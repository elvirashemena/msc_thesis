"""Transform FreeSurfer RAS coordinates to MNI space with SPM y_ fields.

The y_ deformation field (from CONN) stores, at each MNI voxel, the native
coordinate it samples from. For each native RAS point, an inverse search finds
the MNI voxel whose stored coordinate is closest and then refines it to
sub-voxel precision. This is done for the capsule and surface point of every
row.

Inputs
------
COORDS_FILE
    Excel with ID, Timepoint, x/y/z_capsule_ras and x/y/z_surface_ras.
Y_FIELD_DIR
    Root folder with {ID}/{Timepoint}/y_{Timepoint}_T1w_MPR_*.nii.

Outputs
-------
OUTPUT_CSV
    Input table plus x/y/z_capsule_mni, x/y/z_surface_mni,
    capsule_match_dist_mm, surface_match_dist_mm and y_file_used.
mni_coordinates_yfield.xlsx
    Excel copy of the CSV, written in the working directory.

Notes
-----
- The match distance is the residual of the inverse search. Values below
  1 mm are good; values above MATCH_DIST_WARN suggest a coordinate space
  mismatch.

Author: Elvira Shemena
"""
import nibabel as nib
import numpy as np
import pandas as pd
import os
import glob
from scipy.ndimage import map_coordinates

Y_FIELD_DIR = r"C:\Users\decarvfc\Desktop\MRI_Preprocessed"
COORDS_FILE = r"C:\Users\decarvfc\Desktop\scripts_22pat\coordinates.xlsx"
OUTPUT_CSV  = r"C:\Users\decarvfc\Desktop\scripts_22pat\mni_coordinates_yfield.csv"

MATCH_DIST_WARN = 5.0

def voxel_to_mm(voxel_coord, affine):
    """Convert a voxel index to mm coordinates with an affine."""
    homogeneous = np.append(voxel_coord, 1.0)
    return (affine @ homogeneous)[:3]


def find_y_file(y_field_dir, subject_id, timepoint):
    """Return the y_ deformation field of one subject and timepoint.

    Searches {y_field_dir}/{subject_id}/{timepoint}/y_{timepoint}_T1w_MPR_*.nii
    and returns the first match, or None if there is none.
    """
    search_dir = os.path.join(y_field_dir, subject_id, timepoint)
    if not os.path.isdir(search_dir):
        return None

    pattern = os.path.join(search_dir, f"y_{timepoint}_T1w_MPR_*.nii")
    matches = glob.glob(pattern)

    matches = [m for m in matches if os.path.basename(m).startswith(f"y_{timepoint}")]

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(f"  WARNING: {len(matches)} y_ files found, using first: {matches[0]}")
    return matches[0]


def find_mni_coarse(y_data_3d, y_affine, native_mm):
    """Find the MNI voxel whose stored native coordinate is closest to native_mm.

    Returns
    -------
    mni_mm : numpy.ndarray, shape (3,)
        MNI coordinate of the best voxel.
    min_dist : float
        Distance in mm between the stored and the target native coordinate.
    best_voxel : numpy.ndarray, shape (3,)
        Voxel index of the match.
    """
    diff = y_data_3d - native_mm[np.newaxis, np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(diff**2, axis=3))

    flat_idx = np.argmin(dist)
    best_voxel = np.array(np.unravel_index(flat_idx, dist.shape))
    min_dist = dist[best_voxel[0], best_voxel[1], best_voxel[2]]

    mni_mm = voxel_to_mm(best_voxel, y_affine)
    return mni_mm, min_dist, best_voxel


def refine_mni_subvoxel(y_data_3d, y_affine, native_mm, coarse_voxel, radius=2):
    """Refine the coarse match on a 0.25-voxel grid within radius voxels.

    The y_ field is interpolated linearly at each candidate position.

    Returns
    -------
    mni_mm : numpy.ndarray, shape (3,)
        Refined MNI coordinate.
    min_dist : float
        Residual distance in mm.
    """
    step = 0.25
    offsets = np.arange(-radius, radius + step, step)
    grid = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    candidates = np.stack([g.ravel() for g in grid], axis=1)
    candidates += coarse_voxel[np.newaxis, :]

    shape = y_data_3d.shape
    best_dist = np.inf
    best_mni = None

    for c in candidates:
        if np.any(c < 0) or c[0] >= shape[0] or c[1] >= shape[1] or c[2] >= shape[2]:
            continue

        coords = c.reshape(3, 1)
        sampled = np.array([
            map_coordinates(y_data_3d[:, :, :, dim], coords, order=1, mode="nearest")[0]
            for dim in range(3)
        ])
        dist = np.sqrt(np.sum((sampled - native_mm)**2))
        if dist < best_dist:
            best_dist = dist
            best_mni = voxel_to_mm(c, y_affine)

    return best_mni, best_dist


def transform_single_point(y_data_3d, y_affine, native_mm):
    """Map one native RAS point to MNI with a coarse search and refinement.

    Returns
    -------
    mni_mm : numpy.ndarray, shape (3,)
        MNI coordinate.
    match_dist : float
        Residual distance in mm, the precision of the inverse search.
    """
    mni_coarse, dist_coarse, best_vox = find_mni_coarse(y_data_3d, y_affine, native_mm)
    mni_refined, dist_refined = refine_mni_subvoxel(y_data_3d, y_affine, native_mm, best_vox)
    return mni_refined, dist_refined


def main():
    """Transform every row of COORDS_FILE and save the result to OUTPUT_CSV."""
    print("=" * 60)
    print("Batch RAS → MNI via SPM y_ field inverse search")
    print("=" * 60)

    df = pd.read_excel(COORDS_FILE)
    print(f"\nLoaded {len(df)} rows from {os.path.basename(COORDS_FILE)}")

    ras_cols = ["x_capsule_ras", "y_capsule_ras", "z_capsule_ras",
                "x_surface_ras", "y_surface_ras", "z_surface_ras"]
    for col in ras_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    for pt in ("capsule", "surface"):
        for ax in ("x", "y", "z"):
            df[f"{ax}_{pt}_mni"] = np.nan
        df[f"{pt}_match_dist_mm"] = np.nan
    df["y_file_used"] = ""

    n_processed = 0
    n_skipped = 0
    n_warned = 0

    y_cache = {}

    for idx, row in df.iterrows():
        subj_id   = row["ID"]
        timepoint = row["Timepoint"]

        print(f"\n[{idx+1}/{len(df)}] {subj_id} / {timepoint}")

        capsule_ras = np.array([row["x_capsule_ras"], row["y_capsule_ras"], row["z_capsule_ras"]])
        surface_ras = np.array([row["x_surface_ras"], row["y_surface_ras"], row["z_surface_ras"]])

        if np.any(np.isnan(capsule_ras)) or np.any(np.isnan(surface_ras)):
            print(f"  SKIP: NaN in RAS coordinates")
            n_skipped += 1
            continue

        cache_key = (subj_id, timepoint)
        if cache_key in y_cache:
            y_data_3d, y_affine, y_path = y_cache[cache_key]
            print(f"  Using cached y_ field")
        else:
            y_path = find_y_file(Y_FIELD_DIR, subj_id, timepoint)
            if y_path is None:
                print(f"  SKIP: no y_ file found in {os.path.join(Y_FIELD_DIR, subj_id, timepoint)}")
                n_skipped += 1
                continue

            print(f"  Loading: {os.path.basename(y_path)}")
            y_img = nib.load(y_path)
            y_data = y_img.get_fdata()
            y_affine = y_img.affine

            if y_data.ndim == 5:
                y_data_3d = y_data[:, :, :, 0, :]
            else:
                y_data_3d = y_data

            y_cache[cache_key] = (y_data_3d, y_affine, y_path)

        cap_mni, cap_dist = transform_single_point(y_data_3d, y_affine, capsule_ras)
        df.loc[idx, "x_capsule_mni"] = cap_mni[0]
        df.loc[idx, "y_capsule_mni"] = cap_mni[1]
        df.loc[idx, "z_capsule_mni"] = cap_mni[2]
        df.loc[idx, "capsule_match_dist_mm"] = cap_dist

        surf_mni, surf_dist = transform_single_point(y_data_3d, y_affine, surface_ras)
        df.loc[idx, "x_surface_mni"] = surf_mni[0]
        df.loc[idx, "y_surface_mni"] = surf_mni[1]
        df.loc[idx, "z_surface_mni"] = surf_mni[2]
        df.loc[idx, "surface_match_dist_mm"] = surf_dist

        df.loc[idx, "y_file_used"] = os.path.basename(y_path)

        print(f"  Capsule: RAS {capsule_ras} → MNI [{cap_mni[0]:.2f}, {cap_mni[1]:.2f}, {cap_mni[2]:.2f}] (Δ={cap_dist:.3f}mm)")
        print(f"  Surface: RAS {surface_ras} → MNI [{surf_mni[0]:.2f}, {surf_mni[1]:.2f}, {surf_mni[2]:.2f}] (Δ={surf_dist:.3f}mm)")

        if cap_dist > MATCH_DIST_WARN or surf_dist > MATCH_DIST_WARN:
            print(f"  ⚠ LARGE MATCH DISTANCE — possible coordinate space mismatch")
            n_warned += 1

        n_processed += 1

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Processed: {n_processed}")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Warnings:  {n_warned}")
    print(f"  Saved to:  {OUTPUT_CSV}")
    print()
    print("Output columns:")
    print("  x/y/z_capsule_mni  — MNI coordinates (capsule projection)")
    print("  x/y/z_surface_mni  — MNI coordinates (surface entry point)")
    print("  *_match_dist_mm    — inverse search precision (<1mm = good)")
    print("  y_file_used        — which y_ file was loaded (audit trail)")
    print()
    print("QC steps:")
    print("  1. Check match_dist columns — all should be <1mm")
    print("  2. Spot-check 3-4 subjects in MRIcroGL on MNI152")
    print("  3. All surface points should land on cortex")
    print("  4. Feed into dlpfc_anticorrelation_batch.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

import pandas as pd
import openpyxl

df = pd.read_csv("mni_coordinates_yfield.csv")
df.to_excel("mni_coordinates_yfield.xlsx", index=False)

