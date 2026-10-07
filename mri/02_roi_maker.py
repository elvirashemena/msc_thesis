"""Build spherical MNI ROI masks for the DMN and the individual dlPFC targets.

Part 1 makes one DMN sphere per row of a coordinate table. Part 2 makes one
5 mm radius dlPFC sphere per subject, centred on the individual Baseline MNI
target, with a glass-brain QC image. Follow-up rows are skipped because the
Baseline target was used throughout treatment.

Inputs
------
DMN_TABLE
    dmn_roi_coordinates.xlsx or .csv with conn_code, x, y, z,
    Sphere_Diameter (mm) and an optional Circuit column.
DLPFC_TABLE
    dlpfc_coordinates_mni.xlsx (from 01_ras_to_mni_yfield_batch.py) with ID,
    Timepoint and x/y/z_surface_mni.
BETA_DIR
    Optional folder with a background NIfTI per subject for the QC plot;
    leave empty to plot the ROI alone.

Outputs
-------
DMN_OUT_DIR
    {Circuit}_{conn_code}_{diameter}.nii per ROI and dmn_roi_summary.csv.
DLPFC_OUT_DIR
    DLPFC_{ID}.nii per subject, dlpfc_roi_summary.csv and
    qc_plots/{ID}_dlpfc_qc.png.

Notes
-----
- Sphere_Diameter is a diameter and is halved to get the radius.
- APPLY_BRAIN_MASK intersects every sphere with the MNI152 brain mask.

Author: Elvira Shemena
"""

DMN_TABLE      = "dmn_roi_coordinates.xlsx"
DMN_OUT_DIR    = "roi_masks_dmn"

DLPFC_TABLE    = "dlpfc_coordinates_mni.xlsx"
DLPFC_OUT_DIR  = "roi_masks_dlpfc"
DLPFC_RADIUS_MM = 5.0

BETA_DIR = ""

MNI_RESOLUTION   = 2
APPLY_BRAIN_MASK = True

import os
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, plotting


def load_table(path):
    """Load a CSV or XLSX table and replace Unicode minus signs with '-'."""
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace("\u2212", "-", regex=False)
    print(f"  Loaded {len(df)} row(s) from {path}")
    return df


def get_mni_reference(resolution):
    """Return the MNI152 affine, grid shape and optional brain mask."""
    template = datasets.load_mni152_template(resolution=resolution)
    brain_mask = None
    if APPLY_BRAIN_MASK:
        brain_mask = datasets.load_mni152_brain_mask(resolution=resolution).get_fdata().astype(bool)
    return template.affine, template.shape[:3], brain_mask


def make_sphere_mask(center_mni, radius_mm, affine, shape, brain_mask=None):
    """Return a binary sphere mask on the MNI grid, optionally brain-masked."""
    i, j, k = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    vox = np.stack([i, j, k], axis=-1).reshape(-1, 3).astype(float)
    mni = (affine[:3, :3] @ vox.T).T + affine[:3, 3]
    dist = np.sqrt(np.sum((mni - np.asarray(center_mni)) ** 2, axis=1))
    mask = (dist <= radius_mm).reshape(shape)
    if brain_mask is not None:
        mask = mask & brain_mask
    return mask.astype(np.uint8)


def save_roi(mask, affine, out_path):
    """Save a mask as a float32 NIfTI image."""
    nib.save(nib.Nifti1Image(mask.astype(np.float32), affine), out_path)


print("\n=== PART 1: DMN ROI masks ===")

os.makedirs(DMN_OUT_DIR, exist_ok=True)
affine, shape, brain_mask = get_mni_reference(MNI_RESOLUTION)

dmn = load_table(DMN_TABLE)
required = {"conn_code", "x", "y", "z", "Sphere_Diameter (mm)"}
missing = required - set(dmn.columns)
if missing:
    raise ValueError(f"{DMN_TABLE} is missing column(s): {missing}")
for col in ("x", "y", "z", "Sphere_Diameter (mm)"):
    dmn[col] = pd.to_numeric(dmn[col], errors="coerce")
dmn = dmn.dropna(subset=["conn_code", "x", "y", "z", "Sphere_Diameter (mm)"]).reset_index(drop=True)

dmn_summary = []
for _, row in dmn.iterrows():
    circuit  = str(row.get("Circuit", "")).strip() if pd.notna(row.get("Circuit", np.nan)) else ""
    diameter = float(row["Sphere_Diameter (mm)"])
    radius   = diameter / 2.0
    diam_str = str(int(diameter)) if diameter.is_integer() else str(diameter)
    name = f"{circuit}_{row['conn_code']}_{diam_str}" if circuit else f"{row['conn_code']}_{diam_str}"

    center = np.array([row["x"], row["y"], row["z"]])
    mask = make_sphere_mask(center, radius, affine, shape, brain_mask)
    n_vox = int(mask.sum())
    if n_vox == 0:
        print(f"  [WARNING] {name}: 0 voxels after brain masking - check coordinates. Skipped.")
        continue

    save_roi(mask, affine, os.path.join(DMN_OUT_DIR, f"{name}.nii"))
    print(f"  {name:<30s} centre={center} r={radius}mm -> {n_vox} vox")
    dmn_summary.append({"roi_name": name, "conn_code": row["conn_code"],
                         "circuit": circuit, "x": center[0], "y": center[1], "z": center[2],
                         "radius_mm": radius, "n_voxels": n_vox})

pd.DataFrame(dmn_summary).to_csv(os.path.join(DMN_OUT_DIR, "dmn_roi_summary.csv"), index=False)
print(f"  Saved {len(dmn_summary)} DMN ROI mask(s) -> {DMN_OUT_DIR}")


print("\n=== PART 2: dlPFC ROI masks (Baseline only) ===")

os.makedirs(DLPFC_OUT_DIR, exist_ok=True)
qc_dir = os.path.join(DLPFC_OUT_DIR, "qc_plots")
os.makedirs(qc_dir, exist_ok=True)
affine, shape, brain_mask = get_mni_reference(MNI_RESOLUTION)

dlpfc = load_table(DLPFC_TABLE)
required = {"ID", "Timepoint", "x_surface_mni", "y_surface_mni", "z_surface_mni"}
missing = required - set(dlpfc.columns)
if missing:
    raise ValueError(f"{DLPFC_TABLE} is missing column(s): {missing}")

baseline = dlpfc[dlpfc["Timepoint"] == "Baseline"].copy()
for col in ("x_surface_mni", "y_surface_mni", "z_surface_mni"):
    baseline[col] = pd.to_numeric(baseline[col], errors="coerce")
n_dropped = baseline[["x_surface_mni", "y_surface_mni", "z_surface_mni"]].isna().any(axis=1).sum()
baseline = baseline.dropna(subset=["x_surface_mni", "y_surface_mni", "z_surface_mni"]).reset_index(drop=True)
print(f"  {len(baseline)} Baseline subject(s) with valid coordinates "
      f"({n_dropped} dropped for missing coords; Follow-up rows skipped by design)")

dlpfc_summary = []
for _, row in baseline.iterrows():
    subj_id = str(row["ID"])
    center = np.array([row["x_surface_mni"], row["y_surface_mni"], row["z_surface_mni"]])

    mask = make_sphere_mask(center, DLPFC_RADIUS_MM, affine, shape, brain_mask)
    n_vox = int(mask.sum())
    if n_vox == 0:
        print(f"  [WARNING] {subj_id}: 0 voxels after brain masking - check coordinates. Skipped.")
        continue

    roi_path = os.path.join(DLPFC_OUT_DIR, f"DLPFC_{subj_id}.nii")
    save_roi(mask, affine, roi_path)
    roi_img = nib.load(roi_path)

    bg_hits = sorted(Path(BETA_DIR).rglob(f"*{subj_id}*.nii*")) if BETA_DIR else []
    qc_path = os.path.join(qc_dir, f"{subj_id}_dlpfc_qc.png")

    if bg_hits:
        display = plotting.plot_glass_brain(str(bg_hits[0]), colorbar=False, plot_abs=False,
                                             cmap="RdBu_r", title=f"{subj_id} - dlPFC ROI on {bg_hits[0].name}")
        display.add_contours(roi_img, colors="lime", linewidths=2)
    else:
        display = plotting.plot_glass_brain(roi_img, colorbar=False, cmap="Greens",
                                             title=f"{subj_id} - dlPFC ROI (no background found, ROI only)")
    display.savefig(qc_path, dpi=150)
    display.close()

    print(f"  {subj_id:<12s} centre={center} r={DLPFC_RADIUS_MM}mm -> {n_vox} vox "
          f"| QC: {os.path.basename(qc_path)}{' (+ background)' if bg_hits else ''}")

    dlpfc_summary.append({"ID": subj_id, "x": center[0], "y": center[1], "z": center[2],
                           "radius_mm": DLPFC_RADIUS_MM, "n_voxels": n_vox,
                           "background_used": bg_hits[0].name if bg_hits else ""})

pd.DataFrame(dlpfc_summary).to_csv(os.path.join(DLPFC_OUT_DIR, "dlpfc_roi_summary.csv"), index=False)
print(f"  Saved {len(dlpfc_summary)} dlPFC ROI mask(s) -> {DLPFC_OUT_DIR}")
print(f"  QC plots -> {qc_dir}")
