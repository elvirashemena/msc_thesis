"""Extract CONN ROI-to-ROI connectivity into long and wide tables.

Reads every first-level result file, takes its Fisher-z matrix and merges it
with all columns of the ID lookup table.

Inputs
------
RRC_DIR
    resultsROI_Subject*_Condition*.mat with the matrix in 'Z' and the ROI
    labels in 'names' (rows) and 'names2' (columns). _list_sources.mat is
    used as an optional label check.
LOOKUP_XLSX
    id_lookup.xlsx with mri_id and clinical_id; all other columns are carried
    through as demographics.

Outputs
-------
OUT_DIR/connectivity_long.csv
    One row per subject, condition and ROI pair.
OUT_DIR/connectivity_wide.csv
    One row per subject and condition, pairs as '{seed}::{target}' columns.
OUT_DIR/connectivity.xlsx
    Sheets 'long' and 'wide'.

Notes
-----
- TRIANGLE_ONLY keeps unique pairs only (upper triangle without diagonal).
- The wide table is pivoted on mri_id and condition before the demographics
  are merged, so a missing demographic value never drops a subject.

Author: Elvira Shemena
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import scipy.io

RRC_DIR     = Path(r"C:\Users\decarvfc\Documents\MATLAB\22pat_NEW_261125.qlog\251126141255531\conn_ElviraSample_19pat_direct&fmap_210126\results\firstlevel\RRC_02")
LOOKUP_XLSX = Path(r"C:\Users\decarvfc\Desktop\scripts19pat\id_lookup.xlsx")
OUT_DIR     = Path(r"C:\Users\decarvfc\Desktop\scripts19pat")

CONDITIONS    = {"Condition001": "Baseline", "Condition002": "Followup"}
TRIANGLE_ONLY = True
SEP           = "::"
FNAME_RE      = re.compile(r"resultsROI_(Subject\d+)_(Condition\d+)\.mat", re.I)


def cells(arr):
    """Convert a MATLAB cell array of strings to a list of str."""
    out = []
    for x in np.ravel(arr):
        while isinstance(x, np.ndarray):
            x = x.ravel()[0] if x.size else ""
        out.append(str(x).strip())
    return out


files = sorted(RRC_DIR.glob("resultsROI_Subject*_Condition*.mat"))
assert files, f"no resultsROI_*.mat files found in {RRC_DIR}"

m0 = scipy.io.loadmat(files[0])
row_labels = cells(m0["names"])
col_labels = cells(m0["names2"])
nr, nc = m0["Z"].shape
assert len(row_labels) == nr and len(col_labels) == nc, \
    f"names/names2 ({len(row_labels)}/{len(col_labels)}) != Z shape {(nr, nc)}"

try:
    src = cells(scipy.io.loadmat(RRC_DIR / "_list_sources.mat")["sourcenames"])
    if src and src != row_labels:
        print("    note: _list_sources.mat sourcenames differ from result-file 'names'")
except Exception as e:
    print(f"    note: skipped _list_sources.mat ({e})")

if TRIANGLE_ONLY:
    assert row_labels == col_labels, "TRIANGLE_ONLY needs a square symmetric matrix; set it False"
    pairs = [(i, j) for i in range(nr) for j in range(i + 1, nr)]
else:
    pairs = [(i, j) for i in range(nr) for j in range(nc)]
print(f"[1] Z is {nr}x{nc} -> {len(pairs)} pairs/file (TRIANGLE_ONLY={TRIANGLE_ONLY})")

rows = []
for f in files:
    m = FNAME_RE.match(f.name)
    if not m:
        continue
    mri_id, cond = m.group(1), m.group(2)
    Z = scipy.io.loadmat(f)["Z"]
    assert Z.shape == (nr, nc), f"{f.name}: Z is {Z.shape}, expected ({nr},{nc})"
    for i, j in pairs:
        rows.append((mri_id, CONDITIONS.get(cond, cond),
                     row_labels[i], col_labels[j], float(Z[i, j])))
print(f"[2] read {len(files)} files -> {len(rows):,} rows")

look = pd.read_excel(LOOKUP_XLSX)
for req in ("mri_id", "clinical_id"):
    assert req in look.columns, f"id_lookup is missing required column '{req}'"
DEMO = list(look.columns)
conn = pd.DataFrame(rows, columns=["mri_id", "condition", "seed", "target", "connectivity"])
long = look.merge(conn, on="mri_id", how="right")[DEMO + ["condition", "seed", "target", "connectivity"]]

unmatched = sorted(long.loc[long["clinical_id"].isna(), "mri_id"].unique())
print(f"[3] subjects={long['mri_id'].nunique()}  conditions={long['condition'].nunique()}  "
      f"NaN connectivity={long['connectivity'].isna().sum()}  "
      f"range=[{long['connectivity'].min():.3f}, {long['connectivity'].max():.3f}]")
if unmatched:
    print(f"    WARNING: {len(unmatched)} mri_id(s) absent from id_lookup: {unmatched}")
    long["clinical_id"] = long["clinical_id"].fillna(long["mri_id"])
assert len(long) == len(files) * len(pairs), "row count != files x pairs"

long["pair"] = long["seed"] + SEP + long["target"]
wide = (long.pivot_table(index=["mri_id", "condition"], columns="pair",
                         values="connectivity", aggfunc="first").reset_index())
wide.columns.name = None
demo_tbl = long.drop_duplicates("mri_id")[DEMO]
pair_cols = [c for c in wide.columns if SEP in c]
wide = demo_tbl.merge(wide, on="mri_id", how="right")[DEMO + ["condition"] + pair_cols]
long = long.drop(columns="pair")

OUT_DIR.mkdir(parents=True, exist_ok=True)
long.to_csv(OUT_DIR / "connectivity_long.csv", index=False)
wide.to_csv(OUT_DIR / "connectivity_wide.csv", index=False)
with pd.ExcelWriter(OUT_DIR / "connectivity.xlsx") as xl:
    long.to_excel(xl, sheet_name="long", index=False)
    wide.to_excel(xl, sheet_name="wide", index=False)
print(f"[4] saved long {long.shape} + wide {wide.shape} -> {OUT_DIR}")
