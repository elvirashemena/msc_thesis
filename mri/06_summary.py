"""Collect the significant and trend-level rows of all results CSVs.

Scans CSV_ROOT for results tables, detects the p-value and corrected p-value
columns by name and keeps rows with p < TREND_HIGH. Each hit is tagged with
its source file, a description from DESCRIPTIONS and whether it survives
correction. Files without a p-value column are skipped; input files are only
read, never changed.

Inputs
------
CSV_ROOT
    Folder to scan; RECURSIVE also scans subfolders.
DESCRIPTIONS
    One-line analysis description per filename; add an entry for each new
    results CSV.

Outputs
-------
OUTPUT_CSV
    SUMMARY_hits.csv, one row per hit, also printed by source file.

Notes
-----
- Significant means raw p < SIG_THRESHOLD; check survives_correction before
  reporting a result as significant after correction.

Author: Elvira Shemena
"""

CSV_ROOT   = "."
RECURSIVE  = True
OUTPUT_CSV = "SUMMARY_hits.csv"

SIG_THRESHOLD = 0.05
TREND_HIGH    = 0.10

EXCLUDE_FILES = {OUTPUT_CSV}

P_COL_CANDIDATES      = ["p", "p_value", "pval", "p_raw", "p_uncorrected"]
P_CORR_COL_CANDIDATES = ["p_corr", "p_fdr", "q", "fdr", "p_bonferroni_x2",
                          "p_bonferroni", "p_adj", "p_adjusted"]

CONTEXT_COLUMNS = [
    "pair", "feature", "band", "condition", "unit",
    "kind", "stratum", "level", "restriction", "scale",
    "measure",
    "test", "effect_name",
    "n", "n_a", "n_b",
    "stat", "effect", "r", "rho", "mean_diff",
]

DESCRIPTIONS = {
    "prepost_results.csv":
        "fMRI connectivity: general (whole sample) + within-group (descriptive) + "
        "between-group (inferential, Mann-Whitney on delta) Baseline->Followup test, "
        "per DMN ROI pair x stratum. Source: 05_exploratory.py -> fmri_stats.prepost_analysis.",
    "correlation_results.csv":
        "fMRI connectivity: general (whole sample) + within-stratum correlation of "
        "connectivity change (delta, Fisher-z) vs symptom %change (BDI/HAM-D17), per DMN "
        "ROI pair. Source: 05_exploratory.py -> fmri_stats.correlation_analysis.",
    "planned_test_results.csv":
        "fMRI connectivity: ONE planned summary test - mean delta across the 6 within-DMN "
        "pairs (DMN composite) vs symptom %change, unipolar-only (per H3) + all-subjects "
        "sensitivity check. Source: 04_correlation.py.",
}
UNKNOWN_DESCRIPTION = ("UNKNOWN analysis - add a one-line description for this filename "
                       "in DESCRIPTIONS at the top of summarize_hits.py")

from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)


def find_col(df, candidates):
    """Return the first candidate column present in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def summarize_file(path: Path):
    """Return the hit rows of one CSV with source and correction columns added.

    Returns None if the file cannot be read, has no p-value column or has no
    hits.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [skip] {path.name}: could not read ({e})")
        return None

    p_col = find_col(df, P_COL_CANDIDATES)
    if p_col is None:
        return None

    hits = df[pd.to_numeric(df[p_col], errors="coerce") < TREND_HIGH].copy()
    if hits.empty:
        return None

    pcorr_col = find_col(df, P_CORR_COL_CANDIDATES)

    hits["source_file"] = str(path)
    hits["analysis"] = DESCRIPTIONS.get(path.name, UNKNOWN_DESCRIPTION)
    hits["p_value"] = pd.to_numeric(hits[p_col], errors="coerce")
    hits["p_corrected"] = pd.to_numeric(hits[pcorr_col], errors="coerce") if pcorr_col else pd.NA
    hits["p_corrected_column"] = pcorr_col or "(none found)"
    hits["hit_type"] = hits["p_value"].apply(
        lambda p: "significant (uncorrected)" if p < SIG_THRESHOLD else "trend (.05-.10)")
    hits["survives_correction"] = (
        hits["p_corrected"] < SIG_THRESHOLD if pcorr_col else pd.NA
    )

    present_context = [c for c in CONTEXT_COLUMNS if c in hits.columns]
    out_cols = (["source_file", "analysis", "hit_type", "p_value", "p_corrected",
                "p_corrected_column", "survives_correction"] + present_context)
    return hits[out_cols].sort_values("p_value").reset_index(drop=True)


root = Path(CSV_ROOT)
csv_files = sorted(root.rglob("*.csv") if RECURSIVE else root.glob("*.csv"))
csv_files = [f for f in csv_files if f.name not in EXCLUDE_FILES]
print(f"Scanning {len(csv_files)} CSV file(s) under {root.resolve()} (recursive={RECURSIVE}) ...")

all_hits = []
for f in csv_files:
    result = summarize_file(f)
    if result is not None:
        all_hits.append(result)
        print(f"  [hit] {f}: {len(result)} row(s) with p<{TREND_HIGH}")

if not all_hits:
    print("\nNo significant or trend-level rows found in any CSV under this folder.")
else:
    summary = pd.concat(all_hits, ignore_index=True).sort_values(
        ["source_file", "p_value"]).reset_index(drop=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 90}")
    print(f"SUMMARY: {len(summary)} row(s) with p<{TREND_HIGH} across {summary['source_file'].nunique()} file(s)")
    print(f"  significant (p<.05):        {(summary['hit_type'] == 'significant (uncorrected)').sum()}")
    print(f"  trend (.05<=p<.10):          {(summary['hit_type'] == 'trend (.05-.10)').sum()}")
    if summary["survives_correction"].notna().any():
        print(f"  survive correction (p_corr<.05): {(summary['survives_correction'] == True).sum()}")
    print(f"{'=' * 90}\n")

    for src, grp in summary.groupby("source_file", sort=False):
        print(f"--- {src} ---")
        print(f"    {grp['analysis'].iloc[0]}")
        show_cols = [c for c in grp.columns if c not in ("source_file", "analysis")]
        print(grp[show_cols].to_string(index=False))
        print()

    print(f"Saved -> {OUTPUT_CSV}")
    unknown = summary[summary["analysis"] == UNKNOWN_DESCRIPTION]["source_file"].unique()
    if len(unknown):
        print(f"\nNOTE: {len(unknown)} file(s) have no description registered yet - "
              f"add them to DESCRIPTIONS in summarize_hits.py: {list(unknown)}")
