"""Test the Baseline to Followup change of each EEG feature, overall and by stratum.

Cheap step that reads only the feature caches. For each feature (power, FAA,
wPLI) and condition, runs:

- general: paired Wilcoxon on all subjects, the reported test.
- within: paired Wilcoxon inside each subgroup, descriptive only.
- between: Mann-Whitney on the per-subject change, the inferential
  stratified test.

The strata are the five grouping columns of the ID lookup; all stratified
results are exploratory.

Inputs
------
features/*_long.csv
    power_long.csv, faa_long.csv and connectivity_long.csv.
LOOKUP_XLSX
    id_lookup.xlsx with the strata columns.

Outputs
-------
results/{feature}_prepost.csv
    One row per test with p and BH-FDR p_corr.
"""

from pathlib import Path
import pandas as pd
import eeg_compute as ec

FEAT = Path("features")
RES = Path("results"); RES.mkdir(exist_ok=True)
LOOKUP_XLSX = Path(r"C:\Users\jojos\coding\aiTBS-eeg-fmri-biomarkers\eeg\id_lookup.xlsx")

lookup = ec.load_lookup(LOOKUP_XLSX)

for feature, cfg in ec.FEATURES.items():
    path = FEAT / cfg["csv"]
    if not path.exists():
        print(f"  - {feature}: {path} not found, skipped")
        continue
    long = pd.read_csv(path, dtype={"eeg_id": str})
    parts = [ec.prepost_analysis(long, feature, cond, lookup)
             for cond in sorted(long["condition"].unique())]
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RES / f"{feature}_prepost.csv", index=False)

    g = out[out.kind == "general"]
    print(f"\n=== {feature.upper()} — general (all subjects) ===")
    for _, r in g.iterrows():
        u = "" if pd.isna(r.unit) else f"{r.unit:<14}"
        print(f"  {r.condition} {u} n={int(r.n):2d}  effect={r.effect:+.2f}  "
              f"p={r.p:.3f}  q={r.p_corr:.3f} {ec.sig_stars(r.p_corr)}")
    print(f"  (+ {len(out[out.kind=='between'])} between-group and "
          f"{len(out[out.kind=='within'])} within-group exploratory tests in CSV)")

print(f"\nSaved -> {RES}")
