"""Correlate EEG feature change with symptom change, overall and by stratum.

Cheap step that reads only the feature caches. For each feature and
condition, correlates the feature change (percent change for power and wPLI,
delta for FAA) with HAM-D17 and BDI percent change using Spearman's rho with
a bootstrap CI, BH-FDR corrected within each family:

- general: all subjects, the reported test.
- within: inside each subgroup, exploratory.

Clinical percent change is negative for improvement, so a feature that rises
in improvers gives a negative rho. Group differences are tested in
04_statistics_pre_post.py instead.

Inputs
------
features/*_long.csv
    power_long.csv, faa_long.csv and connectivity_long.csv.
QUEST_XLSX
    questionnaire_converted.xlsx with HAMD17_percentage and BDI_percentage.
LOOKUP_XLSX
    id_lookup.xlsx with the strata columns.

Outputs
-------
results/{feature}_correlations.csv
    One row per correlation with rho, its CI, p and p_corr.
"""

from pathlib import Path
import pandas as pd
import eeg_compute as ec

FEAT = Path("features")
RES = Path("results"); RES.mkdir(exist_ok=True)
LOOKUP_XLSX = Path(r"C:\Users\jojos\coding\aiTBS-eeg-fmri-biomarkers\eeg\id_lookup.xlsx")
QUEST_XLSX = Path(r"C:\Users\jojos\coding\aiTBS-eeg-fmri-biomarkers\eeg\questionnaire_converted.xlsx")

lookup = ec.load_lookup(LOOKUP_XLSX)
clinical = ec.load_clinical(QUEST_XLSX)

for feature, cfg in ec.FEATURES.items():
    path = FEAT / cfg["csv"]
    if not path.exists():
        print(f"  - {feature}: {path} not found, skipped")
        continue
    long = pd.read_csv(path, dtype={"eeg_id": str})
    parts = [ec.correlation_analysis(long, clinical, feature, cond, lookup)
             for cond in sorted(long["condition"].unique())]
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RES / f"{feature}_correlations.csv", index=False)

    g = out[out.kind == "general"]
    print(f"\n=== {feature.upper()} — general correlations (all subjects) ===")
    for _, r in g.iterrows():
        u = "" if pd.isna(r.unit) else f"{r.unit:<14}"
        print(f"  {r.condition} {u} {r.measure:<18} n={int(r.n):2d}  "
              f"rho={r.rho:+.2f}  p={r.p:.3f}  q={r.p_corr:.3f} {ec.sig_stars(r.p_corr)}")

print(f"\nSaved -> {RES}")
