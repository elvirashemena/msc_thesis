"""Build the connectivity change table and run the planned DMN composite test.

Computes the per-subject Baseline value and delta (Followup - Baseline,
Fisher-z) of every pair, plus a DMN composite (mean of the within-DMN pairs).
The planned test correlates the composite delta with HAM-D17 and BDI percent
change in unipolar patients (H3), with the full sample as a sensitivity
check. The per-pair analysis is in 05_exploratory.py.

Inputs
------
CONNECTIVITY_XLSX
    Sheet 'wide' of connectivity.xlsx (from 03_conn_connectivity_extractor.py).
QUESTIONNAIRE_XLSX
    questionnaire_converted.xlsx with HAMD17_percentage and BDI_percentage.

Outputs
-------
OUT_DIR/connectivity_change_table.csv
    One row per subject with baseline_* and delta_* columns, the composite,
    demographics and symptom percent change.
OUT_DIR/planned_composite_vs_{scale}_{restriction}.png
    Scatter plot with a linear fit.
OUT_DIR/planned_test_results.csv
    r, p and Bonferroni (x2) p per restriction and scale.

Notes
-----
- Connectivity change is a raw delta because Fisher-z can be zero or
  negative; symptom change is percent change (negative means improvement).
- A positive r means connectivity and symptom change move in the same
  direction across subjects.
- Opposite-signed pair effects cancel in the composite, so a null result here
  does not rule out pair-level effects.

Author: Elvira Shemena
"""

CONNECTIVITY_XLSX  = "connectivity_20pat_5roi.xlsx"
QUESTIONNAIRE_XLSX = "questionnaire_converted.xlsx"
OUT_DIR            = "results_correlation"

SEP            = "::"
BASELINE_LABEL = "Baseline"
FOLLOWUP_LABEL = "Followup"

CORR_METHOD = "spearman"

UNIPOLAR_ONLY_FOR_PLANNED_TEST = True
MIN_N_FOR_CORR                 = 5

UNIPOLAR_COLOR = "#bac6de"
BIPOLAR_COLOR  = "#f2b49b"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from stats_helpers import p_to_stars

os.makedirs(OUT_DIR, exist_ok=True)
corr_func = spearmanr if CORR_METHOD == "spearman" else pearsonr


wide = pd.read_excel(CONNECTIVITY_XLSX, sheet_name="wide")
pair_cols = [c for c in wide.columns if SEP in c]
assert pair_cols, f"No '{SEP}'-separated pair columns found in {CONNECTIVITY_XLSX}"

dmn_internal_cols = [c for c in pair_cols if "DLPFC" not in c.upper()]
dlpfc_cols        = [c for c in pair_cols if "DLPFC" in c.upper()]
print(f"[1] {len(pair_cols)} pairs total: {len(dmn_internal_cols)} within-DMN, "
      f"{len(dlpfc_cols)} dlPFC-DMN")

base = wide[wide["condition"] == BASELINE_LABEL].set_index("clinical_id")
fu   = wide[wide["condition"] == FOLLOWUP_LABEL].set_index("clinical_id")
common = base.index.intersection(fu.index)
missing = set(wide["clinical_id"]) - set(common)
if missing:
    print(f"    NOTE: {len(missing)} subject(s) missing one condition, dropped: {sorted(missing)}")

delta    = (fu.loc[common, pair_cols] - base.loc[common, pair_cols]).add_prefix("delta_")
baseline = base.loc[common, pair_cols].add_prefix("baseline_")

delta["delta_DMN_composite"]    = delta[[f"delta_{c}" for c in dmn_internal_cols]].mean(axis=1)
baseline["baseline_DMN_composite"] = baseline[[f"baseline_{c}" for c in dmn_internal_cols]].mean(axis=1)

demo_cols = ["sex", "age", "clinical_group", "responder_HAMD17", "responder_BDI", "Comorbidity"]
demo_cols = [c for c in demo_cols if c in base.columns]
demo = base.loc[common, demo_cols]

subj = demo.join(baseline).join(delta).reset_index().rename(columns={"index": "clinical_id"})

q = pd.read_excel(QUESTIONNAIRE_XLSX).rename(columns={"ID": "clinical_id"})
q_keep = [c for c in ["clinical_id", "HAMD17_percentage", "BDI_percentage"] if c in q.columns]
subj = subj.merge(q[q_keep], on="clinical_id", how="left")

change_table_path = os.path.join(OUT_DIR, "connectivity_change_table.csv")
subj.to_csv(change_table_path, index=False)
print(f"[2] Saved per-subject change table ({len(subj)} rows) -> {change_table_path}")


print("\n=== PART A: planned SUMMARY test (DMN composite, unipolar-only per H3) ===")
print("    (a null/small result here is expected if pairs have opposite-signed effects -")
print("     that's the reason Part B below tests each pair individually)")

SCALES = ["HAMD17_percentage", "BDI_percentage"]
planned_results = []

for restrict_label, df_test in [
    ("unipolar_only", subj[subj["clinical_group"] == "Unipolar"] if "clinical_group" in subj.columns else subj),
    ("all_subjects_sensitivity", subj),
]:
    if restrict_label == "unipolar_only" and not UNIPOLAR_ONLY_FOR_PLANNED_TEST:
        continue
    for scale in SCALES:
        d = df_test[["delta_DMN_composite", scale]].dropna()
        n = len(d)
        if n < MIN_N_FOR_CORR:
            print(f"  [{restrict_label}] {scale}: n={n} < {MIN_N_FOR_CORR}, skipped")
            continue
        r, p = corr_func(d["delta_DMN_composite"], d[scale])
        p_bonf2 = min(p * 2, 1.0)
        print(f"  [{restrict_label}] delta_DMN_composite vs {scale}: "
              f"n={n}  r={r:.3f}  p={p:.4f}  p(Bonferroni x2)={p_bonf2:.4f}")
        planned_results.append({"restriction": restrict_label, "scale": scale, "n": n,
                                 "r": r, "p": p, "p_bonferroni_x2": p_bonf2})

        if restrict_label == "unipolar_only" or not UNIPOLAR_ONLY_FOR_PLANNED_TEST:
            fig, ax = plt.subplots(figsize=(5, 4.5))
            ax.scatter(d["delta_DMN_composite"], d[scale], s=60,
                       color=UNIPOLAR_COLOR, edgecolor="black", linewidth=0.6, zorder=3)
            if n >= 3:
                m, b = np.polyfit(d["delta_DMN_composite"], d[scale], 1)
                xs = np.linspace(d["delta_DMN_composite"].min(), d["delta_DMN_composite"].max(), 50)
                ax.plot(xs, m * xs + b, color="black", linewidth=1.2, linestyle="--", zorder=2)
            ax.axhline(0, color="grey", linewidth=0.6)
            ax.axvline(0, color="grey", linewidth=0.6)
            ax.set_xlabel("delta DMN composite connectivity (Followup - Baseline, Fisher-z)")
            ax.set_ylabel(f"{scale.replace('_', ' ')} (%, negative = improvement)")
            stars = p_to_stars(p)
            ax.set_title(f"{restrict_label} (n={n})\n{CORR_METHOD} r={r:.2f}{stars}  p={p:.3f}",
                         fontsize=10)
            fig.tight_layout()
            fig_path = os.path.join(OUT_DIR, f"planned_composite_vs_{scale}_{restrict_label}.png")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)

pd.DataFrame(planned_results).to_csv(os.path.join(OUT_DIR, "planned_test_results.csv"), index=False)
print(f"  Saved planned test results + scatter plot(s) -> {OUT_DIR}")
print("\nFor the full per-pair, per-stratum breakdown (general -> within -> between, plus "
      "correlations, split by sex/HAM-D responder/BDI responder/Comorbidity/clinical_group), "
      "run 05_exploratory.py - it uses the shared fmri_stats.py engine directly on "
      f"{CONNECTIVITY_XLSX} and {QUESTIONNAIRE_XLSX}, so nothing here needs to be duplicated.")
