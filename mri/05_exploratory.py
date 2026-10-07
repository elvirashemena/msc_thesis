"""Run the general, within-group and between-group DMN connectivity analysis.

For every within-DMN ROI pair, runs the whole-sample Baseline to Followup test
first, then within-group and between-group tests for each stratum (sex, HAM-D
responder, BDI responder, comorbidity, clinical group), plus correlations of
the connectivity delta with symptom percent change. Statistics come from
fmri_stats.py and plots from fmri_plots.py, mirroring the EEG pipeline.

Inputs
------
CONNECTIVITY_XLSX
    Sheet 'wide' of connectivity.xlsx (from 03_conn_connectivity_extractor.py),
    including the strata columns.
QUESTIONNAIRE_XLSX
    questionnaire_converted.xlsx with HAMD17_percentage and BDI_percentage.

Outputs
-------
OUT_DIR/prepost_results.csv
    All general, within and between tests with p and p_corr.
OUT_DIR/correlation_results.csv
    All general and within correlations.
OUT_DIR/descriptive_stats.csv
    Mean, SD and bootstrap 95% CI summary per test.
OUT_DIR/pair_plots/{pair}_{stratum}.png
    Bar and trajectory panels per pair and stratum.
OUT_DIR/correlation_heatmap_general.png, correlation_heatmap_{stratum}.png
    Spearman rho heatmaps for the whole sample and per stratum.

Notes
-----
- INCLUDE_DLPFC_PAIRS is False so the dlPFC-DMN pairs stay a separate FDR
  family.
- p_corr is BH-FDR within each (kind, stratum) family. All stratified results
  are exploratory at n of about 20.

Author: Elvira Shemena
"""

CONNECTIVITY_XLSX  = "connectivity_20pat_5roi.xlsx"
QUESTIONNAIRE_XLSX = "questionnaire_converted.xlsx"
OUT_DIR            = "results_exploratory"

SEP = "::"
INCLUDE_DLPFC_PAIRS = False

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fmri_stats as fs
import fmri_plots as fp

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "pair_plots"), exist_ok=True)


wide = pd.read_excel(CONNECTIVITY_XLSX, sheet_name="wide")
pair_cols = [c for c in wide.columns if SEP in c]
dmn_cols   = [c for c in pair_cols if "DLPFC" not in c.upper()]
dlpfc_cols = [c for c in pair_cols if "DLPFC" in c.upper()]
test_cols  = dmn_cols + (dlpfc_cols if INCLUDE_DLPFC_PAIRS else [])

long    = fs.build_long_connectivity(wide, test_cols)
lookup  = fs.build_lookup(wide)
q       = pd.read_excel(QUESTIONNAIRE_XLSX)
clinical = fs.build_clinical(q)
pairs   = sorted(long["pair"].unique())

print(f"[1] {len(pairs)} pair(s) tested: {pairs}")
print(f"[2] Strata available: {[pretty for col, pretty in fs.STRATA if col in lookup.columns]}")


prepost = fs.prepost_analysis(long, lookup, pairs=pairs)
prepost.to_csv(os.path.join(OUT_DIR, "prepost_results.csv"), index=False)

n_between_sig = ((prepost["kind"] == "between") & (prepost["p_corr"] < .05)).sum()
n_general_sig = ((prepost["kind"] == "general") & (prepost["p_corr"] < .05)).sum()
print(f"\n[3] prepost_analysis: {len(prepost)} rows saved -> prepost_results.csv")
print(f"    general tests surviving FDR: {n_general_sig}/{(prepost['kind']=='general').sum()}")
print(f"    between-group tests surviving FDR: {n_between_sig}/{(prepost['kind']=='between').sum()}")


corr = fs.correlation_analysis(long, clinical, lookup, pairs=pairs)
corr.to_csv(os.path.join(OUT_DIR, "correlation_results.csv"), index=False)
n_corr_sig = (corr["p_corr"] < .05).sum()
print(f"\n[4] correlation_analysis: {len(corr)} rows saved -> correlation_results.csv "
      f"({n_corr_sig}/{len(corr)} survive within-family FDR)")


print("\n[5] Building pair x stratum panels ...")

for pair in pairs:
    w = fs.pair_wide(long, pair).dropna(subset=[fs.BASELINE_LABEL, fs.FOLLOWUP_LABEL])
    gen_row = prepost[(prepost["pair"] == pair) & (prepost["kind"] == "general")]
    general_p = float(gen_row["p"].iloc[0]) if len(gen_row) else np.nan

    for col, pretty in fs.STRATA:
        if col not in lookup.columns:
            continue
        levels = fs._levels(lookup, col)
        if len(levels) != 2:
            continue
        (lvl_a, ids_a), (lvl_b, ids_b) = sorted(levels.items(), reverse=True)
        wa = w[w.index.isin(ids_a)]
        wb = w[w.index.isin(ids_b)]
        if len(wa) == 0 or len(wb) == 0:
            continue

        within_a = prepost[(prepost.pair == pair) & (prepost.kind == "within") &
                            (prepost.stratum == pretty) & (prepost.level == lvl_a)]
        within_b = prepost[(prepost.pair == pair) & (prepost.kind == "within") &
                            (prepost.stratum == pretty) & (prepost.level == lvl_b)]
        between = prepost[(prepost.pair == pair) & (prepost.kind == "between") &
                           (prepost.stratum == pretty)]
        p_a = float(within_a["p"].iloc[0]) if len(within_a) else np.nan
        p_b = float(within_b["p"].iloc[0]) if len(within_b) else np.nan
        p_between = float(between["p"].iloc[0]) if len(between) else np.nan
        p_between_fdr = float(between["p_corr"].iloc[0]) if len(between) else np.nan

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

        group_specs = [
            dict(label=f"All (n={len(w)})", baseline=w[fs.BASELINE_LABEL].values,
                 followup=w[fs.FOLLOWUP_LABEL].values, p=general_p, bracket_color="black"),
            dict(label=f"{lvl_a} (n={len(wa)})", baseline=wa[fs.BASELINE_LABEL].values,
                 followup=wa[fs.FOLLOWUP_LABEL].values, p=p_a, bracket_color="grey"),
            dict(label=f"{lvl_b} (n={len(wb)})", baseline=wb[fs.BASELINE_LABEL].values,
                 followup=wb[fs.FOLLOWUP_LABEL].values, p=p_b, bracket_color="grey"),
        ]
        fp.prepost_bar_panel(axes[0], group_specs,
                             title=f"General (black) + within-group (grey, descriptive only)")
        axes[0].set_ylabel(f"{pair} connectivity (Fisher-z)")

        fp.trajectory_panel(axes[1], lvl_a, lvl_b,
                            wa[fs.BASELINE_LABEL].values, wa[fs.FOLLOWUP_LABEL].values,
                            wb[fs.BASELINE_LABEL].values, wb[fs.FOLLOWUP_LABEL].values,
                            lvl_a, lvl_b, ylabel=f"{pair} connectivity (Fisher-z)")
        axes[1].set_title(f"Between-group CHANGE (inferential): p={p_between:.3f}, "
                          f"p_FDR={p_between_fdr:.3f}", fontsize=9)

        fig.suptitle(f"{pair} - stratified by {pretty}", fontsize=12)
        fig.tight_layout()
        fig_path = os.path.join(OUT_DIR, "pair_plots", f"{pair}_{col}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

print(f"    Saved panels -> {os.path.join(OUT_DIR, 'pair_plots')}")


print("\n[6] Building correlation heatmaps ...")

def _heatmap_from(sub, col_key, out_name, title):
    """Pivot correlation rows to a pair by column heatmap and save it as out_name."""
    if sub.empty:
        print(f"    {out_name}: no rows, skipped")
        return
    sub = sub.copy()
    sub["col"] = sub[col_key]
    pivot_r = sub.pivot_table(index="pair", columns="col", values="rho", aggfunc="first").loc[pairs]
    pivot_p = sub.pivot_table(index="pair", columns="col", values="p_corr", aggfunc="first").loc[pairs]
    pivot_stars = pivot_p.map(lambda p: fs.sig_stars(p) if pd.notna(p) else "")
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * pivot_r.shape[1]), 0.5 * len(pairs) + 1.5))
    im = fp.correlation_heatmap(ax, pivot_r, pivot_stars, corr_label=title + "\n(stars = FDR-corrected)")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_name}")

general = corr[corr["kind"] == "general"].copy()
general["scale"] = general["measure"].str.replace("_percentage", " %chg", regex=False)
_heatmap_from(general, "scale", "correlation_heatmap_general.png",
             "delta connectivity vs symptom %change (whole sample)")

for col, pretty in fs.STRATA:
    sub = corr[(corr["kind"] == "within") & (corr["stratum"] == pretty)].copy()
    if sub.empty:
        continue
    sub["scale"] = sub["level"] + " | " + sub["measure"].str.replace("_percentage", " %chg", regex=False)
    _heatmap_from(sub, "scale", f"correlation_heatmap_{col}.png",
                 f"delta connectivity vs symptom %change, within {pretty}")

print("\n[7] Building descriptive statistics table (Mean+/-SD, 95% CI) ...")

def _fmt_paired(row):
    """Format a general or within row as a one-line descriptive summary."""
    return (f"Baseline {row.mean_baseline:.3f}±{row.sd_baseline:.3f}, "
            f"Followup {row.mean_followup:.3f}±{row.sd_followup:.3f}, "
            f"Δ={row.mean_delta:+.3f} [95% CI {row.delta_ci95_lo:.3f}, {row.delta_ci95_hi:.3f}], "
            f"n={int(row.n)}")

def _fmt_between(row):
    """Format a between row as a one-line summary of the group deltas."""
    la, lb = [s.strip() for s in row.level.split(" vs ")]
    return (f"Δ({la})={row.mean_delta_a:+.3f}±{row.sd_delta_a:.3f} (n={int(row.n_a)}) vs "
            f"Δ({lb})={row.mean_delta_b:+.3f}±{row.sd_delta_b:.3f} (n={int(row.n_b)}); "
            f"diff={row.diff_in_means:+.3f} [95% CI {row.diff_ci95_lo:.3f}, {row.diff_ci95_hi:.3f}]")

desc = prepost.copy()
desc["summary_text"] = ""
paired_mask = desc["kind"].isin(["general", "within"])
between_mask = desc["kind"] == "between"
desc.loc[paired_mask, "summary_text"] = desc.loc[paired_mask].apply(_fmt_paired, axis=1)
desc.loc[between_mask, "summary_text"] = desc.loc[between_mask].apply(_fmt_between, axis=1)

desc_out = desc[["pair", "kind", "stratum", "level", "p", "p_corr", "summary_text"]].sort_values(
    ["pair", "kind", "stratum", "level"]).reset_index(drop=True)
desc_path = os.path.join(OUT_DIR, "descriptive_stats.csv")
desc_out.to_csv(desc_path, index=False)

print(desc_out.to_string(index=False))
print(f"\n    Saved -> {desc_path}")
print("    (95% CIs are nonparametric percentile-bootstrap, 5000 resamples - matches "
      "Wilcoxon/Mann-Whitney being the primary tests here rather than assuming normality)")


print("\nDone. Remember: p_corr on the correlation heatmaps and p_corr on the between-group "
      "rows in prepost_results.csv are each corrected WITHIN their own (kind, stratum) family - "
      "not across the whole pipeline - by design (see fmri_stats.py docstring).")
