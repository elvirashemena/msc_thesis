"""Summarise the significant and trend-level EEG results in one table.

Cheap, standalone step. Scans the pre to post and correlation results, keeps
significant or trend rows, describes each test in plain language and prints
and saves one sorted table. Tiers, from strongest:

- SIG (FDR): p_corr < 0.05, survives multiple-comparison correction.
- sig (raw): p < 0.05 but p_corr >= 0.05.
- trend: 0.05 <= min(p, p_corr) < 0.10.

Only the general tests are pre-specified; everything else is exploratory.

Inputs
------
results/*_prepost.csv, results/*_correlations.csv
    Outputs of 04_statistics_pre_post.py and 05_correlations.py.

Outputs
-------
results/SUMMARY_hits.csv
    One row per hit, also printed by tier.
"""

from pathlib import Path
import numpy as np
import pandas as pd

RES = Path("results")
OUT = RES / "SUMMARY_hits.csv"
SIG, TREND = 0.05, 0.10


def tier(p, q):
    """Return the tier label for raw p and FDR q, or None below trend level."""
    p = p if np.isfinite(p) else 1.0
    q = q if np.isfinite(q) else 1.0
    if q < SIG:
        return "SIG (FDR)"
    if p < SIG:
        return "sig (raw)"
    if min(p, q) < TREND:
        return "trend"
    return None
 
 
def spatial_spectral(r, feature):
    """Return a readable location of the test: band and region."""
    band = r.get("band")
    has_band = band is not None and pd.notna(band) and str(band).lower() != "nan"
    unit = r.get("unit")
    if feature == "power":
        return f"{unit} power (whole-head)"
    if feature == "wpli":
        return f"{band + ' ' if has_band else ''}wPLI ({unit})"
    if feature == "faa":
        return "frontal alpha asymmetry"
    return f"{feature} {unit}"
 
 
def role_of(kind):
    """Return the evidence role of a test kind: primary, inferential or descriptive."""
    return {"general": "primary (pre-specified)",
            "between": "inferential (stratified, exploratory)",
            "within": "descriptive (stratified)"}.get(kind, kind)
 
 
def describe(r, feature, analysis):
    """Return a one-line plain-language description of a results row."""
    cond = str(r.get("condition", "")).upper()
    loc = spatial_spectral(r, feature)
    kind = r.get("kind")
    stratum, level = r.get("stratum"), r.get("level")
    if analysis == "prepost":
        if kind == "general":
            what = f"{loc} changed Baseline->Followup (whole sample, paired)"
        elif kind == "within":
            what = f"{loc} changed Baseline->Followup WITHIN {stratum}={level} (descriptive)"
        else:
            what = (f"{loc}: GROUP DIFFERENCE in Baseline->Followup change, "
                    f"{stratum} ({level}) — Mann-Whitney (inferential)")
    else:
        meas = r.get("measure")
        base = f"{loc} change vs {meas} (Spearman)"
        what = base if kind == "general" else f"{base} WITHIN {stratum}={level}"
    return f"[{cond}] {what}"
 
 
rows = []
for f in sorted(RES.glob("*_prepost.csv")) + sorted(RES.glob("*_correlations.csv")):
    analysis = "prepost" if f.name.endswith("_prepost.csv") else "correlations"
    feature = f.name.replace(f"_{analysis}.csv", "")
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        p, q = r.get("p", np.nan), r.get("p_corr", np.nan)
        t = tier(p, q)
        if t is None:
            continue
        eff = r["rho"] if analysis == "correlations" else r.get("effect", np.nan)
        eff_type = "rho" if analysis == "correlations" else str(r.get("effect_name", "effect"))
        if np.isfinite(r.get("n", np.nan)):
            n = int(r["n"])
        elif np.isfinite(r.get("n_a", np.nan)):
            n = f"{int(r['n_a'])}v{int(r['n_b'])}"
        else:
            n = ""
        if analysis == "correlations":
            lo, hi = r.get("rho_ci_lo", np.nan), r.get("rho_ci_hi", np.nan)
        else:
            lo, hi = r.get("ci_lo", np.nan), r.get("ci_hi", np.nan)
        ci_txt = (f"[{lo:+.3f}, {hi:+.3f}]"
                  if np.isfinite(lo) and np.isfinite(hi) else "")
        mpre, spre = r.get("mean_pre", np.nan), r.get("sd_pre", np.nan)
        mpost, spost = r.get("mean_post", np.nan), r.get("sd_post", np.nan)
        prepost_txt = (f"{mpre:.3f}±{spre:.3f} -> {mpost:.3f}±{spost:.3f}"
                       if np.isfinite(mpre) and np.isfinite(mpost) else "")
        rows.append({
            "tier": t, "fdr_survives": "yes" if (np.isfinite(q) and q < SIG) else "no",
            "role": role_of(r.get("kind")),
            "feature": feature, "analysis": analysis, "kind": r.get("kind"),
            "condition": r.get("condition"), "band": r.get("band"),
            "region/level": r.get("unit"), "stratum": r.get("stratum"),
            "level": r.get("level"), "measure": r.get("measure"),
            "n": n, "effect": round(float(eff), 3) if np.isfinite(eff) else np.nan,
            "effect_type": eff_type, "ci95": ci_txt,
            "pre_post": prepost_txt,
            "p": round(float(p), 4) if np.isfinite(p) else np.nan,
            "p_corr": round(float(q), 4) if np.isfinite(q) else np.nan,
            "description": describe(r, feature, analysis),
        })
 
hits = pd.DataFrame(rows)
if hits.empty:
    print("No significant or trend results found.")
    raise SystemExit
 
order = {"SIG (FDR)": 0, "sig (raw)": 1, "trend": 2}
hits["_o"] = hits["tier"].map(order)
hits = hits.sort_values(["_o", "p_corr", "p"]).drop(columns="_o").reset_index(drop=True)
hits.to_csv(OUT, index=False)
 
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 80)
print("=" * 100)
print("SUMMARY OF SIGNIFICANT & TREND RESULTS")
print("Tiers: SIG (FDR) p_corr<.05 | sig (raw) p<.05 only | trend .05-.10")
print("Sign: correlations — NEGATIVE rho = feature RISES as symptoms improve "
      "(clinical %change is negative-for-improvement).")
print("Kinds: general=whole sample (pre-specified) | within=subgroup (descriptive) | "
      "between=group difference in change (inferential, exploratory).")
print("ci95: bootstrap 95% CI — on rho (correlations) or on the mean change / "
      "group-difference-in-change (pre/post). WIDE / zero-crossing CIs are expected at n=20.")
print("=" * 100)
for t in ["SIG (FDR)", "sig (raw)", "trend"]:
    sub = hits[hits.tier == t]
    print(f"\n########## {t}  (n = {len(sub)}) ##########")
    if sub.empty:
        print("  (none)")
        continue
    for _, r in sub.iterrows():
        pp = f"  {r['pre_post']}" if r["pre_post"] else ""
        print(f"  {r['effect_type']}={r['effect']:+.2f} {r['ci95']}  "
              f"p={r['p']:.3f}  q={r['p_corr']:.3f}  n={r['n']}{pp}  ::  {r['description']}")
 
print(f"\nSaved {len(hits)} rows -> {OUT}")
print(f"Counts: " + ", ".join(f"{t}={len(hits[hits.tier==t])}"
                               for t in ['SIG (FDR)', 'sig (raw)', 'trend']))