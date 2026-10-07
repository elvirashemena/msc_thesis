"""Statistics and stratification engine for the fMRI connectivity analysis.

Mirrors the statistics and stratification parts of eeg_compute.py, with the
ROI pair as the analysis unit instead of the frequency band. Starts from the
connectivity table made by 03_conn_connectivity_extractor.py.

For every pair, the general (whole-sample) paired test runs first. Each
stratum then gets a descriptive within-group paired test per level and an
inferential between-group Mann-Whitney test on the per-subject delta. A
significant result in one level but not the other is not a group difference.

Notes
-----
- Connectivity change is delta (Followup - Baseline, Fisher-z), never percent
  change, because Fisher-z crosses zero.
- Clinical percent change is negative for improvement.
- Rank-based tests are primary; all stratified results are exploratory.
- group_test effects are positive when the first group has larger values,
  matching eeg_compute.py.

Author: Elvira Shemena
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from stats_helpers import bh_fdr, pretty_pair

BASELINE_LABEL = "Baseline"
FOLLOWUP_LABEL = "Followup"

STRATA = [("sex", "sex"),
          ("responder_HAMD17", "HAM-D responder"),
          ("responder_BDI", "BDI responder"),
          ("Comorbidity", "comorbidity"),
          ("clinical_group", "clinical group")]

CLINICAL_COLS = ["HAMD17_percentage", "BDI_percentage"]

COLOR_PRE, COLOR_POST = "#b9c6dd", "#f2b49a"
COLOR_DOT, COLOR_LINE = "#3d4a5c", "#9aa5b1"
COLOR_A, COLOR_B = "#c0504d", "#4f6d9a"



def build_long_connectivity(wide, pair_cols, id_col="clinical_id", condition_col="condition"):
    """Reshape the wide connectivity table to long format.

    Returns the columns clinical_id, condition, pair and value, with pair as a
    short label such as 'AMPFC-PCC'.
    """
    long = wide.melt(id_vars=[id_col, condition_col], value_vars=pair_cols,
                      var_name="pair", value_name="value")
    long["pair"] = long["pair"].map(pretty_pair)
    return long


def build_lookup(wide, id_col="clinical_id"):
    """Return one row per subject with the STRATA columns as stripped text."""
    cols = [id_col] + [c for c, _ in STRATA if c in wide.columns]
    lk = wide[cols].drop_duplicates(subset=id_col).copy()
    for col, _ in STRATA:
        if col in lk.columns:
            lk[col] = lk[col].astype(str).str.strip()
    return lk


def build_clinical(questionnaire_df, id_col="ID"):
    """Return clinical_id and the clinical percent change columns."""
    q = questionnaire_df.rename(columns={id_col: "clinical_id"})
    cols = ["clinical_id"] + [c for c in CLINICAL_COLS if c in q.columns]
    return q[cols]



def sig_stars(p) -> str:
    """Return '***', '**', '*' or '' for p; '' if p is missing."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def describe(x) -> str:
    """Format the mean and SD of the finite values as text, or 'n/a'."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return "n/a"
    return f"{x.mean():.4g}±{x.std(ddof=1):.4g}" if x.size > 1 else f"{x[0]:.4g}"


def bootstrap_ci(values, n_boot: int = 5000, ci: float = 0.95, seed: int = 42):
    """Return the percentile bootstrap CI for the mean of one sample (NaN if n < 3)."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def bootstrap_ci_diff(a, b, n_boot: int = 5000, ci: float = 0.95, seed: int = 42):
    """Return the percentile bootstrap CI for the difference in means a - b."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    na, nb = a.size, b.size
    if na < 3 or nb < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, na, size=(n_boot, na))
    idx_b = rng.integers(0, nb, size=(n_boot, nb))
    boot_diff = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    lo, hi = np.percentile(boot_diff, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def describe_paired(baseline, followup, n_boot: int = 5000, seed: int = 42) -> dict:
    """Return mean and SD of Baseline, Followup and delta, plus a bootstrap CI.

    Descriptive complement to paired_test; the CI is on the mean delta.
    """
    baseline = np.asarray(baseline, float)
    followup = np.asarray(followup, float)
    ok = np.isfinite(baseline) & np.isfinite(followup)
    b, f = baseline[ok], followup[ok]
    d = f - b
    ci_lo, ci_hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return dict(
        mean_baseline=float(b.mean()) if b.size else np.nan,
        sd_baseline=float(b.std(ddof=1)) if b.size > 1 else np.nan,
        mean_followup=float(f.mean()) if f.size else np.nan,
        sd_followup=float(f.std(ddof=1)) if f.size > 1 else np.nan,
        mean_delta=float(d.mean()) if d.size else np.nan,
        sd_delta=float(d.std(ddof=1)) if d.size > 1 else np.nan,
        delta_ci95_lo=ci_lo,
        delta_ci95_hi=ci_hi,
    )


def describe_between(delta_a, delta_b, n_boot: int = 5000, seed: int = 42) -> dict:
    """Return mean and SD of each group's delta, plus a bootstrap CI on a - b.

    Descriptive complement to group_test; pass the groups in the same order.
    """
    a = np.asarray(delta_a, float); a = a[np.isfinite(a)]
    b = np.asarray(delta_b, float); b = b[np.isfinite(b)]
    ci_lo, ci_hi = bootstrap_ci_diff(a, b, n_boot=n_boot, seed=seed)
    return dict(
        mean_delta_a=float(a.mean()) if a.size else np.nan,
        sd_delta_a=float(a.std(ddof=1)) if a.size > 1 else np.nan,
        mean_delta_b=float(b.mean()) if b.size else np.nan,
        sd_delta_b=float(b.std(ddof=1)) if b.size > 1 else np.nan,
        diff_in_means=float(a.mean() - b.mean()) if a.size and b.size else np.nan,
        diff_ci95_lo=ci_lo,
        diff_ci95_hi=ci_hi,
    )


def paired_test(pre, post, log: bool = False, method: str = "Wilcoxon") -> dict:
    """Run a paired Baseline to Followup test.

    Parameters
    ----------
    pre, post : array-like
        Paired values; pairs with a missing value are dropped.
    log : bool
        Test on log values if all values are positive.
    method : {'Wilcoxon', 'Paired t'}
        Test to run.

    Returns
    -------
    dict
        n, stat, p, effect, effect_name, mean_diff and test. The effect is
        Cohen's d_z (t-test) or the matched-pairs rank-biserial (Wilcoxon) and is
        positive when Followup > Baseline.
    """
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    ok = np.isfinite(pre) & np.isfinite(post)
    b, f = pre[ok], post[ok]
    res = dict(n=int(b.size), stat=np.nan, p=np.nan, effect=np.nan,
               effect_name="", mean_diff=np.nan, test=method)
    if b.size < 3:
        return res
    if log and (b > 0).all() and (f > 0).all():
        b, f = np.log(b), np.log(f)
    d = f - b
    res["mean_diff"] = float(d.mean())
    if np.allclose(d, 0):
        res["p"] = 1.0
        return res
    if method == "Paired t":
        t, p = stats.ttest_rel(f, b)
        sd = d.std(ddof=1)
        res.update(stat=float(t), p=float(p), effect_name="dz",
                   effect=float(d.mean() / sd) if sd > 0 else np.nan)
    else:
        nz = d[d != 0]
        try:
            w, p = stats.wilcoxon(f, b)
        except ValueError:
            return res
        ranks = stats.rankdata(np.abs(nz))
        T = nz.size * (nz.size + 1) / 2
        res.update(stat=float(w), p=float(p), effect_name="rank-biserial",
                   effect=float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / T)
                   if T > 0 else np.nan)
    return res


def group_test(a, b, method: str = "Mann-Whitney") -> dict:
    """Compare two independent groups with a two-sided Mann-Whitney U test.

    Returns n_a, n_b, stat, p and the rank-biserial effect, which is positive
    when a > b. The exact method is used below 30 subjects.
    """
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    res = dict(n_a=int(a.size), n_b=int(b.size), stat=np.nan, p=np.nan,
               effect=np.nan, effect_name="rank-biserial", test=method)
    if a.size < 2 or b.size < 2:
        return res
    mth = "exact" if (a.size + b.size) < 30 else "asymptotic"
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided", method=mth)
    res.update(stat=float(u), p=float(p),
               effect=float(2 * u / (a.size * b.size) - 1))
    return res


def correlation(x, y, method: str = "Spearman") -> dict:
    """Return n, rho and p of a Spearman or Pearson correlation (needs n >= 4)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    res = dict(n=int(x.size), rho=np.nan, p=np.nan, method=method)
    if x.size < 4:
        return res
    rho, p = (stats.spearmanr if method == "Spearman" else stats.pearsonr)(x, y)
    res.update(rho=float(rho), p=float(p))
    return res



def _wide_pair(long, pair):
    """Pivot one pair to a subject by condition table with Baseline and Followup."""
    d = long[long["pair"] == pair]
    w = d.pivot_table(index="clinical_id", columns="condition", values="value", aggfunc="mean")
    for c in (BASELINE_LABEL, FOLLOWUP_LABEL):
        if c not in w.columns:
            w[c] = np.nan
    return w


def pair_wide(long, pair):
    """Return the Baseline and Followup values of one pair per subject."""
    return _wide_pair(long, pair)


def _change(w, metric="delta"):
    """Return the per-subject change: delta, log ratio or percent change."""
    if metric == "delta":
        return w[FOLLOWUP_LABEL] - w[BASELINE_LABEL]
    if metric == "logratio":
        return np.log(w[FOLLOWUP_LABEL] / w[BASELINE_LABEL])
    return (w[FOLLOWUP_LABEL] - w[BASELINE_LABEL]) / w[BASELINE_LABEL] * 100


def _levels(lookup, col) -> dict:
    """Map each level of a strata column to its set of clinical_ids, skipping blanks."""
    out = {}
    for lvl, g in lookup.dropna(subset=[col]).groupby(col):
        if str(lvl).lower() not in ("nan", ""):
            out[str(lvl)] = set(g["clinical_id"])
    return out


def prepost_analysis(long, lookup, pairs=None, n_boot: int = 5000) -> pd.DataFrame:
    """Run general and stratified Baseline to Followup tests for each pair.

    Row kinds are 'general' (all subjects, paired), 'within' (paired inside each
    level, descriptive) and 'between' (Mann-Whitney on the per-subject delta,
    inferential). Each row also carries descriptive statistics with a bootstrap
    CI. p_corr is BH-FDR within each (kind, stratum) family; n_boot only affects
    the CI precision.
    """
    pairs = pairs or sorted(long["pair"].unique())
    rows = []
    for pair in pairs:
        w = _wide_pair(long, pair).dropna(subset=[BASELINE_LABEL, FOLLOWUP_LABEL])
        ch = _change(w, "delta")
        base = dict(pair=pair)

        r = paired_test(w[BASELINE_LABEL], w[FOLLOWUP_LABEL], log=False)
        desc = describe_paired(w[BASELINE_LABEL], w[FOLLOWUP_LABEL], n_boot=n_boot)
        rows.append({**base, "kind": "general", "stratum": "ALL", "level": "ALL", **r, **desc})

        for col, pretty in STRATA:
            if col not in lookup.columns:
                continue
            lv = _levels(lookup, col)
            deltas = {}
            for level, ids in lv.items():
                wl = w[w.index.isin(ids)]
                if len(wl) == 0:
                    continue
                deltas[level] = ch[ch.index.isin(ids)]
                rr = paired_test(wl[BASELINE_LABEL], wl[FOLLOWUP_LABEL], log=False)
                dd = describe_paired(wl[BASELINE_LABEL], wl[FOLLOWUP_LABEL], n_boot=n_boot)
                rows.append({**base, "kind": "within", "stratum": pretty, "level": level, **rr, **dd})
            if len(deltas) == 2:
                (la, da), (lb, db) = deltas.items()
                gg = group_test(da, db)
                bd = describe_between(da, db, n_boot=n_boot)
                rows.append({**base, "kind": "between", "stratum": pretty, "level": f"{la} vs {lb}", **gg, **bd})

    out = pd.DataFrame(rows)
    out["p_corr"] = np.nan
    for keys, idx in out.groupby(["kind", "stratum"]).groups.items():
        out.loc[idx, "p_corr"] = bh_fdr(out.loc[idx, "p"].values)
    return out.sort_values(["kind", "stratum", "pair"], na_position="first").reset_index(drop=True)


def correlation_analysis(long, clinical, lookup, pairs=None) -> pd.DataFrame:
    """Correlate the connectivity delta with clinical percent change per pair.

    Runs general and within-stratum correlations. There is no between-group
    contrast; that question is answered by the 'between' rows of
    prepost_analysis. p_corr is BH-FDR within each (kind, stratum) family.
    """
    pairs = pairs or sorted(long["pair"].unique())
    clin = clinical.set_index("clinical_id")
    measures = [c for c in CLINICAL_COLS if c in clinical.columns]
    rows = []
    for pair in pairs:
        w = _wide_pair(long, pair).dropna(subset=[BASELINE_LABEL, FOLLOWUP_LABEL])
        ch = _change(w, "delta").rename("chg").to_frame().join(clin, how="inner")
        base = dict(pair=pair)

        for meas in measures:
            rows.append({**base, "kind": "general", "stratum": "ALL", "level": "ALL",
                         "measure": meas, **correlation(ch["chg"], ch[meas])})
        for col, pretty in STRATA:
            if col not in lookup.columns:
                continue
            for level, ids in _levels(lookup, col).items():
                sub = ch[ch.index.isin(ids)]
                for meas in measures:
                    rows.append({**base, "kind": "within", "stratum": pretty, "level": level,
                                 "measure": meas, **correlation(sub["chg"], sub[meas])})

    out = pd.DataFrame(rows)
    out["p_corr"] = np.nan
    for keys, idx in out.groupby(["kind", "stratum"]).groups.items():
        out.loc[idx, "p_corr"] = bh_fdr(out.loc[idx, "p"].values)
    return out.sort_values(["kind", "stratum", "pair", "measure"], na_position="first").reset_index(drop=True)
