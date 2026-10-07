"""Shared computation engine for the EEG analysis.

Used by every EEG script and organised in four parts:

1. IO and epoching: load preprocessed EO and EC recordings, make clean 4 s
   epochs and match epoch counts across sessions.
2. Features: global band power, frontal alpha asymmetry (FAA) and wPLI
   connectivity.
3. Statistics: paired tests, correlations, Mann-Whitney tests, BH-FDR and
   bootstrap CIs.
4. Stratification: a general test first, then within-group (descriptive) and
   between-group (inferential) tests for the five strata.

Feature extraction (01 to 03) reads the EEG once and writes *_long.csv
caches; analysis (04, 05) and plotting (06) read only the caches. MNE is
imported inside the compute functions, so the analysis scripts run without
it.

Notes
-----
- Baseline is pre (T1) and Followup is post (T3); EC is the primary
  condition.
- Change metrics (see FEATURES): power uses percent change with the paired
  test on log power, FAA uses delta because it is a log ratio that crosses
  zero, and wPLI uses percent change without log.
- Clinical percent change is negative for improvement, so a feature that
  rises in improvers correlates negatively with it.
- Rank-based tests are primary and all stratified results are exploratory.
  An effect in one level but not the other is not a group difference; the
  between-group test on the change is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}
ALPHA_BAND = (8.0, 13.0)
THETA_BAND = (4.0, 8.0)
PSD_FMIN, PSD_FMAX = 1.0, 45.0
EPOCH_LEN_S = 4.0
MIN_EPOCHS = 15
WARN_EPOCHS = 30
MATCH_EPOCHS_ACROSS_SESSIONS = True
DROP_CHANNELS = ["TP9", "TP10", "M1", "M2", "A1", "A2"]


FAA_LEFT = ["Fp1", "F7", "F3"]
FAA_RIGHT = ["Fp2", "F8", "F4"]
FAA_USE_RELATIVE = True
 
CONN_LEVELS = {
    "global":        None,
    "frontal":       ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "central":       ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4"],
    "parietal":      ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8"],
    "occipital":     ["POz", "O1", "Oz", "O2"],
}
CONN_MIN_CH = 3
 
CONN_BANDS = dict(BANDS)
 
STRATA = [("sex", "sex"),
          ("responder_HAMD17", "HAM-D responder"),
          ("responder_BDI", "BDI responder"),
          ("Comorbidity", "comorbidity"),
          ("clinical_group", "clinical group")]
 
CLINICAL_COLS = ["HAMD17_percentage", "BDI_percentage"]
 
FEATURES = {
    "power": dict(csv="power_long.csv", value="rel_power", unit="band",
                  log=True, change="pct"),
    "faa":   dict(csv="faa_long.csv", value="faa", unit=None,
                  log=False, change="delta"),
    "wpli":  dict(csv="connectivity_long.csv", value="wpli", unit="level",
                  facet="band", primary_band="theta", log=False, change="pct"),
}
 
COLOR_PRE, COLOR_POST = "#b9c6dd", "#f2b49a"
COLOR_DOT, COLOR_LINE = "#3d4a5c", "#9aa5b1"
COLOR_A, COLOR_B = "#c0504d", "#4f6d9a"
 
TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
 
 
 
def parse_filename(path: Path) -> tuple[str, str, str] | None:
    """Parse 'baseline_113_raw_ec.fif.gz' into ('113', 'Baseline', 'ec'), or None."""
    stem = Path(path).name.lower().replace(".fif.gz", "").replace(".fif", "")
    p = stem.split("_")
    if len(p) < 4 or p[-1] not in ("ec", "eo"):
        return None
    session = {"baseline": "Baseline", "followup": "Followup"}.get(p[0])
    return (p[1], session, p[-1]) if session else None
 
 
def index_files(data_dir: Path, cond: str) -> pd.DataFrame:
    """List the recordings of one condition, keeping subjects with both sessions."""
    rows = []
    for f in sorted(Path(data_dir).rglob(f"*_raw_{cond}.fif*")):
        parsed = parse_filename(f)
        if parsed:
            rows.append({"eeg_id": parsed[0], "session": parsed[1], "path": f})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No *_raw_{cond}.fif* under {data_dir}")
    complete = df.groupby("eeg_id")["session"].nunique() == 2
    dropped = sorted(complete[~complete].index)
    if dropped:
        print(f"  ! [{cond}] dropped (missing a session): {dropped}")
    return df[df["eeg_id"].isin(complete[complete].index)].reset_index(drop=True)
 
 
def load_raw(path: Path, picks: list[str] | None = None):
    """Load a recording and drop the reference and mastoid channels.

    picks=None keeps all remaining channels (global level); a list keeps only
    those present. Returns None if too few channels remain.
    """
    import mne
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raw.drop_channels([c for c in DROP_CHANNELS if c in raw.ch_names])
    if picks is None:
        return raw if len(raw.ch_names) >= 2 else None
    keep = [c for c in picks if c in raw.ch_names]
    if len(keep) < len(picks):
        print(f"  ! {Path(path).name}: missing {set(picks) - set(keep)} -> subset")
    return raw.pick(keep) if len(keep) >= 1 else None
 
 
def make_epochs(raw, name: str = ""):
    """Return clean fixed-length epochs, or None if there is too little clean data.

    Epochs overlapping a 'bad_' annotation from preprocessing are rejected.
    """
    import mne
    try:
        ep = mne.make_fixed_length_epochs(raw, duration=EPOCH_LEN_S, overlap=0.0,
                                          preload=True, reject_by_annotation=True,
                                          verbose="ERROR")
    except ValueError:
        print(f"  ! {name}: shorter than one {EPOCH_LEN_S:g}s epoch -> skipped")
        return None
    if len(ep) < MIN_EPOCHS:
        print(f"  ! {name}: {len(ep)} clean epochs "
              f"(<{MIN_EPOCHS * EPOCH_LEN_S:.0f}s) -> skipped")
        return None
    return ep
 
 
def subsample_epochs(epochs, n: int):
    """Keep n evenly spaced epochs across the recording (deterministic)."""
    if len(epochs) <= n:
        return epochs
    idx = np.unique(np.linspace(0, len(epochs) - 1, n).round().astype(int))
    return epochs[idx]
 
 
def paired_epochs(files_sub, picks):
    """Return epoch-matched Baseline and Followup epochs of one subject.

    Returns ({'Baseline': ep, 'Followup': ep}, available epoch counts), or
    (None, None) if a session is unusable.
    """
    ep, n_avail = {}, {}
    for r in files_sub.itertuples(index=False):
        raw = load_raw(r.path, picks)
        if raw is None:
            continue
        e = make_epochs(raw, name=f"{r.session} {r.eeg_id}")
        if e is not None:
            ep[r.session], n_avail[r.session] = e, len(e)
    if set(ep) != {"Baseline", "Followup"}:
        return None, None
    if MATCH_EPOCHS_ACROSS_SESSIONS:
        n = min(len(e) for e in ep.values())
        ep = {s: subsample_epochs(e, n) for s, e in ep.items()}
    return ep, n_avail
 
 
 
def _psd(epochs, picks=None):
    """Return the Welch PSD averaged over epochs as (power[n_ch, n_freq], freqs)."""
    n_fft = len(epochs.times)
    kw = dict(method="welch", fmin=PSD_FMIN, fmax=PSD_FMAX, n_fft=n_fft,
              n_overlap=0, n_per_seg=n_fft, verbose="ERROR")
    if picks is not None:
        kw["picks"] = picks
    spec = epochs.compute_psd(**kw)
    power, freqs = spec.get_data(return_freqs=True)
    return power.mean(axis=0), freqs
 
 
def band_powers_global(epochs) -> list[dict]:
    """Return the absolute and relative power per band, averaged over all channels."""
    power, freqs = _psd(epochs)
    g = power.mean(axis=0)
    total = TRAPZ(g[(freqs >= PSD_FMIN) & (freqs <= PSD_FMAX)],
                  freqs[(freqs >= PSD_FMIN) & (freqs <= PSD_FMAX)])
    out = []
    for band, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs <= hi)
        ap = float(TRAPZ(g[m], freqs[m]))
        out.append({"band": band, "abs_power": ap, "rel_power": ap / total})
    return out
 
 
def faa(epochs) -> float | None:
    """Return FAA as the mean right minus mean left log alpha power.

    The log is taken per electrode before averaging within each cluster. Returns
    None if a cluster has no channels.
    """
    def cluster(names):
        """Return the mean log alpha power of the available channels in names."""
        picks = [c for c in names if c in epochs.ch_names]
        if not picks:
            return None
        power, freqs = _psd(epochs, picks=picks)
        m = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
        alpha = TRAPZ(power[:, m], freqs[m], axis=1)
        if FAA_USE_RELATIVE:
            alpha = alpha / TRAPZ(power, freqs, axis=1)
        return float(np.mean(np.log(alpha)))
    r, l = cluster(FAA_RIGHT), cluster(FAA_LEFT)
    return None if (r is None or l is None) else r - l
 
 
def band_wpli(epochs, bands: dict | None = None) -> dict:
    """Return the mean wPLI over all electrode pairs for each band.

    All bands are computed in one connectivity call, and wPLI is not
    log-transformed. Theta is the primary band, delta and alpha are specificity
    controls, and beta and gamma may contain muscle artefact.
    """
    from mne_connectivity import spectral_connectivity_epochs
    bands = bands or CONN_BANDS
    fmin = tuple(lo for lo, hi in bands.values())
    fmax = tuple(hi for lo, hi in bands.values())
    con = spectral_connectivity_epochs(
        epochs, method="wpli", mode="multitaper", sfreq=epochs.info["sfreq"],
        fmin=fmin, fmax=fmax, faverage=True, verbose="ERROR")
    data = con.get_data(output="dense")
    if data.ndim == 2:
        data = data[:, :, None]
    tril = np.tril_indices(data.shape[0], k=-1)
    return {band: float(np.nanmean(np.abs(data[:, :, i][tril])))
            for i, band in enumerate(bands)}
 
 
def theta_wpli(epochs) -> float:
    """Return the theta wPLI only (kept for backward compatibility)."""
    return band_wpli(epochs, {"theta": THETA_BAND})["theta"]
 
 
 
def describe(x) -> str:
    """Format the mean and SD of the finite values as text, or 'n/a'."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return "n/a"
    return f"{x.mean():.4g}±{x.std(ddof=1):.4g}" if x.size > 1 else f"{x[0]:.4g}"
 
 
def bh_fdr(pvals) -> np.ndarray:
    """Return Benjamini-Hochberg q-values; NaNs pass through."""
    p = np.asarray(pvals, float)
    q = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return q
    pp = p[idx]
    order = np.argsort(pp)
    m = pp.size
    ranked = np.minimum.accumulate((pp[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    q[idx] = out
    return q
 
 
def sig_stars(p) -> str:
    """Return '***', '**', '*' or '' for p; '' if p is missing."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
 
 
def paired_test(pre, post, log: bool = False,
                method: str = "Wilcoxon") -> dict:
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
 
 
CI_NBOOT = 5000
CI_LEVEL = 0.95
CI_SEED = 42
 
 
def bootstrap_ci(values, n_boot: int = CI_NBOOT, ci: float = CI_LEVEL,
                 seed: int = CI_SEED):
    """Return the percentile bootstrap CI for the mean of one sample (NaN if n < 3)."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)
 
 
def bootstrap_ci_diff(a, b, n_boot: int = CI_NBOOT, ci: float = CI_LEVEL,
                      seed: int = CI_SEED):
    """Return the percentile bootstrap CI for the difference in means a - b."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    boot = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)
 
 
def bootstrap_ci_rho(x, y, n_boot: int = CI_NBOOT, ci: float = CI_LEVEL,
                     seed: int = CI_SEED):
    """Return the percentile bootstrap CI for Spearman rho, resampling subject pairs."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    rx = stats.rankdata(x[idx], axis=1)
    ry = stats.rankdata(y[idx], axis=1)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(r, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)
 
 
def paired_descriptives(pre, post) -> dict:
    """Return mean and SD of pre, post and change, plus a bootstrap CI on the change.

    Always on the raw scale, even when the paired test uses log values.
    """
    pre = np.asarray(pre, float); post = np.asarray(post, float)
    ok = np.isfinite(pre) & np.isfinite(post)
    b, f = pre[ok], post[ok]
    d = f - b
    lo, hi = bootstrap_ci(d)
    m = lambda v: float(np.mean(v)) if v.size else np.nan
    s = lambda v: float(np.std(v, ddof=1)) if v.size > 1 else np.nan
    return dict(mean_pre=m(b), sd_pre=s(b), mean_post=m(f), sd_post=s(f),
                mean_change=m(d), sd_change=s(d), ci_lo=lo, ci_hi=hi)
 
 
def between_descriptives(da, db) -> dict:
    """Return the difference in mean change (a - b) with a bootstrap CI.

    The pre and post fields are NaN; each group's own values are in its 'within'
    row.
    """
    da = np.asarray(da, float); db = np.asarray(db, float)
    da, db = da[np.isfinite(da)], db[np.isfinite(db)]
    lo, hi = bootstrap_ci_diff(da, db)
    diff = float(np.mean(da) - np.mean(db)) if (da.size and db.size) else np.nan
    return dict(mean_pre=np.nan, sd_pre=np.nan, mean_post=np.nan, sd_post=np.nan,
                mean_change=diff, sd_change=np.nan, ci_lo=lo, ci_hi=hi)
 
 
 
def load_lookup(xlsx: Path) -> pd.DataFrame:
    """Load the ID lookup as eeg_id (digits only) plus the strata columns as text."""
    d = pd.read_excel(xlsx)
    d["eeg_id"] = d["eeg_id"].astype(str).str.extract(r"(\d+)")[0]
    for col, _ in STRATA:
        if col in d:
            d[col] = d[col].astype(str).str.strip()
    return d[["eeg_id"] + [c for c, _ in STRATA if c in d]]
 
 
def load_clinical(xlsx: Path) -> pd.DataFrame:
    """Load the questionnaire as eeg_id plus the clinical percent change columns."""
    q = pd.read_excel(xlsx)
    id_col = "ID" if "ID" in q.columns else "eeg_id"
    q["eeg_id"] = q[id_col].astype(str).str.extract(r"(\d+)")[0]
    return q[["eeg_id"] + [c for c in CLINICAL_COLS if c in q.columns]]
 
 
def _wide(long: pd.DataFrame, value: str, **filt) -> pd.DataFrame:
    """Filter the long table and pivot one value to a subject by session table."""
    df = long
    for k, v in filt.items():
        df = df[df[k] == v]
    w = df.pivot_table(index="eeg_id", columns="session", values=value,
                       aggfunc="mean")
    for c in ("Baseline", "Followup"):
        if c not in w.columns:
            w[c] = np.nan
    return w
 
 
def _change(w: pd.DataFrame, metric: str) -> pd.Series:
    """Return the per-subject change: delta, log ratio or percent change."""
    if metric == "delta":
        return w["Followup"] - w["Baseline"]
    if metric == "logratio":
        return np.log(w["Followup"] / w["Baseline"])
    return (w["Followup"] - w["Baseline"]) / w["Baseline"] * 100
 
 
def _levels(lookup: pd.DataFrame, col: str) -> dict[str, set]:
    """Map each level of a strata column to its set of eeg_ids, skipping blanks."""
    out = {}
    for lvl, g in lookup.dropna(subset=[col]).groupby(col):
        if str(lvl).lower() not in ("nan", ""):
            out[str(lvl)] = set(g["eeg_id"])
    return out
 
 
def prepost_analysis(long: pd.DataFrame, feature: str, cond: str,
                     lookup: pd.DataFrame) -> pd.DataFrame:
    """Run general and stratified pre to post tests for one feature and condition.

    Row kinds are 'general' (all subjects, paired), 'within' (paired inside each
    level, descriptive) and 'between' (Mann-Whitney on the per-subject change,
    inferential). Each row carries descriptive statistics with a bootstrap CI.
    p_corr is BH-FDR within each kind, condition and stratum (and band for
    wPLI).
    """
    cfg = FEATURES[feature]
    facet = cfg.get("facet")
    facet_vals = [None] if not facet else sorted(long[facet].dropna().unique())
    units = [None] if cfg["unit"] is None else sorted(long[cfg["unit"]].dropna().unique())
    rows = []
    for fv in facet_vals:
        for unit in units:
            filt = {"condition": cond}
            if cfg["unit"]:
                filt[cfg["unit"]] = unit
            if facet:
                filt[facet] = fv
            w = _wide(long, cfg["value"], **filt).dropna(subset=["Baseline", "Followup"])
            ch = _change(w, cfg["change"])
            base = dict(feature=feature, condition=cond, unit=unit)
            if facet:
                base[facet] = fv
 
            r = paired_test(w["Baseline"], w["Followup"], log=cfg["log"])
            rows.append({**base, "kind": "general", "stratum": "ALL", "level": "ALL",
                         **r, **paired_descriptives(w["Baseline"], w["Followup"])})
 
            for col, pretty in STRATA:
                lv = _levels(lookup, col)
                deltas = {}
                for level, ids in lv.items():
                    wl = w[w.index.isin(ids)]
                    deltas[level] = ch[ch.index.isin(ids)]
                    rr = paired_test(wl["Baseline"], wl["Followup"], log=cfg["log"])
                    rows.append({**base, "kind": "within", "stratum": pretty,
                                 "level": level, **rr,
                                 **paired_descriptives(wl["Baseline"], wl["Followup"])})
                if len(deltas) == 2:
                    (la, da), (lb, db) = deltas.items()
                    gg = group_test(da, db)
                    rows.append({**base, "kind": "between", "stratum": pretty,
                                 "level": f"{la} vs {lb}", **gg,
                                 **between_descriptives(da, db)})
 
    out = pd.DataFrame(rows)
    fdr_cols = [c for c in ["kind", "condition", "stratum", facet] if c and c in out]
    out["p_corr"] = np.nan
    for keys, idx in out.groupby(fdr_cols).groups.items():
        out.loc[idx, "p_corr"] = bh_fdr(out.loc[idx, "p"].values)
    return out.sort_values([c for c in [facet, "unit", "kind", "stratum"] if c],
                           na_position="first").reset_index(drop=True)
 
 
def correlation_analysis(long: pd.DataFrame, clinical: pd.DataFrame, feature: str,
                         cond: str, lookup: pd.DataFrame) -> pd.DataFrame:
    """Correlate feature change with clinical change for one feature and condition.

    Runs general and within-stratum Spearman correlations with bootstrap CIs on
    rho. There is no between-group contrast; that question is answered by
    prepost_analysis. p_corr is BH-FDR within each kind, condition and stratum.
    """
    cfg = FEATURES[feature]
    facet = cfg.get("facet")
    facet_vals = [None] if not facet else sorted(long[facet].dropna().unique())
    units = [None] if cfg["unit"] is None else sorted(long[cfg["unit"]].dropna().unique())
    clin = clinical.set_index("eeg_id")
    measures = [c for c in CLINICAL_COLS if c in clinical.columns]
    rows = []
    for fv in facet_vals:
        for unit in units:
            filt = {"condition": cond}
            if cfg["unit"]:
                filt[cfg["unit"]] = unit
            if facet:
                filt[facet] = fv
            w = _wide(long, cfg["value"], **filt).dropna(subset=["Baseline", "Followup"])
            ch = _change(w, cfg["change"]).rename("chg").to_frame().join(clin, how="inner")
            base = dict(feature=feature, condition=cond, unit=unit)
            if facet:
                base[facet] = fv
 
            for meas in measures:
                lo, hi = bootstrap_ci_rho(ch["chg"].values, ch[meas].values)
                rows.append({**base, "kind": "general", "stratum": "ALL", "level": "ALL",
                             "measure": meas,
                             **correlation(ch["chg"], ch[meas]),
                             "rho_ci_lo": lo, "rho_ci_hi": hi})
            for col, pretty in STRATA:
                for level, ids in _levels(lookup, col).items():
                    sub = ch[ch.index.isin(ids)]
                    for meas in measures:
                        lo, hi = bootstrap_ci_rho(sub["chg"].values, sub[meas].values)
                        rows.append({**base, "kind": "within", "stratum": pretty,
                                     "level": level, "measure": meas,
                                     **correlation(sub["chg"], sub[meas]),
                                     "rho_ci_lo": lo, "rho_ci_hi": hi})
 
    out = pd.DataFrame(rows)
    fdr_cols = [c for c in ["kind", "condition", "stratum", facet] if c and c in out]
    out["p_corr"] = np.nan
    for keys, idx in out.groupby(fdr_cols).groups.items():
        out.loc[idx, "p_corr"] = bh_fdr(out.loc[idx, "p"].values)
    return out.sort_values([c for c in [facet, "unit", "kind", "stratum", "measure"] if c],
                           na_position="first").reset_index(drop=True)
 