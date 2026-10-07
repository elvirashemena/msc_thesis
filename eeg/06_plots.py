"""Draw the EEG figures from the cached features and results.

Cheap step: reads only CSV caches, so editing a figure never reruns a PSD or
wPLI, and MNE is not needed. Draws the clinical symptom reduction, pre to
post panels with the general q, rho heatmaps, between-group effect-size
heatmaps, change scatter plots and two-group split panels.

Inputs
------
features/*_long.csv, results/*_prepost.csv, results/*_correlations.csv
    Outputs of 01_spectral_power.py to 05_correlations.py.
QUEST_XLSX, LOOKUP_XLSX
    Questionnaire and ID lookup tables.

Outputs
-------
figures/
    PNG and PDF of every figure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eeg_compute as ec

FEAT = Path("features")
RES = Path("results")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
QUEST_XLSX = Path(r"C:\Users\jojos\OneDrive\Desktop\EEG Dashboard\questionnaire_converted.xlsx")
LOOKUP_XLSX = Path(r"C:\Users\jojos\OneDrive\Desktop\EEG Dashboard\id_lookup.xlsx")

plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})
UNIT_ORDER = {"power": list(ec.BANDS), "wpli": list(ec.CONN_LEVELS), "faa": [None]}


 
def _feature_long(feature):
    """Load the feature cache, or return None if it does not exist."""
    p = FEAT / ec.FEATURES[feature]["csv"]
    return pd.read_csv(p, dtype={"eeg_id": str}) if p.exists() else None
 
 
def _facet_slice(df, feature, band):
    """Filter a table to one band for faceted features such as wPLI.

    Returns (sliced_df, band_used, file_tag); non-faceted features pass through.
    """
    fac = ec.FEATURES[feature].get("facet")
    if not fac or df is None or fac not in df.columns:
        return df, None, ""
    b = band or ec.FEATURES[feature].get("primary_band")
    return df[df[fac] == b], b, f"_{b}"
 
 
def _roi_footer(fig, feature):
    """Add the electrodes of each wPLI region to the figure, read from CONN_LEVELS."""
    if feature != "wpli":
        return
    parts = [f"{lvl} = {', '.join(ch)}" for lvl, ch in ec.CONN_LEVELS.items() if ch]
    if parts:
        fig.text(0.5, -0.02, "ROI:  " + "   |   ".join(parts) +
                 "   (global = all analysed electrodes)",
                 ha="center", va="top", fontsize=7.5, color="#666")
 
 
def _units(feature, long):
    """Return the units (bands or regions) of a feature in plotting order."""
    if ec.FEATURES[feature]["unit"] is None:
        return [None]
    order = UNIT_ORDER.get(feature, sorted(long[ec.FEATURES[feature]["unit"]].unique()))
    present = set(long[ec.FEATURES[feature]["unit"]].unique())
    return [u for u in order if u in present]
 
 
def _paired_panel(ax, pre, post, title, qtext):
    """Draw one pre to post panel with mean and SD, subject dots and lines."""
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    lo = min(pre.min(), post.min()); allpos = lo > 0
    for a, b in zip(pre, post):
        ax.plot([0, 1], [a, b], color=ec.COLOR_LINE, lw=.7, alpha=.55, zorder=2)
    for x, v, c in ((0, pre, ec.COLOR_PRE), (1, post, ec.COLOR_POST)):
        if allpos:
            ax.bar(x, v.mean(), .55, color=c, edgecolor="black", zorder=1)
        else:
            ax.plot([x - .28, x + .28], [v.mean()] * 2, color=c, lw=5, zorder=1)
        ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="none", ecolor="black",
                    capsize=4, zorder=4)
        ax.scatter(np.full(v.size, x), v, s=18, color=ec.COLOR_DOT, alpha=.8,
                   edgecolor="white", linewidth=.4, zorder=5)
    if not allpos:
        ax.axhline(0, color="#bbb", lw=1, ls="--", zorder=0)
    ax.set_xticks([0, 1], ["Baseline", "Followup"])
    ax.set_xlim(-.5, 1.5)
    ax.set_title(f"{title}\n{qtext}", fontsize=10)
 
 
def fig_prepost(feature, cond, band=None):
    """Plot the general pre to post panels of one feature and condition."""
    long = _feature_long(feature)
    if long is None:
        return
    cfg, ucol = ec.FEATURES[feature], ec.FEATURES[feature]["unit"]
    long, bnd, btag = _facet_slice(long, feature, band)
    res = pd.read_csv(RES / f"{feature}_prepost.csv")
    res, _, _ = _facet_slice(res, feature, band)
    gen = res[(res.kind == "general") & (res.condition == cond)]
    units = _units(feature, long)
    fig, axes = plt.subplots(1, len(units), figsize=(3.6 * len(units), 4.4), squeeze=False)
    for ax, unit in zip(axes[0], units):
        filt = {"condition": cond} | ({ucol: unit} if ucol else {})
        w = ec._wide(long, cfg["value"], **filt).dropna(subset=["Baseline", "Followup"])
        row = gen[gen.unit == unit] if ucol else gen
        q = float(row.p_corr.iloc[0]) if len(row) else np.nan
        p = float(row.p.iloc[0]) if len(row) else np.nan
        star = ec.sig_stars(q) or "n.s."
        _paired_panel(ax, w["Baseline"], w["Followup"],
                      f"{unit or feature}", f"p={p:.3f} q={q:.3f} {star}")
    axes[0][0].set_ylabel(cfg["value"].replace("_", " "))
    bt = f" [{bnd}]" if bnd else ""
    fig.suptitle(f"{feature.upper()}{bt} Baseline->Followup — {cond.upper()} (n=20, general)",
                 y=1.02)
    fig.tight_layout()
    _roi_footer(fig, feature)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"{feature}_prepost_{cond}{btag}.{e}")
    plt.close(fig)
 
 
def fig_prepost_stratified(feature, cond, unit=None, band=None):
    """Plot one unit pre to post across all five strata.

    Each panel shows the two subgroups with subject lines, the within-group
    Wilcoxon stars (raw p, descriptive) and a bracket with the between-group
    Mann-Whitney q (inferential).
    """
    long = _feature_long(feature)
    if long is None:
        return
    cfg, ucol = ec.FEATURES[feature], ec.FEATURES[feature]["unit"]
    long, bnd, btag = _facet_slice(long, feature, band)
    lookup = ec.load_lookup(LOOKUP_XLSX)
    res = pd.read_csv(RES / f"{feature}_prepost.csv")
    res, _, _ = _facet_slice(res, feature, band)
    res_c = res[res.condition == cond]
    filt = {"condition": cond} | ({ucol: unit} if ucol else {})
    w = ec._wide(long, cfg["value"], **filt).dropna(subset=["Baseline", "Followup"])
    lo = min(w["Baseline"].min(), w["Followup"].min())
    hi = max(w["Baseline"].max(), w["Followup"].max())
    allpos = lo > 0
    pad = (hi - lo) * 0.28
 
    fig, axes = plt.subplots(1, len(ec.STRATA), figsize=(3.0 * len(ec.STRATA), 4.6),
                             squeeze=False, sharey=True)
    for ax, (col, pretty) in zip(axes[0], ec.STRATA):
        levels = list(ec._levels(lookup, col).items())
        xpos = [(0, 1), (2.4, 3.4)]
        for gi, (level, ids) in enumerate(levels[:2]):
            sub = w[w.index.isin(ids)]
            if sub.empty:
                continue
            xp, xf = xpos[gi]
            for a, b in zip(sub["Baseline"], sub["Followup"]):
                ax.plot([xp, xf], [a, b], color=ec.COLOR_LINE, lw=.6, alpha=.5, zorder=2)
            for x, v, c in ((xp, sub["Baseline"], ec.COLOR_PRE),
                            (xf, sub["Followup"], ec.COLOR_POST)):
                m = v.mean()
                if allpos:
                    ax.bar(x, m, .8, color=c, edgecolor="black", zorder=1)
                else:
                    ax.plot([x - .38, x + .38], [m, m], color=c, lw=5, zorder=1)
                ax.errorbar(x, m, yerr=v.std(ddof=1), fmt="none", ecolor="black",
                            capsize=3, zorder=4)
                ax.scatter(np.full(len(v), x), v, s=12, color=ec.COLOR_DOT,
                           alpha=.75, zorder=5)
            wr = res_c[(res_c.kind == "within") & (res_c.stratum == pretty) &
                       (res_c.level == level) & ((res_c.unit == unit) | (ucol is None))]
            if len(wr):
                ax.text((xp + xf) / 2, hi + pad * .35,
                        ec.sig_stars(wr.p.iloc[0]) or "n.s.", ha="center", fontsize=9)
            ax.text((xp + xf) / 2, lo - pad * .7, str(level)[:10], ha="center",
                    fontsize=8, color="#333")
        br = res_c[(res_c.kind == "between") & (res_c.stratum == pretty) &
                   ((res_c.unit == unit) | (ucol is None))]
        if len(br):
            y = hi + pad
            ax.plot([0.5, 0.5, 2.9, 2.9], [y, y + pad * .12, y + pad * .12, y],
                    color="black", lw=1)
            ax.text(1.7, y + pad * .16,
                    f"Δ {ec.sig_stars(br.p_corr.iloc[0]) or 'n.s.'} (q={br.p_corr.iloc[0]:.3f})",
                    ha="center", fontsize=8)
        ax.set_title(pretty, fontsize=10)
        ax.set_xticks([])
        ax.set_xlim(-.7, 4.1)
        if not allpos:
            ax.axhline(0, color="#bbb", lw=1, ls="--", zorder=0)
    axes[0][0].set_ylabel(cfg["value"].replace("_", " "))
    axes[0][0].set_ylim(lo - pad * 1.1, hi + pad * 1.7)
    bt = f" [{bnd}]" if bnd else ""
    fig.suptitle(f"{feature.upper()}{bt} {unit or ''} — {cond.upper()} stratified "
                 f"(bars=mean±SD; within=Wilcoxon raw p, Δ=Mann-Whitney q on change)",
                 y=1.03, fontsize=10)
    fig.tight_layout()
    _roi_footer(fig, feature)
    tag = unit or feature
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"{feature}_stratified_{cond}_{tag}{btag}.{e}")
    plt.close(fig)
 
 
def fig_corr_heat(feature, cond, band=None):
    """Plot the general rho heatmap of clinical measure by unit for one condition."""
    res = pd.read_csv(RES / f"{feature}_correlations.csv")
    res, bnd, btag = _facet_slice(res, feature, band)
    g = res[(res.kind == "general") & (res.condition == cond)].copy()
    if g.empty:
        return
    ucol = ec.FEATURES[feature]["unit"]
    long0 = _feature_long(feature)
    long0, _, _ = _facet_slice(long0, feature, band)
    g["unit"] = g["unit"].fillna(feature) if ucol else feature
    units = _units(feature, long0) if long0 is not None else sorted(g.unit.unique())
    units = [u if u is not None else feature for u in units]
    measures = sorted(g.measure.unique())
    R = g.pivot(index="measure", columns="unit", values="rho").reindex(index=measures, columns=units)
    Q = g.pivot(index="measure", columns="unit", values="p_corr").reindex(index=measures, columns=units)
    fig, ax = plt.subplots(figsize=(1.1 * len(units) + 2, .9 * len(measures) + 1.6))
    im = ax.imshow(R.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(units)), units)
    ax.set_yticks(range(len(measures)), measures)
    for i in range(len(measures)):
        for j in range(len(units)):
            r, q = R.values[i, j], Q.values[i, j]
            if np.isfinite(r):
                ax.text(j, i, f"{r:+.2f}{ec.sig_stars(q)}", ha="center", va="center",
                        fontsize=9, color="white" if abs(r) > .55 else "black")
    bt = f" [{bnd}]" if bnd else ""
    ax.set_title(f"{feature.upper()}{bt} change vs symptom change — {cond.upper()}\n"
                 f"Spearman rho, * q<.05 (neg = feature rises as symptoms improve)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, label="rho", fraction=.046)
    fig.tight_layout()
    _roi_footer(fig, feature)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"{feature}_corr_{cond}{btag}.{e}")
    plt.close(fig)
 
 
def fig_drivers(feature, band=None):
    """Plot the between-group rank-biserial effects as a stratum by unit heatmap."""
    res = pd.read_csv(RES / f"{feature}_prepost.csv")
    res, bnd, btag = _facet_slice(res, feature, band)
    b = res[res.kind == "between"].copy()
    if b.empty:
        return
    ucol = ec.FEATURES[feature]["unit"]
    b["unit"] = b["unit"].fillna(feature) if ucol else feature
    b["col"] = b["condition"].str.upper() + " / " + b["unit"].astype(str)
    E = b.pivot_table(index="stratum", columns="col", values="effect")
    Q = b.pivot_table(index="stratum", columns="col", values="p_corr")
    fig, ax = plt.subplots(figsize=(1.0 * E.shape[1] + 2, .7 * E.shape[0] + 1.6))
    im = ax.imshow(E.values, cmap="PuOr", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(E.shape[1]), E.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(E.shape[0]), E.index)
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            e, q = E.values[i, j], Q.values[i, j]
            if np.isfinite(e):
                ax.text(j, i, f"{e:+.2f}{ec.sig_stars(q)}", ha="center", va="center",
                        fontsize=8, color="white" if abs(e) > .6 else "black")
    bt = f" [{bnd}]" if bnd else ""
    ax.set_title(f"{feature.upper()}{bt} — between-group change (Mann-Whitney on per-subject "
                 f"Δ)\nrank-biserial effect, * q<.05  |  EXPLORATORY", fontsize=9)
    fig.colorbar(im, ax=ax, label="effect (a>b +)", fraction=.046)
    fig.tight_layout()
    _roi_footer(fig, feature)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"{feature}_drivers{btag}.{e}")
    plt.close(fig)
 
 
GROUP_STYLE = {
    "responder_HAMD17": {"responder": ("#c0392b", "o"), "non-responder": ("#4f6d9a", "s")},
    "responder_BDI":    {"responder": ("#c0392b", "o"), "non-responder": ("#4f6d9a", "s")},
    "sex":              {"Female": ("#c0392b", "o"), "Male": ("#4f6d9a", "s")},
}
_FALLBACK = [("#c0392b", "o"), ("#4f6d9a", "s"), ("#2e8b57", "^"), ("#8e44ad", "D")]
 
 
def fig_scatter(feature, cond, unit, measure, group_by=None, band=None):
    """Plot feature change against symptom change for one unit and measure.

    If group_by is given, points are coloured by subgroup; the fit line and the
    rho, p and q label always use the full sample.
    """
    long = _feature_long(feature)
    if long is None:
        return
    cfg, ucol = ec.FEATURES[feature], ec.FEATURES[feature]["unit"]
    long, bnd, btag = _facet_slice(long, feature, band)
    clin = ec.load_clinical(QUEST_XLSX).set_index("eeg_id")
    filt = {"condition": cond} | ({ucol: unit} if ucol else {})
    w = ec._wide(long, cfg["value"], **filt).dropna(subset=["Baseline", "Followup"])
    ch = ec._change(w, cfg["change"]).rename("chg").to_frame().join(clin, how="inner")
    res = pd.read_csv(RES / f"{feature}_correlations.csv")
    res, _, _ = _facet_slice(res, feature, band)
    q = res[(res.kind == "general") & (res.condition == cond) & (res.measure == measure) &
            ((res.unit == unit) | (ucol is None))]
 
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    if group_by:
        lk = ec.load_lookup(LOOKUP_XLSX).set_index("eeg_id")[group_by].astype(str)
        ch = ch.join(lk.rename("grp"), how="left")
        style = GROUP_STYLE.get(group_by, {})
        for i, (lvl, g) in enumerate(ch.groupby("grp")):
            color, marker = style.get(lvl, _FALLBACK[i % len(_FALLBACK)])
            ax.scatter(g["chg"], g[measure], s=64, color=color, marker=marker,
                       edgecolor="white", linewidth=.7, zorder=3,
                       label=f"{lvl} (n={len(g)})")
        ax.legend(frameon=False, fontsize=8, loc="best")
    else:
        ax.scatter(ch["chg"], ch[measure], s=48, color=ec.COLOR_A,
                   edgecolor="white", zorder=3)
 
    d = ch[["chg", measure]].dropna()
    if len(d) >= 3:
        m = np.polyfit(d["chg"], d[measure], 1)
        xx = np.linspace(d["chg"].min(), d["chg"].max(), 50)
        ax.plot(xx, np.polyval(m, xx), "--", color="#444", zorder=2)
    ax.axhline(0, color="#ccc", lw=.8); ax.axvline(0, color="#ccc", lw=.8)
    rho = float(q.rho.iloc[0]) if len(q) else np.nan
    qq = float(q.p_corr.iloc[0]) if len(q) else np.nan
    pp = float(q.p.iloc[0]) if len(q) else np.nan
    ax.set_xlabel(f"{unit or feature} {cfg['change']} change")
    ax.set_ylabel(f"{measure}  (- = improvement)")
    gtxt = f"  |  coloured by {group_by}" if group_by else ""
    bt = f" [{bnd}]" if bnd else ""
    ax.set_title(f"{feature.upper()}{bt} {unit or ''} vs {measure} — {cond.upper()}{gtxt}\n"
                 f"full-sample Spearman rho={rho:+.2f}, p={pp:.3f}, q={qq:.3f}", fontsize=9)
    fig.tight_layout()
    _roi_footer(fig, feature)
    gtag = f"_by-{group_by}" if group_by else ""
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"{feature}_scatter_{cond}_{unit or 'faa'}_{measure}{gtag}{btag}.{e}")
    plt.close(fig)
 
 
def fig_group_split(feature, cond, unit, group_by="responder_HAMD17", band=None):
    """Plot pre to post for the two subgroups of group_by in one panel.

    Shows mean and SD bars, subject dots and lines, the within-group Wilcoxon p
    over each pair and a bracket with the between-group r_rb, p and q.
    """
    long = _feature_long(feature)
    if long is None:
        return
    cfg, ucol = ec.FEATURES[feature], ec.FEATURES[feature]["unit"]
    long, bnd, btag = _facet_slice(long, feature, band)
    lookup = ec.load_lookup(LOOKUP_XLSX)
    pretty = dict(ec.STRATA)[group_by]
    res = pd.read_csv(RES / f"{feature}_prepost.csv")
    res, _, _ = _facet_slice(res, feature, band)
    res = res[(res.condition == cond) & ((res.unit == unit) | (ucol is None))]
 
    filt = {"condition": cond} | ({ucol: unit} if ucol else {})
    w = ec._wide(long, cfg["value"], **filt).dropna(subset=["Baseline", "Followup"])
    style = GROUP_STYLE.get(group_by, {})
    levels = list(style) if style else sorted(ec._levels(lookup, group_by))
    levels = [l for l in levels if l in ec._levels(lookup, group_by)]
 
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    xpos = [(0, 1), (2.4, 3.4)]
    allvals = []
    for gi, level in enumerate(levels[:2]):
        ids = ec._levels(lookup, group_by)[level]
        sub = w[w.index.isin(ids)]
        if sub.empty:
            continue
        xp, xf = xpos[gi]
        pre, post = sub["Baseline"].values, sub["Followup"].values
        allvals += [pre, post]
        for a, b in zip(pre, post):
            ax.plot([xp, xf], [a, b], color=ec.COLOR_LINE, lw=.7, alpha=.55, zorder=2)
        for x, v, col in ((xp, pre, ec.COLOR_PRE), (xf, post, ec.COLOR_POST)):
            ax.bar(x, v.mean(), .8, color=col, edgecolor="black", zorder=1)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color="black", capsize=4, zorder=4)
            ax.scatter(np.full(len(v), x), v, s=18, color=ec.COLOR_DOT, alpha=.8, zorder=5)
        wr = res[(res.kind == "within") & (res.stratum == pretty) & (res.level == level)]
        pw = float(wr.p.iloc[0]) if len(wr) else np.nan
        gcol = style.get(level, ("#333", "o"))[0]
        ax.text((xp + xf) / 2, 0.5, f"{level} (n={len(sub)})", ha="center",
                transform=ax.get_xaxis_transform(), color=gcol, fontsize=9)
        if np.isfinite(pw):
            ax.text((xp + xf) / 2, max(pre.max(), post.max()) * 1.04,
                    f"p={pw:.3f}", ha="center", fontsize=8)
 
    top = max(v.max() for v in allvals)
    br = res[(res.kind == "between") & (res.stratum == pretty)]
    if len(br):
        y = top * 1.16
        ax.plot([0.5, 0.5, 2.9, 2.9], [y, y * 1.03, y * 1.03, y], color="black", lw=1)
        ax.text(1.7, y * 1.05,
                f"between-group $\\Delta$:  r_rb={float(br.effect.iloc[0]):+.2f}, "
                f"p={float(br.p.iloc[0]):.3f}, q={float(br.p_corr.iloc[0]):.3f}",
                ha="center", fontsize=8.5)
    ax.set_xticks([0, 1, 2.4, 3.4], ["Baseline", "Followup", "Baseline", "Followup"])
    ax.set_xlim(-.6, 4.0); ax.set_ylim(0, top * 1.3)
    val = "wPLI" if feature == "wpli" else cfg["value"].replace("_", " ")
    bt = f"{bnd} " if bnd else ""
    ax.set_ylabel(f"{cond.upper()} {unit or ''} {bt}{val}")
    ax.set_title(f"{feature.upper()} {bt}{unit or ''} — {cond.upper()} by {pretty}",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); _roi_footer(fig, feature)
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"{feature}_split_{cond}_{unit or 'faa'}_{group_by}{btag}.{e}")
    plt.close(fig)
 
 
def fig_symptom_reduction(scales=("HAMD17", "BDI"), group_by="responder_HAMD17"):
    """Plot clinical pre to post per scale, coloured by responder status.

    The titles give mean and SD, percent reduction and the paired Wilcoxon p.
    """
    q = pd.read_excel(QUEST_XLSX)
    q["eeg_id"] = q[("ID" if "ID" in q.columns else "eeg_id")].astype(str).str.extract(r"(\d+)")[0]
    lookup = ec.load_lookup(LOOKUP_XLSX).set_index("eeg_id")[group_by].astype(str)
    style = GROUP_STYLE.get(group_by, {})
    from scipy import stats
    fig, axes = plt.subplots(1, len(scales), figsize=(4.8 * len(scales), 4.6), squeeze=False)
    for ax, sc in zip(axes[0], scales):
        pre_c, post_c = f"{sc}_pre", f"{sc}_post"
        if pre_c not in q or post_c not in q:
            ax.set_title(f"{sc}: columns {pre_c}/{post_c} not found"); continue
        d = q[["eeg_id", pre_c, post_c]].dropna().set_index("eeg_id")
        d = d.join(lookup.rename("grp"), how="left")
        pre, post = d[pre_c].values, d[post_c].values
        for (_, r) in d.iterrows():
            col = style.get(str(r["grp"]), ("#9aa5b1", "o"))[0]
            ax.plot([0, 1], [r[pre_c], r[post_c]], color=col, lw=.8, alpha=.6, zorder=2)
            ax.scatter([0, 1], [r[pre_c], r[post_c]], color=col, s=22, zorder=3)
        for x, v, col in ((0, pre, ec.COLOR_PRE), (1, post, ec.COLOR_POST)):
            ax.plot([x - .2, x + .2], [v.mean()] * 2, color=col, lw=5, zorder=1)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color="black", capsize=5, zorder=4)
        try:
            _, p = stats.wilcoxon(post, pre)
        except ValueError:
            p = np.nan
        red = (1 - post.mean() / pre.mean()) * 100
        ax.set_xticks([0, 1], ["Baseline", "Followup"]); ax.set_xlim(-.4, 1.4)
        ax.set_ylabel(sc); ax.set_ylim(bottom=0)
        ax.set_title(f"{sc}: {pre.mean():.1f}±{pre.std(ddof=1):.1f} -> "
                     f"{post.mean():.1f}±{post.std(ddof=1):.1f}\n"
                     f"-{red:.0f}%,  Wilcoxon p={p:.4f} (n={len(d)})", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c[0],
                          markersize=8, label=l) for l, c in style.items()]
    if handles:
        axes[0][0].legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Clinical response to aiTBS", y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"symptom_reduction_{group_by}.{e}")
    plt.close(fig)
 
 
def main():
    """Draw the selected custom figures, then the standard figures of each feature."""
    fig_symptom_reduction(scales=("HAMD17", "BDI"), group_by="responder_HAMD17")
    for meas in ("HAMD17_percentage", "BDI_percentage"):
        for gb in ("responder_HAMD17", "sex"):
            fig_scatter("power", "ec", "theta", meas, group_by=gb)
    fig_group_split("wpli", "eo", "global", group_by="responder_HAMD17", band="theta")
    for cond in ("ec", "eo"):
        fig_group_split("wpli", cond, "central", group_by="sex", band="alpha")
        fig_group_split("power", cond, "alpha", group_by="sex")
 
    for feature in ec.FEATURES:
        if not (RES / f"{feature}_prepost.csv").exists():
            continue
        bands = list(ec.CONN_BANDS) if ec.FEATURES[feature].get("facet") else [None]
        for band in bands:
            for cond in ("ec", "eo"):
                fig_prepost(feature, cond, band=band)
                fig_corr_heat(feature, cond, band=band)
            fig_drivers(feature, band=band)
    print(f"Figures -> {FIG}")
 
 
if __name__ == "__main__":
    main()
