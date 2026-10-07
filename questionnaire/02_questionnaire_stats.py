"""Test the pre to post change in HAM-D17 and BDI-II scores.

For each scale, runs a Shapiro-Wilk test on the paired differences, a paired
t-test and a paired Cohen's d (d_z) with a noncentral t 95% CI, then corrects
the p-values across scales with CORRECTION_METHOD (Bonferroni by default).
Prints the result tables, saves them and plots pre and post box plots with
individual trajectories.

Inputs
------
DATA_PATH
    questionnaire_converted.xlsx (from 01_convert_questionnaires.py).

Outputs
-------
TABLE_OUT
    questionnaire_stats.csv, one row per scale.
FIG_OUT
    questionnaire_pre_post.png.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import nct
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

DATA_PATH = Path(r"C:\Users\jojos\coding\aiTBS-eeg-fmri-biomarkers\questionnaire\questionnaire_converted.xlsx")
FIG_OUT   = Path(r"C:\Users\jojos\coding\aiTBS-eeg-fmri-biomarkers\results\figures\questionnaire_pre_post.png")
TABLE_OUT = Path(r"C:\Users\jojos\coding\aiTBS-eeg-fmri-biomarkers\results\tables\questionnaire_stats.csv")

QUESTIONNAIRES    = ["HAMD17", "BDI"]
DISPLAY_NAMES      = {"HAMD17": "HAM-D17", "BDI": "BDI-II"}
COLOR_PRE         = "#b9c6dd"
COLOR_POST        = "#f2b49a"
ALPHA             = 0.05
CORRECTION_METHOD = "bonferroni"
RANDOM_SEED       = 42

df_stats = pd.read_excel(DATA_PATH)

def paired_d_ci(t_obs: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Return the CI of a paired Cohen's d (d_z) via the noncentral t distribution.

    Parameters
    ----------
    t_obs : float
        Observed t, computed with the same subtraction direction as d
        (post - pre here, so negative means improvement).
    n : int
        Number of pairs.
    conf : float
        Confidence level.

    Returns
    -------
    tuple of float
        Lower and upper bound of the CI on d.
    """
    df = n - 1
    alpha = 1 - conf
 
    def cdf_diff(ncp, target):
        """Return the noncentral t CDF at t_obs minus the target probability."""
        return nct.cdf(t_obs, df, ncp) - target
 
    def find_root(target, direction, start_step=2.0, max_step=80.0):
        """Find the noncentrality parameter where the CDF equals target."""
        x, f_prev = t_obs, cdf_diff(t_obs, target)
        step = start_step
        while step <= max_step:
            x_next = t_obs + direction * step
            f_next = cdf_diff(x_next, target)
            if np.isnan(f_next):
                step /= 2
                continue
            if np.sign(f_next) != np.sign(f_prev) or f_next == 0:
                lo, hi = sorted([x, x_next])
                return brentq(lambda ncp: cdf_diff(ncp, target), lo, hi)
            x, f_prev = x_next, f_next
            step *= 1.6
        raise RuntimeError(f"CI root not bracketed for t_obs={t_obs}, target={target}")
 
    ncp_lo = find_root(1 - alpha / 2, direction=-1)
    ncp_hi = find_root(alpha / 2,     direction=+1)
    return ncp_lo / np.sqrt(n), ncp_hi / np.sqrt(n)
 
 
def compute_paired_stats(df: pd.DataFrame, questionnaire: str) -> dict:
    """Run the paired statistics for one questionnaire.

    Returns a dict with the pre, post and difference series, n, Shapiro-Wilk W
    and p, paired t and p, and Cohen's d with its 95% CI.
    """
    pair = df[[f"{questionnaire}_pre", f"{questionnaire}_post"]].dropna()
    pre, post = pair[f"{questionnaire}_pre"], pair[f"{questionnaire}_post"]
    diff = post - pre
 
    W, p_norm = st.shapiro(diff)
    t, p_t = st.ttest_rel(pre, post)
    d = diff.mean() / diff.std(ddof=1)
 
    n = len(pair)
    t_for_d = d * np.sqrt(n)
    d_ci_lo, d_ci_hi = paired_d_ci(t_for_d, n)
 
    return dict(questionnaire=questionnaire, pre=pre, post=post, diff=diff,
                n=n, W=W, p_norm=p_norm, t=t, p=p_t, d=d, d_ci_lo=d_ci_lo, d_ci_hi=d_ci_hi)
 
 
results = {q: compute_paired_stats(df_stats, q) for q in QUESTIONNAIRES}
 
p_raw = [results[q]["p"] for q in QUESTIONNAIRES]
if CORRECTION_METHOD:
    _, p_corr, _, _ = multipletests(p_raw, alpha=ALPHA, method=CORRECTION_METHOD)
else:
    p_corr = p_raw
for q, pc in zip(QUESTIONNAIRES, p_corr):
    results[q]["p_corr"] = pc
 
print("\nNormality Test (Shapiro-Wilk on pre-post difference)")
print(f"{'':22}{'W':>8}{'p':>10}")
for q in QUESTIONNAIRES:
    r = results[q]
    print(f"{q + '_diff':22}{r['W']:>8.3f}{r['p_norm']:>10.3f}")
print("Note. A low p-value suggests a violation of the assumption of normality.")
 
print("\nDescriptives")
print(f"{'':14}{'N':>5}{'Mean':>8}{'Median':>9}{'SD':>8}{'SE':>8}")
for q in QUESTIONNAIRES:
    for tp, x in [("pre", results[q]["pre"]), ("post", results[q]["post"])]:
        n = len(x)
        print(f"{q + '_' + tp:14}{n:>5}{x.mean():>8.1f}{x.median():>9.1f}"
              f"{x.std(ddof=1):>8.2f}{x.std(ddof=1) / np.sqrt(n):>8.2f}")
 
print("\nPaired Samples T-Test")
print(f"{'':16}{'t':>9}{'df':>5}{'p':>9}{'p_corr':>9}{'Cohen d':>10}{'95% CI':>18}")
for q in QUESTIONNAIRES:
    r = results[q]
    ci_str = f"[{r['d_ci_lo']:.2f}, {r['d_ci_hi']:.2f}]"
    print(f"{q:16}{r['t']:>9.3f}{r['n'] - 1:>5}{r['p']:>9.4f}{r['p_corr']:>9.4f}{r['d']:>10.3f}{ci_str:>18}")
print("Note. Ha: mean(pre) - mean(post) != 0. 95% CI computed via the noncentral t distribution.")
 
table = pd.DataFrame([{
    "questionnaire": DISPLAY_NAMES.get(q, q),
    "n": results[q]["n"],
    "shapiro_W": results[q]["W"], "shapiro_p": results[q]["p_norm"],
    "t": results[q]["t"], "df": results[q]["n"] - 1,
    "p": results[q]["p"], "p_corr": results[q]["p_corr"],
    "correction_method": CORRECTION_METHOD or "none",
    "cohen_d": results[q]["d"],
    "d_ci_lower": results[q]["d_ci_lo"], "d_ci_upper": results[q]["d_ci_hi"],
} for q in QUESTIONNAIRES])
TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(TABLE_OUT, index=False)
print(f"\nSaved stats table to {TABLE_OUT}")
 
 
def sig_stars(p: float) -> str:
    """Return a significance label ('***', '**', '*' or 'n.s.') for p."""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
 
 
fig, axes = plt.subplots(1, len(QUESTIONNAIRES), figsize=(3.2 * len(QUESTIONNAIRES), 4.6))
axes = np.atleast_1d(axes)
fig.suptitle("Pre vs Post \u2014 Clinical Outcomes", fontsize=15, fontweight="bold")
 
for ax, q in zip(axes, QUESTIONNAIRES):
    r = results[q]
    pre, post = r["pre"], r["post"]
 
    bp = ax.boxplot([pre, post], positions=[1, 2], widths=0.4,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="darkred", linewidth=2),
                     whiskerprops=dict(color="black", linewidth=1.4),
                     capprops=dict(color="black", linewidth=1.4), zorder=2)
    bp["boxes"][0].set_facecolor(COLOR_PRE)
    bp["boxes"][1].set_facecolor(COLOR_POST)
    for b in bp["boxes"]:
        b.set_edgecolor("black"); b.set_linewidth(1.6); b.set_alpha(0.9)
 
    for a, b in zip(pre, post):
        ax.plot([1, 2], [a, b], color="gray", alpha=0.25, lw=0.8, zorder=1)
    ax.scatter(np.ones(r["n"]),     pre,  color="gray", alpha=0.6, s=25, zorder=3)
    ax.scatter(np.ones(r["n"]) * 2, post, color="gray", alpha=0.6, s=25, zorder=3)
 
    y_top = max(pre.max(), post.max())
    y_bar = y_top * 1.12
    ax.plot([1, 1, 2, 2], [y_bar, y_bar * 1.02, y_bar * 1.02, y_bar], color="black", lw=1.2)
    ax.text(1.5, y_bar * 1.03, sig_stars(r["p_corr"]), ha="center", va="bottom",
            fontsize=14, fontweight="bold")
    ax.set_ylim(0, y_bar * 1.18)
 
    stat_txt = f"t({r['n']-1}) = {r['t']:.2f}, p = {r['p']:.4f}\nCohen's d = {r['d']:.2f}, N = {r['n']}"
    ax.text(0.5, -0.16, stat_txt, transform=ax.transAxes, ha="center", va="top", fontsize=9)
 
    ax.set_title(DISPLAY_NAMES.get(q, q), fontsize=12, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Pre-treatment", "Post-treatment"])
    ax.set_xlim(0.55, 2.45)
    ax.set_ylabel("Score")
    ax.spines[["top", "right"]].set_visible(False)
 
plt.tight_layout()
FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_OUT, dpi=300, bbox_inches="tight")
print(f"Saved figure to {FIG_OUT}")
plt.show()

