"""Plot the clinical response trajectories for HAM-D17 and BDI-II.

Draws one panel per scale with pre and post box plots and per-participant
lines coloured by responder status (at least 50% reduction). Saves one
version with a statistics box and one clean version.

Inputs
------
DATA_PATH
    questionnaire_converted.xlsx (from 01_convert_questionnaires.py).

Outputs
-------
clinical_response_stats.svg, clinical_response_clean.svg
    Saved in the folder of FIG_OUT.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

DATA_PATH = Path(r"C:\Users\jojos\coding\aiTBS-eeg-fmri-biomarkers\questionnaire\questionnaire_converted.xlsx")
FIG_OUT   = Path(r"C:\Users\jojos\coding\aiTBS-eeg-fmri-biomarkers\results\figures\clinical_response_lines.png")

QUESTIONNAIRES = ["HAMD17", "BDI"]
DISPLAY_NAMES  = {"HAMD17": "HAM-D17", "BDI": "BDI-II"}
Y_LABELS       = {"HAMD17": "HAM-D17 Score", "BDI": "BDI-II Score"}
COLOR_PRE      = "#b9c6dd"
COLOR_POST     = "#f2b49a"
COLOR_RESP     = "#c0392b"
COLOR_NONRESP  = "#4f6d9a"
RESPONDER_CUT  = -50.0


def plot_clinical_response(results, df, questionnaires=QUESTIONNAIRES,
                           display=DISPLAY_NAMES, out_path=FIG_OUT,
                           show_stats=True):
    """Draw the pre to post response figure, one panel per questionnaire.

    Parameters
    ----------
    results : dict
        Per scale: pre and post series, n, t, p and d.
    df : pandas.DataFrame
        Converted questionnaire table, used for the responder status.
    questionnaires : list of str
        Scales to plot, one panel each.
    display : dict
        Panel title per scale.
    out_path : pathlib.Path
        Output file; the format follows the suffix.
    show_stats : bool
        Add a box with median [IQR], mean change, t-test and Cohen's d.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = len(questionnaires)

    fig, axes = plt.subplots(1, n_panels, figsize=(6.7, 4.5))
    axes = np.atleast_1d(axes)
    fig.suptitle("Clinical Response Trajectories Following aiTBS",
                 fontsize=11, fontweight="bold", y=0.98)

    def sig(p):
        """Return a significance label with its p threshold."""
        return "*** (p < .001)" if p < .001 else "** (p < .01)" if p < .01 \
            else "* (p < .05)" if p < .05 else "n.s."

    xbox_pre,  xbox_post = 0.0, 1.0
    xpt_pre,   xpt_post  = 0.0, 1.0

    for ax, q in zip(axes, questionnaires):
        r    = results[q]
        pre  = np.asarray(r["pre"])
        post = np.asarray(r["post"])
        idx  = r["pre"].index

        pct_col = f"{q}_percentage"
        if pct_col in df.columns:
            resp = (df.loc[idx, pct_col] <= RESPONDER_CUT).to_numpy()
        else:
            resp = ((post - pre) / pre * 100 <= RESPONDER_CUT)

        bp = ax.boxplot(
            [pre, post],
            positions=[xbox_pre, xbox_post],
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            zorder=2,
            medianprops =dict(color="darkred", linewidth=2),
            whiskerprops=dict(color="#555",     linewidth=1.1),
            capprops    =dict(color="#555",     linewidth=1.1),
            boxprops    =dict(edgecolor="#555", linewidth=1.1),
        )
        bp["boxes"][0].set_facecolor(COLOR_PRE);  bp["boxes"][0].set_alpha(0.55)
        bp["boxes"][1].set_facecolor(COLOR_POST); bp["boxes"][1].set_alpha(0.55)

        for a, b, is_r in zip(pre, post, resp):
            c = COLOR_RESP if is_r else COLOR_NONRESP
            ax.plot([xpt_pre, xpt_post], [a, b],
                    color=c, lw=0.9, alpha=0.45, zorder=3)

        for a, b, is_r in zip(pre, post, resp):
            c = COLOR_RESP if is_r else COLOR_NONRESP
            ax.scatter([xpt_pre, xpt_post], [a, b],
                       color=c, s=22, edgecolor="white",
                       linewidth=0.4, zorder=4, alpha=0.9)

        top = max(pre.max(), post.max())
        yb  = top * 1.06
        ax.plot([xbox_pre, xbox_pre, xbox_post, xbox_post],
                [yb, yb * 1.025, yb * 1.025, yb], color="black", lw=1.1)
        ax.text((xbox_pre + xbox_post) / 2, yb * 1.035, sig(r["p"]),
                ha="center", va="bottom", fontsize=9, fontweight="bold")

        if show_stats:
            red = (1 - post.mean() / pre.mean()) * 100
            q1a, q3a = np.percentile(pre,  [25, 75])
            q1b, q3b = np.percentile(post, [25, 75])
            box_txt = (f"Median: {np.median(pre):.0f} [{q1a:.0f}-{q3a:.0f}] "
                       f"\u2192 {np.median(post):.0f} [{q1b:.0f}-{q3b:.0f}]\n"
                       f"Mean Change: {-red:.0f}%\n"
                       f"t({r['n']-1}) = {r['t']:.2f}, p = {r['p']:.4f}\n"
                       f"Cohen's d = {r['d']:.2f} (N = {r['n']})")
            ax.text(0.03, 0.03, box_txt, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#bbb", alpha=0.9))

        ax.set_title(display.get(q, q), fontsize=10, fontweight="bold", pad=4)
        ax.set_xticks([xbox_pre, xbox_post],
                      ["Pre-treatment", "Post-treatment"], fontsize=9)
        ax.set_xlim(-0.55, xbox_post + 0.55)
        ax.set_ylim(0, top * 1.20)
        ax.set_ylabel(Y_LABELS.get(q, "Score"), fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)

    handles = [
        plt.Line2D([0], [0], marker="o", color=COLOR_RESP,
                   markerfacecolor=COLOR_RESP, markersize=6, lw=1.2,
                   label=f"Responder (\u226550% reduction)"),
        plt.Line2D([0], [0], marker="o", color=COLOR_NONRESP,
                   markerfacecolor=COLOR_NONRESP, markersize=6, lw=1.2,
                   label="Non-responder (<50% reduction)")
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout(rect=[0, 0.06, 1, 0.97], w_pad=3.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    return fig


if __name__ == "__main__":
    df = pd.read_excel(DATA_PATH)
    results = {}
    for q in QUESTIONNAIRES:
        pair  = df[[f"{q}_pre", f"{q}_post"]].dropna()
        pre   = pair[f"{q}_pre"]
        post  = pair[f"{q}_post"]
        diff  = post - pre
        t, p  = st.ttest_rel(pre, post)
        results[q] = dict(pre=pre, post=post, n=len(pair), t=t, p=p,
                          d=diff.mean() / diff.std(ddof=1))

    plot_clinical_response(results, df, show_stats=True,
                           out_path=FIG_OUT.with_name("clinical_response_stats.svg"))
    plot_clinical_response(results, df, show_stats=False,
                           out_path=FIG_OUT.with_name("clinical_response_clean.svg"))
    plt.show()
    print("\n" + "="*52)
    print(f"  {'Descriptive Statistics':^48}")
    print("="*52)
    for q in QUESTIONNAIRES:
        r    = results[q]
        pre  = np.asarray(r["pre"])
        post = np.asarray(r["post"])
        name = DISPLAY_NAMES.get(q, q)
        print(f"\n  {name}")
        print(f"  {'-'*30}")
        print(f"  Pre  — Mean: {pre.mean():.2f}  SD: {pre.std(ddof=1):.2f}")
        print(f"  Post — Mean: {post.mean():.2f}  SD: {post.std(ddof=1):.2f}")
    print("\n" + "="*52 + "\n")
