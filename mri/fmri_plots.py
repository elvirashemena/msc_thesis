"""Figure helpers for the fMRI connectivity analysis.

Each function draws already computed data and statistics onto a given
matplotlib Axes; there is no file I/O, statistics or data loading here.
Statistics live in fmri_stats.py, and 05_exploratory.py loads the data and
saves the figures.

Author: Elvira Shemena
"""
import numpy as np

from fmri_stats import COLOR_PRE, COLOR_POST, COLOR_DOT, COLOR_LINE, COLOR_A, COLOR_B, sig_stars



def sig_bracket(ax, x1, x2, y, p, color="black", label_p=True, height=None):
    """Draw a significance bracket from x1 to x2 at height y, labelled with p.

    Black marks the general (whole-sample) test, grey the descriptive
    within-group tests.
    """
    h = height if height is not None else 0.03 * abs(y if y != 0 else 1)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, linewidth=1.1)
    txt = f"p = {p:.3f}{sig_stars(p)}" if label_p else sig_stars(p)
    ax.text((x1 + x2) / 2, y + h * 1.15, txt, ha="center", va="bottom",
            fontsize=8, color=color)



def prepost_bar_panel(ax, group_specs, title=""):
    """Draw grouped Baseline and Followup bars with paired subject lines.

    Each group gets mean and SD bars, jittered subject dots joined by lines and a
    significance bracket.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    group_specs : list of dict
        One dict per group with the keys label, baseline, followup (same subject
        order), p and bracket_color ('black' for the general test, 'grey' for
        within-group tests).
    title : str
        Panel title.
    """
    n_groups = len(group_specs)
    width = 0.32
    xs = np.arange(n_groups)

    all_vals = np.concatenate([np.r_[g["baseline"], g["followup"]] for g in group_specs
                                if len(g["baseline"]) or len(g["followup"])])
    y_top = np.nanmax(all_vals) if all_vals.size else 1.0
    y_bottom = np.nanmin(all_vals) if all_vals.size else 0.0
    span = (y_top - y_bottom) or 1.0

    for i, g in enumerate(group_specs):
        b, f = np.asarray(g["baseline"], float), np.asarray(g["followup"], float)
        for arr, xoff, color in [(b, -width / 2, COLOR_PRE), (f, width / 2, COLOR_POST)]:
            if arr.size == 0:
                continue
            m, sd = np.nanmean(arr), np.nanstd(arr, ddof=1) if arr.size > 1 else 0.0
            ax.bar(xs[i] + xoff, m, width=width, color=color, edgecolor="black", linewidth=0.6, zorder=2)
            ax.errorbar(xs[i] + xoff, m, yerr=sd, color="black", capsize=3, linewidth=1, zorder=3)
        jitter_b = np.random.normal(xs[i] - width / 2, 0.03, size=b.size)
        jitter_f = np.random.normal(xs[i] + width / 2, 0.03, size=f.size)
        ax.scatter(jitter_b, b, color=COLOR_DOT, s=14, alpha=0.6, zorder=4)
        ax.scatter(jitter_f, f, color=COLOR_DOT, s=14, alpha=0.6, zorder=4)
        for xb, xf, vb, vf in zip(jitter_b, jitter_f, b, f):
            ax.plot([xb, xf], [vb, vf], color=COLOR_LINE, linewidth=0.6, alpha=0.5, zorder=1)

        bracket_y = y_top + span * (0.08 + 0.10 * (i == 0))
        sig_bracket(ax, xs[i] - width / 2, xs[i] + width / 2, bracket_y, g["p"],
                    color=g.get("bracket_color", "grey"))

    ax.set_xticks(xs)
    ax.set_xticklabels([g["label"] for g in group_specs])
    ax.set_title(title, fontsize=9)
    ax.axhline(0, color="grey", linewidth=0.5, zorder=0)



def trajectory_panel(ax, level_a, level_b, baseline_col_vals_a, followup_vals_a,
                     baseline_vals_b, followup_vals_b, label_a, label_b, ylabel=""):
    """Draw subject and mean Baseline to Followup lines for two groups.

    Group a uses COLOR_A and group b COLOR_B; level_a and level_b are unused.
    """
    for b, f in zip(baseline_col_vals_a, followup_vals_a):
        ax.plot([0, 1], [b, f], color=COLOR_A, alpha=0.25, linewidth=1)
    for b, f in zip(baseline_vals_b, followup_vals_b):
        ax.plot([0, 1], [b, f], color=COLOR_B, alpha=0.25, linewidth=1)
    if len(baseline_col_vals_a):
        ax.plot([0, 1], [np.nanmean(baseline_col_vals_a), np.nanmean(followup_vals_a)],
                color=COLOR_A, linewidth=3, marker="o", label=f"{label_a} (n={len(baseline_col_vals_a)})")
    if len(baseline_vals_b):
        ax.plot([0, 1], [np.nanmean(baseline_vals_b), np.nanmean(followup_vals_b)],
                color=COLOR_B, linewidth=3, marker="o", label=f"{label_b} (n={len(baseline_vals_b)})")
    ax.set_xticks([0, 1]); ax.set_xticklabels([" Baseline", "Followup"])
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)



def box_with_points(ax, groups, colors, labels):
    """Draw box plots with jittered points, one box per array in groups."""
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                     medianprops=dict(color="black"), showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    for i, g in enumerate(groups):
        x = np.random.normal(i + 1, 0.05, size=len(g))
        ax.scatter(x, g, color=COLOR_DOT, s=18, alpha=0.6, zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)



def correlation_heatmap(ax, pivot_r, pivot_stars, corr_label=""):
    """Draw a correlation heatmap annotated with r and significance stars.

    pivot_r and pivot_stars have the same shape, with pairs as rows and scales as
    columns. Returns the image for a colorbar.
    """
    im = ax.imshow(pivot_r.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot_r.columns)))
    ax.set_xticklabels(pivot_r.columns, fontsize=8)
    ax.set_yticks(range(len(pivot_r.index)))
    ax.set_yticklabels(pivot_r.index, fontsize=9)
    for i in range(pivot_r.shape[0]):
        for j in range(pivot_r.shape[1]):
            r_val = pivot_r.values[i, j]
            if not np.isnan(r_val):
                s = pivot_stars.values[i, j]
                ax.text(j, i, f"{r_val:.2f}{s}", ha="center", va="center", fontsize=8)
    ax.set_title(corr_label, fontsize=9)
    return im


def scatter_with_fit(ax, x, y, color=COLOR_A, xlabel="", ylabel="", title=""):
    """Draw a scatter plot with a dashed linear fit when at least 3 points exist."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ax.scatter(x, y, s=55, color=color, edgecolor="black", linewidth=0.6, zorder=3)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 3:
        m, b = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(xs, m * xs + b, color="black", linewidth=1.2, linestyle="--", zorder=2)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
