"""Shared helpers for 04_correlation.py and fmri_stats.py.

Provides BH-FDR correction, significance stars, a Mann-Whitney effect size
and ROI pair name formatting.

Author: Elvira Shemena
"""
import numpy as np
from scipy.stats import mannwhitneyu


def bh_fdr(pvals):
    """Return Benjamini-Hochberg adjusted p-values in input order.

    NaNs are left out of the correction and returned as NaN.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    result = np.empty(n)
    result[order] = adj
    out[valid] = result
    return out


def p_to_stars(p):
    """Return '***', '**', '*' or '' for p; '' if p is missing."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def mannwhitney_effect(x, y):
    """Run a two-sided Mann-Whitney U test with a rank-biserial effect size.

    NaNs are dropped first, and r_rb = 1 - 2U / (n1 * n2).

    Returns
    -------
    tuple
        U, p, r_rb, n1 and n2; U, p and r_rb are NaN if a sample is empty.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, n1, n2
    U, p = mannwhitneyu(x, y, alternative="two-sided")
    r_rb = 1 - (2 * U) / (n1 * n2)
    return U, p, r_rb, n1, n2



_ROI_PREFIXES = ("delta_", "baseline_")

_ROI_SIMPLIFY = (
    ("DLPFC", "DLPFC"),
    ("AMPFC", "AMPFC"),
    ("PCC", "PCC"),
    ("L_AG", "leftAG"),
    ("R_AG", "rightAG"),
)


def strip_prefix(col):
    """Remove a leading 'delta_' or 'baseline_' from a column name."""
    for p in _ROI_PREFIXES:
        if col.startswith(p):
            return col[len(p):]
    return col


def simplify_roi(token):
    """Return a short ROI label such as 'leftAG', or the token unchanged."""
    t = token.upper()
    for key, label in _ROI_SIMPLIFY:
        if key in t:
            return label
    return token


def pretty_pair(col, sep="::"):
    """Return a short pair label such as 'AMPFC-PCC'.

    Accepts raw pair columns and columns with a delta_ or baseline_ prefix.
    """
    raw = strip_prefix(col)
    parts = raw.split(sep)
    if len(parts) == 2:
        return f"{simplify_roi(parts[0])}-{simplify_roi(parts[1])}"
    return raw
