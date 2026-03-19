"""
statistics.py — Distributional statistics for Huggett (1996).

Contains:
    - Weighted Gini coefficient (allows negative wealth, Gini > 1)
    - Top wealth shares
    - Cross-sectional wealth moments
    - Age-by-age profiles (mean, std, quantiles, Gini, CV)
"""

import numpy as np


def weighted_gini(x, w):
    """
    Gini coefficient allowing negative values (can exceed 1).
    Uses the covariance formula: G = 2*cov(x, F(x)) / |mu|.
    """
    x, w = np.asarray(x).ravel(), np.asarray(w).ravel()
    if np.std(x) < 1e-14:
        return 0.0
    mu = np.sum(w * x) / np.sum(w)
    if abs(mu) < 1e-14:
        return np.nan
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    w = w / w.sum()
    cw = np.cumsum(w) - w / 2
    return max(2.0 * np.sum(w * x * cw) / abs(mu) - 1.0, 0.0)


def _top_share(x, w, p):
    """Fraction of total wealth held by the top p fraction of agents."""
    idx = np.argsort(x)
    x, w = x[idx], w[idx] / w[idx].sum()
    tot = np.sum(w * x)
    if abs(tot) < 1e-14:
        return np.nan
    return np.sum(w[np.cumsum(w) >= 1 - p] * x[np.cumsum(w) >= 1 - p]) / tot


def wealth_moments(k_hist, age_mass):
    """Cross-sectional Gini, top shares, and fraction with wealth <= 0."""
    N, A = k_hist.shape
    w = np.repeat(age_mass / age_mass.sum(), N) / N
    x = k_hist.T.ravel()
    return dict(
        frac_le0=float(np.sum(w * (x <= 0))),
        gini=float(weighted_gini(x, w)),
        top1=float(_top_share(x, w, 0.01)),
        top5=float(_top_share(x, w, 0.05)),
        top20=float(_top_share(x, w, 0.20)),
    )


def age_profiles(k_hist):
    """Mean, std, quantiles, Gini, and CV of wealth at each age."""
    N, A = k_hist.shape
    mean = k_hist.mean(axis=0)
    std  = k_hist.std(axis=0)
    qs = (0.10, 0.25, 0.50, 0.75, 0.90)
    quant = {q: np.quantile(k_hist, q, axis=0) for q in qs}
    gini = np.array([weighted_gini(k_hist[:, a], np.ones(N) / N) for a in range(A)])
    cv = np.zeros_like(mean)
    mask = np.abs(mean) > 1e-12
    cv[mask] = std[mask] / np.abs(mean[mask])
    return dict(mean=mean, std=std, quant=quant, gini=gini, cv=cv)

