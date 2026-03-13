"""
calibration.py — Calibration data and parameters for Huggett (1996).

Contains:
    - Death probabilities from Jordan (1975), ages 20-98
    - Age-earnings profile from paper Figure 1 (loaded from CSV)
    - build_params(): constructs the full parameter dictionary (Table 2, p.478)
"""

import numpy as np
import pandas as pd
import os

# ── Death probabilities: Jordan (1975), ages 20-98 ──
DEATH_PROBS = np.array([
    .00159,.00169,.00174,.00172,.00165,.00156,.00149,.00145,.00145,.00149,
    .00156,.00163,.00171,.00181,.00193,.00207,.00225,.00246,.00270,.00299,
    .00332,.00368,.00409,.00454,.00504,.00558,.00617,.00686,.00766,.00865,
    .00955,.01058,.01162,.01264,.01368,.01475,.01593,.01730,.01891,.02074,
    .02271,.02476,.02690,.02912,.03143,.03389,.03652,.03930,.04225,.04538,
    .04871,.05230,.05623,.06060,.06542,.07066,.07636,.08271,.08986,.09788,
    .10732,.11799,.12895,.13920,.14861,.16039,.17303,.18665,.20194,.21877,
    .23601,.25289,.26973,.28612,.30128,.31416,.32915,.34450,.36018])

# ── Age-earnings profile: loaded from CSV ──
# Paper Figure 1: SSA median male earnings x LFP rates (Handbook of Labor Statistics, 1985)
# Asymmetric hump, peak ~1.4 at age 48
_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "huggett1996_fig1_age_earnings.csv")
if os.path.exists(_csv_path):
    AGE_EARNINGS = pd.read_csv(_csv_path)["earnings_ratio_to_overall_mean"].to_numpy(dtype=np.float64)
else:
    # Fallback: embedded values if CSV not found
    AGE_EARNINGS = np.array([
        .20,.24,.29,.35,.42,.50,.58,.66,.74,.81,
        .87,.93,.98,1.03,1.07,1.11,1.15,1.18,1.21,1.24,
        1.27,1.29,1.31,1.33,1.35,1.37,1.38,1.39,1.40,1.39,
        1.38,1.36,1.34,1.30,1.26,1.20,1.14,1.06,.97,.87,
        .76,.64,.52,.40,.28,
        .16,.10,.06,.04,.03,.02,.015,.010,.007,.005,
        .003,.002,.0015,.001,.0007,
        .0005,.0004,.0003,.0002,.00015,
        .0001,8e-5,6e-5,4e-5,3e-5,
        2e-5,1e-5,1e-5,1e-5,5e-6,3e-6,2e-6,1e-6,1e-6])


def build_params(**overrides):
    """
    Build the default parameter dictionary (Paper Table 2, p.478).
    Any parameter can be overridden via keyword arguments.
    """
    g_y, delta = 0.195, 0.06
    p = dict(
        alpha=0.36, delta=delta, beta=1.011, sigma=1.5,
        theta=0.10, nu=g_y / (1 - delta * 3.0),
        gamma=0.96, sigma_eps=np.sqrt(0.045), sigma1=np.sqrt(0.38),
        age1=20, age_last=98, age_ret=65, pop_g=0.012,
        death_probs=DEATH_PROBS.copy(),
        age_eff=AGE_EARNINGS.copy(),
        nk=301, kmax=120.0, c_floor=1e-10,
        Nsim=500_000, seed=123,
        K0=40.0, T0=0.2, dampK=0.25, dampT=0.35,
        tolK=2e-3, tolT=2e-4, maxit=250,
        lifetimes="uncertain", credit_limit="-w",
    )
    p.update(overrides)
    return p
