"""
calibration.py — Calibration data and parameters for Huggett (1996).
 
Contains:
    - Death probabilities from Jordan (1975), ages 20-98
    - Age-earnings profile from Huggett via Corbae (embedded).
    - build_params(): constructs the full parameter dictionary (Table 2, p.478)
"""
 
import numpy as np
 
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
 
# ── Age-earnings profile: exp(ybar_t) from Huggett via Corbae ──
# Source: Dean Corbae homework handout (data obtained from Mark Huggett).
# Used in Kirkby's VFI Toolkit replication. Values are exp(ybar_t), the deterministic component of labour endowment e(z,t) = exp(z_t + ybar_t).
# Ages 20-98. Hump-shaped, peak ~0.80 at age 47, zero from age 73 onward.

AGE_EARNINGS = np.array([
    0.0911, 0.1573, 0.2268, 0.2752, 0.3218, 0.3669, 0.4114, 0.4559, 0.4859, 0.5164,
    0.5474, 0.5786, 0.6097, 0.6311, 0.6517, 0.6711, 0.6893, 0.7060, 0.7213, 0.7355,
    0.7489, 0.7619, 0.7747, 0.7783, 0.7825, 0.7874, 0.7931, 0.7994, 0.7923, 0.7850,
    0.7771, 0.7679, 0.7567, 0.7351, 0.7105, 0.6822, 0.6500, 0.6138, 0.5675, 0.5183,
    0.4672, 0.3935, 0.3239, 0.2596, 0.1955, 0.1408, 0.0959, 0.0604, 0.0459, 0.0342,
    0.0246, 0.0165, 0.0091, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0])
 
 
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
 
