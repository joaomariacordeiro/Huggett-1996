"""
household.py — Household problem for Huggett (1996).

Contains:
    - CRRA marginal utility and its inverse
    - TFP normalisation
    - Prices from production function
    - Social security benefit
    - Asset grid construction
    - Endogenous Grid Method (EGM) backward induction
"""

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(f=None, **kw):
        if f is not None:
            return f
        def wrapper(fn): return fn
        return wrapper


# ── Preferences ──

@njit
def _u1(c, sigma):
    """Marginal utility of CRRA: u'(c) = c^(-sigma)."""
    return c ** (-sigma)

@njit
def _inv_u1(mu, sigma):
    """Inverse marginal utility: c = mu^(-1/sigma)."""
    return mu ** (-1.0 / sigma)


# ── Technology ──

def _normalize_A(alpha, KY=3.0, w_target=1.0):
    """Choose TFP so that w = w_target when K/Y = KY and L = 1 (paper p.478)."""
    Y = w_target / (1.0 - alpha)
    K = KY * Y
    return Y / (K ** alpha)


def _prices(K, L, A, alpha, delta):
    """Cobb-Douglas output, gross return, and wage."""
    Y = A * K ** alpha * L ** (1 - alpha)
    r = alpha * A * K ** (alpha - 1) * L ** (1 - alpha) - delta
    w = (1 - alpha) * A * K ** alpha * L ** (-alpha)
    return Y, 1 + r, w


# ── Social Security ──

def _ss_benefit(theta, w, L, retire_mass):
    """Paper eq.6: balanced-budget SS benefit per retiree."""
    return theta * w * L / max(retire_mass, 1e-12)


# ── Asset Grid ──

def asset_grid(kmin, kmax, nk):
    """Uniform asset grid (paper p.492: uniform with spacing 0.40)."""
    return np.linspace(kmin, kmax, nk)


# ── EGM Backward Induction ──

@njit
def egm_backward(agrid, zgrid, Pz, age_eff, s_surv, aR,
                  beta, sigma, R, w, T, b, theta, nu,
                  kprime_min_global, c_floor=1e-10):
    """
    Solve the household problem backward from the terminal age using EGM.

    Budget (paper eq.1): c + a' = R*a + y
      working: y = (1 - theta - nu) * w * e(z,t) + T
      retired: y = b + T
    """
    A, nz, nk = len(age_eff), len(zgrid), len(agrid)
    kp = np.zeros((A, nz, nk))
    c  = np.zeros((A, nz, nk))

    # Terminal age: consume everything
    for iz in range(nz):
        y = b + T
        for ik in range(nk):
            c[A-1, iz, ik] = max(R * agrid[ik] + y, c_floor)

    # Backward loop
    for age in range(A - 2, -1, -1):
        working = age < aR
        # Paper p.475: terminal-age assets must be non-negative
        kprime_min = 0.0 if age == A - 2 else kprime_min_global

        # Expected marginal utility tomorrow
        mu_next = np.empty((nz, nk))
        for iz in range(nz):
            for ik in range(nk):
                mu_next[iz, ik] = _u1(max(c[age+1, iz, ik], c_floor), sigma)

        Emu = np.zeros((nz, nk))
        for iz in range(nz):
            for jz in range(nz):
                p = Pz[iz, jz]
                for ik in range(nk):
                    Emu[iz, ik] += p * mu_next[jz, ik]

        for iz in range(nz):
            y = ((1.0 - theta - nu) * w * age_eff[age] * np.exp(zgrid[iz]) + T
                 if working else b + T)

            # EGM: for each k' on grid, find implied k today
            x  = np.empty(nk)
            cc = np.empty(nk)
            for ikp in range(nk):
                kpr = agrid[ikp]
                if kpr < kprime_min:
                    x[ikp], cc[ikp] = 1e18, c_floor
                    continue
                c_now = _inv_u1(beta * s_surv[age] * R * Emu[iz, ikp], sigma)
                c_now = max(c_now, c_floor)
                x[ikp]  = (c_now + kpr - y) / R
                cc[ikp] = c_now

            order = np.argsort(x)
            xs, cs, kps = x[order], cc[order], agrid[order]

            # Interpolate back onto the asset grid
            for ik in range(nk):
                k0 = agrid[ik]
                if k0 <= xs[0]:
                    kpr = kprime_min
                    c[age, iz, ik] = max(R * k0 + y - kpr, c_floor)
                    kp[age, iz, ik] = kpr
                else:
                    j = min(np.searchsorted(xs, k0), nk - 1)
                    wgt = (k0 - xs[j-1]) / (xs[j] - xs[j-1] + 1e-16)
                    kpr = (1 - wgt) * kps[j-1] + wgt * kps[j]
                    kpr = max(kpr, kprime_min)
                    c_now = R * k0 + y - kpr
                    if c_now < c_floor:
                        c_now = c_floor
                        kpr = max(R * k0 + y - c_now, kprime_min)
                    c[age, iz, ik] = c_now
                    kp[age, iz, ik] = kpr

    return c, kp
