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

''' 

@njit tells Numba to compile that function to machine code the first time it's called, so every subsequent call runs at C-like speed instead of Python (slower).
It matters here because _u1 and _inv_u1 are called millions of times — once for every combination of agent × age × earnings state × asset grid point inside the EGM loops. 
Without @njit, each call goes through Python's interpreter, which is slow. 
With it, the function becomes a raw machine instruction that the CPU executes directly.

'''

# ── Preferences ──

@njit
def _u1(c, sigma):
    return c ** (-sigma) # Marginal utility of a CRRA utility function (Euler Equation)


@njit
def _inv_u1(mu, sigma):
    return mu ** (-1.0 / sigma) # Converting marginal utility into the implied consumption level (inverted Euler Equation)


# ── Technology ──

# Paper p.478: "The technology level A is normalized so that the wage equals 1.0 when the capital output ratio equals 3.0 and the labor input per capita is normalized at 1.0."
# Choose TFP (A) so that w = w_target = 1 when K/Y = 3 and L = 1
def _normalize_A(alpha, KY=3.0, w_target=1.0):
    Y = w_target / (1.0 - alpha)
    K = KY * Y 
    return Y / (K ** alpha)


def _prices(K, L, A, alpha, delta):
    Y = A * K ** alpha * L ** (1 - alpha)
    r = alpha * A * K ** (alpha - 1) * L ** (1 - alpha) - delta
    w = (1 - alpha) * A * K ** alpha * L ** (-alpha)
    return Y, 1 + r, w


# ── Social Security ──

# Eq.6: balanced-budget SS benefit per retiree.
def _ss_benefit(theta, w, L, retire_mass):
    return theta * w * L / max(retire_mass, 1e-12)


# ── Asset Grid ──

# Paper p.492: 
# "The number of grid points varies between as little as 41 for economies without earnings uncertainty to as many as 301 for the economies with earnings uncertainty."
# Note: paper used different grids depending on the type of economy. This choice, in my view might have been due to computational power, back in the 90s
# Here, I will stick with a uniform asset grid with 301 and spacing 0.40 

def asset_grid(kmin, kmax, nk):
    return np.linspace(kmin, kmax, nk)


# ── EGM Backward Induction ──

@njit
def egm_backward(agrid, zgrid, Pz, age_eff, s_h, aR,
                  beta, sigma, R, w, T, b, theta, nu,
                  kprime_min_global, c_floor=1e-10):
# Budget constraint (paper eq.1): c + a' = R·a + y
# Worker: y = (1 - θ - ν)·w·e(z,t) + T
# Retiree: y = b + T
    A, nz, nk = len(age_eff), len(zgrid), len(agrid)
    kp = np.zeros((A, nz, nk)) # how much an agent of this age, earnings state, and wealth level saves for next period
    c  = np.zeros((A, nz, nk)) # how much an agent of this age, earnings state, and wealth level consumes

    # Terminal age: consume everything
    # At the last age, the agent dies for sure (no reason to save)
    # Income is just the social security benefit plus transfers (everyone this age is retired). 
    # Cash on hand is R times their assets (return on savings) plus income y. 
    for iz in range(nz):
        y = b + T
        for ik in range(nk):
            c[A-1, iz, ik] = max(R * agrid[ik] + y, c_floor)
    # Backward loop        
    for age in range(A - 2, -1, -1): # start, stop, step. Starts at A-2 since terminal period was handled above.
        working = age < aR
        kprime_min = 0.0 if age == A - 2 else kprime_min_global 
        # Paper p.475: The only additional restriction is that if an agent survives to the terminal age N, then asset holdings must be nonnegative, a'>= 0. 
        # Of course, the credit limit can be set sufficiently low so that the only binding requirement is that in the last period of life the agent holds no debt.
        # Agents at 97 can save but cannot borrow since they die next period (assets cannot be negative)
        
        # Expected marginal utility tomorrow
        # For each (earnings state, asset level) tomorrow, compute the marginal utility of consumption.
        mu_next = np.empty((nz, nk))
        for iz in range(nz):
            for ik in range(nk):
                mu_next[iz, ik] = _u1(max(c[age+1, iz, ik], c_floor), sigma)
        # The agent doesn't know which earnings state they'll have tomorrow. 
        # So they compute the expected marginal utility by weighting over all possible tomorrow states. 
        # If today you're in state iz, the probability of being in state jz tomorrow is Pz[iz, jz]. 
        # This triple loop computes: for each current state iz and each asset level ik, the expectation Σⱼ P(iz,jz) · u'(c_{tomorrow}(jz, ik)).
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
            # Loop over every possible choice of k' on the grid. 
            # Standard VFI would try each k and evaluate the value function (expensive optimization). 
            x  = np.empty(nk)
            cc = np.empty(nk)
            for ikp in range(nk):
                kpr = agrid[ikp]
                if kpr < kprime_min:
                    x[ikp], cc[ikp] = 1e18, c_floor
                    continue
                c_now = _inv_u1(beta * s_h[age + 1] * R * Emu[iz, ikp], sigma) # Invert Euler equation using Huggett's s_{t+1}.
                c_now = max(c_now, c_floor) # Boundaries for consumption
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


