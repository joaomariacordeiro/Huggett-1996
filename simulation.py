"""
simulation.py — Monte Carlo panel simulation for Huggett (1996).

Huggett (1996) uses distributional iteration (Huggett, 1993) to compute
the stationary wealth distribution. This replication uses Monte Carlo
simulation with 500,000 agents, which is simpler to implement but
introduces sampling noise.
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


@njit
def simulate_panel(agrid, zgrid, Pz, pi0, age_eff, aR, kp_pol,
                   N=500_000, seed=123):
    """
    Simulate N agents through their entire life cycle.

    Each agent starts at age 20 with zero assets and an earnings state
    drawn from pi0. At each age, savings are interpolated from the policy
    function, and next-period earnings are drawn via inverse CDF method.

    Returns k_hist (N x A) and z_hist (N x A).
    """
    np.random.seed(seed)
    A, nz, nk = len(age_eff), len(zgrid), len(agrid)

    # Draw initial earnings states from pi0
    cdf0 = np.cumsum(pi0)
    z = np.searchsorted(cdf0, np.random.rand(N))

    # Build cumulative transition matrix for inverse CDF sampling
    Pc = np.empty((nz, nz))
    for i in range(nz):
        s = 0.0
        for j in range(nz):
            s += Pz[i, j]; Pc[i, j] = s

    k = np.zeros(N)
    k_hist = np.zeros((N, A))
    z_hist = np.zeros((N, A), dtype=np.int64)

    # Forward simulation
    for age in range(A):
        k_hist[:, age] = k
        z_hist[:, age] = z
        k_cl = np.minimum(np.maximum(k, agrid[0]), agrid[-1])

        # Interpolate policy function
        kp = np.empty(N)
        for i in range(N):
            j = max(0, min(np.searchsorted(agrid, k_cl[i]) - 1, nk - 2))
            ww = (k_cl[i] - agrid[j]) / (agrid[j+1] - agrid[j] + 1e-16)
            kp[i] = (1 - ww) * kp_pol[age, z[i], j] + ww * kp_pol[age, z[i], j+1]

        if age < A - 1:
            k = kp.copy()
            uu = np.random.rand(N)
            z_next = np.empty(N, dtype=np.int64)
            for i in range(N):
                z_next[i] = np.searchsorted(Pc[z[i]], uu[i])
            z = z_next

    return k_hist, z_hist
