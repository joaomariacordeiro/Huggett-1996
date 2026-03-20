"""
simulation.py — Monte Carlo panel simulation for Huggett (1996).

Huggett (1996) uses distributional iteration (Huggett, 1993) to compute the stationary wealth distribution. 
This replication uses Monte Carlo simulation with 500,000 agents, which is simpler to implement but
introduces sampling noise.

Note:
@njit requires careful version matching between Numba and NumPy. 
Numba 0.60.0 requires NumPy >= 1.22 and < 2.1. 
Newer versions of NumPy (>= 2.0) are incompatible with Numba 0.60.0 and will produce import errors. 

To install these specific versions, please run:

pip install numba==0.60.0 numpy==1.26.4

Or with conda:

conda install numba=0.60.0 numpy=1.26.4

The code includes a fallback that allows it to run without Numba, but performance will be approximately 50–100x slower.
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

"""
Proceadure: 
    1) Simulate N agents through their entire life cycle.

    2) Each agent starts at age 20 with zero assets and an earnings state drawn from pi0. 
    At each age, savings are interpolated from the policy function, and next-period earnings are drawn via inverse CDF method.

    3) Returns k_hist (N x A) and z_hist (N x A).
"""

@njit
def simulate_panel(agrid, zgrid, Pz, pi0, age_eff, aR, kp_pol,
                   N=500_000, seed=123):
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
        k_cl = np.minimum(np.maximum(k, agrid[0]), agrid[-1]) # Ensures nobody is ouside the grid

        # Interpolate policy function
        kp = np.empty(N)
        for i in range(N):
            j = max(0, min(np.searchsorted(agrid, k_cl[i]) - 1, nk - 2))
            ww = (k_cl[i] - agrid[j]) / (agrid[j+1] - agrid[j] + 1e-16) # Interpolation weight
            kp[i] = (1 - ww) * kp_pol[age, z[i], j] + ww * kp_pol[age, z[i], j+1]  # Linear interpolation of the policy function

        if age < A - 1: # Do not update for the last period (age)!
            k = kp.copy()
            uu = np.random.rand(N)
            z_next = np.empty(N, dtype=np.int64)
            for i in range(N):
                z_next[i] = np.searchsorted(Pc[z[i]], uu[i])
            z = z_next

    return k_hist, z_hist
