"""
earnings.py — Earnings process discretisation for Huggett (1996).

Contains:
    - Standard normal CDF
    - Stationary distribution of a Markov chain
    - Tauchen method for AR(1) discretisation (18 states)
    - Initial earnings distribution for age-1 agents
"""

import numpy as np
from math import erf, sqrt


#Standard normal CDF

def _Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

# Long-run fraction of agents in each earnings state
# Crucial to compute aggregate labor supply

def stationary_dist(P, tol=1e-14, maxit=200_000):
    pi = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(maxit):
        pi2 = P.T @ pi # distribution one period ahead
        if np.max(np.abs(pi2 - pi)) < tol:
            return pi2 / pi2.sum() # Normalise probabilities to sum up to 1
        pi = pi2
    return pi / pi.sum() 


# Paper p.479-481: z_t = gamma*z_{t-1} + eps_t, eps~N(0,sigma_eps^2).
# 17 equally-spaced states on [-4σ₁, 4σ₁] plus one extreme earnings shock at 6σ₁.
 

""" Tauchen's Method """

def earnings_markov(gamma, sigma_eps, sigma1, n_states=18, m=4.0, extreme=6.0):
    if sigma_eps <= 0.0:
        return np.array([0.0]), np.array([[1.0]])

    z = np.append(np.linspace(-m * sigma1, m * sigma1, n_states - 1),
                  extreme * sigma1).astype(np.float64)
    nz = len(z)

    # Defining the boundaries
    b = np.empty(nz + 1) # 18 states need 19 bin boundaries
    b[0], b[-1] = -np.inf, np.inf # limit of first and last boundary 
    for j in range(1, nz):
        b[j] = 0.5 * (z[j-1] + z[j]) # interior boundaries are mid-points
    # Given the additional extreme state, there's a non-uniform spacing between the 17th and 18th grid points    
    P = np.zeros((nz, nz))
    for i in range(nz):
        mu = gamma * z[i] # conditional mean
        for j in range(nz):
            # The probability that tomorrow's z lands in bin j.
            #Subtracting mu centers it, dividing by σ_ε standardizes it, then Φ (the standard normal CDF) gives the cumulative probability. 
            #Upper boundary minus lower boundary gives the area in that bin.
            P[i, j] = _Phi((b[j+1] - mu) / sigma_eps) - _Phi((b[j] - mu) / sigma_eps) 
        P[i] /= P[i].sum() or 1.0 # Normalise each row to sum to 1 

    return z, P

# Paper p.479-481: age-1 (20-year olds) agents have z ~ N(0, σ₁²).
# Defines where agents start.
# Earnings inequality increases monotonically with age (i.e. shocks accumulate with age, generating more inequality)

def initial_z_dist(zgrid, sigma1):
    nz = len(zgrid)
    b = np.empty(nz + 1) # 18 states need 19 bin boundaries
    b[0], b[-1] = -np.inf, np.inf # limit of first and last boundary 
    for j in range(1, nz):
        b[j] = 0.5 * (zgrid[j-1] + zgrid[j])  # interior boundaries are mid-points
    # For each of the 18 bins, compute the probability that a draw from N(0, σ²₁) falls in that bin. 
    # Dividing the boundaries by σ₁ standardizes them (converts N(0, σ²₁) to N(0,1) so we can use Φ). 
    # The if sigma1 > 0 guard handles the degenerate case — if σ₁ = 0, everyone starts at exactly z = 0, so all probabilities are zero (to be fixed by the fallback on the last line).
    pi0 = np.array([_Phi(b[j+1] / sigma1) - _Phi(b[j] / sigma1) 
                     for j in range(nz)]) if sigma1 > 0 else np.zeros(nz) 
    s = pi0.sum() # sum of probabilites (should be 1)
    return pi0 / s if s > 0 else np.eye(1, nz, nz // 2).ravel() 
