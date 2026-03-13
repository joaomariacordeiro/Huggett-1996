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


def _Phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def stationary_dist(P, tol=1e-14, maxit=200_000):
    """Long-run (ergodic) distribution of a Markov chain."""
    pi = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(maxit):
        pi2 = P.T @ pi
        if np.max(np.abs(pi2 - pi)) < tol:
            return pi2 / pi2.sum()
        pi = pi2
    return pi / pi.sum()


def earnings_markov(gamma, sigma_eps, sigma1, n_states=18, m=4.0, extreme=6.0):
    """
    Paper p.479-481: z_t = gamma*z_{t-1} + eps_t, eps~N(0,sigma_eps^2).
    17 equally-spaced states on [-4*sigma1, 4*sigma1] plus one extreme state at 6*sigma1.
    Transition probabilities via Tauchen's method.
    """
    if sigma_eps <= 0.0:
        return np.array([0.0]), np.array([[1.0]])

    z = np.append(np.linspace(-m * sigma1, m * sigma1, n_states - 1),
                  extreme * sigma1).astype(np.float64)
    nz = len(z)

    # Bin boundaries: 18 states need 19 boundaries
    b = np.empty(nz + 1)
    b[0], b[-1] = -np.inf, np.inf
    for j in range(1, nz):
        b[j] = 0.5 * (z[j-1] + z[j])

    # Transition matrix
    P = np.zeros((nz, nz))
    for i in range(nz):
        mu = gamma * z[i]
        for j in range(nz):
            P[i, j] = _Phi((b[j+1] - mu) / sigma_eps) - _Phi((b[j] - mu) / sigma_eps)
        P[i] /= P[i].sum() or 1.0

    return z, P


def initial_z_dist(zgrid, sigma1):
    """
    Paper p.479-481: age-1 agents have z ~ N(0, sigma1^2).
    Integrate over Markov chain bins — NOT the stationary distribution.
    """
    nz = len(zgrid)
    b = np.empty(nz + 1)
    b[0], b[-1] = -np.inf, np.inf
    for j in range(1, nz):
        b[j] = 0.5 * (zgrid[j-1] + zgrid[j])

    pi0 = np.array([_Phi(b[j+1] / sigma1) - _Phi(b[j] / sigma1)
                     for j in range(nz)]) if sigma1 > 0 else np.zeros(nz)
    s = pi0.sum()
    return pi0 / s if s > 0 else np.eye(1, nz, nz // 2).ravel()
