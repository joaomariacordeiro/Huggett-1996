"""
equilibrium.py — General equilibrium solver for Huggett (1996).

Contains:
    - solve_ge: iterates on (K, T) until market clearing
    - print_diagnostics: compact summary of equilibrium
    - replicate_table: runs all 8 specs for a given sigma
"""

import numpy as np
import pandas as pd

from earnings import earnings_markov, initial_z_dist
from household import _normalise_A, _prices, _ss_benefit, asset_grid, egm_backward
from simulation import simulate_panel
from statistics import wealth_moments, age_profiles

# Find stationary general equilibrium by iterating on (K, T)

def solve_ge(params, verbose=True, track_convergence=False):
    p = params
    A_len = p["age_last"] - p["age1"] + 1
    aR    = p["age_ret"]  - p["age1"] + 1
    age_eff = p["age_eff"][:A_len].copy()

    # Survival
    if p["lifetimes"] == "certain":
        d = np.zeros(A_len); d[-1] = 1.0; beta_use = 0.994
    else:
        d = p["death_probs"][:A_len].copy(); d[-1] = max(d[-1], 1.0)
        beta_use = p["beta"]
    # Huggett notation: s_h[t] = Pr(survive to age t+1 | alive at age t) in 0-based Python indexing
    # so s_h[1] is survival to model age 2, conditional on reaching age 1.
    s_h = np.ones(A_len)
    s_h[1:] = 1.0 - d[:-1]

    # Stationary age mass (paper p.474, footnote 4)
    age_mass = np.ones(A_len)
    for i in range(1, A_len):
        age_mass[i] = s_h[i] * age_mass[i-1] / (1 + p["pop_g"])
    age_mass /= age_mass.sum()
    retire_mass = age_mass[aR:].sum()

    # Earnings chain + initial distribution
    zgrid, Pz = earnings_markov(p["gamma"], p["sigma_eps"], p["sigma1"])
    pi0 = initial_z_dist(zgrid, p["sigma1"]) if p["sigma_eps"] > 0 else np.array([1.0])

    # Normalise L = 1 using the lifecycle earnings distribution
    z_mass_by_age = np.zeros((A_len, len(zgrid)))
    z_mass_by_age[0] = pi0
    for age in range(1, A_len):
        z_mass_by_age[age] = z_mass_by_age[age - 1] @ Pz
    Ez_by_age = z_mass_by_age @ np.exp(zgrid)
    L_raw = float(np.sum(age_mass[:aR] * age_eff[:aR] * Ez_by_age[:aR]))
    age_eff /= L_raw

    TFP = _normalise_A(p["alpha"])
    K, T = p["K0"], p["T0"]
    hist = {k: [] for k in ('K', 'T', 'errK', 'errT')} if track_convergence else None

    for it in range(1, p["maxit"] + 1):
        # Step 1: Prices from current K guess
        Y, R_pre, w = _prices(K, 1.0, TFP, p["alpha"], p["delta"])
        R = 1 + (R_pre - 1) * (1 - p["nu"])
        b = _ss_benefit(p["theta"], w, 1.0, retire_mass) if p["theta"] > 0 else 0.0

        # Step 2: Borrowing limit and asset grid
        kprime_min = 0.0 if p["credit_limit"] == "0.0" else -w
        agrid = asset_grid(min(kprime_min, 0), p["kmax"], p["nk"])

        # Step 3: Solve household problem (EGM backward)
        _, kp_pol = egm_backward(agrid, zgrid, Pz, age_eff, s_h, aR,
                                  beta_use, p["sigma"], R, w, T, b,
                                  p["theta"], p["nu"], kprime_min, p["c_floor"])

        # Step 4: Simulate panel (Monte Carlo forward)
        k_hist, z_hist = simulate_panel(agrid, zgrid, Pz, pi0, age_eff, aR,
                                         kp_pol, p["Nsim"], p["seed"])

        # Step 5: Reconstruct k' for aggregation
        kp_hist = np.zeros_like(k_hist)
        for age in range(A_len):
            k_cl = np.clip(k_hist[:, age], agrid[0], agrid[-1])
            j = np.clip(np.searchsorted(agrid, k_cl) - 1, 0, len(agrid) - 2)
            wgt = (k_cl - agrid[j]) / (agrid[j+1] - agrid[j] + 1e-16)
            kp_hist[:, age] = (1 - wgt) * kp_pol[age, z_hist[:, age], j] + \
                               wgt * kp_pol[age, z_hist[:, age], j + 1]

        # Step 6: Capital market clearing and accidental bequests (paper eq.7, p.477)
        K_model = max(float(np.mean(kp_hist, 0) @ age_mass) / (1 + p["pop_g"]), 1e-12)
        death_next = np.ones(A_len)
        death_next[:-1] = 1.0 - s_h[1:]
        beq = float(np.mean(R * kp_hist, 0) @ (age_mass * death_next))
        T_model = beq / (1 + p["pop_g"])

        # Step 6: Check convergence
        errK, errT = K_model - K, T_model - T
        if hist:
            for k, v in zip(('K','T','errK','errT'), (K, T, errK, errT)):
                hist[k].append(v)

        if verbose and it % 10 == 0:
            print(f"  it {it:3d}: K={K:.4f} T={T:.4f} | errK={errK:+.2e} errT={errT:+.2e}")

        if abs(errK) < p["tolK"] and abs(errT) < p["tolT"]:
            if verbose: print(f"  Converged at iteration {it}")
            K, T = K_model, T_model
            break

        # Step 7: Update with dampening
        K = (1 - p["dampK"]) * K + p["dampK"] * K_model
        T = (1 - p["dampT"]) * T + p["dampT"] * T_model
    else:
        if verbose: print(f"  Warning: max iterations ({p['maxit']}) reached")

    Y, R_pre, w = _prices(K_model, 1.0, TFP, p["alpha"], p["delta"])
    R = 1 + (R_pre - 1) * (1 - p["nu"])
    b = _ss_benefit(p["theta"], w, 1.0, retire_mass) if p["theta"] > 0 else 0.0
    moms = wealth_moments(k_hist, age_mass)
    prof = age_profiles(k_hist)

    return dict(
        K_over_Y=K_model / Y, K=K_model, Y=Y, w=w, R=R, r_pre=R_pre - 1,
        T=T_model, b=b, z_states=len(zgrid),
        frac_le0=moms["frac_le0"], gini=moms["gini"],
        top1=moms["top1"], top5=moms["top5"], top20=moms["top20"],
        age_mass=age_mass, profiles=prof,
        k_hist=k_hist, z_hist=z_hist, convergence=hist,
    )


def print_diagnostics(out, params):
    """Print a compact summary of equilibrium quantities."""
    p, o = params, out
    L, nu, theta = 1.0, p["nu"], p["theta"]
    r, w, K, Y = o["r_pre"], o["w"], o["K"], o["Y"]
    G = nu * (r * K + w * L)
    SS_rev = theta * w * L
    aR = p["age_ret"] - p["age1"] + 1
    SS_ben = o["b"] * o["age_mass"][aR:].sum()

    print(f"\n{'─'*60}")
    print(f"  K/Y = {o['K_over_Y']:.3f}   w = {w:.4f}   r = {r*100:.2f}%")
    print(f"  G/Y = {G/Y:.3f}   T/Y = {o['T']/Y:.4f}   SS bal = {SS_rev - SS_ben:+.4f}")
    print(f"  Gini = {o['gini']:.3f}   Top1% = {o['top1']*100:.1f}%   "
          f"Top5% = {o['top5']*100:.1f}%   Top20% = {o['top20']*100:.1f}%")
    print(f"  Frac <= 0 = {o['frac_le0']*100:.1f}%")
    print(f"{'─'*60}")


# Run all 8 specifications for a given sigma (1.5 for Table 3, 3.0 for Table 4).
def replicate_table(base_params, sigma_val, verbose=True):
    specs = [(lt, s2, cl)
             for lt in ("certain", "uncertain")
             for s2 in (0.0, 0.045)
             for cl in ("0.0", "-w")]

    rows = []
    for lt, s2, cl in specs:
        label = f"{'Cert' if lt == 'certain' else 'Uncert'}, " \
                f"{'Shocks' if s2 > 0 else 'No shocks'}, " \
                f"{'a>=0' if cl == '0.0' else 'a>=-w'}"
        if verbose: print(f"\n  -- {label} --")
        p = dict(base_params, sigma=sigma_val, lifetimes=lt,
                 sigma_eps=float(np.sqrt(s2)) if s2 > 0 else 0.0,
                 credit_limit=cl)
        o = solve_ge(p, verbose=verbose)
        rows.append(dict(Spec=label, **{k: o[k] for k in
                    ('K_over_Y','gini','top1','top5','top20','frac_le0')}))
        if verbose: print_diagnostics(o, p)

    df = pd.DataFrame(rows)
    df.columns = ['Spec', 'K/Y', 'Gini', 'Top1%', 'Top5%', 'Top20%', 'Frac<=0']
    for c in ('Top1%', 'Top5%', 'Top20%', 'Frac<=0'):
        df[c] = (df[c] * 100).round(1)
    df['Gini'] = df['Gini'].round(3)
    df['K/Y']  = df['K/Y'].round(2)
    print(f"\n{'='*90}\n  TABLE {'3' if sigma_val == 1.5 else '4'}: sigma = {sigma_val}\n{'='*90}")
    print(df.to_string(index=False))
    return df



