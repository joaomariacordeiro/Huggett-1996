
"""
run_and_plot.py — Run Huggett (1996) replication and plot results.

This is the main script. It imports from the module files:
    calibration.py  — parameters and data
    earnings.py     — earnings process discretisation
    household.py    — preferences, technology, EGM solver
    simulation.py   — Monte Carlo panel simulation
    statistics.py   — Gini, top shares, age profiles
    equilibrium.py  — GE solver, diagnostics, table replication

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from calibration import build_params
from equilibrium import solve_ge, print_diagnostics, replicate_table
from statistics import weighted_gini
from datetime import datetime

# Chronomtre
start_time = datetime.now()


# Plot settings
# Follow a style that matches nicely the LaTeX fonts etc.
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'lines.linewidth': 1.8,
    'lines.markersize': 6, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})

save_dir = Path("figures")
save_dir.mkdir(exist_ok=True)


def weighted_quantile(x, w, q):
    x = np.asarray(x).ravel()
    w = np.asarray(w).ravel()
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    w = w / w.sum()
    cw = np.cumsum(w)
    return float(np.interp(q, np.r_[0.0, cw], np.r_[x[0], x]))


def pooled_age_slice(k_hist, age_mass, lo, hi):
    n_agents = k_hist.shape[0]
    x = k_hist[:, lo:hi].T.ravel()
    w = np.repeat(age_mass[lo:hi], n_agents) / n_agents
    w = w / w.sum()
    return x, w


# ══════════════════════════════════════════════════════════════
# 1. Replicate Tables 3 and 4
# ══════════════════════════════════════════════════════════════
# Note: this step take quite some time to run given the multiple simulations for each model economy.
# To run it, please uncomment out the two lines below
base = build_params()
replicate_table(base, 1.5)   # Table 3
replicate_table(base, 3.0)   # Table 4

# ══════════════════════════════════════════════════════════════
# 2. Run the 4 uncertain-lifetimes specs (for Figures 3, 5)
# ══════════════════════════════════════════════════════════════

specs = [
    (0.0,   "0.0",  "No shocks, a>=0"),
    (0.0,   "-w",   "No shocks, a>=-w"),
    (0.045, "0.0",  "Shocks, a>=0"),
    (0.045, "-w",   "Shocks, a>=-w"),
]

results = {}
for s2, cl, label in specs:
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    p = dict(base, lifetimes="uncertain",
             sigma_eps=float(np.sqrt(s2)) if s2 > 0 else 0.0,
             credit_limit=cl)
    o = solve_ge(p)
    print_diagnostics(o, p)
    results[label] = o


# ══════════════════════════════════════════════════════════════
# 3. Run 2 certain-lifetimes shocks specs (for Figure 4)
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}\n  Certain, Shocks, a>=0\n{'='*60}")
p = dict(base, lifetimes="certain", sigma_eps=np.sqrt(0.045), credit_limit="0.0")
cert_a0 = solve_ge(p)
print_diagnostics(cert_a0, p)

print(f"\n{'='*60}\n  Certain, Shocks, a>=-w\n{'='*60}")
p = dict(base, lifetimes="certain", sigma_eps=np.sqrt(0.045), credit_limit="-w")
cert_aw = solve_ge(p)
print_diagnostics(cert_aw, p)


# Convenience
age1    = base["age1"]
age_ret = base["age_ret"]
ages    = np.arange(age1, age1 + len(base["age_eff"]))
A_len   = len(ages)


# ══════════════════════════════════════════════════════════════
# 4. Age-Earnings Profile
# ══════════════════════════════════════════════════════════════

age_eff = base["age_eff"]
idx_ret = age_ret - age1

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ages[:idx_ret+1], age_eff[:idx_ret+1],color='#1a1a2e', lw=2.2, label='Working ages')
ax.plot(ages[idx_ret-1:], age_eff[idx_ret-1:],color='#1a1a2e', lw=1.2, ls='--', alpha=0.4, label='Post-retirement')
ax.axvline(age_ret, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Ratio to overall mean')
ax.set_xlim(age1, 80)
ax.set_ylim(bottom=0)
ax.legend()
plt.tight_layout()
plt.savefig(save_dir / 'fig_earnings.pdf')
plt.show()


# ══════════════════════════════════════════════════════════════
# 5. Mean Wealth by Age (4 specs, every age)
# ══════════════════════════════════════════════════════════════

styles = {
    "No shocks, a>=0":  dict(color='#1f77b4', ls='-',  marker='o', ms=4, markevery=5),
    "No shocks, a>=-w": dict(color='#ff7f0e', ls='--', marker='s', ms=4, markevery=5),
    "Shocks, a>=0":     dict(color='#2ca02c', ls='-.', marker='^', ms=4, markevery=5),
    "Shocks, a>=-w":    dict(color='#d62728', ls=':',  marker='D', ms=4, markevery=5),
}

fig, ax = plt.subplots(figsize=(9, 6))
for label, out in results.items():
    prof_ages = np.arange(age1, age1 + len(out['profiles']['mean']))
    ax.plot(prof_ages, out['profiles']['mean'], label=label, **styles[label])
ax.axvline(age_ret, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.axhline(0, color='black', lw=0.5, alpha=0.3)
ax.set_xlabel('Age')
ax.set_ylabel('Mean wealth')
ax.set_xlim(age1, 98)
ax.set_title('$\\sigma=1.5$, uncertain')
ax.legend(loc='upper left', framealpha=0.9)
plt.tight_layout()
plt.savefig(save_dir / 'fig_mean_wealth.pdf')
plt.show()


# ══════════════════════════════════════════════════════════════
# 6. Wealth Profiles (quantiles, 10-year age bins)
#    Paper shows: Mean, 50% quantile, 25% quantile, 10% quantile
#    For the baseline spec (uncertain, shocks, a'>=-w)
# ══════════════════════════════════════════════════════════════

k_hist = results['Shocks, a>=-w']['k_hist']

bin_centers_f3 = [20, 30, 40, 50, 60, 70, 80]
means_f3, q50_f3, q25_f3, q10_f3 = [], [], [], []
ages_f3 = []

for center in bin_centers_f3:
    lo = max(center - 5 - age1, 0)       # 10-year bins
    hi = min(center + 5 - age1, A_len)
    if lo >= hi:
        continue
    pooled, pooled_w = pooled_age_slice(k_hist, results['Shocks, a>=-w']['age_mass'], lo, hi)
    ages_f3.append(center)
    means_f3.append(float(np.sum(pooled * pooled_w)))
    q50_f3.append(weighted_quantile(pooled, pooled_w, 0.50))
    q25_f3.append(weighted_quantile(pooled, pooled_w, 0.25))
    q10_f3.append(weighted_quantile(pooled, pooled_w, 0.10))

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(ages_f3, means_f3, '-^', color='#1a1a2e', ms=4, mfc='none', mew=1.5, lw=2, markevery=1, label='Mean')
ax.plot(ages_f3, q50_f3,   '-+', color='#1f77b4', ms=4, mew=1.5, lw=1.8, markevery=1, label='50% Quantile')
ax.plot(ages_f3, q25_f3,   '-*', color='#2ca02c', ms=4, mew=1.2, lw=1.8, markevery=1, label='25% Quantile')
ax.plot(ages_f3, q10_f3,   '-s', color='#d62728', ms=4, mfc='none', mew=1.5, lw=1.8, markevery=1, label='10% Quantile')
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Wealth')
ax.set_xlim(15, 90)
ax.set_xticks([20, 30, 40, 50, 60, 70, 80, 90])
ax.legend(loc='upper left', framealpha=0.9)
plt.tight_layout()
plt.savefig(save_dir / 'fig_wealth_profiles.pdf')
plt.show()



# ══════════════════════════════════════════════════════════════
# 7. Gini within age groups — Certain Lifetimes
#    Paper: 5-year bins from age 30 to 75, shocks only
# ══════════════════════════════════════════════════════════════

bin_centers_f4 = list(range(30, 80, 5))   # 30, 35, ..., 75
gini_cert_a0, gini_cert_aw = [], []
ages_f4 = []

for center in bin_centers_f4:
    lo = max(center - 2 - age1, 0)        # 5-year bins
    hi = min(center + 3 - age1, A_len)
    if lo >= hi:
        continue
    pool_a0, w_a0 = pooled_age_slice(cert_a0['k_hist'], cert_a0['age_mass'], lo, hi)
    pool_aw, w_aw = pooled_age_slice(cert_aw['k_hist'], cert_aw['age_mass'], lo, hi)
    ages_f4.append(center)
    gini_cert_a0.append(weighted_gini(pool_a0, w_a0))
    gini_cert_aw.append(weighted_gini(pool_aw, w_aw))

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(ages_f4, gini_cert_a0, color='#2ca02c', ls='-.', marker='^', ms=4, markevery=1, lw=1.8,
        label='Shocks, $\\bar{a} = 0$')
ax.plot(ages_f4, gini_cert_aw, color='#d62728', ls=':', marker='D', ms=4, markevery=1, lw=1.8,
        label='Shocks, $\\bar{a} = -w$')
ax.axhline(1.0, color='gray', lw=0.7, ls='--', alpha=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Wealth Gini')
ax.set_xlim(25, 78)
ax.set_ylim(0, 1.4)
ax.set_xticks(range(25, 80, 5))
ax.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.savefig(save_dir / 'fig_gini_certain.pdf')
plt.show()


# ══════════════════════════════════════════════════════════════
# 8. Gini within age groups — Uncertain Lifetimes
#    Paper: 5-year bins from age 30 to 90, shocks only
# ══════════════════════════════════════════════════════════════

bin_centers_f5 = list(range(30, 90, 5))   # 30, 35, ..., 85 (stop before 90 to avoid sparse bins)
gini_uncert_a0, gini_uncert_aw = [], []
ages_f5 = []

k_hist_ua0 = results['Shocks, a>=0']['k_hist']
k_hist_uaw = results['Shocks, a>=-w']['k_hist']

for center in bin_centers_f5:
    lo = max(center - 2 - age1, 0)        # 5-year bins
    hi = min(center + 3 - age1, A_len)
    if lo >= hi:
        continue
    pool_a0, w_a0 = pooled_age_slice(k_hist_ua0, results['Shocks, a>=0']['age_mass'], lo, hi)
    pool_aw, w_aw = pooled_age_slice(k_hist_uaw, results['Shocks, a>=-w']['age_mass'], lo, hi)
    ages_f5.append(center)
    gini_uncert_a0.append(weighted_gini(pool_a0, w_a0))
    gini_uncert_aw.append(weighted_gini(pool_aw, w_aw))

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(ages_f5, gini_uncert_a0, color='#2ca02c', ls='-.', marker='^', ms=4, markevery=1, lw=1.8, label='Shocks, $\\bar{a} = 0$')
ax.plot(ages_f5, gini_uncert_aw, color='#d62728', ls=':', marker='D', ms=4, markevery=1, lw=1.8,         label='Shocks, $\\bar{a} = -w$')
ax.axhline(1.0, color='gray', lw=0.7, ls='--', alpha=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Wealth Gini')
ax.set_xlim(25, 88)
ax.set_ylim(0, 1.4)
ax.set_xticks(range(25, 90, 5))
ax.legend(loc='upper left', framealpha=0.9)
plt.tight_layout()
plt.savefig(save_dir / 'fig_gini_uncertain.pdf')
plt.show()


# ══════════════════════════════════════════════════════════════
# 9. Lorenz Curves
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, (label, out) in enumerate(results.items()):
    N, A = out['k_hist'].shape
    age_mass = out['age_mass']
    w = np.repeat(age_mass, N) / N
    x = out['k_hist'].T.ravel()
    idx = np.argsort(x)
    ws, xs = w[idx], x[idx]
    cw = np.cumsum(ws) / ws.sum()
    cx = np.cumsum(xs * ws)
    if abs(cx[-1]) > 1e-12:
        cx /= cx[-1]
    ax.plot(cw, cx, color=colors[i], lw=1.8, label=label)

ax.plot([0, 1], [0, 1], 'k--', lw=0.7, alpha=0.4)
ax.set_xlabel('Population share')
ax.set_ylabel('Wealth share')
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 1)
ax.set_title('Lorenz Curves')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(save_dir / 'fig_lorenz.pdf')
plt.show()

# ══════════════════════════════════════════════════════════════
# 10. Wealth Percentiles (for baseline)
# ══════════════════════════════════════════════════════════════


bl = results['Shocks, a>=-w']
pr = bl['profiles']
fan_ages = np.arange(age1, age1 + len(pr['mean']))

fig, ax = plt.subplots(figsize=(9, 6))
ax.fill_between(fan_ages, pr['quant'][0.10], pr['quant'][0.90],alpha=0.15, color='#1f77b4', label='p10-p90')
ax.fill_between(fan_ages, pr['quant'][0.25], pr['quant'][0.75],alpha=0.25, color='#1f77b4', label='p25-p75')
ax.plot(fan_ages, pr['quant'][0.50], color='#1f77b4', lw=2, label='Median')
ax.plot(fan_ages, pr['mean'], color='#d62728', lw=2, ls='--', label='Mean')
ax.axvline(age_ret, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.axhline(0, color='black', lw=0.5, alpha=0.3)
ax.set_xlabel('Age')
ax.set_ylabel('Wealth')
ax.set_xlim(age1, 98)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(save_dir / 'fig_quantiles.pdf')
plt.show()


# ══════════════════════════════════════════════════════════════
# 11. Print summary
# ══════════════════════════════════════════════════════════════

print("\n" + "="*75)
print("  Results Summary: sigma=1.5, uncertain lifetimes")
print("="*75)
print(f"{'Specification':<25} {'K/Y':>6} {'Gini':>6} {'Top1%':>7} {'Top5%':>7} {'Top20%':>7} {'Frac<=0':>8}")
print("-"*75)
for label, out in results.items():
    print(f"{label:<25} {out['K_over_Y']:>6.2f} {out['gini']:>6.3f} "
          f"{out['top1']*100:>6.1f}% {out['top5']*100:>6.1f}% "
          f"{out['top20']*100:>6.1f}% {out['frac_le0']*100:>7.1f}%")
print("-"*75)

bl = results['Shocks, a>=-w']
print("\nBaseline vs Paper:")
print(f"  K/Y:     {bl['K_over_Y']:.3f}  (paper: 3.2)")
print(f"  Gini:    {bl['gini']:.3f}  (paper: 0.76)")
print(f"  Top 1%:  {bl['top1']*100:.1f}%  (paper: 11.8%)")
print(f"  Top 5%:  {bl['top5']*100:.1f}%  (paper: 35.6%)")
print(f"  Top 20%: {bl['top20']*100:.1f}%  (paper: 75.5%)")
print(f"  Frac<=0: {bl['frac_le0']*100:.1f}%  (paper: 24.0%)")

print(f"\nAll figures saved to: {save_dir.resolve()}/")

# Stop chronometer & print running time
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))