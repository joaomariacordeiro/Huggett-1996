"""
run_and_plot.py — Run Huggett (1996) replication and plot results.

This is the main script. It imports from the module files:
    calibration.py  — parameters and data
    earnings.py     — earnings process discretisation
    household.py    — preferences, technology, EGM solver
    simulation.py   — Monte Carlo panel simulation
    statistics.py   — Gini, top shares, age profiles
    equilibrium.py  — GE solver, diagnostics, table replication

Usage:
    python run_and_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from calibration import build_params
from equilibrium import solve_ge, print_diagnostics, replicate_table

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'lines.linewidth': 1.8,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

save_dir = Path("figures")
save_dir.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. Replicate Tables 3 and 4
# ══════════════════════════════════════════════════════════════

base = build_params()
replicate_table(base, 1.5)   # Table 3
replicate_table(base, 3.0)   # Table 4


# ══════════════════════════════════════════════════════════════
# 2. Run the 4 uncertain-lifetimes specifications (for figures)
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

# Convenience variables
age1    = base["age1"]
age_ret = base["age_ret"]
ages    = np.arange(age1, age1 + len(base["age_eff"]))

# Line styles for the 4 specs
styles = {
    "No shocks, a>=0":  dict(color='#1f77b4', ls='-',  marker='o', ms=4, markevery=5),
    "No shocks, a>=-w": dict(color='#ff7f0e', ls='--', marker='s', ms=4, markevery=5),
    "Shocks, a>=0":     dict(color='#2ca02c', ls='-.', marker='^', ms=4, markevery=5),
    "Shocks, a>=-w":    dict(color='#d62728', ls=':',  marker='D', ms=4, markevery=5),
}


# ══════════════════════════════════════════════════════════════
# 3. Figure 1: Age-Earnings Profile
# ══════════════════════════════════════════════════════════════

age_eff = base["age_eff"]
idx_ret = age_ret - age1

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ages[:idx_ret+1], age_eff[:idx_ret+1],
        color='#1a1a2e', lw=2.2, label='Working ages')
ax.plot(ages[idx_ret-1:], age_eff[idx_ret-1:],
        color='#1a1a2e', lw=1.2, ls='--', alpha=0.4, label='Post-retirement')
ax.axvline(age_ret, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Ratio to overall mean')
ax.set_xlim(age1, 80)
ax.set_ylim(bottom=0)
ax.set_title('Figure 1: Age-Earnings Profile')
ax.legend()
plt.tight_layout()
plt.savefig(save_dir / 'fig1_earnings.png')
plt.show()


# ══════════════════════════════════════════════════════════════
# 4. Figure 2: Survival Probabilities
# ══════════════════════════════════════════════════════════════

death_probs = base["death_probs"]
s_surv = 1 - death_probs
uncond = np.cumprod(np.insert(s_surv[:-1], 0, 1.0))
surv_ages = np.arange(age1, age1 + len(death_probs))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(surv_ages, s_surv, color='#1a1a2e', lw=2)
ax1.set_xlabel('Age')
ax1.set_ylabel('$s_t$')
ax1.set_title('Conditional survival')
ax1.set_ylim(0.6, 1.01)

ax2.plot(surv_ages, uncond, color='#d62728', lw=2)
ax2.set_xlabel('Age')
ax2.set_ylabel('$S_t$')
ax2.set_title('Unconditional survival')
ax2.set_ylim(0, 1.05)

fig.suptitle('Figure 2: Survival Probabilities', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(save_dir / 'fig2_survival.png')
plt.show()


# ══════════════════════════════════════════════════════════════
# 5. Figure 3: Mean Wealth by Age
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 6))
for label, out in results.items():
    prof_ages = np.arange(age1, age1 + len(out['profiles']['mean']))
    ax.plot(prof_ages, out['profiles']['mean'], label=label, **styles[label])
ax.axvline(age_ret, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.axhline(0, color='black', lw=0.5, alpha=0.3)
ax.set_xlabel('Age')
ax.set_ylabel('Mean wealth')
ax.set_xlim(age1, 98)
ax.set_title('Figure 3: Mean Wealth by Age ($\\sigma=1.5$, uncertain)')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(save_dir / 'fig3_mean_wealth.png')
plt.show()


# ══════════════════════════════════════════════════════════════
# 6. Figure 4: Gini of Wealth by Age
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 6))
for label, out in results.items():
    prof_ages = np.arange(age1, age1 + len(out['profiles']['gini']))
    ax.plot(prof_ages, out['profiles']['gini'], label=label, **styles[label])
ax.axvline(age_ret, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.axhline(1.0, color='black', ls='--', lw=0.7, alpha=0.3, label='Gini = 1')
ax.set_xlabel('Age')
ax.set_ylabel('Gini coefficient')
ax.set_xlim(age1, 98)
ax.set_title('Figure 4: Gini of Wealth by Age ($\\sigma=1.5$, uncertain)')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(save_dir / 'fig4_gini.png')
plt.show()


# ══════════════════════════════════════════════════════════════
# 7. Figure 5: CV of Wealth by Age
# ══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 6))
for label, out in results.items():
    prof_ages = np.arange(age1, age1 + len(out['profiles']['cv']))
    ax.plot(prof_ages, out['profiles']['cv'], label=label, **styles[label])
ax.axvline(age_ret, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Coefficient of variation')
ax.set_xlim(age1, 98)

all_cvs = np.concatenate([o['profiles']['cv'][:70] for o in results.values()])
ax.set_ylim(0, min(float(all_cvs.max()) * 1.15, 20))
ax.set_title('Figure 5: CV of Wealth by Age ($\\sigma=1.5$, uncertain)')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(save_dir / 'fig5_cv.png')
plt.show()


# ══════════════════════════════════════════════════════════════
# 8. Lorenz Curves
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
plt.savefig(save_dir / 'fig_lorenz.png')
plt.show()


# ══════════════════════════════════════════════════════════════
# 9. Wealth Percentiles Fan Chart (baseline only)
# ══════════════════════════════════════════════════════════════

bl = results['Shocks, a>=-w']
pr = bl['profiles']
fan_ages = np.arange(age1, age1 + len(pr['mean']))

fig, ax = plt.subplots(figsize=(9, 6))
ax.fill_between(fan_ages, pr['quant'][0.10], pr['quant'][0.90],
                alpha=0.15, color='#1f77b4', label='p10-p90')
ax.fill_between(fan_ages, pr['quant'][0.25], pr['quant'][0.75],
                alpha=0.25, color='#1f77b4', label='p25-p75')
ax.plot(fan_ages, pr['quant'][0.50], color='#1f77b4', lw=2, label='Median')
ax.plot(fan_ages, pr['mean'], color='#d62728', lw=2, ls='--', label='Mean')
ax.axvline(age_ret, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.axhline(0, color='black', lw=0.5, alpha=0.3)
ax.set_xlabel('Age')
ax.set_ylabel('Wealth')
ax.set_xlim(age1, 98)
ax.set_title('Wealth Percentiles by Age (Shocks, a>=-w)')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(save_dir / 'fig_quantiles.png')
plt.show()


# ══════════════════════════════════════════════════════════════
# 10. Print summary
# ══════════════════════════════════════════════════════════════

print("\n" + "="*75)
print("  RESULTS SUMMARY: sigma=1.5, uncertain lifetimes")
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
