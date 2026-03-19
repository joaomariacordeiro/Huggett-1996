# Replication Package: Huggett (1996) — Wealth Distribution in Life-Cycle Economies

## Overview

This repository contains the replication code for:

> Huggett, Mark. 1996. "Wealth Distribution in Life-Cycle Economies." *Journal of Monetary Economics*, 38(3): 469–494.

The replication was produced as part of the course *Quantitative Macroeconomics and Numerical Methods* at Goethe University Frankfurt (Winter Term 2025/26), taught by Johannes Goensch.

**Author:** João Maria Cordeiro  
**Date:** March 2026  
**Contact:** joaomariacordeiro@gmail.com

(any errors are my own)

---

## Data Availability Statement

The model is calibrated using data from the following sources, as specified in the original paper (Table 2, p. 478):

| Data | Source | Availability |
|------|--------|-------------|
| Survival probabilities | Jordan (1975) | Embedded in `calibration.py` |
| Age-earnings profile | Huggett via Corbae; see Kirkby (2022) | Embedded in `calibration.py` |
| Earnings process parameters | Abowd and Card (1989), *Econometrica* 57(2) | Published |
| Population growth rate | US Census | Published |

No external data files need to be downloaded. All calibration data are embedded directly in the source code.

---

## Computational Requirements

### Software

| Software | Tested version | Required |
|----------|---------------|----------|
| Python | 3.11.9 | >= 3.10 |
| NumPy | 1.26.4 | >= 1.24, < 2.0 (see note) |
| Numba | 0.60.0 | >= 0.58, <= 0.60 (see note) |
| Matplotlib | 3.7+ | Yes |
| Pandas | 2.0+ | Yes |

**Critical note on Numba and NumPy compatibility.** The `@njit` decorator, which compiles the EGM solver and panel simulation to machine code, requires careful version matching between Numba and NumPy. Numba 0.60.0 requires NumPy >= 1.22 and < 2.1. Newer versions of NumPy (>= 2.0) are incompatible with Numba 0.60.0 and will produce import errors. Additionally, there are reported miscompilation bugs with Numba 0.61.x and NumPy 2.2.x on Windows (see [numba/numba#10126](https://github.com/numba/numba/issues/10126)). The tested and recommended combination is:

```
numba==0.60.0
numpy==1.26.4
```

To install these specific versions:
```bash
pip install numba==0.60.0 numpy==1.26.4
```

Or with conda:
```bash
conda install numba=0.60.0 numpy=1.26.4
```

The code includes a fallback that allows it to run without Numba, but performance will be approximately 50–100x slower.

### Hardware

The code was developed and tested on the following system:

| Component | Specification |
|-----------|--------------|
| Operating system | Windows 11 |
| Processor | Intel(R) Core(TM) 7 240H (2.50 GHz) |
| RAM | 32.0 GB |
| IDE | Spyder 6 (Anaconda) |

The code is platform-independent and should run on any system with the required software. No GPU is required.

### Expected Runtime (with Numba)

Total runtime for the full replication (Tables 3–4, all figures, summary): **approximately 2 hours 30 minutes**.

| Task | Approximate time |
|------|-----------------|
| First run (Numba JIT compilation) | ~1 minute (one-off) |
| Table 3 (8 specifications, σ = 1.5) | ~50 minutes |
| Table 4 (8 specifications, σ = 3.0) | ~50 minutes |
| Figures (4 uncertain + 2 certain specs) | ~40 minutes |
| **Full replication (`run_and_plot.py`)** | **~2 h 30 min** |

Without Numba, the full replication may take 10+ hours.

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install numba==0.60.0 numpy==1.26.4 matplotlib pandas
```

### 2. Verify Numba installation

```bash
python -c "from numba import njit; print('Numba OK')"
```

If this prints `Numba OK`, the installation is correct. If it fails, check that your NumPy version is compatible (see above).

### 3. Place all files in a single directory

```
replication/
├── calibration.py
├── earnings.py
├── household.py
├── simulation.py
├── statistics.py
├── equilibrium.py
├── run_and_plot.py
└── README.md
```

---

## Code Structure

| File | Description | Key functions |
|------|-------------|---------------|
| `calibration.py` | Parameters and embedded data (Table 2) | `build_params()` |
| `earnings.py` | Tauchen discretisation of AR(1) earnings (p. 481) | `earnings_markov()`, `initial_z_dist()` |
| `household.py` | Preferences, technology, EGM solver | `egm_backward()`, `asset_grid()` |
| `simulation.py` | Monte Carlo panel simulation (500K agents) | `simulate_panel()` |
| `statistics.py` | Distributional statistics (Gini, top shares) | `wealth_moments()`, `age_profiles()` |
| `equilibrium.py` | GE solver, diagnostics, table replication | `solve_ge()`, `replicate_table()` |
| **`run_and_plot.py`** | **Main script: produces all output** | — |

### Dependency graph

```
calibration.py ──┐
earnings.py ─────┤
household.py ────┼── equilibrium.py ── run_and_plot.py
simulation.py ───┤
statistics.py ───┘
```

The four core modules (`earnings.py`, `household.py`, `simulation.py`, `statistics.py`) are standalone with no cross-dependencies. `equilibrium.py` imports from all four. `run_and_plot.py` imports from `calibration.py`, `equilibrium.py`, and `statistics.py`.

---

## Reproducing Results

### Full replication

```bash
cd replication
python run_and_plot.py
```

This produces, in order:
1. **Table 3** (σ = 1.5, 8 specifications) — printed to console
2. **Table 4** (σ = 3.0, 8 specifications) — printed to console
3. **6 GE solves** (4 uncertain + 2 certain lifetimes) for figures
4. **All figures** — saved to `figures/` as PDF
5. **Summary table** — printed to console
6. **Total runtime** — printed to console

### Tables only

To replicate only the tables without generating figures, comment out sections 2–10 in `run_and_plot.py` and run:

```python
from calibration import build_params
from equilibrium import replicate_table

base = build_params()
replicate_table(base, 1.5)   # Table 3
replicate_table(base, 3.0)   # Table 4
```

### Single specification

```python
from calibration import build_params
from equilibrium import solve_ge, print_diagnostics

base = build_params()
out = solve_ge(base)
print_diagnostics(out, base)
```

---

## Output

### Key results

Baseline specification (σ = 1.5, uncertain lifetimes, shocks, a' ≥ −w):

| Moment | Replication | Paper | US Data |
|--------|------------|-------|---------|
| K/Y | 3.323 | 3.2 | 3.0 |
| Gini | 0.746 | 0.76 | 0.72 |
| Top 1% | 11.8% | 11.8% | 28.0% |
| Top 5% | 35.4% | 35.6% | 53.5% |
| Top 20% | 74.6% | 75.5% | 79.5% |
| Frac ≤ 0 | 28.9% | 24.0% | 15–20% |

### Figures

All figures are saved to `figures/` in PDF format at 300 DPI.

| File | Description | Paper reference |
|------|-------------|----------------|
| `fig_earnings.pdf` | Age-earnings profile (Corbae/Huggett data) | Figure 1 |
| `fig_mean_wealth.pdf` | Mean wealth by age, 4 uncertain specs | — |
| `fig_wealth_profiles.pdf` | Quantiles (Mean, p50, p25, p10) in 10-year bins | Figure 3 |
| `fig_gini_certain.pdf` | Gini within 5-year age groups, certain lifetimes | Figure 4 |
| `fig_gini_uncertain.pdf` | Gini within 5-year age groups, uncertain lifetimes | Figure 5 |
| `fig_lorenz.pdf` | Lorenz curves, 4 uncertain specs | — |
| `fig_quantiles.pdf` | Wealth percentile fan chart (baseline) | — |

### Tables

Tables are printed to the console. To save as CSV:
```python
df3 = replicate_table(base, 1.5)
df4 = replicate_table(base, 3.0)
df3.to_csv("table3_results.csv", index=False)
df4.to_csv("table4_results.csv", index=False)
```

---

## Methodological Notes

### Departures from the original paper

| Component | This replication | Huggett (1996) |
|-----------|-----------------|----------------|
| Household problem | EGM (Carroll, 2006) | Value function iteration |
| Wealth distribution | Monte Carlo (500K agents) | Distributional iteration |
| Asset grid | Uniform, 301 pts | Uniform; 41–301 pts depending on economy |
| Earnings discretisation | 18-state Tauchen variant | 18-state discretisation |
| Age-earnings profile | Huggett via Corbae/Kirkby (2022) | SSA Bulletin / Handbook of LS |
| GE updating | Damped fixed-point on K and T | Not specified |

### Known discrepancies

1. **Certain-lifetimes K/Y.** The certain-lifetimes specifications produce K/Y ≈ 4.3–5.1, above the paper's 2.0–3.2. This arises because the Corbae earnings profile drops to exactly zero at age 73, creating 25 years of zero income for agents who survive to 98 with certainty. Under uncertain lifetimes, most agents die before this region, so the effect is muted. The discount factor β = 0.994 was likely calibrated with the original (smoother) earnings profile and may require adjustment.

2. **Fraction ≤ 0.** The zero-wealth fraction (28.9% vs 24.0% for the baseline) reflects remaining differences in the age-earnings profile and Monte Carlo sampling noise.

3. **Monte Carlo noise.** Top wealth shares exhibit minor variability (±0.3 pp) across random seeds. The seed is fixed at 123 for reproducibility.

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `ImportError: Numba needs NumPy...` | Install compatible versions: `pip install numba==0.60.0 numpy==1.26.4` |
| `WARNING: Numba not found` | Install Numba; code runs without it but ~100x slower |
| `NameError: name 'cert_a0' is not defined` | Run the full script top-to-bottom; in Spyder, use `%runfile run_and_plot.py` |
| `RuntimeWarning: invalid value encountered in divide` | Harmless; occurs when computing CV at ages where mean wealth is zero |
| Very slow execution | Verify Numba is loaded: check for `Numba loaded — compiled mode` at startup |

---

## References

- Abowd, J.M. and D. Card (1989). "On the Covariance Structure of Earnings and Hours Changes." *Econometrica*, 57(2): 411–445.
- Carroll, C.D. (2006). "The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems." *Economics Letters*, 91(3): 312–320.
- Huggett, M. (1993). "The Risk-Free Rate in Heterogeneous-Agent Incomplete-Insurance Economies." *JEDC*, 17(5–6): 953–969.
- Huggett, M. (1996). "Wealth Distribution in Life-Cycle Economies." *Journal of Monetary Economics*, 38(3): 469–494.
- Hurd, M.D. (1989). "Mortality Risk and Bequests." *Econometrica*, 57(4): 779–813.
- Jordan, C.W. (1975). *Life Contingencies*. Society of Actuaries.
- Kirkby, R. (2022). "Quantitative Macroeconomics: Lessons Learned from Fourteen Replications." *Computational Economics*, 60: 875–896.
- Tauchen, G. (1986). "Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions." *Economics Letters*, 20(2): 177–181.

---

## Licence

This replication package is provided for academic purposes. The original paper is copyright of Elsevier (Journal of Monetary Economics). The code in this repository is released under the MIT Licence.
