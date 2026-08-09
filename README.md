# PINN with Heterogeneous Activation Functions via Multi-Objective NAS

> **Undergraduate Thesis Project** — Investigating whether assigning different activation functions to different layers of a Physics-Informed Neural Network (PINN) can improve accuracy and training stability, using NSGA-II evolutionary multi-objective optimization.

---

## Overview

Standard PINNs (Raissi et al., 2019) use a single fixed activation function (Tanh) across all hidden layers. This project investigates whether assigning **different activation functions to different layers** can improve accuracy and training stability, using the 1D Burgers equation as a benchmark.

**Search space per layer:** `{Tanh, Sine, Swish}`

**Two simultaneous objectives (Multi-Objective):**
- **Objective 1:** Minimize physical accuracy error (L2 relative error on validation grid)
- **Objective 2:** Minimize training instability (loss variance during the final optimization phase)

**Algorithm:** NSGA-II (Deb et al., 2002), implemented from scratch — outputs a **Pareto Front** of architectures, each representing a different trade-off between accuracy and stability.

**Benchmark PDE:** 1D Burgers equation (same as Raissi 2019):

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, \quad \nu = \frac{0.01}{\pi}$$

with $u(x,0) = -\sin(\pi x)$, $u(-1,t) = u(1,t) = 0$, $x \in [-1,1]$, $t \in [0,1]$.

---

## Repository Structure

```
pinn-heterogeneous-nas/
├── data/
│   └── burgers_shock.mat        # Burgers equation reference solution (from Raissi GitHub)
├── src/
│   ├── pinn_baseline.py         # Baseline PINN model class (Raissi 2019 architecture, Tanh only)
│   ├── pinn_heterogeneous.py    # PINN with per-layer activation function support
│   ├── evaluator.py             # Trains a chromosome, returns (f1, f2) objectives
│   ├── nsga2.py                 # NSGA-II implementation (sorting, crowding, diversity control)
│   ├── train_baseline.py        # Baseline training script (Adam + L-BFGS-B)
│   └── utils/
│       └── data_loader.py       # Data loading with Latin Hypercube Sampling
├── results/
│   ├── baseline/                # N_u=100, N_f=10000
│   ├── nu100_nf6000/            # N_u=100, N_f=6000
│   ├── nu200_nf10000/           # N_u=200, N_f=10000
│   └── nsga2_run/                # NSGA-II checkpoints and generation history
├── run_nsga2.py                 # Entry point to run the full NSGA-II search
├── final_eval.py                # Fully trains and compares NSGA-II candidates vs. baseline
├── requirements.txt
└── .gitignore
```

---

## Baseline Results

Replication of Raissi et al. (2019) Table 2 — Burgers equation, architecture: `[2, 20×8, 1]` (3,021 parameters), training: Adam (10,000 epochs, lr=1e-3) → L-BFGS-B (scipy, maxiter=30,000).

Collocation points sampled with **Latin Hypercube Sampling (LHS)** via `scipy.stats.qmc`, which provides better spatial coverage than the random sampling used in the original paper — this accounts for our results consistently outperforming the reported targets.

| N_u | N_f | Raissi 2019 Target | **Our L2 Error** |
|-----|-----|--------------------|------------------|
| 100 | 10000 | 6.7e-04 | **3.02e-04** ✅ |
| 100 | 6000  | 7.2e-03 | **6.45e-04** ✅ |
| 200 | 10000 | 4.9e-04 | **3.31e-04** ✅ |

---

## NSGA-II Search: Method and Findings

### Method

Because a single full training run (Adam=10,000 + L-BFGS-B=30,000) takes ~75–90 minutes on CPU, evaluating dozens of candidate architectures at that cost is impractical. We therefore use a **proxy task**: each candidate chromosome is trained with a reduced protocol (Adam=5,000, L-BFGS-B=10,000) during the search, and only the best candidates from the resulting Pareto front are re-trained with the full protocol for final comparison against the baseline.

The proxy task additionally uses a reduced collocation set (N_f=6,000 instead of 10,000) to further cut per-evaluation cost; the final full-training comparison against the baseline uses the same N_f=10,000 configuration as the baseline itself, ensuring a fair final comparison.

Two safeguards were added to the NSGA-II loop to counter premature convergence, a known failure mode where the population collapses to a handful of duplicate individuals well before the search budget is exhausted:
- **Duplicate prevention:** any offspring or immigrant chromosome that already exists in the current population (or within the same generation's batch) is replaced with a fresh random chromosome before evaluation, so no training run is wasted re-evaluating an architecture we already scored.
- **Diversity injection:** if the fraction of unique chromosomes in the population drops below 50%, the weakest individuals are replaced with random immigrants to keep the search exploring.

### Result: Tanh outperforms all heterogeneous combinations found

Across two independent NSGA-II runs (different population sizes, generation counts, and mutation rates), **no heterogeneous chromosome outperformed the uniform-Tanh baseline** once re-trained with the full protocol and a fair, fixed data split:

| Architecture | L2 Error (full training) |
|---|---|
| **Baseline (all-Tanh)** | **3.02e-04** ✅ |
| Best NSGA-II candidate (`sine,tanh,tanh,swish,tanh,tanh,sine,tanh`) | 4.46e-03 |
| 2nd best (`sine,tanh,tanh,sine,swish,sine,sine,tanh`) | 6.53e-03 |
| 3rd best (`sine,sine,swish,sine,tanh,swish,sine,swish`) | 1.22e-02 |

Every top candidate found by NSGA-II contains at least one **Sine** activation. We interpret this negative result as consistent with a known limitation of periodic activations: Sine is well-suited to smooth, oscillatory targets (e.g., wave or Helmholtz-type problems), but the Burgers equation develops a **sharp, non-periodic shock** near $x=0,\ t\to1$. A periodic activation embedded in the network appears to introduce spurious oscillations near the discontinuity, analogous to the Gibbs phenomenon in Fourier approximations — degrading accuracy despite performing competitively under the reduced-budget proxy task.

### Interpretation

This is treated as a genuine (negative) finding rather than a failed experiment: **for shock-dominated PDEs like Burgers, a uniform smooth/bounded activation (Tanh) appears to be a stronger inductive bias than heterogeneous per-layer combinations drawn from `{Tanh, Sine, Swish}`.** Whether heterogeneous activations help on smoother, multi-scale, or oscillatory PDEs (e.g., Helmholtz, wave equations) remains an open question and is the natural next step for this project.

---

## Installation

```bash
git clone https://github.com/NimaGhayour/pinn-heterogeneous-nas.git
cd pinn-heterogeneous-nas
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, NumPy, SciPy

**Data:** Download `burgers_shock.mat` from the [Raissi et al. GitHub](https://github.com/maziarraissi/PINNs/blob/master/appendix/Data/burgers_shock.mat) (branch: `master` → `appendix/Data/`) and place it in `data/burgers_shock.mat`.

---

## Usage

```bash
# Reproduce baseline (N_u=100, N_f=10000)
python src/train_baseline.py

# Run the full NSGA-II search (proxy task, checkpointed after every evaluation)
python run_nsga2.py

# Fully train and compare top NSGA-II candidates against the baseline
python final_eval.py
```

Results are saved to `results/` as `model.pt` (model weights + training history + L2 error) and, for the NSGA-II search, as `checkpoint.json` / `history.json` under `results/nsga2_run/`.

---

## Roadmap

- [x] Baseline PINN replication (Raissi 2019)
- [x] Sensitivity analysis: effect of N_u and N_f on L2 error
- [x] Heterogeneous PINN: per-layer activation function support
- [x] NSGA-II framework: chromosome encoding, crossover, mutation, diversity control
- [x] Multi-objective fitness evaluation (accuracy + stability)
- [x] Pareto Front analysis and full-training validation against baseline
- [ ] Comparison with LAAF baseline (Jagtap et al., 2020)
- [ ] Extension to a second PDE (Helmholtz 2D or Allen-Cahn) to test whether heterogeneous activations help on smoother/oscillatory problems

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs. *Journal of Computational Physics*, 378, 686–707.
2. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
3. Jagtap, A. D., Kawaguchi, K., & Karniadakis, G. E. (2020). Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks. *Proceedings of the Royal Society A*, 476(2239).
4. Cuomo, S., et al. (2022). Scientific machine learning through physics-informed neural networks: Where we are and what's next. *Journal of Scientific Computing*, 92, 88.
5. Wong, J. C., et al. (2025). Evolutionary optimization of physics-informed neural networks: Evo-PINN frontiers and opportunities. *(Survey)*
