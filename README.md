# PINN with Heterogeneous Activation Functions via Multi-Objective NAS

> **Undergraduate Thesis Project** — Optimizing Physics-Informed Neural Network (PINN) architectures by searching for the best per-layer activation function combination using NSGA-II evolutionary algorithm with multi-objective optimization.

---

## Overview

Standard PINNs (Raissi et al., 2019) use a single fixed activation function (Tanh) across all hidden layers. This project investigates whether assigning **different activation functions to different layers** can improve accuracy and training stability.

**Search space per layer:** `{Tanh, Sine, Swish, ReLU, Sigmoid}`

**Two simultaneous objectives (Multi-Objective):**
- **Objective 1:** Maximize physical accuracy (PDE residual error on validation set)
- **Objective 2:** Maximize training stability (minimize loss variance during training)

**Algorithm:** NSGA-II (Deb et al., 2002) — outputs a **Pareto Front** of architectures, each representing a different trade-off between accuracy and stability.

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
│   ├── pinn_baseline.py         # PINN model class (Raissi 2019 architecture)
│   ├── train_baseline.py        # Training script (Adam + L-BFGS-B)
│   └── utils/
│       ├── data_loader.py       # Data loading with Latin Hypercube Sampling
│       └── plotting.py          # Visualization utilities
├── results/
│   ├── baseline/                # N_u=100, N_f=10000
│   ├── nu100_nf6000/            # N_u=100, N_f=6000
│   └── nu200_nf10000/           # N_u=200, N_f=10000
├── notebooks/
├── requirements.txt
└── .gitignore
```

---

## Baseline Results

Replication of Raissi et al. (2019) Table 2 — Burgers equation, architecture: `[2, 20×8, 1]` (3,021 parameters), training: Adam (10,000 epochs, lr=1e-3) → L-BFGS-B (scipy, maxiter=30,000).

Collocation points sampled with **Latin Hypercube Sampling (LHS)** via `scipy.stats.qmc`, which provides better spatial coverage than the random sampling used in the original paper — this accounts for our results consistently outperforming the reported targets.

| N_u | N_f | Raissi 2019 Target | **Our L2 Error** | Time (CPU) |
|-----|-----|--------------------|------------------|------------|
| 100 | 10000 | 6.7e-04 | **3.02e-04** ✅ | ~4901s |
| 100 | 6000  | 7.2e-03 | **6.45e-04** ✅ | ~3218s |
| 200 | 10000 | 4.9e-04 | **3.31e-04** ✅ | ~6937s |

> **Note on timing:** Raissi reports ~60s on an NVIDIA Titan X GPU. Our CPU runs are ~80× slower (this is consistent with known GPU/CPU speedup ratios for this class of problem).

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
```

Results are saved to `results/` as `model.pt` (model weights + training history + L2 error).

---

## Roadmap

- [x] Baseline PINN replication (Raissi 2019)
- [x] Sensitivity analysis: effect of N_u and N_f on L2 error
- [x] Heterogeneous PINN: per-layer activation function support
- [ ] NSGA-II framework: chromosome encoding, crossover, mutation
- [ ] Multi-objective fitness evaluation (accuracy + stability)
- [ ] Pareto Front analysis and visualization
- [ ] Comparison with LAAF baseline (Jagtap et al., 2020)
- [ ] Extension to second PDE (Helmholtz 2D or Allen-Cahn)

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs. *Journal of Computational Physics*, 378, 686–707.
2. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
3. Jagtap, A. D., Kawaguchi, K., & Karniadakis, G. E. (2020). Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks. *Proceedings of the Royal Society A*, 476(2239).
4. Cuomo, S., et al. (2022). Scientific machine learning through physics-informed neural networks: Where we are and what's next. *Journal of Scientific Computing*, 92, 88.
5. Wong, J. C., et al. (2025). Evolutionary optimization of physics-informed neural networks: Evo-PINN frontiers and opportunities. *(Survey)*

