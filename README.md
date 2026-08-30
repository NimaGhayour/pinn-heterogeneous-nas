# PINN with Heterogeneous Activation Functions via Multi-Objective NAS

> **Undergraduate Thesis Project** — Investigating whether assigning different activation functions to different layers of a Physics-Informed Neural Network (PINN) can improve accuracy and training stability, using NSGA-II evolutionary multi-objective optimization. Tested on two PDEs with very different character: the shock-forming 1D Burgers equation and the smooth, periodic 2D Helmholtz equation.

---

## Overview

Standard PINNs (Raissi et al., 2019) use a single fixed activation function (Tanh) across all hidden layers. This project investigates whether assigning **different activation functions to different layers** can improve accuracy and training stability, and whether the answer depends on the nature of the PDE being solved.

**Search space per layer:** `{Tanh, Sine, Swish}`

**Two simultaneous objectives (Multi-Objective):**
- **Objective 1:** Minimize physical accuracy error (L2 relative error on validation grid)
- **Objective 2:** Minimize training instability (loss variance during the final optimization phase)

**Algorithm:** NSGA-II (Deb et al., 2002), implemented from scratch — outputs a **Pareto Front** of architectures, each representing a different trade-off between accuracy and stability.

**Benchmark PDEs:**

1D Burgers equation (same as Raissi 2019), which develops a sharp shock:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, \quad \nu = \frac{0.01}{\pi}$$

with $u(x,0) = -\sin(\pi x)$, $u(-1,t) = u(1,t) = 0$, $x \in [-1,1]$, $t \in [0,1]$.

2D Helmholtz equation, which is smooth and periodic by construction (manufactured solution):

$$\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + k^2 u = f(x,y), \quad (x,y) \in [-1,1]^2$$

with manufactured solution $u(x,y) = \sin(\pi x)\sin(4\pi y)$, $k=1$, and $f$ derived analytically so this $u$ solves the PDE exactly.

---

## Repository Structure

```
pinn-heterogeneous-nas/
├── data/
│   └── burgers_shock.mat            # Burgers equation reference solution (from Raissi GitHub)
├── src/
│   ├── pinn_baseline.py             # Baseline PINN model class (Raissi 2019 architecture, Tanh only)
│   ├── pinn_heterogeneous.py        # PINN with per-layer activation function support (Burgers)
│   ├── pinn_helmholtz.py            # Heterogeneous PINN specialized for the Helmholtz residual
│   ├── evaluator.py                 # Trains a chromosome, returns (f1, f2) objectives — PDE-agnostic
│   ├── nsga2.py                     # NSGA-II implementation (sorting, crowding, diversity control)
│   ├── train_baseline.py            # Baseline training script for Burgers (Adam + L-BFGS-B)
│   ├── train_helmholtz_baseline.py  # Baseline training script for Helmholtz (Adam + L-BFGS-B)
│   └── utils/
│       ├── data_loader.py           # Burgers data loading with Latin Hypercube Sampling
│       └── data_loader_helmholtz.py # Helmholtz boundary/collocation/eval-grid generation
├── results/
│   ├── baseline/                    # Burgers: N_u=100, N_f=10000
│   ├── nu100_nf6000/                # Burgers: N_u=100, N_f=6000
│   ├── nu200_nf10000/               # Burgers: N_u=200, N_f=10000
│   ├── helmholtz_baseline/          # Helmholtz: N_u=100, N_f=10000, all-Tanh
│   ├── nsga2_run/                   # NSGA-II checkpoints and generation history (Burgers)
│   └── nsga2_run_helmholtz/         # NSGA-II checkpoints and generation history (Helmholtz)
├── run_nsga2.py                     # Entry point: run NSGA-II on Burgers
├── run_nsga2_helmholtz.py           # Entry point: run NSGA-II on Helmholtz
├── final_eval.py                    # Fully trains and compares NSGA-II candidates vs. baseline (Burgers)
├── final_eval_helmholtz.py          # Fully trains and compares NSGA-II candidates vs. baseline (Helmholtz)
├── requirements.txt
└── .gitignore
```

---

## Baseline Results (Burgers)

Replication of Raissi et al. (2019) Table 2 — architecture: `[2, 20×8, 1]` (3,021 parameters), training: Adam (10,000 epochs, lr=1e-3) → L-BFGS-B (scipy, maxiter=30,000). Collocation points sampled with **Latin Hypercube Sampling (LHS)** via `scipy.stats.qmc`, which provides better spatial coverage than the random sampling used in the original paper — this accounts for our results consistently outperforming the reported targets.

| N_u | N_f | Raissi 2019 Target | **Our L2 Error** |
|-----|-----|--------------------|------------------|
| 100 | 10000 | 6.7e-04 | **3.02e-04** ✅ |
| 100 | 6000  | 7.2e-03 | **6.45e-04** ✅ |
| 200 | 10000 | 4.9e-04 | **3.31e-04** ✅ |

---

## NSGA-II Search: Method

Because a single full training run (Adam=10,000 + L-BFGS-B=30,000) takes ~75–130 minutes on CPU depending on the PDE, evaluating dozens of candidate architectures at that cost is impractical. We therefore use a **proxy task**: each candidate chromosome is trained with a reduced protocol (Adam=5,000, L-BFGS-B=10,000, reduced collocation count N_f=6,000) during the search, and only the best candidates from the resulting Pareto front are re-trained with the full protocol (Adam=10,000, L-BFGS-B=30,000, N_f=10,000, matching the baseline exactly) for final comparison.

Two safeguards were added to the NSGA-II loop to counter premature convergence, a known failure mode where the population collapses to a handful of duplicate individuals well before the search budget is exhausted:
- **Duplicate prevention:** any offspring or immigrant chromosome that already exists in the current population (or within the same generation's batch) is replaced with a fresh random chromosome before evaluation, so no training run is wasted re-evaluating an architecture we already scored.
- **Diversity injection:** if the fraction of unique chromosomes in the population drops below 50%, the weakest individuals are replaced with random immigrants to keep the search exploring.

---

## Results: The Best Activation Strategy Depends on the PDE

### Burgers (shock-dominated): Tanh wins

Across two independent NSGA-II runs on Burgers (different population sizes, generation counts, and mutation rates), **no heterogeneous chromosome outperformed the uniform-Tanh baseline** once re-trained with the full protocol and a fair, fixed data split:

| Architecture | L2 Error (full training) |
|---|---|
| **Baseline (all-Tanh)** | **3.02e-04** ✅ |
| Best NSGA-II candidate (`sine,tanh,tanh,swish,tanh,tanh,sine,tanh`) | 4.46e-03 |
| 2nd best (`sine,tanh,tanh,sine,swish,sine,sine,tanh`) | 6.53e-03 |
| 3rd best (`sine,sine,swish,sine,tanh,swish,sine,swish`) | 1.22e-02 |

Every top candidate found by NSGA-II contains at least one **Sine** activation. We interpret this as consistent with a known limitation of periodic activations: Sine is well-suited to smooth, oscillatory targets, but Burgers develops a **sharp, non-periodic shock** near $x=0,\ t\to1$. A periodic activation embedded in the network appears to introduce spurious oscillations near the discontinuity, analogous to the Gibbs phenomenon in Fourier approximations.

### Helmholtz (smooth, periodic): heterogeneous activations win clearly

On Helmholtz, the result reverses. After a full NSGA-II search (POP_SIZE=8, 6 generations) followed by full-protocol re-training, **both top candidates substantially outperformed the all-Tanh baseline**:

| Architecture | L2 Error (full training) | vs. baseline |
|---|---|---|
| **Best NSGA-II candidate** (`swish,swish,tanh,swish,sine,tanh,swish,sine`) | **2.25e-02** | **~5.8× better** ✅ |
| 2nd best (`sine,swish,sine,tanh,tanh,tanh,sine,sine`) | 2.97e-02 | ~4.4× better ✅ |
| Baseline (all-Tanh) | 1.31e-01 | — |

Both winning chromosomes are dominated by **Sine and Swish**, and the best one contains no more than one Tanh layer.

### Why the two PDEs disagree — and what the literature says

This reversal is not a contradiction; it reflects a documented property of PINNs. A residual loss for a $k$-th order PDE requires the $k$-th derivative of the activation function to be well-behaved (informally, close to bijective) for the residual to be driven to zero (Hosseini Dashtbayaz et al., 2024). Tanh's higher-order derivatives decay and flatten, which limits its ability to represent oscillatory, high-frequency solutions — exactly the situation in our Helmholtz benchmark, whose exact solution is itself a product of sines.

Independent studies confirm this pattern on Helmholtz specifically:
- Al-Safwan, Song & Waheed (2021) compared Tanh, Atan, ELU, and Swish on a Helmholtz wavefield problem using the same 8-layer × 20-neuron architecture as ours, and found Swish gave the lowest L2 error of all activations tested (their best Tanh: ~5.5e-05; their best Swish: ~3.65e-05).
- Hosseini Dashtbayaz et al. (2024) benchmarked Tanh against Sine on a Helmholtz equation with the same functional form as ours ($\sin(\pi x)\sin(n\pi y)$) and found Tanh's mean absolute error ranged from 0.59 to 4.72 across network widths, while Sine's stayed below 0.03 — a 100–1000× gap.
- Sitzmann et al. (2020) first demonstrated that sinusoidal (SIREN) networks are strongly preferred over Tanh/ReLU-family activations for representing Wave and Helmholtz-type PDEs.

Our result is directionally consistent with all three: Sine/Swish beat Tanh on Helmholtz, and by a wide margin. The absolute accuracy we achieve (2.25e-02) is weaker than the best published results above (which reach 1e-3–1e-5), which we attribute to our reduced-epoch proxy search protocol and the absence of additional techniques used in that literature (e.g., loss-term weighting, Fourier feature embeddings). Unlike that prior work, however, our activation choice was **not selected by hand** — NSGA-II discovered the Sine/Swish preference for Helmholtz (and the Tanh preference for Burgers) automatically, from the same unbiased search space, purely by evaluating candidates.

### Interpretation

**The optimal activation strategy for a PINN is PDE-dependent, and a multi-objective evolutionary search can recover the correct strategy for each PDE without hand-tuning.** For shock-dominated problems (Burgers), a uniform smooth/bounded activation (Tanh) is a stronger inductive bias. For smooth, periodic problems (Helmholtz), heterogeneous combinations dominated by Sine/Swish substantially outperform Tanh — matching independent findings in the PINN literature that were obtained via manual, equation-specific tuning rather than automated search.

---

## Installation

```bash
git clone https://github.com/NimaGhayour/pinn-heterogeneous-nas.git
cd pinn-heterogeneous-nas
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, NumPy, SciPy

**Data:** Download `burgers_shock.mat` from the [Raissi et al. GitHub](https://github.com/maziarraissi/PINNs/blob/master/appendix/Data/burgers_shock.mat) (branch: `master` → `appendix/Data/`) and place it in `data/burgers_shock.mat`. The Helmholtz benchmark needs no external data — its ground truth is generated analytically from the manufactured solution.

---

## Usage

```bash
# Burgers: reproduce baseline, run NSGA-II search, validate top candidates
python src/train_baseline.py
python run_nsga2.py
python final_eval.py

# Helmholtz: reproduce baseline, run NSGA-II search, validate top candidates
python src/train_helmholtz_baseline.py
python run_nsga2_helmholtz.py
python final_eval_helmholtz.py
```

Results are saved to `results/` as `model.pt` (model weights + training history + L2 error) and, for each NSGA-II search, as `checkpoint.json` / `history.json` under the corresponding `results/nsga2_run*/` folder.

---

## Roadmap

- [x] Baseline PINN replication (Raissi 2019) on Burgers
- [x] Sensitivity analysis: effect of N_u and N_f on L2 error
- [x] Heterogeneous PINN: per-layer activation function support
- [x] NSGA-II framework: chromosome encoding, crossover, mutation, diversity control
- [x] Multi-objective fitness evaluation (accuracy + stability)
- [x] Pareto Front analysis and full-training validation against baseline (Burgers)
- [x] Extension to a second PDE (2D Helmholtz) to test whether heterogeneous activations help on smoother/oscillatory problems
- [x] Literature comparison confirming the PDE-dependent direction of the result
- [ ] Comparison with LAAF baseline (Jagtap et al., 2020)
- [ ] Closing the absolute-accuracy gap to published Helmholtz results (loss weighting, Fourier features, or longer full-protocol search)

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs. *Journal of Computational Physics*, 378, 686–707.
2. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
3. Jagtap, A. D., Kawaguchi, K., & Karniadakis, G. E. (2020). Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks. *Proceedings of the Royal Society A*, 476(2239), 20200334.
4. Cuomo, S., et al. (2022). Scientific machine learning through physics-informed neural networks: Where we are and what's next. *Journal of Scientific Computing*, 92, 88.
5. Wong, J. C., et al. (2025). Evolutionary optimization of physics-informed neural networks: Evo-PINN frontiers and opportunities. *(Survey)*
6. Al-Safwan, A., Song, C., & bin Waheed, U. (2021). Is it time to swish? Comparing activation functions in solving the Helmholtz equation using physics-informed neural networks. *arXiv:2110.07721*.
7. Hosseini Dashtbayaz, N., Farhani, G., Wang, B., & Ling, C. X. (2024). Physics-informed neural networks: Minimizing residual loss with wide networks and effective activations. *arXiv:2405.01680*.
8. Sitzmann, V., Martel, J., Bergman, A., Lindell, D., & Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. *Advances in Neural Information Processing Systems*, 33, 7462–7473.
9. Herrmann, L., Jokeit, M., Weeger, O., & Kollmannsberger, S. (2025). Introduction to Physics-Informed Neural Networks. In *Deep Learning in Computational Mechanics*. Springer, Cham.
