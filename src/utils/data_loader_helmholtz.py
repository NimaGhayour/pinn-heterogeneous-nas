import numpy as np
import torch
from scipy.stats import qmc

from src.pinn_helmholtz import exact_solution


def load_helmholtz_data(N_u=100, N_f=10000, N_star=2500, device='cpu', seed=1234):
    """
    Generate training and evaluation data for the 2D Helmholtz benchmark
    on the domain [-1, 1] x [-1, 1], using the manufactured solution
    u(x, y) = sin(pi*x) * sin(4*pi*y).

    Parameters
    ----------
    N_u     : number of boundary points (sampled from the 4 edges of the square)
    N_f     : number of interior collocation points (Latin Hypercube Sampling)
    N_star  : size of the evaluation grid (N_star = n*n, n = sqrt(N_star))
    device  : 'cpu' or 'cuda'
    seed    : random seed for reproducibility

    Returns
    -------
    X_u, u_train, X_f, X_star, u_star, lb, ub
        Same structure/signature as load_burgers_data(), so it plugs
        directly into evaluator.py and nsga2.py without changes.
    """
    rng = np.random.RandomState(seed)

    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])

    # ── Boundary points: sample uniformly from the 4 edges of the square ──
    n_per_edge = N_u // 4

    # left edge: x = -1
    y_left = rng.uniform(-1, 1, n_per_edge)
    x_left = -np.ones(n_per_edge)

    # right edge: x = 1
    y_right = rng.uniform(-1, 1, n_per_edge)
    x_right = np.ones(n_per_edge)

    # bottom edge: y = -1
    x_bottom = rng.uniform(-1, 1, n_per_edge)
    y_bottom = -np.ones(n_per_edge)

    # top edge: y = 1
    x_top = rng.uniform(-1, 1, N_u - 3 * n_per_edge)  # absorb rounding remainder
    y_top = np.ones(N_u - 3 * n_per_edge)

    x_u_all = np.concatenate([x_left, x_right, x_bottom, x_top])
    y_u_all = np.concatenate([y_left, y_right, y_bottom, y_top])

    X_u_train = np.stack([x_u_all, y_u_all], axis=1)
    u_train = exact_solution(x_u_all, y_u_all)[:, None]

    # ── Collocation points: Latin Hypercube Sampling over the interior ──
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    X_f_train = lb + (ub - lb) * sampler.random(n=N_f)
    # augment with boundary points, consistent with our Burgers data loader
    X_f_train = np.vstack((X_f_train, X_u_train))

    # ── Evaluation grid: uniform n x n grid over the full domain ──
    n = int(np.sqrt(N_star))
    x_grid = np.linspace(-1, 1, n)
    y_grid = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x_grid, y_grid)
    X_star = np.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
    u_star = exact_solution(X.flatten(), Y.flatten())[:, None]

    # ── Convert to torch tensors ──
    X_u_train = torch.tensor(X_u_train, dtype=torch.float64).to(device)
    u_train   = torch.tensor(u_train,   dtype=torch.float64).to(device)
    X_f_train = torch.tensor(X_f_train, dtype=torch.float64).to(device)
    X_star    = torch.tensor(X_star,    dtype=torch.float64).to(device)
    u_star    = torch.tensor(u_star,    dtype=torch.float64).to(device)

    return X_u_train, u_train, X_f_train, X_star, u_star, lb, ub