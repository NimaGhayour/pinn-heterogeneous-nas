import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.float64)

from src.utils.data_loader_helmholtz import load_helmholtz_data
from src.pinn_helmholtz import HeterogeneousPINNHelmholtz
from src.nsga2 import run_nsga2

X_u, u_train, X_f, X_star, u_star, lb, ub = load_helmholtz_data(
    N_u=100, N_f=6000, N_star=2500, device='cpu'
)

pareto_front, history = run_nsga2(
    X_u, u_train, X_f, X_star, u_star, lb, ub,
    device='cpu',
    checkpoint_path='results/nsga2_run_helmholtz/checkpoint.json',
    history_path='results/nsga2_run_helmholtz/history.json',
    model_class=HeterogeneousPINNHelmholtz
)