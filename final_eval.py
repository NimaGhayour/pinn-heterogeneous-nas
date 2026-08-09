import torch
import numpy as np
import sys
import os
import time
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.float64)

from src.utils.data_loader import load_burgers_data
from src.pinn_heterogeneous import HeterogeneousPINN


def l2_relative_error(pred, exact):
    return np.linalg.norm(exact - pred) / np.linalg.norm(exact)


def full_train(chromosome, X_u, u_train, X_f, X_star, u_star, lb, ub, name):
    print(f"\n{'='*50}")
    print(f"{name}: {chromosome}")
    print(f"{'='*50}")
    torch.manual_seed(1234)
    np.random.seed(1234)

    model = HeterogeneousPINN([2, 20, 20, 20, 20, 20, 20, 20, 20, 1], lb, ub, chromosome)
    start = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10000):
        model.train()
        optimizer.zero_grad()
        loss, _, _ = model.compute_loss(X_u, u_train, X_f)
        loss.backward()
        optimizer.step()
        if epoch % 2000 == 0:
            print(f"  Adam {epoch:5d} | loss={loss.item():.3e} | time={time.time()-start:.0f}s")

    def get_weights():
        return np.concatenate([p.detach().numpy().ravel()
                               for p in model.parameters()]).astype(np.float64)

    def set_weights(w):
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(torch.tensor(w[idx:idx+n], dtype=p.dtype).reshape(p.shape))
            idx += n

    iters = [0]

    def loss_and_grad(w):
        set_weights(w)
        model.train()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss, _, _ = model.compute_loss(X_u, u_train, X_f)
        loss.backward()
        iters[0] += 1
        if iters[0] % 5000 == 0:
            print(f"  LBFGS {iters[0]:5d} | loss={loss.item():.3e} | time={time.time()-start:.0f}s")
        grads = np.concatenate([p.grad.detach().numpy().ravel() if p.grad is not None
                                else np.zeros(p.numel()) for p in model.parameters()]).astype(np.float64)
        return float(loss.item()), grads

    minimize(loss_and_grad, get_weights(), method='L-BFGS-B', jac=True,
             options={'maxiter': 30000, 'maxfun': 1000000,
                      'ftol': 1.0 * np.finfo(float).eps, 'gtol': 1e-10, 'iprint': -1})

    model.eval()
    with torch.no_grad():
        u_pred = model.forward(X_star[:, 0:1], X_star[:, 1:2]).numpy()

    f1 = l2_relative_error(u_pred, u_star.numpy())
    elapsed = time.time() - start
    print(f"\n  f1 (L2) : {f1:.4e}")
    print(f"  Time    : {elapsed:.0f}s")
    return f1


# ── Seed BEFORE loading data, so the boundary-point split matches
#    the one used for our original baseline (3.02e-04) ──────────
torch.manual_seed(1234)
np.random.seed(1234)

X_u, u_train, X_f, X_star, u_star, lb, ub = load_burgers_data(
    data_path='data/burgers_shock.mat',
    N_u=100, N_f=10000, device='cpu'
)

candidates = {
    'nsga-best-1 [sine,sine,swish,sine,tanh,swish,sine,swish]': [1, 1, 2, 1, 0, 2, 1, 2],
    'nsga-best-2 [sine,tanh,tanh,sine,swish,sine,sine,tanh]':   [1, 0, 0, 1, 2, 1, 1, 0],
    'nsga-best-3 [sine,tanh,tanh,swish,tanh,tanh,sine,tanh]':   [1, 0, 0, 2, 0, 0, 1, 0],
    'baseline [all-tanh]':                                      [0, 0, 0, 0, 0, 0, 0, 0],
}

results = {}
for name, chrom in candidates.items():
    f1 = full_train(chrom, X_u, u_train, X_f, X_star, u_star, lb, ub, name)
    results[name] = f1

print(f"\n{'='*50}")
print("FINAL RESULTS (Adam=10000, LBFGS=30000, N_f=10000):")
print(f"{'='*50}")
for name, f1 in results.items():
    print(f"  {f1:.4e}  {name}")