import torch
import numpy as np
import sys
import os
import time
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)

from src.utils.data_loader_helmholtz import load_helmholtz_data
from src.pinn_helmholtz import HeterogeneousPINNHelmholtz


def l2_relative_error(pred, exact):
    return np.linalg.norm(exact - pred) / np.linalg.norm(exact)


def full_train(chromosome, X_u, u_train, X_f, X_star, u_star, lb, ub, name,
               adam_epochs=10000, lbfgs_maxiter=30000):
    print(f"\n{'='*50}")
    print(f"{name}: {chromosome} | Adam={adam_epochs} LBFGS={lbfgs_maxiter}")
    print(f"{'='*50}")
    torch.manual_seed(1234)
    np.random.seed(1234)

    model = HeterogeneousPINNHelmholtz(
        [2, 20, 20, 20, 20, 20, 20, 20, 20, 1], lb, ub, chromosome
    )
    start = time.time()

    # ── Phase 1: Adam ──────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(adam_epochs):
        model.train()
        optimizer.zero_grad()
        loss, mse_u, mse_f = model.compute_loss(X_u, u_train, X_f)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"  Adam {epoch:5d} | loss={loss.item():.3e} | "
                  f"mse_u={mse_u.item():.3e} | mse_f={mse_f.item():.3e} | "
                  f"time={time.time()-start:.0f}s")

    # ── Phase 2: scipy L-BFGS-B ─────────────────────────
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
             options={'maxiter': lbfgs_maxiter, 'maxfun': 1000000,
                      'ftol': 1.0 * np.finfo(float).eps, 'gtol': 1e-10, 'iprint': -1})

    # ── Evaluation ───────────────────────────────────────
    model.eval()
    with torch.no_grad():
        u_pred = model.forward(X_star[:, 0:1], X_star[:, 1:2]).numpy()

    f1 = l2_relative_error(u_pred, u_star.numpy())
    elapsed = time.time() - start
    print(f"\n  L2 Relative Error : {f1:.4e}")
    print(f"  Total Time        : {elapsed:.0f}s")
    return f1, model


if __name__ == '__main__':
    X_u, u_train, X_f, X_star, u_star, lb, ub = load_helmholtz_data(
        N_u=100, N_f=10000, N_star=2500, device='cpu'
    )

    print(f"X_u: {X_u.shape}, X_f: {X_f.shape}, X_star: {X_star.shape}")

    chromosome = [0, 0, 0, 0, 0, 0, 0, 0]  # all-tanh baseline
    f1, model = full_train(chromosome, X_u, u_train, X_f, X_star, u_star, lb, ub,
                           name='baseline [all-tanh]')

    os.makedirs('results/helmholtz_baseline', exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'l2_error': f1,
        'lb': lb,
        'ub': ub,
        'config': {'N_u': 100, 'N_f': 10000, 'chromosome': chromosome}
    }, 'results/helmholtz_baseline/model.pt')
    print("\nSaved: results/helmholtz_baseline/model.pt")