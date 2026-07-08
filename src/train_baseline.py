import torch
import numpy as np
import time
import os
import sys
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pinn_baseline import PINN
from src.utils.data_loader import load_burgers_data


def l2_relative_error(pred, exact):
    return np.linalg.norm(exact - pred) / np.linalg.norm(exact)


def train():
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.set_default_dtype(torch.float64)

    device = torch.device('cpu')
    print(f"Device: {device}\n")

    X_u, u_train, X_f, X_star, u_star, lb, ub = load_burgers_data(
        data_path='data/burgers_shock.mat',
        N_u=200,
        N_f=10000,
        device=device
    )

    print(f"lb = {lb},  ub = {ub}")
    print(f"X_u: {X_u.shape}, X_f: {X_f.shape}\n")

    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = PINN(layers, lb, ub).to(device)
    print(f"Architecture: {layers}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    history = {'loss': [], 'mse_u': [], 'mse_f': []}
    start = time.time()

    # ===================================================
    # Phase 1: Adam
    # ===================================================
    print("=== Phase 1: Adam (10000 epochs, lr=1e-3 constant) ===")
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10000):
        model.train()
        optimizer_adam.zero_grad()
        loss, mse_u, mse_f = model.compute_loss(X_u, u_train, X_f)
        loss.backward()
        optimizer_adam.step()

        history['loss'].append(loss.item())
        history['mse_u'].append(mse_u.item())
        history['mse_f'].append(mse_f.item())

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.3e} | "
                  f"MSE_u: {mse_u.item():.3e} | "
                  f"MSE_f: {mse_f.item():.3e} | "
                  f"Time: {time.time()-start:.0f}s")

    # ===================================================
    # Phase 2: scipy L-BFGS-B — Same as Raissi 2019
    # ===================================================
    print("\n=== Phase 2: scipy L-BFGS-B (exact same as Raissi 2019) ===")

    def get_weights():
        weights = []
        for p in model.parameters():
            weights.append(p.detach().cpu().numpy().ravel())
        return np.concatenate(weights).astype(np.float64)

    def set_weights(w_flat):
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(
                torch.tensor(
                    w_flat[idx:idx + n],
                    dtype=p.dtype,
                    device=p.device
                ).reshape(p.shape)
            )
            idx += n

    lbfgs_iter = [0]

    def loss_and_grad(w_flat):
        set_weights(w_flat)
        model.train()

        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # forward + backward
        loss, mse_u, mse_f = model.compute_loss(X_u, u_train, X_f)
        loss.backward()

        lbfgs_iter[0] += 1
        history['loss'].append(loss.item())

        if lbfgs_iter[0] % 1000 == 0:
            print(f"  L-BFGS-B {lbfgs_iter[0]:5d} | "
                  f"Loss: {loss.item():.3e} | "
                  f"Time: {time.time()-start:.0f}s")

        grads = []
        for p in model.parameters():
            g = (p.grad.detach().cpu().numpy().ravel()
                 if p.grad is not None
                 else np.zeros(p.numel()))
            grads.append(g)

        return float(loss.item()), np.concatenate(grads).astype(np.float64)

    result = minimize(
        loss_and_grad,
        get_weights(),
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': 30000,
            'maxfun': 1000000,
            'ftol': 1.0 * np.finfo(float).eps,
            'gtol': 1e-10,
            'iprint': -1,
        }
    )

    set_weights(result.x)

    print(f"\n  Converged: {result.success}")
    print(f"  Message  : {result.message}")
    print(f"  Iters    : {result.nit}")
    print(f"  Fun evals: {result.nfev}")
    print(f"  Final loss: {result.fun:.4e}")

    # ===================================================
    # Results
    # ===================================================
    model.eval()
    with torch.no_grad():
        u_pred = model.forward(
            X_star[:, 0:1],
            X_star[:, 1:2]
        ).cpu().numpy()

    error = l2_relative_error(u_pred, u_star.cpu().numpy())

    print(f"\n{'='*40}")
    print(f" L2 Relative Error : {error:.4e}")
    print(f" Raissi 2019 target: ~4.9e-04")
    print(f" Total Time        : {time.time()-start:.0f}s")
    print(f"{'='*40}")

    os.makedirs('results/nu200_nf10000', exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'history': history,
        'l2_error': error,
        'lb': lb,
        'ub': ub,
        'config': {'layers': layers, 'N_u': 200, 'N_f': 10000}
    }, 'results/nu200_nf10000/model.pt')
    print("Saved: results/nu200_nf10000/model.pt")

    return model, history, error


if __name__ == '__main__':
    train()