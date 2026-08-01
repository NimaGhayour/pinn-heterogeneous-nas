import torch
import numpy as np
import time
import json
import os
from scipy.optimize import minimize

from src.pinn_heterogeneous import HeterogeneousPINN


LAYERS = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

# Training config — lighter than baseline for NAS speed
ADAM_EPOCHS = 5000        # per evaluation (full: 10000, reduced for NAS speed)
ADAM_LR = 1e-3
LBFGS_MAXITER = 10000     # per evaluation (full: 30000, reduced for NAS speed)
STABILITY_WINDOW = 500
EARLY_STOP_PATIENCE = 500  # stop if no improvement for 500 iters
EARLY_STOP_MIN_DELTA = 1e-9


def l2_relative_error(pred, exact):
    return np.linalg.norm(exact - pred) / np.linalg.norm(exact)


def evaluate(chromosome, X_u, u_train, X_f, X_star, u_star, lb, ub,
             device='cpu', seed=1234, verbose=False):
    """
    Train a HeterogeneousPINN with the given chromosome and return
    two objective values for NSGA-II.

    Parameters
    ----------
    chromosome : list[int]  length-8, values in {0,1,2,3,4}
    X_u        : boundary/IC data  (torch.Tensor)
    u_train    : boundary/IC values (torch.Tensor)
    X_f        : collocation points (torch.Tensor)
    X_star     : full grid for evaluation (torch.Tensor)
    u_star     : reference solution (torch.Tensor)
    lb, ub     : domain bounds (numpy arrays)
    device     : 'cpu' or 'cuda'
    seed       : random seed for reproducibility
    verbose    : print training progress

    Returns
    -------
    f1 : float  L2 relative error (lower is better)
    f2 : float  loss variance in last STABILITY_WINDOW L-BFGS-B iters (lower is better)
    info : dict  extra info for logging
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float64)

    model = HeterogeneousPINN(LAYERS, lb, ub, chromosome).to(device)
    start = time.time()

    lbfgs_loss_history = []

    # ─────────────────────────────────────────
    # Phase 1: Adam
    # ─────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR)

    for epoch in range(ADAM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss, mse_u, mse_f = model.compute_loss(X_u, u_train, X_f)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 1000 == 0:
            print(f"  Adam {epoch:5d} | loss={loss.item():.3e}")

    # ─────────────────────────────────────────
    # Phase 2: L-BFGS-B
    # ─────────────────────────────────────────
    def get_weights():
        return np.concatenate([
            p.detach().cpu().numpy().ravel()
            for p in model.parameters()
        ]).astype(np.float64)

    def set_weights(w_flat):
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(
                torch.tensor(
                    w_flat[idx:idx + n],
                    dtype=p.dtype, device=p.device
                ).reshape(p.shape)
            )
            idx += n

    iter_count = [0]
    best_loss = [np.inf]
    no_improve_count = [0]
    stop_flag = [False]

    def loss_and_grad(w_flat):
        if stop_flag[0]:
            # return current loss/grad to let scipy exit cleanly
            set_weights(w_flat)
            loss, _, _ = model.compute_loss(X_u, u_train, X_f)
            return float(loss.item()), np.zeros_like(w_flat)

        set_weights(w_flat)
        model.train()

        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        loss, mse_u, mse_f = model.compute_loss(X_u, u_train, X_f)
        loss.backward()

        loss_val = float(loss.item())
        lbfgs_loss_history.append(loss_val)
        iter_count[0] += 1

        # Early stopping
        if loss_val < best_loss[0] - EARLY_STOP_MIN_DELTA:
            best_loss[0] = loss_val
            no_improve_count[0] = 0
        else:
            no_improve_count[0] += 1
            if no_improve_count[0] >= EARLY_STOP_PATIENCE:
                stop_flag[0] = True
                if verbose:
                    print(f"  Early stop at iter {iter_count[0]}")

        if verbose and iter_count[0] % 500 == 0:
            print(f"  L-BFGS-B {iter_count[0]:5d} | loss={loss_val:.3e}")

        grads = np.concatenate([
            p.grad.detach().cpu().numpy().ravel() if p.grad is not None
            else np.zeros(p.numel())
            for p in model.parameters()
        ]).astype(np.float64)

        return loss_val, grads

    minimize(
        loss_and_grad,
        get_weights(),
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': LBFGS_MAXITER,
            'maxfun': 1000000,
            'ftol': 1.0 * np.finfo(float).eps,
            'gtol': 1e-10,
            'iprint': -1,
        }
    )

    # ─────────────────────────────────────────
    # Compute objectives
    # ─────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        u_pred = model.forward(
            X_star[:, 0:1], X_star[:, 1:2]
        ).cpu().numpy()

    f1 = float(l2_relative_error(u_pred, u_star.cpu().numpy()))

    # f2: variance of loss in last STABILITY_WINDOW iters
    window = lbfgs_loss_history[-STABILITY_WINDOW:] if len(lbfgs_loss_history) >= STABILITY_WINDOW \
        else lbfgs_loss_history
    f2 = float(np.var(window))

    elapsed = time.time() - start

    info = {
        'chromosome': chromosome,
        'chromosome_str': model.chromosome_str(),
        'f1_l2_error': f1,
        'f2_loss_variance': f2,
        'lbfgs_iters': iter_count[0],
        'early_stopped': stop_flag[0],
        'elapsed_sec': round(elapsed, 1),
    }

    if verbose:
        print(f"\n  Chromosome : {model.chromosome_str()}")
        print(f"  f1 (L2)    : {f1:.4e}")
        print(f"  f2 (var)   : {f2:.4e}")
        print(f"  Time       : {elapsed:.1f}s")

    return f1, f2, info


def evaluate_population(population, X_u, u_train, X_f, X_star, u_star,
                         lb, ub, device='cpu', checkpoint_path=None):
    """
    Evaluate a list of chromosomes and return their objective values.
    Saves checkpoint after each evaluation for Colab session safety.

    Parameters
    ----------
    population       : list of chromosomes
    checkpoint_path  : if given, save results to this JSON file after each eval

    Returns
    -------
    results : list of dicts with f1, f2, info for each chromosome
    """
    results = []

    # Load existing checkpoint if available (resume after session crash)
    evaluated_indices = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            existing = json.load(f)
        results = existing
        evaluated_indices = set(range(len(existing)))
        print(f"Resumed from checkpoint: {len(existing)} already evaluated.")

    for i, chromosome in enumerate(population):
        if i in evaluated_indices:
            continue

        print(f"Evaluating [{i+1}/{len(population)}] {chromosome} ...")
        f1, f2, info = evaluate(
            chromosome, X_u, u_train, X_f, X_star, u_star,
            lb, ub, device=device, verbose=False
        )
        results.append({'index': i, 'f1': f1, 'f2': f2, 'info': info})

        # Save checkpoint after every evaluation
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump(results, f, indent=2)

    return results