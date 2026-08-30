import torch
import numpy as np
import time
import json
import os
from scipy.optimize import minimize

from src.pinn_heterogeneous import HeterogeneousPINN
from src.pinn_helmholtz import HeterogeneousPINNHelmholtz

LAYERS = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

ADAM_EPOCHS       = 5000  
ADAM_LR           = 1e-3    
LBFGS_MAXITER     = 10000
STABILITY_WINDOW  = 100

def l2_relative_error(pred, exact):
    return np.linalg.norm(exact - pred) / np.linalg.norm(exact)


def evaluate(chromosome, X_u, u_train, X_f, X_star, u_star, lb, ub,
             device='cpu', seed=1234, verbose=False, model_class=HeterogeneousPINN):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float64)

    model = model_class(LAYERS, lb, ub, chromosome).to(device)
    start = time.time()
    lbfgs_loss_history = []

    # Phase 1: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    for epoch in range(ADAM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss, _, _ = model.compute_loss(X_u, u_train, X_f)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 1000 == 0:
            print(f"  Adam {epoch:5d} | loss={loss.item():.3e} | time={time.time()-start:.0f}s")

    # Phase 2: scipy L-BFGS-B
    def get_weights():
        return np.concatenate([
            p.detach().cpu().numpy().ravel()
            for p in model.parameters()
        ]).astype(np.float64)

    def set_weights(w):
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(torch.tensor(w[idx:idx+n], dtype=p.dtype, device=p.device).reshape(p.shape))
            idx += n

    iter_count = [0]

    def loss_and_grad(w):
        set_weights(w)
        model.train()
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        loss, _, _ = model.compute_loss(X_u, u_train, X_f)
        loss.backward()
        loss_val = float(loss.item())
        lbfgs_loss_history.append(loss_val)
        iter_count[0] += 1
        if verbose and iter_count[0] % 2000 == 0:
            print(f"  LBFGS {iter_count[0]:5d} | loss={loss_val:.3e} | time={time.time()-start:.0f}s")
        grads = np.concatenate([
            p.grad.detach().cpu().numpy().ravel() if p.grad is not None
            else np.zeros(p.numel())
            for p in model.parameters()
        ]).astype(np.float64)
        return loss_val, grads

    minimize(loss_and_grad, get_weights(), method='L-BFGS-B', jac=True,
             options={'maxiter': LBFGS_MAXITER, 'maxfun': 1000000,
                      'ftol': 1.0 * np.finfo(float).eps, 'gtol': 1e-10, 'iprint': -1})

    model.eval()
    with torch.no_grad():
        u_pred = model.forward(X_star[:, 0:1], X_star[:, 1:2]).cpu().numpy()

    f1 = float(l2_relative_error(u_pred, u_star.cpu().numpy()))

    window = lbfgs_loss_history[-STABILITY_WINDOW:] if len(lbfgs_loss_history) >= STABILITY_WINDOW \
        else lbfgs_loss_history
    f2 = float(np.var(window))

    elapsed = time.time() - start
    info = {
        'chromosome':       chromosome,
        'chromosome_str':   model.chromosome_str(),
        'f1_l2_error':      f1,
        'f2_loss_variance': f2,
        'lbfgs_iters':      iter_count[0],
        'elapsed_sec':      round(elapsed, 1),
    }

    if verbose:
        print(f"\n  Chromosome : {model.chromosome_str()}")
        print(f"  f1 (L2)    : {f1:.4e}")
        print(f"  f2 (var)   : {f2:.4e}")
        print(f"  Time       : {elapsed:.1f}s")

    return f1, f2, info


def evaluate_population(population, X_u, u_train, X_f, X_star, u_star,
                         lb, ub, device='cpu', checkpoint_path=None,
                         model_class=HeterogeneousPINN):
    results = []
    evaluated_indices = set()
    chromosome_cache = {}

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            existing = json.load(f)
        results = existing
        evaluated_indices = set(range(len(existing)))
        for r in existing:
            key = tuple(r['info']['chromosome'])
            chromosome_cache[key] = {'f1': r['f1'], 'f2': r['f2'], 'info': r['info']}
        print(f"Resumed from checkpoint: {len(existing)} already evaluated.")

    for i, chromosome in enumerate(population):
        if i in evaluated_indices:
            continue

        key = tuple(chromosome)
        if key in chromosome_cache:
            cached = chromosome_cache[key]
            print(f"Evaluating [{i+1}/{len(population)}] {chromosome} ... [CACHED]")
            results.append({'index': i, 'f1': cached['f1'], 'f2': cached['f2'], 'info': cached['info']})
            print(f"  -> f1={cached['f1']:.4e}  f2={cached['f2']:.4e}  [from cache]")
        else:
            print(f"Evaluating [{i+1}/{len(population)}] {chromosome} ...")
            f1, f2, info = evaluate(
                chromosome, X_u, u_train, X_f, X_star, u_star,
                lb, ub, device=device, verbose=False, model_class=model_class
            )
            chromosome_cache[key] = {'f1': f1, 'f2': f2, 'info': info}
            results.append({'index': i, 'f1': f1, 'f2': f2, 'info': info})
            print(f"  -> f1={f1:.4e}  f2={f2:.4e}  time={info['elapsed_sec']}s")

        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump(results, f, indent=2)

    return results