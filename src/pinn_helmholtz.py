import torch
import torch.nn as nn
import numpy as np

from src.pinn_heterogeneous import HeterogeneousPINN


# ─────────────────────────────────────────────────────────────
# Helmholtz equation:
#     u_xx + u_yy + k^2 * u = f(x, y),   (x, y) in [-1, 1]^2
#
# Manufactured solution:
#     u(x, y) = sin(pi * x) * sin(4 * pi * y)
#
# Forcing term (derived analytically so that u solves the PDE exactly):
#     f(x, y) = -(pi^2 + (4*pi)^2 - k^2) * sin(pi*x) * sin(4*pi*y)
# ─────────────────────────────────────────────────────────────

A1 = 1.0            # frequency multiplier on x  -> sin(A1 * pi * x)
A2 = 4.0            # frequency multiplier on y  -> sin(A2 * pi * y)
K  = 1.0            # Helmholtz wavenumber


def exact_solution(x, y):
    """Analytical solution u(x, y) = sin(A1*pi*x) * sin(A2*pi*y)."""
    return np.sin(A1 * np.pi * x) * np.sin(A2 * np.pi * y)


def forcing_term(x, y):
    """Analytical forcing f(x, y) matching the manufactured solution."""
    lap_coeff = (A1 * np.pi) ** 2 + (A2 * np.pi) ** 2
    return -(lap_coeff - K ** 2) * np.sin(A1 * np.pi * x) * np.sin(A2 * np.pi * y)


class HeterogeneousPINNHelmholtz(HeterogeneousPINN):
    """
    Same architecture and per-layer activation search space as
    HeterogeneousPINN, but with a residual() and compute_loss()
    specialized for the 2D Helmholtz equation instead of Burgers.

    Input convention: forward(x, y) instead of forward(x, t) — the
    parent class's forward() is reused unchanged since it only cares
    about concatenating two spatial coordinates and normalizing them.
    """

    def residual(self, x, y):
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)

        u = self.forward(x, y)

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True)[0]
        u_y = torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u),
            create_graph=True)[0]

        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True)[0]
        u_yy = torch.autograd.grad(
            u_y, y, grad_outputs=torch.ones_like(u_y),
            create_graph=True)[0]

        f = forcing_term_torch(x, y)

        return u_xx + u_yy + (K ** 2) * u - f

    def compute_loss(self, X_u, u, X_f):
        """
        X_u : boundary points (N_u, 2) -> columns [x, y]
        u   : exact boundary values (N_u, 1)
        X_f : collocation points (N_f, 2) -> columns [x, y]
        """
        x_u, y_u = X_u[:, 0:1], X_u[:, 1:2]
        x_f, y_f = X_f[:, 0:1], X_f[:, 1:2]

        u_pred = self.forward(x_u, y_u)
        mse_u = torch.mean((u - u_pred) ** 2)

        f_pred = self.residual(x_f, y_f)
        mse_f = torch.mean(f_pred ** 2)

        return mse_u + mse_f, mse_u, mse_f


def forcing_term_torch(x, y):
    """Torch version of forcing_term(), used inside the autograd graph."""
    lap_coeff = (A1 * np.pi) ** 2 + (A2 * np.pi) ** 2
    return -(lap_coeff - K ** 2) * torch.sin(A1 * np.pi * x) * torch.sin(A2 * np.pi * y)