import torch
import torch.nn as nn
import numpy as np


class PINN(nn.Module):

    def __init__(self, layers, lb, ub):
        super(PINN, self).__init__()

        self.register_buffer('lb', torch.tensor(lb, dtype=torch.float64))
        self.register_buffer('ub', torch.tensor(ub, dtype=torch.float64))

        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.linears.append(linear)

        self.tanh = nn.Tanh()

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)  # (N, 2)

        inp = 2.0 * (inp - self.lb) / (self.ub - self.lb) - 1.0

        out = inp
        for i, layer in enumerate(self.linears):
            out = layer(out)
            if i < len(self.linears) - 1:
                out = self.tanh(out)
        return out

    def residual(self, x, t):
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)

        u = self.forward(x, t)

        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True)[0]

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True)[0]

        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True)[0]

        nu = 0.01 / np.pi
        return u_t + u * u_x - nu * u_xx

    def compute_loss(self, X_u, u, X_f):
        x_u, t_u = X_u[:, 0:1], X_u[:, 1:2]
        x_f, t_f = X_f[:, 0:1], X_f[:, 1:2]

        u_pred = self.forward(x_u, t_u)
        mse_u = torch.mean((u - u_pred) ** 2)

        f_pred = self.residual(x_f, t_f)
        mse_f = torch.mean(f_pred ** 2)

        return mse_u + mse_f, mse_u, mse_f