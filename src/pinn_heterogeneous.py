import torch
import torch.nn as nn
import numpy as np


ACTIVATION_FUNCTIONS = {
    0: 'tanh',
    1: 'sine',
    2: 'swish',
    3: 'relu',
    4: 'sigmoid',
}


class HeterogeneousPINN(nn.Module):

    def __init__(self, layers, lb, ub, chromosome):
        """
        layers     : e.g. [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        lb, ub     : domain bounds (numpy arrays)
        chromosome : list of 8 ints, each in {0,1,2,3,4}
                     one activation per hidden layer
        """
        super(HeterogeneousPINN, self).__init__()

        assert len(chromosome) == len(layers) - 2, (
            f"chromosome length ({len(chromosome)}) must equal "
            f"number of hidden layers ({len(layers) - 2})"
        )

        self.register_buffer('lb', torch.tensor(lb, dtype=torch.float64))
        self.register_buffer('ub', torch.tensor(ub, dtype=torch.float64))

        self.chromosome = chromosome

        # Build linear layers
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            linear = linear.double() 
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.linears.append(linear)

        # Build activation functions from chromosome
        self.activations = nn.ModuleList()
        for gene in chromosome:
            self.activations.append(self._make_activation(gene))

    def _make_activation(self, gene):
        if gene == 0:
            return nn.Tanh()
        elif gene == 1:
            return Sine()
        elif gene == 2:
            return Swish()
        elif gene == 3:
            return nn.ReLU()
        elif gene == 4:
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown gene value: {gene}. Must be in {{0,1,2,3,4}}")

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)  # (N, 2)

        # Normalize to [-1, 1]
        inp = 2.0 * (inp - self.lb) / (self.ub - self.lb) - 1.0

        out = inp
        for i, linear in enumerate(self.linears):
            out = linear(out)
            if i < len(self.linears) - 1:   # no activation after last layer
                out = self.activations[i](out)
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

    def chromosome_str(self):
        """Human-readable chromosome representation."""
        return '[' + ', '.join(ACTIVATION_FUNCTIONS[g] for g in self.chromosome) + ']'


# ─────────────────────────────────────────────
# Custom activation functions
# ─────────────────────────────────────────────

class Sine(nn.Module):
    """Sine activation — good for periodic/wave physics."""
    def forward(self, x):
        return torch.sin(x)


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x) — smooth, non-monotonic."""
    def forward(self, x):
        return x * torch.sigmoid(x)