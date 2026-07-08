import numpy as np
import scipy.io
import torch
from scipy.stats import qmc


def load_burgers_data(data_path, N_u=100, N_f=10000, device='cpu'):

    data = scipy.io.loadmat(data_path)

    t = data['t'].flatten()[:, None]    # (100, 1)
    x = data['x'].flatten()[:, None]    # (256, 1)
    Exact = np.real(data['usol']).T     # (100, 256)

    X, T = np.meshgrid(x, t)
    X_star = np.hstack([X.flatten()[:, None],
                        T.flatten()[:, None]])   # (25600, 2)
    u_star = Exact.flatten()[:, None]            # (25600, 1)

    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0,  1.0])

    xx1 = np.hstack([x, np.zeros_like(x)])
    uu1 = -np.sin(np.pi * x)

    xx2 = np.hstack([-np.ones_like(t), t])
    uu2 = np.zeros_like(t)

    xx3 = np.hstack([np.ones_like(t), t])
    uu3 = np.zeros_like(t)

    X_u_all = np.vstack([xx1, xx2, xx3])
    u_all   = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(X_u_all.shape[0], N_u, replace=False)
    X_u_train = X_u_all[idx, :]
    u_train   = u_all[idx, :]

    sampler = qmc.LatinHypercube(d=2, seed=1234)
    X_f_train = lb + (ub - lb) * sampler.random(n=N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))

    X_u_train = torch.tensor(X_u_train, dtype=torch.float64).to(device)
    u_train   = torch.tensor(u_train,   dtype=torch.float64).to(device)
    X_f_train = torch.tensor(X_f_train, dtype=torch.float64).to(device)
    X_star    = torch.tensor(X_star,    dtype=torch.float64).to(device)
    u_star    = torch.tensor(u_star,    dtype=torch.float64).to(device)

    return X_u_train, u_train, X_f_train, X_star, u_star, lb, ub