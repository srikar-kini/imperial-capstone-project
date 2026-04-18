#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bbo_reference_surrogates.py
---------------------------------
Reference script for Black-Box Optimisation (BBO) capstone:
 • Train a surrogate (Gaussian Process or small Neural Network)
 • Score candidates and propose next inputs that comply with 0.xxxxxx-... format
 • Print quick diagnostics (UCB/mean for GP; gradients for NN)

USAGE (examples)
  # GP with demo data, 8-D, propose 2 inputs
  python bbo_reference_surrogates.py --model gp --dim 8 --k 2 --demo

  # NN with demo data (if torch available)
  python bbo_reference_surrogates.py --model nn --dim 8 --k 2 --demo

  # Load from CSV with columns x1..xN + y (and optional 'week')
  python bbo_reference_surrogates.py --data ./my_data.csv --x-cols x1,x2,x3,x4,x5,x6,x7,x8 --y-col y --dim 8 --model gp --k 1

DEPENDENCIES
  pip install numpy pandas scikit-learn
  pip install torch   # only if using --model nn
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

# Optional torch import (graceful fallback to GP if not installed)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C


# ----------------------------
# Utilities & formatting
# ----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed)


def hyphen_format(row: np.ndarray) -> str:
    """
    Render a 1D array in required 0.xxxxxx-... format (6 decimals).
    - Clips to [0, 1)
    - Ensures each token starts with '0'
    """
    row = np.clip(row, 0.0, np.nextafter(1.0, 0.0))  # [0, 1)
    toks = [f"{float(v):.6f}" for v in row]
    # Force tokens to start with '0' (e.g., '0.123456'); if negative (shouldn't happen), zero it.
    toks = [t if t.startswith("0") else ("0" + t) if not t.startswith("-") else "0.000000" for t in toks]
    return "-".join(toks)


def latin_hypercube(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.random(n)) / n
    return X


# ----------------------------
# Data loading
# ----------------------------
def load_demo(dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace these demo rows with your real historical data.
    Two points per function mimic your Week-1/2 style.
    """
    if dim == 2:
        X = np.array([[0.719095, 0.749053],
                      [0.735737, 0.931443]], dtype=np.float32)
        y = np.array([0.0, 0.389443], dtype=np.float32)
    elif dim == 3:
        X = np.array([[0.507140, 0.627060, 0.312795],
                      [0.527140, 0.647060, 0.292795]], dtype=np.float32)
        y = np.array([-0.066727, -0.071530], dtype=np.float32)
    elif dim == 4:
        X = np.array([[0.246223, 0.833921, 0.846444, 0.882309],
                      [0.251223, 0.828921, 0.866444, 0.902309]], dtype=np.float32)
        y = np.array([918.0415, 1076.3634], dtype=np.float32)
    elif dim == 5:
        X = np.array([[0.712937, 0.180470, 0.707343, 0.707815, 0.068447],
                      [0.732937, 0.200470, 0.727343, 0.687815, 0.088447]], dtype=np.float32)
        y = np.array([-0.610878, -0.684180], dtype=np.float32)
    elif dim == 6:
        X = np.array([[0.076166, 0.476789, 0.260377, 0.224116, 0.409357, 0.726687],
                      [0.096166, 0.466789, 0.280377, 0.214116, 0.429357, 0.706687]], dtype=np.float32)
        y = np.array([1.531873, 1.523771], dtype=np.float32)
    elif dim == 8:
        X = np.array([[0.065255, 0.075394, 0.032198, 0.055903, 0.407636, 0.790016, 0.479884, 0.880923],
                      [0.075255, 0.085394, 0.042198, 0.065903, 0.417636, 0.770016, 0.469884, 0.860923]], dtype=np.float32)
        y = np.array([9.628824, 9.665909], dtype=np.float32)
    else:
        raise ValueError("Unsupported demo dim; plug in your own data loader.")
    return X, y


def load_from_file(path: str, x_cols: List[str], y_col: str,
                   include_weeks: Optional[List[int]] = None,
                   exclude_weeks: Optional[List[int]] = None,
                   sheet: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV or Excel with column names for inputs and the target.
    Optionally filter by a 'week' column.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xls", ".xlsx"}:
        df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if 'week' in df.columns:
        if include_weeks:
            df = df[df['week'].isin(include_weeks)]
        if exclude_weeks:
            df = df[~df['week'].isin(exclude_weeks)]

    X = df[x_cols].to_numpy(dtype=np.float32)
    y = df[y_col].to_numpy(dtype=np.float32)
    return X, y


# ----------------------------
# GP surrogate + UCB
# ----------------------------
@dataclass
class GPConfig:
    kappa: float = 2.0         # UCB exploration weight
    n_restarts: int = 5
    seed: int = 42


def build_gp(dim: int, cfg: GPConfig) -> GaussianProcessRegressor:
    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(dim),
        length_scale_bounds=(1e-3, 1e3),
        nu=2.5
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=cfg.n_restarts,
        random_state=cfg.seed
    )
    return gp


def propose_with_gp(X: np.ndarray, y: np.ndarray, dim: int, k: int, cfg: GPConfig) -> Tuple[np.ndarray, str]:
    set_seed(cfg.seed)
    gp = build_gp(dim, cfg)
    gp.fit(X, y)

    # Candidate pool + UCB scoring
    C = latin_hypercube(2000, dim, seed=cfg.seed)
    mu, std = gp.predict(C, return_std=True)
    score = mu + cfg.kappa * std
    top_idx = np.argsort(score)[-k:]
    props = C[top_idx]

    # Diagnostics
    diag = f"GP: UCB kappa={cfg.kappa} | mean(range)≈[{mu.min():.3f},{mu.max():.3f}] " \
           f"| std(mean)≈{std.mean():.3f}"

    return props, diag


# ----------------------------
# NN surrogate (optional)
# ----------------------------
@dataclass
class NNConfig:
    hidden: int = 32
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 500
    batch_size: int = 16
    seed: int = 42
    step: float = 0.02     # gradient nudge step
    pool: int = 2000       # candidate pool
    topN: int = 50         # keep top-N before nudging


class TinyMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_nn(X: np.ndarray, y: np.ndarray, dim: int, cfg: NNConfig):
    set_seed(cfg.seed)
    y = y.reshape(-1, 1)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y).astype(np.float32)

    Xt = torch.tensor(X.astype(np.float32))
    yt = torch.tensor(y_scaled)

    model = TinyMLP(dim, cfg.hidden)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    return model, y_scaler


def predict_with_grads(model: nn.Module, X: np.ndarray, y_scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray]:
    Xt = torch.tensor(X.astype(np.float32), requires_grad=True)
    out = model(Xt)  # scaled space
    y_pred = y_scaler.inverse_transform(out.detach().numpy())

    grads = []
    for i in range(len(Xt)):
        model.zero_grad()
        if Xt.grad is not None:
            Xt.grad.zero_()
        out[i].backward(retain_graph=True)
        grads.append(Xt.grad[i].detach().clone().numpy())
    grads = np.stack(grads)  # ∂(scaled y)/∂x
    return y_pred.flatten(), grads


def propose_with_nn(X: np.ndarray, y: np.ndarray, dim: int, k: int, cfg: NNConfig) -> Tuple[np.ndarray, str]:
    if not TORCH_OK:
        raise RuntimeError("PyTorch not installed; cannot use --model nn")

    model, y_scaler = train_nn(X, y, dim, cfg)

    # Candidate scoring
    C = latin_hypercube(cfg.pool, dim, seed=cfg.seed)
    preds0, _ = predict_with_grads(model, C, y_scaler)
    top_idx = np.argsort(preds0)[-cfg.topN:]
    top = C[top_idx]

    # Gradient nudge
    _, grads = predict_with_grads(model, top, y_scaler)
    nudged = np.clip(top + cfg.step * np.sign(grads), 0.0, np.nextafter(1.0, 0.0))

    preds1, grads1 = predict_with_grads(model, nudged, y_scaler)
    sel = np.argsort(preds1)[-k:]
    props = nudged[sel]

    # Diagnostics
    grad_mag = np.abs(grads1[sel]).sum(axis=1).mean()
    diag = f"NN: hidden={cfg.hidden}, lr={cfg.lr}, step={cfg.step} | mean(pred)≈{preds1[sel].mean():.3f} | avg|grad|≈{grad_mag:.3f}"

    return props, diag


# ----------------------------
# Main CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="BBO surrogate (GP/NN) + proposal generator.")
    p.add_argument("--dim", type=int, required=True, help="Problem dimension (e.g., 2,3,4,5,6,8)")
    p.add_argument("--model", type=str, default="gp", choices=["gp", "nn"], help="Surrogate model")
    p.add_argument("--k", type=int, default=1, help="How many proposals to output")
    p.add_argument("--seed", type=int, default=42)

    # Data sources
    p.add_argument("--demo", action="store_true", help="Use built-in tiny demo dataset")
    p.add_argument("--data", type=str, help="Path to CSV/Excel with historical data")
    p.add_argument("--sheet", type=str, default=None, help="Excel sheet name (optional)")
    p.add_argument("--x-cols", type=str, default=None, help="Comma-separated input column names (if using --data)")
    p.add_argument("--y-col", type=str, default=None, help="Target column name (if using --data)")
    p.add_argument("--include-weeks", type=str, default=None, help="Comma-separated week numbers to keep (if 'week' column exists)")
    p.add_argument("--exclude-weeks", type=str, default=None, help="Comma-separated week numbers to drop (e.g., 3)")

    # GP knobs
    p.add_argument("--gp-kappa", type=float, default=2.0, help="UCB exploration weight")

    # NN knobs
    p.add_argument("--nn-hidden", type=int, default=32)
    p.add_argument("--nn-lr", type=float, default=3e-3)
    p.add_argument("--nn-wd", type=float, default=1e-4)
    p.add_argument("--nn-epochs", type=int, default=500)
    p.add_argument("--nn-batch", type=int, default=16)
    p.add_argument("--nn-step", type=float, default=0.02)
    p.add_argument("--nn-pool", type=int, default=2000)
    p.add_argument("--nn-topN", type=int, default=50)

    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    # Load data
    if args.demo:
        X, y = load_demo(args.dim)
    else:
        if not (args.data and args.x_cols and args.y_col):
            raise SystemExit("When not using --demo, please provide --data, --x-cols, --y-col")
        x_cols = [c.strip() for c in args.x_cols.split(",")]
        include_weeks = [int(w) for w in args.include_weeks.split(",")] if args.include_weeks else None
        exclude_weeks = [int(w) for w in args.exclude_weeks.split(",")] if args.exclude_weeks else None
        X, y = load_from_file(args.data, x_cols, args.y_col, include_weeks, exclude_weeks, args.sheet)

    # Train & propose
    if args.model == "gp":
        gp_cfg = GPConfig(kappa=args.gp_kappa, seed=args.seed)
        props, diag = propose_with_gp(X, y, dim=args.dim, k=args.k, cfg=gp_cfg)
    else:
        if not TORCH_OK:
            raise SystemExit("PyTorch not installed; cannot use --model nn")
        nn_cfg = NNConfig(hidden=args.nn_hidden, lr=args.nn_lr, weight_decay=args.nn_wd,
                          epochs=args.nn_epochs, batch_size=args.nn_batch,
                          step=args.nn_step, pool=args.nn_pool, topN=args.nn_topN, seed=args.seed)
        props, diag = propose_with_nn(X, y, dim=args.dim, k=args.k, cfg=nn_cfg)

    # Output proposals in the EXACT portal format
    print("\n# Proposed next input(s) (paste into the portal):")
    for i, p in enumerate(props, 1):
        print(f"{i}. {hyphen_format(p)}")

    # Quick diagnostics
    print(f"\n# Surrogate summary: {diag}\n")


if __name__ == "__main__":
    main()