# bo_estimate.py
# Gaussian-Process Bayesian "estimator" for black-box function submissions
# Pre-populated with Srikar's Week-2 data and Week-3 submission inputs.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

@dataclass
class FunctionConfig:
    name: str
    dim: int
    X_train: np.ndarray  # shape (n, dim)
    y_train: np.ndarray  # shape (n,)
    X_submit: np.ndarray  # shape (m, dim), typically m=1
    repeat_prediction_n_times: bool = True
    decimals: int = 6
    enforce_start_with_zero: bool = False
    submission_scaler: str = "none"  # {"none","logistic","divide:K"}

def _build_gp(dim: int) -> GaussianProcessRegressor:
    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(dim), length_scale_bounds=(1e-3, 1e3), nu=2.5
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1))
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=42
    )

def _scale_for_submission(value: float, scaler: str) -> float:
    if scaler == "none":
        return value
    if scaler == "logistic":
        return 1.0 / (1.0 + np.exp(-value))
    if scaler.startswith("divide:"):
        denom = float(scaler.split(":")[1]); return value / denom
    raise ValueError(f"Unknown scaler: {scaler}")

def _format_submission_value(value: float, n_repeat: int, decimals: int) -> str:
    fmt = f"{{:.{decimals}f}}"
    return "-".join([fmt.format(value) for _ in range(n_repeat)])

def fit_predict(cfg: FunctionConfig) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.X_train.ndim != 2 or cfg.X_train.shape[1] != cfg.dim:
        raise ValueError(f"{cfg.name}: X_train must be (n, {cfg.dim})")
    if cfg.y_train.ndim != 1 or cfg.y_train.shape[0] != cfg.X_train.shape[0]:
        raise ValueError(f"{cfg.name}: y_train length must match X_train rows")
    if cfg.X_submit.ndim != 2 or cfg.X_submit.shape[1] != cfg.dim:
        raise ValueError(f"{cfg.name}: X_submit must be (m, {cfg.dim})")
    gp = _build_gp(cfg.dim)
    gp.fit(cfg.X_train, cfg.y_train)
    return gp.predict(cfg.X_submit, return_std=True)

essential_note = (
    "NOTE: If your portal enforces outputs to start with '0' (0.xxxxxx), set\n"
    "      enforce_start_with_zero=True and choose submission_scaler like\n"
    "      'divide:10000' or 'logistic'.\n"
)

def format_for_portal(cfg: FunctionConfig, scalar_pred: float) -> str:
    v = float(scalar_pred)
    if cfg.enforce_start_with_zero:
        v = _scale_for_submission(v, cfg.submission_scaler)
        if v < 0 and abs(v) < 1e-12: v = 0.0
        if v >= 1.0: v = _scale_for_submission(v, "logistic")
        if abs(v) < 0.5 * (10 ** -cfg.decimals): v = 0.0
    n_repeat = cfg.dim if cfg.repeat_prediction_n_times else 1
    return _format_submission_value(v, n_repeat, cfg.decimals)

if __name__ == "__main__":
    print("Running GP-based estimator…\n")

    # ========= Historical (Week‑2): ONE point per function (you can add more) =========
    F1_X_train = np.array([[0.719095, 0.749053]]);  F1_y_train = np.array([0.0])
    F2_X_train = np.array([[0.715737, 0.911443]]);  F2_y_train = np.array([0.6507203817813242])
    F3_X_train = np.array([[0.507140, 0.627060, 0.312795]]);  F3_y_train = np.array([-0.0667274655383587])
    F4_X_train = np.array([[0.550363, 0.435211, 0.441097, 0.234603]]);  F4_y_train = np.array([-3.8213133947055513])
    F5_X_train = np.array([[0.246223, 0.833921, 0.846444, 0.882309]]);  F5_y_train = np.array([918.0415049226765])
    F6_X_train = np.array([[0.712937, 0.180470, 0.707343, 0.707815, 0.068447]]);  F6_y_train = np.array([-0.6108776050648881])
    F7_X_train = np.array([[0.076166, 0.476789, 0.260377, 0.224116, 0.409357, 0.726687]]);  F7_y_train = np.array([1.5318732083108697])
    F8_X_train = np.array([[0.065255, 0.075394, 0.032198, 0.055903, 0.407636, 0.790016, 0.479884, 0.880923]])
    F8_y_train = np.array([9.6288236610841])

    # ========= New inputs (Week‑3 submission) =========
    F1_X_submit = np.array([[0.729095, 0.739053]])
    F2_X_submit = np.array([[0.735737, 0.931443]])
    F3_X_submit = np.array([[0.527140, 0.647060, 0.292795]])
    F4_X_submit = np.array([[0.570363, 0.415211, 0.451097, 0.234603]])
    F5_X_submit = np.array([[0.251223, 0.828921, 0.866444, 0.902309]])
    F6_X_submit = np.array([[0.732937, 0.200470, 0.727343, 0.687815, 0.088447]])
    F7_X_submit = np.array([[0.096166, 0.466789, 0.280377, 0.214116, 0.429357, 0.706687]])
    F8_X_submit = np.array([[0.075255, 0.085394, 0.042198, 0.065903, 0.417636, 0.770016, 0.469884, 0.860923]])

    # ========= Build configs (F5 scaled to 0.xxxxxx; toggle if your checker changes) =========
    functions = [
        FunctionConfig("Function 1", 2, F1_X_train, F1_y_train, F1_X_submit),
        FunctionConfig("Function 2", 2, F2_X_train, F2_y_train, F2_X_submit),
        FunctionConfig("Function 3", 3, F3_X_train, F3_y_train, F3_X_submit),
        FunctionConfig("Function 4", 4, F4_X_train, F4_y_train, F4_X_submit),
        FunctionConfig("Function 5", 4, F5_X_train, F5_y_train, F5_X_submit,
                       enforce_start_with_zero=True, submission_scaler="divide:10000"),
        FunctionConfig("Function 6", 5, F6_X_train, F6_y_train, F6_X_submit),
        FunctionConfig("Function 7", 6, F7_X_train, F7_y_train, F7_X_submit),
        FunctionConfig("Function 8", 8, F8_X_train, F8_y_train, F8_X_submit),
    ]

    results: Dict[str, Dict[str, str]] = {}
    for cfg in functions:
        mean, std = fit_predict(cfg)
        pred = float(mean[0])
        line = format_for_portal(cfg, pred)
        results[cfg.name] = {
            "prediction_line": line,
            "mean": f"{pred:.{cfg.decimals}f}",
            "std": f"{float(std[0]):.{cfg.decimals}f}",
        }

    print("=== PREDICTIONS (paste these lines into the portal) ===\n")
    for name, r in results.items():
        print(f"{name}: {r['prediction_line']}")

    print("\n--- Means & Std (for your records) ---")
    for name, r in results.items():
        print(f"{name}: mean={r['mean']}  std={r['std']}")

    print("\n" + essential_note)