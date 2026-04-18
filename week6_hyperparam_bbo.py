#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
week6_hyperparam_bbo.py

Reference script for Week 6 of the BBO capstone:
  • Uses Weeks 1–5 function data as a tiny training set
  • Excludes Week 3 as an outlier (extreme inputs/outputs)
  • Tunes a simple hyperparameter (Ridge alpha) via small grid search
  • Fits a Ridge model per function F1–F8
  • Predicts outputs for the Week 6 query inputs (already accepted by the portal)

This script is meant as a *reference* for:
  - basic hyperparameter tuning
  - simple surrogate modelling
  - documenting your reasoning in GitHub.
"""

from __future__ import annotations

import itertools
import numpy as np
from sklearn.linear_model import Ridge


# --------------------------
# 1. Training data (Weeks 1–5)
#    Taken from input and outout values weekly iteration.xlsx
#    Week order: [W1, W2, W3, W4, W5]
# --------------------------

# Function 1 (2D)
F1_X = np.array([
    [0.719095, 0.749053],  # W1
    [0.729095, 0.739053],  # W2
    [0.000000, 0.000000],  # W3 (outlier)
    [0.730000, 0.740000],  # W4
    [0.720000, 0.748000],  # W5
])
F1_y = np.array([
    -3.10400768246467e-17,   # W1
     1.1253443948613001e-16, # W2
     2.3088107421261399e-248,# W3 (outlier)
     5.0066003689023798e-17, # W4
    -2.14141970294188e-17,   # W5
])

# Function 2 (2D)
F2_X = np.array([
    [0.715737, 0.911443],  # W1
    [0.735737, 0.931443],  # W2
    [0.540000, 0.540000],  # W3 (valid but poorer region)
    [0.560000, 0.620000],  # W4
    [0.705737, 0.891443],  # W5
])
F2_y = np.array([
    0.650720381781324,   # W1
    0.38944343460576403, # W2
    0.29636006576788798, # W3
    0.182980504445761,   # W4
    0.54745542290314497, # W5
])

# Function 3 (3D)
F3_X = np.array([
    [0.507140, 0.627060, 0.312795],  # W1
    [0.527140, 0.647060, 0.292795],  # W2
    [0.056000, 0.056000, 0.056000],  # W3
    [0.120000, 0.180000, 0.140000],  # W4
    [0.497140, 0.617060, 0.322795],  # W5
])
F3_y = np.array([
    -0.0667274655383587,  # W1
    -0.07152992226253835, # W2
    -0.103100417089051,   # W3
    -0.12289925212224199, # W4
    -0.054712887218655,   # W5
])

# Function 4 (4D)
F4_X = np.array([
    [0.550363, 0.435211, 0.441097, 0.234603],  # W1
    [0.570363, 0.415211, 0.451097, 0.234603],  # W2
    [3.650000, 3.650000, 3.650000, 3.650000],  # W3 (invalid input)
    [0.080000, 0.120000, 0.160000, 0.200000],  # W4
    [0.540363, 0.445211, 0.431097, 0.234603],  # W5
])
F4_y = np.array([
    -3.82131339470555,        # W1
    -4.6106611390395296,      # W2
    -27299.486542392198,      # W3 (extreme outlier)
    -13.257309922431601,      # W4
    -3.5598988632934701,      # W5
])

# Function 5 (4D)
F5_X = np.array([
    [0.246223, 0.833921, 0.846444, 0.882309],  # W1
    [0.251223, 0.828921, 0.866444, 0.902309],  # W2
    [0.123000, 0.123000, 0.123000, 0.123000],  # W3 (poor region)
    [0.260000, 0.860000, 0.880000, 0.910000],  # W4
    [0.255000, 0.824000, 0.886000, 0.922000],  # W5
])
F5_y = np.array([
    918.04150492267604, # W1
    1076.36345165896,   # W2
    161.62476488340599, # W3
    1319.9549883617001, # W4
    1257.44006595833,   # W5
])

# Function 6 (5D)
F6_X = np.array([
    [0.712937, 0.180470, 0.707343, 0.707815, 0.068447],  # W1
    [0.732937, 0.200470, 0.727343, 0.687815, 0.088447],  # W2
    [0.660000, 0.660000, 0.660000, 0.660000, 0.660000],  # W3 (problematic)
    [0.350000, 0.260000, 0.760000, 0.720000, 0.140000],  # W4
    [0.702937, 0.170470, 0.697343, 0.717815, 0.058447],  # W5
])
F6_y = np.array([
    -0.610877605064888,  # W1
    -0.684180393738355,  # W2
    -1.2403699165622499, # W3
    -0.370202477594745,  # W4
    -0.584809896135903,  # W5
])

# Function 7 (6D)
F7_X = np.array([
    [0.076166, 0.476789, 0.260377, 0.224116, 0.409357, 0.726687],  # W1
    [0.096166, 0.466789, 0.280377, 0.214116, 0.429357, 0.706687],  # W2
    [1.530000, 1.530000, 1.530000, 1.530000, 1.530000, 1.530000],  # W3 (invalid input)
    [0.220000, 0.520000, 0.300000, 0.260000, 0.460000, 0.740000],  # W4
    [0.080000, 0.472000, 0.264000, 0.220000, 0.415000, 0.730000],  # W5
])
F7_y = np.array([
    1.53187320831086,       # W1
    1.5237709478196,        # W2
    1.3992220078479999e-18, # W3
    1.2609383091402999,     # W4
    1.5174571796580201,     # W5
])

# Function 8 (8D)
F8_X = np.array([
    [0.065255, 0.075394, 0.032198, 0.055903, 0.407636, 0.790016, 0.479884, 0.880923],  # W1
    [0.075255, 0.085394, 0.042198, 0.065903, 0.417636, 0.770016, 0.469884, 0.860923],  # W2
    [9.700000, 9.700000, 9.700000, 9.700000, 9.700000, 9.700000, 9.700000, 9.700000],  # W3 (invalid)
    [0.090000, 0.120000, 0.070000, 0.100000, 0.450000, 0.820000, 0.520000, 0.900000],  # W4
    [0.085255, 0.095394, 0.052198, 0.075903, 0.427636, 0.760016, 0.459884, 0.850923],  # W5
])
F8_y = np.array([
    9.6288236610841,     # W1
    9.6659089730841004,  # W2
   -944.5057,            # W3 (extreme outlier)
    9.60815,             # W4
    9.6946221190841,     # W5
])


# --------------------------
# 2. Week 6 query inputs (already accepted by the portal)
#    All values in [0,1), formatted as 0.xxxxxx.
# --------------------------

F1_week6 = np.array([[0.725000, 0.743000]])
F2_week6 = np.array([[0.710000, 0.900000]])
F3_week6 = np.array([[0.495000, 0.620000, 0.320000]])
F4_week6 = np.array([[0.538000, 0.448000, 0.428000, 0.234603]])
F5_week6 = np.array([[0.258000, 0.862000, 0.884000, 0.918000]])
F6_week6 = np.array([[0.353000, 0.263000, 0.763000, 0.720000, 0.138000]])
F7_week6 = np.array([[0.078000, 0.478000, 0.262000, 0.226000, 0.411000, 0.728000]])
F8_week6 = np.array([[0.086000, 0.096000, 0.053000, 0.077000, 0.429000, 0.758000, 0.457000, 0.848000]])


# --------------------------
# 3. Helper: simple alpha grid search with leave-one-out
# --------------------------

def tune_ridge_alpha(X: np.ndarray, y: np.ndarray, alphas=(0.01, 0.1, 1.0, 10.0)) -> float:
    """
    Very small hyperparameter tuner for Ridge alpha.
    Uses leave-one-out cross-validation on the provided (X, y).

    NOTE: This is illustrative only; with 4 points, scores are noisy.
    """
    n = len(X)
    best_alpha = alphas[0]
    best_score = float("inf")

    for alpha in alphas:
        errors = []
        for i in range(n):
            # leave one out
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_tr, y_tr = X[mask], y[mask]
            X_val, y_val = X[~mask], y[~mask]

            model = Ridge(alpha=alpha)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            errors.append((y_val - y_pred) ** 2)

        mse = float(np.mean(errors))
        if mse < best_score:
            best_score = mse
            best_alpha = alpha

    return best_alpha


def portal_format_scalar(value: float, dim: int) -> str:
    """
    Convert a scalar prediction into a hyphen-separated string
    repeating the value 'dim' times. This is only for logging
    predictions in the same style as the portal (not for submission).
    Values here are not restricted to [0,1), unlike query inputs.
    """
    return "-".join(f"{value:.6f}" for _ in range(dim))


# --------------------------
# 4. Fit models, tune alpha, predict Week 6
# --------------------------

def main():
    functions = [
        ("Function 1", F1_X, F1_y, F1_week6, 2, True),
        ("Function 2", F2_X, F2_y, F2_week6, 2, False),
        ("Function 3", F3_X, F3_y, F3_week6, 3, False),
        ("Function 4", F4_X, F4_y, F4_week6, 4, True),
        ("Function 5", F5_X, F5_y, F5_week6, 4, False),
        ("Function 6", F6_X, F6_y, F6_week6, 5, True),
        ("Function 7", F7_X, F7_y, F7_week6, 6, True),
        ("Function 8", F8_X, F8_y, F8_week6, 8, True),
    ]

    print("\n=== Week 6 Ridge-based surrogate predictions with alpha tuning ===\n")

    for name, X, y, X6, dim, drop_idx2 in functions:
        # Decide which rows to keep (drop Week 3 where inputs are clearly invalid)
        if drop_idx2:
            mask = np.ones(len(X), dtype=bool)
            mask[2] = False
            X_train, y_train = X[mask], y[mask]
        else:
            X_train, y_train = X, y

        # Tune alpha on this small dataset
        alpha = tune_ridge_alpha(X_train, y_train)
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        y_pred = float(model.predict(X6)[0])

        print(f"{name}:")
        print(f"  chosen alpha       = {alpha}")
        print(f"  predicted y (W6)   = {y_pred:.6f}")
        print(f"  portal-style value = {portal_format_scalar(y_pred, dim)}\n")


if __name__ == "__main__":
    main()