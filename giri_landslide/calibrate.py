"""Data-driven calibration of the susceptibility factor weights.

The heuristic susceptibility index is combined in exponent form,

    log S = sum_i w_i * log(f_i + 1),

so the factor influences ``w_i`` are exactly the coefficients of a logistic
regression of landslide *presence* on the log-transformed factor scores. We fit
that regression on a historical inventory (presence) plus background points
(pseudo-absence), which yields calibrated, region-specific weights and a
discrimination score (ROC AUC).

Only NumPy is used - the logistic model is fitted with gradient descent, so
there is no scikit-learn dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C

FACTOR_NAMES = ["slope", "lithology", "vegetation", "soil_moisture"]


@dataclass
class CalibrationResult:
    weights: Dict[str, float]          # calibrated exponent weights (w_i)
    weights_raw: Dict[str, float]      # unnormalised logistic coefficients
    intercept: float
    auc: float                         # held-out ROC AUC
    auc_train: float
    n_presence: int
    n_background: int
    factor_means_presence: Dict[str, float]
    factor_means_background: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_dataset(presence_feats: np.ndarray,
                  background_feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Stack presence/background factor samples into (X=log(f+1), y)."""
    def clean(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype="float64")
        return a[np.isfinite(a).all(axis=1)]

    p = clean(presence_feats)
    b = clean(background_feats)
    X = np.log(np.vstack([p, b]) + 1.0)
    y = np.concatenate([np.ones(len(p)), np.zeros(len(b))])
    return X, y


# ---------------------------------------------------------------------------
# Logistic regression (numpy)
# ---------------------------------------------------------------------------

def _fit_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 4000,
                  lr: float = 0.1, l2: float = 1e-3,
                  seed: int = 0) -> Tuple[np.ndarray, float]:
    """Return (coefficients, intercept) via standardised gradient descent."""
    rng = np.random.default_rng(seed)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    # class balancing weights
    pos = max(y.sum(), 1.0)
    neg = max((1 - y).sum(), 1.0)
    sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    for _ in range(epochs):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = (p - y) * sw
        gw = Xs.T @ g / n + l2 * w
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    # de-standardise back to raw-feature space
    w_raw = w / sd
    b_raw = b - float((w * mu / sd).sum())
    return w_raw, b_raw


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype="float64")
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Top-level calibration
# ---------------------------------------------------------------------------

def calibrate(presence_feats: np.ndarray, background_feats: np.ndarray,
              test_fraction: float = 0.3, normalise: bool = True,
              seed: int = 0) -> CalibrationResult:
    """Fit weights and report held-out AUC."""
    X, y = build_dataset(presence_feats, background_feats)
    if len(X) < 20:
        raise ValueError(f"too few valid calibration points ({len(X)})")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(4, int(len(X) * test_fraction))
    test, train = idx[:n_test], idx[n_test:]

    w_raw, b = _fit_logistic(X[train], y[train], seed=seed)
    auc_tr = _auc(X[train] @ w_raw, y[train])
    auc_te = _auc(X[test] @ w_raw, y[test])

    # For the deployed index we want positive influences; clamp tiny negatives
    # to 0 and scale so the mean weight is 1 (scale is irrelevant to ranking).
    w_dep = np.clip(w_raw, 0.0, None)
    if normalise and w_dep.sum() > 0:
        w_dep = w_dep * (len(w_dep) / w_dep.sum())

    p_mean = np.log(np.asarray(presence_feats, "float64") + 1.0)
    b_mean = np.log(np.asarray(background_feats, "float64") + 1.0)

    return CalibrationResult(
        weights={k: float(v) for k, v in zip(FACTOR_NAMES, w_dep)},
        weights_raw={k: float(v) for k, v in zip(FACTOR_NAMES, w_raw)},
        intercept=float(b),
        auc=float(auc_te),
        auc_train=float(auc_tr),
        n_presence=int(y.sum()),
        n_background=int((1 - y).sum()),
        factor_means_presence={k: float(np.nanmean(p_mean[:, i]))
                               for i, k in enumerate(FACTOR_NAMES)},
        factor_means_background={k: float(np.nanmean(b_mean[:, i]))
                                 for i, k in enumerate(FACTOR_NAMES)},
    )


def apply_to_config(cfg: C.Config, result: CalibrationResult) -> C.Config:
    """Return a copy of ``cfg`` using the calibrated exponent weights."""
    import copy

    c = copy.deepcopy(cfg)
    c.weights = C.Weights(
        slope=result.weights["slope"],
        lithology=result.weights["lithology"],
        vegetation=result.weights["vegetation"],
        soil_moisture=result.weights["soil_moisture"],
    )
    c.weight_mode = "exponent"
    c.classification = "quantile"
    return c
