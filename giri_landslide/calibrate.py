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
from typing import Dict, List, Tuple

import numpy as np

from . import config as C

FACTOR_NAMES = ["slope", "lithology", "vegetation", "soil_moisture"]


@dataclass
class CalibrationResult:
    weights: Dict[str, float]          # calibrated exponent weights (w_i)
    weights_raw: Dict[str, float]      # unnormalised logistic coefficients
    intercept: float
    auc: float                         # mean cross-validated ROC AUC
    auc_train: float
    auc_folds: List[float]             # per-fold held-out AUC (stability check)
    auc_std: float                     # spread across folds
    n_presence: int
    n_background: int
    factor_means_presence: Dict[str, float]
    factor_means_background: Dict[str, float]
    excluded_factors: List[str]        # uninformative (near-constant) factors
    warnings: List[str]                # data-quality caveats

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

def _fit_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 6000,
                  lr: float = 0.1, l2: float = 0.05,
                  seed: int = 0) -> Tuple[np.ndarray, float]:
    """Return (coefficients, intercept) via standardised gradient descent.

    Only columns with real variance are used; near-constant columns get a
    zero coefficient (they carry no information and would otherwise blow up in
    collinearity with the intercept). L2 regularisation keeps weights bounded.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    active = sd > 1e-6
    sd_safe = np.where(active, sd, 1.0)
    Xs = np.where(active, (X - mu) / sd_safe, 0.0)
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    pos = max(y.sum(), 1.0)
    neg = max((1 - y).sum(), 1.0)
    sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    for _ in range(epochs):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = (p - y) * sw
        w -= lr * (Xs.T @ g / n + l2 * w)
        b -= lr * g.mean()
    w = np.where(active, w, 0.0)
    w_raw = w / sd_safe
    b_raw = b - float((w * mu / sd_safe)[active].sum())
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
              n_folds: int = 5, normalise: bool = True,
              seed: int = 0) -> CalibrationResult:
    """Fit weights and report cross-validated AUC.

    K-fold cross-validation is used rather than a single hold-out split: with
    the small, clustered inventories typical of mountain regions a single split
    gives a noisy, sometimes badly optimistic score. The per-fold spread
    (``auc_std``) is reported so an unstable fit is visible.

    The deployed weights are fitted on *all* the data; the folds only estimate
    how well those weights generalise.
    """
    X, y = build_dataset(presence_feats, background_feats)
    if len(X) < 20:
        raise ValueError(f"too few valid calibration points ({len(X)})")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_folds = max(2, min(n_folds, int(y.sum()), int((1 - y).sum())))
    folds = np.array_split(idx, n_folds)

    fold_aucs: List[float] = []
    for k in range(n_folds):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        if y[train].sum() == 0 or (1 - y[train]).sum() == 0:
            continue
        w_k, _ = _fit_logistic(X[train], y[train], seed=seed)
        a = _auc(X[test] @ w_k, y[test])
        if np.isfinite(a):
            fold_aucs.append(float(a))

    # Deployed model: fitted on the full dataset.
    w_raw, b = _fit_logistic(X, y, seed=seed)
    auc_tr = _auc(X @ w_raw, y)
    auc_te = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    auc_sd = float(np.std(fold_aucs)) if fold_aucs else float("nan")

    # Screen factors that carry no information (near-constant, e.g. a missing
    # lithology layer left uniform) - they cannot be calibrated.
    sd = X.std(axis=0)
    excluded = [FACTOR_NAMES[i] for i in range(len(FACTOR_NAMES))
                if sd[i] <= 1e-6]
    warnings: List[str] = []
    if excluded:
        warnings.append(
            "uninformative (near-constant) factors excluded from calibration: "
            + ", ".join(excluded) + " - supply the underlying dataset "
            "(e.g. GLiM lithology) to calibrate them; their weight is held at "
            "the physical prior 1.0")

    p_mean = np.log(np.asarray(presence_feats, "float64") + 1.0)
    b_mean = np.log(np.asarray(background_feats, "float64") + 1.0)

    # Flag factors that are *negatively* associated with observed landslides -
    # usually a symptom of inventory reporting bias rather than physics.
    for i, k in enumerate(FACTOR_NAMES):
        if k not in excluded and w_raw[i] < -0.05:
            warnings.append(
                f"factor '{k}' is negatively associated with mapped landslides "
                f"(coef {w_raw[i]:.2f}); likely spatial reporting bias in the "
                f"inventory - weight clamped to 0")

    if np.isfinite(auc_te) and auc_te < 0.7:
        warnings.append(
            f"weak discrimination (CV AUC {auc_te:.2f}) - the calibrated "
            "weights are not trustworthy; use a larger/denser inventory or "
            "keep the default weights")
    if np.isfinite(auc_sd) and auc_sd > 0.1:
        warnings.append(
            f"unstable across folds (AUC sd {auc_sd:.2f}) - the inventory is "
            "probably too small or too spatially clustered")
    if int(y.sum()) < 100:
        warnings.append(
            f"only {int(y.sum())} landslide points - a few hundred well-spread "
            "points is the practical minimum for stable weights")

    # Deployment weights (exponent form must be non-negative). Excluded factors
    # keep the physical prior 1.0; fitted negatives are clamped to 0.
    w_dep = np.clip(w_raw, 0.0, None)
    for i, k in enumerate(FACTOR_NAMES):
        if k in excluded:
            w_dep[i] = 1.0
    if normalise and w_dep.sum() > 0:
        w_dep = w_dep * (len(w_dep) / w_dep.sum())

    return CalibrationResult(
        weights={k: float(v) for k, v in zip(FACTOR_NAMES, w_dep)},
        weights_raw={k: float(v) for k, v in zip(FACTOR_NAMES, w_raw)},
        intercept=float(b),
        auc=float(auc_te),
        auc_train=float(auc_tr),
        auc_folds=fold_aucs,
        auc_std=auc_sd,
        n_presence=int(y.sum()),
        n_background=int((1 - y).sum()),
        factor_means_presence={k: float(np.nanmean(p_mean[:, i]))
                               for i, k in enumerate(FACTOR_NAMES)},
        factor_means_background={k: float(np.nanmean(b_mean[:, i]))
                                 for i, k in enumerate(FACTOR_NAMES)},
        excluded_factors=excluded,
        warnings=warnings,
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
