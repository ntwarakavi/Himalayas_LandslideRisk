"""Cross-validation schemes for within-region model assessment.

Two ways to split a single region's landslides into training and test sets, and
they do not measure the same thing.

``random``
    Points are assigned to folds independently. This is what the default
    calibration does. Landslides are spatially clustered and terrain is
    autocorrelated, so a test point usually has training points a few hundred
    metres away on the same hillside. The model is effectively asked to
    interpolate between neighbours, and the resulting score is optimistic as an
    estimate of performance on new ground.

``spatial``
    The area is divided into square blocks and whole blocks are assigned to
    folds, so every test point sits in a block containing no training data. This
    measures what is usually wanted: performance where the model has not seen
    the local terrain. Blocks must be large relative to the autocorrelation
    range, or the scheme degenerates towards the random one.

The gap between the two is itself the quantity of interest. A large gap means
the apparent skill is largely spatial interpolation rather than a transferable
relationship between terrain and failure.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def random_folds(n: int, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index per sample, assigned independently."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_folds, n)


def spatial_block_folds(points: np.ndarray, bbox: Sequence[float],
                        n_folds: int = 5, block_deg: float = 0.25,
                        seed: int = 0) -> np.ndarray:
    """Fold index per point, assigned by spatial block.

    The bounding box is tiled into ``block_deg`` squares and each block is given
    to one fold at random, so a fold's test points are geographically separated
    from its training points.

    ``block_deg`` should exceed the range over which terrain and landslide
    density are correlated. At 0.25 degrees (~25 km) a block is much larger than
    a hillslope, so neighbouring landslides fall in the same block and cannot be
    split across the train/test boundary.
    """
    w, s, e, n = bbox
    rng = np.random.default_rng(seed)

    col = np.floor((points[:, 0] - w) / block_deg).astype(int)
    row = np.floor((points[:, 1] - s) / block_deg).astype(int)
    n_col = max(int(np.ceil((e - w) / block_deg)), 1)
    block_id = row * n_col + col

    unique = np.unique(block_id)
    assignment = rng.permutation(len(unique)) % n_folds
    lookup = dict(zip(unique.tolist(), assignment.tolist()))
    return np.array([lookup[b] for b in block_id])


def cross_validate(presence: np.ndarray, background: np.ndarray,
                   presence_feats: np.ndarray, background_feats: np.ndarray,
                   feature_mode: str, bbox: Sequence[float],
                   scheme: str = "spatial", n_folds: int = 5,
                   block_deg: float = 0.25, seed: int = 0) -> dict:
    """Fit and score fold by fold under the chosen split.

    Presence and background points are split by the same rule, so a spatial fold
    withholds a whole area rather than only its landslides.
    """
    from . import calibrate as CAL
    from . import features as FT

    if scheme == "spatial":
        fp = spatial_block_folds(presence, bbox, n_folds, block_deg, seed)
        fb = spatial_block_folds(background, bbox, n_folds, block_deg, seed)
    elif scheme == "random":
        fp = random_folds(len(presence), n_folds, seed)
        fb = random_folds(len(background), n_folds, seed)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    Xp = FT.design_matrix(np.asarray(presence_feats, "float64"), feature_mode)
    Xb = FT.design_matrix(np.asarray(background_feats, "float64"), feature_mode)

    aucs: List[float] = []
    sizes: List[Tuple[int, int]] = []
    for k in range(n_folds):
        tr_X = np.vstack([Xp[fp != k], Xb[fb != k]])
        tr_y = np.concatenate([np.ones((fp != k).sum()),
                               np.zeros((fb != k).sum())])
        te_X = np.vstack([Xp[fp == k], Xb[fb == k]])
        te_y = np.concatenate([np.ones((fp == k).sum()),
                               np.zeros((fb == k).sum())])

        ok_tr = np.isfinite(tr_X).all(axis=1)
        ok_te = np.isfinite(te_X).all(axis=1)
        tr_X, tr_y = tr_X[ok_tr], tr_y[ok_tr]
        te_X, te_y = te_X[ok_te], te_y[ok_te]
        if tr_y.sum() < 10 or (1 - tr_y).sum() < 10 or te_y.sum() < 5 \
                or (1 - te_y).sum() < 5:
            continue

        w, _ = CAL._fit_logistic(tr_X, tr_y, seed=seed)
        auc = CAL._auc(te_X @ w, te_y)
        if np.isfinite(auc):
            aucs.append(float(auc))
            sizes.append((int(te_y.sum()), int((1 - te_y).sum())))

    return {
        "scheme": scheme,
        "n_folds_scored": len(aucs),
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "auc_folds": [round(a, 4) for a in aucs],
        "test_sizes": sizes,
        "block_deg": block_deg if scheme == "spatial" else None,
    }
