"""Cross-validation splits for within-region model assessment.

Two ways to split a single region's landslides into training and test sets, and
they do not measure the same thing.

``random``
    Points are assigned to folds independently. Landslides are spatially
    clustered and terrain is autocorrelated, so a test point usually has
    training points a few hundred metres away on the same hillside. The model
    is effectively asked to interpolate between neighbours, and the resulting
    score is optimistic as an estimate of performance on new ground.

``spatial``
    The area is divided into square blocks and whole blocks are assigned to
    folds, so every test point sits in a block containing no training data.
    This measures what is usually wanted: performance where the model has not
    seen the local terrain. Blocks must be large relative to the
    autocorrelation range, or the scheme degenerates towards the random one.

The gap between the two is itself the quantity of interest. A large gap means
the apparent skill is largely spatial interpolation rather than a transferable
relationship between terrain and failure.

These functions only assign folds. The fitting and scoring loop lives with the
model being assessed, in :func:`giri_landslide.model.physical.cross_validate`.
"""

from __future__ import annotations

from typing import Sequence

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
