"""Feature construction, shared by calibration and prediction.

The susceptibility index is a logistic model, so it is only as good as the
predictors fed to it. Two feature sets are available.

``ordinal``
    The manuscript's formulation: each input is reclassified to a small integer
    score (slope 0-5, lithology 0-3, land cover 0-5, soil moisture 1-5) and the
    model is fitted on ``log(score + 1)``. Faithful to the published method, but
    the index inherits the coarseness of its inputs: four integer scores admit
    only a few hundred distinct products, so large groups of pixels tie and the
    resulting map cannot be split into evenly sized bins.

``continuous``
    Slope and precipitation enter at full precision instead of as classes.
    Slope is represented by a quadratic, since its relationship with failure is
    not monotonic - susceptibility rises to a maximum near 30-36 degrees and
    falls above it, where slopes exceed the internal friction angle of most
    soils and have already shed their regolith. A linear term alone cannot
    express that shape; a quadratic can, and lets the data place the peak.
    Lithology and land cover remain categorical, because they are.

Both sets are defined here rather than in the calibration or the prediction
code, so the two cannot drift apart. A model fitted on one feature set and
evaluated on another would produce a plausible-looking and entirely wrong map.
"""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple

import numpy as np

from .factors import FACTOR_NODATA


class Feature(NamedTuple):
    """One model predictor: which raster it reads and how it is transformed."""

    name: str
    source: str                       # key into the raster path mapping
    transform: Callable[[np.ndarray], np.ndarray]
    nodata: float                     # value marking nodata in that raster


def _log1p(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 0.0, None) + 1.0)


def _slope_scaled(a: np.ndarray) -> np.ndarray:
    """Slope in degrees, scaled to roughly unit range for conditioning."""
    return np.clip(a, 0.0, 90.0) / 45.0


def _slope_scaled_sq(a: np.ndarray) -> np.ndarray:
    return _slope_scaled(a) ** 2


ORDINAL: List[Feature] = [
    Feature("slope", "slope", _log1p, FACTOR_NODATA),
    Feature("lithology", "litho", _log1p, FACTOR_NODATA),
    Feature("vegetation", "veg", _log1p, FACTOR_NODATA),
    Feature("soil_moisture", "soil", _log1p, FACTOR_NODATA),
]

CONTINUOUS: List[Feature] = [
    Feature("slope", "slope_deg", _slope_scaled, -9999.0),
    Feature("slope_sq", "slope_deg", _slope_scaled_sq, -9999.0),
    Feature("lithology", "litho", _log1p, FACTOR_NODATA),
    Feature("vegetation", "veg", _log1p, FACTOR_NODATA),
    Feature("soil_moisture", "soil_raw", _log1p, -9999.0),
]

SETS: Dict[str, List[Feature]] = {"ordinal": ORDINAL, "continuous": CONTINUOUS}


def get(feature_mode: str) -> List[Feature]:
    if feature_mode not in SETS:
        raise ValueError(f"unknown feature_mode {feature_mode!r}; "
                         f"expected one of {sorted(SETS)}")
    return SETS[feature_mode]


def names(feature_mode: str) -> List[str]:
    return [f.name for f in get(feature_mode)]


def paths(feature_mode: str, raster_paths: Dict[str, str]) -> List[str]:
    """Raster path per feature, in model order."""
    feats = get(feature_mode)
    missing = sorted({f.source for f in feats} - set(raster_paths))
    if missing:
        raise KeyError(
            f"feature_mode {feature_mode!r} needs raster(s) {missing}; "
            "re-run the susceptibility stage so they are written")
    return [raster_paths[f.source] for f in feats]


def design_matrix(samples: np.ndarray, feature_mode: str) -> np.ndarray:
    """Apply each feature's transform to its column of sampled raster values.

    ``samples`` has one column per feature, in the order given by :func:`get`,
    holding the raw values read from that feature's raster.
    """
    feats = get(feature_mode)
    cols = []
    for i, f in enumerate(feats):
        col = np.asarray(samples[:, i], dtype="float64")
        col = np.where(col == f.nodata, np.nan, col)
        cols.append(f.transform(col))
    return np.column_stack(cols)


def linear_predictor(arrays: List[np.ndarray], weights: List[float],
                     feature_mode: str) -> np.ndarray:
    """Weighted sum of transformed features, for raster blocks.

    ``arrays`` holds one block per feature, in model order.
    """
    feats = get(feature_mode)
    z = np.zeros(arrays[0].shape, dtype="float64")
    for f, w, a in zip(feats, weights, arrays):
        a = np.where(a == f.nodata, np.nan, a)
        z += w * np.nan_to_num(f.transform(a), nan=0.0)
    return z


def nodata_mask(arrays: List[np.ndarray], feature_mode: str) -> np.ndarray:
    """True where any feature is missing for that pixel."""
    feats = get(feature_mode)
    masks = [(a == f.nodata) | np.isnan(a) for f, a in zip(feats, arrays)]
    return np.any(masks, axis=0)
