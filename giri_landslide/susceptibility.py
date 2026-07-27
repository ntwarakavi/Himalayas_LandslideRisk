"""Step 3.2 - combine susceptibility factors into the 5-class susceptibility map.

    S = product_i ( w_i * f(S_i) )                      (Eq. 1)

then S is reclassified into categories 1..5 (Very Low .. Very High).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import config as C
from .factors import FACTOR_NODATA
from .grid import combine_rasters, map_raster, reclassify_continuous

SUSC_NODATA = 255


def combine_factors(slope_f: str, litho_f: str, veg_f: str, soil_f: str,
                    out_index_path: str, weights: C.Weights,
                    mode: str = "multiplicative", block: int = 1024) -> str:
    """Compute the susceptibility index S (float raster).

    Two combination modes:

    * ``"multiplicative"`` - S = prod_i (w_i * f_i)  (as in the manuscript).
      Note the weights only rescale S globally; they do not change the relative
      ranking of pixels, so they matter only through the class breaks.
    * ``"exponent"`` - S = prod_i (f_i + 1) ** w_i, i.e.
      log S = sum_i w_i * log(f_i + 1). Here the weights are factor
      *influences* and DO change the ranking, which is what makes data-driven
      weight calibration meaningful.

    In both modes the physical hard constraints are preserved: flat terrain
    (slope factor 0) and open water (vegetation factor 0) give S = 0.
    """
    w = [weights.slope, weights.lithology, weights.vegetation,
         weights.soil_moisture]

    def fn(arrs: List[np.ndarray]) -> np.ndarray:
        slope, litho, veg, soil = arrs
        masks = [(a == FACTOR_NODATA) | np.isnan(a) for a in arrs]
        any_nodata = np.any(masks, axis=0)
        clean = [np.where(np.isnan(a), 0.0, a) for a in arrs]
        if mode == "exponent":
            logS = np.zeros(clean[0].shape, dtype="float64")
            for wi, a in zip(w, clean):
                logS += wi * np.log(a + 1.0)
            S = np.exp(logS)
        else:
            S = np.ones(clean[0].shape, dtype="float64")
            for wi, a in zip(w, clean):
                S *= wi * a
        # Hard constraints: flat terrain or open water -> not susceptible.
        S = np.where((clean[0] == 0) | (clean[2] == 0), 0.0, S)
        return np.where(any_nodata, np.nan, S)

    return combine_rasters([slope_f, litho_f, veg_f, soil_f], out_index_path,
                           fn, "float32", -9999.0, block=block)


def probability_index(slope_f: str, litho_f: str, veg_f: str, soil_f: str,
                      out_path: str, weights: C.Weights,
                      intercept: Optional[float] = None,
                      block: int = 1024) -> str:
    """Continuous 0-1 susceptibility index, from the fitted logistic model.

        P = 1 / (1 + exp(-(b + sum_i w_i * log(f_i + 1))))

    This is the same model the calibration fits, evaluated at every pixel, so it
    needs no class breaks: no empty classes, no quantile ties, and the ordering
    between two pixels is always meaningful.

    IMPORTANT - this is a *relative* index, not an absolute probability of
    failure. It is fitted against background points that stand in for absences,
    so the intercept reflects how many background points were drawn, not how
    often landslides really occur. Pixel A scoring 0.8 against pixel B at 0.4 is
    a statement about their ordering and separation, not a claim that A fails
    80% of the time. Absolute rates need a prevalence correction against a
    known landslide density.

    With ``intercept=None`` the AOI's own median score is used, which centres
    the index on 0.5 and keeps it spread across the full range.
    """
    w = [weights.slope, weights.lithology, weights.vegetation,
         weights.soil_moisture]

    def linear(arrs: List[np.ndarray]) -> np.ndarray:
        clean = [np.where(np.isnan(a), 0.0, a) for a in arrs]
        z = np.zeros(clean[0].shape, dtype="float64")
        for wi, a in zip(w, clean):
            z += wi * np.log(a + 1.0)
        return z

    if intercept is None:                 # centre on the AOI median
        intercept = -_median_linear(slope_f, litho_f, veg_f, soil_f, linear,
                                    block)

    def fn(arrs: List[np.ndarray]) -> np.ndarray:
        masks = [(a == FACTOR_NODATA) | np.isnan(a) for a in arrs]
        any_nodata = np.any(masks, axis=0)
        clean = [np.where(np.isnan(a), 0.0, a) for a in arrs]
        p = 1.0 / (1.0 + np.exp(-np.clip(linear(arrs) + intercept, -30, 30)))
        # Hard physical constraints: flat ground and open water cannot fail.
        p = np.where((clean[0] == 0) | (clean[2] == 0), 0.0, p)
        # Emit the declared nodata value rather than NaN, so the raster is
        # self-consistent for readers that honour the nodata tag.
        return np.where(any_nodata, -9999.0, p)

    return combine_rasters([slope_f, litho_f, veg_f, soil_f], out_path, fn,
                           "float32", -9999.0, block=block)


def _median_linear(slope_f, litho_f, veg_f, soil_f, linear, block) -> float:
    """Median of the linear predictor over valid, non-flat pixels."""
    import rasterio

    from .grid import iter_blocks

    sample = []
    ds = [rasterio.open(p) for p in (slope_f, litho_f, veg_f, soil_f)]
    try:
        ref = ds[0]
        for win in iter_blocks(ref.width, ref.height, block):
            arrs = [d.read(1, window=win).astype("float64") for d in ds]
            ok = ~np.any([(a == FACTOR_NODATA) for a in arrs], axis=0)
            ok &= (arrs[0] > 0) & (arrs[2] > 0)
            if ok.any():
                sample.append(linear(arrs)[ok].ravel()[::7])
    finally:
        for d in ds:
            d.close()
    vals = np.concatenate(sample) if sample else np.array([0.0])
    return float(np.median(vals))


def classify_susceptibility(index_path: str, out_path: str,
                            breaks=None, block: int = 1024) -> str:
    """Reclassify the susceptibility index S into categories 1..5."""
    breaks = breaks or C.SUSCEPTIBILITY_BREAKS

    def fn(arr: np.ndarray) -> np.ndarray:
        arr = np.where(arr == -9999.0, np.nan, arr)
        cls = reclassify_continuous(arr, breaks, inclusive=True)
        # S == 0 (flat, water, etc.) -> Very Low (class 1), not nodata.
        cls = np.where((arr == 0) & np.isnan(cls), 1, cls)
        return np.where(np.isnan(cls), SUSC_NODATA, cls)

    return map_raster(index_path, out_path, fn, "uint8", SUSC_NODATA,
                      src_nodata=-9999.0, block=block)


def quantile_breaks(index_path: str, n_classes: int = 5,
                    max_sample: int = 2_000_000, block: int = 1024):
    """Equal-area quantile breaks of positive S values over the AOI.

    Returns a break list compatible with :func:`classify_susceptibility`.
    S == 0 cells (flat/water) are excluded from the quantiles and later mapped
    to class 1.
    """
    import rasterio
    from .grid import iter_blocks

    sample = []
    stride = 1
    with rasterio.open(index_path) as src:
        total = src.width * src.height
        if total > max_sample:
            stride = int(total / max_sample) + 1
        for win in iter_blocks(src.width, src.height, block):
            a = src.read(1, window=win).astype("float64").ravel()
            a = a[(a != src.nodata) & np.isfinite(a) & (a > 0)]
            if stride > 1:
                a = a[::stride]
            sample.append(a)
        vals = np.concatenate(sample) if sample else np.array([1.0])
    if not vals.size:
        return [(float("inf"), n_classes)]

    qs = np.linspace(0, 1, n_classes + 1)[1:-1]
    cuts = [float(c) for c in np.quantile(vals, qs)]

    # The index S is built from small integer factors, so its distribution is
    # lumpy: several quantiles can land on the same value, which would leave a
    # class with an empty interval (a hole in the map legend). Drop duplicate
    # cuts, and if that leaves too few, spread the cuts evenly over the
    # *distinct* values instead so all n_classes are populated.
    strictly_increasing: List[float] = []
    for c in cuts:
        if not strictly_increasing or c > strictly_increasing[-1]:
            strictly_increasing.append(c)
    if len(strictly_increasing) < n_classes - 1:
        distinct = np.unique(vals)
        if distinct.size >= n_classes:
            idx = np.linspace(0, distinct.size - 1, n_classes + 1)[1:-1]
            strictly_increasing = sorted({float(distinct[int(i)])
                                          for i in idx})

    breaks = [(c, i) for i, c in enumerate(strictly_increasing, start=1)]
    breaks.append((float("inf"), len(breaks) + 1))
    return breaks
