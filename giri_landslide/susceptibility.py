"""Step 3.2 - combine susceptibility factors into the 5-class susceptibility map.

    S = product_i ( w_i * f(S_i) )                      (Eq. 1)

then S is reclassified into categories 1..5 (Very Low .. Very High).
"""

from __future__ import annotations

from typing import List

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
    qs = np.linspace(0, 1, n_classes + 1)[1:-1]
    cuts = list(np.quantile(vals, qs)) if vals.size else []
    breaks = []
    for i, c in enumerate(cuts, start=1):
        breaks.append((float(c), i))
    breaks.append((float("inf"), n_classes))
    return breaks
