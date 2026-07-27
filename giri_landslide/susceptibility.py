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
                    block: int = 1024) -> str:
    """Compute the weighted-product susceptibility index S (float raster)."""
    w = [weights.slope, weights.lithology, weights.vegetation,
         weights.soil_moisture]

    def fn(arrs: List[np.ndarray]) -> np.ndarray:
        # Treat factor nodata as nodata for the whole cell.
        masks = [(a == FACTOR_NODATA) | np.isnan(a) for a in arrs]
        any_nodata = np.any(masks, axis=0)
        prod = np.ones(arrs[0].shape, dtype="float64")
        for wi, a in zip(w, arrs):
            prod *= wi * np.where(np.isnan(a), 0.0, a)
        return np.where(any_nodata, np.nan, prod)

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
