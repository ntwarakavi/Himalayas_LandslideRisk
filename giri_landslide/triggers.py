"""Step 4 - triggering conditions (rainfall return period / earthquake PGA).

Each trigger produces an integer *trigger class* raster (1..5), which is later
crossed with the susceptibility class through a hazard matrix.
"""

from __future__ import annotations

import math
import numpy as np

from . import config as C
from .grid import map_raster

TRIGGER_NODATA = 255

_EULER = 0.5772156649015329
_SQRT6_PI = math.sqrt(6.0) / math.pi


# ---------------------------------------------------------------------------
# 4.1  Rainfall
# ---------------------------------------------------------------------------

def gumbel_return_period(z: np.ndarray) -> np.ndarray:
    """Return period (years) of a normalised 24h rainfall z = (I - mu)/sigma.

    Assuming annual maxima follow a Gumbel distribution, the standardised
    variate maps to the reduced variate y and hence to T:

        z = (sqrt(6)/pi) * (y - gamma),   y = -ln(-ln(1 - 1/T))
        =>  T = 1 / (1 - exp(-exp(-y)))
    """
    y = z / _SQRT6_PI + _EULER
    # exceedance probability p = 1 - exp(-exp(-y)); guard numerics.
    with np.errstate(over="ignore"):
        p = 1.0 - np.exp(-np.exp(-y))
    p = np.clip(p, 1e-9, 1.0)
    return 1.0 / p


def rainfall_class_from_norm(norm_path: str, out_path: str,
                             block: int = 1024) -> str:
    """Classify a normalised-24h-rainfall raster into rainfall classes 1..5."""
    rps = C.RAINFALL_RETURN_PERIODS_YR  # [5, 25, 200, 1000]

    def fn(z: np.ndarray) -> np.ndarray:
        T = gumbel_return_period(z)
        cls = np.ones(T.shape)
        cls = np.where(T >= rps[0], 2, cls)
        cls = np.where(T >= rps[1], 3, cls)
        cls = np.where(T >= rps[2], 4, cls)
        cls = np.where(T >= rps[3], 5, cls)
        return np.where(np.isnan(z), TRIGGER_NODATA, cls)

    return map_raster(norm_path, out_path, fn, "uint8", TRIGGER_NODATA,
                      block=block)


def rainfall_class_from_return_period(grid_template_path: str, out_path: str,
                                      return_period_yr: float,
                                      block: int = 1024) -> str:
    """Uniform-scenario rainfall class from a single return period value.

    Uses ``grid_template_path`` only to copy the grid / valid-data mask.
    """
    rps = C.RAINFALL_RETURN_PERIODS_YR
    cls = 1
    if return_period_yr >= rps[0]:
        cls = 2
    if return_period_yr >= rps[1]:
        cls = 3
    if return_period_yr >= rps[2]:
        cls = 4
    if return_period_yr >= rps[3]:
        cls = 5

    def fn(arr: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(arr), TRIGGER_NODATA, float(cls))

    return map_raster(grid_template_path, out_path, fn, "uint8",
                      TRIGGER_NODATA, block=block)


# ---------------------------------------------------------------------------
# 4.2  Earthquake
# ---------------------------------------------------------------------------

def pga_class(pga_path: str, out_path: str, block: int = 1024) -> str:
    """Classify a PGA raster (g) into seismic hazard classes 1..5 (0 below 0.05g)."""
    thr = C.PGA_THRESHOLDS_G  # [0.05, 0.15, 0.25, 0.35, 0.45]

    def fn(pga: np.ndarray) -> np.ndarray:
        cls = np.zeros(pga.shape)
        for k, t in enumerate(thr, start=1):
            cls = np.where(pga >= t, k, cls)
        return np.where(np.isnan(pga), TRIGGER_NODATA, cls)

    return map_raster(pga_path, out_path, fn, "uint8", TRIGGER_NODATA,
                      block=block)


def pga_class_uniform(grid_template_path: str, out_path: str, pga_g: float,
                      block: int = 1024) -> str:
    """Uniform-scenario PGA class from a single PGA value."""
    thr = C.PGA_THRESHOLDS_G
    cls = 0
    for k, t in enumerate(thr, start=1):
        if pga_g >= t:
            cls = k

    def fn(arr: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(arr), TRIGGER_NODATA, float(cls))

    return map_raster(grid_template_path, out_path, fn, "uint8",
                      TRIGGER_NODATA, block=block)
