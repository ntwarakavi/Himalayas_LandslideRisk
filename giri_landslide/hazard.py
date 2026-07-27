"""Step 5 - combine susceptibility class x trigger class -> landslide hazard.

The hazard index is the probability (per event of the given scenario) that a
significant landslide impacts a 1 km infrastructure stretch, looked up from the
rainfall (Fig. 3) or earthquake (Fig. 4) hazard matrix.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .grid import combine_rasters
from .susceptibility import SUSC_NODATA
from .triggers import TRIGGER_NODATA

HAZARD_NODATA = -9999.0


def apply_hazard_matrix(susc_path: str, trigger_path: str, out_path: str,
                        trigger_kind: str, block: int = 1024) -> str:
    """Cross susceptibility (1..5) with trigger class into a probability raster.

    Rainfall trigger classes are 1..5; earthquake trigger classes are 0..5
    (0 = PGA < 0.05 g -> probability 0).
    """
    if trigger_kind == "rainfall":
        matrix = np.array(C.RAINFALL_MATRIX, dtype="float64")  # rows 1..5
        min_trig = 1
    else:
        matrix = np.array(C.EARTHQUAKE_MATRIX, dtype="float64")  # rows 1..5
        min_trig = 0

    def fn(arrs):
        susc, trig = arrs
        valid = (~np.isnan(susc)) & (~np.isnan(trig)) & \
                (susc != SUSC_NODATA) & (trig != TRIGGER_NODATA)
        out = np.full(susc.shape, HAZARD_NODATA)

        s = np.clip(np.nan_to_num(susc, nan=1), 1, 5).astype("int64")
        t = np.nan_to_num(trig, nan=0).astype("int64")

        active = valid & (t >= max(min_trig, 1))  # class 0 -> probability 0
        # Matrix row index = trigger class - 1; col = susc class - 1.
        ti = np.clip(t - 1, 0, matrix.shape[0] - 1)
        si = np.clip(s - 1, 0, matrix.shape[1] - 1)
        probs = matrix[ti, si]
        out = np.where(valid, 0.0, HAZARD_NODATA)
        out = np.where(active, probs, out)
        return out

    return combine_rasters([susc_path, trigger_path], out_path, fn,
                           "float32", HAZARD_NODATA, block=block)
