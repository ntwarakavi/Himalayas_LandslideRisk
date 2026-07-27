"""Step 3.1 - convert input datasets into susceptibility factor rasters.

Each function reads a grid-aligned input raster (produced by the data stage) and
writes an integer factor raster on the same grid, processed block-by-block.
"""

from __future__ import annotations

import math
import numpy as np
import rasterio
from rasterio.windows import Window

from .. import config as C
from ..utility.grid import iter_blocks, map_raster, remap_categorical, reclassify_continuous

FACTOR_NODATA = 255  # uint8 nodata for factor rasters


# ---------------------------------------------------------------------------
# Slope factor (Table 2)
# ---------------------------------------------------------------------------

def compute_slope_degrees(dem_path: str, out_path: str, block: int = 1024) -> str:
    """Compute terrain slope (degrees) from a lat/lon DEM, block-by-block.

    Uses the Horn (3x3) finite-difference method. Cell spacing is converted from
    degrees to metres per block using the local latitude, so slopes are correct
    for a geographic DEM. A one-pixel halo is read around each block to avoid
    seams at tile boundaries.
    """
    with rasterio.open(dem_path) as src:
        res_deg = src.transform.a
        top = src.bounds.top
        nod = src.nodata
        prof = src.profile.copy()
        prof.update(dtype="float32", nodata=-9999.0, count=1,
                    compress="deflate", tiled=True)
        with rasterio.open(out_path, "w", **prof) as dst:
            for win in iter_blocks(src.width, src.height, block):
                pad = Window(win.col_off - 1, win.row_off - 1,
                             win.width + 2, win.height + 2)
                z = src.read(1, window=pad, boundless=True,
                             fill_value=(nod if nod is not None else 0)
                             ).astype("float64")
                if nod is not None:
                    z = np.where(z == nod, np.nan, z)

                # Latitude at each row of the padded block -> metric spacing.
                row0 = win.row_off - 1
                lats = top - (row0 + np.arange(z.shape[0]) + 0.5) * res_deg
                lat_c = math.radians(float(np.nanmean(lats)))
                dx = res_deg * 111320.0 * max(math.cos(lat_c), 1e-6)
                dy = res_deg * 110540.0

                slope = _horn_slope(z, dx, dy)
                # Trim the halo back to the block interior.
                out = slope[1:1 + win.height, 1:1 + win.width]
                out = np.where(np.isnan(out), -9999.0, out)
                dst.write(out.astype("float32"), 1, window=win)
    return out_path


def _horn_slope(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Horn's method slope in degrees for a 2D elevation array."""
    # Neighbour shifts (edges padded by replication to avoid NaN spread).
    zp = np.pad(z, 1, mode="edge")
    a = zp[:-2, :-2]; b = zp[:-2, 1:-1]; c = zp[:-2, 2:]
    d = zp[1:-1, :-2];                    f = zp[1:-1, 2:]
    g = zp[2:, :-2];  h = zp[2:, 1:-1];   i = zp[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy)
    rise = np.sqrt(dzdx ** 2 + dzdy ** 2)
    return np.degrees(np.arctan(rise))


def slope_factor(slope_deg_path: str, out_path: str, block: int = 1024,
                 breaks=None) -> str:
    """Reclassify slope (degrees) into the Sr factor.

    ``breaks`` defaults to the manuscript's Table 2; pass a calibrated table to
    match a different DEM resolution.
    """
    breaks = breaks or C.SLOPE_BREAKS_DEG

    def fn(arr: np.ndarray) -> np.ndarray:
        cls = reclassify_continuous(arr, breaks, inclusive=False)
        return np.where(np.isnan(cls), FACTOR_NODATA, cls)

    return map_raster(slope_deg_path, out_path, fn, "uint8", FACTOR_NODATA,
                      block=block)


# ---------------------------------------------------------------------------
# Lithology factor (GLiM -> 1..3)
# ---------------------------------------------------------------------------

def lithology_factor(glim_code_path: str, out_path: str,
                     block: int = 1024) -> str:
    """Reclassify a rasterised GLiM level-1 code grid into Sl (0/1/2/3).

    ``glim_code_path`` must be an integer raster whose values are the indices
    produced by :func:`giri_landslide.sources.rasterize_glim` (which stores an
    index->code table alongside the raster). To keep this module dependency
    free, the mapping is applied by numeric index here; see sources.py.
    """
    def fn(arr: np.ndarray) -> np.ndarray:
        # The rasteriser encodes Sl directly as the burn value, so this is a
        # pass-through clamp; kept as a function for uniformity/validation.
        out = np.where(np.isnan(arr), FACTOR_NODATA, arr)
        out = np.where((out < 0) | (out > 3), FACTOR_NODATA, out)
        return out

    return map_raster(glim_code_path, out_path, fn, "uint8", FACTOR_NODATA,
                      block=block)


# ---------------------------------------------------------------------------
# Vegetation / land-cover factor (Table 5)
# ---------------------------------------------------------------------------

def landcover_factor(landcover_path: str, out_path: str, source: str,
                     block: int = 1024) -> str:
    """Reclassify a land-cover code raster into the Sv factor of Table 5."""
    mapping = C.WORLDCOVER_SV if source == "worldcover" else C.LCCS_SV
    default = C.WORLDCOVER_SV_NODATA

    def fn(arr: np.ndarray) -> np.ndarray:
        out = remap_categorical(arr, mapping, default)
        return np.where(np.isnan(arr), FACTOR_NODATA, out)

    return map_raster(landcover_path, out_path, fn, "uint8", FACTOR_NODATA,
                      block=block)


# ---------------------------------------------------------------------------
# Soil-moisture factor (Table 3 rainfall proxy / Table 4 VWC)
# ---------------------------------------------------------------------------

def soil_moisture_factor(source_path: str, out_path: str, trigger: str,
                         block: int = 1024) -> str:
    """Reclassify the soil-moisture proxy into Sp.

    For ``trigger == 'rainfall'`` the input is the mean-year-maximum-monthly
    rainfall (mm, Table 3); for ``'earthquake'`` it is the ERA5 volumetric
    water content (m3/m3, Table 4).
    """
    breaks = C.MYMMR_BREAKS_MM if trigger == "rainfall" else C.VWC_BREAKS

    def fn(arr: np.ndarray) -> np.ndarray:
        cls = reclassify_continuous(arr, breaks, inclusive=True)
        return np.where(np.isnan(cls), FACTOR_NODATA, cls)

    return map_raster(source_path, out_path, fn, "uint8", FACTOR_NODATA,
                      block=block)
