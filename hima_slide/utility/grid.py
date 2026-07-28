"""Raster grid utilities for memory-bounded, tile-based processing.

The whole pipeline is anchored to a single reference :class:`Grid` (derived
from the AOI bounding box and target resolution). Every input dataset is warped
onto that grid, and every per-pixel operation is executed block-by-block with
:func:`iter_blocks`, so a large area can be processed on a laptop without ever
holding a full continent-sized array in memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import Window

WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class Grid:
    """A regular lat/lon raster grid in EPSG:4326."""

    west: float
    south: float
    east: float
    north: float
    res: float
    crs: str = WGS84

    @property
    def width(self) -> int:
        return max(1, int(round((self.east - self.west) / self.res)))

    @property
    def height(self) -> int:
        return max(1, int(round((self.north - self.south) / self.res)))

    @property
    def transform(self):
        # Origin is the top-left corner (north-west).
        return from_origin(self.west, self.north, self.res, self.res)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @classmethod
    def from_bbox(cls, bbox: Sequence[float], res: float) -> "Grid":
        w, s, e, n = bbox
        # Snap extent so width/height are whole pixels.
        width = max(1, int(round((e - w) / res)))
        height = max(1, int(round((n - s) / res)))
        return cls(west=w, south=n - height * res, east=w + width * res,
                   north=n, res=res)

    def profile(self, dtype: str, nodata, count: int = 1,
                block: int = 512) -> dict:
        block = _clamp_block(block, self.width, self.height)
        return {
            "driver": "GTiff",
            "height": self.height,
            "width": self.width,
            "count": count,
            "dtype": dtype,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": nodata,
            "tiled": True,
            "blockxsize": block,
            "blockysize": block,
            "compress": "deflate",
            "predictor": 2,
            "BIGTIFF": "IF_SAFER",
        }


def _clamp_block(block: int, width: int, height: int) -> int:
    """GeoTIFF internal block size must be a multiple of 16 and <= dimensions."""
    block = min(block, ((max(width, 1) + 15) // 16) * 16,
                ((max(height, 1) + 15) // 16) * 16)
    block = max(16, (block // 16) * 16)
    return block


def iter_blocks(width: int, height: int, block: int) -> Iterator[Window]:
    """Yield read/write windows tiling a (height, width) raster."""
    for row in range(0, height, block):
        h = min(block, height - row)
        for col in range(0, width, block):
            w = min(block, width - col)
            yield Window(col_off=col, row_off=row, width=w, height=h)


# ---------------------------------------------------------------------------
# Warping arbitrary sources onto the reference grid
# ---------------------------------------------------------------------------

def warp_to_grid(src_path: str, grid: Grid, out_path: str,
                 resampling: Resampling = Resampling.bilinear,
                 dtype: Optional[str] = None, nodata=None,
                 src_band: int = 1, block: int = 512) -> str:
    """Reproject/resample one band of ``src_path`` onto ``grid``.

    The output is a tiled, compressed GeoTIFF aligned exactly to the grid.
    """
    with rasterio.open(src_path) as src:
        out_dtype = dtype or src.dtypes[src_band - 1]
        out_nodata = nodata if nodata is not None else src.nodata
        prof = grid.profile(out_dtype, out_nodata, block=block)
        with rasterio.open(out_path, "w", **prof) as dst:
            reproject(
                source=rasterio.band(src, src_band),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs or grid.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=resampling,
                num_threads=2,
            )
    return out_path


def mosaic_and_warp(src_paths: Sequence[str], grid: Grid, out_path: str,
                    resampling: Resampling = Resampling.bilinear,
                    dtype: Optional[str] = None, nodata=None,
                    block: int = 512) -> str:
    """Warp several (possibly overlapping) source tiles onto ``grid``.

    Sources are reprojected one after another into the same destination, which
    naturally mosaics AOI-intersecting tiles without a giant in-memory merge.
    """
    if not src_paths:
        raise ValueError("no source paths supplied to mosaic_and_warp")
    with rasterio.open(src_paths[0]) as s0:
        out_dtype = dtype or s0.dtypes[0]
        out_nodata = nodata if nodata is not None else (s0.nodata if s0.nodata
                                                        is not None else 0)
    prof = grid.profile(out_dtype, out_nodata, block=block)
    with rasterio.open(out_path, "w", **prof) as dst:
        for sp in src_paths:
            with rasterio.open(sp) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs or grid.crs,
                    dst_transform=grid.transform,
                    dst_crs=grid.crs,
                    resampling=resampling,
                    dst_nodata=out_nodata,
                    init_dest_nodata=(sp == src_paths[0]),
                    num_threads=2,
                )
    return out_path


# ---------------------------------------------------------------------------
# Block-wise map / reclassify helpers
# ---------------------------------------------------------------------------

def map_raster(src_path: str, out_path: str, func: Callable[[np.ndarray],
               np.ndarray], out_dtype: str, out_nodata, block: int = 512,
               src_nodata=None) -> str:
    """Apply ``func`` to each block of a single-band raster, writing results.

    ``func`` receives a masked-aware float array (nodata -> np.nan) and must
    return an array of the same shape.
    """
    with rasterio.open(src_path) as src:
        prof = src.profile.copy()
        prof.update(dtype=out_dtype, nodata=out_nodata, count=1,
                    compress="deflate", tiled=True,
                    blockxsize=_clamp_block(block, src.width, src.height),
                    blockysize=_clamp_block(block, src.width, src.height))
        nod = src_nodata if src_nodata is not None else src.nodata
        with rasterio.open(out_path, "w", **prof) as dst:
            for win in iter_blocks(src.width, src.height, block):
                arr = src.read(1, window=win).astype("float64")
                if nod is not None:
                    arr = np.where(arr == nod, np.nan, arr)
                out = func(arr)
                dst.write(out.astype(out_dtype), 1, window=win)
    return out_path


def combine_rasters(src_paths: Sequence[str], out_path: str,
                    func: Callable[[List[np.ndarray]], np.ndarray],
                    out_dtype: str, out_nodata, block: int = 512) -> str:
    """Combine several aligned rasters block-by-block via ``func``.

    ``func`` receives a list of float arrays (nodata -> np.nan), one per input,
    and returns a single array.
    """
    datasets = [rasterio.open(p) for p in src_paths]
    try:
        ref = datasets[0]
        prof = ref.profile.copy()
        prof.update(dtype=out_dtype, nodata=out_nodata, count=1,
                    compress="deflate", tiled=True,
                    blockxsize=_clamp_block(block, ref.width, ref.height),
                    blockysize=_clamp_block(block, ref.width, ref.height))
        with rasterio.open(out_path, "w", **prof) as dst:
            for win in iter_blocks(ref.width, ref.height, block):
                arrs = []
                for ds in datasets:
                    a = ds.read(1, window=win).astype("float64")
                    if ds.nodata is not None:
                        a = np.where(a == ds.nodata, np.nan, a)
                    arrs.append(a)
                dst.write(func(arrs).astype(out_dtype), 1, window=win)
    finally:
        for ds in datasets:
            ds.close()
    return out_path


def reclassify_continuous(arr: np.ndarray,
                          breaks: Sequence[Tuple[float, int]],
                          inclusive: bool = True) -> np.ndarray:
    """Map a continuous array to integer classes using (upper_bound, value).

    ``inclusive`` -> bins are (prev, bound]; otherwise [prev, bound).
    NaNs propagate to NaN.
    """
    out = np.full(arr.shape, np.nan)
    prev = -np.inf
    for bound, value in breaks:
        if inclusive:
            mask = (arr > prev) & (arr <= bound)
        else:
            mask = (arr >= prev) & (arr < bound)
        out[mask] = value
        prev = bound
    return out


def remap_categorical(arr: np.ndarray, mapping: Dict[int, int],
                      default: int) -> np.ndarray:
    """Map integer category codes to new values via a lookup dict."""
    out = np.full(arr.shape, float(default))
    finite = ~np.isnan(arr)
    codes = np.where(finite, arr, 0).astype("int64")
    for code, val in mapping.items():
        out[finite & (codes == code)] = val
    return out


def raster_stats(path: str) -> Dict[str, float]:
    """Cheap block-wise min/max/mean/valid-count for reporting."""
    total = 0.0
    count = 0
    vmin = math.inf
    vmax = -math.inf
    with rasterio.open(path) as src:
        for win in iter_blocks(src.width, src.height, 1024):
            a = src.read(1, window=win).astype("float64")
            if src.nodata is not None:
                a = a[a != src.nodata]
            a = a[np.isfinite(a)]
            if a.size:
                total += float(a.sum())
                count += a.size
                vmin = min(vmin, float(a.min()))
                vmax = max(vmax, float(a.max()))
    return {
        "min": vmin if count else float("nan"),
        "max": vmax if count else float("nan"),
        "mean": (total / count) if count else float("nan"),
        "valid": count,
    }
