"""Step 1/2 - download open-source inputs and stage them for the model.

All downloaders fetch only the tiles that intersect the AOI, so a laptop run
pulls a handful of files rather than a global dataset. Every dataset used here
is openly accessible without authentication:

  * DEM         - Copernicus GLO-90 / GLO-30 DEM (AWS Open Data, 1-deg COG tiles)
  * Land cover  - ESA WorldCover 2021 v200 (AWS Open Data, 3-deg tiles)
  * Precip.     - WorldClim v2.1 monthly precipitation climatology (mm)
  * Lithology   - GLiM (Hartmann & Moosdorf 2012) - user-supplied vector/raster
  * PGA         - GEM/GSHAP seismic hazard - user-supplied raster

GLiM and the global PGA layer are distributed from portals that need a manual
(one-click) download or a login, so those are staged from a local path with a
clear pointer to the source; the pipeline degrades gracefully if they are
absent (see pipeline.py).
"""

from __future__ import annotations

import math
import os
import time
import warnings
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import requests

# Public endpoints -----------------------------------------------------------
COP_DEM_90 = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"
COP_DEM_30 = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
WORLDCLIM = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"

GLIM_SOURCE_INFO = (
    "GLiM lithology: download 'LiMW_GIS 2015.gdb' / shapefile from "
    "https://doi.pangaea.de/10.1594/PANGAEA.788537 and pass its path via "
    "config.glim_path (vector .shp/.gdb or a pre-rasterised code GeoTIFF)."
)
PGA_SOURCE_INFO = (
    "PGA: download the GEM Global Seismic Hazard Map (PGA, 475-yr return "
    "period) GeoTIFF from https://www.globalquakemodel.org/product/"
    "global-seismic-hazard-map and pass its path via config.trigger_path."
)


# ---------------------------------------------------------------------------
# Generic HTTP download with retries
# ---------------------------------------------------------------------------

def download_file(url: str, dest: str, retries: int = 4,
                  timeout: int = 120) -> Optional[str]:
    """Download ``url`` to ``dest`` (skip if present). Returns path or None (404)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                tmp = dest + ".part"
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                os.replace(tmp, dest)
                return dest
        except (requests.RequestException, OSError) as exc:
            if attempt == retries:
                raise
            print(f"  download retry {attempt}/{retries} ({exc}); "
                  f"waiting {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    return None


# ---------------------------------------------------------------------------
# Tile enumeration helpers
# ---------------------------------------------------------------------------

def _int_range(lo: float, hi: float, step: int) -> List[int]:
    a = int(math.floor(lo / step) * step)
    b = int(math.floor((hi - 1e-9) / step) * step)
    return list(range(a, b + 1, step))


def _ns(lat: int) -> str:
    return f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"


def _ew(lon: int) -> str:
    return f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"


# ---------------------------------------------------------------------------
# Copernicus DEM
# ---------------------------------------------------------------------------

def download_dem(bbox: Sequence[float], data_dir: str,
                 source: str = "copernicus90") -> List[str]:
    """Download 1-deg Copernicus DEM tiles intersecting ``bbox``."""
    base, res = (COP_DEM_90, "30") if source == "copernicus90" else (COP_DEM_30, "10")
    w, s, e, n = bbox
    out: List[str] = []
    for lat in _int_range(s, n, 1):
        for lon in _int_range(w, e, 1):
            tile = f"Copernicus_DSM_COG_{res}_{_ns(lat)}_00_{_ew(lon)}_00_DEM"
            url = f"{base}/{tile}/{tile}.tif"
            dest = os.path.join(data_dir, "dem", f"{tile}.tif")
            got = download_file(url, dest)
            if got:
                out.append(got)
    if not out:
        raise RuntimeError(
            "No Copernicus DEM tiles found for the AOI (all ocean, or network "
            "blocked). Provide a local DEM via config.dem_path.")
    return out


# ---------------------------------------------------------------------------
# ESA WorldCover
# ---------------------------------------------------------------------------

def download_worldcover(bbox: Sequence[float], data_dir: str) -> List[str]:
    """Download 3-deg ESA WorldCover 2021 tiles intersecting ``bbox``."""
    w, s, e, n = bbox
    out: List[str] = []
    for lat in _int_range(s, n, 3):
        for lon in _int_range(w, e, 3):
            tile = f"{_ns(lat)}{_ew(lon)}"
            fn = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
            url = f"{WORLDCOVER}/{fn}"
            dest = os.path.join(data_dir, "landcover", fn)
            got = download_file(url, dest)
            if got:
                out.append(got)
    if not out:
        raise RuntimeError(
            "No ESA WorldCover tiles found for the AOI. Provide a local land "
            "cover raster via config.landcover_path.")
    return out


# ---------------------------------------------------------------------------
# WorldClim monthly precipitation (soil-moisture proxy)
# ---------------------------------------------------------------------------

def download_worldclim_precip(data_dir: str, res: str = "10m") -> List[str]:
    """Download the 12 WorldClim v2.1 monthly precipitation rasters (mm).

    ``res`` in {"10m", "5m", "2.5m", "30s"}. Returns the 12 GeoTIFF paths.
    """
    zip_name = f"wc2.1_{res}_prec.zip"
    url = f"{WORLDCLIM}/{zip_name}"
    dest_zip = os.path.join(data_dir, "worldclim", zip_name)
    got = download_file(url, dest_zip, timeout=600)
    if not got:
        raise RuntimeError(f"WorldClim precip not available at {url}")
    out_dir = os.path.join(data_dir, "worldclim", res)
    os.makedirs(out_dir, exist_ok=True)
    tifs: List[str] = []
    with zipfile.ZipFile(got) as zf:
        for member in sorted(zf.namelist()):
            if member.endswith(".tif"):
                target = os.path.join(out_dir, os.path.basename(member))
                if not os.path.exists(target):
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                tifs.append(target)
    if len(tifs) != 12:
        raise RuntimeError(f"expected 12 WorldClim precip tiles, got {len(tifs)}")
    return tifs


def max_monthly_precip(monthly_paths: Sequence[str], out_path: str,
                       block: int = 1024) -> str:
    """Pixel-wise maximum across 12 monthly precip rasters -> MYMMR proxy (mm).

    Approximates the paper's "mean year maximum monthly rainfall" from an open
    monthly climatology.
    """
    import numpy as np
    import rasterio
    from .grid import iter_blocks, _clamp_block

    datasets = [rasterio.open(p) for p in monthly_paths]
    try:
        ref = datasets[0]
        blk = _clamp_block(512, ref.width, ref.height)
        prof = ref.profile.copy()
        prof.update(dtype="float32", count=1, nodata=-9999.0,
                    compress="deflate", tiled=True,
                    blockxsize=blk, blockysize=blk)
        with rasterio.open(out_path, "w", **prof) as dst:
            for win in iter_blocks(ref.width, ref.height, block):
                stack = []
                for ds in datasets:
                    a = ds.read(1, window=win).astype("float64")
                    nod = ds.nodata
                    if nod is not None:
                        a = np.where(a == nod, np.nan, a)
                    a = np.where(a < 0, np.nan, a)  # WorldClim ocean = -32768
                    stack.append(a)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mx = np.nanmax(np.stack(stack), axis=0)
                dst.write(np.where(np.isnan(mx), -9999.0, mx).astype("float32"),
                          1, window=win)
    finally:
        for ds in datasets:
            ds.close()
    return out_path


# ---------------------------------------------------------------------------
# GLiM lithology -> rasterised Sl code grid
# ---------------------------------------------------------------------------

def rasterize_glim(glim_path: str, grid, out_path: str,
                   code_field: Optional[str] = None) -> str:
    """Rasterise a GLiM vector onto ``grid``, burning the Sl factor (0..3).

    Requires ``fiona``/``geopandas``. If ``glim_path`` is already a raster of
    GLiM level-1 codes, use :func:`rasterize_glim_from_raster` instead.
    """
    import numpy as np
    import rasterio
    from rasterio.features import rasterize

    try:
        import fiona
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading a GLiM vector needs 'fiona' (pip install fiona). "
            + GLIM_SOURCE_INFO) from exc

    from . import config as C

    shapes = []
    with fiona.open(glim_path) as src:
        field = code_field or _guess_glim_field(src.schema["properties"])
        for feat in src.filter(bbox=(grid.west, grid.south, grid.east,
                                     grid.north)):
            code = (feat["properties"].get(field) or "nd")
            code = str(code)[:2].lower()
            sl = C.GLIM_SL.get(code, C.GLIM_SL_DEFAULT)
            shapes.append((feat["geometry"], sl))

    prof = grid.profile("uint8", 255)
    arr = rasterize(shapes, out_shape=grid.shape, transform=grid.transform,
                    fill=255, dtype="uint8") if shapes else \
        np.full(grid.shape, 255, dtype="uint8")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr, 1)
    return out_path


def _guess_glim_field(props: Dict[str, str]) -> str:
    for cand in ("xx", "Litho", "litho", "LITHO", "level1", "Level1", "class"):
        if cand in props:
            return cand
    # fall back to first string field
    for name, typ in props.items():
        if str(typ).startswith("str"):
            return name
    raise RuntimeError("could not identify GLiM lithology code field")
