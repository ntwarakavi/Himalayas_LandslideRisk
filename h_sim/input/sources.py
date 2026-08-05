"""Step 1/2 - download open-source inputs and stage them for the model.

All downloaders fetch only the tiles that intersect the AOI, so a laptop run
pulls a handful of files rather than a global dataset. Every dataset used here
is openly accessible without authentication:

  * DEM         - Copernicus GLO-30 / GLO-90 DEM (AWS Open Data, 1-deg COG tiles)
  * Precip.     - WorldClim v2.1 monthly precipitation climatology (mm), and
                  downscaled CMIP6 projections for future scenarios
  * Land cover  - ESA WorldCover 2021 v200 (AWS Open Data, 3-deg tiles)
  * Lithology   - GLiM (Hartmann & Moosdorf 2012)
  * PGA         - GEM/GSHAP seismic hazard - user-supplied raster

Only the DEM and the precipitation climatology are needed for a plain run. Land
cover and GLiM are fetched only when a run asks for calibration regions, and the
global PGA layer comes from a portal that needs a manual download, so it is
staged from a local path with a clear pointer to the source; the pipeline
degrades gracefully if any of them are absent (see pipeline.py).
"""

from __future__ import annotations

import math
import os
import time
import warnings
import zipfile
from typing import Dict, List, Optional, Sequence

import requests

# Public endpoints -----------------------------------------------------------
COP_DEM_90 = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"
COP_DEM_30 = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
WORLDCOVER = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
WORLDCLIM = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"
# Downscaled CMIP6 projections (future climate scenarios).
WORLDCLIM_CMIP6 = "https://geodata.ucdavis.edu/cmip6"

# GLiM (Hartmann & Moosdorf 2012). The PANGAEA record distributes a global
# 0.5-degree ASCII grid of level-1 lithology classes plus a legend, which this
# package can download and use directly. The full 1.2M-polygon vector database
# ('LiMW_GIS 2015.gdb') is available from the same record's landing page and can
# be supplied instead via config.glim_path for much finer lithological detail.
GLIM_ASCII_URL = "https://hdl.handle.net/10013/epic.39939.d001"
GLIM_LANDING_PAGE = "https://doi.pangaea.de/10.1594/PANGAEA.788537"
# Full GLiM geodatabase (1,235,259 polygons, ~1.14 GB), as published by the
# authors via the University of Hamburg GLiM page.
GLIM_VECTOR_URL = (
    "https://www.dropbox.com/s/9vuowtebp9f1iud/LiMW_GIS%202015.gdb.zip?dl=1"
)
GLIM_VECTOR_DIRNAME = "LiMW_GIS 2015.gdb"
GLIM_SOURCE_INFO = (
    "GLiM lithology: the global 0.5-degree grid is downloaded automatically "
    f"from {GLIM_ASCII_URL}. For higher lithological resolution, download the "
    f"full vector database ('LiMW_GIS 2015.gdb') from {GLIM_LANDING_PAGE} and "
    "pass its path via config.glim_path."
)

# GLiM raster value -> level-1 lithology code (from the distributed
# Classnames.txt legend).
GLIM_VALUE_TO_CODE = {
    1: "su", 2: "vb", 3: "ss", 4: "pb", 5: "sm", 6: "sc", 7: "va", 8: "mt",
    9: "pa", 10: "vi", 11: "wb", 12: "py", 13: "pi", 14: "ev", 15: "nd",
    16: "ig",
}
PGA_SOURCE_INFO = (
    "PGA: download the GEM Global Seismic Hazard Map (PGA, 475-yr return "
    "period) GeoTIFF from https://www.globalquakemodel.org/product/"
    "global-seismic-hazard-map and pass its path via config.pga_path."
)


# ---------------------------------------------------------------------------
# Generic HTTP download with retries
# ---------------------------------------------------------------------------

def download_file(url: str, dest: str, retries: int = 4,
                  timeout: int = 120) -> Optional[str]:
    """Download ``url`` to ``dest`` (skip if present). Returns path or None (404).

    Interrupted transfers resume: a leftover ``.part`` is continued with an
    HTTP Range request when the server honours it (a 206), and started over
    when it does not (a 200) - so losing the connection at 90% of a large
    climate file costs the last 10%, not the whole file.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, stream=True, timeout=timeout,
                              headers=headers) as r:
                if r.status_code == 404:
                    return None
                if have and r.status_code == 416:
                    # Range not satisfiable: the part is already complete.
                    os.replace(tmp, dest)
                    return dest
                r.raise_for_status()
                resume = have and r.status_code == 206
                with open(tmp, "ab" if resume else "wb") as fh:
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
                 source: str = "copernicus90",
                 max_workers: int = 6) -> List[str]:
    """Download 1-deg Copernicus DEM tiles intersecting ``bbox``.

    Tiles are independent files on a CDN, so they are fetched concurrently -
    the wall-clock win is roughly the worker count on a region-scale sweep.
    Cached tiles cost a stat call each; results keep tile order so the mosaic
    is deterministic.
    """
    from concurrent.futures import ThreadPoolExecutor

    base, res = (COP_DEM_90, "30") if source == "copernicus90" else (COP_DEM_30, "10")
    w, s, e, n = bbox
    jobs = []
    for lat in _int_range(s, n, 1):
        for lon in _int_range(w, e, 1):
            tile = f"Copernicus_DSM_COG_{res}_{_ns(lat)}_00_{_ew(lon)}_00_DEM"
            jobs.append((f"{base}/{tile}/{tile}.tif",
                         os.path.join(data_dir, "dem", f"{tile}.tif")))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        got = list(pool.map(lambda j: download_file(*j), jobs))
    out = [g for g in got if g]
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


def download_worldclim_elevation(data_dir: str, res: str = "5m"
                                 ) -> Optional[str]:
    """Global elevation on the WorldClim grid. One small file, cached.

    Used only to decide which states and provinces are mountainous enough to
    be worth running - see :func:`h_sim.input.admin.relief_stats`. At 5 arc
    minutes it is far too coarse for the stability model, and is never used
    for one: about 9 km per cell, 4.8 MB for the world.
    """
    out_dir = os.path.join(data_dir, "worldclim")
    tif = os.path.join(out_dir, f"wc2.1_{res}_elev.tif")
    if os.path.exists(tif):
        return tif
    dest = os.path.join(out_dir, f"wc2.1_{res}_elev.zip")
    got = download_file(f"{WORLDCLIM}/wc2.1_{res}_elev.zip", dest, timeout=300)
    if not got:
        return None
    with zipfile.ZipFile(got) as zf:
        zf.extractall(out_dir)
    return tif if os.path.exists(tif) else None


def download_worldclim_future(data_dir: str, ssp: str,
                              period: str = "2041-2060",
                              model: str = "IPSL-CM6A-LR",
                              res: str = "2.5m") -> Optional[List]:
    """Download downscaled CMIP6 monthly precipitation for a future scenario.

    WorldClim distributes these as a single 12-band GeoTIFF (one band per
    month), so the return value is a list of ``(path, band)`` pairs matching the
    interface of :func:`download_worldclim_precip`.

    ``ssp`` is e.g. "ssp126"/"ssp585"; ``period`` one of 2021-2040, 2041-2060,
    2061-2080, 2081-2100. The default model, IPSL-CM6A-LR, sits mid-range for
    climate sensitivity among the CMIP6 ensemble.
    """
    fn = f"wc2.1_{res}_prec_{model}_{ssp}_{period}.tif"
    url = f"{WORLDCLIM_CMIP6}/{res}/{model}/{ssp}/{fn}"
    dest = os.path.join(data_dir, "worldclim_future", fn)
    got = download_file(url, dest, timeout=1800)
    if not got:
        print(f"  future climate not available: {url}\n"
              "  check the model/period/ssp combination at "
              "https://worldclim.org/data/cmip6/cmip6climate.html")
        return None
    return [(got, m) for m in range(1, 13)]


def max_monthly_precip(monthly_paths: Sequence, grid, out_path: str,
                       tmp_prefix: str, block: int = 1024) -> str:
    """Pixel-wise maximum of 12 monthly precip rasters on ``grid`` (mm).

    Approximates the paper's "mean year maximum monthly rainfall" from an open
    monthly climatology.

    ``monthly_paths`` entries may be plain paths (one file per month, as in the
    current-climate product) or ``(path, band)`` pairs (a single 12-band file,
    as in the CMIP6 future-climate product).

    Each monthly raster is clipped/resampled onto the AOI grid *first* and the
    maximum is taken afterwards. Doing it in that order matters: the WorldClim
    30s product is a 43200 x 21600 global grid, so taking the maximum globally
    and clipping afterwards would process ~11 billion pixels for an AOI that
    needs a few million.
    """
    import numpy as np
    from rasterio.enums import Resampling

    from ..utility.grid import combine_rasters, warp_to_grid

    clipped: List[str] = []
    for i, entry in enumerate(monthly_paths, start=1):
        p, band = entry if isinstance(entry, (tuple, list)) else (entry, 1)
        t = f"{tmp_prefix}_prec_{i:02d}.tif"
        warp_to_grid(p, grid, t, Resampling.bilinear, dtype="float32",
                     nodata=-9999.0, src_band=band, block=block)
        clipped.append(t)

    def fn(arrs):
        # WorldClim marks ocean with a large negative value.
        stack = np.stack([np.where(a < 0, np.nan, a) for a in arrs])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mx = np.nanmax(stack, axis=0)
        return np.where(np.isnan(mx), -9999.0, mx)

    result = combine_rasters(clipped, out_path, fn, "float32", -9999.0,
                             block=block)
    for t in clipped:  # the per-month clips are not needed once combined
        try:
            os.remove(t)
        except OSError:
            pass
    return result


# ---------------------------------------------------------------------------
# GLiM lithology -> rasterised Sl code grid
# ---------------------------------------------------------------------------

def download_glim_grid(data_dir: str) -> Optional[str]:
    """Download + extract the global GLiM 0.5-degree lithology grid (.asc).

    Returns the path to the ASCII grid, or None if unavailable.
    """
    dest_zip = os.path.join(data_dir, "glim", "glim.zip")
    got = download_file(GLIM_ASCII_URL, dest_zip, timeout=300)
    if not got:
        print("  " + GLIM_SOURCE_INFO)
        return None
    out_dir = os.path.join(data_dir, "glim")
    asc = None
    with zipfile.ZipFile(got) as zf:
        for member in zf.namelist():
            if member.lower().endswith((".asc", ".txt")):
                target = os.path.join(out_dir, os.path.basename(member))
                if not os.path.exists(target):
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                if member.lower().endswith(".asc"):
                    asc = target
    return asc


def download_glim_vector(data_dir: str) -> Optional[str]:
    """Download + extract the full GLiM geodatabase (~1.14 GB).

    Returns the path to the extracted ``.gdb`` directory, or None on failure.
    Safe to re-run: both the download and the extraction are skipped if the
    geodatabase is already present.
    """
    out_dir = os.path.join(data_dir, "glim")
    gdb = os.path.join(out_dir, GLIM_VECTOR_DIRNAME)
    if os.path.isdir(gdb):
        return gdb

    dest_zip = os.path.join(out_dir, "LiMW_GIS_2015.gdb.zip")
    print("  downloading the full GLiM geodatabase (~1.14 GB, one-off)...")
    got = download_file(GLIM_VECTOR_URL, dest_zip, timeout=1800)
    if not got:
        print("  " + GLIM_SOURCE_INFO)
        return None
    with zipfile.ZipFile(got) as zf:
        zf.extractall(out_dir)
    return gdb if os.path.isdir(gdb) else None


def glim_grid_to_codes(asc_path: str, out_path: str) -> str:
    """Convert the GLiM class grid into a lithology-code GeoTIFF.

    Values are mapped GLiM class -> level-1 two-letter code -> the small
    integer in :data:`config.GLIM_CODES`. That mapping is fixed, so a code
    means the same rock type in every area of interest and fitted per-region
    parameters stay comparable between runs.
    """
    import numpy as np
    import rasterio

    from .. import config as C

    with rasterio.open(asc_path) as src:
        arr = src.read(1).astype("float64")
        prof = src.profile.copy()
        nod = src.nodata if src.nodata is not None else -9999.0
        transform = src.transform

    out = np.full(arr.shape, 255, dtype="uint8")
    for value, code in GLIM_VALUE_TO_CODE.items():
        out[arr == value] = C.GLIM_CODES.get(code, 0)
    out[arr == nod] = 255

    prof.update(driver="GTiff", dtype="uint8", nodata=255, count=1,
                compress="deflate", crs=prof.get("crs") or "EPSG:4326",
                transform=transform)
    prof.pop("blockxsize", None)
    prof.pop("blockysize", None)
    prof.pop("tiled", None)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)
    return out_path


def rasterize_glim(glim_path: str, grid, out_path: str,
                   code_field: Optional[str] = None):
    """Rasterise a GLiM vector onto ``grid`` as lithology codes.

    The burned value is the small integer from :data:`config.GLIM_CODES`, which
    is a fixed mapping, so the raster means the same thing in every area of
    interest. Returns ``(path, {code_value: two_letter_code})`` for reporting.
    """
    import numpy as np
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom

    try:
        import fiona
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Reading a GLiM vector needs 'fiona' (pip install fiona). "
            + GLIM_SOURCE_INFO) from exc

    # A .gdb holds one or more layers; pick the first if not specified.
    layer = None
    if os.path.splitext(glim_path)[1].lower() == ".gdb" or \
            os.path.isdir(glim_path):
        layers = fiona.listlayers(glim_path)
        layer = layers[0] if layers else None

    from .. import config as C

    shapes = []
    codes_seen: Dict[int, str] = {}
    with fiona.open(glim_path, layer=layer) as src:
        field = code_field or _guess_glim_field(src.schema["properties"])
        src_crs = src.crs
        # GLiM ships in Eckert IV (ESRI:54012), so the AOI must be projected
        # into the source CRS to filter, and geometries reprojected back.
        same_crs = bool(src_crs) and rasterio.crs.CRS.from_user_input(
            src_crs).to_epsg() == 4326
        if same_crs:
            bbox = (grid.west, grid.south, grid.east, grid.north)
        else:
            from rasterio.warp import transform_bounds
            bbox = transform_bounds("EPSG:4326", src_crs, grid.west,
                                    grid.south, grid.east, grid.north)
        for feat in src.filter(bbox=bbox):
            code = (feat["properties"].get(field) or "nd")
            code = str(code)[:2].lower()
            value = C.GLIM_CODES.get(code, 0)
            codes_seen[value] = code
            geom = feat["geometry"]
            if not same_crs:
                geom = transform_geom(src_crs, "EPSG:4326", geom)
            shapes.append((geom, value))

    prof = grid.profile("uint8", 255)
    arr = rasterize(shapes, out_shape=grid.shape, transform=grid.transform,
                    fill=255, dtype="uint8") if shapes else \
        np.full(grid.shape, 255, dtype="uint8")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr, 1)
    return out_path, codes_seen


def _guess_glim_field(props: Dict[str, str]) -> str:
    for cand in ("xx", "Litho", "litho", "LITHO", "level1", "Level1", "class"):
        if cand in props:
            return cand
    # fall back to first string field
    for name, typ in props.items():
        if str(typ).startswith("str"):
            return name
    raise RuntimeError("could not identify GLiM lithology code field")
