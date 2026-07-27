"""Historical landslide inventory handling for weight calibration.

The calibration needs *presence* points (mapped historical landslides) and
*background* points. This module can:

  * load an inventory from CSV (auto-detecting lat/lon columns) or GeoJSON;
  * restrict it to the South Asia Himalayan region (bounding box + country list);
  * download the NASA Global Landslide Catalog (configurable URL);
  * generate a synthetic Himalayan inventory for fully offline demos/tests;
  * sample factor rasters at points and draw background points.

Primary open source for the region: the NASA Global Landslide Catalog / COOLR
(Kirschbaum et al. 2010; Juang et al. 2019). If the live endpoint is
unreachable, pass a downloaded copy via ``inventory_path``.
"""

from __future__ import annotations

import csv
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Candidate NASA GLC endpoints (may change; override via download_nasa_glc(url=)).
NASA_GLC_URLS = [
    "https://data.nasa.gov/api/views/dd9e-wu2v/rows.csv?accessType=DOWNLOAD",
    "https://maps.nccs.nasa.gov/download/landslide/catalog/nasa_global_landslide_catalog_point.csv",
]
NASA_GLC_INFO = (
    "NASA Global Landslide Catalog / COOLR: browse https://landslides.nasa.gov "
    "and export the point catalogue as CSV or GeoJSON, then pass it via "
    "config.inventory_path (columns latitude/longitude or a GeoJSON of points)."
)

_LAT_KEYS = ("latitude", "lat", "y", "ycoord", "y_coord")
_LON_KEYS = ("longitude", "lon", "long", "lng", "x", "xcoord", "x_coord")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_inventory(path: str,
                   bbox: Optional[Sequence[float]] = None,
                   countries: Optional[Sequence[str]] = None) -> np.ndarray:
    """Load landslide points from CSV/GeoJSON as an (N, 2) [lon, lat] array."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".geojson", ".json"):
        pts = _load_geojson(path)
    else:
        pts = _load_csv(path, countries)
    pts = np.asarray(pts, dtype="float64").reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if bbox is not None and len(pts):
        w, s, e, n = bbox
        m = (pts[:, 0] >= w) & (pts[:, 0] <= e) & \
            (pts[:, 1] >= s) & (pts[:, 1] <= n)
        pts = pts[m]
    return pts


def _load_csv(path: str, countries: Optional[Sequence[str]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    cset = {c.lower() for c in countries} if countries else None
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        lat_c = next((cols[k] for k in _LAT_KEYS if k in cols), None)
        lon_c = next((cols[k] for k in _LON_KEYS if k in cols), None)
        country_c = next((cols[k] for k in ("country_name", "country")
                          if k in cols), None)
        if not lat_c or not lon_c:
            raise ValueError(f"could not find lat/lon columns in {path}; "
                             f"headers were {reader.fieldnames}")
        for row in reader:
            if cset and country_c:
                if str(row.get(country_c, "")).strip().lower() not in cset:
                    continue
            try:
                out.append((float(row[lon_c]), float(row[lat_c])))
            except (TypeError, ValueError):
                continue
    return out


def _load_geojson(path: str) -> List[Tuple[float, float]]:
    with open(path, "r", encoding="utf-8") as fh:
        gj = json.load(fh)
    out: List[Tuple[float, float]] = []
    for feat in gj.get("features", []):
        geom = (feat or {}).get("geometry") or {}
        if geom.get("type") == "Point":
            c = geom.get("coordinates", [])
            if len(c) >= 2:
                out.append((float(c[0]), float(c[1])))
    return out


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def download_nasa_glc(data_dir: str, url: Optional[str] = None) -> Optional[str]:
    """Try to download the NASA Global Landslide Catalog CSV. Returns path/None."""
    from .sources import download_file

    urls = [url] if url else NASA_GLC_URLS
    dest = os.path.join(data_dir, "inventory", "nasa_glc.csv")
    for u in urls:
        try:
            got = download_file(u, dest, retries=2, timeout=120)
            if got and os.path.getsize(got) > 1000:
                return got
        except Exception as exc:  # noqa: BLE001
            print(f"  NASA GLC fetch failed for {u}: {exc}")
    print("  " + NASA_GLC_INFO)
    return None


# ---------------------------------------------------------------------------
# Synthetic inventory (offline demo / tests)
# ---------------------------------------------------------------------------

def make_synthetic_inventory(factor_paths: Sequence[str], n: int,
                             true_weights: Sequence[float],
                             seed: int = 13,
                             candidates: int = 40000) -> np.ndarray:
    """Sample presence points with probability rising with true susceptibility.

    A plausible ground truth is built from the factor rasters using
    ``true_weights`` (exponent form); points are then drawn so that the
    calibration has a real signal to recover. Returns (N, 2) [lon, lat].
    """
    import rasterio

    rng = np.random.default_rng(seed)
    with rasterio.open(factor_paths[0]) as ref:
        H, W = ref.height, ref.width
        transform = ref.transform
    rows = rng.integers(0, H, size=candidates)
    cols = rng.integers(0, W, size=candidates)

    feats = []
    for p in factor_paths:
        with rasterio.open(p) as src:
            band = src.read(1).astype("float64")
            nod = src.nodata
        vals = band[rows, cols]
        vals = np.where(vals == nod, np.nan, vals)
        feats.append(vals)
    feats = np.vstack(feats)  # (4, candidates)

    valid = np.isfinite(feats).all(axis=0)
    # exponent-form true score; flat/water (slope or veg factor 0) -> 0
    logS = np.zeros(candidates)
    for wi, row in zip(true_weights, feats):
        logS += wi * np.log(np.nan_to_num(row, nan=0.0) + 1.0)
    hard0 = (np.nan_to_num(feats[0]) == 0) | (np.nan_to_num(feats[2]) == 0)
    score = np.where(hard0 | ~valid, -np.inf, logS)

    usable = valid & ~hard0
    ref = np.nanmedian(score[usable]) if usable.any() else 0.0
    prob = 1.0 / (1.0 + np.exp(-(score - ref)))
    prob = np.where(usable, prob, 0.0)
    # Acceptance-rejection sampling: keep candidate i with probability prob[i].
    accept = rng.random(candidates) < prob
    idx = np.flatnonzero(accept)
    if len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    xs, ys = rasterio.transform.xy(transform, rows[idx], cols[idx])
    return np.column_stack([np.asarray(xs), np.asarray(ys)])


# ---------------------------------------------------------------------------
# Sampling & background
# ---------------------------------------------------------------------------

def sample_factors_at_points(points: np.ndarray,
                             factor_paths: Sequence[str]) -> np.ndarray:
    """Sample factor rasters at [lon, lat] points -> (N, n_factors) array.

    Cells falling on nodata in any factor yield NaN in that column.
    """
    import rasterio

    n = len(points)
    out = np.full((n, len(factor_paths)), np.nan)
    xy = [(float(x), float(y)) for x, y in points]
    for j, p in enumerate(factor_paths):
        with rasterio.open(p) as src:
            nod = src.nodata
            for i, v in enumerate(src.sample(xy, indexes=1)):
                val = float(v[0])
                out[i, j] = np.nan if (nod is not None and val == nod) else val
    return out


def background_points(bbox: Sequence[float], n: int, reference_raster: str,
                      seed: int = 7) -> np.ndarray:
    """Draw ``n`` random background points that fall on valid raster data."""
    import rasterio

    rng = np.random.default_rng(seed)
    w, s, e, nth = bbox
    with rasterio.open(reference_raster) as src:
        band = src.read(1)
        nod = src.nodata
        transform = src.transform
        H, W = src.height, src.width
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 50:
        tries += 1
        x = rng.uniform(w, e)
        y = rng.uniform(s, nth)
        col, row = ~transform * (x, y)
        r, c = int(row), int(col)
        if 0 <= r < H and 0 <= c < W:
            v = band[r, c]
            if nod is None or v != nod:
                pts.append((x, y))
    return np.asarray(pts, dtype="float64").reshape(-1, 2)
