"""Historical landslide inventory handling for weight calibration.

The calibration needs *presence* points (mapped historical landslides) and
*background* points. This module can:

  * load an inventory from CSV (auto-detecting lat/lon columns) or GeoJSON;
  * restrict it to the South Asia Himalayan region (bounding box + country list);
  * download the NASA Global Landslide Catalog (configurable URL);
  * generate a synthetic Himalayan inventory for fully offline demos/tests;
  * sample factor rasters at points and draw background points.

Inventory sources for the HKH, in rough order of calibration quality:

  * Roback et al. (2018) Gorkha earthquake inventory - 24,795 satellite-mapped
    source areas across Nepal. USGS ScienceBase item 582c74fbe4b04d580bd377e8.
  * Multi-temporal Far-Western Nepal inventory - 26,350 polygons, monsoon
    triggered. Zenodo doi:10.5281/zenodo.4290100.
  * ICIMOD Regional Database System (rds.icimod.org) - HKH-wide, free account.
  * NASA Global Landslide Catalog / COOLR (Kirschbaum et al. 2010; Juang et al.
    2019) - downloaded automatically, but note it is compiled from media
    reports, so it carries a strong spatial reporting bias towards roads and
    settlements. Useful for visual validation; weak for calibration. See
    docs/RUNNING_LOCALLY.md for a measured comparison.

Any of these can be passed via ``inventory_path`` as CSV, GeoJSON or shapefile.
"""

from __future__ import annotations

import csv
import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

# NASA COOLR (Cooperative Open Online Landslide Repository) point catalogue,
# served as an ArcGIS FeatureServer that supports GeoJSON queries + pagination.
COOLR_POINTS_URL = (
    "https://gis.earthdata.nasa.gov/gis05/rest/services/Landslides/"
    "COOLR_Events_Points/FeatureServer/0/query"
)
NASA_GLC_INFO = (
    "NASA Global Landslide Catalog / COOLR: browse https://landslides.nasa.gov"
    "/viewer and export the point catalogue as CSV/GeoJSON, or download it "
    "directly from the ArcGIS FeatureServer:\n  " + COOLR_POINTS_URL +
    "?where=1=1&outFields=*&f=geojson\nThen pass it via config.inventory_path."
)

_LAT_KEYS = ("latitude", "lat", "y", "ycoord", "y_coord")
_LON_KEYS = ("longitude", "lon", "long", "lng", "x", "xcoord", "x_coord")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_inventory(path: str,
                   bbox: Optional[Sequence[float]] = None,
                   countries: Optional[Sequence[str]] = None) -> np.ndarray:
    """Load landslide locations as an (N, 2) [lon, lat] array.

    Accepts CSV (lat/lon columns), GeoJSON, or an ESRI shapefile / geodatabase.
    Polygon inventories - the norm for satellite-mapped landslides - are reduced
    to one representative point per feature, and any projected CRS is
    reprojected to EPSG:4326.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".geojson", ".json"):
        pts = _load_geojson(path)
    elif ext in (".shp", ".gdb", ".gpkg"):
        pts = _load_vector(path)
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


def _load_vector(path: str, layer: Optional[str] = None
                 ) -> List[Tuple[float, float]]:
    """Read a shapefile/geodatabase of landslide points or polygons.

    Each feature is reduced to a single representative location: points are
    used as-is, polygons and lines are reduced to their coordinate centroid.
    Features are reprojected to EPSG:4326 when the source CRS differs.
    """
    try:
        import fiona
        from fiona.transform import transform_geom
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "reading a shapefile inventory needs 'fiona' (pip install fiona)"
        ) from exc

    out: List[Tuple[float, float]] = []
    with fiona.open(path, layer=layer) as src:
        src_crs = src.crs
        for feat in src:
            geom = feat.get("geometry")
            if not geom:
                continue
            if src_crs and str(src_crs).upper() not in ("EPSG:4326",
                                                        "OGC:CRS84"):
                geom = transform_geom(src_crs, "EPSG:4326", geom)
            xy = _representative_point(geom)
            if xy:
                out.append(xy)
    return out


def _representative_point(geom: dict) -> Optional[Tuple[float, float]]:
    """Centroid of a geometry's coordinates (no shapely dependency)."""
    coords: List[Tuple[float, float]] = []

    def walk(c):
        if not c:
            return
        if isinstance(c[0], (int, float)):
            coords.append((float(c[0]), float(c[1])))
        else:
            for part in c:
                walk(part)

    gtype = geom.get("type")
    if gtype == "Point":
        c = geom.get("coordinates") or []
        return (float(c[0]), float(c[1])) if len(c) >= 2 else None
    walk(geom.get("coordinates"))
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


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

def download_coolr_points(data_dir: str,
                          bbox: Optional[Sequence[float]] = None,
                          page: int = 1000, max_records: int = 100000
                          ) -> Optional[str]:
    """Download NASA COOLR landslide points as GeoJSON via the FeatureServer.

    Only records intersecting ``bbox`` are requested (server-side), and results
    are paginated. Returns the written GeoJSON path, or None on failure.
    """
    import requests

    params = {"where": "1=1", "outFields": "latitude,longitude,country_name,"
              "event_date,landslide_category,landslide_trigger",
              "outSR": "4326", "f": "geojson"}
    if bbox is not None:
        w, s, e, n = bbox
        params.update(geometry=f"{w},{s},{e},{n}",
                      geometryType="esriGeometryEnvelope", inSR="4326",
                      spatialRel="esriSpatialRelIntersects")

    features: List[dict] = []
    offset = 0
    try:
        while offset < max_records:
            q = dict(params, resultOffset=offset, resultRecordCount=page)
            r = requests.get(COOLR_POINTS_URL, params=q, timeout=120)
            r.raise_for_status()
            batch = r.json().get("features", [])
            if not batch:
                break
            features.extend(batch)
            if len(batch) < page:
                break
            offset += page
    except Exception as exc:  # noqa: BLE001
        print(f"  COOLR fetch failed: {exc}")
        return None

    if not features:
        return None
    os.makedirs(os.path.join(data_dir, "inventory"), exist_ok=True)
    dest = os.path.join(data_dir, "inventory", "coolr_points.geojson")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh)
    print(f"  COOLR: {len(features)} landslide points -> {dest}")
    return dest


def download_nasa_glc(data_dir: str,
                      bbox: Optional[Sequence[float]] = None) -> Optional[str]:
    """Obtain the NASA COOLR landslide inventory for ``bbox``.

    Returns a path to the downloaded GeoJSON, or None (with a pointer printed).
    """
    got = download_coolr_points(data_dir, bbox=bbox)
    if got:
        return got
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
                      seed: int = 7, near: Optional[np.ndarray] = None,
                      radius_deg: float = 0.15) -> np.ndarray:
    """Draw ``n`` random background points on valid raster data.

    If ``near`` (an array of presence points) is given, points are drawn within
    ``radius_deg`` of a random presence point ("target-group" / density-matched
    background). This controls the spatial reporting bias of citizen-science
    inventories: landslides cluster in accessible valleys, so background drawn
    uniformly across a steep AOI would be systematically steeper than presence
    and the calibration would learn a spurious negative slope effect. Matching
    the background's spatial density to presence removes that bias.
    """
    import rasterio

    rng = np.random.default_rng(seed)
    w, s, e, nth = bbox
    with rasterio.open(reference_raster) as src:
        band = src.read(1)
        nod = src.nodata
        transform = src.transform
        H, W = src.height, src.width
    have_near = near is not None and len(near) > 0
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 100:
        tries += 1
        if have_near:
            cx, cy = near[rng.integers(len(near))]
            x = cx + rng.uniform(-radius_deg, radius_deg)
            y = cy + rng.uniform(-radius_deg, radius_deg)
            if not (w <= x <= e and s <= y <= nth):
                continue
        else:
            x = rng.uniform(w, e)
            y = rng.uniform(s, nth)
        col, row = ~transform * (x, y)
        r, c = int(row), int(col)
        if 0 <= r < H and 0 <= c < W:
            v = band[r, c]
            if nod is None or v != nod:
                pts.append((x, y))
    return np.asarray(pts, dtype="float64").reshape(-1, 2)
