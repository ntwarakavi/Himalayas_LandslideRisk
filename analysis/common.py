"""Shared plumbing for the experiments in this directory.

Each script here answers one question about the model and prints a table. They
share the same sampling and fold logic so their numbers are comparable, which
is the whole point: every comparison below holds the presence points, the
background points and the spatial folds fixed, and varies one thing.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hima_slide import config as C                    # noqa: E402
from hima_slide import pipeline                       # noqa: E402
from hima_slide.input import inventory, sources       # noqa: E402
from hima_slide.model import crossval, physical as P  # noqa: E402
from hima_slide.utility.grid import Grid              # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ROBACK = ("data/raw/inventory/roback/Roback_Nepal_final_files/"
          "Source20170209.shp")
FARWEST = ("data/raw/inventory/farwest/LandslideInventory_FarWesternNepal/"
           "LandslideInventory_FarWesternNepal_Points.shp")
SIKKIM = ("data/raw/inventory/sikkim/"
          "Google_Earth_landslides_polygon_21Dec2021.shp")

#: Polygons delimiting the ground each inventory actually surveyed. All three
#: inventories publish one, and using them is not optional: background points
#: stand in for "terrain that did not fail", so drawing them outside the
#: surveyed area silently labels unmapped ground as landslide-free. Far-West
#: and Sikkim survey only about 60 per cent of their own bounding boxes, so
#: unmasked background would be roughly 40 per cent false negatives.
EXTENTS = {
    "gorkha": ("data/raw/inventory/roback/Roback_Nepal_final_files/"
               "MappingExtent20170209.shp"),
    "farwest": ("data/raw/inventory/farwest/LandslideInventory_FarWesternNepal/"
                "LandslideInventory_FarWesternNepal_AOI.shp"),
    "sikkim": ("data/raw/inventory/sikkim/"
               "Google_Earth_mapped_extent_21Dec2021.shp"),
}

#: Study areas. Gorkha is the 2015 earthquake footprint; Far-West Nepal is a
#: monsoon-driven multi-temporal inventory; Sikkim is small and used only to
#: test transfer. Each bounding box is the surveyed extent's own bounds, so no
#: part of a run falls outside ground somebody looked at.
AREAS = {
    "gorkha":  dict(bbox=(84.5, 27.6, 85.3, 28.2), inventory=ROBACK),
    "farwest": dict(bbox=(80.558, 28.913, 81.592, 29.856), inventory=FARWEST),
    "sikkim":  dict(bbox=(88.048, 27.067, 88.917, 27.554), inventory=SIKKIM),
}


def survey_mask(area: str, grid: Grid, out_path: str) -> Optional[str]:
    """Rasterise the surveyed-extent polygon onto ``grid``.

    Returns the path to a 0/1 mask, or None if the area has no extent polygon.
    """
    import fiona
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom

    shp = EXTENTS.get(area)
    if not shp or not os.path.exists(shp):
        return None
    with fiona.open(shp) as src:
        crs = src.crs or "EPSG:4326"
        shapes = [(transform_geom(crs, "EPSG:4326", f["geometry"]), 1)
                  for f in src]
    arr = rasterize(shapes, out_shape=grid.shape, transform=grid.transform,
                    fill=0, dtype="uint8")
    with rasterio.open(out_path, "w", **grid.profile("uint8", 0)) as dst:
        dst.write(arr, 1)
    return out_path


def masked_reference(area: str, cfg: C.Config, slope_path: str) -> str:
    """A copy of the slope raster blanked outside the surveyed extent.

    Passed to :func:`background_points` as the reference, so background falls
    only on ground the inventory's authors actually examined.
    """
    grid = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
    mask_path = os.path.join(cfg.work_dir, f"{cfg.name}_surveyed.tif")
    if survey_mask(area, grid, mask_path) is None:
        return slope_path

    out = os.path.join(cfg.work_dir, f"{cfg.name}_slope_surveyed.tif")
    with rasterio.open(slope_path) as s, rasterio.open(mask_path) as m:
        a = s.read(1).astype("float32")
        a[m.read(1) == 0] = -9999.0
    with rasterio.open(out, "w", **grid.profile("float32", -9999.0)) as dst:
        dst.write(a, 1)
    frac = float((a != -9999.0).mean())
    print(f"  surveyed extent covers {frac * 100:.1f}% of the AOI")
    return out


def save(name: str, payload: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\n  -> {path}")
    return path


def work_name(area: str, res: float, suffix: str = "") -> str:
    """Canonical run name for an (area, resolution) pair.

    Every script uses the same name so they share one cached copy of the
    terrain rasters. Flow routing over Far-West Nepal at 30 m is 17 million
    cells; computing it once rather than once per experiment is the difference
    between minutes and an hour.
    """
    tag = f"{res:.8f}".rstrip("0").replace(".", "")
    return f"exp_{area}_{tag}{suffix}"


def make_config(area: str, res: float, suffix: str = "", **kw) -> C.Config:
    a = AREAS[area]
    return C.Config(name=work_name(area, res, suffix), bbox=a["bbox"],
                    resolution_deg=res, inventory_path=a["inventory"],
                    dem_source="copernicus30", **kw)


def terrain_layers(cfg: C.Config, want_precip: bool = True) -> Dict[str, str]:
    """Build (or reuse) the rasters an experiment samples from."""
    grid = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
    inputs = pipeline.resolve_inputs(cfg, "download")
    out = dict(pipeline.stage_terrain(cfg, grid, inputs))
    if want_precip:
        path, ref = pipeline.stage_recharge(cfg, grid, inputs)
        out["recharge"] = path
        out["recharge_reference_mm"] = ref
    return out


def region_layer(area: str, res: float, kind: str) -> Optional[str]:
    """Build just a calibration-region raster for an (area, resolution).

    Deliberately separate from :func:`terrain_layers`: a region raster is a
    warp or a rasterisation and takes seconds, while the terrain it would
    otherwise be bundled with is flow routing over millions of cells. Asking
    for lithology should not re-route the DEM.
    """
    cfg = make_config(area, res, suffix=f"_{kind}", calibration_regions=kind)
    grid = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
    inputs = pipeline.resolve_inputs(cfg, "download")
    return pipeline.stage_regions(cfg, grid, inputs)


def sample(cfg: C.Config, layers: Sequence[str],
           n_background: Optional[int] = None, seed: int = 0,
           area: Optional[str] = None
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Presence and background points with their sampled layer values.

    Returns ``(points_pres, values_pres, points_bg, values_bg)``. Background is
    drawn over the surveyed extent and screened by the slope raster, so it
    stands in for "terrain that was looked at and did not fail" rather than for
    "anywhere on Earth" - or, worse, "anywhere nobody checked".
    """
    bbox = cfg.clipped_bbox()
    pres = inventory.load_inventory(cfg.inventory_path, bbox=bbox)
    n_bg = n_background or max(2 * len(pres), 2000)
    reference = (masked_reference(area, cfg, layers[0]) if area
                 else layers[0])
    bg = inventory.background_points(bbox, n_bg, reference, seed=seed)
    vp = inventory.sample_factors_at_points(pres, list(layers))
    vb = inventory.sample_factors_at_points(bg, list(layers))
    vp[vp == -9999.0] = np.nan
    vb[vb == -9999.0] = np.nan
    return pres, vp, bg, vb


def clean(*arrays: np.ndarray) -> np.ndarray:
    """Row mask where every supplied array is finite."""
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        a = np.asarray(a)
        ok &= np.isfinite(a).all(axis=1) if a.ndim > 1 else np.isfinite(a)
    return ok


def folds(points_p: np.ndarray, points_b: np.ndarray, bbox, scheme: str,
          n_folds: int = 5, block_deg: float = 0.25, seed: int = 0):
    """One fold assignment, shared by every model in a comparison."""
    if scheme == "spatial":
        return (crossval.spatial_block_folds(points_p, bbox, n_folds,
                                             block_deg, seed),
                crossval.spatial_block_folds(points_b, bbox, n_folds,
                                             block_deg, seed))
    return (crossval.random_folds(len(points_p), n_folds, seed),
            crossval.random_folds(len(points_b), n_folds, seed))


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    return P._auc(np.asarray(scores, "float64"), np.asarray(y, "float64"))


def capture(scores: np.ndarray, y: np.ndarray,
            fractions=(0.05, 0.10, 0.20)) -> Dict[str, float]:
    """Share of landslides falling in the worst-ranked fraction of background.

    Thresholds come from the background distribution, which stands in for map
    area, so this reads as "what fraction of failures does the worst X% of
    terrain capture".
    """
    s, y = np.asarray(scores, "float64"), np.asarray(y)
    bg = s[y == 0]
    pos = s[y == 1]
    out = {}
    for f in fractions:
        thr = np.quantile(bg, 1.0 - f)
        out[f"capture_top{int(f * 100)}pct"] = round(
            float((pos >= thr).mean() * 100), 2)
    return out


def summarise(name: str, aucs: List[float]) -> dict:
    a = np.asarray(aucs, "float64")
    return {"model": name, "auc_mean": round(float(a.mean()), 4),
            "auc_std": round(float(a.std()), 4),
            "auc_folds": [round(float(x), 4) for x in a],
            "n_folds": len(a)}


def table(rows: List[dict], cols: Sequence[Tuple[str, str, str]]) -> str:
    """Fixed-width table. ``cols`` is (key, header, format)."""
    head = "  " + "  ".join(f"{h:>{max(len(h), 9)}}" for _, h, _ in cols)
    lines = [head, "  " + "-" * (len(head) - 2)]
    for r in rows:
        cells = []
        for key, h, fmt in cols:
            v = r.get(key)
            w = max(len(h), 9)
            cells.append(f"{'':>{w}}" if v is None else
                         f"{format(v, fmt):>{w}}")
        lines.append("  " + "  ".join(cells))
    return "\n".join(lines)
