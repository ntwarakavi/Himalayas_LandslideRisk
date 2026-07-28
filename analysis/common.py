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

from giri_landslide import config as C          # noqa: E402
from giri_landslide import pipeline             # noqa: E402
from giri_landslide.input import inventory, sources  # noqa: E402
from giri_landslide.model import crossval, physical as P  # noqa: E402
from giri_landslide.utility.grid import Grid    # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ROBACK = ("data/raw/inventory/roback/Roback_Nepal_final_files/"
          "Source20170209.shp")
FARWEST = ("data/raw/inventory/farwest/LandslideInventory_FarWesternNepal/"
           "LandslideInventory_FarWesternNepal_Points.shp")
SIKKIM = ("data/raw/inventory/sikkim/"
          "Google_Earth_landslides_polygon_21Dec2021.shp")

#: Study areas. Gorkha is the 2015 earthquake footprint; Far-West Nepal is a
#: monsoon-driven multi-temporal inventory; Sikkim is small and used only to
#: test transfer.
AREAS = {
    "gorkha":  dict(bbox=(84.5, 27.6, 85.3, 28.2), inventory=ROBACK),
    "farwest": dict(bbox=(80.3, 28.8, 81.4, 30.0), inventory=FARWEST),
    "sikkim":  dict(bbox=(88.2, 27.1, 88.8, 27.6), inventory=SIKKIM),
}


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
           n_background: Optional[int] = None, seed: int = 0
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Presence and background points with their sampled layer values.

    Returns ``(points_pres, values_pres, points_bg, values_bg)``. Background is
    drawn over the same extent and screened by the slope raster, so it stands
    in for "terrain that did not fail" rather than for "anywhere on Earth".
    """
    bbox = cfg.clipped_bbox()
    pres = inventory.load_inventory(cfg.inventory_path, bbox=bbox)
    n_bg = n_background or max(2 * len(pres), 2000)
    bg = inventory.background_points(bbox, n_bg, layers[0], seed=seed)
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
