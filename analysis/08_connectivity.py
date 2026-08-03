"""Does flow-connectivity weighting rank impacted ground better than the cone?

The reach score weights every source in the angle-of-reach cone by 1/distance.
Debris paths, though, are mostly channelised: they follow drainage, which is
why regional runout screens (Flow-R; Horton et al. 2013, NHESS) propagate
sources through a flow-spreading algorithm rather than a cone. The proposed
refinement weights each source additionally by the fraction of its D-infinity
flow routed through the target (Tarboton's dependence), floored so
unchannelised near-field delivery keeps a share.

Physically motivated is not the same as better, and this repository has been
burned before (see 06_calibration_regions: lithology zoning, plausible,
measured at -0.0004 AUC). So the weighting ships **off by default** and this
script is the switch: it measures whether connectivity-weighted reach ranks
ground that landslides actually arrived at above ground they did not.

The test
--------
Whole-landslide polygons include the runout, not just the source - which is
what the polygon inventories here provide (Sikkim maps full polygons;
Far-West ships polygons alongside its points file). For each polygon the
**toe** - its lowest cell - stands in for "a place debris arrived". Background
is drawn from non-landslide cells inside the surveyed extent. Both sets are
scored with the settlement-style reach, cone against connectivity, and the
question is one number each: AUC of toe-vs-background ranking.

Decision rule: enable ``connectivity_weighting`` in the product configs only
if the connectivity AUC exceeds the cone AUC by more than the spread across
resampled background draws. A gain inside the noise is a null result and the
default stays off, exactly as it did for calibration regions.

What this cannot test: volumes, velocities, or whether a specific flagged
path is real. It tests ranking, which is what the screening product claims.

    python analysis/08_connectivity.py                 # sikkim at 30 m
    python analysis/08_connectivity.py farwest 0.00083333
"""

from __future__ import annotations

import os
import sys

import numpy as np
import rasterio
from rasterio.features import rasterize

import common as K

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from h_sim.model import hydrology, risk as R          # noqa: E402
from h_sim.utility.grid import Grid                   # noqa: E402

#: Whole-landslide polygon sources per area. The Far-West points file used by
#: the fitting experiments is derived from these polygons.
POLYGONS = {
    "sikkim": ("data/raw/inventory/sikkim/"
               "Google_Earth_landslides_polygon_21Dec2021.shp"),
    "farwest": ("data/raw/inventory/farwest/"
                "LandslideInventory_FarWesternNepal/"
                "LandslideInventory_FarWesternNepal_Polygons.shp"),
}

N_BACKGROUND_DRAWS = 5      # background resamples, to size the noise
BACKGROUND_RATIO = 2        # background points per toe
SEED = 7


def polygon_toes(shp_path: str, grid: Grid, dem: np.ndarray) -> np.ndarray:
    """(row, col) of the lowest cell of every polygon on the grid."""
    import fiona

    toes = []
    with fiona.open(shp_path) as src:
        for feat in src:
            geom = feat["geometry"]
            if geom is None:
                continue
            mask = rasterize([(geom, 1)], out_shape=grid.shape,
                             transform=grid.transform, fill=0, dtype="uint8")
            rows, cols = np.nonzero(mask)
            if rows.size == 0:
                continue
            z = dem[rows, cols]
            ok = np.isfinite(z)
            if not ok.any():
                continue
            k = int(np.argmin(np.where(ok, z, np.inf)))
            toes.append((int(rows[k]), int(cols[k])))
    return np.array(toes, dtype=int)


def reach_scores(index: R.ReachIndex, prob: np.ndarray, grid: Grid,
                 cells: np.ndarray) -> np.ndarray:
    out = np.empty(len(cells))
    for i, (r, c) in enumerate(cells):
        lon, lat = grid.transform * (c + 0.5, r + 0.5)
        s = index.score_point(prob, lon, lat)
        out[i] = s.score if s is not None else np.nan
    return out


def main(area: str = "sikkim", res: float = 0.00027778) -> None:
    shp = POLYGONS[area]
    if not os.path.exists(shp):
        raise SystemExit(f"{shp} not found - run step2-download first, and "
                         "check the polygon shapefile name in POLYGONS.")

    cfg = K.make_config(area, res, suffix="_conn")
    terrain = K.terrain_layers(cfg, want_precip=True)
    # Score against the susceptibility the standard pipeline produces.
    from h_sim import pipeline
    base = pipeline.run_susceptibility(cfg)
    with rasterio.open(base["probability"]) as src:
        prob = src.read(1).astype("float64")
        prob[prob == src.nodata] = np.nan

    grid = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
    with rasterio.open(terrain["dem"]) as src:
        dem = src.read(1).astype("float64")
        dem[dem == src.nodata] = np.nan

    dx, dy = pipeline.metres_per_cell(cfg.clipped_bbox(), cfg.resolution_deg)

    print(f"[08] rasterising polygons and locating toes ({area})")
    toes = polygon_toes(shp, grid, dem)
    print(f"[08] {len(toes)} polygon toes")

    # Landslide mask, to keep background off mapped ground.
    import fiona
    with fiona.open(shp) as src:
        geoms = [f["geometry"] for f in src if f["geometry"]]
    slide_mask = rasterize([(g, 1) for g in geoms], out_shape=grid.shape,
                           transform=grid.transform, fill=0,
                           dtype="uint8").astype(bool)
    extent = K.survey_mask(area, grid,
                           os.path.join(cfg.work_dir, f"{cfg.name}_ext.tif"))
    if extent:
        with rasterio.open(extent) as src:
            inside = src.read(1).astype(bool)
    else:
        inside = np.isfinite(dem)
    candidates = np.argwhere(inside & ~slide_mask & np.isfinite(dem))

    print("[08] building the two indices")
    cone = R.ReachIndex(dem, grid.transform, dx, dy)
    filled = hydrology.fill_depressions(dem)
    ang, _ = hydrology.dinf_flow_direction(filled, dx, dy)
    conn = R.ReachIndex(dem, grid.transform, dx, dy, flow=(filled, ang))

    rng = np.random.default_rng(SEED)
    rows, draws = [], {"cone": [], "connectivity": []}
    s_toe = {"cone": reach_scores(cone, prob, grid, toes),
             "connectivity": reach_scores(conn, prob, grid, toes)}
    for d in range(N_BACKGROUND_DRAWS):
        bg = candidates[rng.choice(len(candidates),
                                   BACKGROUND_RATIO * len(toes),
                                   replace=False)]
        for name, index in (("cone", cone), ("connectivity", conn)):
            s_bg = reach_scores(index, prob, grid, bg)
            s = np.concatenate([s_toe[name], s_bg])
            y = np.concatenate([np.ones(len(toes)), np.zeros(len(bg))])
            ok = np.isfinite(s)
            draws[name].append(K.auc(s[ok], y[ok]))

    for name in ("cone", "connectivity"):
        rows.append(K.summarise(name, draws[name]))
    gain = rows[1]["auc_mean"] - rows[0]["auc_mean"]
    noise = max(rows[0]["auc_std"], rows[1]["auc_std"])

    print(K.table(rows, [("model", "model", "s"),
                         ("AUC", "auc_mean", ".4f"),
                         ("+/-", "auc_std", ".4f")]))
    verdict = ("ADOPT: gain exceeds the draw noise"
               if gain > noise else
               "NULL: gain within noise - leave connectivity_weighting off")
    print(f"[08] gain {gain:+.4f} against noise {noise:.4f} -> {verdict}")

    K.save("08_connectivity", {
        "area": area, "resolution_deg": res, "n_toes": int(len(toes)),
        "background_ratio": BACKGROUND_RATIO, "draws": N_BACKGROUND_DRAWS,
        "rows": rows, "gain": round(gain, 4), "noise": round(noise, 4),
        "verdict": verdict,
    })


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0] if args else "sikkim",
         float(args[1]) if len(args) > 1 else 0.00027778)
