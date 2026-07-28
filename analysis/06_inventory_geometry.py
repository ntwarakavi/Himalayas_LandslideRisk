"""Where on a landslide should the model be scored?

An infinite-slope model is a statement about *initiation*: it predicts where a
soil column stops holding, not where the resulting debris comes to rest. So the
point at which an inventory is sampled matters, and inventories differ in what
they record.

The Roback Gorkha inventory ships mapped *source areas*, whose centroid sits in
the initiation zone. The Far-Western Nepal inventory ships whole-landslide
polygons, whose centroid sits somewhere down the runout path - on gentler,
more convergent ground than the scar it came from. Scoring the second the way
you score the first asks the model to predict a location it never claimed to.

This tests that directly, on the same polygons, by comparing three sampling
conventions:

* **centroid** - the coordinate centroid, which is what the package does by
  default and what the earlier results used.
* **crown** - the highest-elevation vertex of the polygon, a cheap proxy for
  the head scarp where failure begins.
* **upper quartile** - the mean of the vertices in the top quarter of the
  polygon's elevation range, which is less sensitive to a single stray vertex
  than the crown.

If the model really is about initiation, the crown should score best, and the
gap is the cost of sampling an inventory in the wrong place.

    python analysis/06_inventory_geometry.py
"""

from __future__ import annotations

import os

import numpy as np
import rasterio

import common as K

RES = 0.00027778          # 30 m

POLYGONS = {
    "farwest": ("data/raw/inventory/farwest/LandslideInventory_FarWesternNepal/"
                "LandslideInventory_FarWesternNepal_Pol.shp"),
    "gorkha": ("data/raw/inventory/roback/Roback_Nepal_final_files/"
               "Source20170209.shp"),
}


def polygon_points(shp: str, dem_path: str, bbox) -> dict:
    """Centroid, crown and upper-quartile point for every polygon in ``shp``."""
    import fiona
    from rasterio.warp import transform_geom

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float64")
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        transform, height, width = src.transform, src.height, src.width
    inv = ~transform

    def elev(xs, ys):
        cols, rows = inv * (np.asarray(xs), np.asarray(ys))
        cols = np.clip(cols.astype(int), 0, width - 1)
        rows = np.clip(rows.astype(int), 0, height - 1)
        return dem[rows, cols]

    out = {"centroid": [], "crown": [], "upper_quartile": []}
    w, s, e, n = bbox
    with fiona.open(shp) as src:
        crs = src.crs or "EPSG:4326"
        # Some of these files carry no CRS and some carry a projected one, so
        # compare by EPSG code rather than by the string spelling of it.
        try:
            same = rasterio.crs.CRS.from_user_input(crs).to_epsg() == 4326
        except Exception:                                 # noqa: BLE001
            same = True
        for feat in src:
            geom = feat["geometry"]
            if geom is None:                     # null geometries do occur
                continue
            if not same:
                geom = transform_geom(crs, "EPSG:4326", geom)
            rings = geom.get("coordinates") or []
            if geom["type"] == "MultiPolygon":
                rings = [r for poly in rings for r in poly]
            coords = [c for ring in rings for c in ring]
            if len(coords) < 3:
                continue
            xs = np.array([c[0] for c in coords], dtype="float64")
            ys = np.array([c[1] for c in coords], dtype="float64")
            cx, cy = float(xs.mean()), float(ys.mean())
            if not (w <= cx <= e and s <= cy <= n):
                continue

            z = elev(xs, ys)
            if not np.isfinite(z).any():
                continue
            out["centroid"].append((cx, cy))
            k = int(np.nanargmax(z))
            out["crown"].append((float(xs[k]), float(ys[k])))
            cut = np.nanpercentile(z, 75)
            m = np.isfinite(z) & (z >= cut)
            out["upper_quartile"].append((float(xs[m].mean()),
                                          float(ys[m].mean())))
    return {k: np.asarray(v) for k, v in out.items()}


def main() -> None:
    rows = []
    for area, shp in POLYGONS.items():
        if not os.path.exists(shp):
            print(f"  {area}: {shp} not found, skipping")
            continue
        cfg = K.make_config(area, RES)
        bbox = cfg.clipped_bbox()
        print(f"\n=== {area} ===")
        layers = K.terrain_layers(cfg, want_precip=True)
        paths = [layers["slope"], layers["sca"], layers["recharge"]]

        pts = polygon_points(shp, layers["dem"], bbox)
        print(f"  {len(pts['centroid'])} polygons inside the AOI")

        # One background sample, shared by all three conventions.
        ref = K.masked_reference(area, cfg, layers[0] if isinstance(layers, list)
                                 else layers["slope"])
        n_bg = max(2 * len(pts["centroid"]), 2000)
        bg = K.inventory.background_points(bbox, n_bg, ref, seed=0)
        vb = K.inventory.sample_factors_at_points(bg, paths)
        vb[vb == -9999.0] = np.nan
        ok_b = K.clean(vb)
        bg, vb = bg[ok_b], vb[ok_b]

        for how in ("centroid", "upper_quartile", "crown"):
            p = pts[how]
            vp = K.inventory.sample_factors_at_points(p, paths)
            vp[vp == -9999.0] = np.nan
            ok_p = K.clean(vp)
            p, vpc = p[ok_p], vp[ok_p]

            fit = K.P.fit_parameters(vpc[:, 0], vpc[:, 1], vb[:, 0], vb[:, 1],
                                     n_samples=60, recharge_pres=vpc[:, 2],
                                     recharge_bg=vb[:, 2])
            cv = K.P.cross_validate(p, vpc[:, 0], vpc[:, 1], bg, vb[:, 0],
                                    vb[:, 1], bbox, scheme="spatial",
                                    n_folds=5, block_deg=0.25, n_samples=60,
                                    recharge_pres=vpc[:, 2],
                                    recharge_bg=vb[:, 2])
            row = {"area": area, "sampled_at": how, "n_presence": int(len(p)),
                   "median_slope_deg": round(float(np.degrees(np.arctan(
                       np.median(vpc[:, 0])))), 2),
                   "median_sca_m": round(float(np.median(vpc[:, 1])), 1),
                   "in_sample_auc": round(fit["auc"], 4),
                   "cv_spatial_mean": round(cv["auc_mean"], 4),
                   "cv_spatial_std": round(cv["auc_std"], 4)}
            rows.append(row)
            print(f"  {how:15s} AUC {fit['auc']:.4f}  "
                  f"spatialCV {cv['auc_mean']:.4f} +/- {cv['auc_std']:.4f}  "
                  f"median slope {row['median_slope_deg']:.1f} deg")

    print("\n\nWHERE THE INVENTORY IS SAMPLED  (30 m)\n")
    print(K.table(rows, [
        ("area", "area", "s"),
        ("sampled_at", "sampled at", "s"),
        ("n_presence", "slides", "d"),
        ("median_slope_deg", "slope deg", ".2f"),
        ("median_sca_m", "SCA m", ".1f"),
        ("in_sample_auc", "in-samp", ".4f"),
        ("cv_spatial_mean", "spatialCV", ".4f"),
    ]))

    K.save("06_inventory_geometry", {"rows": rows, "resolution_deg": RES})


if __name__ == "__main__":
    main()
