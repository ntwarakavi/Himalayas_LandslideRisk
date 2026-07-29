"""Can imprecise point catalogues be used at all?

The NASA Global Landslide Catalog holds 2,469 records inside the Hindu Kush
Himalaya, 2,099 of them rainfall-triggered - the mechanism SINMAP describes and
far more records than the three polygon inventories put together. They are
unusable *as points* because 85 per cent are placed worse than 1 km, and a
90 m pixel cannot be tested against a position known to a district.

Positional error is not a reason to discard data, though. It is a reason to
test at the scale the data supports. Two methods do that, and this script
measures whether either produces signal.

**Neighbourhood sampling.** Score each record by the susceptibility within a
disc of radius equal to its own stated accuracy, and score background points
through discs drawn from the same radius distribution. The comparison is then
fair: both sides are blurred identically, and the AUC answers "is the
neighbourhood of a reported landslide more susceptible than the neighbourhood
of a random place", which is a real question at the scale the data supports.

The statistic matters. A maximum over a 25 km disc saturates for the same
reason the reaching-susceptibility score did - take enough cells and one of
them is always unstable - so the mean and the 90th percentile are measured too.

**Areal density.** Forget positions entirely. Bin records to a grid coarser
than their error, and test whether landslide *count* per cell rises with mean
predicted susceptibility. A 5 km error does not matter at 25 km support. This
uses every record, including the ones no neighbourhood method can rescue.

Both are honest tests of a coarse claim. Neither can validate a 90 m map at
90 m, and nothing here should be quoted as if it did.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import common  # noqa: E402
from h_sim import config as C  # noqa: E402
from h_sim.model.physical import _auc  # noqa: E402

#: Stated accuracy class -> radius in metres. "exact" is treated as one 90 m
#: cell rather than zero, since even a surveyed point is a polygon somewhere.
ACCURACY_M = {"exact": 100.0, "1km": 1000.0, "5km": 5000.0, "10km": 10000.0,
              "25km": 25000.0, "50km": 50000.0}

#: Records worse than this are hopeless for neighbourhood sampling: the disc
#: covers so much ground that every statistic converges on the map average.
MAX_NEIGHBOURHOOD_M = 10000.0


def load_glc(path, bbox, triggers=None):
    """GLC records inside ``bbox`` as (lon, lat, radius_m)."""
    trig = {t.lower() for t in triggers} if triggers else None
    w, s, e, n = bbox
    out = []
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            acc = (row.get("location_accuracy") or "").strip()
            if acc not in ACCURACY_M:
                continue
            if trig and (row.get("landslide_trigger") or "").strip().lower() \
                    not in trig:
                continue
            try:
                lon, lat = float(row["longitude"]), float(row["latitude"])
            except (TypeError, ValueError, KeyError):
                continue
            if w <= lon <= e and s <= lat <= n:
                out.append((lon, lat, ACCURACY_M[acc]))
    return out


def _auc_pb(pos, neg):
    """AUC of presence against background."""
    scores = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(_auc(scores, y))


def disc_stats(src, band, lon, lat, radius_m):
    """mean, 90th percentile and max of the raster within a disc."""
    tr = src.transform
    dy_m = abs(tr.e) * 110540.0
    dx_m = abs(tr.a) * 111320.0 * np.cos(np.radians(lat))
    ry = max(int(round(radius_m / dy_m)), 0)
    rx = max(int(round(radius_m / dx_m)), 0)
    col, row = ~tr * (lon, lat)
    r0, c0 = int(row), int(col)
    h, w = band.shape
    r1, r2 = max(r0 - ry, 0), min(r0 + ry + 1, h)
    c1, c2 = max(c0 - rx, 0), min(c0 + rx + 1, w)
    if r1 >= r2 or c1 >= c2:
        return None
    win = band[r1:r2, c1:c2]
    ok = np.isfinite(win)
    if not ok.any():
        return None
    v = win[ok]
    return float(v.mean()), float(np.percentile(v, 90)), float(v.max())


def load_target_group(prob_path):
    """Settlements scored for this map, if step 7 has run over it.

    A media-derived catalogue reports landslides where people are to see them.
    Background drawn uniformly is therefore drawn from different ground than
    presence, and the comparison measures reporting rather than stability.
    Drawing background near the same settlements matches the two, which is the
    standard target-group correction for presence-only data.
    """
    import json

    path = prob_path.replace("_susceptibility_prob.tif",
                             "_risk_settlements.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    return np.array([[r["lon"], r["lat"]] for r in rows]) if rows else None


def neighbourhood_test(prob_path, presence, rng, n_background=4,
                       target_group=None, spread_deg=0.15, repeats=1):
    """AUC with presence and background blurred through identical discs.

    With ``target_group``, background is drawn around those locations rather
    than uniformly, so both sides carry the same reporting bias.
    """
    with rasterio.open(prob_path) as src:
        band = src.read(1).astype("float64")
        if src.nodata is not None:
            band[band == src.nodata] = np.nan
        b = src.bounds

        pres = [(lon, lat, r) for lon, lat, r in presence
                if r <= MAX_NEIGHBOURHOOD_M]
        if not pres:
            return None
        radii = np.array([r for _, _, r in pres])

        p_stats, bg_stats = [], []
        for lon, lat, r in pres:
            st = disc_stats(src, band, lon, lat, r)
            if st:
                p_stats.append(st)
        # background: same radius distribution, uniformly placed on valid data
        want = n_background * len(p_stats)
        tries = 0
        while len(bg_stats) < want and tries < want * 40:
            tries += 1
            if target_group is not None and len(target_group):
                cx, cy = target_group[rng.integers(len(target_group))]
                lon = cx + rng.uniform(-spread_deg, spread_deg)
                lat = cy + rng.uniform(-spread_deg, spread_deg)
                if not (b.left <= lon <= b.right and b.bottom <= lat <= b.top):
                    continue
            else:
                lon = rng.uniform(b.left, b.right)
                lat = rng.uniform(b.bottom, b.top)
            r = float(rng.choice(radii))
            st = disc_stats(src, band, lon, lat, r)
            if st:
                bg_stats.append(st)

    if not p_stats or not bg_stats:
        return None
    P, B = np.array(p_stats), np.array(bg_stats)
    return {
        "n_presence": len(P), "n_background": len(B),
        "median_radius_m": float(np.median(radii)),
        "auc_mean": round(_auc_pb(P[:, 0], B[:, 0]), 4),
        "auc_p90": round(_auc_pb(P[:, 1], B[:, 1]), 4),
        "auc_max": round(_auc_pb(P[:, 2], B[:, 2]), 4),
    }


def repeated(prob_path, presence, rng, repeats=15, **kw):
    """Neighbourhood AUC over several background draws.

    Background is random, and with a few dozen presence records the AUC moves
    by several points between draws. A single number would be noise reported
    as a result, so the spread is measured and quoted.
    """
    runs = [neighbourhood_test(prob_path, presence, rng, **kw)
            for _ in range(repeats)]
    runs = [r for r in runs if r]
    if not runs:
        return None
    out = dict(runs[0])
    for key in ("auc_mean", "auc_p90", "auc_max"):
        vals = np.array([r[key] for r in runs])
        out[key] = round(float(vals.mean()), 4)
        out[key + "_sd"] = round(float(vals.std()), 4)
    out["repeats"] = len(runs)
    return out


def areal_test(prob_path, presence, cell_deg=0.25):
    """Does landslide count per coarse cell rise with mean susceptibility?"""
    with rasterio.open(prob_path) as src:
        band = src.read(1).astype("float64")
        if src.nodata is not None:
            band[band == src.nodata] = np.nan
        tr, b = src.transform, src.bounds

    nx = max(int((b.right - b.left) / cell_deg), 1)
    ny = max(int((b.top - b.bottom) / cell_deg), 1)
    pred = np.full((ny, nx), np.nan)
    count = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            x0, x1 = b.left + i * cell_deg, b.left + (i + 1) * cell_deg
            y0, y1 = b.bottom + j * cell_deg, b.bottom + (j + 1) * cell_deg
            c0, r1 = ~tr * (x0, y1)
            c1, r0 = ~tr * (x1, y0)
            win = band[max(int(r1), 0):int(r0), max(int(c0), 0):int(c1)]
            ok = np.isfinite(win)
            if ok.sum() > 100:
                pred[j, i] = float(win[ok].mean())

    for lon, lat, _ in presence:
        i = int((lon - b.left) / cell_deg)
        j = int((lat - b.bottom) / cell_deg)
        if 0 <= j < ny and 0 <= i < nx:
            count[j, i] += 1

    ok = np.isfinite(pred)
    if ok.sum() < 10:
        return None
    x, y = pred[ok], count[ok]

    # Spearman without scipy
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float)
        r[order] = np.arange(len(a), dtype=float)
        return r
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])

    # frequency ratio across susceptibility quintiles of the coarse cells
    edges = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
    binned = np.digitize(x, edges)
    total = y.sum()
    fr = []
    for k in range(5):
        m = binned == k
        share_cells = m.mean()
        share_slides = (y[m].sum() / total) if total else 0.0
        fr.append(round(share_slides / share_cells, 2) if share_cells else 0.0)
    return {"cell_deg": cell_deg, "n_cells": int(ok.sum()),
            "n_landslides": int(total), "spearman_rho": round(rho, 4),
            "frequency_ratio_by_quintile": fr}


def main() -> None:
    rng = np.random.default_rng(11)
    glc = os.path.join("data", "raw", "inventory", "glc_export.csv")
    if not os.path.exists(glc):
        print("GLC export not found; run step2-download first.")
        return

    rain = ("downpour", "rain", "continuous_rain", "monsoon")
    results = {}

    for label, prob, bbox in common.point_test_targets():
        if not os.path.exists(prob):
            print(f"  {label}: no map at {prob}, skipped")
            continue
        pts_all = load_glc(glc, bbox)
        pts_rain = load_glc(glc, bbox, triggers=rain)
        print(f"\n=== {label}")
        print(f"  {len(pts_all)} GLC records in the map, "
              f"{len(pts_rain)} rainfall-triggered")
        entry = {"n_all": len(pts_all), "n_rain": len(pts_rain)}
        tg = load_target_group(prob)
        if tg is not None:
            print(f"  target group: {len(tg)} settlements")
        for tag, pts in (("all", pts_all), ("rainfall", pts_rain)):
            nb = repeated(prob, pts, rng)
            nbt = (repeated(prob, pts, rng, target_group=tg)
                   if tg is not None else None)
            ar = areal_test(prob, pts)
            entry[tag] = {"neighbourhood": nb, "neighbourhood_target_group": nbt,
                          "areal": ar}
            if nb:
                print(f"  {tag:<9} uniform background   (n={nb['n_presence']}):"
                      f" AUC {nb['auc_mean']:.3f} +/- {nb['auc_mean_sd']:.3f}")
            if nbt:
                print(f"  {tag:<9} target-group backgnd (n={nbt['n_presence']}):"
                      f" AUC {nbt['auc_mean']:.3f} +/- {nbt['auc_mean_sd']:.3f}")
            if ar:
                print(f"  {tag:<9} areal ({ar['n_cells']} cells of "
                      f"{ar['cell_deg']} deg, {ar['n_landslides']} slides): "
                      f"rho {ar['spearman_rho']}, FR {ar['frequency_ratio_by_quintile']}")
        results[label] = entry

    common.save("point_validation", results)


if __name__ == "__main__":
    main()
