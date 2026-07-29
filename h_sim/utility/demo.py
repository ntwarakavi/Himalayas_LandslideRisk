"""Synthetic input generator so the full pipeline runs with zero downloads.

``make_demo_inputs`` fabricates small, physically plausible input rasters
(DEM, land cover, lithology, monthly precipitation, PGA) over the configured
AOI. This lets anyone execute the complete end-to-end model - and its tests -
on a laptop offline, then swap in the real downloaders for a production run.

The DEM matters most here: the stability model routes flow across it, so the
synthetic terrain has to have real valleys and divides rather than noise. The
multi-octave surface below produces both, which is why the demo exercises the
hydrology rather than sidestepping it.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import rasterio

from .grid import Grid


def _write(path: str, grid: Grid, arr: np.ndarray, dtype: str, nodata) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prof = grid.profile(dtype, nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
    return path


def _fractal_terrain(shape, seed: int = 7) -> np.ndarray:
    """Low-cost multi-octave value noise -> a smooth mountainous surface."""
    rng = np.random.default_rng(seed)
    h, w = shape
    field = np.zeros(shape, dtype="float64")
    amp = 1.0
    for octave in range(1, 7):
        n = 2 ** octave
        coarse = rng.standard_normal((min(n, h), min(n, w)))
        # bilinear upsample to full shape
        ys = np.linspace(0, coarse.shape[0] - 1, h)
        xs = np.linspace(0, coarse.shape[1] - 1, w)
        y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
        y1 = np.clip(y0 + 1, 0, coarse.shape[0] - 1)
        x1 = np.clip(x0 + 1, 0, coarse.shape[1] - 1)
        fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
        top = coarse[y0][:, x0] * (1 - fx) + coarse[y0][:, x1] * fx
        bot = coarse[y1][:, x0] * (1 - fx) + coarse[y1][:, x1] * fx
        field += amp * (top * (1 - fy) + bot * fy)
        amp *= 0.55
    field -= field.min()
    field /= max(field.max(), 1e-9)
    return field


#: Synthetic wetting per pathway, as a multiplier on monthly precipitation.
#: Roughly the direction and spread the CMIP6 ensemble gives for South Asian
#: monsoon precipitation, so the offline demo exercises the climate path with a
#: signal in it rather than reporting "no change" and leaving a user unable to
#: tell whether the machinery works.
DEMO_CLIMATE_WETTING = {"ssp126": 1.04, "ssp245": 1.09,
                        "ssp370": 1.14, "ssp585": 1.20}


def demo_precip_factor(scen) -> float:
    """Precipitation multiplier for a synthetic climate scenario."""
    if scen is None or getattr(scen, "is_baseline", True):
        return 1.0
    wet = DEMO_CLIMATE_WETTING.get(scen.ssp, 1.10)
    # Later windows get more of the signal, on a straight ramp across the
    # twenty-year periods the archive offers.
    from ..model.climate import PERIODS
    step = (PERIODS.index(scen.period) + 1 if scen.period in PERIODS
            else len(PERIODS))
    return 1.0 + (wet - 1.0) * step / len(PERIODS)


def make_demo_inputs(grid: Grid, data_dir: str,
                     scen=None) -> Dict[str, object]:
    """Create synthetic inputs on ``grid``; return a paths dict for the pipeline.

    ``scen`` is a :class:`h_sim.model.climate.ClimateScenario`. Only
    the precipitation depends on it, which mirrors the real model: climate
    enters through recharge and nothing else.
    """
    h, w = grid.shape
    terrain = _fractal_terrain((h, w))

    # --- DEM: 200 m .. 5200 m relief with a steep central ridge -------------
    ridge = np.exp(-((np.linspace(-2, 2, w)[None, :]) ** 2)) * \
        np.exp(-((np.linspace(-2, 2, h)[:, None]) ** 2))
    dem = 200.0 + 5000.0 * (0.6 * terrain + 0.4 * ridge)
    dem_path = _write(os.path.join(data_dir, "demo", "dem.tif"), grid, dem,
                      "float32", -9999.0)

    # --- Land cover (ESA WorldCover codes) keyed off elevation --------------
    lc = np.full((h, w), 30, dtype="uint8")          # grassland
    lc[dem < 800] = 40                                # cropland lowlands
    lc[(dem >= 800) & (dem < 2500)] = 10             # tree cover
    lc[(dem >= 2500) & (dem < 4000)] = 20            # shrubland
    lc[dem >= 4000] = 60                              # bare / sparse
    lc[dem >= 4800] = 70                             # snow and ice
    lc[terrain < 0.05] = 80                          # water in lowest cells
    lc_path = _write(os.path.join(data_dir, "demo", "landcover.tif"), grid, lc,
                     "uint8", 0)

    # --- GLiM lithology codes, banded by a noise field ----------------------
    litho_noise = _fractal_terrain((h, w), seed=21)
    litho = np.full((h, w), 9, dtype="uint8")        # metamorphics
    litho[litho_noise > 0.6] = 2                      # siliciclastic sediments
    litho[litho_noise < 0.3] = 11                     # acid plutonic rock
    litho[lc == 80] = 15                              # water bodies
    litho_path = _write(os.path.join(data_dir, "demo", "glim_codes.tif"), grid,
                        litho, "uint8", 255)

    # --- 12 monthly precipitation rasters (mm), monsoonal, orographic -------
    tag = getattr(scen, "key", "current")
    wetting = demo_precip_factor(scen)
    monthly_paths: List[str] = []
    base = (40.0 + 260.0 * terrain) * wetting         # orographic gradient
    monthly_factor = [0.3, 0.35, 0.5, 0.8, 1.4, 2.6, 3.2, 3.0, 1.8, 0.9,
                      0.4, 0.3]                        # monsoon peak Jul/Aug
    for m, f in enumerate(monthly_factor, start=1):
        pr = base * f
        monthly_paths.append(
            _write(os.path.join(data_dir, "demo", f"prec_{tag}_{m:02d}.tif"),
                   grid, pr, "float32", -9999.0))

    # --- PGA scenario raster (g): a fault-parallel gradient -----------------
    pga = 0.05 + 0.5 * np.linspace(0, 1, w)[None, :] * (0.5 + 0.5 * terrain)
    pga_path = _write(os.path.join(data_dir, "demo", "pga.tif"), grid, pga,
                      "float32", -9999.0)

    return {
        "dem": dem_path,
        "landcover": lc_path,
        "landcover_source": "worldcover",
        "glim_codes_raster": litho_path,
        "precip_monthly": monthly_paths,
        "pga": pga_path,
    }


# ---------------------------------------------------------------------------
# synthetic exposure, so step10 and step11 run offline too
# ---------------------------------------------------------------------------

def make_demo_exposure(grid: Grid, dem: np.ndarray, seed: int = 11):
    """Fabricate settlements and roads over synthetic terrain.

    Placement is not decorative. Real towns sit on valley floors, which is the
    whole reason the reach model exists, so demo settlements are drawn from the
    low ground and demo roads are traced down it. Scattering them uniformly
    would put most of them on ridges and make the demo answer a question nobody
    asks.
    """
    from ..input.exposure import Road, Settlement

    rng = np.random.default_rng(seed)
    h, w = dem.shape
    cutoff = float(np.nanpercentile(dem, 25))
    rows, cols = np.nonzero(dem <= cutoff)
    if rows.size == 0:
        return [], []

    n = min(60, rows.size)
    pick = rng.choice(rows.size, size=n, replace=False)
    kinds = ["hamlet"] * 40 + ["village"] * 14 + ["town"] * 5 + ["city"]
    settlements = []
    for i, k in enumerate(pick):
        r, c = int(rows[k]), int(cols[k])
        lon, lat = grid.transform * (c + 0.5, r + 0.5)
        place = kinds[i % len(kinds)]
        settlements.append(Settlement(
            name=f"Demo {place} {i + 1:02d}", lon=float(lon), lat=float(lat),
            place=place,
            population=int({"city": 60000, "town": 8000, "village": 1200,
                            "hamlet": 150}[place] * (0.5 + rng.random())),
            source="demo"))

    # Roads: start on the low ground and follow the locally lowest neighbour,
    # which on this terrain traces the valleys the way a real road does.
    roads = []
    starts = rng.choice(rows.size, size=min(4, rows.size), replace=False)
    for i, k in enumerate(starts):
        r, c = int(rows[k]), int(cols[k])
        coords = []
        for _ in range(max(h, w)):
            lon, lat = grid.transform * (c + 0.5, r + 0.5)
            coords.append((float(lon), float(lat)))
            nxt, best = None, np.inf
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr == 0 and dc == 0) or not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if dem[rr, cc] < best:
                        best, nxt = dem[rr, cc], (rr, cc)
            if nxt is None or best >= dem[r, c]:
                break
            r, c = nxt
        if len(coords) > 1:
            roads.append(Road(name=f"Demo highway {i + 1}",
                              highway=("primary" if i == 0 else "secondary"),
                              coords=coords, source="demo"))
    return settlements, roads
