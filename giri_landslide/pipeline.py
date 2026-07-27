"""End-to-end orchestration: data -> factors -> susceptibility -> hazard.

The pipeline runs in discrete, independently inspectable steps and writes every
intermediate raster to ``work_dir`` so a run can be stopped and resumed, or a
single stage re-run, on a local machine.

Three input modes:
  * "demo"     - fabricate synthetic inputs (no network); always works.
  * "download" - fetch open datasets for the AOI (needs network).
  * "local"    - use paths supplied in the Config.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling

from . import config as C
from . import factors, susceptibility, triggers, hazard, sources, demo
from .grid import Grid, warp_to_grid, mosaic_and_warp, raster_stats


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(cfg: C.Config) -> None:
    for d in (cfg.data_dir, cfg.work_dir, cfg.out_dir):
        os.makedirs(d, exist_ok=True)


def _work(cfg: C.Config, name: str) -> str:
    return os.path.join(cfg.work_dir, f"{cfg.name}_{name}")


def _uniform_raster(grid: Grid, value: float, out_path: str,
                    dtype: str = "float32", nodata=-9999.0) -> str:
    prof = grid.profile(dtype, nodata)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(np.full(grid.shape, value, dtype=dtype), 1)
    return out_path


def _log(step: str, msg: str = "") -> None:
    print(f"[giri] {step:<22} {msg}")


# ---------------------------------------------------------------------------
# input resolution
# ---------------------------------------------------------------------------

def resolve_inputs(cfg: C.Config, mode: str) -> Dict[str, object]:
    """Return a dict of raw (ungridded) input source paths for the run."""
    bbox = cfg.clipped_bbox()  # restrict AOI to the South Asia Himalayan region
    if mode == "demo":
        grid = Grid.from_bbox(bbox, cfg.resolution_deg)
        _log("demo", "generating synthetic inputs")
        return demo.make_demo_inputs(grid, cfg.data_dir)

    inputs: Dict[str, object] = {}

    # DEM ------------------------------------------------------------------
    if cfg.dem_path:
        inputs["dem_tiles"] = [cfg.dem_path]
    elif mode == "download":
        _log("download:dem", cfg.dem_source)
        inputs["dem_tiles"] = sources.download_dem(bbox, cfg.data_dir,
                                                   cfg.dem_source)
    else:
        raise ValueError("local mode requires config.dem_path")

    # Land cover -----------------------------------------------------------
    inputs["landcover_source"] = cfg.landcover_source
    if cfg.landcover_path:
        inputs["landcover_tiles"] = [cfg.landcover_path]
    elif mode == "download" and cfg.landcover_source == "worldcover":
        _log("download:landcover", "ESA WorldCover 2021")
        inputs["landcover_tiles"] = sources.download_worldcover(bbox,
                                                               cfg.data_dir)
    else:
        raise ValueError("local mode requires config.landcover_path")

    # Lithology -------------------------------------------------------------
    if cfg.glim_path:
        # A vector database (.shp/.gdb) gives the finest lithological detail;
        # a raster path is used directly.
        if os.path.splitext(cfg.glim_path)[1].lower() in (".tif", ".tiff",
                                                          ".asc"):
            inputs["glim_raster"] = cfg.glim_path
        else:
            inputs["glim_vector"] = cfg.glim_path
    elif mode == "download":
        local_gdb = os.path.join(cfg.data_dir, "glim",
                                 sources.GLIM_VECTOR_DIRNAME)
        if cfg.glim_full or os.path.isdir(local_gdb):
            # Full-resolution lithology (default). Downloaded once, then reused.
            gdb = local_gdb if os.path.isdir(local_gdb) else \
                sources.download_glim_vector(cfg.data_dir)
            if gdb:
                _log("lithology", "full-resolution GLiM geodatabase")
                inputs["glim_vector"] = gdb
        if "glim_vector" not in inputs:
            _log("download:lithology", "GLiM 0.5-deg grid (coarse fallback)")
            asc = sources.download_glim_grid(cfg.data_dir)
            if asc:
                sl_tif = os.path.join(cfg.data_dir, "glim", "glim_sl.tif")
                if not os.path.exists(sl_tif):
                    sources.glim_grid_to_sl(asc, sl_tif)
                inputs["glim_sl_raster"] = sl_tif

    # Soil-moisture proxy --------------------------------------------------
    if cfg.trigger == "rainfall":
        if cfg.precip_monthly_dir:
            inputs["precip_monthly"] = sorted(
                os.path.join(cfg.precip_monthly_dir, f)
                for f in os.listdir(cfg.precip_monthly_dir)
                if f.endswith(".tif"))
        elif mode == "download":
            if cfg.climate == "current":
                _log("download:precip",
                     f"WorldClim v2.1 monthly ({cfg.worldclim_res})")
                inputs["precip_monthly"] = sources.download_worldclim_precip(
                    cfg.data_dir, res=cfg.worldclim_res)
            else:
                _log("download:precip",
                     f"CMIP6 {cfg.climate_model} {cfg.climate} "
                     f"{cfg.climate_period} ({cfg.climate_res})")
                fut = sources.download_worldclim_future(
                    cfg.data_dir, ssp=cfg.climate, period=cfg.climate_period,
                    model=cfg.climate_model, res=cfg.climate_res)
                if not fut:
                    raise RuntimeError(
                        f"future-climate precipitation unavailable for "
                        f"{cfg.climate_model}/{cfg.climate}/{cfg.climate_period}")
                inputs["precip_monthly"] = fut
        # else -> fallback handled in staging
    else:
        if cfg.vwc_path:
            inputs["vwc"] = cfg.vwc_path

    # Trigger --------------------------------------------------------------
    if cfg.trigger_path:
        inputs["trigger_raster"] = cfg.trigger_path
    return inputs


# ---------------------------------------------------------------------------
# staging: warp everything onto the reference grid
# ---------------------------------------------------------------------------

def stage_factors(cfg: C.Config, grid: Grid, inputs: Dict[str, object]) -> Dict[str, str]:
    """Produce grid-aligned factor rasters (Sr, Sl, Sv, Sp). Returns paths."""
    block = cfg.block_size
    paths: Dict[str, str] = {}

    # ---- Slope factor ----------------------------------------------------
    dem_grid = _work(cfg, "dem.tif")
    if "dem" in inputs:                       # demo: already on grid
        warp_to_grid(inputs["dem"], grid, dem_grid, Resampling.bilinear,
                     dtype="float32", nodata=-9999.0, block=block)
    else:
        mosaic_and_warp(inputs["dem_tiles"], grid, dem_grid,
                        Resampling.bilinear, dtype="float32", nodata=-9999.0,
                        block=block)
    _log("slope", "computing slope + factor")
    slope_deg = _work(cfg, "slope_deg.tif")
    factors.compute_slope_degrees(dem_grid, slope_deg, block=block)
    paths["slope"] = factors.slope_factor(slope_deg, _work(cfg, "f_slope.tif"),
                                          block=block,
                                          breaks=cfg.slope_breaks)
    paths["slope_deg"] = slope_deg

    # ---- Vegetation factor ----------------------------------------------
    lc_grid = _work(cfg, "landcover.tif")
    lc_src = inputs.get("landcover") or inputs.get("landcover_tiles")
    if isinstance(lc_src, list):
        mosaic_and_warp(lc_src, grid, lc_grid, Resampling.nearest,
                        dtype="uint8", nodata=0, block=block)
    else:
        warp_to_grid(lc_src, grid, lc_grid, Resampling.nearest,
                     dtype="uint8", nodata=0, block=block)
    _log("vegetation", "reclassifying land cover")
    paths["veg"] = factors.landcover_factor(
        lc_grid, _work(cfg, "f_veg.tif"), inputs["landcover_source"],
        block=block)

    # ---- Lithology factor -----------------------------------------------
    glim_sl_grid = _work(cfg, "glim_sl.tif")
    if "glim_sl_raster" in inputs:            # demo: Sl already burned
        warp_to_grid(inputs["glim_sl_raster"], grid, glim_sl_grid,
                     Resampling.nearest, dtype="uint8", nodata=255, block=block)
    elif "glim_vector" in inputs:
        _log("lithology", "rasterising GLiM vector (full resolution)")
        sources.rasterize_glim(inputs["glim_vector"], grid, glim_sl_grid)
    elif "glim_raster" in inputs:
        _log("lithology", "converting supplied GLiM class raster")
        sl_tif = _work(cfg, "glim_sl_native.tif")
        sources.glim_grid_to_sl(inputs["glim_raster"], sl_tif)
        warp_to_grid(sl_tif, grid, glim_sl_grid, Resampling.nearest,
                     dtype="uint8", nodata=255, block=block)
    else:
        _log("lithology", "GLiM absent -> uniform Sl=2 (see sources.GLIM_SOURCE_INFO)")
        _uniform_raster(grid, 2, glim_sl_grid, dtype="uint8", nodata=255)
    paths["litho"] = factors.lithology_factor(
        glim_sl_grid, _work(cfg, "f_litho.tif"), block=block)

    # ---- Soil-moisture factor -------------------------------------------
    sm_grid = _work(cfg, "soilmoist.tif")
    if cfg.trigger == "rainfall":
        if "precip_monthly" in inputs:
            _log("soil moisture", "max monthly precip (MYMMR proxy)")
            sources.max_monthly_precip(inputs["precip_monthly"], grid, sm_grid,
                                       tmp_prefix=_work(cfg, "tmp"),
                                       block=block)
        else:
            _log("soil moisture", "precip absent -> uniform MYMMR=300 mm")
            _uniform_raster(grid, 300.0, sm_grid)
    else:
        if "vwc" in inputs:
            warp_to_grid(inputs["vwc"], grid, sm_grid, Resampling.bilinear,
                         dtype="float32", nodata=-9999.0, block=block)
        else:
            _log("soil moisture", "VWC absent -> uniform 0.25 m3/m3")
            _uniform_raster(grid, 0.25, sm_grid)
    paths["soil"] = factors.soil_moisture_factor(
        sm_grid, _work(cfg, "f_soil.tif"), cfg.trigger, block=block)

    return paths


def _susceptibility_stage(cfg: C.Config, factor_paths: Dict[str, str]) -> str:
    """Combine factors into S and classify into the 5-class susceptibility map."""
    _log("susceptibility", f"combining factors ({cfg.weight_mode})")
    susc_index = _work(cfg, "susc_index.tif")
    susceptibility.combine_factors(
        factor_paths["slope"], factor_paths["litho"], factor_paths["veg"],
        factor_paths["soil"], susc_index, cfg.weights, mode=cfg.weight_mode,
        block=cfg.block_size)

    if cfg.classification == "quantile":
        breaks = susceptibility.quantile_breaks(susc_index, block=cfg.block_size)
        _log("classify", "equal-area quantile breaks")
    else:
        breaks = cfg.susceptibility_breaks

    susc_class = os.path.join(cfg.out_dir, f"{cfg.name}_susceptibility.tif")
    susceptibility.classify_susceptibility(susc_index, susc_class, breaks,
                                           block=cfg.block_size)
    return susc_class


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def run_calibration(cfg: C.Config, mode: str = "demo",
                    n_background: Optional[int] = None,
                    fit_slope_breaks: bool = False) -> dict:
    """Fine-tune factor weights against a historical Himalayan inventory.

    Steps: stage the four factor rasters -> obtain presence points (from
    ``cfg.inventory_path``, a NASA-GLC download, or a synthetic inventory in
    demo mode) -> sample factors at presence + background points -> fit the
    logistic model -> write a calibrated config JSON. Returns a report dict.
    """
    from . import inventory, calibrate

    _ensure_dirs(cfg)
    bbox = cfg.clipped_bbox()
    grid = Grid.from_bbox(bbox, cfg.resolution_deg)
    _log("grid", f"{grid.width}x{grid.height} px @ {cfg.resolution_deg} deg "
                 f"(calibration over Himalaya)")

    inputs = resolve_inputs(cfg, mode)
    factor_paths = stage_factors(cfg, grid, inputs)
    fpaths = [factor_paths["slope"], factor_paths["litho"],
              factor_paths["veg"], factor_paths["soil"]]

    # ---- presence points -------------------------------------------------
    if cfg.inventory_path:
        _log("inventory", f"loading {cfg.inventory_path}")
        presence = inventory.load_inventory(cfg.inventory_path, bbox=bbox,
                                            countries=C.HIMALAYA_COUNTRIES)
    elif mode == "download":
        _log("inventory", "downloading NASA COOLR landslide catalogue")
        glc = inventory.download_nasa_glc(cfg.data_dir, bbox=cfg.region_bbox)
        if not glc:
            raise RuntimeError("NASA GLC unavailable; supply config.inventory_path")
        presence = inventory.load_inventory(glc, bbox=bbox,
                                            countries=C.HIMALAYA_COUNTRIES)
    else:
        _log("inventory", "synthetic Himalayan inventory (demo)")
        presence = inventory.make_synthetic_inventory(
            fpaths, n=1500, true_weights=[1.6, 0.7, 0.5, 1.1])
    if len(presence) < 20:
        raise RuntimeError(f"only {len(presence)} presence points in region; "
                           "need >= 20 for calibration")
    _log("inventory", f"{len(presence)} presence points in region")

    # ---- background points (density-matched to control reporting bias) ---
    n_bg = n_background or max(2 * len(presence), 1500)
    n_near = int(n_bg * 0.7)
    bg_near = inventory.background_points(bbox, n_near, fpaths[0],
                                          near=presence, radius_deg=0.15)
    bg_wide = inventory.background_points(bbox, n_bg - n_near, fpaths[0],
                                          seed=11)
    background = np.vstack([b for b in (bg_near, bg_wide) if len(b)])
    _log("background", f"{len(background)} background points "
                       f"({len(bg_near)} density-matched + {len(bg_wide)} AOI-wide)")

    pres_feats = inventory.sample_factors_at_points(presence, fpaths)
    bg_feats = inventory.sample_factors_at_points(background, fpaths)

    _log("calibrate", "fitting logistic model on log-factors")
    result = calibrate.calibrate(pres_feats, bg_feats)
    _log("calibrate", f"held-out AUC = {result.auc:.3f}  "
                      f"weights = {result.weights}")

    # ---- optional: fit the slope reclassification table ------------------
    slope_breaks = None
    slope_diag = None
    if fit_slope_breaks:
        _log("slope breaks", "fitting from inventory (frequency ratio)")
        sp = inventory.sample_factors_at_points(presence,
                                                [factor_paths["slope_deg"]])[:, 0]
        sb = inventory.sample_factors_at_points(background,
                                                [factor_paths["slope_deg"]])[:, 0]
        try:
            slope_breaks, slope_diag = calibrate.calibrate_slope_breaks(sp, sb)
            peak = slope_diag.get("peak_fr_slope_deg")
            _log("slope breaks", f"{len(slope_breaks)} classes, landslides "
                                 f"most over-represented near {peak}deg")
        except ValueError as exc:
            _log("slope breaks", f"skipped ({exc})")

    # ---- write calibrated config + report -------------------------------
    cal_cfg = calibrate.apply_to_config(cfg, result)
    if slope_breaks:
        cal_cfg.slope_breaks = slope_breaks
    cal_path = os.path.join(cfg.out_dir, f"{cfg.name}_calibrated_config.json")
    cal_cfg.to_json(cal_path)
    report_path = os.path.join(cfg.out_dir, f"{cfg.name}_calibration.json")
    report = result.to_dict()
    if slope_diag:
        report["slope_breaks"] = slope_breaks
        report["slope_diagnostics"] = slope_diag
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    _log("done", f"calibrated config -> {cal_path}")
    return {"result": report, "calibrated_config": cal_path,
            "report": report_path}


# ---------------------------------------------------------------------------
# scenario comparison (present vs future climate)
# ---------------------------------------------------------------------------

def compare_susceptibility(baseline_path: str, scenario_path: str,
                           out_prefix: str, block: int = 1024) -> Dict[str, str]:
    """Difference map: scenario susceptibility class minus baseline class.

    This is the figure the manuscript uses to show climate-change impact
    (its Fig. 8): where does the susceptibility class move up or down between
    two runs? Positive values mean the scenario is *more* susceptible.

    Both inputs must be on the same grid (same AOI, same resolution).
    """
    from .grid import combine_rasters, iter_blocks
    from .susceptibility import SUSC_NODATA

    with rasterio.open(baseline_path) as a, rasterio.open(scenario_path) as b:
        if (a.width, a.height) != (b.width, b.height):
            raise ValueError(
                f"grids differ: baseline {a.width}x{a.height} vs scenario "
                f"{b.width}x{b.height}. Re-run both with the same --bbox/--res.")

    change_path = f"{out_prefix}_susceptibility_change.tif"

    def fn(arrs):
        base, scen = arrs
        bad = (base == SUSC_NODATA) | (scen == SUSC_NODATA) | \
              np.isnan(base) | np.isnan(scen)
        return np.where(bad, -128, scen - base)

    combine_rasters([baseline_path, scenario_path], change_path, fn,
                    "int16", -128, block=block)

    # ---- area statistics -------------------------------------------------
    counts: Dict[str, int] = {}
    total = 0
    with rasterio.open(change_path) as src:
        for win in iter_blocks(src.width, src.height, block):
            d = src.read(1, window=win)
            d = d[d != -128]
            total += d.size
            for v in np.unique(d):
                counts[str(int(v))] = counts.get(str(int(v)), 0) + \
                    int((d == v).sum())
    inc = sum(n for k, n in counts.items() if int(k) > 0)
    dec = sum(n for k, n in counts.items() if int(k) < 0)
    stats = {
        "pixels_compared": total,
        "class_change_histogram": dict(sorted(counts.items(),
                                              key=lambda kv: int(kv[0]))),
        "pct_increased": round(100.0 * inc / total, 3) if total else 0.0,
        "pct_decreased": round(100.0 * dec / total, 3) if total else 0.0,
        "pct_unchanged": round(100.0 * counts.get("0", 0) / total, 3)
        if total else 0.0,
        "baseline": baseline_path,
        "scenario": scenario_path,
    }
    stats_path = f"{out_prefix}_change_summary.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    out = {"change": change_path, "summary": stats_path}
    try:
        out["quicklook"] = change_quicklook(change_path,
                                            f"{out_prefix}_change_quicklook.png")
    except Exception as exc:  # matplotlib optional
        _log("quicklook", f"skipped ({exc})")
    _log("compare", f"{stats['pct_increased']}% of pixels more susceptible, "
                    f"{stats['pct_decreased']}% less")
    return out


def change_quicklook(change_path: str, out_png: str) -> str:
    """Diverging-colour render of a susceptibility class-change raster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    with rasterio.open(change_path) as src:
        d = src.read(1).astype("float32")
        d[d == -128] = np.nan

    cmap = ListedColormap(["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7",
                           "#fddbc7", "#ef8a62", "#b2182b"])
    norm = BoundaryNorm([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(d, cmap=cmap, norm=norm)
    ax.set_title("Susceptibility class change\n(scenario - baseline)")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb.set_label("classes gained (+) / lost (-)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# full run
# ---------------------------------------------------------------------------

def run(cfg: C.Config, mode: str = "demo") -> Dict[str, str]:
    """Execute the whole model and return a dict of output raster paths."""
    _ensure_dirs(cfg)
    grid = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
    _log("grid", f"{grid.width}x{grid.height} px @ {cfg.resolution_deg} deg "
                 f"({cfg.trigger}, weights={cfg.weight_mode})")

    inputs = resolve_inputs(cfg, mode)
    factor_paths = stage_factors(cfg, grid, inputs)

    susc_class = _susceptibility_stage(cfg, factor_paths)

    # ---- trigger class ---------------------------------------------------
    _log("trigger", cfg.trigger)
    trig_class = _work(cfg, "trigger_class.tif")
    if cfg.trigger == "rainfall":
        if "trigger_raster" in inputs:
            # interpret supplied raster as normalised 24h rainfall (z)
            grid_z = _work(cfg, "rain_z.tif")
            warp_to_grid(inputs["trigger_raster"], grid, grid_z,
                         Resampling.bilinear, dtype="float32", nodata=-9999.0,
                         block=cfg.block_size)
            triggers.rainfall_class_from_norm(grid_z, trig_class,
                                              block=cfg.block_size)
        else:
            triggers.rainfall_class_from_return_period(
                susc_class, trig_class, cfg.scenario_return_period_yr,
                block=cfg.block_size)
    else:
        pga_src = inputs.get("pga") or inputs.get("trigger_raster")
        if pga_src:
            grid_pga = _work(cfg, "pga.tif")
            warp_to_grid(pga_src, grid, grid_pga, Resampling.bilinear,
                         dtype="float32", nodata=-9999.0, block=cfg.block_size)
            triggers.pga_class(grid_pga, trig_class, block=cfg.block_size)
        else:
            triggers.pga_class_uniform(susc_class, trig_class,
                                       cfg.scenario_pga_g,
                                       block=cfg.block_size)

    # ---- hazard ----------------------------------------------------------
    _log("hazard", "applying hazard matrix")
    hazard_path = os.path.join(cfg.out_dir, f"{cfg.name}_hazard_probability.tif")
    hazard.apply_hazard_matrix(susc_class, trig_class, hazard_path,
                               cfg.trigger, block=cfg.block_size)

    outputs = {
        "susceptibility": susc_class,
        "trigger_class": trig_class,
        "hazard_probability": hazard_path,
    }

    # ---- summary + quicklook --------------------------------------------
    summary = _summarise(cfg, mode, grid, factor_paths, outputs)
    summary_path = os.path.join(cfg.out_dir, f"{cfg.name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    outputs["summary"] = summary_path
    try:
        outputs["quicklook"] = quicklook(susc_class, hazard_path,
                                         os.path.join(cfg.out_dir,
                                                      f"{cfg.name}_quicklook.png"))
    except Exception as exc:  # matplotlib optional
        _log("quicklook", f"skipped ({exc})")

    _log("done", f"outputs in {cfg.out_dir}")
    return outputs


def _summarise(cfg, mode, grid, factor_paths, outputs) -> dict:
    susc_hist = _class_histogram(outputs["susceptibility"], 1, 5)
    return {
        "name": cfg.name,
        "mode": mode,
        "trigger": cfg.trigger,
        "bbox": list(cfg.bbox),
        "resolution_deg": cfg.resolution_deg,
        "grid": {"width": grid.width, "height": grid.height},
        "weights": vars(cfg.weights),
        "susceptibility_class_pixels": susc_hist,
        "hazard_probability_stats": raster_stats(outputs["hazard_probability"]),
    }


def _class_histogram(path: str, lo: int, hi: int) -> Dict[str, int]:
    from .grid import iter_blocks
    counts = {str(k): 0 for k in range(lo, hi + 1)}
    with rasterio.open(path) as src:
        for win in iter_blocks(src.width, src.height, 1024):
            a = src.read(1, window=win)
            for k in range(lo, hi + 1):
                counts[str(k)] += int(np.count_nonzero(a == k))
    return counts


def quicklook(susc_path: str, hazard_path: str, out_png: str) -> str:
    """Render a two-panel PNG (susceptibility + hazard) for a quick visual check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    with rasterio.open(susc_path) as s:
        susc = s.read(1).astype("float32")
        susc[susc == s.nodata] = np.nan
    with rasterio.open(hazard_path) as h:
        haz = h.read(1).astype("float32")
        haz[haz == h.nodata] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = ListedColormap(["#1a9850", "#a6d96a", "#fee08b", "#fc8d59",
                           "#d73027"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    im0 = axes[0].imshow(susc, cmap=cmap, norm=norm)
    axes[0].set_title("Landslide susceptibility (1-5)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, ticks=[1, 2, 3, 4, 5])

    im1 = axes[1].imshow(haz, cmap="magma")
    axes[1].set_title("Landslide probability (per scenario event)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
