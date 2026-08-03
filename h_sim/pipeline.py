"""H-SIM orchestration: data -> terrain -> stability -> hazard.

The pipeline runs in discrete, independently inspectable steps and writes every
intermediate raster to ``work_dir`` so a run can be stopped and resumed, or a
single stage re-run, on a local machine.

Three input modes:
  * "demo"     - fabricate synthetic inputs (no network); always works.
  * "download" - fetch open datasets for the AOI (needs network).
  * "local"    - use paths supplied in the Config.

One stage does not tile. Contributing area is a property of the whole drainage
network, so :func:`stage_terrain` holds the area of interest in memory rather
than streaming it in blocks. Everything else is windowed.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling

from . import config as C
from .model import climate as CL
from .model import hazard, hydrology, physical
from .input import sources
from .utility import demo
from .utility.grid import Grid, warp_to_grid, mosaic_and_warp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_dirs(cfg: C.Config) -> None:
    for d in (cfg.data_dir, cfg.work_dir, cfg.out_dir):
        os.makedirs(d, exist_ok=True)


def _work(cfg: C.Config, name: str) -> str:
    return os.path.join(cfg.work_dir, f"{cfg.name}_{name}")


def _out(cfg: C.Config, name: str) -> str:
    return os.path.join(cfg.out_dir, f"{cfg.name}_{name}")


def _uniform_raster(grid: Grid, value: float, out_path: str,
                    dtype: str = "float32", nodata=-9999.0) -> str:
    prof = grid.profile(dtype, nodata)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(np.full(grid.shape, value, dtype=dtype), 1)
    return out_path


def _log(step: str, msg: str = "") -> None:
    print(f"[h-sim] {step:<22} {msg}")


def _read(path: str) -> np.ndarray:
    """Read a single-band raster with nodata as NaN."""
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64")
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    return a


def _write(grid: Grid, arr: np.ndarray, path: str, dtype: str = "float32",
           nodata=-9999.0) -> str:
    """Write a band, atomically.

    Work files are named after the run, so two runs of the same name - two
    experiments sharing a cached grid, say - can target the same path at once.
    Writing through a temporary and renaming means a reader either sees the old
    file or the complete new one, never a half-written raster.
    """
    tmp = f"{path}.tmp{os.getpid()}"
    with rasterio.open(tmp, "w", **grid.profile(dtype, nodata)) as dst:
        dst.write(np.where(np.isfinite(arr), arr, nodata).astype(dtype), 1)
    os.replace(tmp, path)
    return path


def _matches_grid(path: str, grid: Grid) -> bool:
    """True if a cached raster is on exactly the grid being asked for.

    Work files are named after the run, so re-running the same name at a
    different resolution or extent would otherwise pick up a stale raster and
    either crash on a shape mismatch or, worse, quietly mix grids.
    """
    if not os.path.exists(path):
        return False
    try:
        with rasterio.open(path) as src:
            return ((src.width, src.height) == (grid.width, grid.height)
                    and np.allclose(np.asarray(src.transform)[:6],
                                    np.asarray(grid.transform)[:6],
                                    rtol=0, atol=1e-9))
    except rasterio.errors.RasterioError:
        return False


def metres_per_cell(bbox, resolution_deg: float) -> Tuple[float, float]:
    """Cell size in metres at the AOI's mid-latitude.

    Flow routing needs a metric grid. Rather than reproject the DEM, the
    degree spacing is converted at the centre latitude, which is accurate to
    better than a percent over an area of a degree or two - well inside the
    uncertainty in the parameters the result feeds.
    """
    lat = 0.5 * (bbox[1] + bbox[3])
    dx = resolution_deg * 111320.0 * max(np.cos(np.radians(lat)), 1e-6)
    dy = resolution_deg * 110540.0
    return float(dx), float(dy)


# ---------------------------------------------------------------------------
# input resolution
# ---------------------------------------------------------------------------

def resolve_inputs(cfg: C.Config, mode: str,
                   scen: Optional[CL.ClimateScenario] = None
                   ) -> Dict[str, object]:
    """Return a dict of raw (ungridded) input source paths for the run.

    ``scen`` selects which climate the precipitation comes from; it defaults to
    whatever ``cfg.climate`` names. Everything else - terrain, lithology, land
    cover - is climate-independent and is resolved identically either way.
    """
    scen = scen or CL.scenario(cfg.climate, cfg.climate_model, cfg.climate_res)
    bbox = cfg.clipped_bbox()
    if mode == "demo":
        grid = Grid.from_bbox(bbox, cfg.resolution_deg)
        _log("demo", f"generating synthetic inputs ({scen.key})")
        inputs = demo.make_demo_inputs(grid, cfg.data_dir, scen=scen)
        inputs["climate"] = scen
        return inputs

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

    # Land cover - only needed as a calibration-region source ----------------
    if cfg.calibration_regions == "landcover":
        inputs["landcover_source"] = cfg.landcover_source
        if cfg.landcover_path:
            inputs["landcover_tiles"] = [cfg.landcover_path]
        elif mode == "download" and cfg.landcover_source == "worldcover":
            _log("download:landcover", "ESA WorldCover 2021")
            inputs["landcover_tiles"] = sources.download_worldcover(
                bbox, cfg.data_dir)
        else:
            raise ValueError("calibration_regions='landcover' needs "
                             "config.landcover_path in local mode")

    # Lithology - likewise ---------------------------------------------------
    if cfg.calibration_regions == "lithology":
        if cfg.glim_path:
            if os.path.splitext(cfg.glim_path)[1].lower() in (".tif", ".tiff",
                                                              ".asc"):
                inputs["glim_raster"] = cfg.glim_path
            else:
                inputs["glim_vector"] = cfg.glim_path
        elif mode == "download":
            local_gdb = os.path.join(cfg.data_dir, "glim",
                                     sources.GLIM_VECTOR_DIRNAME)
            if cfg.glim_full or os.path.isdir(local_gdb):
                gdb = local_gdb if os.path.isdir(local_gdb) else \
                    sources.download_glim_vector(cfg.data_dir)
                if gdb:
                    _log("lithology", "full-resolution GLiM geodatabase")
                    inputs["glim_vector"] = gdb
            if "glim_vector" not in inputs:
                _log("download:lithology", "GLiM 0.5-deg grid (coarse fallback)")
                asc = sources.download_glim_grid(cfg.data_dir)
                if asc:
                    tif = os.path.join(cfg.data_dir, "glim", "glim_codes.tif")
                    if not os.path.exists(tif):
                        sources.glim_grid_to_codes(asc, tif)
                    inputs["glim_codes_raster"] = tif

    # Recharge climatology ---------------------------------------------------
    if cfg.spatial_recharge:
        if cfg.precip_monthly_dir:
            inputs["precip_monthly"] = sorted(
                os.path.join(cfg.precip_monthly_dir, f)
                for f in os.listdir(cfg.precip_monthly_dir)
                if f.endswith(".tif"))
        elif mode == "download":
            if scen.is_baseline:
                _log("download:precip",
                     f"WorldClim v2.1 monthly ({cfg.worldclim_res})")
                inputs["precip_monthly"] = sources.download_worldclim_precip(
                    cfg.data_dir, res=cfg.worldclim_res)
            else:
                _log("download:precip", f"CMIP6 {scen.label}")
                fut = sources.download_worldclim_future(
                    cfg.data_dir, ssp=scen.ssp, period=scen.period,
                    model=scen.gcm, res=scen.resolution)
                if not fut:
                    raise RuntimeError(
                        "future-climate precipitation unavailable for "
                        f"{scen.gcm}/{scen.ssp}/{scen.period}")
                inputs["precip_monthly"] = fut
    inputs["climate"] = scen

    if cfg.pga_path:
        inputs["pga"] = cfg.pga_path
    return inputs


# ---------------------------------------------------------------------------
# terrain: the DEM, and what flow routing makes of it
# ---------------------------------------------------------------------------

def stage_terrain(cfg: C.Config, grid: Grid, inputs: Dict[str, object],
                  force: bool = False) -> Dict[str, str]:
    """Slope and specific catchment area for the AOI.

    This is the expensive stage and the one that does not tile: depression
    filling and D-infinity accumulation both need the whole drainage network at
    once. Results are cached on disk, so later steps re-read them rather than
    recomputing.
    """
    _ensure_dirs(cfg)
    slope_path, sca_path = _work(cfg, "slope_tan.tif"), _work(cfg, "sca.tif")
    if not force and _matches_grid(slope_path, grid) and \
            _matches_grid(sca_path, grid):
        _log("terrain", "slope and catchment area already computed, reusing")
        return {"slope": slope_path, "sca": sca_path,
                "dem": _work(cfg, "dem.tif")}

    dem_grid = _work(cfg, "dem.tif")
    if "dem" in inputs:                       # demo: already on grid
        warp_to_grid(inputs["dem"], grid, dem_grid, Resampling.bilinear,
                     dtype="float32", nodata=-9999.0, block=cfg.block_size)
    else:
        mosaic_and_warp(inputs["dem_tiles"], grid, dem_grid,
                        Resampling.bilinear, dtype="float32", nodata=-9999.0,
                        block=cfg.block_size)

    dem = _read(dem_grid)
    dx, dy = metres_per_cell(cfg.clipped_bbox(), cfg.resolution_deg)
    _log("hydrology", f"filling depressions, D-inf routing "
                      f"({dem.size:,} cells at {dx:.0f}x{dy:.0f} m)")
    sca, slope = hydrology.specific_catchment_area(dem, dx, dy)
    _log("hydrology", f"specific catchment area "
                      f"{np.nanpercentile(sca, 50):.0f} m median, "
                      f"{np.nanmax(sca):.0f} m max")

    _write(grid, sca, sca_path)
    _write(grid, slope, slope_path)
    return {"slope": slope_path, "sca": sca_path, "dem": dem_grid}


def stage_recharge(cfg: C.Config, grid: Grid, inputs: Dict[str, object],
                   reference_mm: Optional[float] = None,
                   scen: Optional[CL.ClimateScenario] = None
                   ) -> Tuple[str, float]:
    """Dimensionless recharge scale, and the reference it is measured against.

    Recharge R enters the wetness term only through R/T, so spatial variation
    in rainfall enters as a multiplier on the fitted ratio. Wettest-month
    precipitation is the available proxy: it is the season when the soil column
    is closest to saturation and when the inventories were mostly filled.

    ``reference_mm`` fixes what a multiplier of 1 means. It is set once, under
    the present-day baseline, when the soil parameters are fitted, and every
    later scenario is divided by that same number. Normalising a future field
    by its own median instead would divide out exactly the signal being looked
    for - a uniformly wetter future would come back looking like today.

    Work files are tagged with the scenario key, so the baseline and its
    futures coexist in one working directory without overwriting each other.
    """
    _ensure_dirs(cfg)
    scen = scen or inputs.get("climate") or CL.BASELINE
    path = _work(cfg, f"recharge_{scen.key}.tif")
    if not cfg.spatial_recharge or "precip_monthly" not in inputs:
        if cfg.spatial_recharge:
            _log("recharge", "precipitation absent -> uniform recharge")
        _uniform_raster(grid, 1.0, path)
        return path, float(reference_mm or 0.0)

    # Warping twelve monthly global rasters onto the grid is not free, and the
    # result depends only on the grid and the climate, so it is cached like the
    # terrain. A cached file on a different grid, or one left truncated by an
    # interrupted run, fails the grid check and is rebuilt.
    precip = _work(cfg, f"precip_max_month_{scen.key}.tif")
    if _matches_grid(precip, grid):
        _log("recharge", f"{scen.key}: wettest-month precipitation reused")
    else:
        _log("recharge", f"{scen.key}: wettest-month precipitation")
        sources.max_monthly_precip(inputs["precip_monthly"], grid, precip,
                                   tmp_prefix=_work(cfg, f"tmp_{scen.key}"),
                                   block=cfg.block_size)
    p = _read(precip)
    ref = reference_mm or float(np.nanmedian(p))
    if not np.isfinite(ref) or ref <= 0:
        ref = float(np.nanmean(p)) or 1.0
    scale = p / ref
    _log("recharge", f"reference {ref:.0f} mm; scale spans "
                     f"{np.nanmin(scale):.2f}-{np.nanmax(scale):.2f} "
                     f"(median {np.nanmedian(scale):.2f})")
    _write(grid, scale, path)
    return path, float(ref)


def stage_regions(cfg: C.Config, grid: Grid,
                  inputs: Dict[str, object]) -> Optional[str]:
    """Calibration-region raster, or None if the area is fitted as one piece."""
    if not cfg.calibration_regions:
        return None
    _ensure_dirs(cfg)
    path = _work(cfg, "regions.tif")

    if cfg.calibration_regions == "landcover":
        src = inputs.get("landcover") or inputs.get("landcover_tiles")
        if src is None:
            _log("regions", "land cover absent -> single calibration region")
            return None
        _log("regions", "ESA WorldCover classes (root cohesion)")
        if isinstance(src, list):
            mosaic_and_warp(src, grid, path, Resampling.nearest, dtype="uint8",
                            nodata=0, block=cfg.block_size)
        else:
            warp_to_grid(src, grid, path, Resampling.nearest, dtype="uint8",
                         nodata=0, block=cfg.block_size)
        return path

    if cfg.calibration_regions == "lithology":
        if "glim_vector" in inputs:
            _log("regions", "rasterising GLiM vector (full resolution)")
            _, codes = sources.rasterize_glim(inputs["glim_vector"], grid, path)
            _log("regions", f"lithologies present: {sorted(set(codes.values()))}")
            return path
        for key in ("glim_codes_raster", "glim_raster"):
            if key in inputs:
                _log("regions", "GLiM lithology codes")
                warp_to_grid(inputs[key], grid, path, Resampling.nearest,
                             dtype="uint8", nodata=255, block=cfg.block_size)
                return path
        _log("regions", "GLiM absent -> single calibration region")
        return None

    raise ValueError(f"unknown calibration_regions {cfg.calibration_regions!r}")


# ---------------------------------------------------------------------------
# step 3: fit the soil parameters to an inventory
# ---------------------------------------------------------------------------

def run_fit(cfg: C.Config, mode: str = "download", cross_validate: bool = True,
            n_background: Optional[int] = None) -> Dict[str, object]:
    """Fit soil parameter ranges to mapped landslides.

    The physics fixes the form of the response; the inventory supplies the
    parameter values. What comes back is written to a JSON file that the
    susceptibility and hazard steps read, so a map is always traceable to the
    landslides that set its parameters.
    """
    from .input import inventory

    if not cfg.inventory_path:
        raise ValueError("fitting needs config.inventory_path (--inventory)")

    _ensure_dirs(cfg)
    bbox = cfg.clipped_bbox()
    grid = Grid.from_bbox(bbox, cfg.resolution_deg)
    _log("grid", f"{grid.width}x{grid.height} px @ {cfg.resolution_deg} deg")

    inputs = resolve_inputs(cfg, mode)
    terrain = stage_terrain(cfg, grid, inputs)
    recharge_path, reference_mm = stage_recharge(cfg, grid, inputs,
                                                 cfg.recharge_reference_mm)
    region_path = stage_regions(cfg, grid, inputs)

    pres = inventory.load_inventory(cfg.inventory_path, bbox=bbox)
    if len(pres) < 50:
        raise ValueError(f"only {len(pres)} landslides fall inside the AOI - "
                         "too few to constrain the parameters")
    n_bg = n_background or max(2 * len(pres), 2000)
    bg = inventory.background_points(bbox, n_bg, terrain["slope"])
    _log("inventory", f"{len(pres)} landslides, {len(bg)} background points")

    layers = [terrain["slope"], terrain["sca"], recharge_path]
    if region_path:
        layers.append(region_path)
    sp = inventory.sample_factors_at_points(pres, layers)
    sb = inventory.sample_factors_at_points(bg, layers)
    sp[sp == -9999.0] = np.nan
    sb[sb == -9999.0] = np.nan

    scale_p = sp[:, 2] if cfg.spatial_recharge else None
    scale_b = sb[:, 2] if cfg.spatial_recharge else None

    _log("fit", f"searching {len(physical.parameter_grid())} parameter sets")
    fit = physical.fit_parameters(sp[:, 0], sp[:, 1], sb[:, 0], sb[:, 1],
                                  n_samples=cfg.n_samples_fit,
                                  recharge_pres=scale_p, recharge_bg=scale_b)
    params = fit["parameters"]
    _log("fit", f"in-sample AUC {fit['auc']:.3f}  {params.as_dict()}")

    report: Dict[str, object] = {
        "parameters": params.as_dict(),
        "in_sample_auc": round(fit["auc"], 4),
        "n_presence": fit["n_presence"],
        "n_background": fit["n_background"],
        "top_trials": fit["top_trials"],
        "recharge_reference_mm": reference_mm,
        "spatial_recharge": cfg.spatial_recharge,
        "resolution_deg": cfg.resolution_deg,
        "bbox": list(bbox),
        "inventory": cfg.inventory_path,
    }

    # Per-region fits, if a zoning was supplied ------------------------------
    if region_path is not None:
        _log("fit", f"per-region fits ({cfg.calibration_regions})")
        reg = physical.fit_parameters_regional(
            sp[:, 0], sp[:, 1], sp[:, 3], sb[:, 0], sb[:, 1], sb[:, 3],
            n_samples=cfg.n_samples_fit,
            min_presence=cfg.min_region_presence)
        report["calibration_regions"] = cfg.calibration_regions
        report["region_parameters"] = {
            str(k): v.as_dict() for k, v in reg["by_region"].items()}
        report["region_detail"] = reg["regions"]
        # The cross-validation below refits the whole-area parameters inside
        # each fold; it does not refit per region. Say so, so the CV figure is
        # not read as evidence for the zoning.
        report["region_note"] = ("cross-validation scores the whole-area "
                                 "parameters, not the per-region ones")
        _log("fit", f"{reg['n_regions_fitted']} regions fitted, the rest fall "
                    "back to the whole-area parameters")

    # Cross-validation ------------------------------------------------------
    if cross_validate:
        for scheme in ("random", "spatial"):
            _log("cross-validate", f"{scheme} split, {cfg.cv_folds} folds")
            cv = physical.cross_validate(
                pres, sp[:, 0], sp[:, 1], bg, sb[:, 0], sb[:, 1], bbox,
                scheme=scheme, n_folds=cfg.cv_folds,
                block_deg=cfg.cv_block_deg, n_samples=cfg.n_samples_fit,
                recharge_pres=scale_p, recharge_bg=scale_b)
            report[f"cv_{scheme}"] = cv
            _log("cross-validate", f"AUC {cv['auc_mean']:.3f} "
                                   f"+/- {cv['auc_std']:.3f} "
                                   f"over {cv['n_folds_scored']} folds")
        report["warnings"] = _fit_warnings(report)

    path = _out(cfg, "fitted_params.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _log("done", f"fitted parameters -> {path}")
    report["path"] = path
    return report


def _fit_warnings(report: dict) -> list:
    """Flag the ways a fit can look better than it is."""
    out = []
    rnd = report.get("cv_random", {}).get("auc_mean", float("nan"))
    spa = report.get("cv_spatial", {}).get("auc_mean", float("nan"))
    sd = report.get("cv_spatial", {}).get("auc_std", float("nan"))
    ins = report.get("in_sample_auc", float("nan"))

    if np.isfinite(rnd) and np.isfinite(spa) and rnd - spa > 0.03:
        out.append(f"random-split AUC ({rnd:.3f}) exceeds spatial-block AUC "
                   f"({spa:.3f}) by {rnd - spa:.3f}: part of the apparent skill "
                   "is interpolation between nearby landslides, not a "
                   "transferable relationship")
    if np.isfinite(sd) and sd > 0.04:
        out.append(f"spatial-block AUC varies by {sd:.3f} between folds: the "
                   "mean describes no particular place, and this spread is the "
                   "range to expect when applying the map somewhere new")
    if np.isfinite(ins) and np.isfinite(spa) and ins - spa > 0.05:
        out.append(f"in-sample AUC ({ins:.3f}) is well above the held-out "
                   f"figure ({spa:.3f}); quote the held-out one")
    if np.isfinite(spa) and spa < 0.65:
        out.append(f"held-out AUC {spa:.3f} is weak; the map orders terrain "
                   "only slightly better than chance")
    return out


def load_fitted(cfg: C.Config) -> Tuple[physical.SoilParameters,
                                        Dict[int, physical.SoilParameters],
                                        Optional[float]]:
    """Read the fitted parameters, falling back to SINMAP's generic ranges."""
    path = cfg.fitted_params or _out(cfg, "fitted_params.json")
    if not os.path.exists(path):
        _log("parameters", "no fit found -> SINMAP generic ranges "
                           "(run step3-fit for local parameters)")
        return physical.SoilParameters(), {}, cfg.recharge_reference_mm

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    params = physical.SoilParameters.from_dict(raw["parameters"])
    by_region = {int(k): physical.SoilParameters.from_dict(v)
                 for k, v in raw.get("region_parameters", {}).items()}
    ref = raw.get("recharge_reference_mm") or cfg.recharge_reference_mm
    _log("parameters", f"{path} ({len(by_region)} regional overrides)")
    return params, by_region, ref


# ---------------------------------------------------------------------------
# production: stability under a climate, and under a trigger scenario
# ---------------------------------------------------------------------------

def run_stability(cfg: C.Config, mode: str = "download",
                  scenario: Optional[dict] = None,
                  label: str = "susceptibility",
                  climate: Optional[CL.ClimateScenario] = None,
                  reference_mm: Optional[float] = None) -> Dict[str, str]:
    """Failure probability over the AOI.

    Three things vary and everything else is shared:

    * ``climate`` selects which precipitation drives the recharge field.
    * ``scenario`` supplies a trigger - a recharge multiplier, a seismic
      coefficient, or both.
    * with neither, this is susceptibility: failure probability at the recharge
      the parameters were fitted at, with no shaking.

    There is deliberately one code path, because in a physical model
    susceptibility, hazard and a climate projection differ only in the value of
    two scalars and the choice of precipitation raster.
    """
    _ensure_dirs(cfg)
    climate = climate or CL.scenario(cfg.climate, cfg.climate_model,
                                     cfg.climate_res)
    bbox = cfg.clipped_bbox()
    grid = Grid.from_bbox(bbox, cfg.resolution_deg)
    _log("grid", f"{grid.width}x{grid.height} px @ {cfg.resolution_deg} deg")

    inputs = resolve_inputs(cfg, mode, scen=climate)
    terrain = stage_terrain(cfg, grid, inputs)
    params, by_region, fitted_ref = load_fitted(cfg)
    reference_mm = reference_mm or fitted_ref
    if not climate.is_baseline and not reference_mm:
        raise ValueError(
            "a future-climate run needs the present-day recharge reference. "
            "Either run step3-fit, which records it, or evaluate the baseline "
            "first so it can be measured.")
    # stage_recharge measures the reference when none was supplied, so take
    # back what it used: that number is what a multiplier of 1 means, and every
    # later scenario has to be divided by the same one.
    recharge_path, reference_mm = stage_recharge(cfg, grid, inputs,
                                                 reference_mm, scen=climate)
    region_path = stage_regions(cfg, grid, inputs) if by_region else None

    slope, sca = _read(terrain["slope"]), _read(terrain["sca"])
    scale = _read(recharge_path) if cfg.spatial_recharge else None

    k_h = 0.0
    if scenario:
        k_h = scenario["k_h"]
        m = scenario["recharge_multiplier"]
        scale = (m if scale is None else scale * m)
        _log("scenario", scenario["description"])

    out: Dict[str, str] = {}
    if by_region and region_path:
        _log("stability", f"failure probability, {len(by_region)} calibration "
                          "regions")
        region = _read(region_path)
        pfail = physical.failure_probability_regional(
            slope, sca, region, by_region, params, n_samples=cfg.n_samples,
            recharge_scale=scale, k_h=k_h)
    else:
        _log("stability", "failure probability over the parameter ranges")
        pfail = physical.failure_probability(
            slope, sca, params, n_samples=cfg.n_samples,
            recharge_scale=scale, k_h=k_h)

    if cfg.output in ("probability", "both"):
        out["probability"] = _write(grid, pfail, _out(cfg, f"{label}_prob.tif"))
    if cfg.output in ("classes", "both"):
        _log("stability", "SINMAP stability classes")
        cls = physical.stability_classes(slope, sca, params,
                                         recharge_scale=scale, k_h=k_h)
        path = _out(cfg, f"{label}_class.tif")
        with rasterio.open(path, "w", **grid.profile("uint8", 255)) as dst:
            dst.write(np.where(np.isfinite(cls), cls, 255).astype("uint8"), 1)
        out["classes"] = path

    if cfg.write_critical_acceleration and not scenario:
        # The mid-range parameters give the representative critical
        # acceleration; the full range is already carried by the probability.
        c = 0.5 * sum(params.cohesion)
        phi = 0.5 * sum(params.friction_deg)
        rt = 0.5 * sum(params.rt) * (1.0 if scale is None else scale)
        kc = physical.critical_acceleration(slope, sca, c, phi, rt,
                                            depth_k=params.depth_k)
        out["critical_acceleration"] = _write(
            grid, kc, _out(cfg, "critical_acceleration.tif"))

    summary = {
        "label": label,
        "climate": climate.as_dict(),
        "parameters": params.as_dict(),
        "n_calibration_regions": len(by_region),
        "recharge_reference_mm": reference_mm,
        "scenario": scenario,
        "unstable_area_pct": _area_above(pfail, 0.5),
        "mean_probability": float(np.nanmean(pfail)),
        "bbox": list(bbox),
        "resolution_deg": cfg.resolution_deg,
        "outputs": out,
    }
    path = _out(cfg, f"{label}_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    out["summary"] = path
    _log("done", f"{summary['unstable_area_pct']:.1f}% of the mapped area has "
                 "failure probability above 0.5")
    return out


def _area_above(arr: np.ndarray, threshold: float) -> float:
    ok = np.isfinite(arr)
    return float(100.0 * (arr[ok] > threshold).mean()) if ok.any() else 0.0


def run_susceptibility(cfg: C.Config, mode: str = "download",
                       climate: Optional[CL.ClimateScenario] = None,
                       reference_mm: Optional[float] = None) -> Dict[str, str]:
    """STEP 5 - susceptibility under one climate, no trigger applied."""
    climate = climate or CL.scenario(cfg.climate, cfg.climate_model,
                                     cfg.climate_res)
    label = ("susceptibility" if climate.is_baseline
             else f"susceptibility_{climate.key}")
    return run_stability(cfg, mode, scenario=None, label=label,
                         climate=climate, reference_mm=reference_mm)


def run_hazard(cfg: C.Config, mode: str = "download",
               return_period_yr: Optional[float] = None,
               pga_g: Optional[float] = None,
               climate: Optional[CL.ClimateScenario] = None
               ) -> Dict[str, str]:
    """STEP 6 - stability under a stated triggering scenario."""
    rp = return_period_yr or cfg.scenario_return_period_yr
    pga = pga_g if pga_g is not None else cfg.scenario_pga_g
    terms = hazard.scenario_terms(cfg.trigger, return_period_yr=rp, pga_g=pga,
                                  cv=cfg.rainfall_cv,
                                  pga_fraction=cfg.pga_fraction)
    terms["description"] = hazard.describe_scenario(cfg.trigger, terms, rp, pga)
    label = (f"hazard_rp{rp:g}" if cfg.trigger == "rainfall"
             else f"hazard_pga{pga:g}")
    return run_stability(cfg, mode, scenario=terms, label=label,
                         climate=climate)


def run_hazard_suite(cfg: C.Config, mode: str = "download",
                     climate: Optional[CL.ClimateScenario] = None
                     ) -> Dict[str, object]:
    """STEP 6 - every trigger scenario the config asks for.

    Rainfall return periods and peak ground accelerations produce separate
    maps rather than one blended figure, because they are different questions
    and a user needs to know which one a map answers.
    """
    out: Dict[str, object] = {"rainfall": {}, "earthquake": {}}
    trigger = cfg.trigger
    try:
        cfg.trigger = "rainfall"
        for rp in cfg.return_periods_yr:
            _log("hazard", f"rainfall, {rp:g}-year return period")
            out["rainfall"][f"{rp:g}yr"] = run_hazard(
                cfg, mode, return_period_yr=rp, climate=climate)
        cfg.trigger = "earthquake"
        for pga in cfg.pga_scenarios_g:
            _log("hazard", f"earthquake, PGA {pga:g} g")
            out["earthquake"][f"{pga:g}g"] = run_hazard(
                cfg, mode, pga_g=pga, climate=climate)
    finally:
        cfg.trigger = trigger
    return out


# ---------------------------------------------------------------------------
# step 7: the climate sweep
# ---------------------------------------------------------------------------

def run_climate(cfg: C.Config, mode: str = "download",
                specs: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """STEP 7 - susceptibility under present and future climates, and the change.

    The baseline is always evaluated, because every future is reported as a
    difference from it. Each future is normalised by the *present-day* recharge
    reference recorded at fitting time, so a uniformly wetter projection shows
    up as a shift rather than cancelling against its own median.

    Terrain is routed once and shared by every scenario; only the recharge
    field and the outputs differ.
    """
    scenarios = CL.parse_all(list(specs or cfg.climate_suite),
                             cfg.climate_model, cfg.climate_res)
    if not any(s.is_baseline for s in scenarios):
        scenarios.insert(0, CL.BASELINE)

    _log("climate", f"{len(scenarios)} scenarios: "
                    + ", ".join(s.key for s in scenarios))

    # The baseline goes first, and not only for reporting: it fixes the
    # recharge reference every future is divided by. Taking it from the fit is
    # preferable, since that is the recharge the soil parameters describe, but
    # measuring it from the present-day field keeps the sweep runnable without
    # an inventory - the futures then still shift against a fixed present day
    # rather than against their own medians.
    maps: Dict[str, Dict[str, str]] = {}
    _log("climate", CL.BASELINE.label)
    maps[CL.BASELINE.key] = run_susceptibility(cfg, mode, climate=CL.BASELINE)
    with open(maps[CL.BASELINE.key]["summary"], encoding="utf-8") as fh:
        reference_mm = json.load(fh).get("recharge_reference_mm")
    _log("climate", f"present-day recharge reference {reference_mm or 0:.0f} mm")

    for s in scenarios:
        if s.is_baseline:
            continue
        _log("climate", s.label)
        maps[s.key] = run_susceptibility(cfg, mode, climate=s,
                                         reference_mm=reference_mm)

    baseline = maps[CL.BASELINE.key]["probability"]
    changes, rows = {}, []
    with rasterio.open(baseline) as src:
        b = src.read(1).astype("float64")
        b[b == src.nodata] = np.nan
    rows.append({"scenario": CL.BASELINE.key, "label": CL.BASELINE.label,
                 "mean_probability": round(float(np.nanmean(b)), 4),
                 "unstable_area_pct": round(_area_above(b, 0.5), 2),
                 "mean_change": 0.0, "pct_more_likely": 0.0})

    for s in scenarios:
        if s.is_baseline:
            continue
        prefix = _out(cfg, f"climate_{s.key}")
        changes[s.key] = compare_probability(baseline,
                                             maps[s.key]["probability"],
                                             prefix, block=cfg.block_size)
        with open(changes[s.key]["summary"], encoding="utf-8") as fh:
            st = json.load(fh)
        with rasterio.open(maps[s.key]["probability"]) as src:
            a = src.read(1).astype("float64")
            a[a == src.nodata] = np.nan
        rows.append({"scenario": s.key, "label": s.label,
                     "mean_probability": round(float(np.nanmean(a)), 4),
                     "unstable_area_pct": round(_area_above(a, 0.5), 2),
                     "mean_change": st["mean_change"],
                     "pct_more_likely": st["pct_more_likely"]})

    report = {"scenarios": rows,
              "maps": {k: v.get("probability") for k, v in maps.items()},
              "changes": {k: v["change"] for k, v in changes.items()}}
    path = _out(cfg, "climate_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    report["summary"] = path
    _log("done", f"climate sweep -> {path}")
    return report


# ---------------------------------------------------------------------------
# step 7: comparing two runs
# ---------------------------------------------------------------------------

def compare_probability(baseline_path: str, scenario_path: str,
                        out_prefix: str, block: int = 1024) -> Dict[str, str]:
    """Difference map: scenario failure probability minus baseline.

    Used for climate scenarios (present against SSP) and for trigger scenarios
    (one return period against another). Both inputs must be on the same grid.
    """
    from .utility.grid import combine_rasters, iter_blocks

    with rasterio.open(baseline_path) as a, rasterio.open(scenario_path) as b:
        if (a.width, a.height) != (b.width, b.height):
            raise ValueError(
                f"grids differ: baseline {a.width}x{a.height} vs scenario "
                f"{b.width}x{b.height}. Re-run both with the same --bbox/--res.")

    change_path = f"{out_prefix}_change.tif"

    def fn(arrs):
        base, scen = arrs
        bad = ~np.isfinite(base) | ~np.isfinite(scen) | \
            (base == -9999.0) | (scen == -9999.0)
        return np.where(bad, -9999.0, scen - base)

    combine_rasters([baseline_path, scenario_path], change_path, fn,
                    "float32", -9999.0, block=block)

    total = inc = dec = 0
    shifted = 0
    acc = 0.0
    with rasterio.open(change_path) as src:
        for win in iter_blocks(src.width, src.height, block):
            d = src.read(1, window=win).astype("float64")
            d = d[(d != -9999.0) & np.isfinite(d)]
            total += d.size
            inc += int((d > 0.01).sum())
            dec += int((d < -0.01).sum())
            shifted += int((np.abs(d) > 0.10).sum())
            acc += float(d.sum())

    stats = {
        "pixels_compared": total,
        "pct_more_likely": round(100.0 * inc / total, 3) if total else 0.0,
        "pct_less_likely": round(100.0 * dec / total, 3) if total else 0.0,
        "pct_shifted_over_0.10": round(100.0 * shifted / total, 3)
        if total else 0.0,
        "mean_change": round(acc / total, 5) if total else 0.0,
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
    _log("compare", f"mean change {stats['mean_change']:+.4f}; "
                    f"{stats['pct_more_likely']}% of pixels more likely to "
                    f"fail, {stats['pct_less_likely']}% less")
    return out


def change_quicklook(change_path: str, out_png: str) -> str:
    """Diverging-colour render of a failure-probability change raster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(change_path) as src:
        d = src.read(1).astype("float32")
        d[d == -9999.0] = np.nan

    lim = float(np.nanpercentile(np.abs(d), 99)) or 0.1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title("Change in failure probability\n(scenario - baseline)")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("more likely (+) / less likely (-)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def quicklook(prob_path: str, out_png: str) -> str:
    """Render a failure-probability raster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with rasterio.open(prob_path) as src:
        p = src.read(1).astype("float32")
        p[p == src.nodata] = np.nan

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(p, cmap="magma_r", vmin=0, vmax=1)
    ax.set_title("Probability of slope failure")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(FS < 1)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# whole workflow
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# step 10: what the susceptibility means for towns and roads
# ---------------------------------------------------------------------------

def scenario_susceptibility(cfg: C.Config, mode: str,
                            scenarios: Sequence[CL.ClimateScenario],
                            override: Optional[str] = None
                            ) -> "Dict[str, str]":
    """Susceptibility rasters for a list of climates, computing what is missing.

    The baseline map is the one step5 already wrote. Future maps are the same
    calculation with the recharge field switched, so rather than demand the
    user run step5 once per scenario by hand, any that are absent are produced
    here.

    The baseline is always resolved first, because it fixes the recharge
    reference every future is divided by. Taking that number from the fit is
    preferable - it is the recharge the soil parameters describe - but the
    present-day summary records it either way, and normalising a future field
    by anything other than a fixed present day would divide out the signal.
    """
    paths: Dict[str, str] = {}
    reference_mm: Optional[float] = None

    base_path = _out(cfg, "susceptibility_prob.tif")
    base_summary = _out(cfg, "susceptibility_summary.json")
    if not os.path.exists(base_path) and not override:
        raise FileNotFoundError(
            f"no susceptibility map at {base_path}. Run step5-susceptibility "
            "first, or pass --susceptibility.")
    paths[CL.BASELINE.key] = override or base_path

    futures = [s for s in scenarios if not s.is_baseline]
    if futures:
        if not os.path.exists(base_summary):
            _log("scenario", "measuring the present-day recharge reference")
            run_susceptibility(cfg, mode, climate=CL.BASELINE)
        with open(base_summary, encoding="utf-8") as fh:
            reference_mm = json.load(fh).get("recharge_reference_mm")

    for scen in futures:
        path = _out(cfg, f"susceptibility_{scen.key}_prob.tif")
        if not os.path.exists(path):
            _log("scenario", f"{scen.label} - computing")
            run_susceptibility(cfg, mode, climate=scen,
                               reference_mm=reference_mm)
        if not os.path.exists(path):
            raise FileNotFoundError(f"expected {path} after computing "
                                    f"{scen.key}")
        paths[scen.key] = path
    return paths


def run_risk(cfg: C.Config, mode: str = "download",
             susceptibility: Optional[str] = None,
             climate: Optional[Sequence[str]] = None,
             assets: Sequence[str] = ("settlements", "roads")
             ) -> Dict[str, object]:
    """Score settlements and road segments by reaching susceptibility.

    ``assets`` selects which layers to do. They are separable because they are
    separate questions with separate audiences - who lives under an unstable
    slope, and which stretches of road are cut when it fails - and because the
    road pass over a province is much the more expensive of the two.

    Not by the susceptibility under them. Towns sit on flat ground, so sampling
    the map at a town's coordinates reports "safe" for precisely the
    settlements a slope above is about to bury. See ``model/risk.py``.

    Every asset is scored under each climate in ``climate`` (default
    ``cfg.risk_climate``: the present day plus near-term CMIP6 windows), so the
    output carries today's exposure and how it moves, per settlement and per
    road segment, rather than one undated number.
    """
    from .input import exposure
    from .model import risk as R

    _ensure_dirs(cfg)
    bbox = cfg.clipped_bbox()
    scens = CL.parse_all(list(climate) if climate else cfg.risk_climate,
                         cfg.climate_model, cfg.climate_res)
    if not any(s.is_baseline for s in scens):
        scens = [CL.BASELINE] + scens
    prob_paths = scenario_susceptibility(cfg, mode, scens, susceptibility)

    grid = Grid.from_bbox(bbox, cfg.resolution_deg)
    inputs = resolve_inputs(cfg, mode)
    terrain = stage_terrain(cfg, grid, inputs)
    dem = _read(terrain["dem"])

    probs: Dict[str, np.ndarray] = {}
    for key, path in prob_paths.items():
        arr = _read(path)
        if arr.shape != dem.shape:
            raise ValueError(
                f"susceptibility {arr.shape} for {key} and DEM {dem.shape} are "
                "on different grids; re-run both at the same --bbox and --res.")
        probs[key] = arr
    dx, dy = metres_per_cell(bbox, cfg.resolution_deg)

    index = R.ReachIndex(dem, grid.transform, dx, dy,
                         travel_angle_deg=cfg.travel_angle_deg,
                         search_radius_m=cfg.reach_radius_m,
                         flow=_reach_flow(cfg, dem, dx, dy),
                         connectivity_floor=cfg.connectivity_floor)
    _log("risk", f"angle of reach {cfg.travel_angle_deg} deg, "
                 f"search radius {cfg.reach_radius_m:.0f} m")
    _log("risk", f"{len(scens)} climate scenarios: "
                 + ", ".join(s.key for s in scens))

    want = set(assets)
    if mode == "demo":
        towns, roads = demo.make_demo_exposure(grid, dem)
        if "settlements" not in want:
            towns = []
        if "roads" not in want:
            roads = []
        _log("exposure", f"synthetic: {len(towns)} settlements, "
                         f"{len(roads)} ways")
    else:
        key = f"{cfg.name}"
        _log("exposure", "settlements")
        towns = exposure.load_settlements(bbox, cfg.data_dir, cache_key=key)
        _log("exposure", "roads")
        roads = exposure.load_roads(bbox, cfg.data_dir, cache_key=key,
                                    classes=cfg.road_classes)
        _log("exposure", f"{len(towns)} settlements, {len(roads)} ways")

    # The run covers the buffered bounding box, but assets outside the clip
    # polygon (the province, in a regional sweep) belong to a neighbour: they
    # would be scored against ground this unit just blanked, shown floating
    # beyond the border, and counted twice across the sweep. Settlements are
    # dropped before scoring; roads after segmentation, by segment midpoint,
    # so a road crossing the border keeps exactly its in-province stretch.
    clip = _clip_lookup(cfg, grid)
    if clip:
        n = len(towns)
        towns = [t for t in towns if clip(t.lon, t.lat)]
        _log("clip", f"{n - len(towns)} of {n} settlements outside the "
                     "unit boundary dropped")

    base = CL.BASELINE.key
    scored_towns = R.score_settlements(index, towns, probs, baseline=base,
                                       footprints=cfg.settlement_footprints)
    scored_roads = R.score_roads(index, roads, probs,
                                 segment_m=cfg.road_segment_m, baseline=base)
    if clip:
        n = len(scored_roads)
        scored_roads = [r for r in scored_roads
                        if clip(*r["coords"][len(r["coords"]) // 2])]
        _log("clip", f"{n - len(scored_roads)} of {n} road segments outside "
                     "the unit boundary dropped")

    # Two failure modes the reach score cannot see, flagged from terrain
    # geometry alone: cut-slope (steep ground immediately above the segment)
    # and washout (the segment touches a channel). Annotated after clipping,
    # so only kept segments pay for the neighbourhood scan.
    if scored_roads:
        mech = R.MechanismIndex(dem, _read(terrain["sca"]), grid.transform,
                                dx, dy,
                                cut_slope_deg=cfg.cut_slope_angle_deg,
                                washout_sca_m=cfg.washout_sca_m)
        for rec in scored_roads:
            rec.update(mech.assess(rec["coords"]))
        n_cut = sum(1 for r in scored_roads if r["cut_slope"])
        n_wash = sum(1 for r in scored_roads if r["washout"])
        _log("mechanisms", f"{n_cut} segments flagged cut-slope "
                           f"(>{cfg.cut_slope_angle_deg:.0f} deg adjacent), "
                           f"{n_wash} at channel crossings "
                           f"(SCA>{cfg.washout_sca_m:.0f} m)")

    _log("risk", f"{len(scored_towns)} settlements and "
                 f"{len(scored_roads)} road segments scored")

    summary = R.summarise(scored_towns, scored_roads,
                          [s.key for s in scens], baseline=base)
    summary["travel_angle_deg"] = cfg.travel_angle_deg
    summary["reach_radius_m"] = cfg.reach_radius_m
    summary["climate"] = [s.as_dict() for s in scens]
    summary["susceptibility"] = prob_paths

    out = {}
    pairs = [(n, r) for n, r in (("settlements", scored_towns),
                                 ("roads", scored_roads)) if n in want]
    for name, rows in pairs:
        path = _out(cfg, f"risk_{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        out[name] = path
    path = _out(cfg, "risk_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    out["summary"] = path
    out["stats"] = summary
    _log("done", f"{summary['n_settlements_exposed']} settlements and "
                 f"{summary['road_km_exposed']:.0f} km of road exposed today")
    for k, ch in (summary.get("change") or {}).items():
        _log("change", f"{k}: {ch['settlements_exposed']:+d} settlements, "
                       f"{ch['road_km_exposed']:+.1f} km road")
    return out


# ---------------------------------------------------------------------------
# step 11: the web map
# ---------------------------------------------------------------------------

def _short_scenario(scen: CL.ClimateScenario) -> str:
    """A label that fits in a table column: 'SSP2-4.5 2041-60'."""
    if scen.is_baseline:
        return "present day"
    ssp = scen.ssp.replace("ssp", "")
    lo, _, hi = (scen.period or "").partition("-")
    return f"SSP{ssp[0]}-{ssp[1]}.{ssp[2]} {lo}-{hi[-2:]}"


def run_webmap(cfg: C.Config, susceptibility: Optional[str] = None,
               open_after: bool = False) -> Dict[str, str]:
    """STEP 11 - assemble a browsable Leaflet page from a finished run.

    Reads what step10 wrote rather than recomputing it, including the climate
    scenarios step10 scored under: the page's selector offers exactly those,
    because a scenario with no scores behind it would be a blank layer.
    """
    from . import webmap
    from .input import inventory as INV

    _ensure_dirs(cfg)
    out_dir = os.path.join(cfg.out_dir, f"{cfg.name}_webmap")
    os.makedirs(out_dir, exist_ok=True)
    layers: Dict[str, object] = {}

    summary = {}
    sp = _out(cfg, "risk_summary.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as fh:
            summary = json.load(fh)

    scens = ([CL.from_dict(d) for d in summary.get("climate", [])]
             or CL.parse_all([cfg.climate], cfg.climate_model,
                             cfg.climate_res))
    prob_paths = summary.get("susceptibility")
    if not isinstance(prob_paths, dict):
        prob_paths = {scens[0].key: (susceptibility
                                     or _out(cfg, "susceptibility_prob.tif"))}
    if susceptibility:
        prob_paths = dict(prob_paths, **{CL.BASELINE.key: susceptibility})

    bounds = None
    scen_layers = []
    for scen in scens:
        path = prob_paths.get(scen.key)
        if not path or not os.path.exists(path):
            _log("webmap", f"{scen.key}: no raster, skipped")
            continue
        png = f"susceptibility_{scen.key}.png"
        _log("webmap", f"rendering {scen.key}")
        b = webmap.raster_to_png(path, os.path.join(out_dir, png))
        bounds = bounds or b
        scen_layers.append({"key": scen.key, "label": scen.label,
                            "short": _short_scenario(scen),
                            "detail": scen.label, "raster": png})
    if bounds is None:
        raise FileNotFoundError(
            "no susceptibility raster to render; run step5 (and step10 for "
            "the future scenarios) first.")
    layers["scenarios"] = scen_layers
    layers["raster"] = scen_layers[0]["raster"]

    data_files: List[str] = []

    def dump(name: str, obj) -> None:
        data_files.append(webmap.write_data(out_dir, name, obj))

    # The unit outline, so the page shows where the clip is rather than
    # leaving assets to stop at an invisible line.
    clip = None
    if cfg.clip_geometry:
        dump("boundary", {"type": "Feature", "properties": {},
                          "geometry": cfg.clip_geometry})
        # The scored files may predate boundary clipping - re-running only
        # this stage is seconds, re-scoring a province is not - so the same
        # test is applied here at page build. Already-clipped files pass
        # through unchanged.
        clip = _clip_lookup(
            cfg, Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg))

    kept: Dict[str, list] = {}
    dropped = 0
    for kind in ("settlements", "roads"):
        src = _out(cfg, f"risk_{kind}.json")
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as fh:
            rows = json.load(fh)
        if clip:
            n = len(rows)
            rows = [r for r in rows
                    if (clip(r["lon"], r["lat"]) if kind == "settlements"
                        else clip(*r["coords"][len(r["coords"]) // 2]))]
            if n - len(rows):
                dropped += n - len(rows)
                _log("webmap", f"{n - len(rows)} of {n} {kind} outside the "
                               "unit boundary hidden")
        kept[kind] = rows
        gj = (webmap.points_geojson(rows) if kind == "settlements"
              else webmap.lines_geojson(rows))
        dump(kind, gj)
        if kind == "settlements":
            # The sidebar list re-ranks itself when the scenario changes, so it
            # needs each place's whole set of scores, not just today's.
            layers["worst"] = [
                {"name": r["name"], "score": r["score"], "band": r["band"],
                 "scenarios": {k: {"score": v["score"], "band": v["band"]}
                               for k, v in (r.get("scenarios") or {}).items()}}
                for r in rows[:40]]

    if dropped:
        # The stored summary counted the unclipped set. The page must not say
        # 60 settlements over a map that draws 41, so the per-scenario stats
        # are recomputed here from exactly the rows the page ships; the keys
        # the page reads for scenarios and rasters are left as stored.
        from .model import risk as R
        summary.update(R.summarise(kept.get("settlements", []),
                                   kept.get("roads", []),
                                   scenarios=[s.key for s in scens],
                                   baseline=CL.BASELINE.key))

    # The training data, so the fit is visible next to what it produced.
    if cfg.inventory_path and os.path.exists(cfg.inventory_path):
        try:
            pts = INV.load_inventory(cfg.inventory_path,
                                     bbox=cfg.clipped_bbox())
            if len(pts):
                dump("inventory", webmap.inventory_geojson(pts, "landslide"))
                bg = INV.background_points(
                    cfg.clipped_bbox(), min(len(pts), 3000),
                    prob_paths[CL.BASELINE.key])
                dump("background", webmap.inventory_geojson(bg, "background"))
                _log("webmap", f"{len(pts)} training landslides, "
                               f"{len(bg)} background points")
        except Exception as exc:                          # noqa: BLE001
            _log("webmap", f"inventory skipped ({exc})")

    w, s, e, n = cfg.clipped_bbox()
    meta = {"area": f"{w:.2f}, {s:.2f} to {e:.2f}, {n:.2f}",
            "resolution": f"{cfg.resolution_deg} deg"}
    path = webmap.build(out_dir, f"H-SIM \u2014 {cfg.name}", bounds,
                        layers, summary, meta, cache_dir=cfg.data_dir,
                        data_files=data_files,
                        maptiler_key=(cfg.maptiler_key
                                      or os.environ.get("MAPTILER_KEY")))
    _log("done", f"web map -> {path} ({len(scen_layers)} climate scenarios)")
    if open_after:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(path)}")
    return {"webmap": path, "dir": out_dir}


# ---------------------------------------------------------------------------
# step 9: regional sweep, one state or province at a time
# ---------------------------------------------------------------------------

def _reach_flow(cfg: C.Config, dem, dx: float, dy: float):
    """Filled DEM and D-infinity angles for connectivity weighting, or None.

    Computed in memory when ``cfg.connectivity_weighting`` is on: filling plus
    the angle pass is a fraction of the full routing cost, and caching another
    raster is not worth an off-by-default feature's while.
    """
    if not cfg.connectivity_weighting:
        return None
    from .model import hydrology

    _log("risk", "connectivity weighting on: filling + D-inf flow angles")
    filled = hydrology.fill_depressions(dem)
    ang, _ = hydrology.dinf_flow_direction(filled, dx, dy)
    return (filled, ang)


def _clip_lookup(cfg: C.Config, grid: Grid):
    """A ``(lon, lat) -> bool`` test for ``cfg.clip_geometry``, or None.

    The polygon is rasterised once onto the run's own grid and points are
    looked up in the resulting mask. Cell-resolution accuracy is the right
    accuracy: the scores being clipped were computed on that grid, so a finer
    boundary test would draw a distinction the model cannot.
    """
    if not cfg.clip_geometry:
        return None
    from .input import admin

    mask = admin.geometry_mask(cfg.clip_geometry, grid)
    h, w = mask.shape
    res = grid.res

    def inside(lon: float, lat: float) -> bool:
        col = int((lon - grid.west) / res)
        row = int((grid.north - lat) / res)
        return 0 <= row < h and 0 <= col < w and bool(mask[row, col])

    return inside


def _clip_to_unit(path: str, mask, grid: Grid, out_path: str) -> str:
    """Blank a raster outside the administrative unit it belongs to."""
    with rasterio.open(path) as src:
        arr = src.read(1)
        prof = src.profile.copy()
        nod = src.nodata
    arr = arr.copy()
    arr[~mask] = nod if nod is not None else 0
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr, 1)
    return out_path


#: What a regional pass can do to one province, in the order the workflow
#: runs them. Each is a separate command, because each is a separate question
#: and a regional pass over any one of them takes hours to days: being able to
#: finish susceptibility everywhere before starting roads anywhere is the
#: difference between a workflow and a single long gamble.
REGION_STAGES = ("susceptibility", "climate", "hazard",
                 "settlements", "roads", "webmap")


def run_admin_unit(cfg: C.Config, unit, mode: str = "download",
                   stages: Sequence[str] = ("susceptibility",)
                   ) -> Dict[str, object]:
    """Produce maps for one state or province.

    The run is done over the unit's bounding box **grown by
    ``cfg.admin_buffer_deg``**, and the outputs are clipped back to the unit
    afterwards. That ordering is the point: a provincial border cuts
    catchments, so routing over the clipped box alone starts every catchment at
    the border and hands the cells just inside it too little upslope area.
    Running wide and clipping late lets flow enter from outside.

    The error is smaller than it sounds - measured at 1% of cells, all in the
    outer ring, and removed entirely by 3 km of buffer - because hillslope
    contributing areas are hundreds of metres, not tens of kilometres. See
    ``admin_buffer_deg`` and analysis/07_boundary_buffer.py.
    """
    from .input import admin

    sub = copy.deepcopy(cfg)
    sub.name = f"{cfg.name}_{unit.slug}"
    sub.bbox = admin.buffered_bbox(unit.bbox, cfg.admin_buffer_deg)
    # The buffer may push the box outside the study region; clipping keeps it
    # legal without shrinking the unit itself.
    sub.bbox = (max(sub.bbox[0], cfg.region_bbox[0]),
                max(sub.bbox[1], cfg.region_bbox[1]),
                min(sub.bbox[2], cfg.region_bbox[2]),
                min(sub.bbox[3], cfg.region_bbox[3]))
    # Fitted parameters are regional, not per unit: resolve them once from the
    # parent run rather than looking for a fit named after the province.
    sub.fitted_params = (cfg.fitted_params
                         or _out(cfg, "fitted_params.json"))
    # The polygon travels with the sub-run so exposure is clipped to the unit,
    # not its bounding box, and the web map can draw the boundary.
    sub.clip_geometry = unit.geometry

    _log("unit", f"{unit.country} / {unit.name}  "
                 f"({sub.cell_count():,} cells incl. buffer)")

    want = set(stages)
    out: Dict[str, object] = {"unit": unit.as_dict(), "stages": list(stages)}
    # Susceptibility underpins everything else, so it is computed whenever it
    # is missing rather than only when asked for. It is cached, so a later
    # stage costs nothing extra when an earlier one already ran.
    base = run_susceptibility(sub, mode=mode)
    if "hazard" in want:
        out["hazard"] = run_hazard_suite(sub, mode=mode)
    if "climate" in want:
        out["climate"] = run_climate(sub, mode=mode)

    # Clip every raster the unit produced back to the province outline.
    grid = Grid.from_bbox(sub.clipped_bbox(), sub.resolution_deg)
    mask = admin.unit_mask(unit, grid)
    coverage = float(mask.mean())
    clipped = {}
    for key, path in base.items():
        if isinstance(path, str) and path.endswith(".tif"):
            clipped[key] = _clip_to_unit(path, mask, grid, path)
    out["maps"] = clipped
    out["summary"] = base.get("summary")

    with rasterio.open(base["probability"]) as src:
        p = src.read(1).astype("float64")
        p[(p == src.nodata) | ~mask] = np.nan
    out["stats"] = {
        "cells_in_unit": int(mask.sum()),
        "buffer_fraction": round(1.0 - coverage, 3),
        "mean_probability": _safe(np.nanmean, p),
        "unstable_area_pct": _area_above(p, 0.5),
        "p90_probability": _safe(lambda a: np.nanpercentile(a, 90), p),
    }
    _log("unit", f"{unit.name}: {out['stats']['unstable_area_pct']:.1f}% "
                 f"above 0.5, mean P {out['stats']['mean_probability']:.4f}")

    # Exposure and the browsable page come last, because both read the rasters
    # this unit has just finished writing - including the clip, so a settlement
    # is never scored against a neighbouring province's ground.
    assets = tuple(a for a in ("settlements", "roads") if a in want)
    if assets:
        try:
            r = run_risk(sub, mode=mode, assets=assets)
            out["risk"] = {k: v for k, v in r.items() if isinstance(v, str)}
            out["exposure"] = r["stats"]
        except Exception as exc:                          # noqa: BLE001
            _log("unit", f"{unit.name}: exposure skipped ({exc})")
    if "webmap" in want:
        try:
            out["webmap"] = run_webmap(sub)["webmap"]
        except Exception as exc:                          # noqa: BLE001
            _log("unit", f"{unit.name}: web map skipped ({exc})")
    return out


def _safe(fn, arr) -> float:
    ok = np.isfinite(arr)
    return round(float(fn(arr[ok])), 4) if ok.any() else float("nan")


def run_region(cfg: C.Config, mode: str = "download",
               countries: Optional[Sequence[str]] = None,
               names: Optional[Sequence[str]] = None,
               stages: Sequence[str] = ("susceptibility",),
               dry_run: bool = False, resume: bool = True
               ) -> Dict[str, object]:
    """Sweep the region one administrative unit at a time, for one stage.

    Units are run independently and their outputs written per unit, so the
    sweep is restartable: a unit whose summary already exists is skipped unless
    ``resume`` is off. That matters because a full regional pass is measured in
    days, and something will interrupt it.

    Units needing more than ``cfg.admin_max_cells`` are reported and skipped
    rather than attempted. Flow routing holds the area in memory, so the
    alternative to skipping is an out-of-memory kill part-way through a sweep.
    """
    from .input import admin

    _ensure_dirs(cfg)
    shp = cfg.admin_path or admin.download_admin1(cfg.data_dir)
    if not shp:
        raise RuntimeError("no administrative boundaries available; "
                           + admin.ADMIN_SOURCE_INFO)

    elev = None
    if cfg.admin_elevation_res:
        elev = sources.download_worldclim_elevation(cfg.data_dir,
                                                    cfg.admin_elevation_res)
        if not elev:
            _log("region", "no elevation grid; selecting on the region box "
                           "alone, which will include plains provinces")

    units = admin.load_units(
        shp, cfg.region_bbox,
        countries=countries or cfg.admin_countries or list(C.HKH_COUNTRIES),
        names=names,
        elevation_path=elev,
        min_mountain_fraction=cfg.admin_min_mountain_fraction,
        mountain_elevation_m=cfg.admin_mountain_elevation_m,
        local_relief_m=cfg.admin_local_relief_m,
        min_mountain_area_km2=cfg.admin_min_mountain_area_km2,
        min_mountain_peak_m=cfg.admin_min_mountain_peak_m,
        exclude=(admin.NOT_HKH if cfg.admin_exclude is None
                 else cfg.admin_exclude))
    if elev:
        _log("region", f"{len(units)} mountain units (mountain = above "
                       f"{cfg.admin_mountain_elevation_m:.0f} m with "
                       f"{cfg.admin_local_relief_m:.0f} m local relief; a unit "
                       f"needs {cfg.admin_min_mountain_fraction:.0%} of it, or "
                       f"{cfg.admin_min_mountain_area_km2:,.0f} km2 and a "
                       f"{cfg.admin_min_mountain_peak_m:.0f} m peak)")
    else:
        _log("region", f"{len(units)} units inside the study region")

    rows = admin.summarise(units, cfg.resolution_deg, cfg.admin_buffer_deg)
    too_big = [r for r in rows if r["cells"] > cfg.admin_max_cells]
    runnable = [u for u in units
                if u.cell_count(cfg.resolution_deg, cfg.admin_buffer_deg)
                <= cfg.admin_max_cells]

    if too_big:
        _log("region", f"{len(too_big)} units exceed "
                       f"{cfg.admin_max_cells:,} cells and are skipped")
    total = sum(u.cell_count(cfg.resolution_deg, cfg.admin_buffer_deg)
                for u in runnable)
    _log("region", f"{len(runnable)} runnable, {total / 1e6:,.0f} million "
                   "cells in total")

    report: Dict[str, object] = {
        "stages": list(stages),
        "resolution_deg": cfg.resolution_deg,
        "buffer_deg": cfg.admin_buffer_deg,
        "n_units_found": len(units),
        "n_units_runnable": len(runnable),
        "skipped_too_large": too_big,
        "plan": rows,
        "units": [],
    }
    if dry_run:
        path = _out(cfg, "region_plan.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        report["summary"] = path
        _log("region", f"plan only -> {path}")
        return report

    done, failed = 0, []
    for i, unit in enumerate(runnable, start=1):
        tag = "_".join(sorted(stages))
        marker = os.path.join(cfg.out_dir,
                              f"{cfg.name}_{unit.slug}_{tag}.json")
        if resume and os.path.exists(marker):
            with open(marker, encoding="utf-8") as fh:
                report["units"].append(json.load(fh))
            _log("skip", f"[{i}/{len(runnable)}] {unit.name} already done")
            continue
        _log("region", f"[{i}/{len(runnable)}] {unit.country} / {unit.name}")
        try:
            res = run_admin_unit(cfg, unit, mode=mode, stages=stages)
        except Exception as exc:                          # noqa: BLE001
            _log("FAILED", f"{unit.name}: {type(exc).__name__}: {exc}")
            failed.append({"unit": unit.as_dict(), "error": str(exc)})
            continue
        with open(marker, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, default=str)
        report["units"].append(res)
        done += 1

    report["n_completed"] = done
    report["failed"] = failed
    path = _out(cfg, f"region_{'_'.join(sorted(stages))}_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    report["summary"] = path

    # One page over the whole sweep. Fifty province folders are an archive; a
    # ranked table with a link per province is something a meeting can work
    # from, which is the point of running the region rather than a catchment.
    if report["units"]:
        try:
            report["index"] = run_region_index(cfg, _merge_stage_reports(cfg,
                                                                        report))
        except Exception as exc:                          # noqa: BLE001
            _log("region", f"index skipped ({exc})")

    _log("done", f"{done} units produced, {len(failed)} failed -> {path}")
    return report


def _merge_stage_reports(cfg: C.Config,
                         report: Dict[str, object]) -> Dict[str, object]:
    """Fold every stage that has run into one view of each province.

    The stages are separate commands writing separate summaries, but the index
    has to show them together: unstable area from step 5 beside settlements
    from step 7 and roads from step 8. Each province's per-stage marker files
    are re-read and merged, so the index is complete after whichever stages
    have actually been run and grows as more of them finish.
    """
    merged: Dict[str, dict] = {}
    prefix = f"{cfg.name}_"
    stage_suffixes = tuple(f"_{s}.json" for s in REGION_STAGES)
    for fname in sorted(os.listdir(cfg.out_dir)):
        if not (fname.startswith(prefix) and fname.endswith(stage_suffixes)):
            continue
        try:
            with open(os.path.join(cfg.out_dir, fname), encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:                                # noqa: BLE001
            continue
        # risk_settlements.json and risk_roads.json share those suffixes and
        # are lists of scored assets, not province markers.
        if not isinstance(rec, dict):
            continue
        unit = rec.get("unit") or {}
        slug = unit.get("slug")
        if not slug:
            continue
        cur = merged.setdefault(slug, {"unit": unit})
        for key in ("stats", "exposure", "webmap", "maps", "risk"):
            if rec.get(key):
                if key in ("stats", "exposure") and isinstance(cur.get(key), dict):
                    cur[key].update(rec[key])
                else:
                    cur[key] = rec[key]

    out = dict(report)
    out["units"] = list(merged.values()) or report.get("units", [])
    return out


def run_region_index(cfg: C.Config, report: Dict[str, object]) -> str:
    """A single ranked page over every province a sweep produced."""
    from . import webmap

    out_dir = os.path.join(cfg.out_dir, f"{cfg.name}_region")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for u in report.get("units", []):
        unit = u.get("unit") or {}
        stats = u.get("stats") or {}
        exp = u.get("exposure") or {}
        page = u.get("webmap")
        rows.append({
            "country": unit.get("country", ""),
            "name": unit.get("name", ""),
            "slug": unit.get("slug", ""),
            "bbox": unit.get("bbox"),
            "cells": stats.get("cells_in_unit"),
            "unstable_pct": stats.get("unstable_area_pct"),
            "mean_probability": stats.get("mean_probability"),
            "p90_probability": stats.get("p90_probability"),
            "settlements": exp.get("n_settlements"),
            "settlements_exposed": exp.get("n_settlements_exposed"),
            "road_km": exp.get("road_km_total"),
            "road_km_exposed": exp.get("road_km_exposed"),
            "map": (os.path.relpath(page, out_dir) if page else None),
        })
    rows.sort(key=lambda r: -(r["unstable_pct"] or 0))

    path = webmap.build_region_index(
        out_dir, f"H-SIM \u2014 {cfg.name}", rows,
        {"resolution_deg": report.get("resolution_deg"),
         "buffer_deg": report.get("buffer_deg"),
         "n_units_found": report.get("n_units_found"),
         "n_completed": report.get("n_completed"),
         "skipped_too_large": [r["name"] for r in
                               report.get("skipped_too_large", [])],
         "failed": [f["unit"]["name"] for f in report.get("failed", [])]})
    _log("region", f"index -> {path}")
    return path


# ---------------------------------------------------------------------------
# step 8: package the deliverables
# ---------------------------------------------------------------------------

def _sources_in(path: str) -> List[dict]:
    """Rows of a scored-asset file, or nothing if it was never written.

    Only the ``source`` field is wanted, but the files are a few megabytes and
    reading them whole is still cheaper than a second format to maintain.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                    # noqa: BLE001
        return []


def run_package(cfg: C.Config) -> Dict[str, str]:
    """STEP 8 - collect what was produced, with the provenance to defend it.

    Nothing is recomputed and nothing is copied. This walks the output
    directory, records what exists, and attaches the things a reader needs in
    order to know what a raster means: the fitted parameters, the held-out
    score, the grid, the data sources and the two trigger conventions.

    A map without this file is not a deliverable, it is a picture.
    """
    _ensure_dirs(cfg)
    prefix = f"{cfg.name}_"
    products = sorted(f for f in os.listdir(cfg.out_dir)
                      if f.startswith(prefix)
                      and f.endswith((".tif", ".png", ".json")))

    fit_path = cfg.fitted_params or _out(cfg, "fitted_params.json")
    fit = {}
    if os.path.exists(fit_path):
        with open(fit_path, encoding="utf-8") as fh:
            fit = json.load(fh)

    def group(kind: str) -> List[str]:
        return [f for f in products if kind in f and f.endswith(".tif")]

    # Describe the rasters that exist, not the config that happens to be in
    # hand: step8 is routinely run without the --bbox/--res the products were
    # made with, and a manifest that misreports their extent is worse than no
    # manifest at all.
    area = {"bbox": list(cfg.clipped_bbox()),
            "resolution_deg": cfg.resolution_deg,
            "cells": cfg.cell_count(),
            "dem_source": cfg.dem_source,
            "read_from": "config"}
    rasters = [f for f in products if f.endswith(".tif")]
    if rasters:
        with rasterio.open(os.path.join(cfg.out_dir, rasters[0])) as src:
            b = src.bounds
            area = {"bbox": [b.left, b.bottom, b.right, b.top],
                    "resolution_deg": abs(src.transform.a),
                    "cells": src.width * src.height,
                    "width": src.width, "height": src.height,
                    "crs": str(src.crs),
                    "dem_source": cfg.dem_source,
                    "read_from": rasters[0]}

    # If step10 ran, carry forward what it was scored under. A reader of the
    # manifest needs to know which climates a settlement table covers, and
    # which of them is the reference the change columns are against.
    exposure_meta = {}
    risk_summary = _out(cfg, "risk_summary.json")
    if os.path.exists(risk_summary):
        with open(risk_summary, encoding="utf-8") as fh:
            rs = json.load(fh)
        exposure_meta = {
            "baseline": rs.get("baseline"),
            "climate": rs.get("climate"),
            "travel_angle_deg": rs.get("travel_angle_deg"),
            "reach_radius_m": rs.get("reach_radius_m"),
            "exposed_threshold": rs.get("exposed_threshold"),
            "n_settlements": rs.get("n_settlements"),
            "n_road_segments": rs.get("n_road_segments"),
            "settlement_sources": sorted({
                s.get("source") for s in _sources_in(
                    _out(cfg, "risk_settlements.json"))}),
            "road_sources": sorted({
                s.get("source") for s in _sources_in(
                    _out(cfg, "risk_roads.json"))}),
            "note": rs.get("note"),
        }

    manifest = {
        "name": cfg.name,
        "model": "SINMAP infinite-slope stability over D-infinity flow routing",
        "package_version": __import__("h_sim").__version__,
        "area": area,
        "calibration": {
            "inventory": fit.get("inventory"),
            "n_presence": fit.get("n_presence"),
            "n_background": fit.get("n_background"),
            "parameters": fit.get("parameters"),
            "recharge_reference_mm": fit.get("recharge_reference_mm"),
            "in_sample_auc": fit.get("in_sample_auc"),
            "held_out_auc": (fit.get("cv_spatial") or {}).get("auc_mean"),
            "held_out_auc_sd": (fit.get("cv_spatial") or {}).get("auc_std"),
            "warnings": fit.get("warnings", []),
        },
        "conventions": {
            "rainfall_cv": cfg.rainfall_cv,
            "pga_fraction": cfg.pga_fraction,
            "note": "neither is fitted here; see docs/RESULTS.md section 5. "
                    "The rainfall value is effectively inert; the PGA fraction "
                    "is not, so quote seismic scenarios as a range over it.",
        },
        "products": {
            "susceptibility": group("susceptibility"),
            "hazard": group("hazard"),
            "climate_change": group("climate"),
            "critical_acceleration": group("critical_acceleration"),
            "exposure": [f for f in products if f.startswith(prefix + "risk_")],
            "reports": [f for f in products if f.endswith(".json")],
            "quicklooks": [f for f in products if f.endswith(".png")],
        },
        "exposure": exposure_meta,
        "interpretation": [
            "The continuous failure probability is the product; the six-class "
            "map is a legend whose lower three bands are not ordered.",
            "Values are relative. Differences between pixels are meaningful; "
            "the value at a pixel is not a frequency of failure per year.",
            "Skill was measured on soil-mantled crystalline terrain. It is "
            "markedly lower in weak sedimentary hill country, and nothing in "
            "the parameters warns which case an area is.",
            "Where settlements and roads were scored, the number is exposure "
            "screening by angle of reach, not risk: no runout model, no "
            "vulnerability, no damage function.",
        ],
    }
    path = _out(cfg, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    n = sum(len(v) for v in manifest["products"].values())
    _log("package", f"{n} products catalogued -> {path}")
    return {"manifest": path}


# ---------------------------------------------------------------------------
# the whole sequence
# ---------------------------------------------------------------------------

def run(cfg: C.Config, mode: str = "demo",
        climate_suite: bool = True,
        region: bool = False) -> Dict[str, object]:
    """Calibrate if an inventory is supplied, then produce every output.

    With ``region``, production is the province-by-province sweep over the
    whole Hindu Kush Himalaya, which is what this model is for. Without it,
    production is the single area of interest in ``cfg`` - useful for
    calibrating, validating and debugging, not a deliverable.
    """
    out: Dict[str, object] = {}
    if cfg.inventory_path:
        fit = run_fit(cfg, mode=mode)
        cfg.fitted_params = fit["path"]
        out["fitted_params"] = fit["path"]

    if region:
        # One stage at a time, in workflow order: finish susceptibility
        # everywhere before starting roads anywhere, so an interrupted run
        # leaves a complete product rather than a fragment of every product.
        stages = ["susceptibility"]
        if climate_suite:
            stages.append("climate")
        stages += ["settlements", "roads", "webmap"]
        out["region"] = {}
        for stage in stages:
            _log("workflow", f"region-wide: {stage}")
            out["region"][stage] = run_region(cfg, mode=mode,
                                              stages=(stage,))
        out.update(run_package(cfg))
        return out

    base = run_susceptibility(cfg, mode=mode)
    out["susceptibility"] = base
    out["hazard"] = run_hazard_suite(cfg, mode=mode)
    if climate_suite:
        out["climate"] = run_climate(cfg, mode=mode)

    try:
        out["quicklook"] = quicklook(base["probability"],
                                     _out(cfg, "quicklook.png"))
    except Exception as exc:  # matplotlib optional
        _log("quicklook", f"skipped ({exc})")
    out.update(run_package(cfg))
    return out
