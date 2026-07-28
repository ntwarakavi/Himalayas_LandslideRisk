"""End-to-end orchestration: data -> terrain -> stability -> hazard.

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
    print(f"[giri] {step:<22} {msg}")


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
        kc = physical.critical_acceleration(slope, sca, c, phi, rt)
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
# step 8: package the deliverables
# ---------------------------------------------------------------------------

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

    manifest = {
        "name": cfg.name,
        "model": "SINMAP infinite-slope stability over D-infinity flow routing",
        "package_version": __import__("giri_landslide").__version__,
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
            "reports": [f for f in products if f.endswith(".json")],
            "quicklooks": [f for f in products if f.endswith(".png")],
        },
        "interpretation": [
            "The continuous failure probability is the product; the six-class "
            "map is a legend whose lower three bands are not ordered.",
            "Values are relative. Differences between pixels are meaningful; "
            "the value at a pixel is not a frequency of failure per year.",
            "Skill was measured on soil-mantled crystalline terrain. It is "
            "markedly lower in weak sedimentary hill country, and nothing in "
            "the parameters warns which case an area is.",
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
        climate_suite: bool = True) -> Dict[str, object]:
    """Calibrate if an inventory is supplied, then produce every output.

    This is the whole workflow end to end: fit, susceptibility under the
    present day, every trigger scenario, the climate sweep, and the manifest.
    """
    out: Dict[str, object] = {}
    if cfg.inventory_path:
        fit = run_fit(cfg, mode=mode)
        cfg.fitted_params = fit["path"]
        out["fitted_params"] = fit["path"]

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
