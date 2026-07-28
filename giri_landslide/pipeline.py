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
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling

from . import config as C
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

def resolve_inputs(cfg: C.Config, mode: str) -> Dict[str, object]:
    """Return a dict of raw (ungridded) input source paths for the run."""
    bbox = cfg.clipped_bbox()
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
                   reference_mm: Optional[float] = None
                   ) -> Tuple[str, float]:
    """Dimensionless recharge scale, and the reference it is measured against.

    Recharge R enters the wetness term only through R/T, so spatial variation
    in rainfall enters as a multiplier on the fitted ratio. Wettest-month
    precipitation is the available proxy: it is the season when the soil column
    is closest to saturation and when the inventories were mostly filled.

    The field is normalised by a fixed reference in millimetres rather than by
    its own median, so that a scenario in which the whole area gets wetter
    shows up as a scale above 1 instead of cancelling out.
    """
    path = _work(cfg, "recharge_scale.tif")
    if not cfg.spatial_recharge or "precip_monthly" not in inputs:
        if cfg.spatial_recharge:
            _log("recharge", "precipitation absent -> uniform recharge")
        _uniform_raster(grid, 1.0, path)
        return path, float(reference_mm or 0.0)

    # Warping twelve monthly global rasters onto the grid is not free, and the
    # result depends only on the grid and the climate, so it is cached like the
    # terrain. A cached file on a different grid, or one left truncated by an
    # interrupted run, fails the grid check and is rebuilt.
    precip = _work(cfg, "precip_max_month.tif")
    if _matches_grid(precip, grid):
        _log("recharge", "wettest-month precipitation already staged, reusing")
    else:
        _log("recharge", "wettest-month precipitation")
        sources.max_monthly_precip(inputs["precip_monthly"], grid, precip,
                                   tmp_prefix=_work(cfg, "tmp"),
                                   block=cfg.block_size)
    p = _read(precip)
    ref = reference_mm or float(np.nanmedian(p))
    if not np.isfinite(ref) or ref <= 0:
        ref = float(np.nanmean(p)) or 1.0
    scale = p / ref
    _log("recharge", f"reference {ref:.0f} mm; scale spans "
                     f"{np.nanmin(scale):.2f}-{np.nanmax(scale):.2f}")
    _write(grid, scale, path)
    return path, float(ref)


def stage_regions(cfg: C.Config, grid: Grid,
                  inputs: Dict[str, object]) -> Optional[str]:
    """Calibration-region raster, or None if the area is fitted as one piece."""
    if not cfg.calibration_regions:
        return None
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
# steps 4 and 5: stability under reference conditions, and under a scenario
# ---------------------------------------------------------------------------

def run_stability(cfg: C.Config, mode: str = "download",
                  scenario: Optional[dict] = None,
                  label: str = "susceptibility") -> Dict[str, str]:
    """Failure probability over the AOI.

    With ``scenario`` None this is susceptibility: the probability of failure
    at the recharge the parameters were fitted at, with no seismic loading.
    With a scenario it is hazard: the same calculation with recharge scaled or
    an inertial term added. There is deliberately only one code path, because
    in a physical model the two differ by the value of two scalars.
    """
    _ensure_dirs(cfg)
    bbox = cfg.clipped_bbox()
    grid = Grid.from_bbox(bbox, cfg.resolution_deg)
    _log("grid", f"{grid.width}x{grid.height} px @ {cfg.resolution_deg} deg")

    inputs = resolve_inputs(cfg, mode)
    terrain = stage_terrain(cfg, grid, inputs)
    params, by_region, reference_mm = load_fitted(cfg)
    recharge_path, _ = stage_recharge(cfg, grid, inputs, reference_mm)
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
        "parameters": params.as_dict(),
        "n_calibration_regions": len(by_region),
        "recharge_reference_mm": reference_mm,
        "scenario": scenario,
        "unstable_area_pct": _area_above(pfail, 0.5),
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


def run_susceptibility(cfg: C.Config, mode: str = "download") -> Dict[str, str]:
    """STEP 4 - stability under the conditions the parameters were fitted at."""
    return run_stability(cfg, mode, scenario=None, label="susceptibility")


def run_hazard(cfg: C.Config, mode: str = "download",
               return_period_yr: Optional[float] = None,
               pga_g: Optional[float] = None) -> Dict[str, str]:
    """STEP 5 - stability under a stated triggering scenario."""
    rp = return_period_yr or cfg.scenario_return_period_yr
    pga = pga_g if pga_g is not None else cfg.scenario_pga_g
    terms = hazard.scenario_terms(cfg.trigger, return_period_yr=rp, pga_g=pga,
                                  cv=cfg.rainfall_cv,
                                  pga_fraction=cfg.pga_fraction)
    terms["description"] = hazard.describe_scenario(cfg.trigger, terms, rp, pga)
    label = (f"hazard_rp{rp:g}" if cfg.trigger == "rainfall"
             else f"hazard_pga{pga:g}")
    return run_stability(cfg, mode, scenario=terms, label=label)


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

def run(cfg: C.Config, mode: str = "demo") -> Dict[str, str]:
    """Fit (if an inventory is supplied), then susceptibility, then hazard."""
    out: Dict[str, str] = {}
    if cfg.inventory_path:
        fit = run_fit(cfg, mode=mode)
        cfg.fitted_params = fit["path"]
        out["fitted_params"] = fit["path"]
    out.update(run_susceptibility(cfg, mode=mode))
    out.update({f"hazard_{k}": v for k, v in run_hazard(cfg, mode=mode).items()})
    try:
        out["quicklook"] = quicklook(out["probability"],
                                     _out(cfg, "quicklook.png"))
    except Exception as exc:  # matplotlib optional
        _log("quicklook", f"skipped ({exc})")
    return out
