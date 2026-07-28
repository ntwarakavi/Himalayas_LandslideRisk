"""Command-line interface, organised as a numbered workflow.

    step1  check      which datasets do I have, and can I get the rest?
    step2  download   fetch everything (skips anything already cached)
    step3  fit        fit the soil parameters to real landslides
    step4  stability  WHERE is the ground unstable?
    step5  hazard     IF a storm or quake hits, how likely is failure?
    step6  validate   does the map hold up on landslides it never saw?
    step7  compare    how does that change between two scenarios?

    run-all           step2 -> step3 -> step4 -> step5 in one go
    info              dataset sources and licences

Every step writes files and prints what it produced, so you can stop after any
step, look at the output, and carry on. Nothing is ever re-downloaded.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import config as C
from .input import datasets, sources
from . import pipeline


# ---------------------------------------------------------------------------
# config assembly
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace) -> C.Config:
    cfg = C.Config.from_json(args.config) if getattr(args, "config", None) \
        else C.Config()
    simple = ["name", "trigger", "data_dir", "work_dir", "out_dir",
              "dem_source", "output", "climate", "climate_period",
              "climate_model", "worldclim_res", "calibration_regions",
              "fitted_params"]
    for attr in simple:
        val = getattr(args, attr, None)
        if val:
            setattr(cfg, attr, val)
    if getattr(args, "bbox", None):
        cfg.bbox = tuple(args.bbox)
    if getattr(args, "res", None):
        cfg.resolution_deg = args.res
    if getattr(args, "block", None):
        cfg.block_size = args.block
    if getattr(args, "pga", None) is not None:
        cfg.scenario_pga_g = args.pga
    if getattr(args, "return_period", None) is not None:
        cfg.scenario_return_period_yr = args.return_period
    if getattr(args, "inventory", None):
        cfg.inventory_path = args.inventory
    if getattr(args, "glim_grid", False):
        cfg.glim_full = False
    if getattr(args, "uniform_recharge", False):
        cfg.spatial_recharge = False
    if getattr(args, "samples", None):
        cfg.n_samples = args.samples
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("area and grid")
    g.add_argument("--config", help="JSON config file (see configs/)")
    g.add_argument("--name", help="run label; prefixes every output file")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="area of interest in degrees")
    g.add_argument("--res", type=float,
                   help="grid resolution in degrees "
                        "(0.00027778 = 30 m, 0.00083333 = 90 m)")
    g.add_argument("--block", type=int, help="tile size in pixels")

    g = p.add_argument_group("model options")
    g.add_argument("--trigger", choices=["rainfall", "earthquake"])
    g.add_argument("--pga", type=float, help="earthquake scenario, g")
    g.add_argument("--return-period", dest="return_period", type=float,
                   help="rainfall scenario, years")
    g.add_argument("--output", choices=["probability", "classes", "both"],
                   help="continuous failure probability, SINMAP classes, "
                        "or both (default)")
    g.add_argument("--calibration-regions", dest="calibration_regions",
                   choices=["lithology", "landcover"],
                   help="fit separate soil parameters per rock type or per "
                        "land-cover class instead of one set for the area")
    g.add_argument("--uniform-recharge", dest="uniform_recharge",
                   action="store_true",
                   help="hold recharge uniform, isolating the effect of "
                        "terrain alone")
    g.add_argument("--samples", type=int,
                   help="Monte Carlo draws per pixel (default 200)")
    g.add_argument("--fitted-params", dest="fitted_params",
                   help="JSON from step3 (default: from --name)")

    g = p.add_argument_group("data options")
    g.add_argument("--dem-source", dest="dem_source",
                   choices=["copernicus90", "copernicus30"])
    g.add_argument("--glim-grid", dest="glim_grid", action="store_true",
                   help="coarse 0.5-deg lithology instead of full GLiM")
    g.add_argument("--worldclim-res", dest="worldclim_res",
                   choices=["30s", "2.5m", "5m", "10m"])
    g.add_argument("--climate",
                   choices=["current", "ssp126", "ssp245", "ssp370", "ssp585"])
    g.add_argument("--climate-period", dest="climate_period",
                   choices=["2021-2040", "2041-2060", "2061-2080", "2081-2100"])
    g.add_argument("--climate-model", dest="climate_model")
    g.add_argument("--inventory",
                   help="landslide inventory (.shp/.kml/.csv/.geojson)")
    g.add_argument("--data-dir", dest="data_dir")
    g.add_argument("--work-dir", dest="work_dir")
    g.add_argument("--out-dir", dest="out_dir")


def _mode(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["demo", "download", "local"],
                   default="download",
                   help="demo = synthetic offline data; download = real data")


def _warn_if_large(cfg: C.Config) -> None:
    """Flow routing is not tiled, so the AOI bounds memory directly."""
    n = cfg.cell_count()
    if n > 40_000_000:
        print(f"  note: {n / 1e6:.0f} million cells. Flow accumulation holds "
              "the whole area in memory and runs a single pass over it, so "
              "expect this to be slow and memory-hungry. Consider a smaller "
              "--bbox or a coarser --res.\n")


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def _step_check(args) -> int:
    cfg = _build_config(args)
    print("STEP 1  Dataset availability")
    print(f"        cache: {cfg.data_dir}")
    rows = datasets.check_all(cfg.data_dir, probe=not args.offline)
    print(datasets.format_report(rows))
    need = [r for r in rows if not r["cached"] and r["reachable"] is not False]
    mb = sum(r["approx_mb"] for r in need)
    print(f"\n{len(need)} dataset(s) still to fetch, roughly {mb:.0f} MB.")
    print("Next:  python -m giri_landslide.cli step2-download")
    return 0


def _step_download(args) -> int:
    cfg = _build_config(args)
    bbox = cfg.clipped_bbox()
    print("STEP 2  Download (anything already cached is skipped)\n")
    from .input import inventory

    got = sources.download_dem(bbox, cfg.data_dir, cfg.dem_source)
    print(f"  DEM tiles            {len(got)} ({cfg.dem_source})")

    if cfg.spatial_recharge:
        if cfg.climate == "current":
            pr = sources.download_worldclim_precip(cfg.data_dir,
                                                   res=cfg.worldclim_res)
            print(f"  Precipitation        {len(pr)} months "
                  f"({cfg.worldclim_res})")
        else:
            pr = sources.download_worldclim_future(
                cfg.data_dir, ssp=cfg.climate, period=cfg.climate_period,
                model=cfg.climate_model, res=cfg.climate_res)
            print(f"  CMIP6 precipitation  {len(pr) if pr else 0} months "
                  f"({cfg.climate} {cfg.climate_period})")

    # Calibration-region sources are optional; fetch only what is asked for.
    if cfg.calibration_regions == "landcover":
        lc = sources.download_worldcover(bbox, cfg.data_dir)
        print(f"  Land cover tiles     {len(lc)}")
    elif cfg.calibration_regions == "lithology":
        if cfg.glim_full:
            gdb = sources.download_glim_vector(cfg.data_dir)
            print(f"  GLiM lithology       "
                  f"{'full geodatabase' if gdb else 'FAILED'}")
        else:
            print(f"  GLiM lithology       "
                  f"{sources.download_glim_grid(cfg.data_dir)}")
    else:
        print("  (no calibration regions requested: land cover and GLiM "
              "skipped)")

    if not args.no_inventories:
        print("\n  Landslide inventories (for step 3):")
        for key, (fetch, label) in inventory.INVENTORY_FETCHERS.items():
            try:
                p = fetch(cfg.data_dir)
                print(f"    {'OK   ' if p else 'FAIL '} {key:14s} {label}")
            except Exception as exc:                     # noqa: BLE001
                print(f"    FAIL  {key:14s} {type(exc).__name__}: {exc}")
        inv = inventory.download_nasa_glc(cfg.data_dir, bbox=cfg.region_bbox)
        print(f"    {'OK   ' if inv else 'FAIL '} coolr          NASA GLC/COOLR")

    print("\nNext:  python -m giri_landslide.cli step3-fit --inventory <path>")
    return 0


def _step_fit(args) -> int:
    cfg = _build_config(args)
    print("STEP 3  Fit soil parameters to a landslide inventory\n")
    _warn_if_large(cfg)
    report = pipeline.run_fit(cfg, mode=args.mode,
                              cross_validate=not args.no_cv,
                              n_background=args.background)

    p = report["parameters"]
    print("\n  Fitted parameter ranges")
    print(f"    cohesion C        {p['cohesion'][0]:.3f} .. {p['cohesion'][1]:.3f}"
          "   (dimensionless, root + soil, over depth x unit weight)")
    print(f"    friction phi      {p['friction_deg'][0]:.1f} .. "
          f"{p['friction_deg'][1]:.1f} deg")
    print(f"    R/T               {p['rt'][0]:.2e} .. {p['rt'][1]:.2e} 1/m")
    if report.get("recharge_reference_mm"):
        print(f"    recharge ref      "
              f"{report['recharge_reference_mm']:.0f} mm wettest-month precip")

    print(f"\n  In-sample AUC       {report['in_sample_auc']:.3f}   "
          f"({report['n_presence']} landslides, "
          f"{report['n_background']} background)")
    for scheme in ("random", "spatial"):
        cv = report.get(f"cv_{scheme}")
        if cv:
            print(f"  {scheme:<8} CV AUC     {cv['auc_mean']:.3f} +/- "
                  f"{cv['auc_std']:.3f}   folds "
                  f"{[f'{a:.3f}' for a in cv['auc_folds']]}")

    if report.get("region_detail"):
        print(f"\n  Per-region fits ({report['calibration_regions']})")
        for code, d in sorted(report["region_detail"].items(),
                              key=lambda kv: -kv[1]["n_presence"]):
            rp = d["parameters"]
            print(f"    region {code:>3}  n={d['n_presence']:>6}  "
                  f"AUC {d['auc']:.3f}  phi {rp['friction_deg'][0]:.0f}-"
                  f"{rp['friction_deg'][1]:.0f}  C<={rp['cohesion'][1]:.2f}")

    for w in report.get("warnings", []):
        print(f"\n  ! {w}")

    print(f"\n  Fitted parameters -> {report['path']}")
    print("\nNext:  python -m giri_landslide.cli step4-stability "
          f"--name {cfg.name}")
    return 0


def _step_stability(args) -> int:
    cfg = _build_config(args)
    print("STEP 4  Slope stability (SINMAP infinite slope + D-inf hydrology)\n")
    _warn_if_large(cfg)
    out = pipeline.run_susceptibility(cfg, mode=args.mode)
    print("\n  Outputs:")
    for k, v in out.items():
        print(f"    {k:22s} {v}")
    prob = out.get("probability")
    if prob:
        print("\n  Validate it against landslides the fit never saw:")
        print(f"    python -m giri_landslide.cli step6-validate "
              f"--susceptibility {prob} --inventory <path>")
    return 0


def _step_hazard(args) -> int:
    cfg = _build_config(args)
    print("STEP 5  Hazard under a triggering scenario\n")
    _warn_if_large(cfg)
    out = pipeline.run_hazard(cfg, mode=args.mode)
    print("\n  Outputs:")
    for k, v in out.items():
        print(f"    {k:22s} {v}")
    return 0


def _step_validate(args) -> int:
    import json

    import numpy as np

    from .input import inventory
    from .model import validate

    susc = args.susceptibility
    if not susc:
        for suffix in ("_susceptibility_prob.tif", "_susceptibility_class.tif"):
            cand = os.path.join(args.out_dir, f"{args.name}{suffix}")
            if os.path.exists(cand):
                susc = cand
                break
    if not susc or not os.path.exists(susc):
        print(f"error: no stability map found for '{args.name}' in "
              f"{args.out_dir}")
        return 1
    print("STEP 6  Validate against a held-out inventory\n")
    kind = ("continuous failure probability" if validate.is_continuous(susc)
            else "stability classes")
    print(f"  map       : {susc}  ({kind})")
    print("  inventories:")

    import rasterio
    with rasterio.open(susc) as src:
        b = src.bounds
        bbox = (b.left, b.bottom, b.right, b.top)
    paths = args.inventory if isinstance(args.inventory, list) \
        else [args.inventory]
    parts = []
    for src in paths:
        sub = inventory.load_inventory(src, bbox=bbox)
        print(f"    {len(sub):6d} from {os.path.basename(src)}")
        if len(sub):
            parts.append(sub)
    pts = np.vstack(parts) if parts else np.empty((0, 2))
    print(f"  {len(pts)} landslides fall inside the map extent\n")
    if len(pts) == 0:
        print("  nothing to validate: the inventory does not overlap the map.")
        return 1

    bg = inventory.background_points(bbox, max(4 * len(pts), 2000), susc)
    result = validate.validate_susceptibility(susc, pts, bg,
                                              block=args.block or 1024)
    print(validate.format_report(result))

    os.makedirs(args.out_dir, exist_ok=True)
    label = args.name or os.path.splitext(os.path.basename(susc))[0]
    path = os.path.join(args.out_dir, f"{label}_validation.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    print(f"\n  Report -> {path}")
    return 0


def _step_compare(args) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    print("STEP 7  Compare two scenarios\n")
    out = pipeline.compare_probability(
        args.baseline, args.scenario,
        os.path.join(args.out_dir, args.name or "comparison"),
        block=args.block or 1024)
    print("\n  Outputs:")
    for k, v in out.items():
        print(f"    {k:22s} {v}")
    return 0


def _run_all(args) -> int:
    rc = _step_download(args)
    if rc:
        return rc
    print("\n" + "=" * 68 + "\n")
    cfg = _build_config(args)
    _warn_if_large(cfg)
    out = pipeline.run(cfg, mode=args.mode)
    print("\nOutputs:")
    for k, v in out.items():
        print(f"  {k:22s} {v}")
    return 0


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="giri_landslide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Hindu Kush Himalaya landslide hazard model: SINMAP "
                    "infinite-slope stability over D-infinity flow routing.",
        epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("step1-check", aliases=["check"],
                       help="which datasets are present or reachable")
    p.add_argument("--offline", action="store_true",
                   help="only report the cache, do not probe the network")
    _add_common(p)

    p = sub.add_parser("step2-download", aliases=["download"],
                       help="fetch datasets (skips whatever is cached)")
    p.add_argument("--no-inventories", action="store_true",
                   help="skip the landslide inventories")
    _add_common(p)

    p = sub.add_parser("step3-fit", aliases=["fit"],
                       help="fit soil parameters to a landslide inventory")
    p.add_argument("--no-cv", action="store_true",
                   help="skip cross-validation (faster, but then only the "
                        "optimistic in-sample score is available)")
    p.add_argument("--background", type=int,
                   help="background sample size (default: twice the "
                        "inventory, minimum 2000)")
    _mode(p); _add_common(p)

    p = sub.add_parser("step4-stability", aliases=["stability",
                                                   "susceptibility"],
                       help="failure probability at the fitted conditions")
    _mode(p); _add_common(p)

    p = sub.add_parser("step5-hazard", aliases=["hazard"],
                       help="failure probability under a trigger scenario")
    _mode(p); _add_common(p)

    p = sub.add_parser("step6-validate", aliases=["validate"],
                       help="test a map against a held-out inventory")
    p.add_argument("--susceptibility",
                   help="stability GeoTIFF (default: from --name)")
    p.add_argument("--inventory", required=True, nargs="+",
                   help="one or more INDEPENDENT inventories; multiple paths "
                        "are pooled into a single validation set")
    p.add_argument("--name")
    p.add_argument("--out-dir", dest="out_dir", default="outputs")
    p.add_argument("--block", type=int)

    p = sub.add_parser("step7-compare", aliases=["compare"],
                       help="difference two failure-probability maps")
    p.add_argument("--baseline", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--name")
    p.add_argument("--out-dir", dest="out_dir", default="outputs")
    p.add_argument("--block", type=int)

    p = sub.add_parser("run-all", aliases=["run"],
                       help="download + fit + stability + hazard in one go")
    p.add_argument("--no-inventories", action="store_true")
    _mode(p); _add_common(p)

    sub.add_parser("info", help="dataset sources, licences and citations")

    args = parser.parse_args(argv)
    cmd = args.command

    if cmd == "info":
        print("Datasets used by this model\n")
        print(datasets.format_report(
            datasets.check_all("data/raw", probe=False)))
        print("\n" + sources.GLIM_SOURCE_INFO)
        print("\n" + sources.PGA_SOURCE_INFO)
        return 0

    handlers = {
        "step1-check": _step_check, "check": _step_check,
        "step2-download": _step_download, "download": _step_download,
        "step3-fit": _step_fit, "fit": _step_fit,
        "step4-stability": _step_stability, "stability": _step_stability,
        "susceptibility": _step_stability,
        "step5-hazard": _step_hazard, "hazard": _step_hazard,
        "step6-validate": _step_validate, "validate": _step_validate,
        "step7-compare": _step_compare, "compare": _step_compare,
        "run-all": _run_all, "run": _run_all,
    }
    return handlers[cmd](args)


if __name__ == "__main__":
    sys.exit(main())
