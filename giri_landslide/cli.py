"""Command-line interface for the SINMAP landslide model.

The workflow runs in four phases. Each step writes files and prints what it
produced, so you can stop after any step, inspect the output, and carry on.
Nothing is ever re-downloaded and the expensive stage is cached.

    SET UP
      step1-check          which datasets do I have, and can I get the rest?
      step2-download       fetch them

    CALIBRATE AND VALIDATE          (needs a landslide inventory)
      step3-fit            fit the soil parameters, cross-validated
      step4-validate       score the map against an inventory it never saw

    PRODUCE                          (applies the validated parameters)
      step5-susceptibility present-day failure probability
      step6-hazard         rainfall and earthquake scenarios
      step7-climate        CMIP6 futures, and the change from today

    PACKAGE
      step8-package        manifest: what was produced and what it means

    run-all                every phase in sequence
    info                   data sources, licences and citations

Phase order matters. Do not produce a map from parameters that have not been
through step4 on an independent inventory - the fit will always look good on
the landslides it was fitted to.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import config as C
from .input import datasets, sources
from . import pipeline
from .model import climate as CL


# ---------------------------------------------------------------------------
# config assembly
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace) -> C.Config:
    cfg = C.Config.from_json(args.config) if getattr(args, "config", None) \
        else C.Config()
    simple = ["name", "trigger", "data_dir", "work_dir", "out_dir",
              "dem_source", "output", "climate", "climate_model",
              "worldclim_res", "calibration_regions", "fitted_params"]
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
    if getattr(args, "scenarios", None):
        cfg.climate_suite = list(args.scenarios)
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("area and grid")
    g.add_argument("--config", help="JSON config file (see configs/)")
    g.add_argument("--name", help="run label; prefixes every output file")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="area of interest in degrees")
    g.add_argument("--res", type=float,
                   help="grid resolution in degrees (0.00027778 = 30 m, "
                        "0.00083333 = 90 m). The single most consequential "
                        "setting; refit after changing it")
    g.add_argument("--block", type=int, help="tile size in pixels")

    g = p.add_argument_group("model options")
    g.add_argument("--trigger", choices=["rainfall", "earthquake"])
    g.add_argument("--pga", type=float, help="earthquake scenario, g")
    g.add_argument("--return-period", dest="return_period", type=float,
                   help="rainfall scenario, years")
    g.add_argument("--climate",
                   help="climate for a single run: 'current' or "
                        "'ssp585:2061-2080'")
    g.add_argument("--output", choices=["probability", "classes", "both"],
                   help="continuous failure probability, SINMAP classes, "
                        "or both (default)")
    g.add_argument("--calibration-regions", dest="calibration_regions",
                   choices=["lithology", "landcover"],
                   help="per-region soil parameters. Measured at -0.0004 AUC; "
                        "off by default")
    g.add_argument("--uniform-recharge", dest="uniform_recharge",
                   action="store_true",
                   help="hold recharge uniform, isolating terrain alone")
    g.add_argument("--samples", type=int,
                   help="Monte Carlo draws per pixel (default 200)")
    g.add_argument("--fitted-params", dest="fitted_params",
                   help="parameter JSON from step3 (default: from --name)")

    g = p.add_argument_group("data options")
    g.add_argument("--dem-source", dest="dem_source",
                   choices=["copernicus90", "copernicus30"])
    g.add_argument("--glim-grid", dest="glim_grid", action="store_true",
                   help="coarse 0.5-deg lithology instead of full GLiM")
    g.add_argument("--worldclim-res", dest="worldclim_res",
                   choices=["30s", "2.5m", "5m", "10m"])
    g.add_argument("--climate-model", dest="climate_model",
                   help=f"CMIP6 GCM (default {CL.DEFAULT_GCM})")
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
              "the whole area in memory in a single pass, so expect this to be "
              "slow and memory-hungry. Consider a smaller --bbox or a coarser "
              "--res.\n")


def _needs_fit(cfg: C.Config) -> bool:
    path = cfg.fitted_params or os.path.join(cfg.out_dir,
                                             f"{cfg.name}_fitted_params.json")
    return not os.path.exists(path)


# ---------------------------------------------------------------------------
# phase 1: set up
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
        pr = sources.download_worldclim_precip(cfg.data_dir,
                                               res=cfg.worldclim_res)
        print(f"  Precipitation        {len(pr)} months ({cfg.worldclim_res})")
        for spec in cfg.climate_suite:
            s = CL.scenario(spec, cfg.climate_model, cfg.climate_res)
            if s.is_baseline:
                continue
            fut = sources.download_worldclim_future(
                cfg.data_dir, ssp=s.ssp, period=s.period, model=s.gcm,
                res=s.resolution)
            print(f"  CMIP6 {s.key:<16} {len(fut) if fut else 0} months")

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


# ---------------------------------------------------------------------------
# phase 2: calibrate and validate
# ---------------------------------------------------------------------------

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
          "   (dimensionless: root + soil, over depth x unit weight)")
    print(f"    friction phi      {p['friction_deg'][0]:.1f} .. "
          f"{p['friction_deg'][1]:.1f} deg")
    print(f"    R/T               {p['rt'][0]:.2e} .. {p['rt'][1]:.2e} 1/m")
    if report.get("recharge_reference_mm"):
        print(f"    recharge ref      "
              f"{report['recharge_reference_mm']:.0f} mm wettest-month precip"
              "   (fixes what a multiplier of 1 means)")

    print(f"\n  In-sample AUC       {report['in_sample_auc']:.3f}   "
          f"({report['n_presence']} landslides, "
          f"{report['n_background']} background)")
    for scheme in ("random", "spatial"):
        cv = report.get(f"cv_{scheme}")
        if cv:
            mark = "  <- quote this" if scheme == "spatial" else ""
            print(f"  {scheme:<8} CV AUC     {cv['auc_mean']:.3f} +/- "
                  f"{cv['auc_std']:.3f}{mark}")

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
    print("\nNext:  step4-validate against an inventory this fit never saw")
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
              f"{args.out_dir}. Run step5-susceptibility first.")
        return 1
    print("STEP 4  Validate against a held-out inventory\n")
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
    print("\n  If this inventory was also used to fit, the numbers above are "
          "in-sample.\n  An independent inventory is what makes this step "
          "mean anything.")
    return 0


# ---------------------------------------------------------------------------
# phase 3: produce
# ---------------------------------------------------------------------------

def _report_outputs(out) -> None:
    print("\n  Outputs:")
    for k, v in out.items():
        if isinstance(v, str):
            print(f"    {k:22s} {v}")


def _step_susceptibility(args) -> int:
    cfg = _build_config(args)
    print("STEP 5  Susceptibility (SINMAP infinite slope + D-inf hydrology)\n")
    _warn_if_large(cfg)
    if _needs_fit(cfg):
        print("  note: no fitted parameters found, so SINMAP's generic ranges "
              "are used.\n        The pattern is meaningful; the level is not. "
              "Run step3-fit.\n")
    out = pipeline.run_susceptibility(cfg, mode=args.mode)
    _report_outputs(out)
    print("\n  Use the continuous probability raster. The six-class map is a "
          "legend:\n  its lower three bands all have failure probability zero "
          "and are not ordered.")
    return 0


def _step_hazard(args) -> int:
    cfg = _build_config(args)
    print("STEP 6  Hazard under triggering scenarios\n")
    _warn_if_large(cfg)
    if args.all:
        out = pipeline.run_hazard_suite(cfg, mode=args.mode)
        print("\n  Outputs:")
        for kind, runs in out.items():
            for key, paths in runs.items():
                print(f"    {kind:<11} {key:<8} {paths.get('probability')}")
        print("\n  Seismic scenarios depend on pga_fraction, which is a "
              "convention, not a fit.\n  Quote them as a range over it "
              "(0.3 to 1.0); rainfall needs no such hedge.")
    else:
        out = pipeline.run_hazard(cfg, mode=args.mode)
        _report_outputs(out)
    return 0


def _step_climate(args) -> int:
    cfg = _build_config(args)
    print("STEP 7  Climate scenarios: present day, futures, and the change\n")
    _warn_if_large(cfg)
    if _needs_fit(cfg):
        print("  note: no fitted parameters, so the present-day recharge "
              "reference is\n        measured from the baseline field rather "
              "than taken from a fit.\n        Changes between scenarios stay "
              "meaningful; the absolute level\n        does not. Run step3-fit "
              "for a calibrated run.\n")
    report = pipeline.run_climate(cfg, mode=args.mode, specs=args.scenarios)

    print("\n  scenario              mean P   unstable %   mean change   "
          "% more likely")
    print("  " + "-" * 72)
    for r in report["scenarios"]:
        print(f"  {r['scenario']:<20} {r['mean_probability']:>7.4f}   "
              f"{r['unstable_area_pct']:>9.2f}   {r['mean_change']:>+11.4f}   "
              f"{r['pct_more_likely']:>13.2f}")
    print("\n  Each future is normalised by the present-day recharge "
          "reference, so a\n  uniformly wetter projection shows as a shift "
          "rather than cancelling out.")
    print(f"\n  Summary -> {report['summary']}")
    return 0


# ---------------------------------------------------------------------------
# phase 4: package
# ---------------------------------------------------------------------------

def _step_package(args) -> int:
    cfg = _build_config(args)
    print("STEP 8  Package the deliverables\n")
    out = pipeline.run_package(cfg)

    import json
    with open(out["manifest"], encoding="utf-8") as fh:
        m = json.load(fh)
    cal = m["calibration"]
    print(f"  area        {m['area']['bbox']} at {m['area']['resolution_deg']} "
          f"deg ({m['area']['cells']:,} cells)")
    if cal.get("parameters"):
        print(f"  parameters  C {cal['parameters']['cohesion']}  "
              f"phi {cal['parameters']['friction_deg']}  "
              f"R/T {cal['parameters']['rt']}")
    if cal.get("held_out_auc"):
        print(f"  held-out    AUC {cal['held_out_auc']:.3f} "
              f"+/- {cal.get('held_out_auc_sd') or 0:.3f}  "
              f"({cal.get('n_presence')} landslides)")
    for kind, files in m["products"].items():
        if files:
            print(f"  {kind:<22} {len(files)}")
    for w in cal.get("warnings", []):
        print(f"\n  ! {w}")
    print(f"\n  Manifest -> {out['manifest']}")
    return 0


def _run_all(args) -> int:
    rc = _step_download(args)
    if rc:
        return rc
    print("\n" + "=" * 68 + "\n")
    cfg = _build_config(args)
    _warn_if_large(cfg)
    out = pipeline.run(cfg, mode=args.mode, climate_suite=not args.no_climate)
    print("\nOutputs:")
    for k, v in out.items():
        print(f"  {k:22s} {v if isinstance(v, str) else '(group)'}")
    return 0


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="giri_landslide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Hindu Kush Himalaya landslide model: SINMAP "
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

    p = sub.add_parser("step4-validate", aliases=["validate"],
                       help="test a map against a held-out inventory")
    p.add_argument("--susceptibility",
                   help="stability GeoTIFF (default: from --name)")
    p.add_argument("--inventory", required=True, nargs="+",
                   help="one or more INDEPENDENT inventories; multiple paths "
                        "are pooled into a single validation set")
    p.add_argument("--name")
    p.add_argument("--out-dir", dest="out_dir", default="outputs")
    p.add_argument("--block", type=int)

    p = sub.add_parser("step5-susceptibility",
                       aliases=["susceptibility", "stability"],
                       help="present-day failure probability")
    _mode(p); _add_common(p)

    p = sub.add_parser("step6-hazard", aliases=["hazard"],
                       help="failure probability under trigger scenarios")
    p.add_argument("--all", action="store_true",
                   help="every return period and PGA in the config, rather "
                        "than the single scenario named on the command line")
    _mode(p); _add_common(p)

    p = sub.add_parser("step7-climate", aliases=["climate"],
                       help="present and CMIP6 future climates, and the change")
    p.add_argument("--scenarios", nargs="+", metavar="SPEC",
                   help="e.g. current ssp245:2061-2080 ssp585:2081-2100 "
                        "(default: config climate_suite)")
    _mode(p); _add_common(p)

    p = sub.add_parser("step8-package", aliases=["package"],
                       help="write the manifest of products and provenance")
    _add_common(p)

    p = sub.add_parser("run-all", aliases=["run"],
                       help="every phase in sequence")
    p.add_argument("--no-inventories", action="store_true")
    p.add_argument("--no-climate", action="store_true",
                   help="skip the climate sweep")
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
        "step4-validate": _step_validate, "validate": _step_validate,
        "step5-susceptibility": _step_susceptibility,
        "susceptibility": _step_susceptibility,
        "stability": _step_susceptibility,
        "step6-hazard": _step_hazard, "hazard": _step_hazard,
        "step7-climate": _step_climate, "climate": _step_climate,
        "step8-package": _step_package, "package": _step_package,
        "run-all": _run_all, "run": _run_all,
    }
    return handlers[cmd](args)


if __name__ == "__main__":
    sys.exit(main())
