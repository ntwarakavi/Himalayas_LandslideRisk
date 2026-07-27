"""Command-line interface, organised as a numbered workflow.

    step1  check          which datasets do I have, and can I get the rest?
    step2  download       fetch everything (skips anything already cached)
    step3  calibrate      fit the factor weights to real landslides
    step4  susceptibility WHERE is the ground fragile?
    step5  hazard         IF a storm/quake hits, how likely is a landslide?
    step6  validate       does the map hold up on landslides it never saw?
    step7  compare        how does that change between two scenarios?

    run-all               step2 -> step4 -> step5 in one go
    info                  dataset sources and licences

Every step writes files and prints what it produced, so you can stop after any
step, look at the output, and carry on. Nothing is ever re-downloaded.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import config as C
from . import datasets, pipeline, sources


# ---------------------------------------------------------------------------
# config assembly
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace) -> C.Config:
    cfg = C.Config.from_json(args.config) if getattr(args, "config", None) \
        else C.Config()
    simple = ["name", "trigger", "data_dir", "work_dir", "out_dir",
              "dem_source", "weight_mode", "classification", "output", "climate",
              "climate_period", "climate_model", "worldclim_res"]
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
    if cfg.trigger == "earthquake" and not getattr(args, "no_eq_preset", False):
        cfg.weights.soil_moisture = min(cfg.weights.soil_moisture, 0.5)
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("area and grid")
    g.add_argument("--config", help="JSON config file (see examples/)")
    g.add_argument("--name", help="run label; prefixes every output file")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="area of interest in degrees")
    g.add_argument("--res", type=float,
                   help="grid resolution in degrees (0.0008333 = 90 m)")
    g.add_argument("--block", type=int, help="tile size in pixels")

    g = p.add_argument_group("model options")
    g.add_argument("--trigger", choices=["rainfall", "earthquake"])
    g.add_argument("--pga", type=float, help="earthquake scenario, g")
    g.add_argument("--return-period", dest="return_period", type=float,
                   help="rainfall scenario, years")
    g.add_argument("--weight-mode", dest="weight_mode",
                   choices=["multiplicative", "exponent"])
    g.add_argument("--classification", choices=["fixed", "quantile"])
    g.add_argument("--output", choices=["probability", "classes", "both"],
                   help="continuous 0-1 index (default), 5 classes, or both")
    g.add_argument("--no-eq-preset", dest="no_eq_preset", action="store_true",
                   help="keep the full soil-moisture weight for earthquakes")

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
    g.add_argument("--inventory", help="landslide inventory (.shp/.kml/.csv/.geojson)")
    g.add_argument("--data-dir", dest="data_dir")
    g.add_argument("--work-dir", dest="work_dir")
    g.add_argument("--out-dir", dest="out_dir")


def _mode(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["demo", "download", "local"],
                   default="download",
                   help="demo = synthetic offline data; download = real data")


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
    from . import inventory

    got = sources.download_dem(bbox, cfg.data_dir, cfg.dem_source)
    print(f"  DEM tiles            {len(got)}")
    lc = sources.download_worldcover(bbox, cfg.data_dir)
    print(f"  Land cover tiles     {len(lc)}")

    if cfg.climate == "current":
        pr = sources.download_worldclim_precip(cfg.data_dir,
                                               res=cfg.worldclim_res)
        print(f"  Precipitation        {len(pr)} months ({cfg.worldclim_res})")
    else:
        pr = sources.download_worldclim_future(
            cfg.data_dir, ssp=cfg.climate, period=cfg.climate_period,
            model=cfg.climate_model, res=cfg.climate_res)
        print(f"  CMIP6 precipitation  {len(pr) if pr else 0} months "
              f"({cfg.climate} {cfg.climate_period})")

    if cfg.glim_full:
        gdb = sources.download_glim_vector(cfg.data_dir)
        print(f"  GLiM lithology       {'full geodatabase' if gdb else 'FAILED'}")
    else:
        print(f"  GLiM lithology       {sources.download_glim_grid(cfg.data_dir)}")

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

    print("\nNext:  python -m giri_landslide.cli step3-calibrate  "
          "(or step4-susceptibility to use default weights)")
    return 0


def _step_calibrate(args) -> int:
    cfg = _build_config(args)
    print("STEP 3  Calibrate factor weights against real landslides\n")
    report = pipeline.run_calibration(
        cfg, mode=args.mode,
        fit_slope_breaks=not args.no_slope_breaks,
        fit_lithology=getattr(args, "fit_lithology", False))
    res = report["result"]

    print("\n  Fitted weights (how much each factor matters):")
    for k, v in res["weights"].items():
        bar = "#" * int(round(v * 8))
        print(f"    {k:14s} {v:6.3f}  {bar}")
    print(f"\n  Cross-validated AUC : {res['auc']:.3f} +/- {res['auc_std']:.3f}")
    print("    0.5 = no skill, 0.7 = fair, 0.8 = good, 0.9 = excellent")
    print(f"  Landslides used     : {res['n_presence']}")
    for w in res.get("warnings", []):
        print(f"    ! {w}")
    if res.get("slope_breaks"):
        print("\n  Fitted slope classes (degrees -> factor 0-5):")
        lo = 0.0
        for hi, sc in res["slope_breaks"]:
            hs = "inf" if hi == float("inf") else f"{hi:.1f}"
            print(f"    {lo:5.1f} - {hs:>5s}  {'*' * sc}")
            lo = hi
    if res.get("lithology_diagnostics"):
        ld = res["lithology_diagnostics"]
        print("\n  Fitted rock-type factors (expert -> fitted, by landslide"
              " over-representation):")
        for code, fr in sorted(ld["frequency_ratio"].items(),
                               key=lambda kv: -kv[1]):
            exp, fit = ld["expert"][code], ld["fitted"][code]
            flag = "  <-- changed" if exp != fit else ""
            print(f"    {code}  ratio={fr:5.2f}   {exp} -> {fit}{flag}")
    print(f"\n  Calibrated config -> {report['calibrated_config']}")
    print(f"  Full report       -> {report['report']}")
    print("\nNext:  python -m giri_landslide.cli step4-susceptibility "
          f"--config {report['calibrated_config']}")
    return 0


def _step_susceptibility(args) -> int:
    cfg = _build_config(args)
    print("STEP 4  Susceptibility - where is the ground fragile?\n")
    out = pipeline.run_susceptibility(cfg, mode=args.mode)
    print("\n  Outputs:")
    for k, v in out.items():
        print(f"    {k:22s} {v}")
    print("\nNext:  python -m giri_landslide.cli step5-hazard"
          + (f" --config {args.config}" if getattr(args, "config", None) else ""))
    return 0


def _step_hazard(args) -> int:
    cfg = _build_config(args)
    susc = args.susceptibility or os.path.join(
        cfg.out_dir, f"{cfg.name}_susceptibility.tif")
    if not os.path.exists(susc):
        print(f"error: susceptibility class map not found: {susc}\n"
              "Run step4-susceptibility first, or pass --susceptibility PATH.\n"
              "(The hazard matrix is indexed by class, so step 5 needs the "
              "class map rather than the continuous index.)")
        return 1
    print("STEP 5  Hazard - how likely is a landslide in this scenario?\n")
    print(f"  using susceptibility: {susc}")
    out = pipeline.run_hazard(cfg, susc, mode=args.mode)
    print("\n  Outputs:")
    for k, v in out.items():
        print(f"    {k:22s} {v}")
    try:
        png = pipeline.quicklook(susc, out["hazard_probability"],
                                 os.path.join(cfg.out_dir,
                                              f"{cfg.name}_quicklook.png"))
        print(f"    {'quicklook':22s} {png}")
    except Exception as exc:                              # noqa: BLE001
        print(f"    quicklook skipped ({exc})")
    return 0


def _step_validate(args) -> int:
    import json

    import numpy as np

    from . import inventory, validate

    susc = args.susceptibility or os.path.join(
        args.out_dir, f"{args.name}_susceptibility.tif")
    if not os.path.exists(susc):
        print(f"error: susceptibility map not found: {susc}")
        return 1
    print("STEP 6  Validate against a held-out inventory\n")
    print(f"  map       : {susc}")
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
    label = args.name or os.path.basename(susc).replace("_susceptibility.tif", "")
    path = os.path.join(args.out_dir, f"{label}_validation.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    print(f"\n  Report -> {path}")
    return 0


def _step_compare(args) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    print("STEP 6  Compare two scenarios\n")
    out = pipeline.compare_susceptibility(
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
        description="Hindu Kush Himalaya landslide susceptibility and hazard "
                    "model (GIRI/NGI method).",
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

    p = sub.add_parser("step3-calibrate", aliases=["calibrate"],
                       help="fit factor weights to a landslide inventory")
    p.add_argument("--no-slope-breaks", action="store_true",
                   help="do not refit the slope table")
    p.add_argument("--fit-lithology", action="store_true",
                   help="also fit the rock-type factor from the inventory "
                        "(needs the full GLiM geodatabase)")
    _mode(p); _add_common(p)

    p = sub.add_parser("step4-susceptibility", aliases=["susceptibility"],
                       help="build the susceptibility map")
    _mode(p); _add_common(p)

    p = sub.add_parser("step5-hazard", aliases=["hazard"],
                       help="apply a trigger scenario to get probabilities")
    p.add_argument("--susceptibility",
                   help="susceptibility GeoTIFF (default: from --name)")
    _mode(p); _add_common(p)

    p = sub.add_parser("step6-validate", aliases=["validate"],
                       help="test a susceptibility map against a held-out "
                            "inventory (ideally another region)")
    p.add_argument("--susceptibility",
                   help="susceptibility GeoTIFF (default: from --name)")
    p.add_argument("--inventory", required=True, nargs="+",
                   help="one or more INDEPENDENT inventories; multiple paths "
                        "are pooled into a single validation set")
    p.add_argument("--name")
    p.add_argument("--out-dir", dest="out_dir", default="outputs")
    p.add_argument("--block", type=int)

    p = sub.add_parser("step7-compare", aliases=["compare"],
                       help="difference two susceptibility maps")
    p.add_argument("--baseline", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--name")
    p.add_argument("--out-dir", dest="out_dir", default="outputs")
    p.add_argument("--block", type=int)

    p = sub.add_parser("run-all", aliases=["run"],
                       help="download + susceptibility + hazard in one go")
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
        "step3-calibrate": _step_calibrate, "calibrate": _step_calibrate,
        "step4-susceptibility": _step_susceptibility,
        "susceptibility": _step_susceptibility,
        "step5-hazard": _step_hazard, "hazard": _step_hazard,
        "step6-validate": _step_validate, "validate": _step_validate,
        "step7-compare": _step_compare, "compare": _step_compare,
        "run-all": _run_all, "run": _run_all,
    }
    return handlers[cmd](args)


if __name__ == "__main__":
    sys.exit(main())
