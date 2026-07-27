"""Command-line interface for the GIRI landslide model.

Examples
--------
    # Run the whole model on synthetic data (no downloads):
    python -m giri_landslide.cli run --mode demo

    # Run on real open data for an AOI (needs network):
    python -m giri_landslide.cli run --mode download \
        --bbox 83.0 27.5 85.0 29.0 --res 0.0025 --trigger rainfall

    # Earthquake scenario from a config file:
    python -m giri_landslide.cli run --config examples/himalayas_eq.json

    # Only fetch data for an AOI:
    python -m giri_landslide.cli download --bbox 83 27.5 85 29
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import config as C
from . import pipeline, sources


def _build_config(args: argparse.Namespace) -> C.Config:
    cfg = C.Config.from_json(args.config) if args.config else C.Config()
    if args.name:
        cfg.name = args.name
    if args.bbox:
        cfg.bbox = tuple(args.bbox)
    if args.res:
        cfg.resolution_deg = args.res
    if args.trigger:
        cfg.trigger = args.trigger
    if args.block:
        cfg.block_size = args.block
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.dem_source:
        cfg.dem_source = args.dem_source
    if args.pga is not None:
        cfg.scenario_pga_g = args.pga
    if args.return_period is not None:
        cfg.scenario_return_period_yr = args.return_period
    if getattr(args, "inventory", None):
        cfg.inventory_path = args.inventory
    if getattr(args, "weight_mode", None):
        cfg.weight_mode = args.weight_mode
    if getattr(args, "classification", None):
        cfg.classification = args.classification
    if cfg.trigger == "earthquake" and not args.no_eq_preset:
        cfg.weights.soil_moisture = min(cfg.weights.soil_moisture, 0.5)
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="JSON config file")
    p.add_argument("--name", help="run/output label")
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("W", "S", "E", "N"), help="AOI bounding box (deg)")
    p.add_argument("--res", type=float, help="grid resolution (deg)")
    p.add_argument("--trigger", choices=["rainfall", "earthquake"])
    p.add_argument("--block", type=int, help="processing block size (px)")
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--work-dir", dest="work_dir")
    p.add_argument("--out-dir", dest="out_dir")
    p.add_argument("--dem-source", dest="dem_source",
                   choices=["copernicus90", "copernicus30"])
    p.add_argument("--pga", type=float, help="uniform PGA scenario (g)")
    p.add_argument("--return-period", dest="return_period", type=float,
                   help="uniform rainfall return period scenario (yr)")
    p.add_argument("--no-eq-preset", dest="no_eq_preset", action="store_true",
                   help="do not auto-reduce soil-moisture weight for earthquake")
    p.add_argument("--inventory", help="historical landslide inventory CSV/GeoJSON")
    p.add_argument("--weight-mode", dest="weight_mode",
                   choices=["multiplicative", "exponent"])
    p.add_argument("--classification", choices=["fixed", "quantile"])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="giri_landslide",
        description="Open-source GIRI-style landslide hazard model (local).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline")
    p_run.add_argument("--mode", choices=["demo", "download", "local"],
                       default="demo")
    _add_common(p_run)

    p_dl = sub.add_parser("download", help="download open datasets for an AOI")
    _add_common(p_dl)

    p_cal = sub.add_parser(
        "calibrate",
        help="fine-tune factor weights against a Himalayan landslide inventory")
    p_cal.add_argument("--mode", choices=["demo", "download", "local"],
                       default="demo")
    _add_common(p_cal)

    p_info = sub.add_parser("info", help="print dataset source information")

    args = parser.parse_args(argv)

    if args.command == "info":
        print(sources.GLIM_SOURCE_INFO)
        print(sources.PGA_SOURCE_INFO)
        return 0

    cfg = _build_config(args)

    if args.command == "download":
        paths = sources.download_dem(cfg.bbox, cfg.data_dir, cfg.dem_source)
        print(f"DEM tiles: {len(paths)}")
        if cfg.landcover_source == "worldcover":
            lc = sources.download_worldcover(cfg.bbox, cfg.data_dir)
            print(f"Land cover tiles: {len(lc)}")
        if cfg.trigger == "rainfall":
            pr = sources.download_worldclim_precip(cfg.data_dir)
            print(f"WorldClim precip tiles: {len(pr)}")
        return 0

    if args.command == "calibrate":
        report = pipeline.run_calibration(cfg, mode=args.mode)
        res = report["result"]
        print("\nCalibrated exponent weights (factor influence):")
        for k, v in res["weights"].items():
            print(f"  {k:14s} {v:6.3f}")
        print(f"\nHeld-out ROC AUC : {res['auc']:.3f}  "
              f"(train {res['auc_train']:.3f})")
        print(f"Presence / background points: {res['n_presence']} / "
              f"{res['n_background']}")
        print(f"\nCalibrated config : {report['calibrated_config']}")
        print(f"Full report       : {report['report']}")
        print("\nRun the model with the calibrated weights:")
        print(f"  python -m giri_landslide.cli run --mode {args.mode} "
              f"--config {report['calibrated_config']}")
        return 0

    if args.command == "run":
        outputs = pipeline.run(cfg, mode=args.mode)
        print("\nOutputs:")
        for k, v in outputs.items():
            print(f"  {k:20s} {v}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
