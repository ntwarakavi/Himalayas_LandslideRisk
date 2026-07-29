"""H-SIM command-line interface.

Himalayan Slope Instability Model - SINMAP infinite-slope stability over
D-infinity flow routing.

The product is region-wide: failure probability, exposed settlements and
exposed road segments for every mountain province in the Hindu Kush Himalaya,
present day and to 2060. Six commands get you there, in this order.

Each writes files and prints what it produced, so you can stop after any one,
inspect the output, and carry on. Nothing is re-downloaded and the expensive
stage is cached.

The number inside a command name is part of the name, not its position in the
sequence - they were numbered as they were added and never renumbered. Run them
top to bottom as listed here, not in numerical order.

     1  step1-check          what is cached, what is reachable, what it costs
     2  step2-download       fetch it
     3  step3-fit            fit the soil parameters, once, for the region
     4  step4-validate       check they travel: score against a SECOND
                             inventory. --build applies them over the held-out
                             inventory's own ground, which is the only honest
                             test of a regional extrapolation.
     5  step9-region         THE PRODUCT. Every mountain province in the Hindu
                             Kush Himalaya: susceptibility, trigger scenarios,
                             climate futures, settlement and road exposure and
                             a page each, plus one ranked index over the sweep.
                             --everything turns all of that on.
     6  step8-package        manifest: every product and its provenance

    run-all                  the whole thing, calibration through region
    info                     dataset sources, licences and citations

These run one area of interest at a time and step9-region calls them per
province, so they are for debugging a single province rather than for making
deliverables:

        step5-susceptibility present-day failure probability
        step6-hazard         rainfall and earthquake scenarios
        step7-climate        CMIP6 futures, and the change from today
        step10-risk          settlements and road segments, per climate
        step11-map           a browsable page of one run

Order matters. Do not ship maps from parameters that have not been through
step4 on an independent inventory - a fit always looks good on the landslides
it was fitted to, and the regional product applies one fit 95 times.
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
                        "'ssp585:2041-2060'")
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
    print("Next:  python -m h_sim.cli step2-download")
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

    print("\nNext:  python -m h_sim.cli step3-fit --inventory <path>")
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
    print("\nNext:  step4-validate against an inventory this fit never saw.")
    print("       Validation needs a map, and if the held-out inventory is in "
          "another\n       catchment it needs a map over *that* ground - "
          "which --build will make:\n")
    print(f"    python -m h_sim.cli step4-validate --build --name {cfg.name} \\"
          f"\n        --inventory <an inventory this fit never saw>")
    return 0


def _inventory_extent(paths, pad_deg: float = 0.02):
    """Bounding box of the inventories, padded. None if none of them load."""
    import numpy as np

    from .input import inventory

    lo = []
    for src in paths:
        try:
            pts = inventory.load_inventory(src)
        except Exception:                                # noqa: BLE001
            continue
        if len(pts):
            lo.append(pts)
    if not lo:
        return None
    pts = np.vstack(lo)
    return (float(pts[:, 0].min()) - pad_deg, float(pts[:, 1].min()) - pad_deg,
            float(pts[:, 0].max()) + pad_deg, float(pts[:, 1].max()) + pad_deg)


def _build_for_validation(args, bbox) -> Optional[str]:
    """Compute a susceptibility map over ``bbox`` from an existing fit.

    This is what transfer validation needs: the parameters fitted in one
    catchment, applied over another, then scored against that catchment's own
    inventory. Doing it by hand means calling step5 with a second name and
    remembering to pass --fitted-params, which is where the sequence usually
    goes wrong.
    """
    cfg = _build_config(args)
    cfg.bbox = tuple(bbox)
    if getattr(args, "res", None):
        cfg.resolution_deg = args.res
    if not cfg.fitted_params:
        cand = os.path.join(cfg.out_dir, f"{args.name}_fitted_params.json") \
            if args.name else None
        if cand and os.path.exists(cand):
            cfg.fitted_params = cand
    if not cfg.fitted_params or not os.path.exists(cfg.fitted_params):
        print("error: --build needs fitted parameters. Run step3-fit, or "
              "pass --fitted-params.")
        return None

    cfg.name = args.build_name or (f"{args.name}_on_target" if args.name
                                   else "validation")
    print(f"  building a map over the inventory's extent as '{cfg.name}'")
    print(f"  parameters from {cfg.fitted_params}\n")
    _warn_if_large(cfg)
    out = pipeline.run_susceptibility(cfg, mode=args.mode)
    print()
    return out.get("probability")


def _step_validate(args) -> int:
    import json

    import numpy as np

    from .input import inventory
    from .model import validate

    paths = args.inventory if isinstance(args.inventory, list) \
        else [args.inventory]
    extent = _inventory_extent(paths)

    susc = args.susceptibility
    if not susc and args.name:
        for suffix in ("_susceptibility_prob.tif", "_susceptibility_class.tif"):
            cand = os.path.join(args.out_dir, f"{args.name}{suffix}")
            if os.path.exists(cand):
                susc = cand
                break

    print("STEP 4  Validate against a held-out inventory\n")

    # A map that exists but sits somewhere else is the same problem as no map,
    # and the common one: inventories in this region are geographically
    # disjoint, so the map fitted on Gorkha does not cover Sikkim at all.
    elsewhere = False
    if susc and os.path.exists(susc) and extent:
        import rasterio
        with rasterio.open(susc) as src:
            b = src.bounds
        elsewhere = not (b.left < extent[2] and b.right > extent[0]
                         and b.bottom < extent[3] and b.top > extent[1])
        if elsewhere:
            print(f"  {os.path.basename(susc)} covers "
                  f"{b.left:.2f}, {b.bottom:.2f} to {b.right:.2f}, {b.top:.2f}")
            print(f"  the inventory covers "
                  f"{extent[0]:.2f}, {extent[1]:.2f} to "
                  f"{extent[2]:.2f}, {extent[3]:.2f}")
            print("  These do not overlap, so this map cannot be scored "
                  "against this inventory.\n")

    if not susc or not os.path.exists(susc) or elsewhere:
        if not extent:
            print(f"error: no usable map for '{args.name}', and the inventory "
                  "could not be read either.")
            return 1
        if args.build:
            susc = _build_for_validation(args, extent)
            if not susc:
                return 1
        else:
            if not elsewhere:
                print(f"  no stability map found for '{args.name}' in "
                      f"{args.out_dir}.\n")
            print("  Validating a fit against an inventory somewhere else is "
                  "transfer\n  validation, and it needs a map over *that* "
                  "ground, built with the\n  parameters you are testing. "
                  "Add --build to do it here:\n")
            res = getattr(args, "res", None) or 0.00083333
            print(f"    python -m h_sim.cli step4-validate --build \\\n"
                  f"        --name {args.name or '<fitted run>'} "
                  f"--res {res:g} \\\n"
                  f"        --inventory {' '.join(paths)}\n")
            print("  or do the two steps yourself:\n")
            print(f"    python -m h_sim.cli step5-susceptibility "
                  f"--name {(args.name or 'run')}_on_target \\\n"
                  f"        --bbox {extent[0]:.2f} {extent[1]:.2f} "
                  f"{extent[2]:.2f} {extent[3]:.2f} --res {res:g} \\\n"
                  f"        --fitted-params {args.out_dir}/"
                  f"{args.name or '<run>'}_fitted_params.json")
            print(f"    python -m h_sim.cli step4-validate "
                  f"--name {(args.name or 'run')}_on_target \\\n"
                  f"        --inventory {' '.join(paths)}")
            return 1

    kind = ("continuous failure probability" if validate.is_continuous(susc)
            else "stability classes")
    print(f"  map       : {susc}  ({kind})")
    print("  inventories:")

    import rasterio
    with rasterio.open(susc) as src:
        b = src.bounds
        bbox = (b.left, b.bottom, b.right, b.top)
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

    reference = susc
    if args.survey_extent:
        masked = os.path.join(args.out_dir,
                              f"{args.name or 'validation'}_surveyed.tif")
        reference, frac = inventory.survey_masked_reference(
            args.survey_extent, susc, masked)
        print(f"  surveyed extent covers {frac * 100:.1f}% of the map; "
              "background is drawn only inside it\n")

    bg = inventory.background_points(bbox, max(4 * len(pts), 2000), reference)
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


def _step_region(args) -> int:
    cfg = _build_config(args)
    print("STEP 9  Regional sweep, one state or province at a time\n")
    if _needs_fit(cfg) and not args.dry_run:
        print("  note: no fitted parameters found, so every unit uses SINMAP's "
              "generic\n        ranges. Calibrate once for the region "
              "(step3-fit) before a real sweep.\n")
    report = pipeline.run_region(
        cfg, mode=args.mode, countries=args.countries, names=args.units,
        hazard=args.with_hazard or args.everything,
        climate=args.with_climate or args.everything,
        risk=args.with_risk or args.everything,
        webmap=args.with_map or args.everything,
        dry_run=args.dry_run,
        resume=not args.no_resume)

    if args.dry_run:
        print(f"\n  {report['n_units_found']} units, "
              f"{report['n_units_runnable']} runnable at "
              f"{report['resolution_deg']} deg\n")
        print("  cells (M)  country / unit")
        print("  " + "-" * 56)
        for r in report["plan"][:25]:
            flag = "  SKIP" if r["cells"] > cfg.admin_max_cells else ""
            print(f"  {r['cells'] / 1e6:>9.1f}  {r['country']} / "
                  f"{r['name']}{flag}")
        if len(report["plan"]) > 25:
            print(f"  ... and {len(report['plan']) - 25} more")
        print(f"\n  Plan -> {report['summary']}")
        return 0

    rows = sorted((u for u in report["units"] if u.get("stats")),
                  key=lambda u: -u["stats"]["unstable_area_pct"])
    if rows:
        print("\n  unstable %   mean P   unit")
        print("  " + "-" * 56)
        for u in rows[:20]:
            s = u["stats"]
            print(f"  {s['unstable_area_pct']:>9.2f}   {s['mean_probability']:>6.4f}   "
                  f"{u['unit']['country']} / {u['unit']['name']}")
        if len(rows) > 20:
            print(f"  ... and {len(rows) - 20} more")
    if report.get("failed"):
        print(f"\n  {len(report['failed'])} unit(s) failed:")
        for f in report["failed"][:5]:
            print(f"    {f['unit']['name']}: {f['error'][:70]}")
    print(f"\n  Summary -> {report['summary']}")
    if report.get("index"):
        print(f"  Index   -> {report['index']}")
        print(f"  Open it :  file://{os.path.abspath(report['index'])}")
    print("\n  Each unit was routed over its bounding box plus a "
          f"{cfg.admin_buffer_deg} deg buffer\n  and clipped back afterwards, "
          "so catchments are not truncated at borders.")
    return 0


def _step_risk(args) -> int:
    cfg = _build_config(args)
    print("STEP 10  Exposure of settlements and roads\n")
    out = pipeline.run_risk(cfg, mode=args.mode,
                            susceptibility=args.susceptibility,
                            climate=args.risk_climate)
    s = out["stats"]
    scen = s.get("scenarios") or {}
    base = s.get("baseline", "current")

    print(f"\n  Settlements assessed   {s['n_settlements']:,}")
    print(f"  Road segments          {s['n_road_segments']:,}"
          f"   ({s['road_km_total']:,.0f} km)")

    hdr = f"\n  {'scenario':<18}{'exposed':>9}{'people':>10}" \
          f"{'road km':>10}{'road %':>9}{'mean':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    for key, st in scen.items():
        tag = f"{key} *" if key == base else key
        print(f"  {tag:<18}{st['n_settlements_exposed']:>9,}"
              f"{st['population_exposed']:>10,}"
              f"{st['road_km_exposed']:>10,.0f}"
              f"{st['road_pct_exposed']:>9.1f}"
              f"{st['mean_settlement_score']:>8.3f}")
    print("  * present day. 'exposed' means a score at or above "
          f"{s.get('exposed_threshold', 0.08)}.")

    if s.get("change"):
        print("\n  Change from the present day")
        for key, ch in s["change"].items():
            print(f"    {key:<18}{ch['settlements_exposed']:>+7,} settlements"
                  f"{ch['road_km_exposed']:>+9,.1f} km"
                  f"{ch['mean_settlement_score']:>+9.4f} mean score")

    base_bands = (scen.get(base) or s).get("settlements_by_band", {})
    print("\n  Present-day settlement bands")
    for b in ("very high", "high", "moderate", "low", "very low"):
        n = base_bands.get(b)
        if n:
            print(f"    {b:<12} {int(n):>6,}")

    print(f"\n  Angle of reach {s['travel_angle_deg']} deg, "
          f"search radius {s['reach_radius_m']:.0f} m")
    print("\n  This is screening, not risk: assets are scored by the "
          "proximity-weighted\n  fraction of upslope ground that could reach "
          "them and that the model calls\n  unstable. There is no runout "
          "model and no vulnerability or damage function.")
    print(f"\n  Settlements -> {out['settlements']}")
    print(f"  Roads       -> {out['roads']}")
    print("\nNext:  python -m h_sim.cli step11-map "
          f"--name {cfg.name}")
    return 0


def _step_map(args) -> int:
    cfg = _build_config(args)
    print("STEP 11  Build the web map\n")
    out = pipeline.run_webmap(cfg, susceptibility=args.susceptibility,
                              open_after=args.open)
    print(f"\n  Open it:  file://{os.path.abspath(out['webmap'])}")
    print("\n  Leaflet and the basemap tiles load from the network; the "
          "model outputs\n  themselves sit next to the page and load from "
          "disk.")
    return 0


def _run_all(args) -> int:
    rc = _step_download(args)
    if rc:
        return rc
    print("\n" + "=" * 68 + "\n")
    cfg = _build_config(args)
    _warn_if_large(cfg)
    out = pipeline.run(cfg, mode=args.mode, climate_suite=not args.no_climate,
                       region=not args.single_area)
    print("\nOutputs:")
    for k, v in out.items():
        print(f"  {k:22s} {v if isinstance(v, str) else '(group)'}")
    return 0


# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="h-sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="H-SIM - Himalayan Slope Instability Model. SINMAP "
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
    p.add_argument("--build", action="store_true",
                   help="if no map covers the inventory, compute one over its "
                        "extent from --fitted-params first. This is transfer "
                        "validation: the parameters fitted in one catchment, "
                        "scored in another")
    p.add_argument("--build-name", dest="build_name",
                   help="name for the map --build creates "
                        "(default: <name>_on_target)")
    p.add_argument("--survey-extent", dest="survey_extent",
                   help="polygon of the ground the inventory's authors "
                        "actually surveyed. Background is drawn only inside "
                        "it; without this, unmapped terrain is scored as "
                        "landslide-free and the AUC comes out low")
    p.add_argument("--res", type=float,
                   help="grid resolution for --build, in degrees")
    p.add_argument("--fitted-params", dest="fitted_params",
                   help="parameter JSON to test (default: from --name)")
    p.add_argument("--config", help="JSON config file, for --build settings")
    p.add_argument("--mode", choices=["demo", "download", "local"],
                   default="download")
    p.add_argument("--name")
    p.add_argument("--out-dir", dest="out_dir", default="outputs")
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--work-dir", dest="work_dir")
    p.add_argument("--dem-source", dest="dem_source",
                   choices=["copernicus90", "copernicus30"])
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
                   help="e.g. current ssp245:2041-2060 ssp585:2081-2100 "
                        "(default: config climate_suite)")
    _mode(p); _add_common(p)

    p = sub.add_parser("step9-region", aliases=["region"],
                       help="sweep the region one state or province at a time")
    p.add_argument("--dry-run", action="store_true",
                   help="list the units and their cost, run nothing")
    p.add_argument("--countries", nargs="+",
                   help="restrict to these countries (default: all HKH)")
    p.add_argument("--units", nargs="+",
                   help="restrict to these state/province names")
    p.add_argument("--with-hazard", dest="with_hazard", action="store_true",
                   help="also run every trigger scenario per unit")
    p.add_argument("--with-climate", dest="with_climate", action="store_true",
                   help="also run the climate sweep per unit")
    p.add_argument("--with-risk", dest="with_risk", action="store_true",
                   help="also score settlements and roads per unit, under "
                        "every climate in risk_climate")
    p.add_argument("--with-map", dest="with_map", action="store_true",
                   help="also build a browsable page per unit. The ranked "
                        "index over all units is written either way")
    p.add_argument("--everything", action="store_true",
                   help="shorthand for --with-hazard --with-climate "
                        "--with-risk --with-map")
    p.add_argument("--no-resume", action="store_true",
                   help="redo units that already have outputs")
    _mode(p); _add_common(p)

    p = sub.add_parser("step10-risk", aliases=["risk"],
                       help="score settlements and roads by the "
                            "susceptibility that can reach them")
    p.add_argument("--susceptibility",
                   help="present-day susceptibility GeoTIFF "
                        "(default: from --name)")
    p.add_argument("--risk-climate", nargs="+", metavar="SPEC",
                   help="climates to score assets under, e.g. current "
                        "ssp245:2021-2040 ssp585:2041-2060. The present day "
                        "is always included. Default: config.risk_climate")
    _mode(p); _add_common(p)

    p = sub.add_parser("step11-map", aliases=["map", "webmap"],
                       help="build a browsable Leaflet page for a run")
    p.add_argument("--susceptibility",
                   help="susceptibility GeoTIFF (default: from --name)")
    p.add_argument("--open", action="store_true",
                   help="open the page in a browser when it is written")
    _add_common(p)

    p = sub.add_parser("step8-package", aliases=["package"],
                       help="write the manifest of products and provenance")
    _add_common(p)

    p = sub.add_parser("run-all", aliases=["run"],
                       help="every phase in sequence")
    p.add_argument("--no-inventories", action="store_true")
    p.add_argument("--no-climate", action="store_true",
                   help="skip the climate sweep")
    p.add_argument("--single-area", dest="single_area", action="store_true",
                   help="produce for the config's bbox instead of sweeping "
                        "the region. For debugging, not for deliverables")
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
        "step9-region": _step_region, "region": _step_region,
        "step10-risk": _step_risk, "risk": _step_risk,
        "step11-map": _step_map, "map": _step_map, "webmap": _step_map,
        "run-all": _run_all, "run": _run_all,
    }
    return handlers[cmd](args)


if __name__ == "__main__":
    sys.exit(main())
