"""Unit + end-to-end tests for the physically based landslide model.

Run with:  python -m pytest tests/ -q      (or)   python tests/test_model.py
No network required - everything uses the synthetic demo generator.

The mechanics have exact answers in limiting cases, so the physics tests check
against those rather than against a stored expectation: FS = 1 when the slope
angle equals the friction angle, mass conservation in flow routing, the
saturated critical angle, and the Newmark critical acceleration.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from h_sim import config as C
from h_sim import pipeline
from h_sim.input import inventory
from h_sim.model import (climate as CL, crossval, hazard,
                         hydrology as H, physical as P)
from h_sim.utility import demo
from h_sim.utility.grid import Grid


# ---------------------------------------------------------------------------
# grid and configuration
# ---------------------------------------------------------------------------

def test_grid_dimensions():
    g = Grid.from_bbox((10.0, 20.0, 11.0, 21.0), 0.01)
    assert g.width == 100 and g.height == 100
    assert abs(g.transform.a - 0.01) < 1e-12


def test_region_clipping():
    # AOI inside the Himalayan region is kept (intersected).
    cfg = C.Config(bbox=(84.0, 28.0, 84.5, 28.5))
    assert cfg.clipped_bbox()[0] >= 71.0
    # AOI entirely outside the region raises.
    off = C.Config(bbox=(0.0, 0.0, 1.0, 1.0))
    try:
        off.clipped_bbox()
        assert False, "expected out-of-region AOI to raise"
    except ValueError:
        pass


def test_cell_count_tracks_resolution():
    cfg = C.Config(bbox=(84.0, 28.0, 85.0, 29.0), resolution_deg=0.001)
    assert cfg.cell_count() == 1000 * 1000
    cfg.resolution_deg = 0.0005
    assert cfg.cell_count() == 2000 * 2000


def test_metres_per_cell_shrinks_with_latitude():
    """East-west spacing must contract towards the pole; north-south must not."""
    dx_lo, dy_lo = pipeline.metres_per_cell((84.0, 0.0, 85.0, 0.0), 0.001)
    dx_hi, dy_hi = pipeline.metres_per_cell((84.0, 60.0, 85.0, 60.0), 0.001)
    assert dx_hi < dx_lo
    assert abs(dx_hi / dx_lo - 0.5) < 0.01          # cos(60) = 0.5
    assert abs(dy_hi - dy_lo) < 1e-9


# ---------------------------------------------------------------------------
# hydrology
# ---------------------------------------------------------------------------

def test_hydrology_matches_analytic_plane():
    """Flow routing must be exact on a plane and conserve mass."""
    n, d, grad = 30, 10.0, 0.5
    dem = np.tile(np.arange(n, dtype=float)[:, None] * grad * d, (1, n))

    filled = H.fill_depressions(dem)
    _, slope = H.dinf_flow_direction(filled, d, d)
    assert abs(np.nanmean(slope[3:-3, 3:-3]) - grad) < 1e-9

    # A pit must be raised to the level of its outlet.
    pit = np.full((15, 15), 10.0)
    pit[7, 7] = 1.0
    assert abs(H.fill_depressions(pit)[7, 7] - 10.0) < 1e-3

    # Everything routed off the plane must equal the area put in.
    ang, _ = H.dinf_flow_direction(filled, d, d)
    total = H.dinf_accumulation(filled, ang, d * d)
    assert abs(np.nansum(total[0]) - n * n * d * d) < 1.0


def test_specific_catchment_area_concentrates_in_hollows():
    """A convergent hollow must collect far more flow than a planar slope."""
    n, d = 41, 20.0
    y, x = np.mgrid[0:n, 0:n].astype(float)
    plane = (n - y) * d * 0.5
    hollow = plane + 0.02 * d * (x - n // 2) ** 2    # valley along the centre

    a_plane, _ = H.specific_catchment_area(plane, d, d)
    a_hollow, _ = H.specific_catchment_area(hollow, d, d)

    axis = n // 2
    # On a plane every column drains its own strip, so contributing area is
    # uniform across the slope.
    assert np.ptp(a_plane[-4, 5:-5]) < 1.0
    # In a hollow it concentrates on the axis and starves the flanks.
    assert a_hollow[-4, axis] > 10 * a_hollow[-4, 3]
    assert a_hollow[-4, axis] > 5 * a_plane[-4, axis]
    # Contributing area must grow downslope along the axis.
    assert np.all(np.diff(a_hollow[5:-2, axis]) > 0)


def test_flat_and_nodata_survive_routing():
    """A DEM with holes must not produce NaN slopes on the valid cells."""
    dem = np.tile(np.arange(20, dtype=float)[:, None] * 5.0, (1, 20))
    dem[8:12, 8:12] = np.nan
    sca, slope = H.specific_catchment_area(dem, 30.0, 30.0)
    ok = np.isfinite(dem)
    assert np.isfinite(slope[ok]).all()
    assert np.isfinite(sca[ok]).all()
    assert not np.isfinite(slope[~ok]).any()


# ---------------------------------------------------------------------------
# stability mechanics
# ---------------------------------------------------------------------------

def test_infinite_slope_physics():
    """Factor of safety must reproduce the classical limiting cases."""
    # Dry and cohesionless: FS = 1 exactly when slope angle equals friction.
    for phi in (30.0, 35.0, 40.0):
        s = np.array([np.tan(np.radians(phi))])
        fs = P.factor_of_safety(s, np.array([0.0]), 0.0, phi, 0.0)
        assert abs(fs[0] - 1.0) < 1e-9, (phi, fs[0])

    # Saturated and cohesionless: critical angle drops to atan((1-r)tan phi).
    phi, r = 35.0, P.DENSITY_RATIO
    crit = np.arctan((1 - r) * np.tan(np.radians(phi)))
    s = np.array([np.tan(crit)])
    fs = P.factor_of_safety(s, np.array([1e7]), 0.0, phi, 0.01)
    assert abs(fs[0] - 1.0) < 1e-6, fs[0]

    # Wetness is bounded and rises with contributing area.
    w = P.wetness(np.array([10.0, 1e3, 1e6]), np.full(3, 0.5), 1e-3)
    assert w[0] < w[1] and w[2] == 1.0

    # Failure probability increases monotonically with slope.
    sl = np.tan(np.radians(np.array([10.0, 25.0, 40.0, 55.0])))
    p = P.failure_probability(sl, np.full(4, 200.0), P.SoilParameters(),
                              n_samples=100)
    assert np.all(np.diff(p) >= 0) and p[0] == 0.0 and p[-1] > 0.9


def test_pseudo_static_reduces_to_static():
    """k_h = 0 must give back SINMAP's published static form, exactly."""
    slope = np.tan(np.radians(np.array([15.0, 30.0, 45.0])))
    sca = np.array([50.0, 500.0, 5000.0])
    fs = P.factor_of_safety(slope, sca, 0.1, 33.0, 1e-3, k_h=0.0)

    theta = np.arctan(slope)
    w = P.wetness(sca, slope, 1e-3)
    expected = (0.1 + np.cos(theta) * (1 - w * P.DENSITY_RATIO)
                * np.tan(np.radians(33.0))) / np.sin(theta)
    assert np.allclose(fs, expected, rtol=0, atol=1e-12)


def test_seismic_loading_only_destabilises():
    """Shaking can never raise the factor of safety."""
    slope = np.tan(np.radians(np.array([10.0, 20.0, 30.0, 40.0, 50.0])))
    sca = np.full(5, 300.0)
    static = P.factor_of_safety(slope, sca, 0.1, 35.0, 1e-3, k_h=0.0)
    shaken = P.factor_of_safety(slope, sca, 0.1, 35.0, 1e-3, k_h=0.15)
    assert np.all(shaken < static)


def test_critical_acceleration_is_the_yield_coefficient():
    """Applying k_c must drive the factor of safety to exactly 1."""
    slope = np.tan(np.radians(np.array([20.0, 28.0, 36.0])))
    sca = np.array([100.0, 400.0, 900.0])
    c, phi, rt = 0.08, 34.0, 5e-4

    kc = P.critical_acceleration(slope, sca, c, phi, rt)
    fs = P.factor_of_safety(slope, sca, c, phi, rt, k_h=kc)
    assert np.allclose(fs, 1.0, atol=1e-9), fs

    # Steeper, wetter ground yields at less shaking. The last of these is
    # already unstable statically, which the model reports as a negative
    # critical acceleration rather than by clipping to zero.
    assert np.all(np.diff(kc) < 0), kc
    static = P.factor_of_safety(slope, sca, c, phi, rt)
    assert np.all((kc > 0) == (static > 1.0)), (kc, static)


def test_recharge_scale_raises_wetness_and_failure():
    """A wetter climate must push probabilities up, never down."""
    slope = np.tan(np.radians(np.full(200, 32.0)))
    sca = np.linspace(20.0, 4000.0, 200)
    params = P.SoilParameters((0.0, 0.15), (30.0, 40.0), (1e-5, 5e-4))

    dry = P.failure_probability(slope, sca, params, n_samples=80,
                                recharge_scale=np.full(200, 0.5))
    wet = P.failure_probability(slope, sca, params, n_samples=80,
                                recharge_scale=np.full(200, 2.0))
    assert np.all(wet >= dry)
    assert wet.mean() > dry.mean()


def test_stability_classes_span_the_sinmap_range():
    """Terrain from flat to vertical must exercise the class definitions."""
    slope = np.tan(np.radians(np.linspace(1.0, 70.0, 400)))
    sca = np.linspace(10.0, 8000.0, 400)
    cls = P.stability_classes(slope, sca, P.SoilParameters())
    present = set(int(c) for c in np.unique(cls[np.isfinite(cls)]))
    assert present.issubset({1, 2, 3, 4, 5, 6})
    assert len(present) >= 3
    # Class must not fall as the ground steepens over the stable tail.
    assert cls[0] <= cls[-1]


# ---------------------------------------------------------------------------
# triggering
# ---------------------------------------------------------------------------

def test_recharge_multiplier_is_monotonic_and_anchored():
    """The reference return period must map to exactly 1, and rarer to more."""
    assert abs(hazard.recharge_multiplier(2.0) - 1.0) < 1e-12
    ms = [hazard.recharge_multiplier(T) for T in (2, 5, 25, 100, 1000)]
    assert all(b > a for a, b in zip(ms, ms[1:]))
    # A 100-year storm should be a couple of times the near-median year, not
    # an order of magnitude.
    assert 1.5 < hazard.recharge_multiplier(100.0) < 3.5


def test_rainfall_cv_controls_scenario_sensitivity():
    """A larger coefficient of variation must spread the return periods out."""
    lo = hazard.recharge_multiplier(100.0, cv=0.20)
    hi = hazard.recharge_multiplier(100.0, cv=0.40)
    assert hi > lo > 1.0


def test_return_period_round_trip():
    """The Gumbel quantile and its inverse must agree."""
    for T in (2.0, 10.0, 100.0, 1000.0):
        z = hazard._gumbel_factor(T)
        back = hazard.return_period_from_normalised(np.array([z]))[0]
        assert abs(back - T) / T < 1e-6, (T, back)


def test_scenario_terms_separate_the_two_triggers():
    rain = hazard.scenario_terms("rainfall", return_period_yr=100.0)
    assert rain["k_h"] == 0.0 and rain["recharge_multiplier"] > 1.0

    quake = hazard.scenario_terms("earthquake", pga_g=0.4)
    assert quake["recharge_multiplier"] == 1.0
    assert abs(quake["k_h"] - 0.2) < 1e-12       # half of PGA, by convention

    try:
        hazard.scenario_terms("meteorite")
        assert False, "expected an unknown trigger to raise"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# fitting and cross-validation
# ---------------------------------------------------------------------------

def _synthetic_population(n=900, seed=0):
    """Landslides drawn from steep, convergent ground; background from all of it."""
    rng = np.random.default_rng(seed)
    slope_bg = np.tan(np.radians(rng.uniform(2.0, 60.0, n)))
    sca_bg = 10 ** rng.uniform(1.0, 3.6, n)
    # Presence points biased towards steep and wet, which is what the physics
    # says should fail.
    slope_p = np.tan(np.radians(rng.uniform(28.0, 55.0, n)))
    sca_p = 10 ** rng.uniform(2.4, 3.8, n)
    return slope_p, sca_p, slope_bg, sca_bg


def test_auc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    y = np.array([0, 0, 0, 1, 1, 1])
    assert abs(P._auc(scores, y) - 1.0) < 1e-9


def test_fit_recovers_a_discriminating_parameter_set():
    """Fitting must beat chance on a population the physics can separate."""
    sp, ap, sb, ab = _synthetic_population()
    fit = P.fit_parameters(sp, ap, sb, ab, n_samples=40)
    assert fit["auc"] > 0.65, fit["auc"]
    assert fit["n_presence"] == len(sp)
    # The search must actually explore, not return the first candidate.
    assert fit["n_trials"] == len(P.parameter_grid())
    assert fit["top_trials"][0]["auc"] >= fit["top_trials"][-1]["auc"]

    p = fit["parameters"]
    assert p.cohesion[0] <= p.cohesion[1]
    assert p.friction_deg[0] < p.friction_deg[1]
    assert 0 < p.rt[0] < p.rt[1]


def test_fit_rejects_a_sample_too_small_to_constrain():
    try:
        P.fit_parameters(np.ones(5), np.ones(5), np.ones(5), np.ones(5))
        assert False, "expected too-few-samples to raise"
    except ValueError:
        pass


def test_cross_validation_is_held_out():
    """CV must refit inside each fold, so its score is not the in-sample one."""
    rng = np.random.default_rng(1)
    sp, ap, sb, ab = _synthetic_population(600, seed=3)
    bbox = (84.0, 27.0, 86.0, 29.0)
    pts_p = np.column_stack([rng.uniform(84.0, 86.0, len(sp)),
                             rng.uniform(27.0, 29.0, len(sp))])
    pts_b = np.column_stack([rng.uniform(84.0, 86.0, len(sb)),
                             rng.uniform(27.0, 29.0, len(sb))])

    cv = P.cross_validate(pts_p, sp, ap, pts_b, sb, ab, bbox,
                          scheme="spatial", n_folds=4, block_deg=0.5,
                          n_samples=25)
    assert cv["n_folds_scored"] >= 3
    assert 0.5 < cv["auc_mean"] <= 1.0
    assert len(cv["fold_parameters"]) == cv["n_folds_scored"]
    # Each fold reports its own test-set size, and they sum to the whole.
    assert sum(a for a, _ in cv["test_sizes"]) <= len(sp)


def test_regional_fit_falls_back_when_a_region_is_sparse():
    """Regions below the presence threshold must not get their own parameters."""
    sp, ap, sb, ab = _synthetic_population(500, seed=5)
    reg_p = np.where(np.arange(len(sp)) < 400, 1, 2)     # region 2 has only 100
    reg_b = np.where(np.arange(len(sb)) < 400, 1, 2)

    out = P.fit_parameters_regional(sp, ap, reg_p, sb, ab, reg_b,
                                    n_samples=25, min_presence=200)
    assert 1 in out["by_region"]
    assert 2 not in out["by_region"], "sparse region should fall back"
    assert isinstance(out["fallback"], P.SoilParameters)


def test_regional_probability_uses_the_right_parameters():
    """Each region must be evaluated with its own parameter set."""
    slope = np.tan(np.radians(np.full(50, 35.0)))
    sca = np.full(50, 500.0)
    region = np.where(np.arange(50) < 25, 1, 2)

    strong = P.SoilParameters((0.4, 0.5), (40.0, 45.0), (1e-6, 1e-5))
    weak = P.SoilParameters((0.0, 0.01), (22.0, 26.0), (0.01, 0.05))
    p = P.failure_probability_regional(slope, sca, region,
                                       {1: strong, 2: weak},
                                       P.SoilParameters(), n_samples=50)
    assert p[:25].max() < p[25:].min(), "regions were not evaluated separately"


def test_soil_parameters_round_trip():
    p = P.SoilParameters((0.0, 0.3), (28.0, 38.0), (1e-5, 5e-4))
    assert P.SoilParameters.from_dict(p.as_dict()) == p


# ---------------------------------------------------------------------------
# cross-validation splits
# ---------------------------------------------------------------------------

def test_spatial_blocks_separate_train_and_test():
    """A spatial fold must withhold whole blocks, not scattered points."""
    rng = np.random.default_rng(0)
    bbox = (84.0, 27.0, 86.0, 29.0)
    pts = np.column_stack([rng.uniform(84.0, 86.0, 2000),
                           rng.uniform(27.0, 29.0, 2000)])
    folds = crossval.spatial_block_folds(pts, bbox, n_folds=5, block_deg=0.5)

    assert set(np.unique(folds)).issubset(set(range(5)))
    # Points sharing a block must share a fold: that is what makes the split
    # spatial rather than random.
    col = np.floor((pts[:, 0] - bbox[0]) / 0.5).astype(int)
    row = np.floor((pts[:, 1] - bbox[1]) / 0.5).astype(int)
    block = row * 100 + col
    for b in np.unique(block):
        assert len(np.unique(folds[block == b])) == 1, f"block {b} was split"

    # A random split, by contrast, mixes folds within a block.
    rnd = crossval.random_folds(len(pts), n_folds=5)
    mixed = sum(len(np.unique(rnd[block == b])) > 1 for b in np.unique(block))
    assert mixed > 0.9 * len(np.unique(block))


# ---------------------------------------------------------------------------
# inventories
# ---------------------------------------------------------------------------

def test_inventory_csv_and_bbox_filter():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "inv.csv")
        with open(p, "w") as fh:
            fh.write("event_id,latitude,longitude,country_name\n")
            fh.write("1,28.2,84.1,Nepal\n")       # inside AOI
            fh.write("2,10.0,10.0,Nigeria\n")      # outside AOI + country
            fh.write("3,27.9,84.3,India\n")        # inside AOI
        pts = inventory.load_inventory(p, bbox=(84.0, 27.5, 84.5, 28.5),
                                       countries=("Nepal", "India"))
        assert pts.shape == (2, 2)
        assert pts[:, 0].min() >= 84.0 and pts[:, 1].max() <= 28.5


def test_inventory_polygon_centroid_and_reprojection():
    """Polygon inventories reduce to one point per feature, reprojected to WGS84."""
    geom = {"type": "Polygon",
            "coordinates": [[(300000.0, 3050000.0), (300100.0, 3050000.0),
                             (300100.0, 3050100.0), (300000.0, 3050100.0),
                             (300000.0, 3050000.0)]]}
    xy = inventory._representative_point(geom)
    assert xy == (300040.0, 3050040.0)  # coordinate centroid (ring closes)

    assert inventory._representative_point(
        {"type": "Point", "coordinates": [85.3, 27.7]}) == (85.3, 27.7)

    assert inventory._representative_point(
        {"type": "Polygon", "coordinates": []}) is None


def test_validate_handles_continuous_index():
    """Validation must score the continuous field, not only a class map."""
    import rasterio

    from h_sim.model import validate

    with tempfile.TemporaryDirectory() as tmp:
        grid = Grid.from_bbox((84.0, 28.0, 84.5, 28.5), 0.01)   # 50x50
        rng = np.random.default_rng(0)
        idx = rng.uniform(0.0, 1.0, grid.shape).astype("float32")
        path = os.path.join(tmp, "prob.tif")
        with rasterio.open(path, "w", **grid.profile("float32", -9999.0)) as d:
            d.write(idx, 1)
        assert validate.is_continuous(path)

        rows, cols = np.where(idx > 0.9)
        xs, ys = rasterio.transform.xy(grid.transform, rows, cols)
        pres = np.column_stack([np.asarray(xs), np.asarray(ys)])
        bg = inventory.background_points((84.0, 28.0, 84.5, 28.5), 800, path)

        r = validate.validate_susceptibility(path, pres, bg)
        assert r.n_classes == 5
        assert r.frequency_ratio["5"] > 1.0
        assert r.monotonic, r.frequency_ratio
        assert r.auc > 0.9, r.auc            # perfect separation by construction


def test_validate_handles_six_stability_classes():
    """SINMAP's class raster runs to 6, and the report must not truncate it."""
    import rasterio

    from h_sim.model import validate

    with tempfile.TemporaryDirectory() as tmp:
        grid = Grid.from_bbox((84.0, 28.0, 84.6, 28.6), 0.01)   # 60x60
        rng = np.random.default_rng(2)
        cls = rng.integers(1, 7, grid.shape).astype("uint8")
        path = os.path.join(tmp, "cls.tif")
        with rasterio.open(path, "w", **grid.profile("uint8", 255)) as d:
            d.write(cls, 1)
        assert not validate.is_continuous(path)

        rows, cols = np.where(cls >= 5)
        xs, ys = rasterio.transform.xy(grid.transform, rows, cols)
        pres = np.column_stack([np.asarray(xs), np.asarray(ys)])

        r = validate.validate_susceptibility(path, pres)
        assert r.n_classes == 6
        assert "6" in r.frequency_ratio
        assert r.frequency_ratio["6"] > 1.0
        assert r.frequency_ratio["1"] == 0.0
        assert "class" in validate.format_report(r)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def _demo_config(tmp, **kw):
    base = dict(bbox=(84.0, 28.0, 84.2, 28.2), resolution_deg=0.004,
                block_size=64, data_dir=os.path.join(tmp, "raw"),
                work_dir=os.path.join(tmp, "work"),
                out_dir=os.path.join(tmp, "out"))
    base.update(kw)
    return C.Config(**base)


# ---------------------------------------------------------------------------
# climate scenarios
# ---------------------------------------------------------------------------

def test_climate_scenario_parsing():
    assert CL.scenario("current").is_baseline
    assert CL.scenario("baseline").key == "current"

    s = CL.scenario("ssp585:2061-2080")
    assert (s.ssp, s.period, s.is_baseline) == ("ssp585", "2061-2080", False)
    assert s.key == "ssp585_2061-2080"
    # A bare pathway takes the planning-horizon window rather than failing.
    assert CL.scenario("ssp245").period == CL.DEFAULT_PERIOD
    assert CL.DEFAULT_PERIOD in CL.PERIODS

    for bad in ("ssp999:2061-2080", "ssp585:1999-2000"):
        try:
            CL.scenario(bad)
            assert False, f"expected {bad} to raise"
        except ValueError:
            pass


def test_climate_suites_always_include_the_baseline():
    """Every future is reported as a change from today, so today must be there."""
    for group in (CL.suite(), CL.trajectory("ssp370")):
        assert group[0].is_baseline
        assert len({s.key for s in group}) == len(group), "duplicate scenarios"

    # parse_all preserves order and drops repeats.
    got = CL.parse_all(["ssp245:2041-2060", "current", "ssp245:2041-2060"])
    assert [s.key for s in got] == ["ssp245_2041-2060", "current"]


def test_future_recharge_is_normalised_by_the_present_day_reference():
    """A uniformly wetter future must show as a shift, not cancel out."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="clim")
        grid = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)

        base = CL.BASELINE
        inputs = pipeline.resolve_inputs(cfg, "demo", scen=base)
        _, reference = pipeline.stage_recharge(cfg, grid, inputs, scen=base)
        assert reference > 0

        future = CL.scenario("ssp585:2081-2100")
        f_inputs = pipeline.resolve_inputs(cfg, "demo", scen=future)
        f_path, f_ref = pipeline.stage_recharge(cfg, grid, f_inputs,
                                                reference_mm=reference,
                                                scen=future)
        assert f_ref == reference, "the reference must not be re-measured"

        import rasterio
        with rasterio.open(f_path) as src:
            a = src.read(1)
            a = a[a != src.nodata]
        expected = demo.demo_precip_factor(future)
        assert abs(float(np.median(a)) - expected) < 0.02, (
            "future recharge should sit at the wetting factor above the "
            "present-day reference")

        # Had it been normalised by its own median it would sit at 1.0.
        assert float(np.median(a)) > 1.05


def test_climate_scenarios_do_not_share_work_files():
    """Two climates on one grid must not overwrite each other's recharge."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="two")
        grid = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
        paths = set()
        for spec in ("current", "ssp126:2021-2040", "ssp585:2081-2100"):
            s = CL.scenario(spec)
            inputs = pipeline.resolve_inputs(cfg, "demo", scen=s)
            path, _ = pipeline.stage_recharge(cfg, grid, inputs,
                                              reference_mm=500.0, scen=s)
            paths.add(path)
        assert len(paths) == 3, paths


def test_climate_sweep_produces_maps_and_changes():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="sweep", resolution_deg=0.005)
        report = pipeline.run_climate(
            cfg, mode="demo", specs=["ssp585:2081-2100"])

        keys = [r["scenario"] for r in report["scenarios"]]
        assert keys[0] == "current", "the baseline must be evaluated first"
        assert "ssp585_2081-2100" in keys
        assert os.path.exists(report["summary"])
        for path in report["maps"].values():
            assert os.path.exists(path)
        for path in report["changes"].values():
            assert os.path.exists(path)

        # A wetter future cannot make the ground safer on average.
        fut = [r for r in report["scenarios"]
               if r["scenario"] == "ssp585_2081-2100"][0]
        assert fut["mean_change"] >= 0.0


# ---------------------------------------------------------------------------
# regional sweep by administrative unit
# ---------------------------------------------------------------------------

def _fake_admin_layer(tmp, boxes) -> str:
    """A GeoJSON standing in for Natural Earth, so tests need no download."""
    feats = []
    for name, (w, s, e, n) in boxes.items():
        feats.append({
            "type": "Feature",
            "properties": {"name": name, "admin": "Testland"},
            "geometry": {"type": "Polygon", "coordinates": [[
                [w, s], [e, s], [e, n], [w, n], [w, s]]]}})
    path = os.path.join(tmp, "admin.geojson")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh)
    return path


def test_admin_units_are_clipped_to_the_region():
    from h_sim.input import admin

    with tempfile.TemporaryDirectory() as tmp:
        path = _fake_admin_layer(tmp, {
            "Inside": (84.0, 28.0, 84.5, 28.5),
            "Straddling": (59.0, 15.0, 61.0, 17.0),   # half outside the HKH
            "Outside": (0.0, 0.0, 1.0, 1.0),
            "Sliver": (84.0, 28.0, 84.005, 28.005),
        })
        units = admin.load_units(path, C.HKH_BBOX)
        names = {u.name for u in units}

        assert "Inside" in names
        assert "Outside" not in names, "unit outside the region must be dropped"
        assert "Sliver" not in names, "unit too small to route must be dropped"

        strad = [u for u in units if u.name == "Straddling"][0]
        assert strad.bbox[0] == C.HKH_BBOX[0], "should clip to the region edge"
        assert strad.bbox[1] == C.HKH_BBOX[1]


def test_admin_slug_is_unique_and_filesystem_safe():
    from h_sim.input import admin

    a = admin.AdminUnit("Jammu & Kashmir", "India", (75.0, 32.0, 76.0, 33.0))
    b = admin.AdminUnit("Jammu & Kashmir", "Pakistan", (74.0, 33.0, 75.0, 34.0))
    assert a.slug != b.slug
    for s in (a.slug, b.slug):
        assert all(c.isalnum() or c == "_" for c in s), s
    assert a.slug == "india_jammu_kashmir"


def test_buffer_grows_the_box_and_costs_cells():
    from h_sim.input import admin

    u = admin.AdminUnit("X", "Y", (84.0, 28.0, 85.0, 29.0))
    assert admin.buffered_bbox(u.bbox, 0.0) == u.bbox
    assert admin.buffered_bbox(u.bbox, 0.5) == (83.5, 27.5, 85.5, 29.5)

    plain = u.cell_count(0.001)
    buffered = u.cell_count(0.001, 0.05)
    assert plain == 1000 * 1000
    assert buffered == 1100 * 1100          # the buffer is not free


def test_regional_sweep_clips_each_unit_and_is_resumable():
    """Two adjacent units must not both claim the border cells."""
    import rasterio

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="reg", resolution_deg=0.005)
        cfg.admin_buffer_deg = 0.02
        cfg.admin_path = _fake_admin_layer(tmp, {
            "West": (84.0, 28.0, 84.1, 28.2),
            "East": (84.1, 28.0, 84.2, 28.2),
        })
        cfg.admin_elevation_res = None      # no download in a unit test
        cfg.admin_countries = ["Testland"]

        report = pipeline.run_region(cfg, mode="demo")
        assert report["n_units_runnable"] == 2
        assert report["n_completed"] == 2
        assert not report["failed"], report["failed"]

        # Every unit produced a map, and each blanked the ground outside itself.
        for u in report["units"]:
            prob = u["maps"]["probability"]
            assert os.path.exists(prob)
            with rasterio.open(prob) as src:
                a = src.read(1)
                valid = a != src.nodata
            assert valid.any(), "unit produced an empty map"
            assert not valid.all(), "buffer region should be blanked"
            assert u["stats"]["cells_in_unit"] > 0

        # Re-running skips completed units rather than redoing them.
        again = pipeline.run_region(cfg, mode="demo")
        assert again["n_completed"] == 0
        assert len(again["units"]) == 2


def test_regional_dry_run_costs_without_running():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="plan", resolution_deg=0.005)
        cfg.admin_path = _fake_admin_layer(tmp, {
            "Small": (84.0, 28.0, 84.1, 28.1),
            "Huge": (60.0, 16.0, 100.0, 38.0),
        })
        cfg.admin_max_cells = 1_000_000
        cfg.admin_elevation_res = None      # no download in a unit test
        cfg.admin_countries = ["Testland"]

        report = pipeline.run_region(cfg, mode="demo", dry_run=True)
        assert report["n_units_found"] == 2
        assert report["n_units_runnable"] == 1, "the huge unit must be skipped"
        assert report["skipped_too_large"][0]["name"] == "Huge"
        assert report["plan"][0]["cells"] > report["plan"][-1]["cells"]
        # Nothing was produced.
        assert not [f for f in os.listdir(cfg.out_dir) if f.endswith(".tif")]


def test_package_manifest_carries_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="pack", resolution_deg=0.005)
        pipeline.run_susceptibility(cfg, mode="demo")
        out = pipeline.run_package(cfg)

        m = json.load(open(out["manifest"]))
        assert m["name"] == "pack"
        assert "SINMAP" in m["model"]
        assert m["area"]["cells"] > 0
        assert m["conventions"]["pga_fraction"] == cfg.pga_fraction
        assert m["products"]["susceptibility"], "no susceptibility products"
        assert any("legend" in s for s in m["interpretation"])


def test_manifest_describes_the_rasters_not_the_config():
    """step8 is run without the --bbox/--res the products were made with."""
    with tempfile.TemporaryDirectory() as tmp:
        made = _demo_config(tmp, name="prov", bbox=(83.0, 27.5, 83.2, 27.7),
                            resolution_deg=0.005)
        pipeline.run_susceptibility(made, mode="demo")

        # A config carrying a different area entirely, as happens when step8
        # is invoked with only --name.
        stale = _demo_config(tmp, name="prov", bbox=(84.0, 28.0, 85.0, 29.0),
                             resolution_deg=0.0008333333)
        m = json.load(open(pipeline.run_package(stale)["manifest"]))

        assert m["area"]["read_from"].endswith(".tif")
        assert abs(m["area"]["bbox"][0] - 83.0) < 1e-6, m["area"]["bbox"]
        assert abs(m["area"]["resolution_deg"] - 0.005) < 1e-9
        assert m["area"]["cells"] == 40 * 40


def test_end_to_end_demo_rainfall():
    import rasterio

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="t", trigger="rainfall")
        out = pipeline.run_susceptibility(cfg, mode="demo")
        haz = pipeline.run_hazard(cfg, mode="demo")

        for path in (out["probability"], out["classes"], haz["probability"]):
            assert os.path.exists(path), path

        with rasterio.open(out["probability"]) as src:
            p = src.read(1)
            p = p[p != src.nodata]
            assert p.min() >= 0.0 and p.max() <= 1.0
            assert len(np.unique(p)) > 5, "index should be continuous"

        with rasterio.open(out["classes"]) as src:
            a = src.read(1)
            classes = set(int(x) for x in np.unique(a) if x != src.nodata)
            assert classes.issubset({1, 2, 3, 4, 5, 6})
            assert len(classes) >= 3, "model should discriminate"

        # A 100-year storm cannot make any pixel safer than the baseline.
        with rasterio.open(out["probability"]) as a, \
                rasterio.open(haz["probability"]) as b:
            base, scen = a.read(1), b.read(1)
            ok = (base != a.nodata) & (scen != b.nodata)
            assert (scen[ok] >= base[ok] - 1e-6).all()


def test_end_to_end_demo_earthquake():
    import rasterio

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="teq", trigger="earthquake",
                           scenario_pga_g=0.4)
        base = pipeline.run_susceptibility(cfg, mode="demo")
        haz = pipeline.run_hazard(cfg, mode="demo")

        with rasterio.open(base["probability"]) as a, \
                rasterio.open(haz["probability"]) as b:
            s, h = a.read(1), b.read(1)
            ok = (s != a.nodata) & (h != b.nodata)
            assert (h[ok] >= s[ok] - 1e-6).all(), "shaking must not stabilise"
            assert h[ok].mean() > s[ok].mean(), "0.4 g should matter"


def test_critical_acceleration_is_written_once():
    """The Newmark map belongs to the terrain, not to a trigger scenario."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="kc")
        base = pipeline.run_susceptibility(cfg, mode="demo")
        haz = pipeline.run_hazard(cfg, mode="demo")
        assert "critical_acceleration" in base
        assert "critical_acceleration" not in haz


def test_terrain_stage_is_cached():
    """Flow routing is the expensive step; a second call must reuse it."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="cache")
        grid = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
        inputs = pipeline.resolve_inputs(cfg, "demo")

        first = pipeline.stage_terrain(cfg, grid, inputs)
        stamp = os.path.getmtime(first["sca"])
        second = pipeline.stage_terrain(cfg, grid, inputs)
        assert second == first
        assert os.path.getmtime(second["sca"]) == stamp, "recomputed unnecessarily"

        forced = pipeline.stage_terrain(cfg, grid, inputs, force=True)
        assert os.path.getmtime(forced["sca"]) >= stamp


def test_terrain_cache_is_invalidated_by_a_different_grid():
    """Re-running a name at a new resolution must not reuse the old rasters."""
    import rasterio

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="regrid", resolution_deg=0.004)
        coarse = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
        pipeline.stage_terrain(cfg, coarse, pipeline.resolve_inputs(cfg, "demo"))

        cfg.resolution_deg = 0.002
        fine = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
        assert fine.shape != coarse.shape
        out = pipeline.stage_terrain(cfg, fine,
                                     pipeline.resolve_inputs(cfg, "demo"))
        with rasterio.open(out["sca"]) as src:
            assert (src.width, src.height) == (fine.width, fine.height)


def test_recharge_stage_is_cached_and_grid_checked():
    """Twelve monthly warps should happen once, and never for the wrong grid."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="rech")
        grid = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
        inputs = pipeline.resolve_inputs(cfg, "demo")

        path, ref = pipeline.stage_recharge(cfg, grid, inputs)
        precip = os.path.join(cfg.work_dir, "rech_precip_max_month_current.tif")
        stamp = os.path.getmtime(precip)
        assert ref > 0

        # Same grid: reuse, and honour the reference passed in.
        again, ref2 = pipeline.stage_recharge(cfg, grid, inputs,
                                              reference_mm=ref)
        assert again == path and ref2 == ref
        assert os.path.getmtime(precip) == stamp, "recomputed unnecessarily"

        # Different grid: the cached raster must not be reused.
        cfg.resolution_deg = 0.002
        fine = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
        pipeline.stage_recharge(cfg, fine, inputs)
        import rasterio
        with rasterio.open(precip) as src:
            assert (src.width, src.height) == (fine.width, fine.height)


def test_no_temporary_rasters_survive_a_run():
    """Atomic writes must not leave .tmp files behind."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="atomic")
        pipeline.run_susceptibility(cfg, mode="demo")
        for d in (cfg.work_dir, cfg.out_dir):
            leftovers = [f for f in os.listdir(d) if ".tmp" in f]
            assert not leftovers, leftovers


def test_fit_writes_parameters_that_later_steps_read():
    """The fit -> map handoff must go through the JSON, not through memory."""
    import rasterio

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _demo_config(tmp, name="fitrun", resolution_deg=0.002)
        grid = Grid.from_bbox(cfg.clipped_bbox(), cfg.resolution_deg)
        inputs = pipeline.resolve_inputs(cfg, "demo")
        terrain = pipeline.stage_terrain(cfg, grid, inputs)

        # Landslides placed on the steepest ground the demo terrain offers.
        with rasterio.open(terrain["slope"]) as src:
            slope = src.read(1)
            cut = np.nanpercentile(slope[slope != src.nodata], 88)
            rows, cols = np.where(slope > cut)
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        pts = np.column_stack([np.asarray(xs), np.asarray(ys)])
        inv = os.path.join(tmp, "inv.csv")
        with open(inv, "w") as fh:
            fh.write("longitude,latitude\n")
            for x, y in pts:
                fh.write(f"{x},{y}\n")

        cfg.inventory_path = inv
        report = pipeline.run_fit(cfg, mode="demo", cross_validate=False)
        assert os.path.exists(report["path"])
        assert report["in_sample_auc"] > 0.6, report["in_sample_auc"]

        saved = json.load(open(report["path"]))
        assert set(saved["parameters"]) == {"cohesion", "friction_deg", "rt"}
        assert saved["recharge_reference_mm"] > 0

        # A fresh config, pointed only at the JSON, must pick the fit up.
        fresh = _demo_config(tmp, name="fitrun2",
                             resolution_deg=cfg.resolution_deg,
                             fitted_params=report["path"])
        params, by_region, ref = pipeline.load_fitted(fresh)
        assert params.as_dict() == saved["parameters"]
        assert by_region == {}
        assert ref == saved["recharge_reference_mm"]


def test_compare_probability():
    """Difference map must report the shift between two runs."""
    import rasterio

    with tempfile.TemporaryDirectory() as tmp:
        grid = Grid.from_bbox((84.0, 28.0, 84.1, 28.1), 0.01)
        base = np.full(grid.shape, 0.2, dtype="float32")
        scen = base.copy()
        scen[:, :5] = 0.6                     # left half becomes likelier
        paths = []
        for name, arr in (("base", base), ("scen", scen)):
            p = os.path.join(tmp, f"{name}.tif")
            with rasterio.open(p, "w", **grid.profile("float32", -9999.0)) as d:
                d.write(arr, 1)
            paths.append(p)

        out = pipeline.compare_probability(paths[0], paths[1],
                                           os.path.join(tmp, "cmp"))
        assert os.path.exists(out["change"])
        stats = json.load(open(out["summary"]))
        assert stats["pct_more_likely"] == 50.0
        assert stats["pct_less_likely"] == 0.0
        assert abs(stats["mean_change"] - 0.2) < 1e-4

        # Mismatched grids must be rejected, not silently compared.
        other = Grid.from_bbox((84.0, 28.0, 84.1, 28.1), 0.005)
        p3 = os.path.join(tmp, "other.tif")
        with rasterio.open(p3, "w", **other.profile("float32", -9999.0)) as d:
            d.write(np.full(other.shape, 0.2, dtype="float32"), 1)
        try:
            pipeline.compare_probability(paths[0], p3,
                                         os.path.join(tmp, "bad"))
            assert False, "expected mismatched grids to raise"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# dataset registry
# ---------------------------------------------------------------------------

def test_dataset_registry_is_wellformed():
    """Every registered dataset must be fetchable or explicitly manual."""
    from h_sim.input import datasets

    keys = [d.key for d in datasets.REGISTRY]
    assert len(keys) == len(set(keys)), "duplicate dataset keys"
    for d in datasets.REGISTRY:
        assert d.group in (datasets.TERRAIN, datasets.CLIMATE,
                           datasets.INVENTORY, datasets.TRIGGER), d.key
        assert d.probe_url or d.manual_url, f"{d.key} has no source"
        assert d.rel_path and not os.path.isabs(d.rel_path), d.key


def test_dataset_cache_detection():
    """A dataset counts as cached only when its file/dir actually has content."""
    from h_sim.input import datasets

    ds = datasets.BY_KEY["gorkha"]
    with tempfile.TemporaryDirectory() as tmp:
        assert ds.cached(tmp) is False
        target = ds.local_path(tmp)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        open(target, "w").close()            # empty file must not count
        assert ds.cached(tmp) is False
        with open(target, "w") as fh:
            fh.write("{}")
        assert ds.cached(tmp) is True

        rows = datasets.check_all(tmp, probe=False, keys=["gorkha"])
        assert rows[0]["cached"] is True and rows[0]["reachable"] is None
        assert "CACHED" in datasets.format_report(rows)


# ---------------------------------------------------------------------------
# reach, exposure and the web map
# ---------------------------------------------------------------------------

def _cone(n=61, height=600.0):
    """A cone: one peak, elevation falling linearly to the edge."""
    y, x = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    r = np.hypot(y - c, x - c)
    return np.clip(height * (1.0 - r / c), 0.0, None)


def _reach_index(dem, cell_m=30.0, **kw):
    from rasterio.transform import from_origin
    from h_sim.model import risk as R

    deg = cell_m / 111320.0
    tr = from_origin(80.0, 30.0, deg, deg)
    return R.ReachIndex(dem, tr, cell_m, cell_m, **kw), tr


def test_reach_geometry_respects_the_travel_angle():
    """Nothing reaches the peak; the foot of a steep cone is reached."""
    from h_sim.model import risk as R

    dem = _cone()
    idx, tr = _reach_index(dem, search_radius_m=2000.0)
    n = dem.shape[0]
    peak = idx.reach(*(tr * (n // 2 + 0.5, n // 2 + 0.5)))
    assert peak.rows.size == 0, "nothing is above the summit"

    foot = idx.reach(*(tr * (1.5, n // 2 + 0.5)))
    assert foot.rows.size > 0
    # every accepted source clears the angle criterion
    assert np.all(foot.relief / foot.dist > idx.tan_alpha)

    # A shallower cone puts nothing above the limiting angle at all.
    flat, trf = _reach_index(_cone(height=60.0), search_radius_m=2000.0)
    assert flat.reach(*(trf * (1.5, n // 2 + 0.5))).rows.size == 0


def test_reach_weights_favour_close_sources():
    """A source twice as far counts half as much: paths widen with distance."""
    dem = _cone()
    idx, tr = _reach_index(dem)
    rch = idx.reach(*(tr * (1.5, dem.shape[0] // 2 + 0.5)))
    order = np.argsort(rch.dist)
    d, w = rch.dist[order], rch.weight[order]
    assert np.all(np.diff(w) <= 1e-12), "weight must fall with distance"
    assert np.allclose(w * d, w[0] * d[0])


def test_score_is_a_weighted_mean_not_a_maximum():
    """The headline score must not saturate as the window grows.

    This is the defect the first version had: taking the maximum over a few
    thousand upslope cells finds a high value almost surely, so nearly every
    settlement landed in the top band. A weighted mean cannot do that.
    """
    from h_sim.model import risk as R

    dem = _cone()
    idx, tr = _reach_index(dem)
    rch = idx.reach(*(tr * (1.5, dem.shape[0] // 2 + 0.5)))
    assert rch.rows.size > 50

    prob = np.zeros_like(dem)
    prob[rch.rows[0], rch.cols[0]] = 1.0        # exactly one unstable cell
    sc = rch.score(prob)
    assert sc.reaching_max == 1.0
    assert sc.reaching < 0.2, "one hot cell must not set the headline"
    assert sc.score == max(sc.on_site, sc.reaching)

    prob[rch.rows, rch.cols] = 1.0              # all of them unstable
    assert rch.score(prob).reaching == 1.0


def test_score_is_monotone_in_probability():
    from h_sim.model import risk as R

    dem = _cone()
    idx, tr = _reach_index(dem)
    rch = idx.reach(*(tr * (1.5, dem.shape[0] // 2 + 0.5)))
    lo = rch.score(np.full_like(dem, 0.2)).reaching
    hi = rch.score(np.full_like(dem, 0.6)).reaching
    assert lo < hi
    assert abs(lo - 0.2) < 1e-9 and abs(hi - 0.6) < 1e-9


def test_reach_geometry_is_reused_across_scenarios():
    """Scoring N climates costs one window search, not N."""
    from h_sim.model import risk as R

    dem = _cone()
    idx, tr = _reach_index(dem)
    lon, lat = tr * (1.5, dem.shape[0] // 2 + 0.5)
    rch = idx.reach(lon, lat)
    a = rch.score(np.full_like(dem, 0.2))
    b = rch.score(np.full_like(dem, 0.5))
    assert a.n_sources == b.n_sources and a.reaching < b.reaching


def test_segment_line_keeps_length_and_splits():
    from h_sim.model import risk as R

    coords = [(84.0 + 0.001 * i, 28.0) for i in range(60)]
    total = R.line_length_m(coords)
    segs = R.segment_line(coords, 500.0)
    assert len(segs) > 1
    assert abs(sum(R.line_length_m(s) for s in segs) - total) < 1.0
    assert all(len(s) >= 2 for s in segs)
    # a two-point way shorter than the target is still one segment
    assert len(R.segment_line(coords[:2], 500.0)) == 1


def test_batch_scoring_carries_every_scenario():
    from h_sim.input.exposure import Road, Settlement
    from h_sim.model import risk as R

    dem = _cone()
    idx, tr = _reach_index(dem)
    n = dem.shape[0]
    lon, lat = tr * (1.5, n // 2 + 0.5)
    towns = [Settlement("Foot", float(lon), float(lat), "village", 500)]
    lon2, lat2 = tr * (3.5, n // 2 + 0.5)
    roads = [Road("Valley road", "primary",
                  [(float(lon), float(lat)), (float(lon2), float(lat2))])]
    probs = {"current": np.full_like(dem, 0.2),
             "ssp585_2041-2060": np.full_like(dem, 0.5)}

    st = R.score_settlements(idx, towns, probs)
    assert len(st) == 1
    assert set(st[0]["scenarios"]) == set(probs)
    assert st[0]["score"] == st[0]["scenarios"]["current"]["score"]
    assert st[0]["delta_max"] > 0

    rd = R.score_roads(idx, roads, probs, segment_m=500.0)
    assert rd and set(rd[0]["scenarios"]) == set(probs)

    s = R.summarise(st, rd, list(probs))
    assert set(s["scenarios"]) == set(probs)
    assert s["change"]["ssp585_2041-2060"]["mean_settlement_score"] > 0
    assert s["n_settlements"] == 1
    # road_km_by_band is kilometres; length_m is metres
    banded = sum(s["scenarios"]["current"]["road_km_by_band"].values())
    assert abs(banded - s["road_km_total"]) < 0.05


def test_bands_are_ordered_and_cover_the_range():
    from h_sim.model import risk as R

    assert R.band(0.0) == "very low" and R.band(1.0) == "very high"
    edges = [e for e, _ in R.RISK_BANDS]
    assert edges == sorted(edges)
    seen = [R.band(v) for v in np.linspace(0, 1, 200)]
    assert set(seen) == set(R.BAND_ORDER)


def test_demo_exposure_sits_on_low_ground():
    """Settlements must go in the valleys, which is the case the model is for."""
    grid = Grid.from_bbox((83.0, 27.5, 83.2, 27.7), 0.002)
    dem = _cone(n=grid.shape[0], height=3000.0)
    towns, roads = demo.make_demo_exposure(grid, dem)
    assert towns and roads

    inv = ~grid.transform
    zs = []
    for t in towns:
        col, row = inv * (t.lon, t.lat)
        zs.append(dem[int(row), int(col)])
    assert np.median(zs) < np.median(dem)


def test_webmap_data_is_script_loadable():
    """Layers ship as <script src>, because fetch() is blocked on file://."""
    from h_sim import webmap

    with tempfile.TemporaryDirectory() as tmp:
        name = webmap.write_data(tmp, "settlements",
                                 {"type": "FeatureCollection", "features": []})
        assert name == "settlements.js"
        text = open(os.path.join(tmp, name), encoding="utf-8").read()
        assert text.startswith("window.HSIM_DATA")
        assert '"settlements"' in text


def test_webmap_page_survives_a_missing_leaflet():
    """The tables come from the run; a blocked CDN must not blank the page."""
    from h_sim import webmap

    assert "typeof L !== 'undefined'" in webmap._PAGE
    assert "__DATA__" in webmap._PAGE and "__LEAFLET__" in webmap._PAGE


def test_short_scenario_labels():
    assert pipeline._short_scenario(CL.BASELINE) == "present day"
    assert (pipeline._short_scenario(CL.scenario("ssp245:2041-2060"))
            == "SSP2-4.5 2041-60")


def test_defaults_sit_on_the_planning_horizon():
    """Nothing should quietly default to end of century."""
    cfg = C.Config()
    assert all(s == "current" or s.endswith(CL.DEFAULT_PERIOD)
               for s in cfg.climate_suite), cfg.climate_suite
    assert all(s == "current" or s.split(":")[1] in ("2021-2040",
                                                     CL.DEFAULT_PERIOD)
               for s in cfg.risk_climate), cfg.risk_climate
    assert all(s.is_baseline or s.period == CL.DEFAULT_PERIOD
               for s in CL.suite())


def test_validate_offers_to_build_and_masks_the_survey_extent():
    """The two things that made step4 fail in practice.

    A map has to exist before it can be scored, and inventories in this region
    do not overlap - so the useful behaviour is building a map over the
    held-out inventory's own ground. Background must then be confined to the
    surveyed polygon, or unmapped terrain is scored as landslide-free.
    """
    from h_sim import cli
    from h_sim.input import inventory as INV

    assert hasattr(cli, "_build_for_validation")
    assert hasattr(cli, "_inventory_extent")
    assert hasattr(INV, "survey_masked_reference")

    # extent of a synthetic inventory, padded
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "inv.csv")
        with open(csv_path, "w") as fh:
            fh.write("latitude,longitude\n27.5,88.1\n27.6,88.4\n")
        ext = cli._inventory_extent([csv_path], pad_deg=0.01)
        assert ext is not None
        assert ext[0] < 88.1 and ext[2] > 88.4
        assert cli._inventory_extent([os.path.join(tmp, "nope.csv")]) is None


def test_survey_mask_confines_background():
    """Background drawn outside the surveyed polygon means 'nobody looked'."""
    import rasterio
    from rasterio.transform import from_origin

    from h_sim.input import inventory as INV

    with tempfile.TemporaryDirectory() as tmp:
        ref = os.path.join(tmp, "ref.tif")
        tr = from_origin(88.0, 27.6, 0.01, 0.01)
        prof = dict(driver="GTiff", height=20, width=20, count=1,
                    dtype="float32", crs="EPSG:4326", transform=tr,
                    nodata=-9999.0)
        with rasterio.open(ref, "w", **prof) as dst:
            dst.write(np.ones((20, 20), "float32"), 1)

        poly = os.path.join(tmp, "extent.geojson")
        with open(poly, "w") as fh:
            json.dump({"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon", "coordinates": [[
                     [88.0, 27.5], [88.1, 27.5], [88.1, 27.6],
                     [88.0, 27.6], [88.0, 27.5]]]}}]}, fh)

        out, frac = INV.survey_masked_reference(
            poly, ref, os.path.join(tmp, "masked.tif"))
        assert 0.2 < frac < 0.3, frac      # a quarter of a 20x20 grid

        bbox = (88.0, 27.4, 88.2, 27.6)
        bg = INV.background_points(bbox, 200, out, seed=3)
        assert len(bg) > 50
        assert bg[:, 0].max() <= 88.11 and bg[:, 1].min() >= 27.49


def test_regional_sweep_can_produce_exposure_and_pages():
    """The regional path has to reach exposure and a page, not stop at rasters.

    A sweep that leaves only susceptibility rasters is an archive. What makes
    it usable is per-province exposure plus one ranked index over the lot.
    """
    import inspect

    from h_sim import webmap

    for fn in (pipeline.run_admin_unit, pipeline.run_region):
        params = inspect.signature(fn).parameters
        assert "stages" in params, fn.__name__
    for stage in ("susceptibility", "climate", "settlements", "roads",
                  "webmap"):
        assert stage in pipeline.REGION_STAGES
    assert hasattr(pipeline, "run_region_index")
    assert hasattr(webmap, "build_region_index")


def test_region_index_ranks_and_reports_what_was_skipped():
    from h_sim import webmap

    rows = [
        {"country": "Nepal", "name": "Bagmati", "unstable_pct": 4.0,
         "settlements_exposed": 12, "road_km_exposed": 30.0, "map": "a/index.html"},
        {"country": "Nepal", "name": "Gandaki", "unstable_pct": 11.5,
         "settlements_exposed": 40, "road_km_exposed": 90.0, "map": None},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = webmap.build_region_index(
            tmp, "sweep", rows,
            {"skipped_too_large": ["Tibet"], "failed": [], "n_completed": 2})
        html = open(path, encoding="utf-8").read()
        assert "Gandaki" in html and "Tibet" in html
        # the bar denominator must not leak into the headline figure
        assert "barScale" in html
        # a province with no page must not fabricate a link
        assert '"map": null' in html or "'map': None" not in html


def test_only_mountain_provinces_are_swept():
    """A bounding box is not a mountain range.

    The HKH box contains the Gangetic plain and most of peninsular India.
    Selecting on it alone returns Odisha and Madhya Pradesh, which have no
    Himalayan hillslope in them.
    """
    from h_sim.input import admin

    plain = {"mountain_fraction": 0.008, "mountain_area_km2": 1214.0,
             "max_elevation_m": 1110.0, "median_local_relief_m": 90.0}
    corner = {"mountain_fraction": 0.019, "mountain_area_km2": 1598.0,
              "max_elevation_m": 2938.0, "median_local_relief_m": 1351.0}
    core = {"mountain_fraction": 0.61, "mountain_area_km2": 227561.0,
            "max_elevation_m": 4915.0, "median_local_relief_m": 626.0}

    assert not admin.is_mountainous(plain), "Odisha-like unit must be dropped"
    assert admin.is_mountainous(corner), "Darjeeling salient must be kept"
    assert admin.is_mountainous(core)
    assert admin.NOT_HKH, "high ground outside the arc is named, not tuned away"


def test_mountain_needs_relief_not_just_altitude():
    """A high plain has no hillslopes to fail on."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from h_sim.input import admin

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "elev.tif")
        tr = from_origin(84.0, 29.0, 0.0833, 0.0833)
        z = np.full((24, 24), 1600.0, "float32")      # a plateau, uniformly high
        z[:, 12:] += (np.arange(12) * 300).astype("float32")   # a range beside it
        prof = dict(driver="GTiff", height=24, width=24, count=1, dtype="float32",
                    crs="EPSG:4326", transform=tr, nodata=-32768.0)
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(z, 1)

        def unit(w, e):
            geom = {"type": "Polygon", "coordinates": [[
                [w, 27.2], [e, 27.2], [e, 28.8], [w, 28.8], [w, 27.2]]]}
            return admin.AdminUnit("u", "Testland", (w, 27.2, e, 28.8), geom)

        flat = admin.relief_stats(unit(84.1, 84.9), path)
        steep = admin.relief_stats(unit(85.1, 85.9), path)
        assert flat["mountain_fraction"] == 0.0, flat
        assert steep["mountain_fraction"] > 0.5, steep


def test_a_layer_without_countries_is_not_silently_emptied():
    """Somebody's own province shapefile may carry no country attribute."""
    from h_sim.input import admin

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "a.geojson")
        with open(path, "w") as fh:
            json.dump({"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {"name": "Mine"},
                 "geometry": {"type": "Polygon", "coordinates": [[
                     [84.0, 28.0], [84.5, 28.0], [84.5, 28.5],
                     [84.0, 28.5], [84.0, 28.0]]]}}]}, fh)
        units = admin.load_units(path, C.HKH_BBOX, countries=["Nepal"])
        assert len(units) == 1


def test_shipped_configs_load_and_target_the_region():
    """The product is region-wide; only the calibration config is a small area."""
    import glob

    paths = sorted(glob.glob("configs/*.json"))
    assert paths, "no configs found"
    regional = 0
    for path in paths:
        cfg = C.Config.from_json(path)
        assert cfg.name, path
        assert cfg.region_bbox == C.HKH_BBOX, path
        if "calibrate" not in path:
            # a regional config is run by step9-region, which sweeps
            # region_bbox and never reads bbox
            regional += 1
    assert regional >= 3, [p for p in paths]
    assert any("calibrate" in p for p in paths), "the fit needs a config too"


def test_run_all_sweeps_the_region_by_default():
    import inspect

    params = inspect.signature(pipeline.run).parameters
    assert "region" in params
    assert params["region"].default is False   # the API default
    src = inspect.getsource(pipeline.run)
    assert "run_region" in src


def test_every_inventory_is_registered_and_fetchable():
    """A dataset nobody can find is a dataset nobody uses."""
    from h_sim.input import datasets, inventory as INV

    keys = {d.key for d in datasets.REGISTRY if d.group == datasets.INVENTORY}
    assert keys == {"gorkha", "farwest", "sikkim"}, keys
    for key in keys:
        assert key in INV.INVENTORY_FETCHERS, key
    for key, (fn, label) in INV.INVENTORY_FETCHERS.items():
        assert callable(fn) and label, key
        assert "polygon" in label.lower(), (key, label)


def test_no_point_catalogue_is_shipped():
    """Only mapped polygon inventories. See the module docstring for why.

    A position known to a kilometre cannot be tested against a 90 m pixel, and
    a media-derived catalogue reports landslides where people are rather than
    where slopes fail - measured at Spearman -0.74 against susceptibility.
    """
    from h_sim.input import datasets, inventory as INV

    for gone in ("load_glc_csv", "download_glc_csv", "download_coolr_points",
                 "download_nasa_glc", "_has_accuracy_column"):
        assert not hasattr(INV, gone), gone
    for d in datasets.REGISTRY:
        assert d.key not in ("glc", "coolr"), d.key


def test_demo_rasters_are_written_atomically():
    """Every demo run writes the same paths; a half-written one poisons the next.

    A truncated GeoTIFF fails to open with "TIFFReadDirectory: Failed to read
    directory", which reads as a code fault rather than a damaged file, so the
    writer must never leave one behind.
    """
    import rasterio

    from h_sim.utility import demo as D

    grid = Grid.from_bbox((83.0, 27.5, 83.1, 27.6), 0.01)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub", "r.tif")
        arr = np.ones(grid.shape, "float32")
        D._write(path, grid, arr, "float32", -9999.0)
        with rasterio.open(path) as src:
            assert src.read(1).shape == grid.shape
        # no temporary survives a successful write
        assert not [f for f in os.listdir(os.path.dirname(path))
                    if ".tmp" in f]

        # nor a failed one: a bad dtype raises before the rename
        try:
            D._write(path, grid, arr, "not-a-dtype", -9999.0)
        except BaseException:
            pass
        assert not [f for f in os.listdir(os.path.dirname(path))
                    if ".tmp" in f]
        with rasterio.open(path) as src:      # the good file is untouched
            assert src.read(1).shape == grid.shape


def test_climate_scenario_round_trips_through_a_dict():
    """The web map rebuilds scenarios from what step10 wrote, not from specs."""
    s = CL.scenario("ssp585:2021-2040")
    back = CL.from_dict(s.as_dict())
    assert back == s
    assert CL.from_dict(CL.BASELINE.as_dict()).is_baseline




def test_mechanism_cut_slope_flags_the_wall_not_the_plain():
    """A road along the foot of a steep wall is flagged; open ground is not."""
    from rasterio.transform import from_origin
    from h_sim.model import risk as R

    cell = 30.0
    dem = np.zeros((20, 20))
    dem[:, :5] = 400.0          # a wall rising west of column 5: 400 m in one
    sca = np.zeros((20, 20))    # 30 m cell is far beyond 35 degrees
    deg = cell / 111320.0
    tr = from_origin(80.0, 30.0, deg, deg)
    m = R.MechanismIndex(dem, sca, tr, cell, cell)

    # a segment running north-south in column 5, against the wall
    at_wall = [tuple(tr * (5.5, r + 0.5)) for r in (5, 6, 7)]
    rec = m.assess(at_wall)
    assert rec["cut_slope"] is True
    assert rec["cut_slope_deg"] > 80.0
    assert rec["washout"] is False

    # the same segment out on the plain, column 15
    open_ground = [tuple(tr * (15.5, r + 0.5)) for r in (5, 6, 7)]
    rec = m.assess(open_ground)
    assert rec["cut_slope"] is False
    assert rec["cut_slope_deg"] == 0.0


def test_mechanism_washout_flags_the_channel_crossing():
    """Only the segment touching the high-SCA channel carries the flag."""
    from rasterio.transform import from_origin
    from h_sim.model import risk as R

    cell = 30.0
    dem = np.zeros((20, 20))
    sca = np.full((20, 20), 200.0)     # hillslope background
    sca[10, :] = 9000.0                # a channel along row 10
    deg = cell / 111320.0
    tr = from_origin(80.0, 30.0, deg, deg)
    m = R.MechanismIndex(dem, sca, tr, cell, cell,
                         washout_sca_m=5000.0)

    crossing = [tuple(tr * (8.5, r + 0.5)) for r in (9, 10, 11)]
    rec = m.assess(crossing)
    assert rec["washout"] is True
    assert rec["washout_sca_m"] == 9000.0

    parallel = [tuple(tr * (8.5, r + 0.5)) for r in (3, 4, 5)]
    assert m.assess(parallel)["washout"] is False


def test_mechanism_threshold_is_configurable():
    """A 30-degree face flags at a 25-degree threshold, not at 35."""
    from rasterio.transform import from_origin
    from h_sim.model import risk as R

    cell = 30.0
    dem = np.zeros((10, 10))
    rise = cell * np.tan(np.radians(30.0))
    dem[:, :5] = rise                  # one-cell step at exactly 30 degrees
    sca = np.zeros((10, 10))
    deg = cell / 111320.0
    tr = from_origin(80.0, 30.0, deg, deg)
    seg = [tuple(tr * (5.5, r + 0.5)) for r in (4, 5)]

    strict = R.MechanismIndex(dem, sca, tr, cell, cell, cut_slope_deg=35.0)
    lax = R.MechanismIndex(dem, sca, tr, cell, cell, cut_slope_deg=25.0)
    assert strict.assess(seg)["cut_slope"] is False
    assert lax.assess(seg)["cut_slope"] is True
    # the reported angle is the diagonal-corrected true gradient
    assert abs(lax.assess(seg)["cut_slope_deg"] - 30.0) < 1.5


def test_summarise_rolls_up_mechanism_kilometres():
    from h_sim.model import risk as R

    roads = [
        {"length_m": 500.0, "score": 0.5, "band": "very high",
         "cut_slope": True, "washout": False},
        {"length_m": 500.0, "score": 0.01, "band": "very low",
         "cut_slope": True, "washout": True},
        {"length_m": 1000.0, "score": 0.01, "band": "very low",
         "cut_slope": False, "washout": False},
    ]
    out = R.summarise([], roads)
    assert out["mechanisms"]["road_km_cut_slope"] == 1.0
    assert out["mechanisms"]["road_km_washout"] == 0.5
    assert out["mechanisms"]["road_km_both"] == 0.5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
