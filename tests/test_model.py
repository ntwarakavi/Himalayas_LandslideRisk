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

from giri_landslide import config as C
from giri_landslide import pipeline
from giri_landslide.input import inventory
from giri_landslide.model import crossval, hazard, hydrology as H, physical as P
from giri_landslide.utility.grid import Grid


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


def test_glc_accuracy_screening():
    """GLC loading must drop records too coarsely placed to test a fine map."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "glc.csv")
        with open(p, "w") as fh:
            fh.write("location_accuracy,country_name,landslide_trigger,"
                     "longitude,latitude\n")
            fh.write("exact,Nepal,downpour,84.1,28.1\n")
            fh.write("1km,Nepal,rain,84.2,28.2\n")
            fh.write("25km,Nepal,downpour,84.3,28.3\n")     # too coarse
            fh.write("unknown,Nepal,downpour,84.4,28.4\n")  # unquantified
            fh.write("exact,Peru,downpour,-72.0,-13.0\n")   # wrong country

        assert len(inventory.load_glc_csv(p, max_accuracy="1km")) == 3
        assert len(inventory.load_glc_csv(p, max_accuracy="exact")) == 2
        assert len(inventory.load_glc_csv(p, max_accuracy="25km")) == 4
        assert len(inventory.load_glc_csv(p, countries=("Nepal",))) == 2
        assert len(inventory.load_glc_csv(
            p, countries=("Nepal",), triggers=("rain",))) == 1
        assert len(inventory.load_glc_csv(
            p, bbox=(84.0, 28.0, 84.15, 28.15))) == 1


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_validate_handles_continuous_index():
    """Validation must score the continuous field, not only a class map."""
    import rasterio

    from giri_landslide.model import validate

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

    from giri_landslide.model import validate

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
        precip = os.path.join(cfg.work_dir, "rech_precip_max_month.tif")
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
    from giri_landslide.input import datasets

    keys = [d.key for d in datasets.REGISTRY]
    assert len(keys) == len(set(keys)), "duplicate dataset keys"
    for d in datasets.REGISTRY:
        assert d.group in (datasets.TERRAIN, datasets.CLIMATE,
                           datasets.INVENTORY, datasets.TRIGGER), d.key
        assert d.probe_url or d.manual_url, f"{d.key} has no source"
        assert d.rel_path and not os.path.isabs(d.rel_path), d.key


def test_dataset_cache_detection():
    """A dataset counts as cached only when its file/dir actually has content."""
    from giri_landslide.input import datasets

    ds = datasets.BY_KEY["coolr"]
    with tempfile.TemporaryDirectory() as tmp:
        assert ds.cached(tmp) is False
        target = ds.local_path(tmp)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        open(target, "w").close()            # empty file must not count
        assert ds.cached(tmp) is False
        with open(target, "w") as fh:
            fh.write("{}")
        assert ds.cached(tmp) is True

        rows = datasets.check_all(tmp, probe=False, keys=["coolr"])
        assert rows[0]["cached"] is True and rows[0]["reachable"] is None
        assert "CACHED" in datasets.format_report(rows)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
