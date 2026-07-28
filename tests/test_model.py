"""Unit + end-to-end smoke tests for the GIRI landslide model.

Run with:  python -m pytest tests/ -q      (or)   python tests/test_model.py
No network required - everything uses the synthetic demo generator.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from giri_landslide import config as C
from giri_landslide import pipeline
from giri_landslide.input import inventory
from giri_landslide.model import calibrate, triggers
from giri_landslide.utility.grid import (Grid, reclassify_continuous,
                                         remap_categorical)


def test_grid_dimensions():
    g = Grid.from_bbox((10.0, 20.0, 11.0, 21.0), 0.01)
    assert g.width == 100 and g.height == 100
    assert abs(g.transform.a - 0.01) < 1e-12


def test_slope_reclass_table():
    arr = np.array([[0.0, 6.0, 30.0, 38.0, 55.0]])
    out = reclassify_continuous(arr, C.SLOPE_BREAKS_DEG, inclusive=False)
    # 0->0(<6), 6->1, 30->5, 38->4(36-40), 55->1(>50)
    assert out.tolist() == [[0, 1, 5, 4, 1]]


def test_landcover_remap():
    arr = np.array([[10, 50, 60, 80]])  # tree, built, bare, water
    out = remap_categorical(arr, C.WORLDCOVER_SV, C.WORLDCOVER_SV_NODATA)
    assert out.tolist() == [[2, 1, 5, 0]]


def test_gumbel_return_period_monotonic():
    z = np.array([-1.0, 0.0, 0.72, 2.6, 4.9])
    T = triggers.gumbel_return_period(z)
    assert np.all(np.diff(T) > 0)
    # z ~ 0.72 should map to roughly a 5-10 yr return period (paper Table 6)
    assert 4.0 < T[2] < 12.0


def test_pga_classes():
    thr = C.PGA_THRESHOLDS_G
    assert len(thr) == 5 and thr[0] == 0.05


def test_hazard_matrix_shapes():
    assert np.array(C.EARTHQUAKE_MATRIX).shape == (5, 5)
    assert np.array(C.RAINFALL_MATRIX).shape == (5, 5)
    # earthquake matrix values match the manuscript corners
    assert C.EARTHQUAKE_MATRIX[4][4] == 0.40
    assert C.EARTHQUAKE_MATRIX[0][3] == 0.001


def test_end_to_end_demo_rainfall():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = C.Config(
            name="t", bbox=(84.0, 28.0, 84.2, 28.2), resolution_deg=0.004,
            trigger="rainfall", block_size=64,
            data_dir=os.path.join(tmp, "raw"),
            work_dir=os.path.join(tmp, "work"),
            out_dir=os.path.join(tmp, "out"),
        )
        out = pipeline.run(cfg, mode="demo")
        for key in ("susceptibility_classes", "susceptibility_probability",
                    "hazard_probability", "summary"):
            assert os.path.exists(out[key]), key

        import rasterio
        # the continuous index must be a genuine 0-1 field
        with rasterio.open(out["susceptibility_probability"]) as src:
            p = src.read(1)
            p = p[p != src.nodata]
            assert p.min() >= 0.0 and p.max() <= 1.0
            assert len(np.unique(p)) > 5, "index should be continuous"
        with rasterio.open(out["susceptibility_classes"]) as src:
            a = src.read(1)
            classes = set(int(x) for x in np.unique(a) if x != src.nodata)
            assert classes.issubset({1, 2, 3, 4, 5})
            assert len(classes) >= 3  # model discriminates
        with rasterio.open(out["hazard_probability"]) as src:
            h = src.read(1)
            h = h[h != src.nodata]
            assert h.min() >= 0.0 and h.max() <= 1.0


def test_end_to_end_demo_earthquake():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = C.Config(
            name="teq", bbox=(84.0, 28.0, 84.2, 28.2), resolution_deg=0.004,
            trigger="earthquake", block_size=64, scenario_pga_g=0.4,
            data_dir=os.path.join(tmp, "raw"),
            work_dir=os.path.join(tmp, "work"),
            out_dir=os.path.join(tmp, "out"),
        )
        out = pipeline.run(cfg, mode="demo")
        assert os.path.exists(out["hazard_probability"])


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


def test_inventory_csv_and_bbox_filter(tmp_path=None):
    import tempfile
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
    # A square polygon in UTM 45N (EPSG:32645) over central Nepal.
    geom = {"type": "Polygon",
            "coordinates": [[(300000.0, 3050000.0), (300100.0, 3050000.0),
                             (300100.0, 3050100.0), (300000.0, 3050100.0),
                             (300000.0, 3050000.0)]]}
    xy = inventory._representative_point(geom)
    assert xy == (300040.0, 3050040.0)  # coordinate centroid (ring closes)

    # A point geometry is passed through untouched.
    assert inventory._representative_point(
        {"type": "Point", "coordinates": [85.3, 27.7]}) == (85.3, 27.7)

    # Empty geometry yields nothing rather than raising.
    assert inventory._representative_point(
        {"type": "Polygon", "coordinates": []}) is None


def test_glc_accuracy_screening():
    """GLC loading must drop records too coarsely placed to test a 90 m map."""
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
        # country and trigger filters
        assert len(inventory.load_glc_csv(p, countries=("Nepal",))) == 2
        assert len(inventory.load_glc_csv(
            p, countries=("Nepal",), triggers=("rain",))) == 1
        # bbox clipping
        assert len(inventory.load_glc_csv(
            p, bbox=(84.0, 28.0, 84.15, 28.15))) == 1


def test_auc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    y = np.array([0, 0, 0, 1, 1, 1])
    assert abs(calibrate._auc(scores, y) - 1.0) < 1e-9


def test_calibration_recovers_signal():
    """Synthetic presence drawn from susceptibility -> high AUC, slope leads.

    Pinned to ordinal features: the demo inventory is generated from the ordinal
    factor scores, so that is the space the planted signal lives in.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = C.Config(
            name="cal", bbox=(83.0, 27.5, 85.0, 29.0), resolution_deg=0.006,
            block_size=256, feature_mode="ordinal",
            data_dir=os.path.join(tmp, "raw"),
            work_dir=os.path.join(tmp, "work"),
            out_dir=os.path.join(tmp, "out"),
        )
        report = pipeline.run_calibration(cfg, mode="demo")
        res = report["result"]
        assert res["auc"] > 0.7, res["auc"]
        assert res["weights"]["slope"] == max(res["weights"].values())
        assert os.path.exists(report["calibrated_config"])
        # calibrated config must switch to exponent + quantile
        cal = C.Config.from_json(report["calibrated_config"])
        assert cal.weight_mode == "exponent"
        assert cal.classification == "quantile"


def test_slope_break_calibration_recovers_shape():
    """Frequency-ratio fit should peak where landslides are over-represented."""
    rng = np.random.default_rng(0)
    # Background terrain spans 0-60 deg; landslides concentrate near 30 deg.
    background = rng.uniform(0, 60, 4000)
    presence = np.clip(rng.normal(30, 4, 1200), 0, 60)
    breaks, diag = calibrate.calibrate_slope_breaks(presence, background)

    assert 24.0 <= diag["peak_fr_slope_deg"] <= 36.0, diag["peak_fr_slope_deg"]
    # Flat ground must stay at factor 0 regardless of the data.
    assert breaks[0][1] == 0 and breaks[0][0] >= 6.0
    # Breaks must be ascending and end at infinity.
    bounds = [b for b, _ in breaks]
    assert bounds == sorted(bounds) and bounds[-1] == float("inf")
    # The fitted table must be usable as a config value.
    cfg = C.Config(slope_breaks=breaks)
    assert cfg.slope_breaks[0][1] == 0


def test_quantile_breaks_never_empty_class():
    """Lumpy S distributions must not leave a class with an empty interval."""
    import rasterio
    from giri_landslide.model import susceptibility as S
    from giri_landslide.utility.grid import Grid

    with tempfile.TemporaryDirectory() as tmp:
        grid = Grid.from_bbox((84.0, 28.0, 84.1, 28.1), 0.001)
        # Only three distinct positive values -> naive quantiles collapse.
        rng = np.random.default_rng(0)
        arr = rng.choice([0.0, 12.0, 12.0, 40.0, 90.0],
                         size=grid.shape).astype("float32")
        p = os.path.join(tmp, "idx.tif")
        with rasterio.open(p, "w", **grid.profile("float32", -9999.0)) as dst:
            dst.write(arr, 1)

        breaks = S.quantile_breaks(p)
        cuts = [b for b, _ in breaks[:-1]]
        assert cuts == sorted(set(cuts)), f"duplicate cuts: {cuts}"
        labels = [c for _, c in breaks]
        assert labels == list(range(1, len(breaks) + 1))


def test_continuous_features_reduce_ties():
    """Continuous features must give a far smoother index than ordinal scores."""
    import rasterio

    def distinct_values(cfg):
        out = pipeline.run_susceptibility(cfg, mode="demo")
        with rasterio.open(out["susceptibility_probability"]) as src:
            a = src.read(1)
            return len(np.unique(a[a != src.nodata]))

    with tempfile.TemporaryDirectory() as tmp:
        common = dict(bbox=(83.0, 27.5, 84.5, 28.8), resolution_deg=0.005,
                      block_size=256, data_dir=os.path.join(tmp, "raw"),
                      work_dir=os.path.join(tmp, "work"),
                      out_dir=os.path.join(tmp, "out"))
        fw = {"slope": 1.0, "slope_sq": -0.4, "lithology": 0.6,
              "vegetation": 0.3, "soil_moisture": 0.8}
        n_ord = distinct_values(C.Config(
            name="ord", feature_mode="ordinal", **common,
            feature_weights={k: 1.0 for k in
                             ("slope", "lithology", "vegetation",
                              "soil_moisture")}))
        n_cont = distinct_values(C.Config(
            name="cont", feature_mode="continuous", feature_weights=fw,
            **common))

        assert n_cont > 10 * n_ord, (n_ord, n_cont)


def test_spatial_blocks_separate_train_and_test():
    """A spatial fold must withhold whole blocks, not scattered points."""
    from giri_landslide.model import crossval

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


def test_validate_handles_continuous_index():
    """Validation must score the continuous index, not only the class map."""
    import rasterio

    from giri_landslide.model import validate
    from giri_landslide.utility.grid import Grid

    with tempfile.TemporaryDirectory() as tmp:
        grid = Grid.from_bbox((84.0, 28.0, 84.5, 28.5), 0.01)   # 50x50
        rng = np.random.default_rng(0)
        idx = rng.uniform(0.0, 1.0, grid.shape).astype("float32")
        path = os.path.join(tmp, "prob.tif")
        with rasterio.open(path, "w", **grid.profile("float32", -9999.0)) as d:
            d.write(idx, 1)
        assert validate.is_continuous(path)

        # Landslides placed only where the index is high must score FR > 1 in
        # the top bin and produce a monotonic ordering.
        rows, cols = np.where(idx > 0.9)
        xs, ys = rasterio.transform.xy(grid.transform, rows, cols)
        pres = np.column_stack([np.asarray(xs), np.asarray(ys)])
        bg = inventory.background_points((84.0, 28.0, 84.5, 28.5), 800, path)

        r = validate.validate_susceptibility(path, pres, bg)
        assert r.frequency_ratio["5"] > 1.0
        assert r.monotonic, r.frequency_ratio
        assert r.auc > 0.9, r.auc            # perfect separation by construction


def test_compare_susceptibility():
    """Difference map should report the class shift between two runs."""
    import rasterio
    from giri_landslide.utility.grid import Grid

    with tempfile.TemporaryDirectory() as tmp:
        grid = Grid.from_bbox((84.0, 28.0, 84.1, 28.1), 0.01)
        base = np.full(grid.shape, 2, dtype="uint8")
        scen = base.copy()
        scen[:, :5] = 4                       # left half gains 2 classes
        paths = []
        for name, arr in (("base", base), ("scen", scen)):
            p = os.path.join(tmp, f"{name}.tif")
            with rasterio.open(p, "w", **grid.profile("uint8", 255)) as dst:
                dst.write(arr, 1)
            paths.append(p)

        out = pipeline.compare_susceptibility(paths[0], paths[1],
                                              os.path.join(tmp, "cmp"))
        assert os.path.exists(out["change"])
        stats = json.load(open(out["summary"]))
        assert stats["class_change_histogram"]["2"] == 50
        assert stats["pct_increased"] == 50.0
        assert stats["pct_decreased"] == 0.0

        # Mismatched grids must be rejected, not silently compared.
        other = Grid.from_bbox((84.0, 28.0, 84.1, 28.1), 0.005)
        p3 = os.path.join(tmp, "other.tif")
        with rasterio.open(p3, "w", **other.profile("uint8", 255)) as dst:
            dst.write(np.full(other.shape, 2, dtype="uint8"), 1)
        try:
            pipeline.compare_susceptibility(paths[0], p3,
                                            os.path.join(tmp, "bad"))
            assert False, "expected mismatched grids to raise"
        except ValueError:
            pass


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

        # Offline check never probes the network and reports the cache.
        rows = datasets.check_all(tmp, probe=False, keys=["coolr"])
        assert rows[0]["cached"] is True and rows[0]["reachable"] is None
        assert "CACHED" in datasets.format_report(rows)


def test_susceptibility_and_hazard_steps_compose():
    """Step 4 then step 5 must give the same result as the combined run."""
    import rasterio

    with tempfile.TemporaryDirectory() as tmp:
        common = dict(bbox=(84.0, 28.0, 84.2, 28.2), resolution_deg=0.004,
                      block_size=64, data_dir=os.path.join(tmp, "raw"),
                      work_dir=os.path.join(tmp, "work"),
                      out_dir=os.path.join(tmp, "out"))
        stepwise = C.Config(name="stepwise", **common)
        s = pipeline.run_susceptibility(stepwise, mode="demo")
        h = pipeline.run_hazard(stepwise, s["susceptibility_classes"],
                                mode="demo")

        oneshot = C.Config(name="oneshot", **common)
        combined = pipeline.run(oneshot, mode="demo")

        with rasterio.open(h["hazard_probability"]) as a, \
                rasterio.open(combined["hazard_probability"]) as b:
            assert np.array_equal(a.read(1), b.read(1))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
