"""Unit + end-to-end smoke tests for the GIRI landslide model.

Run with:  python -m pytest tests/ -q      (or)   python tests/test_model.py
No network required - everything uses the synthetic demo generator.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from giri_landslide import config as C
from giri_landslide import pipeline, triggers
from giri_landslide.grid import Grid, reclassify_continuous, remap_categorical


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
        for key in ("susceptibility", "hazard_probability", "summary"):
            assert os.path.exists(out[key]), key

        import rasterio
        with rasterio.open(out["susceptibility"]) as src:
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
