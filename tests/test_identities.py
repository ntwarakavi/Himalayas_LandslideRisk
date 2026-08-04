"""Physics and geometry verified as identities, independently derived.

Unit tests elsewhere check behaviour; these check *algebra*. Each test
recomputes the quantity from the underlying mathematics - the infinite-slope
force balance, the Gumbel quantile, planar-slope routing - and requires the
code to agree to numerical precision. If a refactor ever changes a formula,
this file is what notices.
"""

import math

import numpy as np
from rasterio.transform import from_origin

from h_sim.model import hazard as HZ
from h_sim.model import hydrology as H
from h_sim.model import physical as P
from h_sim.model import risk as R


def test_fs_reduces_to_friction_over_slope_when_dry_and_cohesionless():
    th = np.radians(37.0)
    fs = P.factor_of_safety(np.array([np.tan(th)]), np.array([100.0]),
                            0.0, 33.0, 1e-12)
    assert abs(fs[0] - np.tan(np.radians(33.0)) / np.tan(th)) < 1e-6


def test_fs_matches_the_force_balance_written_out_by_hand():
    slope, sca = np.tan(np.radians(35.0)), 800.0
    C, phi, rt, kh, r = 0.12, 33.0, 4e-4, 0.12, 0.5
    s35, c35 = np.sin(np.radians(35.0)), np.cos(np.radians(35.0))
    w = min(rt * sca / s35, 1.0)
    expect = ((C + (c35 - kh * s35 - w * r * c35) * np.tan(np.radians(phi)))
              / (s35 + kh * c35))
    got = P.factor_of_safety(np.array([slope]), np.array([sca]),
                             C, phi, rt, k_h=kh)[0]
    assert abs(got - expect) < 1e-9


def test_newmark_critical_acceleration_is_the_exact_root_of_fs():
    slope, sca = np.array([np.tan(np.radians(35.0))]), np.array([800.0])
    for depth_k in (0.0, 2.0):
        kc = P.critical_acceleration(slope, sca, 0.12, 33.0, 4e-4,
                                     depth_k=depth_k)[0]
        fs = P.factor_of_safety(slope, sca, 0.12, 33.0, 4e-4,
                                k_h=kc, depth_k=depth_k)[0]
        assert abs(fs - 1.0) < 1e-9


def test_negative_critical_acceleration_means_statically_unstable():
    slope, sca = np.array([np.tan(np.radians(55.0))]), np.array([3000.0])
    kc = P.critical_acceleration(slope, sca, 0.0, 30.0, 2e-3)[0]
    fs0 = P.factor_of_safety(slope, sca, 0.0, 30.0, 2e-3)[0]
    assert (kc < 0) == (fs0 < 1)


def test_gumbel_multiplier_matches_independent_algebra():
    assert abs(HZ.recharge_multiplier(2.0) - 1.0) < 1e-12
    y = lambda T: -math.log(-math.log(1.0 - 1.0 / T))       # noqa: E731
    k = lambda T: math.sqrt(6) / math.pi * (y(T) - 0.5772156649015329)  # noqa: E731
    for T in (10.0, 100.0, 1000.0):
        expect = (1 + 0.30 * k(T)) / (1 + 0.30 * k(2.0))
        assert abs(HZ.recharge_multiplier(T) - expect) < 1e-9
    m = [HZ.recharge_multiplier(T) for T in (10.0, 100.0, 1000.0)]
    assert m[0] < m[1] < m[2] < 3.0


def test_sca_grows_linearly_down_a_planar_slope():
    n, cell = 40, 30.0
    dem = np.arange(n, 0, -1, dtype=float)[:, None] * np.ones((1, n)) * 10
    sca, _ = H.specific_catchment_area(dem, cell, cell)
    mid = sca[:, n // 2]
    assert abs(mid[30] / mid[10] - 31 / 11) < 0.15


def test_dependence_conserves_mass_through_the_target_patch():
    n, cell = 40, 30.0
    dem = np.arange(n, 0, -1, dtype=float)[:, None] * np.ones((1, n)) * 10
    filled = H.fill_depressions(dem)
    ang, _ = H.dinf_flow_direction(filled, cell, cell)
    dep = H.dinf_dependence(filled, ang, row=30, col=20)
    # everything that crosses a full row upslope of a 3-wide patch is exactly
    # the water that will pass through the patch: the row must sum to 3
    assert abs(dep[10].sum() - 3.0) < 0.05


def test_reach_azimuth_criterion_and_weights():
    n, cell = 40, 30.0
    dem = np.arange(n, 0, -1, dtype=float)[:, None] * np.ones((1, n)) * 10
    deg = cell / 111320.0
    tr = from_origin(80.0, 30.0, deg, deg)
    idx = R.ReachIndex(dem, tr, cell, cell)
    rch = idx.reach(*tuple(tr * (20.5, 35.5)))
    due_n = (rch.cols == 20) & (rch.rows == 25)
    assert due_n.any() and abs(rch.az[due_n][0]) < 1e-9
    assert np.all(rch.relief / rch.dist
                  > np.tan(np.radians(idx.travel_angle_deg)) - 1e-12)
    assert np.allclose(rch.weight, 1.0 / rch.dist)


def test_sector_octant_labels():
    lab = lambda az: R.SECTORS[int(round(az / 45.0)) % 8]  # noqa: E731
    assert lab(350) == "N" and lab(40) == "NE" and lab(100) == "E"
    assert lab(200) == "S" and lab(230) == "SW" and lab(280) == "W"
