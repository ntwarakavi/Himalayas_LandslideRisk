"""Physically based slope stability: infinite slope with steady-state wetness.

This is the SINMAP formulation (Pack, Tarboton & Goodwin 1998), which couples
the infinite-slope factor of safety to a TOPMODEL-style steady-state hydrology.
It replaces the heuristic index with a mechanical statement about when a
hillslope fails.

Factor of safety
----------------

For a planar failure surface parallel to the ground, with the slide much wider
and longer than it is deep, the balance of driving and resisting forces reduces
to

    FS = [ C + cos(theta) * (1 - w * r) * tan(phi) ] / sin(theta)

where

    C      dimensionless cohesion, (Cr + Cs) / (h * rho_s * g), combining root
           and soil cohesion, normalised by the weight of the soil column
    theta  slope angle
    phi    angle of internal friction
    r      density ratio rho_w / rho_s, about 0.5
    w      relative wetness, the fraction of the soil column that is saturated

Wetness comes from a steady-state balance: recharge R falling on the upslope
contributing area a must pass through the soil column, whose ability to
transmit water is the transmissivity T.

    w = min( R * a / (T * sin(theta)), 1 )

Only the ratio R/T matters, which is convenient because it is far better
constrained than either term alone. Wetness is capped at 1: any excess becomes
overland flow rather than deeper saturation.

Stability index
---------------

The parameters are not known per pixel. SINMAP treats C, tan(phi) and R/T as
uniform over plausible ranges and reports the probability that the slope
stands. Here the complement is returned - the probability of failure - so the
output is directly comparable with the heuristic susceptibility index and can
be validated the same way.

Two limits are worth naming because they appear as constant regions on the map:

* **Unconditionally stable** - stable even fully saturated, at the most
  pessimistic parameters. Failure probability 0.
* **Unconditionally unstable** - unstable even completely dry, at the most
  optimistic parameters. Failure probability 1. Such terrain stands only
  through cohesion the model does not represent, or is actively eroding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

#: Density of water over density of saturated soil. Fairly stable across soils;
#: SINMAP's default.
DENSITY_RATIO = 0.5


@dataclass
class SoilParameters:
    """Uniform ranges for the three uncertain parameters.

    Defaults are SINMAP's generic values, appropriate for a soil-mantled
    mountain catchment before any local calibration.

    cohesion
        Dimensionless, already normalised by soil depth and unit weight. 0
        means cohesionless; values above about 0.5 hold very steep slopes.
    friction_deg
        Internal friction angle in degrees. Most soils fall between 30 and 45.
    rt
        R/T in units of 1/m. Small values mean a highly transmissive soil that
        drains readily; large values mean water backs up and wetness builds.
    """

    cohesion: Tuple[float, float] = (0.0, 0.25)
    friction_deg: Tuple[float, float] = (30.0, 45.0)
    rt: Tuple[float, float] = (0.0001, 0.01)

    def as_dict(self) -> dict:
        return {"cohesion": list(self.cohesion),
                "friction_deg": list(self.friction_deg),
                "rt": list(self.rt)}


def wetness(sca: np.ndarray, slope: np.ndarray, rt: float) -> np.ndarray:
    """Relative wetness w = min(R a / (T sin theta), 1)."""
    theta = np.arctan(slope)
    sin_t = np.sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = rt * sca / np.where(sin_t > 1e-6, sin_t, np.nan)
    return np.clip(w, 0.0, 1.0)


def factor_of_safety(slope: np.ndarray, sca: np.ndarray, cohesion: float,
                     friction_deg: float, rt: float,
                     density_ratio: float = DENSITY_RATIO) -> np.ndarray:
    """Infinite-slope factor of safety under steady-state wetness."""
    theta = np.arctan(slope)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    w = wetness(sca, slope, rt)
    tan_phi = np.tan(np.radians(friction_deg))

    with np.errstate(divide="ignore", invalid="ignore"):
        fs = (cohesion + cos_t * (1.0 - w * density_ratio) * tan_phi) / \
            np.where(sin_t > 1e-6, sin_t, np.nan)
    # Flat ground cannot fail by sliding; report it as unconditionally stable
    # rather than as a division blow-up.
    return np.where(sin_t <= 1e-6, np.inf, fs)


def failure_probability(slope: np.ndarray, sca: np.ndarray,
                        params: SoilParameters, n_samples: int = 200,
                        seed: int = 0,
                        density_ratio: float = DENSITY_RATIO) -> np.ndarray:
    """Probability that FS < 1, over the uniform parameter ranges.

    Parameter triples are drawn once and applied to every pixel, which is what
    the marginal per-pixel probability requires and keeps the cost to
    ``n_samples`` passes over the block rather than a draw per pixel.
    """
    rng = np.random.default_rng(seed)
    c = rng.uniform(*params.cohesion, n_samples)
    phi = rng.uniform(*params.friction_deg, n_samples)
    rt = rng.uniform(*params.rt, n_samples)

    valid = np.isfinite(slope) & np.isfinite(sca)
    fails = np.zeros(slope.shape, dtype="float64")
    for i in range(n_samples):
        fs = factor_of_safety(slope, sca, c[i], phi[i], rt[i], density_ratio)
        fails += (fs < 1.0)
    p = fails / n_samples
    return np.where(valid, p, np.nan)


def stability_classes(slope: np.ndarray, sca: np.ndarray,
                      params: SoilParameters,
                      density_ratio: float = DENSITY_RATIO) -> np.ndarray:
    """SINMAP stability classes, from the extremes of the parameter ranges.

    1 unconditionally stable   - stable when saturated at worst-case parameters
    2 stable                   - FS > 1.25 at worst case
    3 quasi-stable             - FS > 1.0 at worst case
    4 lower threshold          - failure possible, probability below 0.5
    5 upper threshold          - failure probability at least 0.5
    6 unconditionally unstable - unstable when dry at best-case parameters
    """
    c_lo, c_hi = params.cohesion
    phi_lo, phi_hi = params.friction_deg
    rt_lo, rt_hi = params.rt

    # Worst case: least cohesion, least friction, wettest.
    fs_worst = factor_of_safety(slope, sca, c_lo, phi_lo, rt_hi, density_ratio)
    # Best case, and dry, so wetness plays no part.
    fs_best_dry = factor_of_safety(slope, np.zeros_like(sca), c_hi, phi_hi,
                                   rt_lo, density_ratio)
    p = failure_probability(slope, sca, params, n_samples=100,
                            density_ratio=density_ratio)

    cls = np.full(slope.shape, np.nan)
    cls = np.where(p >= 0.5, 5.0, cls)
    cls = np.where((p > 0) & (p < 0.5), 4.0, cls)
    cls = np.where(p <= 0, 3.0, cls)
    cls = np.where(fs_worst > 1.25, 2.0, cls)
    cls = np.where(np.isfinite(fs_worst) & (fs_worst > 1.5), 1.0, cls)
    cls = np.where(fs_best_dry < 1.0, 6.0, cls)
    return np.where(np.isfinite(slope), cls, np.nan)


# ---------------------------------------------------------------------------
# Fitting the parameter ranges to an inventory
# ---------------------------------------------------------------------------

def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype="float64")
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def fit_parameters(slope_pres: np.ndarray, sca_pres: np.ndarray,
                   slope_bg: np.ndarray, sca_bg: np.ndarray,
                   n_samples: int = 60, seed: int = 0) -> dict:
    """Search soil parameter ranges that best separate landslides from terrain.

    The physics fixes the *form* of the response; what a region's soils and
    hydrology supply are the parameter values. Those are searched over ranges
    reported for soil-mantled mountain terrain, scored by how well the
    resulting failure probability ranks mapped landslides above background.

    Only the ratio R/T is identifiable from a static map, not R and T
    separately, and cohesion is identifiable only jointly with soil depth,
    since the model sees the combination C = (Cr + Cs) / (h rho_s g).
    """
    sp = np.asarray(slope_pres, "float64")
    ap = np.asarray(sca_pres, "float64")
    sb = np.asarray(slope_bg, "float64")
    ab = np.asarray(sca_bg, "float64")
    ok_p = np.isfinite(sp) & np.isfinite(ap)
    ok_b = np.isfinite(sb) & np.isfinite(ab)
    sp, ap, sb, ab = sp[ok_p], ap[ok_p], sb[ok_b], ab[ok_b]
    if len(sp) < 20 or len(sb) < 20:
        raise ValueError("too few valid samples to fit soil parameters")

    slope = np.concatenate([sp, sb])
    sca = np.concatenate([ap, ab])
    y = np.concatenate([np.ones(len(sp)), np.zeros(len(sb))])

    # Ranges spanning what is reported for soil-mantled mountain hillslopes.
    grid = []
    for c_hi in (0.05, 0.15, 0.25, 0.40):
        for phi_lo, phi_hi in ((25.0, 35.0), (30.0, 40.0), (35.0, 45.0)):
            for rt_hi in (0.0005, 0.002, 0.01, 0.05):
                grid.append(SoilParameters((0.0, c_hi), (phi_lo, phi_hi),
                                           (rt_hi / 50.0, rt_hi)))

    best, best_auc = None, -np.inf
    trials = []
    for params in grid:
        p = failure_probability(slope, sca, params, n_samples=n_samples,
                                seed=seed)
        ok = np.isfinite(p)
        if ok.sum() < 20:
            continue
        auc = _auc(p[ok], y[ok])
        if np.isfinite(auc):
            trials.append({"params": params.as_dict(), "auc": round(auc, 4)})
            if auc > best_auc:
                best, best_auc = params, auc

    if best is None:
        raise ValueError("no parameter set produced a usable score")
    trials.sort(key=lambda t: -t["auc"])
    return {"parameters": best, "auc": float(best_auc),
            "n_presence": int(len(sp)), "n_background": int(len(sb)),
            "top_trials": trials[:5], "n_trials": len(trials)}
