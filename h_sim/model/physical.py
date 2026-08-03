"""Slope stability: infinite slope with steady-state wetness (SINMAP).

This is the model. It follows SINMAP (Pack, Tarboton & Goodwin 1998), which
couples an infinite-slope factor of safety to a TOPMODEL-style steady-state
hydrology, extended here with a pseudo-static term so the same balance covers
seismic triggering.

Factor of safety
----------------

For a planar failure surface parallel to the ground, with the slide much wider
and longer than it is deep, the balance of driving and resisting forces on a
column of unit plan area reduces to

    FS = [ C + (cos(t) - k*sin(t) - w*r*cos(t)) * tan(phi) ]
         / ( sin(t) + k*cos(t) )

where

    C      dimensionless cohesion, (Cr + Cs) / (h * rho_s * g), combining root
           and soil cohesion, normalised by the weight of the soil column
    t      slope angle
    phi    angle of internal friction
    r      density ratio rho_w / rho_s, about 0.5
    w      relative wetness, the fraction of the soil column that is saturated
    k      horizontal seismic coefficient, zero for rainfall triggering

With k = 0 this is SINMAP's published form, FS = [C + cos(t)(1 - w r)tan(phi)]
/ sin(t). The seismic terms enter as an extra driving force k*W along the slope
and a matching reduction k*W*sin(t) in the normal force; pore pressure is
unaffected by inertia, which is why the w term keeps its static form.

Wetness comes from a steady-state balance: recharge R falling on the upslope
contributing area a must pass through the soil column, whose ability to
transmit water is the transmissivity T.

    w = min( R * a / (T * sin(t)), 1 )

Only the ratio R/T matters, which is convenient because it is far better
constrained than either term alone. Wetness is capped at 1: any excess becomes
overland flow rather than deeper saturation.

Stability index
---------------

The parameters are not known per pixel. SINMAP treats C, tan(phi) and R/T as
uniform over plausible ranges and reports the probability that the slope
stands. Here the complement is returned - the probability of failure - so the
output is a continuous field in [0, 1] that can be validated against an
inventory the same way any susceptibility map can.

Two limits are worth naming because they appear as constant regions on the map:

* **Unconditionally stable** - stable even fully saturated, at the most
  pessimistic parameters. Failure probability 0.
* **Unconditionally unstable** - unstable even completely dry, at the most
  optimistic parameters. Failure probability 1. Such terrain stands only
  through cohesion the model does not represent, or is actively eroding.

What is identifiable
--------------------

Fitting to a map of past failures constrains less than the parameter list
suggests, and the limits are worth stating plainly:

* R and T are identifiable only as their ratio.
* Cohesion is identifiable only jointly with soil depth, since the model sees
  the combination C = (Cr + Cs) / (h rho_s g). A soil-depth map would separate
  them; none is used here.
* The absolute level of the failure probability depends on how background
  points were drawn. Differences between pixels are meaningful; the value at a
  pixel is not a frequency of failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import crossval

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
    depth_k
        Rate of soil-depth decline with gradient, ``h = h0 exp(-depth_k
        tan(theta))`` - the exponential thinning of regolith on steepening
        slopes measured in soil-mantled terrain (DeRose 1996) and used for
        distributed effective depth in TOPMODEL (Saulnier et al. 1997; cf.
        Catani et al. 2010, GIST). The model never sees ``h`` itself, only
        the two parameters that carry ``1/h`` - dimensionless cohesion and
        R/T - so a nonzero ``depth_k`` multiplies both by the per-pixel
        factor of :func:`depth_factor`, anchored so the fitted ranges keep
        their face values at a 30-degree slope. Zero, the default and the
        published SINMAP, is uniform depth.
    """

    cohesion: Tuple[float, float] = (0.0, 0.25)
    friction_deg: Tuple[float, float] = (30.0, 45.0)
    rt: Tuple[float, float] = (0.0001, 0.01)
    depth_k: float = 0.0

    def as_dict(self) -> dict:
        return {"cohesion": list(self.cohesion),
                "friction_deg": list(self.friction_deg),
                "rt": list(self.rt),
                "depth_k": self.depth_k}

    @classmethod
    def from_dict(cls, raw: dict) -> "SoilParameters":
        return cls(cohesion=tuple(raw["cohesion"]),
                   friction_deg=tuple(raw["friction_deg"]),
                   rt=tuple(raw["rt"]),
                   # Fits written before the depth term existed load as the
                   # uniform-depth model they were.
                   depth_k=float(raw.get("depth_k", 0.0)))


# ---------------------------------------------------------------------------
# the mechanics
# ---------------------------------------------------------------------------

def wetness(sca: np.ndarray, slope: np.ndarray, rt) -> np.ndarray:
    """Relative wetness w = min(R a / (T sin theta), 1).

    ``rt`` may be a scalar or an array the shape of ``sca``, which is how a
    spatially varying recharge enters.
    """
    theta = np.arctan(slope)
    sin_t = np.sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = rt * sca / np.where(sin_t > 1e-6, sin_t, np.nan)
    return np.clip(w, 0.0, 1.0)


#: The depth term is anchored here: fitted cohesion and R/T keep their face
#: values at tan(30 deg), the middle of the band where shallow failures
#: happen, and scale up (thinner soil) or down (thicker) either side.
DEPTH_REF_TAN = float(np.tan(np.radians(30.0)))

#: The depth law is capped at this gradient. Regolith-thinning relations are
#: measured on soil-mantled slopes; past about 60 degrees there is no soil
#: column left for an exponential to describe, and extrapolating it there
#: would only inflate cohesion on rock faces the model cannot speak to.
DEPTH_CAP_TAN = float(np.tan(np.radians(60.0)))


def depth_factor(slope, depth_k: float):
    """Per-pixel ``1/h`` scaling for slope-dependent soil depth.

    With ``h = h0 exp(-depth_k tan(theta))``, everything the model reads
    through ``1/h`` - the dimensionless cohesion and R/T - gains the factor
    ``exp(depth_k (tan(theta) - tan 30 deg))``, capped at 60 degrees.
    Returns the scalar 1.0 when ``depth_k`` is zero, so the default model
    pays nothing.
    """
    if not depth_k:
        return 1.0
    t = np.minimum(np.asarray(slope, "float64"), DEPTH_CAP_TAN)
    return np.exp(depth_k * (t - DEPTH_REF_TAN))


def factor_of_safety(slope: np.ndarray, sca: np.ndarray, cohesion: float,
                     friction_deg: float, rt,
                     density_ratio: float = DENSITY_RATIO,
                     k_h=0.0, depth_k: float = 0.0) -> np.ndarray:
    """Infinite-slope factor of safety under steady-state wetness.

    ``k_h`` is the horizontal seismic coefficient (dimensionless, a fraction of
    g). Zero gives the static case. A nonzero ``depth_k`` applies the
    slope-dependent soil depth of :func:`depth_factor`: thinner soil on
    steeper ground is at once relatively more root-bound (cohesion up) and
    less transmissive (R/T up, so wetter), which is the physically coupled
    pair - both are the same ``1/h``.
    """
    theta = np.arctan(slope)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    f = depth_factor(slope, depth_k)
    w = wetness(sca, slope, rt * f)
    tan_phi = np.tan(np.radians(friction_deg))

    numer = (cohesion * f
             + (cos_t - k_h * sin_t - w * density_ratio * cos_t) * tan_phi)
    denom = sin_t + k_h * cos_t
    with np.errstate(divide="ignore", invalid="ignore"):
        fs = numer / np.where(denom > 1e-6, denom, np.nan)
    # Flat ground with no seismic driving force cannot fail by sliding; report
    # it as unconditionally stable rather than as a division blow-up.
    return np.where(denom <= 1e-6, np.inf, fs)


def critical_acceleration(slope: np.ndarray, sca: np.ndarray, cohesion: float,
                          friction_deg: float, rt,
                          density_ratio: float = DENSITY_RATIO,
                          depth_k: float = 0.0) -> np.ndarray:
    """Newmark critical acceleration k_c, in g: the k_h at which FS reaches 1.

    Solving FS = 1 for the seismic coefficient gives

        k_c = [C + cos(t)(1 - w r)tan(phi) - sin(t)] / (cos(t) + sin(t)tan(phi))

    whose numerator is sin(t) * (FS_static - 1). Slopes already unstable under
    static conditions return a negative value, which is the honest answer: no
    shaking is required.
    """
    theta = np.arctan(slope)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    f = depth_factor(slope, depth_k)
    w = wetness(sca, slope, rt * f)
    tan_phi = np.tan(np.radians(friction_deg))
    numer = cohesion * f + cos_t * (1.0 - w * density_ratio) * tan_phi - sin_t
    return numer / (cos_t + sin_t * tan_phi)


def failure_probability(slope: np.ndarray, sca: np.ndarray,
                        params: SoilParameters, n_samples: int = 200,
                        seed: int = 0,
                        density_ratio: float = DENSITY_RATIO,
                        recharge_scale: Optional[np.ndarray] = None,
                        k_h=0.0) -> np.ndarray:
    """Probability that FS < 1, over the uniform parameter ranges.

    Parameter triples are drawn once and applied to every pixel, which is what
    the marginal per-pixel probability requires and keeps the cost to
    ``n_samples`` passes over the block rather than a draw per pixel.

    ``recharge_scale`` multiplies R/T per pixel, carrying spatial variation in
    recharge (wetter places drain more water through the same soil). It is
    dimensionless and centred on 1 at the reference climate.
    """
    rng = np.random.default_rng(seed)
    c = rng.uniform(*params.cohesion, n_samples)
    phi = rng.uniform(*params.friction_deg, n_samples)
    rt = rng.uniform(*params.rt, n_samples)

    valid = np.isfinite(slope) & np.isfinite(sca)
    fails = np.zeros(np.shape(slope), dtype="float64")
    for i in range(n_samples):
        rt_i = rt[i] if recharge_scale is None else rt[i] * recharge_scale
        fs = factor_of_safety(slope, sca, c[i], phi[i], rt_i, density_ratio,
                              k_h=k_h, depth_k=params.depth_k)
        fails += (fs < 1.0)
    p = fails / n_samples
    return np.where(valid, p, np.nan)


def stability_classes(slope: np.ndarray, sca: np.ndarray,
                      params: SoilParameters,
                      density_ratio: float = DENSITY_RATIO,
                      recharge_scale: Optional[np.ndarray] = None,
                      k_h=0.0) -> np.ndarray:
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
    scale = 1.0 if recharge_scale is None else recharge_scale

    # Worst case: least cohesion, least friction, wettest.
    fs_worst = factor_of_safety(slope, sca, c_lo, phi_lo, rt_hi * scale,
                                density_ratio, k_h=k_h,
                                depth_k=params.depth_k)
    # Best case, and dry, so wetness plays no part.
    fs_best_dry = factor_of_safety(slope, np.zeros_like(sca), c_hi, phi_hi,
                                   rt_lo, density_ratio, k_h=k_h,
                                   depth_k=params.depth_k)
    p = failure_probability(slope, sca, params, n_samples=100,
                            density_ratio=density_ratio,
                            recharge_scale=recharge_scale, k_h=k_h)

    cls = np.full(np.shape(slope), np.nan)
    cls = np.where(p >= 0.5, 5.0, cls)
    cls = np.where((p > 0) & (p < 0.5), 4.0, cls)
    cls = np.where(p <= 0, 3.0, cls)
    cls = np.where(fs_worst > 1.25, 2.0, cls)
    cls = np.where(np.isfinite(fs_worst) & (fs_worst > 1.5), 1.0, cls)
    cls = np.where(fs_best_dry < 1.0, 6.0, cls)
    return np.where(np.isfinite(slope), cls, np.nan)


# ---------------------------------------------------------------------------
# calibration regions
# ---------------------------------------------------------------------------
#
# SINMAP's own answer to spatially varying soils is the "calibration region": a
# zoning of the map within which the parameter ranges are taken as uniform, and
# between which they may differ. Lithology controls friction and soil cohesion;
# land cover controls root cohesion. Both are available as open rasters, so
# either can serve as the zoning.

def failure_probability_regional(slope: np.ndarray, sca: np.ndarray,
                                 region: np.ndarray,
                                 params_by_region: Dict[int, SoilParameters],
                                 fallback: SoilParameters,
                                 n_samples: int = 200, seed: int = 0,
                                 density_ratio: float = DENSITY_RATIO,
                                 recharge_scale: Optional[np.ndarray] = None,
                                 k_h=0.0) -> np.ndarray:
    """Failure probability with per-region soil parameters.

    Regions are evaluated independently; a region with no fitted parameters
    falls back to the supplied ranges.
    """
    out = np.full(np.shape(slope), np.nan, dtype="float64")
    codes = np.unique(region[np.isfinite(region)])
    for code in codes:
        m = region == code
        if not m.any():
            continue
        p = params_by_region.get(int(code), fallback)
        scale = None if recharge_scale is None else np.asarray(recharge_scale)[m]
        out[m] = failure_probability(slope[m], sca[m], p, n_samples=n_samples,
                                     seed=seed, density_ratio=density_ratio,
                                     recharge_scale=scale, k_h=k_h)
    return out


# ---------------------------------------------------------------------------
# fitting the parameter ranges to an inventory
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


#: Depth-decline candidates for the augmented fit grid, in units of
#: 1/tan(theta). k = 1 roughly halves the soil column between flat ground
#: and 35 degrees; 2.5 thins it near six-fold - together bracketing the
#: DeRose-type decline rates reported for steep soil-mantled terrain. Zero
#: is always searched alongside them, so the augmented grid is free to
#: reject the term by choosing the uniform-depth model.
DEPTH_K_CANDIDATES = (1.0, 2.5)


def parameter_grid(depth_candidates: Sequence[float] = (0.0,)
                   ) -> Sequence[SoilParameters]:
    """Candidate parameter ranges spanning soil-mantled mountain hillslopes.

    Cohesion runs from bare (0) up to a well-rooted forest soil. Friction
    brackets the 25-45 degree band reported for colluvium and weathered
    regolith. R/T spans four orders of magnitude, which is the range over which
    published transmissivities and monsoon recharge rates place it. The lower
    end of each R/T range is set to a fiftieth of the upper, keeping the ratio
    of the range width to its level constant.
    """
    grid = []
    for depth_k in depth_candidates:
        for c_hi in (0.05, 0.15, 0.25, 0.40):
            for phi_lo, phi_hi in ((25.0, 35.0), (30.0, 40.0), (35.0, 45.0)):
                for rt_hi in (0.0005, 0.002, 0.01, 0.05):
                    grid.append(SoilParameters((0.0, c_hi), (phi_lo, phi_hi),
                                               (rt_hi / 50.0, rt_hi),
                                               depth_k=depth_k))
    return grid


def _search(slope: np.ndarray, sca: np.ndarray, y: np.ndarray,
            n_samples: int, seed: int,
            recharge_scale: Optional[np.ndarray] = None,
            grid: Optional[Sequence[SoilParameters]] = None
            ) -> Tuple[Optional[SoilParameters], float, list]:
    best, best_auc, trials = None, -np.inf, []
    for params in (parameter_grid() if grid is None else grid):
        p = failure_probability(slope, sca, params, n_samples=n_samples,
                                seed=seed, recharge_scale=recharge_scale)
        ok = np.isfinite(p)
        if ok.sum() < 20:
            continue
        auc = _auc(p[ok], y[ok])
        if np.isfinite(auc):
            trials.append({"params": params.as_dict(), "auc": round(auc, 4)})
            if auc > best_auc:
                best, best_auc = params, auc
    trials.sort(key=lambda t: -t["auc"])
    return best, float(best_auc), trials


def fit_parameters(slope_pres: np.ndarray, sca_pres: np.ndarray,
                   slope_bg: np.ndarray, sca_bg: np.ndarray,
                   n_samples: int = 60, seed: int = 0,
                   recharge_pres: Optional[np.ndarray] = None,
                   recharge_bg: Optional[np.ndarray] = None,
                   grid: Optional[Sequence[SoilParameters]] = None) -> dict:
    """Search soil parameter ranges that best separate landslides from terrain.

    ``grid`` overrides the candidate set - pass
    ``parameter_grid((0.0,) + DEPTH_K_CANDIDATES)`` to let the search
    consider slope-dependent soil depth. Do that in the product configs only
    on the strength of analysis/09_soil_depth.py.

    The physics fixes the *form* of the response; what a region's soils and
    hydrology supply are the parameter values. Those are searched over ranges
    reported for soil-mantled mountain terrain, scored by how well the
    resulting failure probability ranks mapped landslides above background.

    The score here is in-sample: every candidate sees every point. Use
    :func:`cross_validate` for a figure that means something about new ground.
    """
    sp, ap, sb, ab = (np.asarray(x, "float64") for x in
                      (slope_pres, sca_pres, slope_bg, sca_bg))
    rp = None if recharge_pres is None else np.asarray(recharge_pres, "float64")
    rb = None if recharge_bg is None else np.asarray(recharge_bg, "float64")

    ok_p = np.isfinite(sp) & np.isfinite(ap)
    ok_b = np.isfinite(sb) & np.isfinite(ab)
    if rp is not None:
        ok_p &= np.isfinite(rp)
        ok_b &= np.isfinite(rb)
    sp, ap, sb, ab = sp[ok_p], ap[ok_p], sb[ok_b], ab[ok_b]
    if len(sp) < 20 or len(sb) < 20:
        raise ValueError("too few valid samples to fit soil parameters")

    slope = np.concatenate([sp, sb])
    sca = np.concatenate([ap, ab])
    y = np.concatenate([np.ones(len(sp)), np.zeros(len(sb))])
    scale = None if rp is None else np.concatenate([rp[ok_p], rb[ok_b]])

    best, best_auc, trials = _search(slope, sca, y, n_samples, seed, scale,
                                     grid=grid)
    if best is None:
        raise ValueError("no parameter set produced a usable score")
    return {"parameters": best, "auc": best_auc,
            "n_presence": int(len(sp)), "n_background": int(len(sb)),
            "top_trials": trials[:5], "n_trials": len(trials)}


def fit_parameters_regional(slope_pres, sca_pres, region_pres,
                            slope_bg, sca_bg, region_bg,
                            n_samples: int = 60, seed: int = 0,
                            min_presence: int = 100) -> dict:
    """Fit a separate parameter range per calibration region.

    A region is fitted only if it holds enough landslides to constrain three
    parameters; sparser regions keep the whole-area fit, which is what
    ``fallback`` carries. Splitting the data buys spatial detail at the cost of
    sample size per fit, and below roughly a hundred failures the second cost
    dominates.
    """
    whole = fit_parameters(slope_pres, sca_pres, slope_bg, sca_bg,
                           n_samples=n_samples, seed=seed)
    rp = np.asarray(region_pres)
    rb = np.asarray(region_bg)

    by_region: Dict[int, SoilParameters] = {}
    detail = {}
    for code in np.unique(rp[np.isfinite(rp)]):
        code = int(code)
        mp, mb = (rp == code), (rb == code)
        if mp.sum() < min_presence or mb.sum() < 20:
            continue
        try:
            r = fit_parameters(np.asarray(slope_pres)[mp],
                               np.asarray(sca_pres)[mp],
                               np.asarray(slope_bg)[mb],
                               np.asarray(sca_bg)[mb],
                               n_samples=n_samples, seed=seed)
        except ValueError:
            continue
        by_region[code] = r["parameters"]
        detail[str(code)] = {"auc": round(r["auc"], 4),
                             "n_presence": r["n_presence"],
                             "parameters": r["parameters"].as_dict()}

    return {"fallback": whole["parameters"], "by_region": by_region,
            "fallback_auc": whole["auc"], "regions": detail,
            "n_regions_fitted": len(by_region)}


def cross_validate(points_pres: np.ndarray, slope_pres: np.ndarray,
                   sca_pres: np.ndarray, points_bg: np.ndarray,
                   slope_bg: np.ndarray, sca_bg: np.ndarray,
                   bbox: Sequence[float], scheme: str = "spatial",
                   n_folds: int = 5, block_deg: float = 0.25,
                   n_samples: int = 40, seed: int = 0,
                   recharge_pres: Optional[np.ndarray] = None,
                   recharge_bg: Optional[np.ndarray] = None,
                   grid: Optional[Sequence[SoilParameters]] = None) -> dict:
    """Fit the parameters fold by fold and score on the withheld fold.

    The parameter search is rerun inside each fold, so the reported AUC is not
    contaminated by having chosen the parameters on the test points. Under the
    spatial scheme whole blocks are withheld, which is the figure that says
    what to expect on ground the fit has not seen.

    The recharge field, if one is in use, must be passed here too - otherwise
    the cross-validated figure describes a different model from the one that
    produces the map.
    """
    if scheme == "spatial":
        fp = crossval.spatial_block_folds(points_pres, bbox, n_folds,
                                          block_deg, seed)
        fb = crossval.spatial_block_folds(points_bg, bbox, n_folds,
                                          block_deg, seed)
    elif scheme == "random":
        fp = crossval.random_folds(len(points_pres), n_folds, seed)
        fb = crossval.random_folds(len(points_bg), n_folds, seed)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    sp, ap = np.asarray(slope_pres, "float64"), np.asarray(sca_pres, "float64")
    sb, ab = np.asarray(slope_bg, "float64"), np.asarray(sca_bg, "float64")
    rp = None if recharge_pres is None else np.asarray(recharge_pres, "float64")
    rb = None if recharge_bg is None else np.asarray(recharge_bg, "float64")

    aucs, sizes, chosen = [], [], []
    for k in range(n_folds):
        tr_p, tr_b = (fp != k), (fb != k)
        te_p, te_b = (fp == k), (fb == k)
        if tr_p.sum() < 50 or tr_b.sum() < 50 or te_p.sum() < 10 \
                or te_b.sum() < 10:
            continue
        try:
            fit = fit_parameters(
                sp[tr_p], ap[tr_p], sb[tr_b], ab[tr_b], n_samples=n_samples,
                seed=seed,
                recharge_pres=None if rp is None else rp[tr_p],
                recharge_bg=None if rb is None else rb[tr_b], grid=grid)
        except ValueError:
            continue
        params = fit["parameters"]

        slope_te = np.concatenate([sp[te_p], sb[te_b]])
        sca_te = np.concatenate([ap[te_p], ab[te_b]])
        y_te = np.concatenate([np.ones(int(te_p.sum())),
                               np.zeros(int(te_b.sum()))])
        scale_te = (None if rp is None
                    else np.concatenate([rp[te_p], rb[te_b]]))
        p = failure_probability(slope_te, sca_te, params,
                                n_samples=n_samples, seed=seed,
                                recharge_scale=scale_te)
        ok = np.isfinite(p)
        auc = _auc(p[ok], y_te[ok])
        if np.isfinite(auc):
            aucs.append(float(auc))
            sizes.append((int(te_p.sum()), int(te_b.sum())))
            chosen.append(params.as_dict())

    return {
        "scheme": scheme,
        "n_folds_scored": len(aucs),
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "auc_folds": [round(a, 4) for a in aucs],
        "test_sizes": sizes,
        "fold_parameters": chosen,
        "block_deg": block_deg if scheme == "spatial" else None,
    }
