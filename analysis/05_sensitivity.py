"""How much do the two unfitted conventions actually decide?

Everything else in the model is either mechanics or fitted to data. Two numbers
are neither:

* ``rainfall_cv`` - the coefficient of variation of annual maximum daily
  rainfall, which sets how much a rarer storm raises recharge. Station analyses
  put monsoon Asia at roughly 0.25-0.35.
* ``pga_fraction`` - the fraction of peak ground acceleration applied as a
  sustained pseudo-static force. One half is the long-standing convention;
  values from 0.3 to 1.0 appear in practice.

A reader is entitled to know what rests on them. Two things are reported, and
they behave very differently:

* the **ranking** of terrain within a scenario map, measured as the AUC against
  the inventory and as the rank correlation with the default map. If this is
  insensitive, the pattern a user reads off the map does not depend on the
  convention.
* the **level**, measured as the area above a failure-probability threshold. If
  this moves, absolute statements about a scenario do depend on the convention,
  and should be quoted as a range.

    python analysis/05_sensitivity.py
"""

from __future__ import annotations

import numpy as np

import common as K
from giri_landslide.model import hazard as HZ

RES = 0.00027778          # 30 m
RAINFALL_CV = [0.20, 0.25, 0.30, 0.35, 0.40]
PGA_FRACTION = [0.3, 0.5, 0.7, 1.0]
RETURN_PERIODS = [10.0, 100.0, 1000.0]
PGA_G = 0.35


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed without scipy."""
    ra = np.argsort(np.argsort(a)).astype("float64")
    rb = np.argsort(np.argsort(b)).astype("float64")
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def main() -> None:
    cfg = K.make_config("gorkha", RES)
    print(f"Sensitivity on Gorkha at 30 m ({cfg.cell_count():,} cells)\n")

    layers = K.terrain_layers(cfg, want_precip=True)
    paths = [layers["slope"], layers["sca"], layers["recharge"]]
    pres, vp, bg, vb = K.sample(cfg, paths, area="gorkha")
    ok_p, ok_b = K.clean(vp), K.clean(vb)
    vp, vb = vp[ok_p], vb[ok_b]

    fit = K.P.fit_parameters(vp[:, 0], vp[:, 1], vb[:, 0], vb[:, 1],
                             n_samples=60, recharge_pres=vp[:, 2],
                             recharge_bg=vb[:, 2])
    params = fit["parameters"]
    print(f"  parameters {params.as_dict()}")

    slope = np.concatenate([vp[:, 0], vb[:, 0]])
    sca = np.concatenate([vp[:, 1], vb[:, 1]])
    scale = np.concatenate([vp[:, 2], vb[:, 2]])
    y = np.concatenate([np.ones(len(vp)), np.zeros(len(vb))])

    def evaluate(mult: float, k_h: float) -> np.ndarray:
        return K.P.failure_probability(slope, sca, params, n_samples=200,
                                       recharge_scale=scale * mult, k_h=k_h)

    # ---- rainfall -------------------------------------------------------
    rain_rows = []
    ref = {}
    for T in RETURN_PERIODS:
        ref[T] = evaluate(HZ.recharge_multiplier(T, cv=0.30), 0.0)
    for cv in RAINFALL_CV:
        for T in RETURN_PERIODS:
            m = HZ.recharge_multiplier(T, cv=cv)
            p = evaluate(m, 0.0)
            ok = np.isfinite(p) & np.isfinite(ref[T])
            rain_rows.append({
                "convention": "rainfall_cv", "value": cv,
                "return_period_yr": T, "multiplier": round(m, 3),
                "auc": round(K.auc(p[ok], y[ok]), 4),
                "rank_corr_vs_default": round(spearman(p[ok], ref[T][ok]), 4),
                "pct_above_0.5": round(float((p[ok] > 0.5).mean() * 100), 2),
                "mean_probability": round(float(p[ok].mean()), 4),
            })
        print(f"  rainfall_cv {cv:.2f} done")

    # ---- earthquake -----------------------------------------------------
    quake_rows = []
    ref_q = evaluate(1.0, HZ.seismic_coefficient(PGA_G, 0.5))
    for frac in PGA_FRACTION:
        k = HZ.seismic_coefficient(PGA_G, frac)
        p = evaluate(1.0, k)
        ok = np.isfinite(p) & np.isfinite(ref_q)
        quake_rows.append({
            "convention": "pga_fraction", "value": frac,
            "pga_g": PGA_G, "k_h": round(k, 4),
            "auc": round(K.auc(p[ok], y[ok]), 4),
            "rank_corr_vs_default": round(spearman(p[ok], ref_q[ok]), 4),
            "pct_above_0.5": round(float((p[ok] > 0.5).mean() * 100), 2),
            "mean_probability": round(float(p[ok].mean()), 4),
        })
        print(f"  pga_fraction {frac:.1f} done")

    print("\n\nRAINFALL: coefficient of variation of annual maximum daily rain\n")
    for T in RETURN_PERIODS:
        print(f"  {T:g}-year storm")
        print(K.table([r for r in rain_rows if r["return_period_yr"] == T], [
            ("value", "cv", ".2f"),
            ("multiplier", "R/T x", ".3f"),
            ("auc", "AUC", ".4f"),
            ("rank_corr_vs_default", "rank rho", ".4f"),
            ("pct_above_0.5", "% P>0.5", ".2f"),
            ("mean_probability", "mean P", ".4f"),
        ]))
        print()

    print(f"\nEARTHQUAKE: pseudo-static fraction of PGA (at {PGA_G} g)\n")
    print(K.table(quake_rows, [
        ("value", "fraction", ".1f"),
        ("k_h", "k_h", ".4f"),
        ("auc", "AUC", ".4f"),
        ("rank_corr_vs_default", "rank rho", ".4f"),
        ("pct_above_0.5", "% P>0.5", ".2f"),
        ("mean_probability", "mean P", ".4f"),
    ]))

    K.save("05_sensitivity", {"rainfall": rain_rows, "earthquake": quake_rows,
                              "parameters": params.as_dict(),
                              "resolution_deg": RES})


if __name__ == "__main__":
    main()
