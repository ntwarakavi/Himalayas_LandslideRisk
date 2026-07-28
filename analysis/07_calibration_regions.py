"""Do per-lithology soil parameters ever earn their keep?

SINMAP's calibration regions let each rock type carry its own cohesion,
friction angle and R/T. On the Gorkha area that found nothing, but Gorkha is
97 per cent metamorphics - there was no variation for the zoning to exploit.
Far-Western Nepal is the opposite case: its own inventory attributes name a
dozen formations, from Siwalik sandstones to basic rocks, each with several
thousand mapped failures.

So the honest test is here, and it must be held out. Fitting a separate
parameter set per region always improves the in-sample score - more parameters
always do - so the comparison below refits both the whole-area and the
per-region parameters *inside each spatial fold* and scores them on the fold
withheld. If the zoning is capturing real mechanical differences it will win
there; if it is fitting noise, it will not.

    python analysis/07_calibration_regions.py
"""

from __future__ import annotations

import numpy as np

import common as K

RES = 0.00027778          # 30 m
AREAS = ["farwest", "gorkha"]
N_FOLDS = 5
BLOCK_DEG = 0.25
MIN_PRESENCE = 100


def main() -> None:
    rows = []
    for area in AREAS:
        cfg = K.make_config(area, RES)
        bbox = cfg.clipped_bbox()
        print(f"\n=== {area} ===")

        layers = K.terrain_layers(cfg, want_precip=True)
        litho = K.region_layer(area, RES, "lithology")
        if litho is None:
            print("  no lithology raster; skipping")
            continue

        paths = [layers["slope"], layers["sca"], layers["recharge"], litho]
        pres, vp, bg, vb = K.sample(cfg, paths, area=area)
        ok_p, ok_b = K.clean(vp[:, :3]), K.clean(vb[:, :3])
        pres, vp, bg, vb = pres[ok_p], vp[ok_p], bg[ok_b], vb[ok_b]

        codes = np.concatenate([vp[:, 3], vb[:, 3]])
        present = {int(c): int((vp[:, 3] == c).sum())
                   for c in np.unique(codes[np.isfinite(codes)])}
        print(f"  {len(pres)} landslides, {len(bg)} background")
        print(f"  lithology codes (landslides each): {present}")

        fp, fb = K.folds(pres, bg, bbox, "spatial", N_FOLDS, BLOCK_DEG)
        whole, regional, n_fitted = [], [], []

        for k in range(N_FOLDS):
            tr_p, tr_b = fp != k, fb != k
            te_p, te_b = fp == k, fb == k
            if min(tr_p.sum(), tr_b.sum()) < 50 or \
                    min(te_p.sum(), te_b.sum()) < 10:
                continue

            s_te = np.concatenate([vp[te_p, 0], vb[te_b, 0]])
            a_te = np.concatenate([vp[te_p, 1], vb[te_b, 1]])
            r_te = np.concatenate([vp[te_p, 2], vb[te_b, 2]])
            g_te = np.concatenate([vp[te_p, 3], vb[te_b, 3]])
            y_te = np.concatenate([np.ones(int(te_p.sum())),
                                   np.zeros(int(te_b.sum()))])

            reg = K.P.fit_parameters_regional(
                vp[tr_p, 0], vp[tr_p, 1], vp[tr_p, 3],
                vb[tr_b, 0], vb[tr_b, 1], vb[tr_b, 3],
                n_samples=60, min_presence=MIN_PRESENCE)
            n_fitted.append(reg["n_regions_fitted"])

            p_w = K.P.failure_probability(s_te, a_te, reg["fallback"],
                                          n_samples=200, recharge_scale=r_te)
            p_r = K.P.failure_probability_regional(
                s_te, a_te, g_te, reg["by_region"], reg["fallback"],
                n_samples=200, recharge_scale=r_te)
            for store, p in ((whole, p_w), (regional, p_r)):
                ok = np.isfinite(p)
                store.append(K.auc(p[ok], y_te[ok]))
            print(f"  fold {k}: whole {whole[-1]:.4f}  "
                  f"regional {regional[-1]:.4f}  "
                  f"({reg['n_regions_fitted']} regions fitted)")

        for name, aucs in (("whole area", whole), ("per lithology", regional)):
            r = K.summarise(name, aucs)
            r["area"] = area
            r["mean_regions_fitted"] = round(float(np.mean(n_fitted)), 1)
            r["n_lithologies_present"] = len(present)
            rows.append(r)

    print("\n\nCALIBRATION REGIONS, SPATIAL-BLOCK CV  (30 m)\n")
    print(K.table(rows, [
        ("area", "area", "s"),
        ("model", "parameters", "s"),
        ("n_lithologies_present", "litho", "d"),
        ("mean_regions_fitted", "fitted", ".1f"),
        ("auc_mean", "AUC", ".4f"),
        ("auc_std", "+/-", ".4f"),
    ]))

    for area in AREAS:
        a = [r for r in rows if r["area"] == area]
        if len(a) == 2:
            d = a[1]["auc_mean"] - a[0]["auc_mean"]
            print(f"\n  {area}: zoning changes held-out AUC by {d:+.4f} "
                  f"(fold spread +/-{a[0]['auc_std']:.4f})")

    K.save("07_calibration_regions", {"rows": rows, "resolution_deg": RES,
                                      "min_presence": MIN_PRESENCE})


if __name__ == "__main__":
    main()
