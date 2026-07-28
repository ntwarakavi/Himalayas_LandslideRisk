"""Does the inventory's triggering mechanism cap what the model can explain?

The Gorkha inventory is earthquake-triggered. Where those landslides actually
happened depended on distance to the rupture, directivity and topographic
amplification of the shaking - none of which appears in a static stability map.
So scoring a static map against an earthquake inventory measures the model
against a target it was never asked to hit, and some of the unexplained
variance belongs to the earthquake rather than to the model.

Far-Western Nepal is a multi-temporal, monsoon-driven inventory. Its triggering
mechanism is the one the wetness term actually represents. If mechanism
mismatch is a real constraint, the same model fitted and cross-validated there
should score higher than on Gorkha.

This is not a like-for-like comparison of two areas - the terrain and the
mapping differ too - so the number to read is the direction and size of the
gap, not a ranking of the two catchments.

    python analysis/04_monsoon.py
"""

from __future__ import annotations

import numpy as np

import common as K

RES = 0.00027778          # 30 m
AREAS = ["gorkha", "farwest"]
MECHANISM = {"gorkha": "earthquake (2015 Mw 7.8 Gorkha)",
             "farwest": "monsoon rainfall, multi-temporal 1992-2018"}


def main() -> None:
    rows = []
    for area in AREAS:
        cfg = K.make_config(area, RES)
        print(f"\n=== {area}: {MECHANISM[area]} ===")
        print(f"    {cfg.cell_count():,} cells at 30 m")

        layers = K.terrain_layers(cfg, want_precip=True)
        paths = [layers["slope"], layers["sca"], layers["recharge"]]
        pres, vp, bg, vb = K.sample(cfg, paths)
        ok_p, ok_b = K.clean(vp), K.clean(vb)
        pres, vp, bg, vb = pres[ok_p], vp[ok_p], bg[ok_b], vb[ok_b]
        print(f"    {len(pres)} landslides, {len(bg)} background")

        fit = K.P.fit_parameters(vp[:, 0], vp[:, 1], vb[:, 0], vb[:, 1],
                                 n_samples=60, recharge_pres=vp[:, 2],
                                 recharge_bg=vb[:, 2])
        row = {"area": area, "mechanism": MECHANISM[area],
               "n_presence": int(len(pres)), "n_background": int(len(bg)),
               "in_sample_auc": round(fit["auc"], 4),
               "parameters": fit["parameters"].as_dict()}
        print(f"    fitted {fit['parameters'].as_dict()}")

        for scheme in ("random", "spatial"):
            cv = K.P.cross_validate(
                pres, vp[:, 0], vp[:, 1], bg, vb[:, 0], vb[:, 1],
                cfg.clipped_bbox(), scheme=scheme, n_folds=5,
                block_deg=0.25, n_samples=60,
                recharge_pres=vp[:, 2], recharge_bg=vb[:, 2])
            row[f"cv_{scheme}_mean"] = round(cv["auc_mean"], 4)
            row[f"cv_{scheme}_std"] = round(cv["auc_std"], 4)
            row[f"cv_{scheme}_folds"] = cv["auc_folds"]
            print(f"    {scheme:8s} CV AUC {cv['auc_mean']:.4f} "
                  f"+/- {cv['auc_std']:.4f}")

        p = K.P.failure_probability(
            np.concatenate([vp[:, 0], vb[:, 0]]),
            np.concatenate([vp[:, 1], vb[:, 1]]), fit["parameters"],
            n_samples=200,
            recharge_scale=np.concatenate([vp[:, 2], vb[:, 2]]))
        y = np.concatenate([np.ones(len(vp)), np.zeros(len(vb))])
        ok = np.isfinite(p)
        row.update(K.capture(p[ok], y[ok]))
        rows.append(row)

    print("\n\nTRIGGERING MECHANISM AND SKILL  (both at 30 m)\n")
    print(K.table(rows, [
        ("area", "area", "s"),
        ("n_presence", "slides", "d"),
        ("in_sample_auc", "in-samp", ".4f"),
        ("cv_random_mean", "randomCV", ".4f"),
        ("cv_spatial_mean", "spatialCV", ".4f"),
        ("cv_spatial_std", "+/-", ".4f"),
        ("capture_top20pct", "top20%", ".1f"),
    ]))
    for r in rows:
        print(f"\n  {r['area']:8s} {r['mechanism']}")
        print(f"           phi {r['parameters']['friction_deg']}  "
              f"C {r['parameters']['cohesion']}  "
              f"R/T {r['parameters']['rt']}")

    K.save("04_monsoon", {"areas": rows, "resolution_deg": RES})


if __name__ == "__main__":
    main()
