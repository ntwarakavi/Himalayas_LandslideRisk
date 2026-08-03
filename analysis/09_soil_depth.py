"""Does slope-dependent soil depth earn its place, on held-out ground?

Soil depth is the model's largest structural gap on the soil side: cohesion is
identifiable only as (Cr + Cs) / (h rho g) and h is treated as uniform
(README, limit 5). Regolith is not uniform - it thins as slopes steepen,
measured as an exponential decline with gradient in steep soil-mantled
terrain (DeRose 1996) and applied as h = h0 exp(-k tan theta) for distributed
effective depth (Saulnier et al. 1997; cf. Catani et al. 2010). Both the
dimensionless cohesion and R/T carry 1/h, so one per-pixel factor scales the
pair together; the mechanics live in physical.depth_factor and the depth_k
field of SoilParameters.

Physically motivated is not measurably better: per-lithology calibration was
equally plausible and scored -0.0004 held-out (06_calibration_regions), which
is why depth ships off (depth_k = 0, the published SINMAP) and why this
script exists. It answers one question the way 06 did: with the presence
points, background points and spatial-block folds held identical, does
letting the fit choose a depth-decline rate improve the AUC on withheld
blocks?

Per fold, cross_validate runs twice on the same folds: once with the standard
48-candidate grid, once with the grid augmented by DEPTH_K_CANDIDATES (the
augmented grid always contains k = 0 too, so it can reject the term by
choosing it). Reported alongside the AUCs: the depth_k each fold's augmented
search actually chose, because "it picked zero every time" is a verdict
before any AUC is read.

Decision rule, same as ever: adopt - fit the product with
``parameter_grid((0.0,) + DEPTH_K_CANDIDATES)`` - only if the augmented AUC
beats the baseline by more than the fold spread in both areas. Otherwise the
default stays and this file is the record of why.

    python analysis/09_soil_depth.py
"""

from __future__ import annotations

import numpy as np

import common as K

from h_sim.model import physical as P

RES = 0.00027778          # 30 m, the resolution the headline fit uses
AREAS = ["gorkha", "farwest"]
N_FOLDS = 5
BLOCK_DEG = 0.25
N_SAMPLES = 60


def main() -> None:
    base_grid = P.parameter_grid()
    depth_grid = P.parameter_grid((0.0,) + P.DEPTH_K_CANDIDATES)
    print(f"[09] grids: {len(base_grid)} uniform-depth candidates vs "
          f"{len(depth_grid)} with depth_k in {(0.0,) + P.DEPTH_K_CANDIDATES}")

    rows = []
    for area in AREAS:
        cfg = K.make_config(area, RES)
        bbox = cfg.clipped_bbox()
        print(f"\n=== {area} ===")

        layers = K.terrain_layers(cfg, want_precip=True)
        paths = [layers["slope"], layers["sca"], layers["recharge"]]
        pres, vp, bg, vb = K.sample(cfg, paths, area=area)
        ok_p, ok_b = K.clean(vp), K.clean(vb)
        pres, vp, bg, vb = pres[ok_p], vp[ok_p], bg[ok_b], vb[ok_b]
        print(f"  {len(pres)} landslides, {len(bg)} background")

        results = {}
        for name, grid in (("uniform depth", base_grid),
                           ("slope-dependent", depth_grid)):
            cv = P.cross_validate(
                pres, vp[:, 0], vp[:, 1], bg, vb[:, 0], vb[:, 1],
                bbox, scheme="spatial", n_folds=N_FOLDS,
                block_deg=BLOCK_DEG, n_samples=N_SAMPLES,
                recharge_pres=vp[:, 2], recharge_bg=vb[:, 2], grid=grid)
            results[name] = cv
            r = {"model": name, "area": area,
                 "auc_mean": round(cv["auc_mean"], 4),
                 "auc_std": round(cv["auc_std"], 4),
                 "auc_folds": cv["auc_folds"],
                 "n_folds": cv["n_folds_scored"]}
            if name == "slope-dependent":
                ks = [c.get("depth_k", 0.0)
                      for c in cv.get("fold_parameters", [])]
                r["chosen_depth_k"] = ks
                print(f"  {name}: {cv['auc_mean']:.4f} +/- "
                      f"{cv['auc_std']:.4f}  (chose depth_k {ks})")
            else:
                print(f"  {name}: {cv['auc_mean']:.4f} +/- "
                      f"{cv['auc_std']:.4f}")
            rows.append(r)

    print("\n\nSLOPE-DEPENDENT SOIL DEPTH, SPATIAL-BLOCK CV  (30 m)\n")
    print(K.table(rows, [
        ("area", "area", "s"),
        ("model", "model", "s"),
        ("auc_mean", "AUC", ".4f"),
        ("auc_std", "+/-", ".4f"),
    ]))

    verdicts = []
    for area in AREAS:
        a = {r["model"]: r for r in rows if r["area"] == area}
        if len(a) < 2:
            continue
        gain = a["slope-dependent"]["auc_mean"] - a["uniform depth"]["auc_mean"]
        noise = max(a["slope-dependent"]["auc_std"],
                    a["uniform depth"]["auc_std"])
        ks = a["slope-dependent"].get("chosen_depth_k", [])
        adopt = gain > noise and any(k > 0 for k in ks)
        verdicts.append(adopt)
        print(f"[09] {area}: gain {gain:+.4f} vs fold spread {noise:.4f}; "
              f"chosen depth_k per fold {ks}")

    adopt = bool(verdicts) and all(verdicts)
    print("[09] " + ("ADOPT - fit the product with "
                     "parameter_grid((0.0,) + DEPTH_K_CANDIDATES)" if adopt
                     else "NULL - the depth term stays off (depth_k = 0)"))

    K.save("09_soil_depth", {
        "resolution_deg": RES,
        "depth_candidates": list(P.DEPTH_K_CANDIDATES),
        "rows": rows,
        "adopt": adopt,
    })


if __name__ == "__main__":
    main()
