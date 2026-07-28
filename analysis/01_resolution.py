"""Does grid resolution matter, and if so why?

The physical argument for a SINMAP-type model over a slope-and-lithology index
is flow convergence: wetness distinguishes a hollow from a planar hillside at
the same gradient. Hollows are tens of metres across. A 250 m grid smooths them
away, so if convergence is really the mechanism, skill should rise as the grid
gets finer - and specific catchment area should become more variable.

This fits and cross-validates the same inventory over the same area at three
grids, holding everything else fixed.

    python analysis/01_resolution.py
"""

from __future__ import annotations

import numpy as np

import common as K


GRIDS = [
    (0.0025,      "250 m"),
    (0.00083333,  "90 m"),
    (0.00027778,  "30 m"),
]


def run_one(res: float, label: str) -> dict:
    cfg = K.make_config("gorkha", res)
    print(f"\n=== {label}  ({cfg.cell_count():,} cells) ===")

    layers = K.terrain_layers(cfg, want_precip=True)
    paths = [layers["slope"], layers["sca"], layers["recharge"]]
    pres, vp, bg, vb = K.sample(cfg, paths)

    ok_p, ok_b = K.clean(vp), K.clean(vb)
    pres, vp, bg, vb = pres[ok_p], vp[ok_p], bg[ok_b], vb[ok_b]
    print(f"  {len(pres)} landslides, {len(bg)} background")

    # How much structure does the wetness term actually have at this grid?
    sca_all = np.concatenate([vp[:, 1], vb[:, 1]])
    sca_iqr = float(np.percentile(sca_all, 75) - np.percentile(sca_all, 25))

    fit = P_fit(vp, vb)
    row = {
        "resolution_deg": res, "label": label,
        "cells": cfg.cell_count(),
        "n_presence": len(pres), "n_background": len(bg),
        "sca_median_m": round(float(np.median(sca_all)), 1),
        "sca_iqr_m": round(sca_iqr, 1),
        "sca_p99_m": round(float(np.percentile(sca_all, 99)), 1),
        "in_sample_auc": round(fit["auc"], 4),
        "parameters": fit["parameters"].as_dict(),
    }

    for scheme in ("random", "spatial"):
        cv = K.P.cross_validate(
            pres, vp[:, 0], vp[:, 1], bg, vb[:, 0], vb[:, 1],
            cfg.clipped_bbox(), scheme=scheme, n_folds=5, block_deg=0.25,
            n_samples=60, recharge_pres=vp[:, 2], recharge_bg=vb[:, 2])
        row[f"cv_{scheme}_mean"] = round(cv["auc_mean"], 4)
        row[f"cv_{scheme}_std"] = round(cv["auc_std"], 4)
        row[f"cv_{scheme}_folds"] = cv["auc_folds"]
        print(f"  {scheme:8s} CV AUC {cv['auc_mean']:.4f} "
              f"+/- {cv['auc_std']:.4f}")

    # Capture, in-sample, at the fitted parameters.
    p = K.P.failure_probability(
        np.concatenate([vp[:, 0], vb[:, 0]]),
        np.concatenate([vp[:, 1], vb[:, 1]]), fit["parameters"],
        n_samples=200,
        recharge_scale=np.concatenate([vp[:, 2], vb[:, 2]]))
    y = np.concatenate([np.ones(len(vp)), np.zeros(len(vb))])
    ok = np.isfinite(p)
    row.update(K.capture(p[ok], y[ok]))
    return row


def P_fit(vp, vb):
    return K.P.fit_parameters(vp[:, 0], vp[:, 1], vb[:, 0], vb[:, 1],
                              n_samples=60,
                              recharge_pres=vp[:, 2], recharge_bg=vb[:, 2])


def main() -> None:
    rows = [run_one(res, label) for res, label in GRIDS]

    print("\n\nRESOLUTION AND SKILL  (Gorkha, Roback inventory)\n")
    print(K.table(rows, [
        ("label", "grid", "s"),
        ("in_sample_auc", "in-samp", ".4f"),
        ("cv_random_mean", "randomCV", ".4f"),
        ("cv_spatial_mean", "spatialCV", ".4f"),
        ("cv_spatial_std", "+/-", ".4f"),
        ("capture_top10pct", "top10%", ".1f"),
        ("capture_top20pct", "top20%", ".1f"),
    ]))
    print("\nWETNESS STRUCTURE  (specific catchment area at the sample points)\n")
    print(K.table(rows, [
        ("label", "grid", "s"),
        ("sca_median_m", "median", ".1f"),
        ("sca_iqr_m", "IQR", ".1f"),
        ("sca_p99_m", "p99", ".1f"),
    ]))

    K.save("01_resolution", {"grids": rows,
                             "area": "gorkha", "inventory": K.ROBACK})


if __name__ == "__main__":
    main()
