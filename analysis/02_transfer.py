"""Does a fit made in one place work in another?

Everything reported so far scores the map against the landslides its parameters
were fitted to. Spatial-block cross-validation withholds ground, but it still
withholds ground inside one area, one inventory and one triggering mechanism.
The harder question is whether parameters fitted on the Gorkha earthquake
footprint say anything useful about a monsoon-driven catchment 400 km west, or
about Sikkim 300 km east.

Two directions are run:

* **transfer** - parameters fitted on Gorkha, applied unchanged to another area
  and scored against that area's own inventory. Nothing about the target area
  informed the parameters.
* **local** - parameters refitted on the target area, as an upper bound. The
  gap between the two is the cost of transferring.

    python analysis/02_transfer.py
"""

from __future__ import annotations

import numpy as np

import common as K

RES = 0.00027778          # 30 m
SOURCE = "gorkha"
TARGETS = ["farwest", "sikkim"]


def sample_area(area: str, res: float):
    cfg = K.make_config(area, res)
    layers = K.terrain_layers(cfg, want_precip=True)
    paths = [layers["slope"], layers["sca"], layers["recharge"]]
    pres, vp, bg, vb = K.sample(cfg, paths)
    ok_p, ok_b = K.clean(vp), K.clean(vb)
    return cfg, pres[ok_p], vp[ok_p], bg[ok_b], vb[ok_b], layers


def score(params, vp, vb) -> dict:
    s = np.concatenate([vp[:, 0], vb[:, 0]])
    a = np.concatenate([vp[:, 1], vb[:, 1]])
    r = np.concatenate([vp[:, 2], vb[:, 2]])
    y = np.concatenate([np.ones(len(vp)), np.zeros(len(vb))])
    p = K.P.failure_probability(s, a, params, n_samples=200,
                                recharge_scale=r)
    ok = np.isfinite(p)
    out = {"auc": round(K.auc(p[ok], y[ok]), 4),
           "n_presence": int(len(vp)), "n_background": int(len(vb))}
    out.update(K.capture(p[ok], y[ok]))
    return out


def main() -> None:
    print(f"=== source area: {SOURCE} at 30 m ===")
    src_cfg, sp, svp, sbg, svb = sample_area(SOURCE, RES)[:5]
    print(f"  {len(sp)} landslides, {len(sbg)} background")
    src_fit = K.P.fit_parameters(svp[:, 0], svp[:, 1], svb[:, 0], svb[:, 1],
                                 n_samples=60, recharge_pres=svp[:, 2],
                                 recharge_bg=svb[:, 2])
    src_params = src_fit["parameters"]
    print(f"  fitted {src_params.as_dict()}")
    print(f"  in-sample AUC {src_fit['auc']:.4f}")

    rows = [dict(area=SOURCE, mode="fitted here",
                 parameters=src_params.as_dict(), **score(src_params, svp, svb))]

    for area in TARGETS:
        print(f"\n=== target area: {area} at 30 m ===")
        try:
            cfg, pres, vp, bg, vb, _ = sample_area(area, RES)
        except Exception as exc:                          # noqa: BLE001
            print(f"  SKIPPED: {type(exc).__name__}: {exc}")
            rows.append(dict(area=area, mode="failed", error=str(exc)))
            continue
        print(f"  {len(pres)} landslides, {len(bg)} background")

        # Transferred: Gorkha parameters, untouched.
        t = score(src_params, vp, vb)
        t.update(area=area, mode="transferred from gorkha",
                 parameters=src_params.as_dict())
        rows.append(t)
        print(f"  transferred AUC {t['auc']:.4f}   "
              f"top20% capture {t['capture_top20pct']:.1f}%")

        # Local upper bound, if the inventory is big enough to fit.
        if len(pres) >= 100:
            local = K.P.fit_parameters(vp[:, 0], vp[:, 1], vb[:, 0], vb[:, 1],
                                       n_samples=60, recharge_pres=vp[:, 2],
                                       recharge_bg=vb[:, 2])
            lo = score(local["parameters"], vp, vb)
            lo.update(area=area, mode="fitted here",
                      parameters=local["parameters"].as_dict())
            rows.append(lo)
            print(f"  refitted    AUC {lo['auc']:.4f}   "
                  f"cost of transfer {lo['auc'] - t['auc']:+.4f}")
        else:
            print(f"  too few landslides ({len(pres)}) to refit locally")

    print("\n\nTRANSFER  (parameters fitted on Gorkha, 30 m)\n")
    print(K.table(rows, [
        ("area", "area", "s"),
        ("mode", "parameters", "s"),
        ("n_presence", "slides", "d"),
        ("auc", "AUC", ".4f"),
        ("capture_top10pct", "top10%", ".1f"),
        ("capture_top20pct", "top20%", ".1f"),
    ]))

    K.save("02_transfer", {"rows": rows, "source": SOURCE,
                           "resolution_deg": RES})


if __name__ == "__main__":
    main()
