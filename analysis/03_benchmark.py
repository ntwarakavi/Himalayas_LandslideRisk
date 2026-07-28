"""Does the physics beat a statistical model given the same data?

The comparison every reviewer will ask for, run under one protocol: the same
presence points, the same background points, the same spatial-block folds, and
every model fitted inside each fold and scored on the fold withheld.

Two predictor sets, because they answer different questions.

``terrain``
    Slope and log specific catchment area - exactly what the stability model
    sees, and nothing else. This isolates the question of interest: does
    imposing the SINMAP functional form beat learning a free function from the
    same two numbers? The physical model has three parameters; the random
    forest has hundreds of splits.

``context``
    The above plus elevation, wettest-month precipitation, lithology and land
    cover. This is what a well-resourced statistical susceptibility model would
    actually be given, and the stability model cannot use any of it. If the
    statistical models win here but not on ``terrain``, the advantage is the
    extra data rather than the method.

    python analysis/03_benchmark.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common as K

RES = 0.00027778          # 30 m: where the physics has its best case
N_FOLDS = 5
BLOCK_DEG = 0.25


def build_features(cfg, layers, region_paths):
    """Sample every predictor at presence and background points."""
    paths = [layers["slope"], layers["sca"], layers["dem"], layers["recharge"],
             region_paths["lithology"], region_paths["landcover"]]
    pres, vp, bg, vb = K.sample(cfg, paths, area="gorkha")
    ok_p, ok_b = K.clean(vp[:, :4]), K.clean(vb[:, :4])
    return pres[ok_p], vp[ok_p], bg[ok_b], vb[ok_b]


def design(v: np.ndarray, which: str, litho_codes, lc_codes) -> np.ndarray:
    """Predictor matrix. Column 0/1 are slope and log10 SCA throughout."""
    slope = v[:, 0]
    log_sca = np.log10(np.clip(v[:, 1], 1.0, None))
    if which == "terrain":
        return np.column_stack([slope, log_sca])

    elev, precip = v[:, 2], v[:, 3]
    litho = np.column_stack([(v[:, 4] == c).astype(float) for c in litho_codes])
    lc = np.column_stack([(v[:, 5] == c).astype(float) for c in lc_codes])
    return np.column_stack([slope, log_sca, elev, precip, litho, lc])


def main() -> None:
    cfg = K.make_config("gorkha", RES)
    print(f"Benchmark on Gorkha at 30 m ({cfg.cell_count():,} cells)\n")

    layers = K.terrain_layers(cfg, want_precip=True)

    region_paths = {kind: K.region_layer("gorkha", RES, kind)
                    for kind in ("lithology", "landcover")}

    pres, vp, bg, vb = build_features(cfg, layers, region_paths)
    print(f"  {len(pres)} landslides, {len(bg)} background points")

    codes = np.concatenate([vp[:, 4], vb[:, 4]])
    litho_codes = [c for c in np.unique(codes[np.isfinite(codes)])
                   if (codes == c).mean() > 0.01]
    lcv = np.concatenate([vp[:, 5], vb[:, 5]])
    lc_codes = [c for c in np.unique(lcv[np.isfinite(lcv)])
                if (lcv == c).mean() > 0.01]
    print(f"  lithologies used: {[int(c) for c in litho_codes]}")
    print(f"  land-cover classes used: {[int(c) for c in lc_codes]}")

    results = []
    for scheme in ("spatial", "random"):
        fp, fb = K.folds(pres, bg, cfg.clipped_bbox(), scheme, N_FOLDS,
                         BLOCK_DEG)
        print(f"\n--- {scheme} split ---")

        scores = {name: [] for name in
                  ("SINMAP (physics)", "logistic [terrain]",
                   "random forest [terrain]", "logistic [context]",
                   "random forest [context]")}
        captures = {name: [] for name in scores}

        for k in range(N_FOLDS):
            tr_p, tr_b = fp != k, fb != k
            te_p, te_b = fp == k, fb == k
            if min(tr_p.sum(), tr_b.sum()) < 50 or \
                    min(te_p.sum(), te_b.sum()) < 10:
                continue
            y_te = np.concatenate([np.ones(int(te_p.sum())),
                                   np.zeros(int(te_b.sum()))])

            # --- the physical model -------------------------------------
            fit = K.P.fit_parameters(
                vp[tr_p, 0], vp[tr_p, 1], vb[tr_b, 0], vb[tr_b, 1],
                n_samples=60, recharge_pres=vp[tr_p, 3],
                recharge_bg=vb[tr_b, 3])
            s_te = np.concatenate([vp[te_p, 0], vb[te_b, 0]])
            a_te = np.concatenate([vp[te_p, 1], vb[te_b, 1]])
            r_te = np.concatenate([vp[te_p, 3], vb[te_b, 3]])
            p = K.P.failure_probability(s_te, a_te, fit["parameters"],
                                        n_samples=200, recharge_scale=r_te)
            ok = np.isfinite(p)
            scores["SINMAP (physics)"].append(K.auc(p[ok], y_te[ok]))
            captures["SINMAP (physics)"].append(K.capture(p[ok], y_te[ok]))

            # --- statistical models -------------------------------------
            for which in ("terrain", "context"):
                Xtr = np.vstack([design(vp[tr_p], which, litho_codes, lc_codes),
                                 design(vb[tr_b], which, litho_codes, lc_codes)])
                ytr = np.concatenate([np.ones(int(tr_p.sum())),
                                      np.zeros(int(tr_b.sum()))])
                Xte = np.vstack([design(vp[te_p], which, litho_codes, lc_codes),
                                 design(vb[te_b], which, litho_codes, lc_codes)])
                good_tr = np.isfinite(Xtr).all(axis=1)
                good_te = np.isfinite(Xte).all(axis=1)
                Xtr, ytr = Xtr[good_tr], ytr[good_tr]

                sc = StandardScaler().fit(Xtr)
                lr = LogisticRegression(max_iter=2000, C=1.0)
                lr.fit(sc.transform(Xtr), ytr)
                pl = lr.predict_proba(sc.transform(Xte[good_te]))[:, 1]

                rf = RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                    n_jobs=-1, random_state=0)
                rf.fit(Xtr, ytr)
                pr = rf.predict_proba(Xte[good_te])[:, 1]

                yy = y_te[good_te]
                for nm, pp in ((f"logistic [{which}]", pl),
                               (f"random forest [{which}]", pr)):
                    scores[nm].append(K.auc(pp, yy))
                    captures[nm].append(K.capture(pp, yy))

            print(f"  fold {k}: " + "  ".join(
                f"{nm.split(' [')[0][:8]}={scores[nm][-1]:.3f}"
                for nm in scores if scores[nm]))

        for nm, aucs in scores.items():
            if not aucs:
                continue
            row = K.summarise(nm, aucs)
            row["scheme"] = scheme
            for frac in ("capture_top5pct", "capture_top10pct",
                         "capture_top20pct"):
                row[frac] = round(float(np.mean(
                    [c[frac] for c in captures[nm]])), 2)
            results.append(row)

    for scheme in ("spatial", "random"):
        rows = [r for r in results if r["scheme"] == scheme]
        print(f"\n\n{scheme.upper()} 5-FOLD CV  (Gorkha, 30 m, "
              f"identical points and folds)\n")
        print(K.table(rows, [
            ("model", "model", "s"),
            ("auc_mean", "AUC", ".4f"),
            ("auc_std", "+/-", ".4f"),
            ("capture_top5pct", "top5%", ".1f"),
            ("capture_top10pct", "top10%", ".1f"),
            ("capture_top20pct", "top20%", ".1f"),
        ]))

    K.save("03_benchmark", {"results": results, "resolution_deg": RES,
                            "n_folds": N_FOLDS, "block_deg": BLOCK_DEG,
                            "area": "gorkha", "inventory": K.ROBACK})


if __name__ == "__main__":
    main()
