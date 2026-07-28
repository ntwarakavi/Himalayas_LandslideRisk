"""Which model should actually be used?

Experiment 03 showed that on one area, given the same two predictors, the
mechanical model and a logistic regression are indistinguishable. That leaves
the choice open, and it is the choice a user actually has to make.

Discrimination on home ground is only half the question. The other half is what
happens when a model is carried somewhere it was not fitted - which is the
normal case, since most of the Hindu Kush Himalaya has no inventory at all. A
statistical model's coefficients are tied to the predictor distribution they
were fitted in; the mechanical model's parameters claim to be properties of
soil. That claim is testable.

Every model is fitted on Gorkha at 30 m and applied unchanged to Far-Western
Nepal and Sikkim, scored against those areas' own inventories. "Home" is the
in-sample Gorkha score, for reference.

    python analysis/08_transfer_benchmark.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common as K

RES = 0.00027778          # 30 m
SOURCE = "gorkha"
TARGETS = ["farwest", "sikkim"]


def load(area: str):
    """Slope, SCA, recharge, elevation and lithology at presence + background."""
    cfg = K.make_config(area, RES)
    layers = K.terrain_layers(cfg, want_precip=True)
    litho = K.region_layer(area, RES, "lithology")
    paths = [layers["slope"], layers["sca"], layers["recharge"],
             layers["dem"], litho]
    pres, vp, bg, vb = K.sample(cfg, paths, area=area)
    ok_p, ok_b = K.clean(vp[:, :4]), K.clean(vb[:, :4])
    return cfg, pres[ok_p], vp[ok_p], bg[ok_b], vb[ok_b]


def design(v, which, litho_codes):
    slope = v[:, 0]
    log_sca = np.log10(np.clip(v[:, 1], 1.0, None))
    if which == "terrain":
        return np.column_stack([slope, log_sca])
    litho = np.column_stack([(v[:, 4] == c).astype(float) for c in litho_codes])
    return np.column_stack([slope, log_sca, v[:, 3], v[:, 2], litho])


def stack(vp, vb, which, litho_codes):
    X = np.vstack([design(vp, which, litho_codes),
                   design(vb, which, litho_codes)])
    y = np.concatenate([np.ones(len(vp)), np.zeros(len(vb))])
    ok = np.isfinite(X).all(axis=1)
    return X[ok], y[ok]


def main() -> None:
    print(f"=== fitting every model on {SOURCE} at 30 m ===")
    _, sp, svp, sbg, svb = load(SOURCE)
    print(f"  {len(sp)} landslides, {len(sbg)} background")

    codes = np.concatenate([svp[:, 4], svb[:, 4]])
    litho_codes = [c for c in np.unique(codes[np.isfinite(codes)])
                   if (codes == c).mean() > 0.01]

    # Physical model.
    fit = K.P.fit_parameters(svp[:, 0], svp[:, 1], svb[:, 0], svb[:, 1],
                             n_samples=60, recharge_pres=svp[:, 2],
                             recharge_bg=svb[:, 2])
    params = fit["parameters"]
    print(f"  SINMAP {params.as_dict()}")

    # Statistical models, on the two predictor sets.
    trained = {}
    for which in ("terrain", "context"):
        X, y = stack(svp, svb, which, litho_codes)
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                    max_features="sqrt", n_jobs=-1,
                                    random_state=0).fit(X, y)
        trained[f"logistic [{which}]"] = ("lr", sc, lr, which)
        trained[f"random forest [{which}]"] = ("rf", sc, rf, which)

    def score_all(vp, vb, label):
        out = {}
        s = np.concatenate([vp[:, 0], vb[:, 0]])
        a = np.concatenate([vp[:, 1], vb[:, 1]])
        r = np.concatenate([vp[:, 2], vb[:, 2]])
        y = np.concatenate([np.ones(len(vp)), np.zeros(len(vb))])
        p = K.P.failure_probability(s, a, params, n_samples=200,
                                    recharge_scale=r)
        ok = np.isfinite(p)
        out["SINMAP (physics)"] = K.auc(p[ok], y[ok])
        for nm, (kind, sc, mdl, which) in trained.items():
            X, yy = stack(vp, vb, which, litho_codes)
            pp = (mdl.predict_proba(sc.transform(X))[:, 1] if kind == "lr"
                  else mdl.predict_proba(X)[:, 1])
            out[nm] = K.auc(pp, yy)
        print(f"  {label}: " + "  ".join(f"{k.split(' [')[0][:8]}={v:.3f}"
                                         for k, v in out.items()))
        return out

    results = {"home": score_all(svp, svb, "gorkha")}
    for area in TARGETS:
        print(f"\n=== applying to {area}, unchanged ===")
        try:
            _, pres, vp, bg, vb = load(area)
        except Exception as exc:                          # noqa: BLE001
            print(f"  skipped: {exc}")
            continue
        print(f"  {len(pres)} landslides, {len(bg)} background")
        results[area] = score_all(vp, vb, area)

    rows = []
    for m in list(results["home"]):
        row = {"model": m}
        for area, sc in results.items():
            row[area] = round(float(sc[m]), 4)
        away = [sc[m] for a, sc in results.items() if a in TARGETS]
        row["mean_away"] = round(float(np.mean(away)), 4) if away else None
        row["drop"] = (round(float(results["home"][m] - np.mean(away)), 4)
                       if away else None)
        rows.append(row)

    print("\n\nTRANSFER FROM GORKHA, EVERY MODEL FITTED ONLY THERE  (30 m)\n")
    cols = [("model", "model", "s"), ("home", "gorkha", ".4f")]
    cols += [(a, a, ".4f") for a in TARGETS if a in results]
    cols += [("mean_away", "mean away", ".4f"), ("drop", "drop", ".4f")]
    print(K.table(rows, cols))

    K.save("08_transfer_benchmark", {"rows": rows, "source": SOURCE,
                                     "resolution_deg": RES})


if __name__ == "__main__":
    main()
