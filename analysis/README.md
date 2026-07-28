# Experiments

Six scripts, each answering one question about the model. They share sampling
and fold logic through `common.py`, so every comparison holds the presence
points, the background points and the spatial folds fixed and varies one thing.

```bash
python analysis/01_resolution.py           # does grid resolution matter, and why?
python analysis/02_transfer.py             # does a fit made in one place work elsewhere?
python analysis/03_domain.py               # where does the model stop working?
python analysis/04_sensitivity.py          # what rests on the two unfitted conventions?
python analysis/05_inventory_geometry.py   # where should a polygon inventory be sampled?
python analysis/06_calibration_regions.py  # do per-lithology parameters ever help?
```

These study the SINMAP model, which is the only model in the repository. Two
further experiments compared it against logistic regression and a random forest;
those scripts instantiated other models, so they were removed when the codebase
was consolidated on SINMAP. Their results stand and are reported in
[`../docs/RESULTS.md`](../docs/RESULTS.md) sections 2 and 8, and the scripts
themselves are recoverable from git at commit `407d976`:

```bash
git show 407d976:analysis/03_benchmark.py
git show 407d976:analysis/08_transfer_benchmark.py
```

Their **results are kept**, since the conclusions in RESULTS.md rest on them:

```
results/archived_benchmark_vs_statistical.json   RESULTS.md section 2
results/archived_benchmark_transfer.json         RESULTS.md section 8
```

Reproducing them needs scikit-learn, which is not a dependency of the model.

Run them from the repository root. Each prints its tables and writes a JSON to
`analysis/results/`. Findings are written up in [`../docs/RESULTS.md`](../docs/RESULTS.md).

## Notes on cost

`01` and `03` build the Gorkha terrain at three grids and at 30 m respectively;
`02` and `04` additionally route Far-Western Nepal, which is 17 million cells.
Flow routing is not tiled, so those are the expensive steps — a few minutes each
at 30 m, and they dominate the runtime.

All scripts name their work files `exp_<area>_<resolution>` through
`common.work_name`, so the terrain is routed once and reused across
experiments. Deleting `data/work/` forces a recompute.

Do not run two scripts covering the same area and resolution concurrently on a
cold cache: they would both route the same grid. Raster writes are atomic, so
the result is correct either way, but the work is wasted. Run one to warm the
cache, then the rest can go in parallel.

## Protocol

Every cross-validated number follows the same rules:

- Presence points come from the area's own inventory; background is drawn over
  the same extent and screened by the slope raster.
- Folds are assigned once per comparison and shared by every model in it.
- Spatial-block folds use 0.25° blocks (~25 km), assigned whole to a fold, so
  no test point has training data nearby.
- Every model — including the parameter search — is refitted inside each fold.
  Nothing selects on the test set.
- AUC is Mann-Whitney with tie correction. Capture is the share of landslides
  falling above the background distribution's (1−f) quantile, which stands in
  for map area.
