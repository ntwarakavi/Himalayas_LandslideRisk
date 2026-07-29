# H-SIM experiments

Seven scripts, each answering one question about the model. They share sampling
and fold logic through `common.py`, so every comparison holds the presence
points, the background points and the spatial folds fixed and varies one thing.

```bash
python analysis/01_resolution.py           # does grid resolution matter, and why?
python analysis/02_transfer.py             # does a fit made in one place work elsewhere?
python analysis/03_domain.py               # where does the model stop working?
python analysis/04_sensitivity.py          # what rests on the two unfitted conventions?
python analysis/05_inventory_geometry.py   # where should a polygon inventory be sampled?
python analysis/06_calibration_regions.py  # do per-lithology parameters ever help?
python analysis/07_boundary_buffer.py      # how wide a buffer does a province sweep need?
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

### 08_point_validation.py — can the point catalogue be used?

The NASA GLC holds 2,469 records in the region against roughly 51,000 polygons
in the three mapped inventories, but 85 % of them are placed worse than 1 km.
This tests whether they can be used at a coarser scale: neighbourhood sampling
through a disc the size of each record's own accuracy, with background blurred
identically, and an areal density test over cells coarser than the error.

The finding is that reporting bias dominates and inverts the result - uniform
background gives AUC 0.346 and Spearman -0.74, because media reports come from
roads and settlements on gentle ground. Target-group background raises Gorkha
from 0.585 to 0.633, but that background is also 6% less susceptible, so part
of the gain is an easier baseline rather than bias removal; and with 37 records
the 95% interval is 0.528-0.738, barely excluding chance.

Inconclusive at catchment scale, and currently used for nothing. Re-run after
the regional sweep, where about 1,455 records would cut the standard error from
0.054 to roughly 0.01. See RESULTS section 14.
