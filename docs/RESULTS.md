# Results

What the model does, measured. Every number here comes from a script in
[`../analysis/`](../analysis/) and a JSON in `analysis/results/`; the protocol is
described in [`../analysis/README.md`](../analysis/README.md).

Read the **spatial-block** column throughout. A random split scatters test
points among training points on the same hillsides, so it measures spatial
interpolation as much as skill. The gap between the two columns turns out to be
the most informative quantity in this document.

## Summary

| Question | Answer |
|---|---|
| Does resolution matter? | Decisively. Spatial-block AUC 0.728 → 0.816 from 250 m to 30 m |
| Does the physics beat statistics? | No. It ties logistic regression given the same inputs |
| Then what does it buy? | Stability across space, interpretable parameters, and extrapolation |
| Do extra predictors help? | Under random CV, hugely. Under spatial CV, not at all |
| Does a fit transfer? | *(§3)* |
| Does the trigger mechanism cap the score? | *(§4)* |
| What rests on the unfitted conventions? | *(§5)* |

## 1. Resolution

`analysis/01_resolution.py` — Gorkha, Roback inventory, 5,193 landslides and
10,386 background points, identical at every grid. Only the grid changes.

| Grid | Cells | In-sample | Random CV | **Spatial CV** | ± | Capture top 10 % | top 20 % |
|---|---|---|---|---|---|---|---|
| 250 m | 76.8 k | 0.7382 | 0.7382 | **0.7278** | 0.0330 | 43.1 % | 59.0 % |
| 90 m | 691 k | 0.8091 | 0.8091 | **0.8059** | 0.0320 | 53.6 % | 69.0 % |
| 30 m | 6.22 M | 0.8214 | 0.8215 | **0.8155** | 0.0204 | 53.1 % | 71.0 % |

The mechanism is visible in the wetness term. Specific catchment area at the
sample points:

| Grid | Median | IQR | 99th percentile |
|---|---|---|---|
| 250 m | 631 m | 1372 m | 266 km |
| 90 m | 258 m | 527 m | 118 km |
| 30 m | 132 m | 260 m | 51.8 km |

A coarse grid cannot represent a hollow. At 250 m a cell is wider than most of
the convergent features that concentrate subsurface flow, so specific catchment
area is smeared towards a single large value and the wetness term loses the
contrast it depends on. Refining the grid restores that contrast, and skill
follows: **+0.078 AUC from 250 m to 90 m**, a further +0.010 to 30 m.

Two things worth noting:

- **Most of the gain is realised by 90 m.** If compute is the binding
  constraint, 90 m captures nine tenths of the benefit at a ninth of the cost.
- **The fold-to-fold spread narrows** from ±0.033 to ±0.020, so the finer grid
  is not only better on average, it is more consistent from place to place.

This also settles a question the earlier configuration left open: the 250 m
result reported before this work (0.729) was not the model's ceiling, it was the
grid's.

## 2. Benchmark against statistical models

`analysis/03_benchmark.py` — Gorkha at 30 m. Same presence points, same
background points, same spatial folds for every model. Every model, including
the SINMAP parameter search, is refitted inside each fold.

Two predictor sets:

- **terrain** — slope and log₁₀ specific catchment area. Exactly what the
  stability model sees, and nothing else.
- **context** — the above plus elevation, wettest-month precipitation, lithology
  (GLiM, one-hot) and land cover (WorldCover, one-hot). What a well-resourced
  statistical susceptibility model would actually be given. The stability model
  cannot use any of it.

### Spatial-block 5-fold CV

| Model | AUC | ± | top 5 % | top 10 % | top 20 % |
|---|---|---|---|---|---|
| SINMAP (physics) | 0.8161 | 0.0208 | 35.3 % | 48.5 % | 68.5 % |
| logistic [terrain] | **0.8218** | 0.0221 | 32.5 % | 49.0 % | 69.4 % |
| random forest [terrain] | 0.8052 | 0.0187 | 29.0 % | 45.3 % | 67.5 % |
| logistic [context] | 0.8070 | 0.0662 | 36.1 % | 52.1 % | 66.0 % |
| random forest [context] | 0.8190 | 0.0516 | 36.3 % | 51.1 % | 68.1 % |

### Random 5-fold CV

| Model | AUC | ± | top 5 % | top 10 % | top 20 % |
|---|---|---|---|---|---|
| SINMAP (physics) | 0.8222 | 0.0059 | 36.9 % | 53.4 % | 71.0 % |
| logistic [terrain] | 0.8274 | 0.0051 | 36.7 % | 54.3 % | 71.0 % |
| random forest [terrain] | 0.8188 | 0.0037 | 33.6 % | 52.1 % | 69.8 % |
| logistic [context] | 0.8818 | 0.0064 | 54.8 % | 69.4 % | 80.6 % |
| random forest [context] | **0.9190** | 0.0046 | 59.9 % | 75.7 % | 88.5 % |

### What this says

**The physics does not beat statistics on discrimination.** Given the same two
predictors, SINMAP (0.8161) and logistic regression (0.8218) are indistinguishable
— the difference is a quarter of the fold-to-fold spread. SINMAP does edge out
the random forest (0.8052). A three-parameter mechanical model matches a fitted
statistical one; it does not surpass it. Any claim otherwise would be an artefact
of an unequal comparison.

**The apparent advantage of extra predictors is almost entirely spatial
leakage.** This is the sharpest result here:

| Model | Random CV | Spatial CV | Drop |
|---|---|---|---|
| SINMAP (physics) | 0.8222 | 0.8161 | **0.006** |
| logistic [terrain] | 0.8274 | 0.8218 | **0.006** |
| random forest [terrain] | 0.8188 | 0.8052 | 0.014 |
| logistic [context] | 0.8818 | 0.8070 | **0.075** |
| random forest [context] | 0.9190 | 0.8190 | **0.100** |

A random forest with lithology, land cover, elevation and precipitation reports
0.919 — a figure that would be publishable on its face and is in line with much
of the landslide susceptibility literature. Under spatial-block validation the
same model scores 0.819, no better than a two-predictor mechanical model. The
0.10 it appeared to gain was memorised geography: categorical predictors let the
model identify *where* it is and recall the local landslide density, which is
worth nothing on ground it has not seen.

**And that memorisation costs stability.** The context models' fold-to-fold
spread is ±0.052 and ±0.066, against ±0.021 for SINMAP and logistic [terrain] —
two and a half to three times more variable from place to place. They are not
merely no better on average; they are markedly less reliable anywhere in
particular.

### Methods note

The SINMAP output is a Monte Carlo probability, so it is discretised to
1/`n_samples`, while the statistical models emit continuous scores. Ties are
handled by rank averaging in the AUC, but coarse discretisation could still cost
the physical model something. It does not: scoring the same points at
`n_samples` of 100, 200, 1000 and 4000 gives AUC 0.8223, 0.8221, 0.8235, 0.8233.
The benchmark's 200 draws cost at most 0.001, well below the differences under
discussion.

## 3. Transfer to other areas

*Pending — `analysis/02_transfer.py` running.*

## 4. Triggering mechanism

*Pending — `analysis/04_monsoon.py`.*

## 5. Sensitivity to the unfitted conventions

*Pending — `analysis/05_sensitivity.py`.*

## What this means for the model

*Written once §3–5 are in.*
