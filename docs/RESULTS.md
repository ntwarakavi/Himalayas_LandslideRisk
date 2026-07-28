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
| Do extra predictors help? | Under random CV, hugely. Under spatial CV, not at all |
| Do the fitted parameters transfer? | Yes, nearly for free — 0.004 to 0.022 AUC |
| Does the model work everywhere? | No. 0.816 on Gorkha, 0.656 in Far-Western Nepal |
| Is that the trigger mechanism? | No — that hypothesis is refuted |
| Do calibration regions help? | No, in either area: −0.0004 AUC |
| What rests on the unfitted conventions? | Rainfall, almost nothing. Seismic, a great deal |
| So which model should I use? | The mechanical one — it degrades least when moved (§8) |

Then what *does* the physics buy? Not discrimination. It buys three interpretable
parameters instead of a fitted surface, near-total insensitivity to spatial
leakage, the smallest loss of any model tested when carried to new ground, and
the ability to state a scenario the data never contained. Sections 8 and 9 set
that out.

## 1. Resolution

`analysis/01_resolution.py` — Gorkha, Roback inventory, 5,193 landslides and
10,386 background points, identical at every grid. Only the grid changes.

| Grid | Cells | In-sample | Random CV | **Spatial CV** | ± | Capture top 10 % | top 20 % |
|---|---|---|---|---|---|---|---|
| 250 m | 76.8 k | 0.7380 | 0.7379 | **0.7281** | 0.0328 | 43.1 % | 58.8 % |
| 90 m | 691 k | 0.8091 | 0.8091 | **0.8059** | 0.0321 | 53.6 % | 69.0 % |
| 30 m | 6.22 M | 0.8215 | 0.8215 | **0.8156** | 0.0204 | 53.1 % | 71.0 % |

The mechanism is visible in the wetness term. Specific catchment area at the
sample points:

| Grid | Median | IQR | 99th percentile |
|---|---|---|---|
| 250 m | 631 m | 1372 m | 266 km |
| 90 m | 258 m | 529 m | 118 km |
| 30 m | 132 m | 260 m | 51.8 km |

A coarse grid cannot represent a hollow. At 250 m a cell is wider than most of
the convergent features that concentrate subsurface flow, so specific catchment
area is smeared towards a single large value and the wetness term loses the
contrast it depends on. Refining the grid restores that contrast, and skill
follows: **+0.078 AUC from 250 m to 90 m**, a further +0.010 to 30 m.

Three things worth noting:

- **Most of the gain is realised by 90 m.** If compute is the binding
  constraint, 90 m captures nine tenths of the benefit at a ninth of the cost.
- **The fold-to-fold spread narrows** from ±0.033 to ±0.020, so the finer grid
  is not only better on average but more consistent from place to place.
- **The measurement is conservative.** Point-sampling a raster at a mapped
  landslide's coordinate gets *more* sensitive to inventory positional error as
  the grid refines, which works against the fine grids. The gain is real
  despite that, not because of it.

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
| SINMAP (physics) | 0.8161 | 0.0207 | 35.2 % | 48.5 % | 68.8 % |
| logistic [terrain] | **0.8220** | 0.0221 | 32.5 % | 49.0 % | 69.4 % |
| random forest [terrain] | 0.8063 | 0.0193 | 28.7 % | 46.0 % | 68.1 % |
| logistic [context] | 0.8068 | 0.0666 | 36.0 % | 52.0 % | 66.0 % |
| random forest [context] | 0.8185 | 0.0505 | 35.9 % | 51.0 % | 69.0 % |

### Random 5-fold CV

| Model | AUC | ± | top 5 % | top 10 % | top 20 % |
|---|---|---|---|---|---|
| SINMAP (physics) | 0.8222 | 0.0029 | 36.5 % | 53.8 % | 71.1 % |
| logistic [terrain] | 0.8275 | 0.0033 | 37.4 % | 54.4 % | 71.0 % |
| random forest [terrain] | 0.8180 | 0.0044 | 33.2 % | 51.6 % | 70.1 % |
| logistic [context] | 0.8815 | 0.0077 | 55.0 % | 69.5 % | 80.7 % |
| random forest [context] | **0.9182** | 0.0061 | 61.3 % | 75.6 % | 87.7 % |

### What this says

**The physics does not beat statistics on discrimination.** Given the same two
predictors, SINMAP (0.8161) and logistic regression (0.8220) are
indistinguishable — the difference is a quarter of the fold-to-fold spread.
SINMAP does edge out the random forest (0.8063). A three-parameter mechanical
model matches a fitted statistical one; it does not surpass it. Any claim
otherwise would be an artefact of an unequal comparison.

**The apparent advantage of extra predictors is almost entirely spatial
leakage.** This is the sharpest result in this document:

| Model | Random CV | Spatial CV | Drop |
|---|---|---|---|
| SINMAP (physics) | 0.8222 | 0.8161 | **0.006** |
| logistic [terrain] | 0.8275 | 0.8220 | **0.006** |
| random forest [terrain] | 0.8180 | 0.8063 | 0.012 |
| logistic [context] | 0.8815 | 0.8068 | **0.075** |
| random forest [context] | 0.9182 | 0.8185 | **0.100** |

A random forest with lithology, land cover, elevation and precipitation reports
0.918 — a figure that would pass without comment in the landslide susceptibility
literature. Under spatial-block validation the same model scores 0.819, no
better than a two-predictor mechanical model. The 0.10 it appeared to gain was
memorised geography: categorical predictors let it identify *where* it is and
recall the local landslide density, which is worth nothing on ground it has not
seen.

**And that memorisation costs stability.** The context models' fold-to-fold
spread is ±0.051 and ±0.067, against ±0.021 for SINMAP and logistic [terrain] —
two and a half to three times more variable from place to place. They are not
merely no better on average; they are markedly less reliable anywhere in
particular.

### Methods note

The SINMAP output is a Monte Carlo probability, discretised to 1/`n_samples`,
while the statistical models emit continuous scores. Ties are handled by rank
averaging in the AUC, but coarse discretisation could still cost the physical
model something. It does not: scoring the same points at `n_samples` of 100,
200, 1000 and 4000 gives AUC 0.8223, 0.8221, 0.8235, 0.8233. The benchmark's
200 draws cost at most 0.001, well below the differences under discussion.

## 3. Transfer between catchments

`analysis/02_transfer.py` — parameters fitted on Gorkha at 30 m, applied
unchanged elsewhere and scored against that area's own inventory. "Fitted here"
is the local upper bound; the difference is the cost of transferring.

| Area | Parameters | Landslides | AUC | top 10 % | top 20 % |
|---|---|---|---|---|---|
| Gorkha | fitted here | 5,193 | 0.8221 | 53.1 % | 71.0 % |
| Far-West Nepal | **transferred from Gorkha** | 25,679 | 0.6557 | 21.7 % | 37.5 % |
| Far-West Nepal | fitted here | 25,679 | 0.6595 | 22.3 % | 38.5 % |
| Sikkim | **transferred from Gorkha** | 255 | 0.7801 | 41.6 % | 58.4 % |
| Sikkim | fitted here | 255 | 0.8019 | 42.4 % | 64.3 % |

**The parameters transfer almost for free.** Refitting locally buys 0.004 AUC in
Far-West Nepal and 0.022 in Sikkim — and the Sikkim figure is generous to the
local fit, since fitting three parameters to 255 landslides will absorb some
noise. A parameter set derived from one Nepali catchment carries to another 400
km west and to Sikkim 300 km east with essentially no loss.

This matters more than it first appears. It means the low Far-West score is
**not** a transfer failure. Whatever is wrong there is not fixed by refitting,
so it is not the parameters — it is the predictors.

## 4. Where the model does not work

`analysis/04_monsoon.py` — the same model, fitted and cross-validated
independently in each area at 30 m.

| Area | Landslides | In-sample | Random CV | **Spatial CV** | ± | top 20 % |
|---|---|---|---|---|---|---|
| Gorkha | 5,193 | 0.8215 | 0.8215 | **0.8156** | 0.0204 | 71.0 % |
| Far-West Nepal | 25,679 | 0.6596 | 0.6596 | **0.6563** | 0.0119 | 38.5 % |

This was set up to test a hypothesis, and **the hypothesis was wrong**. The
reasoning went: Gorkha is earthquake-triggered, so a static stability map is
being scored against failures whose locations were partly set by shaking it does
not represent; Far-Western Nepal is monsoon-driven, which is the mechanism the
wetness term actually models, so it should score *higher*. It scores 0.16 lower.

Two candidate explanations were tested and both were largely ruled out.

### 4a. It is not mainly where the inventory is sampled

`analysis/06_inventory_geometry.py`. An infinite-slope model predicts
*initiation*. Roback ships mapped source areas, whose centroid sits in the
initiation zone; Far-West ships whole-landslide polygons, whose centroid sits
somewhere down the runout on gentler, more convergent ground. Sampling the
second like the first asks the model to predict a location it never claimed.

| Area | Sampled at | Landslides | Median slope | Median SCA | Spatial CV |
|---|---|---|---|---|---|
| Far-West | centroid | 26,347 | 35.1° | 202 m | 0.6448 |
| Far-West | upper quartile | 26,344 | 36.6° | 134 m | 0.6585 |
| Far-West | **crown** | 26,344 | 36.7° | 109 m | **0.6619** |
| Gorkha | centroid | 5,193 | 41.1° | 119 m | **0.8156** |
| Gorkha | upper quartile | 5,193 | 40.8° | 112 m | 0.8137 |
| Gorkha | crown | 5,192 | 40.5° | 107 m | 0.8099 |

The effect is real and its *sign is exactly as predicted*: moving to the crown
helps Far-West (+0.017) and hurts Gorkha (−0.006), because Roback's polygons are
already source areas and the crown overshoots onto the ridge above. But +0.017
does not close a 0.16 gap. Sampling convention is a second-order effect —
worth getting right, not the explanation.

### 4b. It is not lithological zoning either

`analysis/07_calibration_regions.py`. Far-West's own attributes name a dozen
formations — Ranimatta, Lakharpata, Siwalik sandstones, basic rocks — with
thousands of failures each, exactly the heterogeneity SINMAP's calibration
regions exist for, and exactly what Gorkha (97 % metamorphics) lacks. Both
whole-area and per-region parameters were refitted inside each spatial fold and
scored on the fold withheld.

| Area | Parameters | GLiM classes | Regions fitted | AUC | ± |
|---|---|---|---|---|---|
| Far-West | whole area | 5 | 3.8 | 0.6557 | 0.0125 |
| Far-West | per lithology | 5 | 3.8 | 0.6553 | 0.0137 |
| Gorkha | whole area | 6 | 1.6 | 0.8183 | 0.0215 |
| Gorkha | per lithology | 6 | 1.6 | 0.8179 | 0.0224 |

**−0.0004 in both areas.** The capability does not work, and it does not work in
the place built to give it its best chance. One caveat on scope: GLiM's level-1
classification collapses Far-West's dozen named formations into five classes,
16,757 of the 25,679 landslides landing in a single one. The negative result is
therefore about *GLiM level-1 zoning*, not about lithological control in
general; a local geological map might do better. As implemented, calibration
regions should stay off.

### So what is it?

Local refitting does not help, sampling geometry explains a tenth of the gap,
and lithological zoning explains none. That points at the predictors themselves.
Slope and topographic wetness separate Gorkha's landslides well and Far-West's
poorly, which is what you would expect if failures there are controlled by
things neither variable sees: bedding orientation and weak mudstone horizons in
the Siwalik sequence, and road cuts, which are pervasive in that hill country
and fail for reasons no terrain model represents.

**This is a statement about the model's domain of validity, and it should travel
with any map it produces.** The model does well where shallow translational
failure on soil-mantled crystalline terrain dominates. It does considerably less
well in weak sedimentary hill country. Nothing in the fitted parameters warns
you which case you are in — only a local inventory does.

## 5. Sensitivity to the unfitted conventions

`analysis/05_sensitivity.py` — Gorkha at 30 m. Two numbers in the model are
neither mechanics nor fitted: the rainfall coefficient of variation and the
pseudo-static fraction of PGA. What rests on them?

### Rainfall coefficient of variation

100-year storm (10- and 1000-year behave the same way):

| cv | R/T multiplier | AUC | Rank ρ vs default | Area P > 0.5 | Mean P |
|---|---|---|---|---|---|
| 0.20 | 1.683 | 0.8193 | 0.993 | 40.1 % | 0.404 |
| 0.25 | 1.861 | 0.8185 | 0.995 | 40.7 % | 0.409 |
| **0.30** | **2.042** | **0.8177** | 1.000 | **41.1 %** | **0.414** |
| 0.35 | 2.226 | 0.8170 | 0.991 | 41.6 % | 0.419 |
| 0.40 | 2.413 | 0.8164 | 0.989 | 42.1 % | 0.423 |

Across the full plausible range the ranking is essentially untouched (rank
correlation ≥ 0.989, AUC varies by 0.003) and even the *level* moves only two
points of map area. **This convention can be treated as inert.** Doubling the
uncertainty in it changes nothing a user would act on.

### Pseudo-static fraction of PGA

At 0.35 g:

| Fraction | k_h | AUC | Rank ρ vs default | Area P > 0.5 | Mean P |
|---|---|---|---|---|---|
| 0.3 | 0.105 | 0.8167 | 0.955 | 57.3 % | 0.565 |
| **0.5** | **0.175** | **0.7961** | 1.000 | **69.5 %** | **0.680** |
| 0.7 | 0.245 | 0.7609 | 0.941 | 79.6 % | 0.778 |
| 1.0 | 0.350 | 0.6867 | 0.755 | 90.0 % | 0.887 |

This one matters. Over its practical range the unstable area runs from 57 % to
90 % of the map, and the ranking itself degrades (rank correlation falls to
0.755, AUC to 0.687) because at high k_h almost everything fails and there is
little left to discriminate.

**A seismic scenario must be quoted as a range over this parameter, never as a
single number.** A rainfall scenario need not be.

## 6. Protocol corrections made along the way

Two defects in the experimental setup were found and fixed while running these,
and both changed results, so they are recorded rather than quietly repaired.

**Background was drawn outside the surveyed extent.** Background points stand in
for "terrain that did not fail". Drawn beyond the ground an inventory actually
examined, they instead mean "nobody looked", and every landslide-prone hillside
in unmapped terrain becomes a false negative. All three inventories publish a
mapping-extent polygon; checking against them showed the problem was real and
uneven — Gorkha's study box is 99.8 % surveyed, but Far-West's was 60.5 % and
Sikkim's 63.8 %. Study extents are now the surveyed polygons' bounds, and
background is drawn only inside the polygon. Gorkha's numbers were unaffected;
the others moved.

**Work rasters were not written atomically and were rebuilt needlessly.** Two
experiments sharing a cached grid could target one path at once, and an
interrupted write left a truncated GeoTIFF that later runs failed to open. Fixed
by writing through a temporary and renaming, and by caching the wettest-month
raster with the same grid check the terrain already used.

## 7. What did not work

Collected in one place, because a method paper that reports only its successes
is not much use.

| Idea | Result | Kept? |
|---|---|---|
| Spatial recharge field from precipitation | +0.005 AUC, inside the fold spread | Yes, on by default — physically correct, and the gradient across the full HKH is far larger than in one catchment |
| Lithology calibration regions | −0.0004 AUC held out, in both areas | Yes, off by default; the negative result is specific to GLiM level-1 |
| Crown rather than centroid sampling | +0.017 Far-West, −0.006 Gorkha | Not wired into the pipeline; documented |
| Refitting on the monsoon inventory to escape the earthquake trigger | Hypothesis refuted — scores 0.16 *lower* | Reported as a domain-of-validity limit |

## 8. Which model to use

`analysis/08_transfer_benchmark.py` — every model fitted on Gorkha at 30 m and
applied **unchanged** to two other catchments, scored against their own
inventories. The Gorkha column is in-sample for every model alike, so it is a
fair "apparent performance" comparison; the away columns are what a user
actually gets on ground with no inventory, which is most of the region.

| Model | Gorkha (in-sample) | Far-West | Sikkim | **Mean away** | Drop |
|---|---|---|---|---|---|
| **SINMAP (physics)** | 0.8221 | 0.6557 | **0.7801** | **0.7179** | **0.104** |
| logistic [terrain] | 0.8274 | 0.6547 | 0.7679 | 0.7113 | 0.116 |
| random forest [terrain] | 0.9422 | 0.6318 | 0.7403 | 0.6860 | 0.256 |
| logistic [context] | 0.8763 | 0.6017 | 0.5917 | 0.5967 | 0.280 |
| random forest [context] | **0.9742** | 0.5916 | 0.6702 | 0.6309 | 0.343 |

**The ranking at home is very nearly the reverse of the ranking away.** The
random forest with the full predictor set is the best model on the data it was
fitted to (0.974) and the worst or near-worst everywhere else. The mechanical
model is mid-table at home and first away, with the smallest drop of any model
tested.

That is the whole argument for using it. It does not out-discriminate a logistic
regression — 0.822 against 0.827 at home, 0.718 against 0.711 away, differences
well inside the noise. What it does is degrade least when moved, while also
being the only one of the five that can state a scenario the training data never
contained.

**Recommendation.** Use the mechanical model, at 30 m where compute allows and
90 m otherwise, with the continuous failure-probability output, spatial recharge
on and calibration regions off. Fit it to a source-area inventory in terrain
resembling the target, and validate locally before relying on it. Do not choose
a model on its home-ground score — on this evidence that selects almost exactly
the wrong one.

## 9. What the physics actually buys

The benchmark says plainly that the mechanical model does not out-discriminate a
logistic regression on the same two variables. Four things it does do, none of
which is an AUC:

1. **It is nearly immune to spatial leakage.** 0.006 between random and
   spatial-block CV, against 0.075 and 0.100 for the context models. Its skill
   is a relationship, not a memory of places.
2. **Its parameters transfer.** 0.004–0.022 AUC to refit locally. A statistical
   model's coefficients are tied to the predictor distribution they were fitted
   in; `φ ∈ [25°, 35°]` is a property of soil.
3. **Its parameters are checkable against something other than the inventory.**
   The Gorkha fit returns `φ ∈ [25°, 35°]` and `C ≤ 0.25` — quantities a
   geotechnical laboratory measures independently. A random forest with the same
   AUC offers nothing to check.
4. **It extrapolates to conditions the data never contained.** A 1000-year storm
   or 0.35 g of shaking enters as a term in the force balance, not as a region of
   predictor space with no training data in it. Section 5 quantifies the
   confidence that deserves: high for rainfall, guarded for seismic.

And what it does not buy: better separation of mapped landslides than
conventional statistics, or any protection against being applied where its
assumptions do not hold. Section 4 is the honest boundary.
