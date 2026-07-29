# Results

What H-SIM does, measured. Every number here comes from a script in
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
| How far does one fit travel? | 0.472 to 0.780 over five independent inventories, and the low end is no skill at all (§12) |
| What do the maps mean for towns and roads? | Two-thirds of settlements are scored by what is above them, not what they sit on (§11) |
| Does near-term climate change move that? | Barely — +2 settlements and +35 km of road by 2041-2060, inside the model's own error (§11) |

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

Gorkha at 30 m. (The benchmark script was removed when the codebase was consolidated on SINMAP, since it instantiated other models; it is recoverable with `git show 407d976:analysis/03_benchmark.py`.) Same presence points, same
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

`analysis/03_domain.py` — the same model, fitted and cross-validated
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

`analysis/05_inventory_geometry.py`. An infinite-slope model predicts
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

`analysis/06_calibration_regions.py`. Far-West's own attributes name a dozen
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

`analysis/04_sensitivity.py` — Gorkha at 30 m. Two numbers in the model are
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

## 6. How wide a buffer a province-by-province sweep needs

`analysis/07_boundary_buffer.py` — a regional run is cut into states and
provinces, and a provincial border crosses catchments. Routing over a box
clipped at the border starts every catchment there, so cells just inside get
too little upslope area. `pipeline.run_admin_unit` routes over a buffered box
and clips the output back; this measures whether that is necessary and how wide
the buffer has to be.

A 400 × 500 cell window inside the cached 30 m Gorkha DEM stands in for a
province. It is routed alone and with increasing buffers, and compared against
the same cells taken from routing the full extent.

| Buffer | | Median SCA ratio | Cells losing >½ | ... in the outer ring | Mean \|ΔP\| | Cells with \|ΔP\| > 0.05 |
|---|---|---|---|---|---|---|
| none | 0 km | 1.0000 | 1.00 % | 3.77 % | 0.00110 | 0.449 % |
| 0.028° | 3.1 km | 1.0000 | **0.00 %** | **0.00 %** | **0.00000** | **0.000 %** |
| 0.083° | 9.2 km | 1.0000 | 0.00 % | 0.00 % | 0.00000 | 0.000 % |
| 0.167° | 18.5 km | 1.0000 | 0.00 % | 0.00 % | 0.00000 | 0.000 % |
| 0.222° | 24.6 km | 1.0000 | 0.00 % | 0.00 % | 0.00000 | 0.000 % |

**The truncation is real but far smaller and more local than expected, and 3 km
of buffer removes it entirely.** The median cell is unaffected even with no
buffer at all; the damage is confined to the outermost ring, where 3.8 % of
cells lose more than half their catchment area.

The reason is the scale mismatch. Hillslope contributing areas — the ones the
wetness term actually discriminates on — are hundreds of metres. Cells with
genuinely long flow paths are valley floors, and those are already saturated at
`w = 1`, where losing upslope area changes nothing because the term is capped.

This changed the default: `admin_buffer_deg` was set to 0.25° on the assumption
that catchments needed tens of kilometres, and is now 0.05° (5.5 km), roughly
twice the measured requirement. That is not a cosmetic change — buffering a
1° × 1° province by 0.25° grows it to 2.25× the cells, against 1.21× at 0.05°,
for no measured gain.

One caveat on scope: this was measured on steep crystalline terrain at 30 m.
Flatter ground with longer flow paths could push the requirement out, which is
why the default carries a factor of two rather than sitting at the measured
edge.

## 7. Protocol corrections made along the way

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

## 8. What did not work

Collected in one place, because a method paper that reports only its successes
is not much use.

| Idea | Result | Kept? |
|---|---|---|
| Spatial recharge field from precipitation | +0.005 AUC, inside the fold spread | Yes, on by default — physically correct, and the gradient across the full HKH is far larger than in one catchment |
| Lithology calibration regions | −0.0004 AUC held out, in both areas | Yes, off by default; the negative result is specific to GLiM level-1 |
| Crown rather than centroid sampling | +0.017 Far-West, −0.006 Gorkha | Not wired into the pipeline; documented |
| Refitting on the monsoon inventory to escape the earthquake trigger | Hypothesis refuted — scores 0.16 *lower* | Reported as a domain-of-validity limit |
| Scoring an asset by the worst reaching cell | Saturates: 56 % of settlements in the top band | Replaced by a proximity-weighted mean (§11) |

## 9. Which model to use

Every model fitted on Gorkha at 30 m and
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

## 10. What the physics actually buys

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

## 11. Exposure: what the maps mean for settlements and roads

Measured on the Gorkha 30 m map, over 84.5-85.3 E, 27.6-28.2 N: **639
OpenStreetMap settlements** and **7,851 road ways**, cut into **18,109 segments
of 500 m** totalling 6,994 km.

### The scoring statistic matters more than the angle

The first implementation scored an asset by the *highest* failure probability
among the cells that could reach it. That statistic does not survive contact
with Himalayan relief.

| | Maximum over reaching cells | Proximity-weighted mean |
|---|---|---|
| Settlements in the top band | 3,199 of 5,677 (**56 %**) | 23 of 639 (**3.6 %**) |
| Mean over all settlements | 0.547 | 0.106 |

A 2 km search radius at 30 m puts a few thousand cells above a typical valley
settlement — the median settlement here has **2,300 reaching cells**, the 90th
percentile 2,175 and the 99th over 5,000. With 7.3 % of the Gorkha landscape
above P = 0.6, the probability that *none* of several thousand upslope cells
clears that bar is negligible. The maximum was therefore reporting the size of
the search window, not the exposure of the town.

`reaching_max` is still written per asset, because "the worst single cell above
this place" is a real diagnostic. It is not banded and not the headline.

### Reach dominates the on-site term, which is the point

| Quantity | Mean over 639 settlements |
|---|---|
| Failure probability of the cell the settlement sits on | 0.024 |
| Proximity-weighted probability of ground that can reach it | **0.106** |
| Settlements where the reaching term exceeds the on-site term | **67 %** |

Sampling susceptibility at a town's coordinates would have reported a mean of
0.024 and called two-thirds of these settlements safe. The exception is the top
of the ranking, which is dominated by the on-site term: the highest-scoring
places are OSM nodes that fall on ground the model calls unconditionally
unstable. Those are worth checking individually — a hamlet node placed on a
cliff is as likely to be a geocoding artefact as a settlement in real danger.

### Present day, and four near-term futures

| Climate | Settlements exposed | Road km exposed | Mean settlement score |
|---|---|---|---|
| present day | 321 | 4,873.0 | 0.116 |
| SSP2-4.5 2021-2040 | 322 | 4,886.7 | 0.117 |
| SSP5-8.5 2021-2040 | 322 | 4,887.2 | 0.117 |
| SSP2-4.5 2041-2060 | 323 | 4,908.1 | 0.118 |
| SSP5-8.5 2041-2060 | 323 | 4,897.6 | 0.118 |

Exposed means a score at or above 0.08. Over a 20 to 30 year horizon the signal
is **small**: two more settlements and 35 km more road at worst, on a mean score
that moves by 0.0017. That is consistent with the raster-level result — the
unstable fraction moves from 10.5 % to 10.7 % — and with the mechanism: once the
wetness term is capped at saturation on the convergent ground where failures
concentrate, more water changes nothing there.

Two things in that table are worth not glossing over. **SSP2-4.5 exceeds
SSP5-8.5 in the 2041-2060 window**, in both columns. This is not an error: the
IPSL-CM6A-LR wettest-month field over this box gives a recharge multiplier with
median 1.12 under SSP2-4.5 against 1.08 under SSP5-8.5. Monsoon precipitation
does not respond monotonically to forcing in a single model over a single
catchment, and a suite of one GCM cannot separate that from noise. Quote the
spread, not the ordering.

**And the changes are smaller than the model's own uncertainty.** Held-out AUC
on this terrain is 0.816 ± 0.020; a shift of 0.0017 in a mean exposure score is
far inside that. The honest reading is that near-term climate change is not what
determines exposure in this catchment — where the roads and houses are is.

### Data provenance, and one failure worth recording

Both layers came from OpenStreetMap on the run reported here. An earlier run of
the same area returned **6 road ways** instead of 7,851, because Overpass
answered 406 from one mirror and 429 from another, and the model fell back to
Natural Earth trunk routes. Neither code meant what it appeared to: both mirrors
reject requests carrying no User-Agent, and the 429 body — "please include a
meaningful User-Agent string to avoid rate-limiting" — reads as throttling. The
fallback is still there and still labelled in each feature's `source` field, but
a road layer of a few dozen segments across a district is the signature of it
having fired.

## 12. Three more inventories, tested and rejected

Three further open inventories were found, wired in, and scored. None of them
improved or usefully checked the model, so none is shipped. They are recorded
here with their DOIs so the search is not repeated.

| Inventory | Where | Size | Result |
|---|---|---|---|
| Nepal monsoon, Sentinel-1 timed (Zenodo 7970874) | west-central Nepal | 499 polygons | AUC **0.543**, not monotonic |
| Shimla district (Zenodo 10492992) | Himachal Pradesh | 3,176 landslides | AUC **0.472**, ordering **inverted** |
| Eastern Himalaya large landslides (Zenodo 18931430) | Bhutan / Arunachal / S Tibet | 420 points | not testable |

All three are CC BY 4.0 and load with the existing reader. Why each was
dropped:

**Nepal monsoon — redundant and weaker.** It is monsoon-triggered in Nepal,
which is what Far-Western Nepal already provides, at 499 polygons against
26,348, and without the mapped-extent polygon that makes the Far-West set
usable. Background over its bounding box therefore labels ground nobody
examined as landslide-free, so 0.543 is a lower bound that cannot be corrected.
It adds nothing the existing set does not do better.

**Shimla — the right answer to a different question.** The dataset accompanies
a paper on anthropogenic landslides, and its failures cluster along road cuts
and construction benches below the town. SINMAP has no term for an undercut
slope, so the inverted frequency ratio is what the mechanics predict rather
than a defect in the data. It cannot calibrate the model and it cannot fairly
validate it either, because it is not measuring what the model computes. The
*finding* is kept in the limits - the model has nothing to say about road-cut
failures around hill towns, and that is a large share of what kills people in
Himachal - but the data is not a calibration or validation source.

**Eastern Himalaya — wrong mechanism, wrong geometry.** 420 *large* landslides
as point locations. SINMAP describes shallow translational failure, and the
fitting needs mapped source areas. Neither condition holds.

### Still missing, and checked

- **Pakistan and Afghanistan.** The USGS earthquake-triggered ground-failure
  repository (Data Series 1064) catalogues three inventories for the 2005
  Kashmir earthquake - Sato 2007, Basharat 2014, Basharat 2016 - but hosts only
  their metadata; the geometry is not attached and its KML and WFS endpoints
  return 404. The same is true of the four Chinese events it lists.
- **The Indian Himalaya at scale.** Chen et al. (2024) mapped 265,000
  landslides across Jammu and Kashmir, Himachal, Uttarakhand, Sikkim and
  Arunachal, 1992-2021, which would be transformative. The repository holds a
  README only; the products are annual *density rasters* on Google Drive under
  CC BY-NC, not an inventory of features.
- **Bhutan.** Nothing open was found beyond the 420 Eastern Himalaya points.
- HR-GLDD and GDCLD are deep-learning image patches with binary masks, not
  geolocated inventories, and cannot fit or validate anything here.

So the calibration and validation base is unchanged: Gorkha, Far-Western Nepal
and Sikkim. That is three areas, all in the eastern half of the region, and it
is the single biggest limit on what the regional product can claim.

## 13. The NASA catalogue is mostly too coarse to use

The NASA Global Landslide Catalog publishes a ``location_accuracy`` field, and
in this region most of it fails a 90 m test:

| Screen | Records in the HKH |
|---|---|
| none | 2,469 |
| accuracy <= 1 km | **367** |
| accuracy "exact" | 39 |
| <= 1 km and rainfall-triggered | 295 |

**85 % of the catalogue's HKH records are placed worse than 1 km**, many to a
district or a state. The loader screens to 1 km automatically for any CSV that
publishes an accuracy field, so passing the GLC export as an inventory now
yields 367 usable records rather than 2,469 mostly-unusable ones. Before that
fix the screening function existed and nothing called it.

367 records over 4,400 km of mountain range is not a calibration set. Use the
catalogue to see where landslides get *reported*; do not fit to it.
