# Methods

A technical explanation of what H-SIM computes and why, in pipeline order:
terrain, stability, calibration, climate, exposure. Measured performance
lives in [RESULTS.md](RESULTS.md); this document explains the machinery that
produced it. Section references of the form `module.function` point into
`h_sim/`.

## 1. Scope and model class

H-SIM maps relative susceptibility to **shallow translational landsliding**
- failures of the soil mantle on a plane roughly parallel to the surface -
and screens the settlements and roads positioned to be reached by them. It is
a SINMAP-class model (Pack, Tarboton & Goodwin 1998): an infinite-slope limit
equilibrium coupled to steady-state TOPMODEL hydrology, made probabilistic by
Monte Carlo over uncertain soil parameters, and extended here with
pseudo-static seismic loading, CMIP6 recharge scenarios, and an
angle-of-reach exposure screen.

Three framing decisions govern everything downstream:

* **Relative, not absolute.** The output probability is the chance that a
  parameter draw makes a pixel's factor of safety fall below one - a
  statement about ranking under parameter uncertainty, not an annual
  frequency. Compare pixels to pixels and assets to assets; never read a
  value as "a 30 % chance of a landslide".
* **One failure mode.** Deep-seated rotational slides, rockfall, and
  road-cut collapse obey different mechanics; the model does not claim them,
  and where an inventory is dominated by them it measurably fails
  (RESULTS §12: Shimla, AUC 0.472 with the ordering inverted).
* **Defaults are earned.** Every refinement beyond the published SINMAP is
  either validated on held-out ground or ships switched off next to the
  experiment that will decide it (§10).

## 2. Input data

| Layer | Source | Role |
|---|---|---|
| Elevation | Copernicus GLO-30 / GLO-90 DEM | slope, flow routing, reach geometry |
| Precipitation | WorldClim v2.1 monthly climatology | recharge (wettest month) |
| Futures | WorldClim downscaled CMIP6 | recharge scaling per SSP and period |
| Landslides | Roback et al. (Gorkha 2015), Far-Western Nepal, Sikkim | calibration and validation |
| Admin units | geoBoundaries ADM1 | the regional sweep's units |
| Exposure | OSM places and highways (Overpass) | settlements and road segments |

Calibration runs at 30 m (GLO-30); the 95-province regional sweep at 90 m,
a cost-driven compromise whose price is measured, not assumed (RESULTS §1:
spatial-block AUC falls 0.816 → 0.728 from 30 m to 250 m, making resolution
worth more than any modelling choice tested).

## 3. Terrain processing (`model/hydrology.py`)

The DEM is depression-filled (priority-flood), then routed with **D-infinity**
(Tarboton 1997): each cell's flow direction is the steepest descent across
eight triangular facets, and its discharge is split between the two neighbours
bracketing that angle in proportion to angular closeness. Accumulation
processed in descending elevation order yields contributing area; divided by
the mean cell width it becomes the **specific catchment area** `a` (m), the
per-unit-contour drainage the wetness term requires - used rather than total
area because total area scales with cell size and would make the model
resolution-dependent by construction.

Two derived quantities added for the exposure screen reuse this machinery:

* `dinf_angles` - the flow-direction grid alone, skipping accumulation.
* `dinf_dependence` - Tarboton's dependence map, the reverse traversal: the
  fraction of each cell's flow that passes through a target patch, computed
  on a window in ascending elevation order (each cell's two receivers are
  strictly lower, so they are always resolved first). This powers the
  connectivity weighting of §7.5.

## 4. The stability model (`model/physical.py`)

### 4.1 Factor of safety

For a soil column on an infinite slope of gradient `tan θ`, with relative
wetness `w`, dimensionless cohesion `C`, friction angle `φ`, water-to-soil
density ratio `r` (0.5), and horizontal seismic coefficient `k_h`:

```
FS = [ C + (cos θ − k_h sin θ − w · r · cos θ) · tan φ ]
     / ( sin θ + k_h cos θ )
```

`k_h = 0` recovers the static SINMAP expression. Flat ground with no seismic
driving force is reported unconditionally stable rather than as a division
blow-up. Wetness is steady-state TOPMODEL:

```
w = min( (R/T) · a / sin θ , 1 )
```

with `R/T` (recharge over transmissivity, 1/m) a single fitted parameter -
recharge and transmissivity are not separately identifiable from topography,
so the model does not pretend to separate them. `C` is likewise the lump
`(C_root + C_soil)/(h ρ_s g)`: cohesion is only identifiable jointly with
soil depth `h` (see §4.4).

### 4.2 From FS to probability

`failure_probability` draws `n` parameter triples `(C, φ, R/T)` uniformly
from fitted ranges and reports, per pixel, the fraction of draws with
`FS < 1`. One draw is applied to every pixel per pass - that is what the
marginal per-pixel probability requires, it keeps the cost to `n` passes over
the block, and it has one statistical consequence used later: failures across
pixels are **strongly positively dependent** (they share the draw), which is
why the exposure screen aggregates by expectation rather than by an
independence union (§7.4). SINMAP's six stability classes are also emitted,
from the best-case-dry and worst-case-wet corners of the ranges.

### 4.3 Seismic term

Two uses of the same mechanics. Scenario maps apply a fixed pseudo-static
`k_h` (0.15 g, 0.35 g). Independently, `critical_acceleration` solves
`FS = 1` for `k_h`, giving the Newmark critical acceleration `k_c` per pixel
- negative where static conditions already fail, which is reported as the
honest answer. RESULTS §5 shows conclusions are sensitive to the seismic
conventions in a way they are not to the rainfall ones; treat seismic
scenario maps accordingly.

### 4.4 Slope-dependent soil depth (off by default)

Regolith thins as slopes steepen - an exponential decline with gradient in
steep soil-mantled terrain (DeRose 1996), used as distributed effective depth
`h = h₀ · exp(−k · tan θ)` since Saulnier et al. (1997; cf. Catani et al.
2010). The model never sees `h` itself, only the two parameters that carry
`1/h`, so a single per-pixel factor

```
f = exp( depth_k · (min(tan θ, tan 60°) − tan 30°) )
```

scales dimensionless cohesion and `R/T` **together**: thinner soil on steeper
ground is at once relatively more root-bound and less transmissive - one
physical quantity, two coupled effects. The anchor at 30° keeps fitted ranges
meaning what they mean on a typical failing slope; the cap at 60°
acknowledges that regolith-thinning laws were measured on soil-mantled
slopes and have nothing to say about rock faces. `depth_k = 0` (the published
SINMAP, and the shipped default) costs nothing; candidates (1.0, 2.5)
bracket DeRose-type decline rates and enter the fit only through the
augmented grid of §5.2, gated on `analysis/09_soil_depth.py`.

## 5. Calibration and validation (`model/physical.py`, `model/crossval.py`)

### 5.1 Sampling

Presence points come from mapped landslide inventories; background points are
drawn only **inside each inventory's published survey extent**, because
background stands in for "terrain that did not fail" and drawing it on
unmapped ground silently labels unsurveyed terrain landslide-free (Far-West
and Sikkim survey only ~60 % of their own bounding boxes).

### 5.2 Parameter search

The physics fixes the form of the response; the region supplies the values.
48 candidate range-triples spanning soil-mantled mountain terrain (cohesion
caps 0.05-0.40; friction bands 25-45°; `R/T` spanning four decades with a
fixed 50:1 range ratio) are each run through the full Monte Carlo and scored
by AUC - how well the resulting probability ranks landslides above
background. The augmented grid (144 candidates) adds the depth axis with
`k ∈ {0, 1.0, 2.5}`; zero is always searched, so the augmented grid can
reject the depth term by choosing it.

### 5.3 Cross-validation and transfer

In-sample AUC flatters any fitted model, and ordinary random CV still leaks
spatial autocorrelation. The reported figure is **spatial-block CV**: the
area is tiled into 0.25° blocks, whole blocks are withheld, and the parameter
search is rerun inside every fold - so no test point ever influenced the
parameters that score it. Transfer is the harder test: parameters fitted on
Gorkha applied unchanged 300-400 km away cost only 0.004-0.022 AUC
(RESULTS §3), which is the model's main claim to usefulness; the physics
buys portability, not discrimination (it ties logistic regression given the
same inputs, RESULTS §2).

The one spatial refinement tried - separate fits per GLiM lithology class -
measured **−0.0004** held-out AUC (RESULTS, `analysis/06`) and ships off.
That result is the template for every gate in §10.

## 6. Climate scenarios (`model/climate.py`)

Recharge enters as wettest-month precipitation - the season the soil column
is closest to the steady-state assumption - normalised to a dimensionless
per-pixel scale centred on 1 at the reference climate, multiplying `R/T`.
Futures replace the climatology with WorldClim's downscaled CMIP6 ensemble
for SSP2-4.5 and SSP5-8.5 over 2021-2040 and 2041-2060: two pathways because
the spread between them is the honest statement of scenario uncertainty, two
windows because the trajectory across the planning horizon is the question.
Every settlement and road segment is scored under every scenario;
RESULTS §11 finds the near-term signal small against the model's own error.

## 7. The exposure screen (`model/risk.py`)

### 7.1 Why not sample the map at the asset?

Settlements sit on flat valley floors; what threatens them arrives from
above. Sampling susceptibility at a town's coordinates reports "safe" for
exactly the towns most at risk.

### 7.2 Angle of reach

Every cell within 2 km whose line of sight to the asset is steeper than
`α = 18°` is a potential source - the empirical mobility criterion for
debris flows (Corominas 1996; Rickenmann 1999): material rarely travels
further than the line dipping `α` from source to toe. The **reaching score**
is the weighted mean failure probability over those sources with weight
`1/d` (a debris fan widens roughly with distance, so the share a fixed-width
target occupies falls as `1/d`), floored by the asset's own on-site
probability:

```
score = max( p_on_site , Σ wᵢ pᵢ / Σ wᵢ )
```

A weighted **mean** rather than a maximum because, over thousands of source
cells, the max saturates (an earlier build put 56 % of Gorkha settlements in
the top band). Bands at 0.02 / 0.08 / 0.20 / 0.40 are legend conveniences
over a continuous score - "moderate" is the quantitative statement that a
fifth of the ground positioned to reach you is called unstable - and
"exposed" means score ≥ 0.08 throughout the products.

### 7.3 Settlement footprints and the worst sector

OSM place nodes are centroids; the hillside ward a reach score exists for
sits at the edge. Settlements are therefore scored over every cell of a disc
scaled by place type (city 1 000 m, town 500, village 250, hamlet 100 - a
stated heuristic standing in for mapped built-up extent such as GHS-BUILT,
which would replace it behind the same interface) and headlined at the
**90th-percentile cell**: the exposed edge of town rather than its safe
centre, and deliberately not the max, which is the saturation artefact
above. The whole headline cell's record is reported so its sector, worst
source and supply all describe one place. Every score also carries the
compass octant whose sources hold the highest weighted mean - "the threat is
from the NE" - so a field visit knows which slope to walk.

### 7.4 Expected delivering area

The mean of §7.2 cannot distinguish thirty unstable source cells from three
thousand at the same mean. Alongside it, each asset reports

```
E[delivering area] = Σ pᵢ · A_cell        (m²)
```

the expected unstable area positioned to reach it. An expectation is used
deliberately: the shared parameter draw (§4.2) makes per-pixel failures
strongly positively dependent, so the independence union `1 − Π(1 − pᵢ)`
is not merely approximate but invalid, while expectation is linear under any
dependence structure. Like everything else it is relative - compare between
assets.

### 7.5 Connectivity weighting (off by default)

The cone of §7.2 is omnidirectional, but the `α` values it borrows were
measured on **channelised** flows, and regional runout practice (Flow-R,
Horton et al. 2013; spreading after Holmgren 1994) propagates debris along
drainage. When enabled, each source's weight becomes

```
w = (1/d) · ( floor + (1 − floor) · dep )
```

where `dep` is the D-infinity dependence of §3 - the fraction of the
source's flow routed through the target's 3×3 patch - and `floor` (0.2)
preserves unchannelised near-field delivery, which follows no channel and
must not be zeroed. A settlement at a gully mouth then outranks one under a
planar slope with the same cone statistics. Gated on
`analysis/08_connectivity.py`: polygon **toes** (the lowest cell of each
whole-landslide polygon - a place debris demonstrably arrived) versus
background, cone against connectivity, adopted only if the AUC gain clears
the background-resampling noise.

## 8. Road failure mechanisms (`model/risk.py`)

Roads are cut into 500 m segments; each inherits its worst vertex's reach
score - a stretch of road is closed by its most exposed point. That covers
one of the three ways Himalayan roads are actually lost, so each segment
carries three columns, the last two honestly labelled as **terrain
geometry, not model output**:

* **from_above** - the reach score: burial by material arriving from
  upslope. Per climate scenario.
* **cut_slope** - the adjacent upslope gradient (one grid cell, so ~30-90 m)
  exceeds 35°. Where a road traverses such ground it does so on a cut, and
  cut faces oversteepen the very slope the infinite-slope model assumes
  undisturbed; measured on a road-cut-dominated inventory that assumption
  inverts (RESULTS §12, Shimla). The flag marks where the model's blind spot
  is; it does not score it.
* **washout** - the segment touches a cell whose specific catchment area
  exceeds 5 000 m: a channel. Crossings there are taken by debris flows and
  scour arriving *along the channel*, from far beyond any local search
  radius.

On the web maps, colour is exposure and shape is mechanism (triangle
cut-slope, diamond washout), so a red diamond reads as: a channel crossing
fed by ground the model calls unstable.

## 9. Regional sweep mechanics (`pipeline.py`, `input/admin.py`)

Provinces are processed independently at 90 m, each on its polygon's
bounding box **buffered by 0.05°** - catchments must be routed wider than
the ground being scored, and the sufficiency of that buffer is measured, not
assumed (RESULTS §6). Rasters are then clipped back to the polygon so no two
provinces claim the same border cells, and exposure is clipped the same way:
settlements are dropped before scoring and road segments after segmentation
(by midpoint) when they fall outside the polygon, so nothing is scored
against a neighbour's blanked ground, drawn floating past the border, or
counted twice across the sweep. Fitted parameters are regional (one Gorkha
fit, transferred - §5.3), never per-province.

## 10. Gated features and the adoption rule

Three refinements are physically motivated, fully implemented, and **off**:

| Feature | Switch | Experiment | Decision rule |
|---|---|---|---|
| Connectivity weighting (§7.5) | `connectivity_weighting` | `analysis/08` | AUC gain > background-draw noise |
| Slope-dependent depth (§4.4) | augmented fit grid | `analysis/09` | held-out gain > fold spread, both areas, nonzero `k` chosen |
| Calibration regions | `calibration_regions` | `analysis/06` | **decided: −0.0004, stays off** |

The rule is uniform and inherited from the lithology result: plausible is
not adopted; measured is. Each experiment holds points, background and folds
fixed and varies exactly one thing, and each prints its own verdict.

## 11. Limitations

Stated once here, enforced throughout the code and copy: probabilities are
relative; only shallow translational failure is modelled; soil depth is not
mapped (§4.4 is a slope proxy awaiting its measurement); all three
calibration inventories sit in the eastern half of the region, so western
skill is asserted by transfer, not measured; road cut-slope and washout are
flags, not scores; the exposure screen is a reach criterion, not a runout
model - no volumes, velocities or specific paths; and there is no
vulnerability or damage function anywhere, so nothing in the outputs is
risk in the quantitative sense, which is why the products say "exposure"
and "priority" and never "expected loss".

## 12. References

* Pack, R. T., Tarboton, D. G., Goodwin, C. N. (1998). *The SINMAP approach
  to terrain stability mapping.* 8th Congress IAEG.
* Tarboton, D. G. (1997). A new method for the determination of flow
  directions and upslope areas in grid DEMs. *Water Resources Research* 33.
* Corominas, J. (1996). The angle of reach as a mobility index for small and
  large landslides. *Canadian Geotechnical Journal* 33.
* Rickenmann, D. (1999). Empirical relationships for debris flows.
  *Natural Hazards* 19.
* Horton, P., Jaboyedoff, M., Rudaz, B., Zimmermann, M. (2013). Flow-R, a
  model for susceptibility mapping of debris flows and other gravitational
  hazards at a regional scale. *NHESS* 13.
* Holmgren, P. (1994). Multiple flow direction algorithms for runoff
  modelling in grid based elevation models. *Hydrological Processes* 8.
* DeRose, R. C. (1996). Relationships between slope morphology, regolith
  depth, and the incidence of shallow landslides. *Zeitschrift für
  Geomorphologie* Suppl. 105.
* Saulnier, G.-M., Beven, K., Obled, C. (1997). Including spatially variable
  effective soil depths in TOPMODEL. *Journal of Hydrology* 202.
* Catani, F., Segoni, S., Falorni, G. (2010). An empirical geomorphology-
  based approach to the spatial prediction of soil thickness at catchment
  scale. *Water Resources Research* 46.
* Newmark, N. M. (1965). Effects of earthquakes on dams and embankments.
  *Géotechnique* 15.
* Roback, K., et al. (2018). The size, distribution, and mobility of
  landslides caused by the 2015 Mw 7.8 Gorkha earthquake, Nepal.
  *Geomorphology* 301.
