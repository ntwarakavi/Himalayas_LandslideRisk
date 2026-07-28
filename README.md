# Hindu Kush Himalaya Landslide Hazard Model

A physically based slope-stability model for the Hindu Kush Himalaya:
Afghanistan, Pakistan, India, Nepal, Bhutan, Bangladesh, China and Myanmar.

Terrain is routed for flow with D-infinity methods after TauDEM, and stability
is evaluated with the SINMAP infinite-slope formulation extended with a
pseudo-static term for seismic loading. Soil parameters are fitted to mapped
landslide inventories and reported with spatial-block cross-validation.

Data is fetched only for the area of interest, processed in tiles where it can
be, and cached — nothing is downloaded twice.

## Model

### Definitions

| Term | Question answered | Status |
|---|---|---|
| Stability / susceptibility | Where is the ground close to failing? | Implemented |
| Hazard | Given a trigger of stated severity, how likely is failure? | Implemented |
| Risk | What are the expected consequences? | Not implemented |

Stability is a property of the terrain under the conditions the parameters were
fitted at. Hazard is conditional on a trigger the user specifies; the model does
not forecast triggers. Risk additionally requires exposure and vulnerability
data, which this package does not hold.

### Factor of safety

For a planar failure surface parallel to the ground, with the slide much wider
and longer than it is deep, the balance of driving and resisting forces on a
column of unit plan area is

```
FS = [ C + (cosθ − k·sinθ − w·r·cosθ) · tanφ ] / ( sinθ + k·cosθ )
```

| Symbol | Meaning |
|---|---|
| `C` | dimensionless cohesion, `(Cr + Cs) / (h·ρs·g)` — root plus soil, normalised by the weight of the soil column |
| `θ` | slope angle |
| `φ` | angle of internal friction |
| `r` | density ratio `ρw/ρs`, about 0.5 |
| `w` | relative wetness, the saturated fraction of the soil column |
| `k` | horizontal seismic coefficient; zero for rainfall triggering |

With `k = 0` this is SINMAP's published form. The seismic terms enter as an
extra driving force `k·W` along the slope and a matching reduction `k·W·sinθ` in
the normal force. Pore pressure is unaffected by inertia, which is why the `w`
term keeps its static form.

Wetness closes the system through a steady-state balance — recharge `R` falling
on the upslope contributing area `a` must pass through a soil column of
transmissivity `T`:

```
w = min( R·a / (T·sinθ), 1 )
```

Only the ratio `R/T` matters, which is convenient because it is far better
constrained than either term alone. Wetness is capped at 1: any excess becomes
overland flow rather than deeper saturation.

### Terrain hydrology

The wetness term needs upslope drainage, which `model/hydrology.py` supplies in
pure NumPy, following TauDEM:

| Stage | Method | Reference |
|---|---|---|
| Depression filling | Priority flood | Barnes, Lehman & Mulla 2014 |
| Flow direction | D-infinity, eight triangular facets | Tarboton 1997 |
| Contributing area | D-infinity accumulation, elevation-ordered | Tarboton 1997 |
| Specific catchment area | Contributing area ÷ contour width | — |

D-infinity rather than D8: the 45° quantisation of D8 produces artificial
parallel flow lines on exactly the planar hillslopes that shallow failure
occupies. Specific catchment area rather than total contributing area, so the
result does not scale with cell size.

### From factor of safety to a map

The three parameters are not known per pixel. SINMAP treats them as uniform over
plausible ranges and reports the probability that `FS < 1` — a continuous field
in [0, 1] that can be validated against an inventory like any other map. Two
regions appear as constants and are worth naming:

- **Unconditionally stable** — stable even fully saturated at the most
  pessimistic parameters. Probability 0.
- **Unconditionally unstable** — unstable even dry at the most optimistic
  parameters. Probability 1. Such terrain stands only through cohesion the model
  does not represent, or is actively eroding.

A six-class SINMAP stability map is also written. **Use the continuous field.**
The classes are a legend, and their lower three bands are not ordered by failure
probability — see [the class map](#the-class-map-is-not-fully-ordered).

### Triggering

In a physical model hazard is not a separate calculation. Each trigger reduces
to one scalar that the factor of safety already accepts:

- **Rainfall** raises recharge, so it enters as a multiplier on `R/T`. Under a
  Gumbel distribution of annual maximum daily rainfall, the ratio of a scenario
  depth to the reference depth cancels the location parameter and leaves a
  dependence on the coefficient of variation alone:
  `m(T) = [1 + cv·k(T)] / [1 + cv·k(T_ref)]`, with `k(T) = (√6/π)(y_T − γ)`.
- **Earthquakes** add an inertial force, entering as `k_h`. The pseudo-static
  coefficient is taken as half of PGA, the long-standing convention
  (Hynes-Griffin & Franklin 1984), since a sustained force stands in for a brief
  oscillation.

Two numbers in the model are not derived from the data here: the rainfall
coefficient of variation (default 0.30; station analyses put monsoon Asia at
0.25–0.35) and that PGA fraction. Both are single interpretable parameters whose
influence can be checked by rerunning at the ends of their range.

### Fitting

The physics fixes the *form* of the response; the inventory supplies the
parameter values. Ranges are searched over a 48-point grid spanning what is
reported for soil-mantled mountain hillslopes — cohesion from bare to
well-rooted, friction across 25–45°, `R/T` over four orders of magnitude — and
scored by how well the resulting failure probability ranks mapped landslides
above background.

What that constrains is less than the parameter list suggests, and the limits
matter:

- `R` and `T` are identifiable only as their ratio.
- Cohesion is identifiable only jointly with soil depth, since the model sees
  `C = (Cr + Cs)/(h·ρs·g)`. No soil-depth map is used.
- The absolute level of the probability depends on how background points were
  drawn. Differences between pixels are meaningful; the value at a pixel is not
  a frequency of failure.

**Calibration regions** (optional) are SINMAP's own answer to spatially varying
soils: a zoning within which the parameters are taken as uniform. Lithology
(GLiM) controls friction and soil cohesion; land cover (WorldCover) controls
root cohesion. A region is fitted separately only if it holds enough landslides
to constrain three parameters; the rest fall back to the whole-area fit.

## Installation

```bash
git clone https://github.com/ntwarakavi/Himalayas_LandslideRisk.git
cd Himalayas_LandslideRisk
./setup.sh
source .venv/bin/activate
```

If `rasterio` fails to build from source:

```bash
conda create -n hkh python=3.11 -c conda-forge rasterio fiona numpy matplotlib requests pytest
conda activate hkh
```

Offline verification, no downloads:

```bash
./scripts/run_demo.sh
```

## Workflow

Each step writes its outputs to disk and can be run independently.

| Step | Command | Produces |
|---|---|---|
| 1 | `step1-check` | Dataset availability report |
| 2 | `step2-download` | Cached source data |
| 3 | `step3-fit` | Soil parameters, cross-validated |
| 4 | `step4-stability` | Failure probability, stability classes, critical acceleration |
| 5 | `step5-hazard` | Failure probability under a trigger scenario |
| 6 | `step6-validate` | Validation against a held-out inventory |
| 7 | `step7-compare` | Difference between two runs |

### 1. Check data availability

```bash
python -m giri_landslide.cli step1-check
```

Reports each dataset as cached, reachable, blocked or manual-only, and totals
the outstanding download. `--offline` restricts the check to the local cache.

### 2. Download

```bash
python -m giri_landslide.cli step2-download --bbox 84.5 27.6 85.3 28.2
```

Cached files are skipped. Only the DEM and the precipitation climatology are
fetched by default; land cover and the 1.1 GB GLiM geodatabase are downloaded
only when `--calibration-regions` asks for them.

### 3. Fit

```bash
python -m giri_landslide.cli step3-fit \
    --name gorkha --bbox 84.5 27.6 85.3 28.2 --res 0.0025 \
    --inventory "data/raw/inventory/roback/Roback_Nepal_final_files/Source20170209.shp"
```

Writes `outputs/<name>_fitted_params.json`, which steps 4 and 5 read. The report
holds the parameter ranges, the recharge reference, both cross-validation
schemes and any data-quality warnings.

Two cross-validation schemes are reported. A **random** split assigns points to
folds independently; because landslides cluster and terrain is autocorrelated,
test points usually have training points on the same hillside, so the score
flatters the model. A **spatial-block** split (`cv_block_deg`, 0.25° by default)
assigns whole blocks to folds, so no test point has training data nearby. The
parameter search is rerun inside every fold, so neither figure is contaminated
by having chosen the parameters on the test points.

### 4. Stability

```bash
python -m giri_landslide.cli step4-stability \
    --name gorkha --bbox 84.5 27.6 85.3 28.2 --res 0.0025
```

Writes `outputs/<name>_susceptibility_prob.tif` (the field to use),
`<name>_susceptibility_class.tif` (SINMAP classes 1–6) and
`<name>_critical_acceleration.tif` (the Newmark yield coefficient in g, the
shaking needed to bring the slope to `FS = 1`). Slope and specific catchment
area go to `data/work/` for inspection.

### 5. Hazard

```bash
python -m giri_landslide.cli step5-hazard --name gorkha --return-period 100
python -m giri_landslide.cli step5-hazard --name gorkha --trigger earthquake --pga 0.35
```

### 6. Validate

```bash
python -m giri_landslide.cli step6-validate --name gorkha \
    --inventory data/raw/inventory/sikkim/Google_Earth_landslides_polygon_21Dec2021.shp
```

Reports the frequency ratio per class — the share of landslides in a class
divided by the share of map area it occupies — which must increase with class
for the ordering to be meaningful. The continuous field is binned into map-area
quintiles for the table; AUC is computed on the raw values. Multiple
`--inventory` paths are pooled.

### 7. Climate scenarios

```bash
python -m giri_landslide.cli step4-stability --name ssp585 \
    --climate ssp585 --climate-period 2061-2080
python -m giri_landslide.cli step7-compare \
    --baseline outputs/gorkha_susceptibility_prob.tif \
    --scenario outputs/ssp585_susceptibility_prob.tif --name climate
```

Substitutes downscaled CMIP6 projections for the recharge field: a wetter future
raises `R`, raises wetness, and lowers the factor of safety. The scenario field
is normalised by the **present-day** reference held in the fitted-parameters
file, so a uniform wetting shows up rather than cancelling out. The triggering
return period retains its present-day definition.

## Results

Full write-up, with all seven experiments and their scripts, in
[docs/RESULTS.md](docs/RESULTS.md). Headlines below. Every figure is
**spatial-block cross-validated** unless marked otherwise: whole 0.25° blocks
are withheld, and the parameter search is rerun inside each fold.

### Resolution is the single biggest factor

Gorkha, Roback inventory, identical points at every grid:

| Grid | Spatial CV AUC | ± | Capture top 20 % |
|---|---|---|---|
| 250 m | 0.7281 | 0.0328 | 58.8 % |
| 90 m | 0.8059 | 0.0321 | 69.0 % |
| **30 m** | **0.8156** | 0.0204 | 71.0 % |

The mechanism is visible in the wetness term: median specific catchment area
falls from 631 m to 132 m as the grid refines, because a 250 m cell is wider
than the hollows that concentrate subsurface flow. Most of the gain is realised
by 90 m. Fitted parameters at 30 m: `C ∈ [0, 0.25]`, `φ ∈ [25°, 35°]`,
`R/T ∈ [1×10⁻⁵, 5×10⁻⁴] m⁻¹`.

### It ties statistics, and that is the interesting part

Same points, same folds, every model refitted per fold, Gorkha at 30 m:

| Model | Random CV | **Spatial CV** | Drop | Fold ± |
|---|---|---|---|---|
| SINMAP (physics) | 0.8222 | **0.8161** | 0.006 | 0.021 |
| logistic — slope + log SCA | 0.8275 | **0.8220** | 0.006 | 0.022 |
| random forest — slope + log SCA | 0.8180 | 0.8063 | 0.012 | 0.019 |
| logistic — + lithology, cover, elevation, precip | 0.8815 | 0.8068 | 0.075 | 0.067 |
| random forest — + lithology, cover, elevation, precip | **0.9182** | 0.8185 | **0.100** | 0.051 |

Given the same two predictors the mechanical model and a logistic regression are
indistinguishable. The headline is the last column but one: a random forest with
the usual extra predictors reports 0.918 under random CV — unremarkable by the
standards of the literature — and 0.819 under spatial-block CV. That 0.10 was
memorised geography, and buying it also triples the between-place variance.

### The parameters transfer; the model does not work everywhere

| Area | Parameters | Landslides | AUC |
|---|---|---|---|
| Gorkha | fitted here | 5,193 | 0.8221 |
| Far-West Nepal | transferred from Gorkha | 25,679 | 0.6557 |
| Far-West Nepal | fitted here | 25,679 | 0.6595 |
| Sikkim | transferred from Gorkha | 255 | 0.7801 |
| Sikkim | fitted here | 255 | 0.8019 |

Refitting locally buys 0.004–0.022 AUC, so the fitted parameters are portable.
But skill in Far-Western Nepal is 0.16 lower than at Gorkha, and refitting does
not recover it — so the limit is the predictors, not the parameters. Slope and
topographic wetness separate Gorkha's failures well and Far-West's poorly, which
is what to expect where bedding, weak mudstone horizons and road cuts control
failure. **The model suits shallow translational failure on soil-mantled
crystalline terrain, and suits weak sedimentary hill country considerably less
well.** Only a local inventory tells you which case you are in.

An appealing hypothesis — that Gorkha's earthquake trigger caps what a static
map can explain, so a monsoon inventory should score higher — was tested and
**refuted**: the monsoon inventory scores lower. See
[docs/RESULTS.md §4](docs/RESULTS.md) for the two candidate explanations that
were tested and largely ruled out.

### Trigger conventions

Of the two numbers that are neither mechanics nor fitted, one is inert and one
is not. Over their plausible ranges the rainfall coefficient of variation moves
AUC by 0.003 and the unstable area by two points; the pseudo-static fraction of
PGA moves the unstable area from 57 % to 90 % and degrades the ranking itself.
**Quote a seismic scenario as a range over `pga_fraction`; a rainfall scenario
needs no such hedge.**

### What did not work

- **Lithology calibration regions**: −0.0004 AUC held out, in both Gorkha and
  the geologically varied Far-West. Off by default. The result is specific to
  GLiM level-1 zoning, which collapses Far-West's dozen named formations into
  five classes.
- **Spatial recharge**: +0.005 AUC, inside the fold spread. Left on as the
  physically correct treatment; `--uniform-recharge` disables it.
- **Crown rather than centroid sampling** of polygon inventories: +0.017 for
  whole-landslide polygons, −0.006 where polygons are already source areas.
  Real, correctly signed, second-order.

### The class map is not fully ordered

The continuous field is monotonic; the six-class SINMAP map is not, and cannot
be. Classes 1–3 all have failure probability zero by definition and differ only
in margin of stability, so landslide density cannot order them. Classes 4–6 are
ordered correctly, and class 6 is the strongest single signal in the map — 0.75 %
of the area holding 8.8 % of the landslides, a frequency ratio of 11.7. **Use
the continuous field**; the classes are a legend.

## Repository layout

```
giri_landslide/
├── config.py              Run configuration and region definitions
├── pipeline.py            Stage orchestration
├── cli.py                 Command-line workflow
├── input/
│   ├── datasets.py        Dataset registry, cache and availability checks
│   ├── sources.py         Terrain and climate downloaders
│   └── inventory.py       Landslide inventories: fetch, load, sample
├── model/
│   ├── hydrology.py       Depression filling, D-inf flow, catchment area
│   ├── physical.py        SINMAP stability, parameter fitting, cross-validation
│   ├── hazard.py          Trigger scenarios: recharge and seismic coefficient
│   ├── crossval.py        Random and spatial-block fold assignment
│   ├── validate.py        Held-out validation
│   └── risk.py            Not implemented; scope documented
└── utility/
    ├── grid.py            Reference grid, warping, tiled processing
    └── demo.py            Synthetic inputs for offline testing

analysis/                  Experiments behind docs/RESULTS.md
├── common.py              Shared sampling, folds and survey masking
├── 01_resolution.py       Grid resolution and the wetness term
├── 02_transfer.py         Applying one area's parameters to another
├── 03_benchmark.py        Against logistic regression and random forest
├── 04_monsoon.py          Triggering mechanism and skill
├── 05_sensitivity.py      The two unfitted conventions
├── 06_inventory_geometry.py  Where a polygon inventory should be sampled
├── 07_calibration_regions.py Whether per-lithology parameters help
└── results/               JSON output, version-controlled

configs/                   Run configurations
docs/                      Operating guide and measured results
scripts/                   Offline demonstration
tests/                     Test suite, no network required
data/, outputs/            Generated, not version-controlled
```

## Data

| Role | Dataset | Resolution | Acquisition |
|---|---|---|---|
| Terrain | Copernicus GLO-30 DEM | 30 m | Automatic |
| Recharge | WorldClim v2.1 monthly precipitation | 1 km | Automatic, 1.0 GB once |
| Future climate | WorldClim CMIP6 | 4.6 km | Automatic |
| Calibration regions (optional) | GLiM, 1.2 M polygons | Vector | Automatic, 1.1 GB once |
| Calibration regions (optional) | ESA WorldCover 2021 | 10 m | Automatic |
| Inventories | Gorkha, Far-West Nepal, Sikkim, NASA GLC | Points, polygons | Automatic |
| Earthquake PGA | GEM seismic hazard map | — | Manual, or scenario value |

### Inventory selection

| Inventory | Records | Mapping | Suitability |
|---|---|---|---|
| Roback Gorkha, Nepal | 24,795 | Satellite, earthquake-triggered | Best available; the reference fit |
| Far-Western Nepal | 26,350 | Satellite, multi-temporal | Rainfall-driven fit |
| Southern Sikkim, India | 255 | Satellite | Validation |
| NASA GLC | 11,033 global | Media reports | Screen by `location_accuracy`; only 32 % are placed to 1 km or better |

The physical model has no trigger-specific parameters, so an inventory from one
mechanism can fit a model applied to another — but the wetness the parameters
absorb reflects the conditions during the mapped events, so matching the
mechanism is still preferable. Validate against a local inventory before relying
on a map somewhere new.

### Resolution

**This is the most consequential choice a user makes**, and it is measured:
spatial-block AUC runs 0.728 at 250 m, 0.806 at 90 m, 0.816 at 30 m. Flow
convergence is what the wetness term depends on, and it is exactly what a coarse
grid smooths away.

The default DEM is Copernicus GLO-30, but the grid set by `--res` is what
actually matters: at `--res 0.0025` (~250 m) a 30 m DEM is simply downsampled
and the source is irrelevant. The package default of `0.00083333` (~90 m) is a
deliberate compromise — it captures nine tenths of the benefit at a ninth of the
cost of 30 m, and flow routing is not tiled, so cost rises steeply.

Specific catchment area is resolution-sensitive by construction, so **refit
after changing `--res`**; parameters fitted at one grid do not describe another.

## Status

Implemented and tested:

- D-infinity flow routing, verified against analytic cases
- SINMAP failure probability and stability classes
- Pseudo-static seismic loading and Newmark critical acceleration
- Soil parameter fitting with random and spatial-block cross-validation
- Rainfall and earthquake trigger scenarios
- Present and CMIP6 future-climate recharge
- Held-out validation, and the seven experiments in `analysis/`

Outstanding:

1. **Flow routing is not tiled.** Contributing area is a property of the whole
   drainage network, so the model holds the area of interest in memory.
   Basin-wise decomposition is the standard remedy and is not implemented; the
   CLI warns above 40 million cells.
2. **Soil depth is not mapped.** Cohesion is fitted as `C = (Cr + Cs)/(h·ρs·g)`,
   so depth cannot be separated from cohesion. A soil-depth model would
   constrain both independently.
3. **Two trigger parameters are conventions, not fits** — the rainfall
   coefficient of variation and the PGA fraction. The rainfall one is measurably
   inert; the PGA fraction is not, and moves the unstable area from 57 % to 90 %
   across its plausible range. Quote seismic scenarios as a range over it.
4. **Domain of validity.** Skill is 0.816 on soil-mantled crystalline terrain at
   Gorkha and 0.656 in the weak sedimentary hill country of Far-Western Nepal,
   and refitting locally does not recover the difference. There is nothing in a
   fitted parameter set that warns which case an area is; only a local inventory
   does.
5. **Calibration regions do not work as implemented** (−0.0004 AUC held out,
   including where the geology is varied). Off by default; the result is
   specific to GLiM level-1 zoning.
6. **Risk.** No exposure or vulnerability data, and no runout model — the
   stability model says where material detaches, not where it arrives. Scope
   documented in `model/risk.py`.
7. **Inventory coverage.** Nepal and Sikkim only. Pakistan, Afghanistan,
   Uttarakhand, Himachal, Bhutan and Myanmar have no usable inventory, so
   parameters there are extrapolated.
8. **Region-wide execution.** The HKH at 30 m is far beyond a single run and
   must be tiled by basin; no tiling driver exists.

## Testing

```bash
python -m pytest tests/ -q
```

The mechanics have exact answers in limiting cases, and the tests check against
those rather than against stored expectations: `FS = 1` when the slope angle
equals the friction angle, mass conservation in flow routing, the saturated
critical angle `atan((1−r)tanφ)`, and that applying the critical acceleration
drives `FS` to exactly 1.

## Licence

MIT. Source datasets retain their own licences (Copernicus, ESA, WorldClim,
GLiM, GEM, NASA, Zenodo records); cite them accordingly.

## References

- Pack, R. T., Tarboton, D. G., Goodwin, C. N. (1998). The SINMAP approach to
  terrain stability mapping. *8th Congress of the IAEG*.
- Tarboton, D. G. (1997). A new method for the determination of flow directions
  and upslope areas in grid digital elevation models. *Water Resources Research*
  33(2), 309–319.
- Barnes, R., Lehman, C., Mulla, D. (2014). Priority-flood: an optimal
  depression-filling and watershed-labeling algorithm for digital elevation
  models. *Computers & Geosciences* 62, 117–127.
- Hynes-Griffin, M. E., Franklin, A. G. (1984). Rationalizing the seismic
  coefficient method. *US Army Engineer Waterways Experiment Station*, MP GL-84-13.
- Roback, K. et al. (2018). The size, distribution, and mobility of landslides
  caused by the 2015 Mw 7.8 Gorkha earthquake, Nepal. *Geomorphology* 301, 121–138.
