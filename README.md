# Hindu Kush Himalaya Landslide Susceptibility Model

An implementation of the GIRI landslide model (Palau, Nadim, Paulsen &
Storrøsten, 2023, NGI / CDRI —
[manuscript](https://giri.unepgrid.ch/sites/default/files/2023-06/20230615-NGI_manuscript_GIRI_landlside_hazard_model.pdf)),
scoped to the Hindu Kush Himalaya: Afghanistan, Pakistan, India, Nepal, Bhutan,
Bangladesh, China and Myanmar.

Two susceptibility models are provided: the manuscript's heuristic index, and a
physically based slope-stability model (SINMAP) driven by D-infinity flow
routing. On the Gorkha test area the physical model is the stronger of the two
(see [Model comparison](#model-comparison)).

The current scope is **susceptibility**. Hazard is implemented but depends on an
uncalibrated parameter (see [Status](#status)); risk is not implemented.

Data is fetched only for the area of interest, processed in tiles, and cached —
nothing is downloaded twice.

## Model

### Definitions

| Term | Question answered | Status |
|---|---|---|
| Susceptibility | Where is the terrain predisposed to fail? | Implemented |
| Hazard | Given a trigger of stated severity, how likely is failure? | Implemented; rainfall matrix uncalibrated |
| Risk | What are the expected consequences? | Not implemented |

Susceptibility is a property of the terrain, independent of any triggering
event. Hazard is conditional on a trigger the user specifies; the model does not
forecast triggers. Risk additionally requires exposure and vulnerability data,
which this package does not hold.

### Susceptibility formulation

Four conditioning factors are derived from open data and reclassified to ordinal
scores:

| Factor | Source | Score |
|---|---|---|
| Slope | Copernicus GLO-90 DEM, Horn method | 0–5 |
| Lithology | GLiM level-1 classes | 0–3 |
| Land cover | ESA WorldCover 2021 | 0–5 |
| Soil moisture | WorldClim wettest-month precipitation | 1–5 |

The slope score is non-monotonic: it rises to a maximum near 30–36° and falls
above it, since slopes steeper than the internal friction angle of most soils
have already shed their regolith.

Factors are combined by the logistic model the calibration fits:

```
P = 1 / (1 + exp(-(b + Σ wᵢ · log(fᵢ + 1))))
```

Two predictor sets are available, selected by `feature_mode`:

- `continuous` (default) — slope and precipitation enter at full precision.
  Slope is represented by a quadratic, so the data may place a peak rather than
  having one imposed. Lithology and land cover stay categorical, because they
  are.
- `ordinal` — the manuscript's formulation, fitted on `log(score + 1)` of the
  four integer factor scores.

The ordinal set makes the index inherit the coarseness of its inputs: four
integer scores admit few distinct products, so large groups of pixels tie and
equal-area bins cannot be formed. Measured over Sikkim, ordinal features put
51 % of the map in a single bin; continuous features give bins of 20.1, 20.1,
19.5, 20.1 and 20.2 %.

The output is a continuous index in [0, 1]. Flat terrain and open water are
constrained to zero. A five-class map can also be produced (`--output classes`)
for compatibility with the manuscript's hazard matrix, which is indexed by
class.

The index is **relative**. It is fitted against background points standing in
for absences, so its level reflects the background sampling rather than observed
landslide frequency. Differences between pixels are meaningful; absolute values
are not failure probabilities.

### Calibration

The manuscript does not publish its factor weights. They are estimated here from
mapped landslide inventories by logistic regression on `log(f + 1)`, which makes
the fitted coefficients the factor weights directly. Two further tables can be
fitted from the same data:

- **Slope classes**, by frequency ratio per slope bin.
- **Lithology scores**, by frequency ratio per rock type. The expert default is
  a poor fit in the Himalaya: four of the seven classes present in the Gorkha
  area disagree with it, two of them inverted.

### Physically based formulation

The heuristic index scores terrain by correlation. The alternative is to state
the mechanics directly. `model/physical.py` implements SINMAP (Pack, Tarboton &
Goodwin 1998): an infinite-slope factor of safety closed by a TOPMODEL
steady-state wetness term.

```
FS = [ C + cosθ · (1 − w·r) · tanφ ] / sinθ        w = min( R·a / (T·sinθ), 1 )
```

`C` is dimensionless cohesion (root plus soil, normalised by the weight of the
soil column), `φ` the friction angle, `r ≈ 0.5` the water/soil density ratio,
`a` the specific catchment area and `R/T` the recharge-to-transmissivity ratio.
Only the ratio `R/T` is identifiable from a static map, and `C` only jointly
with soil depth.

The wetness term needs upslope drainage, which `model/hydrology.py` supplies
following TauDEM, in pure NumPy:

| Stage | Method | Reference |
|---|---|---|
| Depression filling | Priority flood | Barnes, Lehman & Mulla 2014 |
| Flow direction | D-infinity, eight triangular facets | Tarboton 1997 |
| Contributing area | D-infinity accumulation, elevation-ordered | Tarboton 1997 |
| Specific catchment area | Area ÷ contour width | — |

D-infinity rather than D8 because the 45° quantisation of D8 produces
artificial parallel flow lines on exactly the planar hillslopes that shallow
failure occupies.

The three parameters are not known per pixel, so SINMAP treats them as uniform
over plausible ranges and reports the probability that `FS < 1` — a continuous
field directly comparable with the heuristic index. Ranges are fitted by grid
search (48 combinations of cohesion, friction and `R/T`), scored by how well the
resulting failure probability ranks mapped landslides above background. On
Gorkha the fit selects `C ∈ [0, 0.4]`, `φ ∈ [25°, 35°]`, `R/T ∈ [1e-5, 5e-4]`.

Two regions appear as constants on the map and are worth naming: terrain stable
even when fully saturated at the worst parameters (probability 0), and terrain
unstable even when dry at the best parameters (probability 1) — the latter
stands only through cohesion the model does not represent, or is actively
eroding.

Flow accumulation is global to the drainage network, so this stage runs over the
whole area of interest at once rather than in tiles.

### Model comparison

Both models run on the same area (84.5–85.3° E, 27.6–28.2° N at 0.0025°), scored
against the same 5,193 Roback Gorkha landslides and the same background sample.
Capture is the share of landslides falling in the worst-ranked fraction of map
area:

| Model | AUC | Worst 5 % | Worst 10 % | Worst 20 % |
|---|---|---|---|---|
| Heuristic, logistic on log factors | 0.663 | 18.9 % | 30.0 % | 44.3 % |
| Physical, SINMAP | **0.748** | **32.2 %** | **45.1 %** | **60.0 %** |

The physical model separates better at every threshold. The heuristic index sees
slope but not convergence; SINMAP's wetness term distinguishes a planar hillside
from a hollow of the same gradient, and mapped landslides sit in the hollows.

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
| 3 | `step3-calibrate` | Fitted weights, slope and lithology tables |
| 4 | `step4-susceptibility` | Susceptibility index (heuristic) |
| 4b | `step4b-physical` | Failure probability and stability classes (SINMAP) |
| 5 | `step5-hazard` | Scenario hazard probability |
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
python -m giri_landslide.cli step2-download --bbox 84.0 28.0 84.6 28.5
```

Cached files are skipped. The first run fetches approximately 2.2 GB, most of it
the global GLiM and WorldClim files; subsequent areas require only their own DEM
and land-cover tiles.

### 3. Calibrate

```bash
python -m giri_landslide.cli step3-calibrate \
    --bbox 84.1 27.1 86.95 28.75 --res 0.0025 --fit-lithology \
    --inventory "data/raw/inventory/roback/Roback_Nepal_final_files/Source20170209.shp"
```

Writes `outputs/<name>_calibrated_config.json` for use in step 4, and a report
containing the weights, cross-validated AUC and any data-quality warnings.

Two cross-validation schemes are reported. A **random** split assigns points to
folds independently; because landslides cluster and terrain is autocorrelated,
test points usually have training points on the same hillside, so the score
flatters the model. A **spatial-block** split (`cv_block_deg`, 0.25° by default)
assigns whole blocks to folds, so no test point has training data nearby.

Measured on Gorkha, the two give almost the same mean — 0.726 random against
0.724 spatial — so the relationship does hold on ground the fit has not seen.
The difference is in the spread: fold-to-fold standard deviation is 0.004 under
the random split and **0.055** under the spatial one, with block AUC ranging
from 0.635 to 0.802. The mean therefore describes no particular place, and the
spatial spread is the range to expect when applying the map somewhere new.

### 4. Susceptibility

```bash
python -m giri_landslide.cli step4-susceptibility \
    --name nepal --bbox 84.0 28.0 84.6 28.5 --res 0.0008333 \
    --config outputs/nepal_calibrated_config.json
```

Writes `outputs/<name>_susceptibility_prob.tif`, and the four factor rasters to
`data/work/` for inspection.

### 4b. Physically based stability

```bash
python -m giri_landslide.cli step4b-physical \
    --name gorkha --bbox 84.5 27.6 85.3 28.2 --res 0.0025 \
    --inventory "data/raw/inventory/roback/Roback_Nepal_final_files/Source20170209.shp"
```

Writes `outputs/<name>_failure_probability.tif`, `outputs/<name>_stability_class.tif`
and a report holding the fitted soil parameters and their AUC. Specific
catchment area and slope go to `data/work/` for inspection. `--no-fit` uses
SINMAP's generic parameter ranges instead of fitting.

The output is validated by the same command as the heuristic map:

```bash
python -m giri_landslide.cli step6-validate \
    --susceptibility outputs/gorkha_failure_probability.tif --inventory <path>
```

Cost is dominated by flow routing, which is not tiled. The example above
(320 × 240 px) takes about 30 seconds; cost scales with cell count.

### 5. Hazard

```bash
python -m giri_landslide.cli step5-hazard --name nepal --return-period 100
python -m giri_landslide.cli step5-hazard --name nepal --trigger earthquake --pga 0.35
```

### 6. Validate

```bash
python -m giri_landslide.cli step6-validate --name nepal \
    --inventory data/raw/inventory/sikkim/Google_Earth_landslides_polygon_21Dec2021.shp
```

Reports the frequency ratio per class — the share of landslides in a class
divided by the share of map area it occupies — which must increase with class
for the ordering to be meaningful. Multiple `--inventory` paths are pooled.

### Climate scenarios

```bash
python -m giri_landslide.cli step4-susceptibility --name ssp585 \
    --climate ssp585 --climate-period 2061-2080
```

Substitutes downscaled CMIP6 projections (default IPSL-CM6A-LR, as in the
manuscript) for the soil-moisture factor. Only susceptibility responds; the
triggering return period retains its present-day definition.

## Repository layout

```
giri_landslide/
├── config.py              All tables, weights, thresholds and matrices
├── pipeline.py            Stage orchestration
├── cli.py                 Command-line workflow
├── input/
│   ├── datasets.py        Dataset registry, cache and availability checks
│   ├── sources.py         Terrain and climate downloaders
│   └── inventory.py       Landslide inventories: fetch, load, sample
├── model/
│   ├── factors.py         Inputs to ordinal factor scores
│   ├── susceptibility.py  Factor combination, continuous index, classes
│   ├── calibrate.py       Weight, slope-table and lithology fitting
│   ├── hydrology.py       Depression filling, D-inf flow, catchment area
│   ├── physical.py        SINMAP infinite-slope stability
│   ├── validate.py        Held-out validation
│   ├── triggers.py        Rainfall and earthquake severity classes
│   ├── hazard.py          Susceptibility × trigger → probability
│   └── risk.py            Not implemented; scope documented
└── utility/
    ├── grid.py            Reference grid, warping, tiled processing
    └── demo.py            Synthetic inputs for offline testing

configs/                   Run configurations
docs/                      Detailed operating guide
scripts/                   Offline demonstration
tests/                     Test suite, no network required
data/, outputs/            Generated, not version-controlled
```

Model constants are confined to `config.py`.

## Data

| Role | Dataset | Resolution | Acquisition |
|---|---|---|---|
| Slope | Copernicus GLO-90 DEM | 90 m | Automatic |
| Land cover | ESA WorldCover 2021 | 10 m | Automatic |
| Lithology | GLiM, 1.2 M polygons | Vector | Automatic, 1.1 GB once |
| Soil moisture | WorldClim v2.1 | 1 km | Automatic, 1.0 GB once |
| Future climate | WorldClim CMIP6 | 4.6 km | Automatic |
| Inventories | Gorkha, Far-West Nepal, Sikkim, NASA GLC | Points, polygons | Automatic |
| Earthquake PGA | GEM seismic hazard map | — | Manual, or scenario value |

### Inventory selection

| Inventory | Records | Mapping | Suitability |
|---|---|---|---|
| Roback Gorkha, Nepal | 24,795 | Satellite, earthquake-triggered | Best available; earthquake calibration |
| Far-Western Nepal | 26,350 | Satellite, multi-temporal | Rainfall calibration |
| Southern Sikkim, India | 255 | Satellite | Validation |
| NASA GLC | 11,033 global | Media reports | Screen by `location_accuracy`; only 32 % are placed to 1 km or better |

Match the inventory's triggering mechanism to the run. Weights fitted in one
sub-region do not necessarily transfer to another; validate against a local
inventory before relying on a map elsewhere.

### Resolution

The default is 90 m. The slope reclassification table was calibrated on ~90 m
slope statistics; a 30 m DEM yields systematically steeper slopes and inflates
the high-susceptibility area by a factor of thirteen over the same area. To run
at 30 m, refit the slope table first with `step3-calibrate --fit-slope-breaks`.

## Status

Implemented and tested:

- Susceptibility index, continuous and classified
- Physically based stability: D-infinity hydrology, SINMAP failure probability,
  soil parameters fitted to an inventory
- Weight, slope-table and lithology calibration from inventories
- Held-out validation
- Present and CMIP6 future-climate scenarios
- Earthquake hazard, using the manuscript's transcribed matrix

Outstanding:

1. **Rainfall hazard matrix.** Published only as a figure; the values in
   `config.py` are illustrative. Relative patterns are meaningful, absolute
   rainfall-triggered probabilities are not.
2. **Risk.** No exposure or vulnerability data. Scope documented in
   `model/risk.py`.
3. **PGA and ERA5 soil moisture** fall back to uniform values unless supplied.
4. **Inventory coverage.** Nepal and Sikkim only. Pakistan, Afghanistan,
   Uttarakhand, Himachal, Bhutan and Myanmar have no usable inventory.
5. **Region-wide execution.** The HKH at 90 m is approximately 1.5 gigapixels
   and must be run as manual tiles; no tiling driver exists.
6. **Flow routing is not tiled.** Contributing area is a property of the whole
   drainage network, so the physical model holds the area of interest in memory.
   Basin-wise decomposition is the standard remedy and is not implemented.
7. **Soil depth is not mapped.** Cohesion is fitted as the combination
   `C = (Cr + Cs) / (h·ρs·g)`, so soil depth cannot be separated from cohesion.
   A soil-depth model would let both be constrained independently.

## Testing

```bash
python -m pytest tests/ -q
```

## Licence

MIT. Source datasets retain their own licences (Copernicus, ESA, WorldClim,
GLiM, GEM, NASA, Zenodo records); cite them accordingly.
