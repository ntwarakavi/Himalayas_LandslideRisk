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
probability — see [Validation](#validation).

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

Measured on the 2015 Gorkha earthquake area (84.5–85.3° E, 27.6–28.2° N) at
0.0025°, against the 5,193 Roback landslides falling inside it.

**Fitted parameters.** `C ∈ [0, 0.15]`, `φ ∈ [25°, 35°]`, `R/T ∈ [1×10⁻⁵,
5×10⁻⁴] m⁻¹`, recharge reference 473 mm wettest-month precipitation. The
friction range is at the low end of the search grid, which is what weathered
Himalayan metamorphic regolith should give.

**Skill.**

| Measure | AUC |
|---|---|
| In-sample | 0.740 |
| Random 5-fold CV | 0.745 ± 0.012 |
| Spatial-block 5-fold CV | **0.729 ± 0.024** |

The spatial figure is the one to quote. That it sits so close to the random one
is the substantive result: the relationship holds on ground the fit never saw,
so the skill is mechanical rather than spatial interpolation.

**Concentration.** Binning the continuous field into map-area quintiles. These
are **in-sample**: they score the map against the same Roback inventory the
parameters were fitted to, so they describe how well the fitted map concentrates
the landslides it was built from, not how it would behave on new ground. The
held-out figure is the spatial-block AUC above.

| Quintile | Map area | Landslides | Frequency ratio |
|---|---|---|---|
| 1 (lowest) | 57.1 % | 24.4 % | 0.43 |
| 2 | 11.0 % | 6.5 % | 0.59 |
| 3 | 10.5 % | 8.9 % | 0.84 |
| 4 | 10.8 % | 14.6 % | 1.35 |
| 5 (highest) | 10.7 % | 45.6 % | **4.29** |

The top 21 % of terrain by failure probability holds 60 % of the landslides, a
concentration of 2.8×. The ordering is monotonic.

### What did not help

Two additions were built, measured, and found not to earn their keep on this
area. Both are kept as options because the reason they fail here is specific to
the test area, but neither is on by default without cause.

- **Spatial recharge** (wettest-month precipitation modulating `R/T`) moves
  held-out AUC from 0.729 to 0.733 — well inside the ±0.024 fold spread. Over
  0.8° × 0.6° the precipitation gradient spans only 0.16–1.34× the median. It is
  left on by default because it is the physically correct treatment and because
  the gradient across the full HKH is far larger. `--uniform-recharge` turns it
  off.
- **Lithology calibration regions** find nothing here because the area is 97 %
  metamorphics: one region holds 5,038 of the 5,193 landslides, and the only
  other region large enough to fit (mixed sedimentary, n=122) scores worse
  (0.624) than the whole-area fit. Zoning needs lithological diversity to pay.
  Note that cross-validation scores the whole-area parameters, not the
  per-region ones.

### Validation of the class map

The continuous field is monotonic; the six-class SINMAP map is not.

| Class | Map area | Landslides | Frequency ratio |
|---|---|---|---|
| 1 unconditionally stable | 17.6 % | 7.0 % | 0.40 |
| 2 stable | 8.5 % | 4.6 % | 0.54 |
| 3 quasi-stable | 19.8 % | 7.3 % | 0.37 |
| 4 lower threshold | 40.9 % | 30.9 % | 0.75 |
| 5 upper threshold | 12.5 % | 41.5 % | 3.33 |
| 6 unconditionally unstable | 0.75 % | 8.8 % | **11.73** |

Also in-sample, and on the same caveat as the table above. Classes 4–6 are
ordered correctly and class 6 is the strongest single signal in
the map: three quarters of a percent of the area holds nearly nine percent of
the landslides. Classes 1–3 are not ordered, and cannot be: all three have
failure probability zero by definition and are separated only by how far above 1
the worst-case factor of safety sits. Landslides falling in them reflect
mapping and DEM positional error, and processes the model does not represent.
This is a property of the SINMAP class definition, not a defect in the fit — and
it is why the continuous field is the product.

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

configs/                   Run configurations
docs/                      Detailed operating guide
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

The default DEM is Copernicus GLO-30. Unlike a heuristic index with tables
calibrated at a particular cell size, the physics carries nothing that a finer
DEM would invalidate — and flow convergence, which the wetness term depends on,
is exactly what a coarse DEM smooths away.

The grid resolution is set separately by `--res`, and it is the one that
matters: at `--res 0.0025` (about 250 m) the 30 m DEM is downsampled and the
source barely matters. Specific catchment area is resolution-sensitive by
construction, so **refit after changing `--res`**.

## Status

Implemented and tested:

- D-infinity flow routing, verified against analytic cases
- SINMAP failure probability and stability classes
- Pseudo-static seismic loading and Newmark critical acceleration
- Soil parameter fitting with random and spatial-block cross-validation
- Optional calibration regions by lithology or land cover
- Rainfall and earthquake trigger scenarios
- Present and CMIP6 future-climate recharge
- Held-out validation

Outstanding:

1. **Flow routing is not tiled.** Contributing area is a property of the whole
   drainage network, so the model holds the area of interest in memory.
   Basin-wise decomposition is the standard remedy and is not implemented; the
   CLI warns above 40 million cells.
2. **Soil depth is not mapped.** Cohesion is fitted as `C = (Cr + Cs)/(h·ρs·g)`,
   so depth cannot be separated from cohesion. A soil-depth model would
   constrain both independently.
3. **Two trigger parameters are conventions, not fits** — the rainfall
   coefficient of variation and the PGA fraction. Relative patterns across a map
   are unaffected; the absolute level of a scenario probability is not.
4. **Risk.** No exposure or vulnerability data. Scope documented in
   `model/risk.py`.
5. **Inventory coverage.** Nepal and Sikkim only. Pakistan, Afghanistan,
   Uttarakhand, Himachal, Bhutan and Myanmar have no usable inventory, so
   parameters there are extrapolated.
6. **Region-wide execution.** The HKH at 30 m is far beyond a single run and
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
