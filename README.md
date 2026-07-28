# H-SIM

**H**imalayan **S**lope **I**nstability **M**odel — a physically based
slope-stability model for the Hindu Kush Himalaya: Afghanistan, Pakistan,
India, Nepal, Bhutan, Bangladesh, China and Myanmar.

The model is SINMAP — infinite-slope stability closed by a steady-state
hydrology — computed over D-infinity flow routing, extended with a pseudo-static
term for seismic loading, and fitted to mapped landslide inventories. It runs
under present-day climate and under CMIP6 futures.

There is one model in this repository. Everything below describes it.

**Contents** — [The model](#the-model) · [Install](#install) ·
[How to run](#how-to-run) · [Climate](#climate-current-and-future) ·
[Outputs](#outputs) · [Measured performance](#measured-performance) ·
[Limits](#limits)

---

## The model

### 1. The question SINMAP answers

Given a hillslope with soil on it, will the soil stay put?

SINMAP (Pack, Tarboton & Goodwin 1998) answers by writing down the forces on a
column of soil and asking whether the ones holding it exceed the ones pulling it
downhill. It is a mechanical statement, not a correlation, which is why it can
be applied where no landslides have ever been mapped and why its parameters mean
something a laboratory can measure.

### 2. The infinite-slope assumption

Shallow landslides in mountain terrain are typically much wider and longer than
they are deep — a metre or two of soil sliding over tens of metres of hillside.
When that is true, the forces on the upslope and downslope ends of the block are
small compared with those on its base, and can be neglected. What is left is a
one-dimensional balance on a slice of unit area, which is what makes the model
cheap enough to run over millions of cells.

This is also the model's first limit: it describes *shallow translational*
failure. It does not describe deep-seated rotational slides, rock falls, or
debris-flow runout.

### 3. The factor of safety

The ratio of resisting to driving forces on that slice:

```
        C + ( cosθ − k·sinθ − w·r·cosθ ) · tanφ
FS  =  ─────────────────────────────────────────
                  sinθ + k·cosθ
```

| Symbol | Meaning | Units |
|---|---|---|
| `C` | cohesion, root plus soil, normalised by the weight of the soil column: `(Cr + Cs) / (h·ρs·g)` | dimensionless |
| `θ` | slope angle, from the DEM | degrees |
| `φ` | angle of internal friction of the soil | degrees |
| `r` | density ratio `ρw / ρs`, about 0.5 | dimensionless |
| `w` | relative wetness: the saturated fraction of the soil column | 0–1 |
| `k` | horizontal seismic coefficient; zero without an earthquake | fraction of g |

`FS > 1` stands, `FS < 1` fails, `FS = 1` is on the point of failing.

Reading the terms: **cohesion** and **friction** resist; **gravity down the
slope** drives. **Water** does not push the block over — it reduces the
effective normal stress that friction acts on, which is why `w` appears
multiplied by `tanφ` and not on its own. **Shaking** both adds a driving force
along the slope and lifts weight off the failure surface, which is why `k`
appears in both the numerator and the denominator.

With `k = 0` this is SINMAP's published form,
`FS = [C + cosθ(1 − w·r)tanφ] / sinθ`. Pore pressure is unaffected by inertia,
so the `w` term keeps its static form under shaking.

### 4. Where the water comes from

The wetness `w` is not guessed. It comes from a steady-state mass balance: water
recharging the slope over the area draining through a point must be carried away
by the soil's ability to transmit it.

```
w = min( R·a / (T·sinθ), 1 )
```

- `R` — recharge, the rate water enters the soil
- `a` — **specific catchment area**: the upslope area draining through unit
  contour width, in metres
- `T` — transmissivity, the soil's capacity to move water laterally

Two consequences matter:

**Only the ratio `R/T` is identifiable.** Doubling recharge and doubling
transmissivity give the same map. This is convenient — the ratio is far better
constrained than either term — and it is a hard limit on what fitting can
recover.

**Convergence is the physics.** Two hillslopes at the same gradient behave
differently if one is a planar spur and the other a hollow: the hollow collects
drainage from above, so `a` is larger, so `w` is larger, so `FS` is lower. This
is the single thing SINMAP knows that a slope-and-lithology index does not — and
it is why grid resolution turns out to dominate everything else (see
[Measured performance](#measured-performance)).

Wetness is capped at 1. Beyond saturation the excess runs off over the surface
rather than raising pore pressure further.

### 5. Getting `a`: terrain hydrology

Specific catchment area requires routing water across the DEM. `model/hydrology.py`
implements this in pure NumPy, following TauDEM:

| Stage | Method | Why |
|---|---|---|
| Depression filling | Priority flood (Barnes, Lehman & Mulla 2014) | Raw DEMs contain sinks, both real and artefacts, which trap flow and truncate catchments |
| Flow direction | D-infinity over eight triangular facets (Tarboton 1997) | Flow leaves a cell along a continuous angle, split between the two bracketing neighbours |
| Contributing area | D-infinity accumulation, elevation-ordered | Each cell's inflow is complete before it passes anything on |
| Specific catchment area | Contributing area ÷ contour width | Removes the dependence on cell size that total area would carry |

D-infinity rather than D8 because D8's 45° quantisation produces artificial
parallel flow lines on exactly the planar hillslopes shallow failure occupies.

### 6. From one factor of safety to a map

`C`, `φ` and `R/T` are not known per pixel and never will be. SINMAP's answer is
not to pretend otherwise: it treats each as uniform over a plausible range and
reports the **probability that `FS < 1`** across that range. The output is a
continuous field in [0, 1].

Two regions appear as constants and are worth naming:

- **Unconditionally stable** — stable even fully saturated at the most
  pessimistic parameters. Probability 0.
- **Unconditionally unstable** — unstable even bone dry at the most optimistic
  parameters. Probability 1. Such ground stands only through cohesion the model
  does not represent, or is actively eroding.

A six-class stability map is also written, but **the continuous field is the
product**. Classes 1–3 all have failure probability zero by definition and are
separated only by how far above 1 the worst-case `FS` sits, so landslide density
cannot order them — measured, and confirmed, in
[docs/RESULTS.md](docs/RESULTS.md).

### 7. Fitting the parameters

The physics fixes the *form* of the response. The inventory supplies the
*values*. A grid of 48 candidate parameter ranges — cohesion from bare to
well-rooted, friction across 25–45°, `R/T` over four orders of magnitude — is
scored by how well the resulting failure probability ranks mapped landslides
above background points, and the best is kept.

What that can and cannot recover:

- `R` and `T`, only as their ratio.
- Cohesion, only jointly with soil depth, since the model sees
  `C = (Cr + Cs)/(h·ρs·g)`. No soil-depth map is used here.
- The absolute *level* of the probability depends on how background points were
  drawn. Differences between pixels are meaningful; the value at a pixel is not
  a frequency of failure per year.

Cross-validation is run in two schemes, and the spatial one is the one to quote —
see [How to run, phase 2](#phase-2--calibrate-and-validate).

### 8. Triggering

In a physical model, hazard is not a separate calculation. Each trigger reduces
to a scalar the factor of safety already accepts:

**Rainfall** raises recharge, entering as a multiplier on `R/T`. Under a Gumbel
distribution of annual maximum daily rainfall, the ratio of a scenario depth to
the reference depth cancels the location parameter, leaving a dependence on the
coefficient of variation alone:

```
m(T) = [1 + cv·k(T)] / [1 + cv·k(T_ref)]        k(T) = (√6/π)(y_T − γ)
```

**Earthquakes** add an inertial force, entering as `k_h = fraction × PGA`. The
fraction is one half by convention (Hynes-Griffin & Franklin 1984), because a
pseudo-static analysis applies a sustained force where the real loading is a
brief oscillation. Solving `FS = 1` for `k_h` gives the **Newmark critical
acceleration**, written as a separate output.

Two numbers here are neither mechanics nor fitted, and they behave very
differently. The rainfall `cv` is effectively inert — across 0.20–0.40 it moves
AUC by 0.003. The PGA fraction is not: across 0.3–1.0 it moves the unstable area
from 57 % to 90 %. **Quote seismic scenarios as a range over `pga_fraction`;
rainfall scenarios need no such hedge.**

---

## Install

```bash
git clone https://github.com/ntwarakavi/Himalayas_LandslideRisk.git
cd Himalayas_LandslideRisk
./setup.sh
source .venv/bin/activate
```

`setup.sh` installs the package editable, so both of these work from any
directory:

```bash
h-sim --help                     # console script
python -m h_sim.cli --help       # module form, used throughout these docs
```

If `rasterio` will not build from source, use conda, which ships binary wheels:

```bash
conda create -n hkh python=3.11 -c conda-forge \
    rasterio fiona numpy matplotlib requests pytest
conda activate hkh
```

Verify, with no network:

```bash
python -m pytest tests/ -q     # 48 tests, seconds
./scripts/run_demo.sh          # all four phases on synthetic data, ~1 minute
```

The tests check the mechanics against exact analytic answers — `FS = 1` at the
friction angle, mass conservation in flow routing, the saturated critical angle,
and that applying the critical acceleration drives `FS` to exactly 1.

---

## How to run

Four phases. **The order matters**: do not produce a map from parameters that
have not been through phase 2 on an independent inventory, because a fit always
looks good on the landslides it was fitted to.

| Phase | Step | Command | Produces |
|---|---|---|---|
| **1 Set up** | 1 | `step1-check` | Dataset availability report |
| | 2 | `step2-download` | Cached source data |
| **2 Calibrate & validate** | 3 | `step3-fit` | Soil parameters, cross-validated |
| | 4 | `step4-validate` | Score against a held-out inventory |
| **3 Produce** | 5 | `step5-susceptibility` | Present-day failure probability |
| | 6 | `step6-hazard` | Rainfall and earthquake scenarios |
| | 7 | `step7-climate` | CMIP6 futures and the change from today |
| **4 Package** | 8 | `step8-package` | Manifest: products and provenance |

`run-all` executes every phase in sequence.

### Phase 1 — set up

```bash
python -m h_sim.cli step1-check
python -m h_sim.cli step2-download --config configs/02_calibrate_gorkha.json
```

`step1-check` reports each dataset as cached, reachable, blocked or manual-only,
and totals the outstanding download; `--offline` restricts it to the local cache.
`step2-download` skips anything already present. Only the DEM and the
precipitation climatology are fetched by default — land cover and the 1.1 GB
GLiM geodatabase come down only if a run asks for calibration regions.

### Phase 2 — calibrate and validate

```bash
python -m h_sim.cli step3-fit --config configs/02_calibrate_gorkha.json
```

```
  Fitted parameter ranges
    cohesion C        0.000 .. 0.250    (dimensionless: root + soil, over depth x unit weight)
    friction phi      25.0 .. 35.0 deg
    R/T               1.00e-05 .. 5.00e-04 1/m
    recharge ref      473 mm wettest-month precip   (fixes what a multiplier of 1 means)

  In-sample AUC       0.822   (5193 landslides, 10386 background)
  random   CV AUC     0.822 +/- 0.003
  spatial  CV AUC     0.816 +/- 0.020  <- quote this
```

**Reading those numbers.**

- **Quote the spatial figure.** A random split scatters test points among
  training points on the same hillsides, so it measures interpolation as much as
  skill. A spatial-block split withholds whole 0.25° blocks.
- **The gap between them is the diagnostic.** Here 0.006, small enough that the
  relationship is mechanical rather than spatial. A gap above 0.03 triggers a
  warning.
- **The fold spread matters as much as the mean.** ±0.020 across five blocks is
  the range to expect when applying the map somewhere new.

Then validate against an inventory the fit never saw:

```bash
python -m h_sim.cli step4-validate --name gorkha \
    --inventory data/raw/inventory/sikkim/Google_Earth_landslides_polygon_21Dec2021.shp
```

Read the report in this order: **monotonic ordering** first (frequency ratio must
rise with class, or nothing else matters), then **efficiency** (below ~1.5 the map
is not selective enough to act on), then **AUC**.

Two traps worth avoiding: never fit and validate on the same inventory, and
check the inventory's **mapped extent** — background points drawn outside the
ground its authors surveyed mean "nobody looked", not "no landslide". Far-West
Nepal and Sikkim survey only about 60 % of their own bounding boxes.

### Phase 3 — produce

```bash
python -m h_sim.cli step5-susceptibility --config configs/03_production_gorkha.json
python -m h_sim.cli step6-hazard        --config configs/03_production_gorkha.json --all
python -m h_sim.cli step7-climate       --config configs/03_production_gorkha.json
```

`step6 --all` runs every return period and PGA in the config, each producing its
own map — they are different questions and a user needs to know which one a map
answers. A single scenario instead:

```bash
python -m h_sim.cli step6-hazard --name gorkha --return-period 100
python -m h_sim.cli step6-hazard --name gorkha --trigger earthquake --pga 0.35
```

### Phase 4 — package

```bash
python -m h_sim.cli step8-package --name gorkha
```

Writes `<name>_manifest.json`: every product, plus the fitted parameters, the
held-out score, the grid read from the rasters themselves, the data sources, the
two trigger conventions, and the interpretation notes that must travel with the
maps. A map without this file is a picture, not a deliverable.

---

## Climate: current and future

Climate enters at exactly one place — the recharge field. Nothing else changes:
not the soil parameters, which are properties of soil; not the terrain; and not
the meaning of a return period, because terrain takes far longer than a century
to adjust and redefining the trigger at the same time would confound two effects
in one map.

**The reference is the part that is easy to get wrong.** A future field must be
normalised by the **present-day** reference in millimetres, recorded when the
parameters were fitted — not by its own median. Normalising each scenario by its
own statistics would divide out exactly the signal being looked for: a uniformly
wetter future would come back looking identical to today. `step7-climate` always
evaluates the baseline first, for this reason.

```bash
# the config's climate_suite
python -m h_sim.cli step7-climate --config configs/03_production_gorkha.json

# or name scenarios directly
python -m h_sim.cli step7-climate --name gorkha \
    --scenarios current ssp245:2061-2080 ssp585:2081-2100
```

```
  scenario              mean P   unstable %   mean change   % more likely
  ------------------------------------------------------------------------
  current               0.1931       16.88       +0.0000            0.00
  ssp245_2061-2080      0.1941       16.95       +0.0010            2.36
  ssp585_2081-2100      0.1956       17.17       +0.0024            7.34
```

Scenarios are `current`, or `<ssp>:<period>`:

| | |
|---|---|
| Pathways | `ssp126` `ssp245` `ssp370` `ssp585` |
| Periods | `2021-2040` `2041-2060` `2061-2080` `2081-2100` |
| GCM | `--climate-model`, default IPSL-CM6A-LR (mid-range sensitivity) |

Two ready-made sweeps: `configs/04_climate_pathways.json` holds the window fixed
and varies forcing; `configs/05_climate_trajectory.json` holds forcing fixed and
varies the window. Each future gets a susceptibility map, a change raster against
the present day, and a row in `<name>_climate_summary.json`.

---

## Outputs

| File | What it is |
|---|---|
| `<name>_susceptibility_prob.tif` | **The product.** Failure probability, 0–1, present-day climate |
| `<name>_susceptibility_class.tif` | SINMAP classes 1–6. A legend — lower bands are not ordered |
| `<name>_critical_acceleration.tif` | Newmark yield coefficient, g: the shaking needed to reach `FS = 1` |
| `<name>_hazard_rp<T>_prob.tif` | Failure probability under a `T`-year storm |
| `<name>_hazard_pga<g>_prob.tif` | Failure probability under `g` of shaking |
| `<name>_susceptibility_<scenario>_prob.tif` | Failure probability under a CMIP6 future |
| `<name>_climate_<scenario>_change.tif` | Future minus present day, signed |
| `<name>_fitted_params.json` | Parameters, cross-validation, warnings |
| `<name>_validation.json` | Held-out validation report |
| `<name>_climate_summary.json` | One row per scenario |
| `<name>_manifest.json` | Everything above, with provenance |

Intermediates in `data/work/` are worth inspecting: `*_slope_tan.tif`,
`*_sca.tif` (specific catchment area — if it does not show a drainage network,
something is wrong with the DEM), and `*_recharge_<scenario>.tif`.

---

## Measured performance

Full write-up and scripts: [docs/RESULTS.md](docs/RESULTS.md),
[`analysis/`](analysis/). All spatial-block cross-validated on Gorkha.

**Resolution dominates everything else.**

| Grid | Spatial CV AUC | ± | Capture in worst 20 % | Median specific catchment area |
|---|---|---|---|---|
| 250 m | 0.7281 | 0.033 | 58.8 % | 631 m |
| 90 m | 0.8059 | 0.032 | 69.0 % | 258 m |
| **30 m** | **0.8156** | 0.020 | 71.0 % | 132 m |

A 250 m cell is wider than the hollows that concentrate subsurface flow, so the
wetness term loses the contrast it depends on. Most of the gain arrives by 90 m.
**Refit after changing `--res`** — specific catchment area is resolution-sensitive
by construction.

**Validation at 30 m**, frequency ratio by map-area quintile:

| Quintile | Map area | Landslides | Freq. ratio |
|---|---|---|---|
| 1 (lowest) | 53.3 % | 10.8 % | 0.20 |
| 5 (highest) | 11.7 % | 56.6 % | **4.85** |

Monotonic, efficiency 3.15×, top 23 % of area holding 73.5 % of landslides.

**Parameters transfer between catchments** at a cost of 0.004–0.022 AUC, so a
fit from one Nepali catchment carries to another 400 km away.

**Against statistical alternatives**, the physics ties rather than wins: given
the same two predictors, SINMAP scores 0.816 and a logistic regression 0.822.
What it does buy is robustness — carried unchanged to two other catchments,
SINMAP had the smallest drop of five models tested, while a random forest with
lithology and land cover scored 0.974 at home and 0.592 away. Full table in
[docs/RESULTS.md §8](docs/RESULTS.md).

---

## Limits

1. **Shallow translational failure only.** No deep-seated slides, rock falls,
   or debris runout. The model says where material detaches, not where it
   arrives; a runout stage is scoped in `model/risk.py`.
2. **Domain of validity.** Skill is 0.816 on soil-mantled crystalline terrain
   and 0.656 in the weak sedimentary hill country of Far-Western Nepal, and
   refitting locally does not recover the difference — the limit is the
   predictors, not the parameters. Nothing in a fitted parameter set warns which
   case an area is; only a local inventory does.
3. **Relative, not absolute.** The probability's level depends on background
   sampling. Pixel-to-pixel differences are meaningful; the value is not an
   annual failure frequency.
4. **Soil depth is not mapped**, so cohesion cannot be separated from depth.
5. **Two trigger conventions are not fitted.** The rainfall `cv` is inert; the
   PGA fraction is not.
6. **Flow routing is not tiled.** Contributing area is a property of the whole
   drainage network, so the area of interest is held in memory. The CLI warns
   above 40 million cells; beyond that, split by basin.
7. **Calibration regions do not work as implemented** (−0.0004 AUC held out,
   including where geology is varied). Off by default.
8. **Risk is not implemented.** No exposure, vulnerability or runout model.
   Scope documented in `model/risk.py`.
9. **Inventory coverage** is Nepal and Sikkim. Pakistan, Afghanistan,
   Uttarakhand, Himachal, Bhutan and Myanmar have none, so parameters there are
   extrapolated.

---

## Repository layout

```
h_sim/
├── config.py              Run configuration, region and scenario defaults
├── pipeline.py            Stage orchestration, the four phases
├── cli.py                 Command-line workflow
├── input/
│   ├── datasets.py        Dataset registry, cache and availability checks
│   ├── sources.py         Terrain and climate downloaders
│   └── inventory.py       Landslide inventories: fetch, load, sample
├── model/
│   ├── hydrology.py       Priority flood, D-infinity flow, catchment area
│   ├── physical.py        SINMAP stability, parameter fitting, cross-validation
│   ├── climate.py         Present-day baseline and CMIP6 scenarios
│   ├── hazard.py          Trigger scenarios: recharge and seismic coefficient
│   ├── crossval.py        Random and spatial-block fold assignment
│   ├── validate.py        Held-out validation
│   └── risk.py            Not implemented; scope documented
└── utility/
    ├── grid.py            Reference grid, warping, tiled processing
    └── demo.py            Synthetic inputs for offline testing

analysis/                  Six experiments behind docs/RESULTS.md
configs/                   Seven run configurations, one per workflow shape
docs/                      Operating guide and measured results
scripts/run_demo.sh        Offline walk through all four phases
tests/                     48 tests, no network required
```

## Data

| Role | Dataset | Resolution | Acquisition |
|---|---|---|---|
| Terrain | Copernicus GLO-30 DEM | 30 m | Automatic |
| Recharge, present | WorldClim v2.1 monthly precipitation | 1 km | Automatic, 1.0 GB once |
| Recharge, future | WorldClim CMIP6 downscaled | 4.6 km | Automatic |
| Calibration regions (optional) | GLiM, 1.2 M polygons | Vector | Automatic, 1.1 GB once |
| Calibration regions (optional) | ESA WorldCover 2021 | 10 m | Automatic |
| Inventories | Gorkha, Far-West Nepal, Sikkim, NASA GLC | Points, polygons | Automatic |
| Earthquake PGA | GEM seismic hazard map | — | Manual, or scenario value |

| Inventory | Records | Mapping | Use |
|---|---|---|---|
| Roback Gorkha, Nepal | 24,795 | Satellite, **source areas** | The reference fit |
| Far-Western Nepal | 26,350 | Satellite, multi-temporal | Monsoon-driven terrain |
| Southern Sikkim, India | 255 | Satellite | Validation |
| NASA GLC | 11,033 global | Media reports | Screen by `location_accuracy` |

Source areas matter: the model predicts initiation, so an inventory mapping
whole-landslide polygons is sampled downslope of what the model describes.
Measured at +0.017 AUC for crown sampling on such an inventory, and −0.006 where
polygons are already source areas.

## Licence

MIT. Source datasets retain their own licences (Copernicus, ESA, WorldClim,
GLiM, GEM, NASA, Zenodo records); cite them accordingly.

## References

- Pack, R. T., Tarboton, D. G., Goodwin, C. N. (1998). The SINMAP approach to
  terrain stability mapping. *8th Congress of the IAEG.*
- Tarboton, D. G. (1997). A new method for the determination of flow directions
  and upslope areas in grid digital elevation models. *Water Resources Research*
  33(2), 309–319.
- Barnes, R., Lehman, C., Mulla, D. (2014). Priority-flood: an optimal
  depression-filling and watershed-labeling algorithm for digital elevation
  models. *Computers & Geosciences* 62, 117–127.
- Montgomery, D. R., Dietrich, W. E. (1994). A physically based model for the
  topographic control on shallow landsliding. *Water Resources Research* 30(4),
  1153–1171.
- Hynes-Griffin, M. E., Franklin, A. G. (1984). Rationalizing the seismic
  coefficient method. *US Army Engineer Waterways Experiment Station*, MP GL-84-13.
- Roback, K. et al. (2018). The size, distribution, and mobility of landslides
  caused by the 2015 Mw 7.8 Gorkha earthquake, Nepal. *Geomorphology* 301, 121–138.
