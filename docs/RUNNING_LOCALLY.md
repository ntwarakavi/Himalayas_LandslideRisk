# Running H-SIM locally

A practical guide: what to install, what to run, how long each step takes, and
what the output means. For the model itself — the mechanics, the fitting, and
the measured results — see the [README](../README.md).

## 0. What you need

| | |
|---|---|
| Python | 3.9 or newer (3.11+ recommended) |
| Disk | ~3 GB for a fitted Gorkha run; ~15 GB if you also fetch GLiM and the 30 s climatology |
| Memory | Flow routing holds the whole area of interest in memory. About 4 GB covers a 6-million-cell run; see [Sizing a run](#4-sizing-a-run) |
| Network | Only for the download step. Everything else works offline once cached |
| OS | Linux, macOS or Windows (WSL recommended on Windows) |

The one heavy dependency is `rasterio` (it bundles GDAL). `fiona` is needed only
for vector inventories and GLiM; `matplotlib` only for quicklook PNGs.

## 1. Install

```bash
git clone https://github.com/ntwarakavi/Himalayas_LandslideRisk.git
cd Himalayas_LandslideRisk
./setup.sh
source .venv/bin/activate
```

If `rasterio` will not build from source — common on macOS without Homebrew
GDAL — use conda instead, which ships binary wheels:

```bash
conda create -n hkh python=3.11 -c conda-forge \
    rasterio fiona numpy matplotlib requests pytest
conda activate hkh
```

Check the install:

```bash
python -m pytest tests/ -q
```

48 tests, no network, a few seconds. They check the mechanics against exact
analytic answers — FS = 1 at the friction angle, mass conservation in flow
routing, the Newmark yield coefficient — so if they pass, the physics is wired
up correctly.

## 2. Offline smoke test (no downloads)

```bash
./scripts/run_demo.sh
```

Fabricates a synthetic mountain catchment and walks all four phases against it:
susceptibility, every trigger scenario, a climate sweep and the manifest. About
a minute. It proves the code runs, not the science — the terrain is noise with
valleys in it, and phase 2 is skipped because calibration needs a real
inventory.

## 3. Your first real run

Two commands. The first fetches data, the second builds a map.

```bash
python -m h_sim.cli step2-download --config configs/01_quickstart.json
python -m h_sim.cli step5-susceptibility --config configs/01_quickstart.json
```

About 120 MB and a few minutes. This uses SINMAP's generic parameter ranges
because no inventory was supplied: **the pattern is meaningful, the level is
not.** Section 5 fits the parameters properly.

Outputs land in `outputs/`:

| File | What it is |
|---|---|
| `quickstart_susceptibility_prob.tif` | Probability of failure, 0–1. **This is the product.** |
| `quickstart_susceptibility_class.tif` | SINMAP stability classes 1–6 (see the README on why the lower bands are a legend, not an ordering) |
| `quickstart_critical_acceleration.tif` | Newmark yield coefficient in g: the shaking needed to bring each slope to FS = 1 |
| `quickstart_susceptibility_summary.json` | Parameters used, unstable area fraction, scenario if any |

Intermediates land in `data/work/` and are worth a look:
`*_slope_tan.tif` (slope as a gradient) and `*_sca.tif` (specific catchment
area, in metres). If the catchment-area raster does not show a drainage network
when you open it, something upstream is wrong with the DEM.

## 4. Sizing a run

Every stage streams in tiles **except flow routing**, which cannot: contributing
area is a property of the whole drainage network, so depression filling and
D-infinity accumulation need the full area at once.

Cell count is `(width_deg / res) × (height_deg / res)`:

| Extent | `--res` | Cells | Rough behaviour |
|---|---|---|---|
| 0.8° × 0.6° | 0.0025 (~250 m) | 77 k | seconds |
| 1.0° × 0.8° | 0.0025 | 128 k | seconds |
| 2.9° × 1.7° | 0.0025 | 750 k | a minute or two |
| 0.8° × 0.6° | 0.00027778 (30 m) | 6.2 M | tens of minutes, several GB |

The CLI prints a warning above 40 million cells. Above that, split the area by
basin and run the pieces separately — there is no tiling driver.

Note that `--res` and `--dem-source` are independent. At `--res 0.0025` a 30 m
DEM is downsampled to ~250 m and the source barely matters; the finer DEM only
pays once the grid can resolve it.

**Resolution is the most consequential setting here.** Measured on Gorkha,
spatial-block AUC runs 0.728 at 250 m, 0.806 at 90 m and 0.816 at 30 m, because
a coarse cell cannot represent the hollows that concentrate subsurface flow. Use
90 m unless you have the compute for 30 m; do not use 250 m for anything but a
first look. Full numbers in [RESULTS.md](RESULTS.md).

## 5. Fitting the parameters to real landslides

This is the step that turns a plausible map into a defensible one.

```bash
python -m h_sim.cli step2-download --config configs/02_calibrate_gorkha.json
python -m h_sim.cli step3-fit --config configs/02_calibrate_gorkha.json
```

The fit searches 48 parameter sets and cross-validates twice. On the Gorkha
area at 30 m it takes a few minutes and prints:

```
  Fitted parameter ranges
    cohesion C        0.000 .. 0.250
    friction phi      25.0 .. 35.0 deg
    R/T               1.00e-05 .. 5.00e-04 1/m
    recharge ref      473 mm wettest-month precip

  In-sample AUC       0.822   (5193 landslides, 10386 background)
  random   CV AUC     0.822 +/- 0.003
  spatial  CV AUC     0.816 +/- 0.020
```

### Reading those numbers

**Quote the spatial figure.** A random split scatters test points among training
points on the same hillsides, so it measures interpolation as much as skill. A
spatial-block split withholds whole 0.25° blocks, so no test point has training
data nearby — that is the number that says what to expect somewhere new.

**The gap between them is the diagnostic.** Here it is 0.006, small enough that
the relationship is genuinely mechanical rather than spatial. A gap above 0.03
triggers a warning. For comparison, a random forest given lithology and land
cover as well shows a gap of 0.100 on this same data — almost all of its
apparent skill is memorised geography.

**The fold spread matters as much as the mean.** ±0.020 across five blocks means
the mean describes no particular place; that spread is the range to expect when
you apply the map somewhere the fit never saw.

The parameters go to `outputs/gorkha_fitted_params.json`. Steps 4 and 5 read it
automatically when `--name` matches, or point at it explicitly with
`--fitted-params`.

### Choosing an inventory

```bash
python -m h_sim.cli step3-fit --name farwest \
    --bbox 80.558 28.913 81.592 29.856 --res 0.00083333 \
    --inventory data/raw/inventory/farwest/<file>.shp
```

| Inventory | Records | Mechanism |
|---|---|---|
| Roback Gorkha | 24,795 | Earthquake |
| Far-Western Nepal | 26,350 | Monsoon, multi-temporal |
| Southern Sikkim | 255 | Mixed — too small to fit, use for validation |
| NASA GLC | 11,033 global | Media reports; screen by `location_accuracy` |

The physical model has no trigger-specific parameters, so an earthquake
inventory can fit a model you then apply to rainfall. But the wetness the
parameters absorb reflects conditions during the mapped events, so matching the
mechanism is still preferable.

Never fit and validate on the same inventory. Fit on one, validate on another.

**Check the mapped extent.** Background points stand in for terrain that did not
fail, so any drawn outside the ground the inventory's authors surveyed are
really saying "nobody looked". Far-West Nepal and Sikkim survey only about 60 %
of their own bounding boxes. All three inventories publish an extent polygon;
`analysis/common.py` shows how to mask against one.

**Expect the model to work better in some terrain than others.** Held-out AUC is
0.816 on Gorkha's soil-mantled crystalline terrain and 0.656 in the weak
sedimentary hill country of Far-Western Nepal, and refitting locally does not
close the gap. See [RESULTS.md §4](RESULTS.md).

## 6. Validating

```bash
python -m h_sim.cli step4-validate --name gorkha \
    --inventory data/raw/inventory/sikkim/Google_Earth_landslides_polygon_21Dec2021.shp
```

Multiple `--inventory` paths are pooled into one validation set. The report:

```
  class   map area   landslides   freq. ratio
    1      57.06%     24.38%       0.43  ##
    ...
    5      10.65%     45.64%       4.29  #################

  landslides in classes 4-5 : 60.2%  (those classes cover 21.4% of the map)
  efficiency                : 2.81x
  AUC                       : 0.740
  monotonic class ordering  : yes
```

Read it in this order:

1. **Monotonic ordering.** Frequency ratio must rise with class. If it does not,
   the map's ordering is not supported by the data and nothing else matters.
2. **Efficiency.** Landslide share in the top two bands over their area share.
   Below about 1.5 the map is not selective enough to act on.
3. **AUC.** Useful for comparing maps; it hides non-monotonicity, so never read
   it first.

The continuous field is binned into map-area quintiles for the table, but AUC is
computed on the raw values, so binning costs no resolution.

## 7. Trigger scenarios

Rainfall raises recharge; earthquakes add an inertial term. Both reduce to one
scalar in the factor of safety, so hazard and stability share a code path.

```bash
# a 100-year storm
python -m h_sim.cli step6-hazard --name gorkha --return-period 100

# 0.35 g of shaking
python -m h_sim.cli step6-hazard --name gorkha \
    --trigger earthquake --pga 0.35
```

Outputs are named for the scenario (`gorkha_hazard_rp100_prob.tif`,
`gorkha_hazard_pga0.35_prob.tif`) so scenarios do not overwrite each other.

Two conventions set the absolute level and are worth knowing about:

- `rainfall_cv` (0.30) — the coefficient of variation of annual maximum daily
  rainfall, which controls how much rarer storms scale recharge. Monsoon Asia
  is roughly 0.25–0.35; rerun at both ends to see the sensitivity.
- `pga_fraction` (0.5) — the fraction of PGA used as the pseudo-static
  coefficient, the standard convention since a sustained force stands in for a
  brief oscillation.

Relative patterns across a map are unaffected by either. The absolute
probability under a scenario is not.

For a real seismic hazard grid rather than a uniform value, set `pga_path` to a
PGA raster in g (the GEM global model is the usual source — see
`python -m h_sim.cli info`).

## 8. Current and future climate

Climate enters the model at one place only — the recharge field. Soil
parameters, terrain and the meaning of a return period are all unchanged.

```bash
# the scenarios named in the config
python -m h_sim.cli step7-climate --config configs/03_production_gorkha.json

# or name them directly
python -m h_sim.cli step7-climate --name gorkha \
    --scenarios current ssp245:2041-2060 ssp585:2041-2060
```

```
  scenario              mean P   unstable %   mean change   % more likely
  ------------------------------------------------------------------------
  current               0.1931       16.88       +0.0000            0.00
  ssp245_2041-2060      0.1938       16.93       +0.0006            1.22
  ssp585_2041-2060      0.1945       17.03       +0.0014            4.02
```

One step produces every map, every change raster against the present day, and a
summary row per scenario. The baseline is always evaluated first and always
included, because every future is reported as a difference from it.

**Why the baseline goes first.** A future precipitation field is normalised by
the *present-day* recharge reference in millimetres, recorded when the
parameters were fitted. Normalising it by its own median instead would divide
out exactly the signal being looked for — a uniformly wetter future would come
back looking identical to today. If no fit exists, the reference is measured
from the present-day field, which keeps the *changes* meaningful even though the
absolute level is uncalibrated.

Scenario specifications are `current`, or `<pathway>:<period>`:

| | |
|---|---|
| Pathways | `ssp126` `ssp245` `ssp370` `ssp585` |
| Periods | `2021-2040` **`2041-2060`** (default) `2061-2080` `2081-2100` |
| GCM | `--climate-model`, default IPSL-CM6A-LR |

Everything defaults to **2041-2060**, a twenty to thirty year planning horizon.
That is the window a road alignment or a settlement plan is decided over, and it
is what `step10-risk` scores assets against. Later windows show a bigger signal
and are one flag away, but a map of 2090 is not a decision anyone can act on,
and the further out the window the more of its spread is the choice of GCM
rather than the pathway. `configs/05_climate_trajectory.json` runs all four
windows when the question is the time course rather than the horizon.

Two ready-made sweeps ask different questions.
`configs/04_climate_pathways.json` holds the window fixed and varies forcing, so
the spread between maps is the pathway alone.
`configs/05_climate_trajectory.json` holds forcing fixed and varies the window,
giving the time course.

The triggering return period keeps its present-day definition: terrain takes
far longer than a century to adjust, and redefining the trigger at the same time
would confound two effects in one map.

## 8b. Regional production, one province at a time

The whole region cannot be a single run: 4,400 × 2,500 km at 30 m is thirteen
billion cells and flow routing is not tiled. `step9-region` sweeps
administrative units instead.

Always cost it first — nothing is computed:

```bash
python -m h_sim.cli step9-region --dry-run --res 0.00083333 \
    --countries Nepal Bhutan
```

```
  cells (M)  country / unit
  --------------------------------------------------------
        5.3  Nepal / Bheri
        5.2  Nepal / Janakpur
        ...
```

Then run it, pointing at the parameters fitted once for the region:

```bash
python -m h_sim.cli step9-region --name hkh --res 0.00083333 \
    --fitted-params outputs/gorkha_fitted_params.json
```

Useful flags: `--countries`, `--units` to restrict; `--with-hazard` and
`--with-climate` to run those per unit as well; `--no-resume` to redo units
that already have output.

Each unit writes `<name>_<country>_<unit>_susceptibility_prob.tif` and friends,
plus a `_unit.json` marker. A region-wide table lands in
`<name>_region_summary.json`, sorted so the worst provinces come first:

```
  unstable %   mean P   unit
  --------------------------------------------------------
      20.97   0.2307   Nepal / Bagmati
      15.06   0.1701   Nepal / Bhojpur
```

Four things to know before starting a long sweep:

- **Calibrate once, sweep many.** Every unit reads the same fitted parameters.
  Nothing is refitted per province, and it should not be — a province is an
  administrative unit, not a soil unit.
- **It is resumable.** A unit whose `_unit.json` exists is skipped. A full pass
  is measured in days; something will interrupt it.
- **Each unit is routed wide and clipped late**, by `admin_buffer_deg`
  (0.05°, 5.5 km), so catchments are not truncated at borders. That default is
  measured, not guessed — see [RESULTS.md §6](RESULTS.md).
- **Budget about twice the cells you keep.** Provinces are irregular and runs
  are over bounding boxes; roughly half of each box falls outside its province
  and is discarded at the clip.

Oversize units are listed and skipped rather than attempted:

```
[h-sim] region   3 units exceed 40,000,000 cells and are skipped
```

Lower `--res` for those, or leave them for a basin-level split that is not
implemented yet.

## 8c. Settlements, roads and the web map

Steps 10 and 11 turn a susceptibility raster into something a planner can act
on, then into something they can open.

```bash
python -m h_sim.cli step10-risk --name gorkha30 \
    --bbox 84.5 27.6 85.3 28.2 --res 0.00027778
python -m h_sim.cli step11-map  --name gorkha30
```

Step 10 fetches settlements and roads for the area, then scores each one against
every climate scenario in `risk_climate` — by default the present day plus
`ssp245` and `ssp585` over 2021-2040 and 2041-2060, the windows that fall inside
a 20 to 30 year planning horizon. Any future susceptibility map it needs and
cannot find is computed on the spot, normalised by the present-day recharge
reference. Override the list:

```bash
python -m h_sim.cli step10-risk --name gorkha30 \
    --risk-climate current ssp585:2041-2060
```

What comes back:

```
  scenario            exposed    people   road km   road %    mean
  ----------------------------------------------------------------
  current *              1,204    82,510       412     31.7   0.121
  ssp245_2021-2040       1,211    82,910       414     31.9   0.123
  ssp585_2041-2060       1,248    85,120       427     32.9   0.129
```

`exposed` counts assets at or above a score of 0.08. The score is **not** the
susceptibility under the asset — towns sit on flat ground, and sampling there
answers "safe" for exactly the settlements a slope above is about to bury. It is
the proximity-weighted fraction of upslope ground that could reach the asset
under an 18 degree angle of reach and that the model calls unstable. Tune with
`travel_angle_deg` and `reach_radius_m`; a smaller angle means a longer reach
and a more conservative screen.

Do not read the maximum instead. `reaching_max` is recorded per asset because
"the worst single cell above this town" is a useful diagnostic, but a maximum
over a few thousand cells saturates: on Gorkha at 30 m it put over half the
settlements in the top band and stopped discriminating.

Step 11 writes `outputs/<name>_webmap/index.html`. Open it directly — Leaflet is
vendored beside it and the layers ship as scripts rather than fetches, so no web
server is needed and nothing but the basemap tiles wants a network. The climate
selector switches the raster, the settlement colours, the road colours and every
table at once; each popup carries the asset's whole set of scenario scores and
the change from today.

If Overpass is unreachable, settlements fall back to GeoNames and roads to
Natural Earth — trunk routes only, generalised to about 1:10 M. That fallback is
recorded in each feature's `source` field, and a road layer of a few dozen
segments over a whole district is the signature of it having happened. Rerun
after deleting `data/raw/exposure/roads_<name>.json` to try Overpass again.

## 9. Packaging the deliverables

```bash
python -m h_sim.cli step8-package --name gorkha
```

Nothing is recomputed and nothing is copied. This catalogues what exists and
attaches what a reader needs in order to know what a raster means: the fitted
parameters, the held-out score, the grid read from the rasters themselves, the
data sources, the two trigger conventions, and the interpretation notes.

Run it last, and ship it with the maps. A map without this file is a picture,
not a deliverable.

## 9b. Calibration regions (optional; measured, and they do not help)

Fits separate soil parameters per rock type or land-cover class:

```bash
python -m h_sim.cli step3-fit --name gorkha \
    --calibration-regions lithology --inventory <path>
```

A region gets its own parameters only if it holds at least
`min_region_presence` landslides (default 100); the rest fall back to the
whole-area fit. Lithology needs the 1.1 GB GLiM geodatabase, downloaded only
when regions are requested.

Measured at **−0.0004 AUC held out**, in both Gorkha and the geologically varied
Far-West — including the area chosen to give the idea its best chance. Off by
default. The negative result is specific to GLiM level-1 zoning, which collapses
Far-West's dozen named formations into five classes; a local geological map
might do better.

## 10. Where everything lands

```
data/raw/         downloaded sources, never re-fetched
  dem/            Copernicus DEM tiles
  worldclim/      monthly precipitation
  glim/           lithology (only if you asked for regions)
  inventory/      landslide inventories
  exposure/       settlements and roads, cached per run name
data/work/        per-run intermediates, safe to delete
  <name>_dem.tif             DEM on the run grid
  <name>_slope_tan.tif       slope as a gradient
  <name>_sca.tif             specific catchment area, m
  <name>_recharge_<scenario>.tif  dimensionless recharge multiplier,
                             one per climate scenario
  <name>_regions.tif         calibration regions, if any
outputs/          the products
  <name>_fitted_params.json
  <name>_susceptibility_prob.tif
  <name>_susceptibility_class.tif
  <name>_critical_acceleration.tif
  <name>_hazard_*_prob.tif
  <name>_susceptibility_<scenario>_prob.tif   one per future climate
  <name>_risk_settlements.json     every settlement, every scenario
  <name>_risk_roads.json           every 500 m segment, every scenario
  <name>_risk_summary.json
  <name>_webmap/index.html         the browsable page and its assets
  <name>_validation.json
```

`data/work/` can be deleted at any time; it rebuilds. Deleting `data/raw/` means
re-downloading.

Work files are keyed on `--name` alone, but the model checks a cached raster's
grid before reusing it, so re-running a name at a different `--res` recomputes
rather than mixing grids.

## 11. Customising a run

Any config field can be overridden on the command line; the CLI wins over the
JSON. The fields that matter most:

| Field / flag | Effect |
|---|---|
| `--res` | Grid resolution. Changes specific catchment area, so **refit after changing it** |
| `--dem-source` | `copernicus30` (default) or `copernicus90` |
| `--samples` | Monte Carlo draws per pixel (default 200; the probability resolves to 1/n) |
| `--uniform-recharge` | Hold recharge uniform, isolating terrain |
| `--calibration-regions` | `lithology` or `landcover` (measured at -0.0004 AUC) |
| `--climate` | scenario for a single run: `current` or `ssp585:2041-2060` |
| `--scenarios` | step7 only: the list to sweep |
| `--output` | `probability`, `classes` or `both` |
| `cv_block_deg` | Spatial-block size in degrees (default 0.25 ≈ 25 km) |
| `rainfall_cv`, `pga_fraction` | The two trigger conventions |

## 12. Troubleshooting

**`AOI ... is outside the Hindu Kush Himalaya region`** — the model is scoped to
the HKH and clips every area of interest to it. Widen `region_bbox` only if you
mean to.

**`only N landslides fall inside the AOI`** — fitting needs at least 50. Either
widen the extent or pick an inventory that covers it.

**Flow routing is slow or runs out of memory** — the AOI is too large for a
single pass. Halve the extent or coarsen `--res`; see
[Sizing a run](#4-sizing-a-run).

**`no fit found -> SINMAP generic ranges`** — steps 4 and 5 could not find a
fitted-parameters JSON. Run `step3-fit` first, or pass `--fitted-params`. The
map will still build, but its level is not calibrated.

**Specific catchment area looks like noise** — the DEM did not warp correctly.
Open `data/work/<name>_dem.tif` and check it holds real elevations over the
whole extent rather than nodata.

**`frequency ratio does not rise monotonically`** on the class map — expected
for SINMAP classes 1–3, which all have failure probability zero and are ordered
only by margin of stability. Validate the continuous field instead.

**Downloads fail behind a proxy** — every source is a plain HTTPS GET, so
`HTTPS_PROXY` is respected. `step1-check` reports which hosts are reachable.

**Overpass returns 406, or a 429 saying you are rate limited** — both mean the
request carried no User-Agent, not that the service is busy. The model sends
one; if you are calling `input/exposure.py` yourself, send one too.

**Only a handful of road segments over a whole district** — the Overpass fetch
failed and the run fell back to Natural Earth, which carries trunk routes only.
Delete `data/raw/exposure/roads_<name>.json` and rerun step 10.

**The web map opens but nothing is on it** — check that `settlements.js` and
`roads.js` sit beside `index.html`. If the panel says the map could not be
drawn, Leaflet is missing from `<name>_webmap/leaflet/`, which happens when the
page was built with no network and no cached copy; rebuild once online.
