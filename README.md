# H-SIM

**H**imalayan **S**lope **I**nstability **M**odel — a physically based
slope-stability model for the Hindu Kush Himalaya: Afghanistan, Pakistan,
India, Nepal, Bhutan, Bangladesh, China and Myanmar.

The model is SINMAP — infinite-slope stability closed by a steady-state
hydrology — computed over D-infinity flow routing, extended with a pseudo-static
term for seismic loading, and fitted to mapped landslide inventories. It runs
under present-day climate and under CMIP6 futures.

There is one model in this repository, and one product: **failure probability,
exposed settlements and exposed road segments for every mountain province in
the region, present day and to 2060.** Everything below is either that model or
the route to that product.

```bash
python -m h_sim.cli step5-susceptibility --config configs/02_hkh_region.json
```

95 states and provinces across seven of the eight member countries. Start at
[All eight countries](#all-eight-countries); the steps before it exist to make
that run defensible.

**Contents** — [The model](#the-model) · [Install](#install) ·
[Run it](#run-it) · [All eight countries](#all-eight-countries) ·
[Climate](#climate-current-and-future) · [Outputs](#outputs) ·
[Measured performance](#measured-performance) · [Limits](#limits)

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
see [Run it, step 3](#3-fit-the-soil-parameters).

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

`setup.sh` creates `.venv`, installs the dependencies and installs the package
editable. It changes to its own directory first, so it works from anywhere.

To do it by hand instead, **from the repository root** — `requirements.txt`
lives there, and `pip install -r requirements.txt` fails with
`No such file or directory` if you are anywhere else:

```bash
cd /path/to/Himalayas_LandslideRisk    # the directory containing setup.sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Or skip `requirements.txt` altogether — `pyproject.toml` declares the same
dependencies, so this is equivalent and works from any directory that contains
the project:

```bash
pip install -e ".[viz,dev]"
```

Either way the package is installed editable, so both of these work from any
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
python -m pytest tests/ -q     # 77 tests, seconds
./scripts/run_demo.sh          # the whole sequence on synthetic data, ~2 minutes
```

The tests check the mechanics against exact analytic answers — `FS = 1` at the
friction angle, mass conservation in flow routing, the saturated critical angle,
and that applying the critical acceleration drives `FS` to exactly 1.

---

## Run it

Five commands. The product is region-wide; the single-area steps exist only to
fit and check the parameters that the regional sweep then applies everywhere.

```bash
python -m h_sim.cli step1-check                                            # 1
python -m h_sim.cli step2-download --config configs/01_calibrate.json      # 2
python -m h_sim.cli step3-calibrate      --config configs/01_calibrate.json      # 3
python -m h_sim.cli step4-validate --build --name gorkha \
    --inventory data/raw/inventory/sikkim/Google_Earth_landslides_polygon_21Dec2021.shp
python -m h_sim.cli step5-susceptibility --config configs/02_hkh_region.json  # 5
python -m h_sim.cli step6-climate        --config configs/02_hkh_region.json  # 6
python -m h_sim.cli step7-settlements    --config configs/02_hkh_region.json  # 7
python -m h_sim.cli step8-roads          --config configs/02_hkh_region.json  # 8
python -m h_sim.cli step9-webapp         --config configs/02_hkh_region.json  # 9
python -m h_sim.cli package              --name hkh
```

`run-all --config configs/01_calibrate.json` does the same in one command.

| | Command | Scope | Produces |
|---|---|---|---|
| 1 | `step1-check` | — | What is cached, what is reachable, what it will cost |
| 2 | `step2-download` | — | The source data |
| 3 | `step3-calibrate` | one calibration area | Soil parameters, cross-validated |
| 4 | `step4-validate` | one held-out area | Skill on landslides the fit never saw |
| 5 | `step5-susceptibility` | **the whole region** | Present-day failure probability, every province |
| 6 | `step6-climate` | **the whole region** | The same under CMIP6 scenarios |
| 7 | `step7-settlements` | **the whole region** | Settlements and villages, now and later |
| 8 | `step8-roads` | **the whole region** | Roads per 500 m segment, now and later |
| 9 | `step9-webapp` | **the whole region** | A page per province, plus one ranked index |
| | `package` | — | Manifest: every product and its provenance |

Steps 5–9 are stages of one regional sweep, each resumable on its own, so an
interrupted pass leaves a complete product rather than a fragment of every
product. The `area-*` commands (`area-susceptibility`, `area-hazard`,
`area-climate`, `area-risk`, `area-map`) do the same work over a single area of
interest — for debugging one province, not for making a deliverable.

The number inside a command name is part of the name, not its position: the
commands were numbered as they were added and never renumbered.

---

### 1–2. Get the data

```bash
python -m h_sim.cli step1-check
python -m h_sim.cli step2-download --config configs/01_calibrate.json
```

`step1-check` reports each dataset as cached, reachable, blocked or manual-only,
and totals the outstanding download. `--offline` restricts it to the local cache.

The regional sweep fetches DEM tiles per province as it goes, so step 2 only has
to bring down what the calibration needs plus the global layers.

### 3. Fit the soil parameters — once, for the whole region

```bash
python -m h_sim.cli step3-calibrate --config configs/01_calibrate.json
```

**This is the only step that is deliberately not region-wide.** Fitting needs a
mapped inventory, and the region has three: Gorkha, Far-Western Nepal and
Sikkim. The Gorkha earthquake footprint is used because Roback et al. ship
mapped *source areas*, which is what an infinite-slope model predicts. Nothing
from this run is a product — the product of step 3 is one JSON file that every
province then reads.

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

**Quote the spatial figure.** A random split scatters test points among training
points on the same hillsides, so it measures interpolation as much as skill; a
spatial-block split withholds whole 0.25° blocks. **The gap between them is the
diagnostic** — 0.006 here, small enough that the relationship is mechanical
rather than spatial. A gap above 0.03 triggers a warning. **The fold spread
matters as much as the mean**: ±0.020 is the range to expect somewhere new.

### 4. Check the parameters travel

```bash
python -m h_sim.cli step4-validate --build --name gorkha --res 0.00083333 \
    --inventory data/raw/inventory/sikkim/Google_Earth_landslides_polygon_21Dec2021.shp
```

The whole regional product rests on one parameter set applied 95 times, so the
question that matters is not "does it fit Gorkha" but "does it survive being
moved". `--build` applies the fitted parameters over the held-out inventory's
own ground and scores there. That is transfer validation, and it is the only
honest test of a regional extrapolation.

The region's inventories do not overlap — Gorkha spans 84.5–85.3° E and Sikkim
88.1–88.9° E — so without `--build` there is nothing to score and step 4 stops
and prints the commands to run.

```
  landslides in classes 4-5 : 60.8%  (those classes cover 30.4% of the map)
  efficiency                : 2.00x (>1 means the map concentrates landslides)
  AUC                       : 0.702
  monotonic class ordering  : yes
  VERDICT: fair - ordering holds but discrimination is modest
```

Read it in this order: **monotonic ordering** first (the frequency ratio must
rise with class, or nothing else matters), then **efficiency** (below ~1.5 the
map is not selective enough to act on), then **AUC**.

Two traps. **Never fit and validate on the same inventory.** And **check the
inventory's mapped extent** — background drawn outside the ground its authors
surveyed means "nobody looked", not "no landslide". Far-West Nepal and Sikkim
survey only about 60 % of their own bounding boxes, so pass the polygon with
`--survey-extent`.

Masking does not reliably *raise* the score and is not meant to. On this run it
moved AUC from 0.702 to 0.690, because the unsurveyed ground there was supplying
easy true negatives. What it does is make the negatives mean what they claim to.

Do not compare either figure with the 0.780 transfer result in
[docs/RESULTS.md §3](docs/RESULTS.md) — that is 30 m rather than 90 m, a
different extent, and target-group rather than uniform background. Only figures
produced the same way belong side by side.

### 5–6. Produce the region, then package it

Covered in full under [All eight countries](#all-eight-countries), which is the
rest of this page.

### Configurations

| File | What it is |
|---|---|
| `01_calibrate.json` | The fit. Gorkha at 30 m. **Not a product** — its output is a parameter file |
| `02_hkh_region.json` | **The product.** Every mountain province at 90 m, with everything |
| `03_hkh_recon.json` | The whole region at 250 m in hours, including the units 02 must skip. Lower skill; for planning, not for delivery |
| `04_hkh_climate.json` | All four CMIP6 pathways over the planning window, region-wide |
| `05_hkh_earthquake.json` | Seismic scenarios region-wide, three PGA levels |

For a single province, pass `--units`:

```bash
python -m h_sim.cli step5-susceptibility --config configs/02_hkh_region.json \
    --units Sikkim
```

---

## All eight countries

This is what the repository is for. Everything above is the machinery; this is
the product.

The region is 4,400 × 2,500 km. At 30 m that is thirteen billion cells and flow
routing is not tiled, so a regional run is a **sweep over states and provinces**,
one at a time. Fit once, then sweep.

```bash
# 1. plan: what would run, and what it would cost. Nothing is computed.
python -m h_sim.cli step5-susceptibility --dry-run --config configs/02_hkh_region.json

# 2. run it: every mountain province, with exposure and a page for each
python -m h_sim.cli step5-susceptibility --config configs/02_hkh_region.json

# 3. the manifest
python -m h_sim.cli package --name hkh
```

### Which provinces, and why those

**95 states and provinces** across Afghanistan, Pakistan, India, Nepal, Bhutan,
China and Myanmar.

| Country | Units | |
|---|---|---|
| Afghanistan | 31 | all |
| Bhutan | 20 | all |
| Nepal | 14 | all |
| India | 11 | Arunachal Pradesh, Himachal Pradesh, Jammu and Kashmir, Ladakh, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Uttarakhand, West Bengal |
| Myanmar | 7 | Chin, Kachin, Kayah, Kayin, Mandalay, Sagaing, Shan |
| Pakistan | 6 | Azad Kashmir, Baluchistan, F.A.T.A., K.P., Northern Areas, Punjab |
| China | 6 | Gansu, Qinghai, Sichuan, Xinjiang, Xizang, Yunnan |
| Bangladesh | 0 | no mountain terrain — see below |

**A bounding box is not a mountain range.** The HKH box spans 60–105° E and
16–39° N, which contains the whole Gangetic plain and most of peninsular India.
Selecting on it alone returns 137 units including Odisha, Madhya Pradesh,
Telangana, Andhra Pradesh and Gujarat, none of which have a Himalayan hillslope
in them.

Units are therefore selected on **relief**, from a 4.8 MB global elevation grid:

- A cell is mountain if it is **above 1,000 m** *and* has **500 m of local
  elevation range** within about 27 km. Both conditions are needed. Elevation
  alone cannot tell a mountain range from a high plain, and a high plain has no
  hillslopes to fail on: Inner Mongolia's Alashan plateau sits above 1,000 m
  with 60 m of local relief, against 1,512 m for Nepal's Bagmati. This is the
  standard definition of mountain terrain (Kapos et al. 2000).
- A unit is kept if **10 % of it is mountain**, *or* if it has **1,000 km² of
  mountain with a 1,400 m peak**. The second clause exists for West Bengal,
  which is 1.9 % mountain — and that 1.9 % is Darjeeling and Kalimpong, among
  the most landslide-prone ground in India. The altitude condition stops that
  clause readmitting the Eastern Ghats: Odisha clears the same area but tops out
  at 1,110 m.

Two honest caveats on that rule:

- **Relief finds mountains, not *these* mountains.** Inner Mongolia's Helan Shan
  and the Yunnan–Guizhou plateau pass every test and are not in the HKH arc.
  They are excluded by name in `admin.NOT_HKH` rather than by tuning a threshold
  until they vanish, which would take half the Himalaya with them. Override with
  `admin_exclude`.
- **Bangladesh drops out entirely**, though it is an HKH member country. The
  Chittagong Hill Tracts reach 597 m on this grid with 49 m of local relief:
  hills, not mountains, and outside what an infinite-slope model fitted in Nepal
  can speak to. Lower `admin_mountain_elevation_m` to include them, and treat
  the result with corresponding caution.

Narrow the sweep with `--countries Nepal Bhutan` or `--units Bagmati Gandaki`.

### What you get

Each stage resumes independently: a province whose output exists is skipped,
so an interrupted pass picks up where it stopped.

**`outputs/hkh_region/index.html` is the deliverable.** One ranked page over the
whole sweep: every province sorted by unstable area, sortable by any column,
filterable by name, settlements and road kilometres exposed beside each, and a
link into that province's own map. Fifty province folders are an archive; a
ranked table with a link per province is something a meeting can work from.

Per province, on disk:

```
outputs/hkh_<country>_<province>_susceptibility_prob.tif    failure probability
outputs/hkh_<country>_<province>_hazard_*_prob.tif          trigger scenarios
outputs/hkh_<country>_<province>_risk_settlements.json      every settlement
outputs/hkh_<country>_<province>_risk_roads.json            every 500 m segment
outputs/hkh_<country>_<province>_webmap/index.html          browsable page
outputs/hkh_region_summary.json                             machine-readable
outputs/hkh_manifest.json                                   provenance
```

### Before you present it

- **Provinces are comparable only because they share one parameter set.** It was
  fitted on soil-mantled crystalline terrain in Nepal. Across Pakistan,
  Afghanistan, Uttarakhand, Himachal, Bhutan and Myanmar there is no inventory,
  so skill there is **unknown rather than measured**. Measured transfer within
  the region ranges from 0.780 to 0.656.
- **The values are relative.** Differences between pixels and between provinces
  are meaningful; the number is not an annual probability of failure.
- **Exposure is screening, not risk.** No runout model, no vulnerability, no
  damage function.

### Practical points

- **The sweep is resumable.** A province whose output exists is skipped. A full
  pass at 90 m is measured in days and something will interrupt it.
- **Eight units exceed the memory ceiling and are skipped**, not attempted —
  above `admin_max_cells` (40 M), because the alternative is an out-of-memory
  kill part-way through. Xizang, Xinjiang, Qinghai, Gansu, Sichuan, Yunnan and
  Baluchistan need a coarser grid or a basin-level split. The index page names
  them.
- **A sweep routes roughly twice the cells it keeps.** Provinces are irregular
  and runs are over bounding boxes: measured on Nepal's Bagmati and Bhojpur,
  only 51 % and 54 % of the box fell inside the province.
- **Borders cut catchments, so each province is routed wide and clipped late.**
  The run covers the unit's box grown by `admin_buffer_deg` and is masked back
  afterwards. Without it, a cell just inside a border loses its upslope area and
  comes out too stable. Measured (`analysis/07_boundary_buffer.py`):

| Buffer | Cells losing >½ their catchment | Cells shifting P by >0.05 |
|---|---|---|
| none | 1.00 % (3.8 % in the outer ring) | 0.45 % |
| 0.028° (3 km) | **0.00 %** | **0.00 %** |

  Smaller than it sounds: hillslope contributing areas are hundreds of metres,
  and cells with long flow paths are valley floors already saturated at `w = 1`.
  The default 0.05° is about twice what was needed, as margin.

## Package the deliverables

```bash
python -m h_sim.cli package --name hkh
```

Writes `<name>_manifest.json`: every product, the fitted parameters, the
held-out score, the grid read from the rasters themselves, the data sources, the
two trigger conventions, which climates the exposure was scored under, and the
interpretation notes that must travel with the maps. Run it last and ship it
with them. A map without this file is a picture, not a deliverable.

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
python -m h_sim.cli step7-climate --config configs/04_hkh_climate.json

# or name scenarios directly
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

Scenarios are `current`, or `<ssp>:<period>`:

| | |
|---|---|
| Pathways | `ssp126` `ssp245` `ssp370` `ssp585` |
| Periods | `2021-2040` **`2041-2060`** (default) `2061-2080` `2081-2100` |
| GCM | `--climate-model`, default IPSL-CM6A-LR (mid-range sensitivity) |

**Every default lands on 2041-2060** — a twenty to thirty year planning horizon,
which is the horizon a road alignment or a settlement plan is actually decided
over. The end-of-century windows show a larger signal, which is why they are
usually quoted and why they are not the default here: a map of 2090 is not a
decision anyone can act on, and the further out the window, the more of its
spread is the choice of general circulation model rather than the pathway. Ask
for them explicitly when the question is how bad it eventually gets.

`configs/04_hkh_climate.json` runs all four pathways over the planning window
across the region, so the spread between maps is the forcing rather than a
mixture of forcing and date. Each future gets a susceptibility map, a change
raster against the present day, and a row in `<name>_climate_summary.json`.

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

A twelve-slide overview deck with speaker notes — product, physics,
measured evidence, display rules, limits, and run commands — lives at
[docs/H-SIM_overview.pptx](docs/H-SIM_overview.pptx).

Full write-up and scripts: [docs/RESULTS.md](docs/RESULTS.md),
[`analysis/`](analysis/). All spatial-block cross-validated on Gorkha.
A technical explanation of every method in the pipeline — terrain routing,
the stability model and its depth term, calibration protocol, the exposure
screen, road mechanisms, and the gated-adoption rule — lives at
[docs/METHODS.md](docs/METHODS.md).

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
   arrives. Step 10 screens what could arrive at a town or a road with an
   angle-of-reach criterion, which is a geometric bound, not a runout model:
   no volume, no channel geometry, no entrainment, no rheology.
2. **Domain of validity, and it is narrower than it looks.** Skill is 0.816 on
   the soil-mantled crystalline terrain it was fitted on. Carried elsewhere and
   measured against five independent inventories it spans **0.472 to 0.780**:
   0.702 in Sikkim, 0.656 in Far-Western Nepal, 0.543 against rainfall-triggered
   landslides in west-central Nepal, and **0.472 — worse than random — against
   the anthropogenic landslides around Shimla in Himachal Pradesh**. Refitting
   locally does not recover the difference; the limit is the predictors, not the
   parameters. Nothing in a fitted parameter set warns which case an area is.
   See [docs/RESULTS.md §12](docs/RESULTS.md).
3. **No term for human undercutting.** Road cuts, benches and construction on
   hillslopes are a large share of what fails around hill towns, and the model
   has nothing to say about them — measurably so: on the Shimla inventory the
   frequency ratio is *inverted*. Do not use these maps to reason about
   road-cut failures.
4. **Relative, not absolute.** The probability's level depends on background
   sampling. Pixel-to-pixel differences are meaningful; the value is not an
   annual failure frequency.
5. **Soil depth is not mapped**, so cohesion cannot be separated from depth.
6. **Two trigger conventions are not fitted.** The rainfall `cv` is inert; the
   PGA fraction is not.
7. **Flow routing is not tiled.** Contributing area is a property of the whole
   drainage network, so the area of interest is held in memory. The CLI warns
   above 40 million cells. The regional sweep works around this by taking
   administrative units, but units above that threshold — Xinjiang, Tibet —
   still need a coarser grid or a basin-level split that is not implemented.
8. **Calibration regions do not work as implemented** (−0.0004 AUC held out,
   including where geology is varied). Off by default.
9. **Exposure is screened; risk is not computed.** Risk is
   `hazard × exposure × vulnerability` and step 10 supplies the first two.
   There is no damage function, so nothing converts to expected loss or
   casualties, and population is carried for ranking only. Settlement and road
   coverage is whatever OpenStreetMap has, which is uneven across the region;
   where Overpass is unreachable the road layer falls back to Natural Earth
   trunk routes and is labelled as such.
10. **Inventory coverage is three areas, all in the eastern half.** Gorkha,
    Far-Western Nepal and Sikkim. Pakistan, Afghanistan, Ladakh, Uttarakhand,
    Himachal, Bhutan and Myanmar have none, so parameters there are
    extrapolated with no way to check them. Three further open inventories were
    found, tested and rejected — what exists, what does not and why is in
    [docs/RESULTS.md §12](docs/RESULTS.md). This is the single biggest limit on
    what the regional product can claim.

---

## Repository layout

```
h_sim/
├── config.py              Run configuration, region and scenario defaults
├── pipeline.py            Stage orchestration, one function per step
├── cli.py                 Command-line workflow
├── input/
│   ├── datasets.py        Dataset registry, cache and availability checks
│   ├── sources.py         Terrain and climate downloaders
│   ├── admin.py           States and provinces: the regional sweep's tiles
│   ├── exposure.py        Settlements and roads: what is there to be harmed
│   └── inventory.py       Landslide inventories: fetch, load, sample
├── model/
│   ├── hydrology.py       Priority flood, D-infinity flow, catchment area
│   ├── physical.py        SINMAP stability, parameter fitting, cross-validation
│   ├── climate.py         Present-day baseline and CMIP6 scenarios
│   ├── hazard.py          Trigger scenarios: recharge and seismic coefficient
│   ├── crossval.py        Random and spatial-block fold assignment
│   ├── validate.py        Held-out validation
│   └── risk.py            Angle of reach: what can arrive at a town or road
├── webmap.py              Standalone Leaflet page from a finished run
└── utility/
    ├── grid.py            Reference grid, warping, tiled processing
    └── demo.py            Synthetic inputs for offline testing

analysis/                  Seven experiments behind docs/RESULTS.md
configs/                   Five configurations; 02 is the regional product
docs/                      Operating guide and measured results
scripts/run_demo.sh        Offline walk through the whole sequence
tests/                     66 tests, no network required
```

## Data

| Role | Dataset | Resolution | Acquisition |
|---|---|---|---|
| Terrain | Copernicus GLO-30 DEM | 30 m | Automatic |
| Recharge, present | WorldClim v2.1 monthly precipitation | 1 km | Automatic, 1.0 GB once |
| Recharge, future | WorldClim CMIP6 downscaled | 4.6 km | Automatic |
| Calibration regions (optional) | GLiM, 1.2 M polygons | Vector | Automatic, 1.1 GB once |
| Calibration regions (optional) | ESA WorldCover 2021 | 10 m | Automatic |
| Inventories | Gorkha, Far-West Nepal, Sikkim | Mapped polygons | Automatic |
| Earthquake PGA | GEM seismic hazard map | — | Manual, or scenario value |
| States and provinces | Natural Earth 1:10m admin-1 | ~1:10 M | Automatic, 15 MB once |

### Landslide inventories

Every inventory the model uses, what it is, and **where in the workflow it is
used**. Nothing else is fitted or validated against.

| Inventory | Records | Mapping | Licence | Used in |
|---|---|---|---|---|
| Roback 2018, Gorkha, Nepal | 24,795 | Satellite, **source areas**, earthquake-triggered | Public domain (USGS) | **(step 3 — the calibration fit.** `configs/01_calibrate.json`. The one parameter set every province then reads.**)** |
| Far-Western Nepal, multi-temporal | 26,350 | Satellite, monsoon-triggered, with a mapped-extent polygon | CC BY 4.0 | **(step 4 — transfer validation, and `analysis/02`–`06` for the domain-of-validity and resolution experiments)** |
| Southern Sikkim, India | 255 polygons, with a mapped-extent polygon | Satellite | CC BY 4.0 | **(step 4 — transfer validation, `--survey-extent`; and `analysis/02_transfer.py`)** |


**Polygon inventories only.** Point catalogues — the NASA Global Landslide
Catalog and COOLR — are not used and not shipped. Two reasons, both measured
rather than assumed. Their positions are mostly known to a kilometre or worse,
which cannot be tested against a 90 m pixel. And being compiled from media
reports they carry a bias towards roads and settlements strong enough to invert
the answer: scored against a susceptibility map in west-central Nepal they give
AUC 0.346 and a Spearman correlation of **−0.74** between landslide count and
predicted susceptibility, because reports come from valley floors where people
are, not from the slopes that fail.

**Source areas matter.** The model predicts initiation, so an inventory mapping
whole-landslide polygons is sampled downslope of what the model describes.
Measured at +0.017 AUC for crown sampling on such an inventory, and −0.006 where
polygons are already source areas.

**Three areas, all in the eastern half of the region.** That is the whole
calibration and validation base, and it is the biggest limit on what the
regional product can claim. Three further open inventories were found, wired in
and rejected — Nepal monsoon (redundant with Far-West and without a mapped
extent), Shimla (anthropogenic road-cut failures, which the model has no term
for), and Eastern Himalaya large landslides (wrong failure mechanism, points not
source areas). What was searched for, what does not exist, and why, is in
[docs/RESULTS.md §12](docs/RESULTS.md).

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
