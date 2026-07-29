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

Four phases, run in order. Inside the table, **read top to bottom** — that is the
order to run things in. The number inside a command name is part of its name, not
its position in the sequence: `step8-package` runs last and `step9-region` runs
before it, because the commands were numbered as they were added rather than
renumbered every time.

| # | Phase | Command | Needs | Produces |
|---|---|---|---|---|
| 1 | **Set up** | `step1-check` | — | Dataset availability report |
| 2 | | `step2-download` | — | Cached source data |
| 3 | **Calibrate** | `step3-fit` | an inventory | Soil parameters, cross-validated |
| 4 | **Produce** | `step5-susceptibility` | step 3 | Present-day failure probability |
| 5 | **Validate** | `step4-validate` | step 4 + a *second* inventory | Score against a held-out inventory |
| 6 | **Produce** | `step6-hazard` | step 3 | Rainfall and earthquake scenarios |
| 7 | | `step7-climate` | step 3 | CMIP6 futures and the change from today |
| 8 | | `step9-region` | step 3 | The above, one state or province at a time |
| 9 | | `step10-risk` | step 4 | Settlements and road segments, scored per climate |
| 10 | | `step11-map` | step 9 | A browsable Leaflet page of the whole run |
| 11 | **Package** | `step8-package` | anything above | Manifest: products and provenance |

`run-all` executes the whole sequence.

**`step4-validate` runs after `step5-susceptibility`, not before it.** This is the
one place the naming misleads, and it is worth being explicit about: validation
scores a *map*, so a map has to exist first. If the held-out inventory sits in a
different catchment from the map — which is the normal case here, since the
region's inventories are geographically disjoint — pass `--build` and step 4 will
compute a map over that inventory's own extent using the parameters you are
testing. That is transfer validation, and it is the honest test.

**Do not skip phase 2.** A fit always looks good on the landslides it was fitted
to. Nothing in a parameter set tells you whether it travels; only a second
inventory does.

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

### Phase 2 — calibrate

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

Validation comes next, but it needs a map, so **run `step5-susceptibility`
first** — see the phase 3 block below, then come back.

### Phase 2b — validate (after the first map exists)

```bash
# same catchment: score the map you just built
python -m h_sim.cli step4-validate --name gorkha \
    --inventory <an inventory the fit never saw>

# different catchment: build a map over that inventory's ground first
python -m h_sim.cli step4-validate --build --name gorkha --res 0.00083333 \
    --inventory data/raw/inventory/sikkim/Google_Earth_landslides_polygon_21Dec2021.shp
```

The second form is the one you usually want here. The region's inventories do
not overlap — Gorkha spans 84.5–85.3° E and Sikkim 88.1–88.9° E — so a Gorkha map
cannot be scored against Sikkim landslides at all. `--build` applies the fitted
parameters over the held-out inventory's own extent and scores there, which is
transfer validation and the honest test of whether a fit travels. Without it,
step 4 stops and prints the commands to run.

```
  landslides in classes 4-5 : 60.8%  (those classes cover 30.4% of the map)
  efficiency                : 2.00x (>1 means the map concentrates landslides)
  AUC                       : 0.702
  monotonic class ordering  : yes
  VERDICT: fair - ordering holds but discrimination is modest
```

Read the report in this order: **monotonic ordering** first (frequency ratio must
rise with class, or nothing else matters), then **efficiency** (below ~1.5 the map
is not selective enough to act on), then **AUC**.

Two traps worth avoiding. **Never fit and validate on the same inventory.** And
**check the inventory's mapped extent** — background points drawn outside the
ground its authors surveyed mean "nobody looked", not "no landslide". Far-West
Nepal and Sikkim survey only about 60 % of their own bounding boxes, so pass the
polygon:

```bash
python -m h_sim.cli step4-validate --name gorkha_on_target \
    --inventory  data/raw/inventory/sikkim/Google_Earth_landslides_polygon_21Dec2021.shp \
    --survey-extent data/raw/inventory/sikkim/Google_Earth_mapped_extent_21Dec2021.shp
```

Masking does not reliably *raise* the score, and it is not meant to. On this run
it moved AUC from 0.702 to 0.690, because the unsurveyed ground here is mostly
low-susceptibility terrain that was supplying easy true negatives. What it does
is make the negatives mean what they claim to mean, which is the point.

Do not compare these numbers with the 0.780 transfer figure in
[docs/RESULTS.md §3](docs/RESULTS.md). That is a different protocol — 30 m rather
than 90 m, the surveyed polygon's own bounds rather than the inventory's, and
target-group background rather than uniform. Both are honest; they answer
slightly different questions, and only figures produced the same way should be
put side by side.

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

### Phase 3 at regional scale — one province at a time

The region is 4,400 × 2,500 km; at 30 m that is thirteen billion cells, and
flow routing is not tiled. A regional product is therefore a **sweep over
administrative units**, not a single run.

```bash
# what would run, and what it would cost — nothing is computed
python -m h_sim.cli step9-region --dry-run --res 0.00083333 --countries Nepal Bhutan

# then run it
python -m h_sim.cli step9-region --name hkh --res 0.00083333 \
    --fitted-params outputs/gorkha_fitted_params.json
```

States and provinces come from Natural Earth admin-1 (public domain, one 15 MB
download). 165 units fall inside the region across the eight countries. Restrict
with `--countries` or `--units`; add `--with-hazard` or `--with-climate` to run
those per unit too.

**Calibration is regional, production is per unit.** Fit once (phase 2), then
sweep. Each unit reads the same fitted parameters; nothing is refitted per
province.

**Borders cut catchments, so each unit is routed wide and clipped late.** The
run covers the unit's bounding box grown by `admin_buffer_deg`, and the outputs
are masked back to the province afterwards. Without that, a cell just inside a
border gets its catchment truncated at the border and comes out too stable.

The size of that error was measured rather than assumed
(`analysis/07_boundary_buffer.py`):

| Buffer | Cells losing >½ their catchment | Cells shifting P by >0.05 |
|---|---|---|
| none | 1.00 % (3.8 % in the outer ring) | 0.45 % |
| 0.028° (3 km) | **0.00 %** | **0.00 %** |

Smaller than it sounds, because hillslope contributing areas are hundreds of
metres, and cells with genuinely long flow paths are valley floors already
saturated at `w = 1` where more water changes nothing. The default is 0.05°
(5.5 km) — about twice what was needed on steep crystalline terrain, as margin
for flatter ground with longer flow paths.

Two practical points:

- **The sweep is resumable.** A unit whose output exists is skipped. A full
  regional pass is measured in days and something will interrupt it.
- **Oversize units are reported and skipped**, not attempted — above
  `admin_max_cells` (40 M by default), since the alternative is an
  out-of-memory kill part-way through. Xinjiang and Tibet need a coarser grid
  or a basin-level split.
- **A sweep routes roughly twice the cells it keeps.** Provinces are irregular
  and the run is over their bounding boxes: measured on Nepal's Bagmati and
  Bhojpur, only 51 % and 54 % of the box fell inside the province. That is the
  price of rectangular tiling, and it is not currently optimised away.

### Phase 3 — what it means for towns and roads

A susceptibility map says which ground is unstable. It does not say who is under
it. Steps 10 and 11 close that gap.

```bash
python -m h_sim.cli step10-risk --config configs/03_production_gorkha.json
python -m h_sim.cli step11-map  --name gorkha30
```

**Do not sample the map at a town's coordinates.** Towns sit on flat ground —
valley floors, terraces, the insides of meanders — where the factor of safety is
high and failure probability is near zero. The model is right about that: the
ground under the town is not going to slide. What destroys mountain towns is
material arriving *from above*. Sampling at the point answers "safe" for exactly
the settlements most at risk.

**Angle of reach.** Debris from a source can reach a target if the line between
them is steeper than a limiting travel angle α — the Fahrböschung, or Heim
ratio:

```
(z_source − z_target) / horizontal_distance  >  tan α
```

Reported values cluster at 11–25° for channelised debris flows and shallow
slides in mountain terrain, falling with volume (Corominas 1996; Rickenmann
1999; Hunter & Fell 2003). The default is **18°** with a 2 km search radius,
towards the conservative end because this is a screening product.

**The score is a weighted mean, not a maximum.** An earlier version scored an
asset by the highest failure probability among the cells that could reach it.
That number saturates. In Himalayan relief a 2 km radius puts a few thousand
cells above a valley settlement, and if 7 % of the landscape exceeds P = 0.6 —
which is what the Gorkha 30 m fit gives — then the chance that *none* of several
thousand upslope cells does is negligible. Scored that way, **56 % of
settlements in the Gorkha box landed in the top band**, which is not a finding
about the Himalaya, it is an artefact of taking a maximum over a large sample.

The score is instead the **proximity-weighted mean failure probability over the
ground positioned to reach the asset**: of the terrain that could deliver
material here, what fraction does the model call unstable. Weights come from one
geometric argument — a debris path widens roughly in proportion to how far it
has travelled, so a fixed-width target occupies a share of the possible fan that
falls about as 1/d. Nothing else is tuned. `reaching_max` is still reported as a
diagnostic, and is deliberately not banded.

Roads are cut into 500 m segments before scoring, because a way can be fifty
kilometres long and one number for all of it tells a maintainer nothing about
where to go.

On Gorkha at 30 m — 639 settlements, 18,109 road segments, 6,994 km — the
reaching term averages **0.106** against **0.024** for the ground the settlement
stands on, and exceeds it for **67 %** of settlements. Sampling the map at the
point would have called two-thirds of them safe. Full numbers in
[docs/RESULTS.md §11](docs/RESULTS.md).

**Every asset is scored under several climates.** The default is the present day
plus the two CMIP6 windows inside a 20–30 year planning horizon, under an
intermediate and a very high pathway:

```
current   ssp245:2021-2040   ssp585:2021-2040   ssp245:2041-2060   ssp585:2041-2060
```

Change against the present day is the only comparison that means anything: a
future recharge field is normalised by the present-day reference, so its
absolute value is interpretable only relative to today. Override with
`--risk-climate`; the present day is always included.

Measured on Gorkha, that change is **small**: 321 settlements exposed today
against 323 by 2041-2060, and 4,873 km of road against 4,908 km, on a mean score
that moves by 0.0017 — well inside the ±0.020 spread on the model's own held-out
AUC. Note also that SSP2-4.5 comes out slightly *wetter* than SSP5-8.5 over this
box in the 2041-2060 window. That is a property of one GCM over one catchment,
not a result: quote the spread across pathways, not their order.

**Exposure data.** Settlements are OpenStreetMap `place=city|town|village|hamlet`
nodes via Overpass, with GeoNames as a fallback; roads are OSM `highway=*` ways
with geometry, falling back to Natural Earth (trunk routes only, ~1:10 M, and
labelled as such). Both are cached after the first fetch. Overpass rejects
anonymous clients — a missing User-Agent returns 406 from one mirror and a
misleading 429 from another — so the model sends one.

**This is screening, not risk.** Risk is `hazard × exposure × vulnerability` and
only the first two are here. There is no runout model, no damage function, and
nothing converts to expected loss or casualties. Population is carried for
ranking and never multiplied into anything.

`step11-map` assembles a single HTML page: the susceptibility raster per
scenario, the scored settlements and road segments, the training inventory and
its background points, a climate selector that switches every model-dependent
layer, and the summary tables. Leaflet is vendored next to the page and the
vector layers ship as `<script src>` rather than `fetch`, so the page works when
opened straight from disk and without a network — only the basemap tiles need
one.

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
   arrives. Step 10 screens what could arrive at a town or a road with an
   angle-of-reach criterion, which is a geometric bound, not a runout model:
   no volume, no channel geometry, no entrainment, no rheology.
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
   above 40 million cells. `step9-region` works around this by sweeping
   administrative units, but units above that threshold — Xinjiang, Tibet —
   still need a coarser grid or a basin-level split that is not implemented.
7. **Calibration regions do not work as implemented** (−0.0004 AUC held out,
   including where geology is varied). Off by default.
8. **Exposure is screened; risk is not computed.** Risk is
   `hazard × exposure × vulnerability` and step 10 supplies the first two.
   There is no damage function, so nothing converts to expected loss or
   casualties, and population is carried for ranking only. Settlement and road
   coverage is whatever OpenStreetMap has, which is uneven across the region;
   where Overpass is unreachable the road layer falls back to Natural Earth
   trunk routes and is labelled as such.
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
configs/                   Seven run configurations, one per workflow shape
docs/                      Operating guide and measured results
scripts/run_demo.sh        Offline walk through all four phases
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
| Inventories | Gorkha, Far-West Nepal, Sikkim, NASA GLC | Points, polygons | Automatic |
| Earthquake PGA | GEM seismic hazard map | — | Manual, or scenario value |
| States and provinces | Natural Earth 1:10m admin-1 | ~1:10 M | Automatic, 15 MB once |

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
