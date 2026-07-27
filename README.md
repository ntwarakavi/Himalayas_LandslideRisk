# Hindu Kush Himalaya Landslide Model

An open-source implementation of the **GIRI landslide model** (Palau, Nadim,
Paulsen & Storrøsten, 2023, NGI / CDRI —
[manuscript](https://giri.unepgrid.ch/sites/default/files/2023-06/20230615-NGI_manuscript_GIRI_landlside_hazard_model.pdf)),
scoped to the **Hindu Kush Himalaya**: Afghanistan, Pakistan, India, Nepal,
Bhutan, Bangladesh, China and Myanmar.

It runs on a normal laptop. Data is downloaded only for your area of interest,
processed in tiles, and **never re-downloaded once cached**.

---

## First: what the model actually produces

These three words get used interchangeably in casual talk. They are different
things, and the model produces the first two.

### 1. Susceptibility — *"where is the ground fragile?"*

A property of the **landscape**. It does not change from day to day. It asks:
if something were to shake or soak this hillside, how ready is it to fail?

Built from four things: **how steep** the ground is, **what rock** is under it,
**what is growing** on it, and **how wet** it usually gets. Output is a map with
5 classes, Very Low → Very High.

> A steep, deforested slope on weak sandstone is *highly susceptible* — even
> though nothing is happening there today.

### 2. Hazard — *"if a storm or earthquake hits, how likely is a landslide?"*

Susceptibility **plus a trigger of a stated size**. This is a probability, and
it is always tied to a specific scenario you choose — a 100-year storm, or a
0.35 g earthquake.

> That same slope has a *5% chance* of failing in a 1-in-100-year storm.

Hazard is **conditional**. The model does not forecast the storm; you specify
it. Change the scenario, get a different hazard map.

### 3. Risk — *"what would it cost?"* — **not implemented**

Risk = hazard × **exposure** × **vulnerability**. It needs to know what is in
harm's way — roads, buildings, people — and how badly each is damaged.

> That slope sits above 2 km of highway carrying 5,000 vehicles a day, so the
> expected annual loss is £X.

**This model stops at hazard.** Despite the repository name, the exposure layer
is not built yet. See [What is missing](#what-is-missing).

```
   TERRAIN DATA                 SUSCEPTIBILITY            HAZARD              RISK
   slope, rock,      ───────►   "where is the    ──┐
   plants, wetness               ground weak?"      │
                                   (step 4)         ├──►  "how likely,   ──►  "what does
   TRIGGER SCENARIO                                 │      in THIS         it cost?"
   storm size, or    ─────────────────────────────┘       scenario?"        NOT BUILT
   earthquake size                                         (step 5)
```

---

## Install

```bash
git clone https://github.com/ntwarakavi/Himalayas_LandslideRisk.git
cd Himalayas_LandslideRisk
./setup.sh                      # virtualenv, dependencies, tests
source .venv/bin/activate
```

If `pip install rasterio` fails, use conda:

```bash
conda create -n hkh python=3.11 -c conda-forge rasterio fiona numpy matplotlib requests pytest
conda activate hkh
```

Check it works, offline, in under a minute:

```bash
./run_demo.sh                   # synthetic data, no downloads
```

---

## The run sequence

Six steps. Each writes files and tells you what to run next. **Stop after any
step, inspect the output, carry on.** Steps 1–2 are one-off; 3 is optional;
4–5 are the model.

### Step 1 — Check what data you have

```bash
python -m giri_landslide.cli step1-check
```

Reports every dataset as `CACHED` / `READY` / `BLOCKED` / `MANUAL`, grouped by
what it is used for, and totals the download still outstanding. Probes the
network; add `--offline` to only look at the cache.

*Produces:* nothing — it just tells you where you stand.

### Step 2 — Download

```bash
python -m giri_landslide.cli step2-download --bbox 84.0 28.0 84.6 28.5
```

Fetches terrain, climate and landslide-inventory data for your area.
**Anything already on disk is skipped**, so re-running is cheap and safe.
First full run is ~2.2 GB (mostly the one-off global GLiM and WorldClim files);
after that, a new area only pulls its own DEM and land-cover tiles.

*Produces:* files under `data/raw/`.

### Step 3 — Calibrate the weights *(optional but recommended)*

```bash
python -m giri_landslide.cli step3-calibrate \
    --bbox 84.1 27.1 86.95 28.75 --res 0.0025 \
    --inventory "data/raw/inventory/roback/Roback_Nepal_final_files/Source20170209.shp"
```

The four factors do not matter equally, and the manuscript never published its
weights. This step **learns them from real landslides**: it takes an inventory
of places that *did* fail, samples the factors there and at random background
points, and fits a logistic model. The coefficients are the weights.

It also refits the **slope table** — how much each steepness band contributes —
by frequency ratio.

*Produces:* `outputs/<name>_calibrated_config.json` (feed it to step 4) and a
report with the weights, the cross-validated AUC and any warnings.

**Read the AUC**: 0.5 = no skill, 0.7 = fair, 0.8 = good. And read the
warnings — see [Choosing an inventory](#choosing-an-inventory).

### Step 4 — Susceptibility map

```bash
python -m giri_landslide.cli step4-susceptibility \
    --name nepal --bbox 84.0 28.0 84.6 28.5 --res 0.0008333
```

Turns each input into a 0–5 factor score and combines them into a **continuous
0–1 susceptibility index** — the fitted logistic model evaluated at every pixel,
so there are no arbitrary class breaks and any two pixels are directly
comparable. Add `--config outputs/..._calibrated_config.json` to use step 3's
weights. `--output classes` or `--output both` also writes the 5-class map.

> The index is **relative, not an absolute probability of failure**. It is fitted
> against background points standing in for absences, so its level depends on how
> many background points were drawn. 0.8 versus 0.4 tells you the ordering and
> the separation, not that the first fails 80% of the time.

*Produces:* `outputs/<name>_susceptibility.tif` plus the four factor rasters in
`data/work/` so you can check any single input.

### Step 5 — Hazard for a scenario

```bash
python -m giri_landslide.cli step5-hazard --name nepal --return-period 100
# or an earthquake:
python -m giri_landslide.cli step5-hazard --name nepal --trigger earthquake --pga 0.35
```

Takes step 4's map, classifies your chosen trigger into 5 severity bands, and
looks the pair up in a hazard matrix to get a probability per pixel.

*Produces:* `outputs/<name>_hazard_probability.tif` and a two-panel
`<name>_quicklook.png`.

### Step 6 — Compare two scenarios *(optional)*

```bash
python -m giri_landslide.cli step6-compare --name future_vs_now \
    --baseline outputs/now_susceptibility.tif \
    --scenario outputs/ssp585_susceptibility.tif
```

*Produces:* a class-change map (red = worse), plus the share of area that moved.

### All at once

```bash
python -m giri_landslide.cli run-all --name nepal \
    --bbox 84.0 28.0 84.6 28.5 --res 0.0008333 --return-period 100
```

---

## Reading the outputs

| File | What it means |
|---|---|
| `<name>_susceptibility_prob.tif` | **Continuous 0–1 susceptibility index** (default). Straight from the fitted logistic model — no class breaks, every pixel comparable. |
| `<name>_susceptibility.tif` | 1 = Very Low … 5 = Very High. Built only when a hazard step follows, since the hazard matrix is indexed by class. |
| `<name>_hazard_probability.tif` | Probability of a damaging landslide **in the scenario you specified**. |
| `<name>_quicklook.png` | Two panels: susceptibility, then hazard. |
| `<name>_summary.json` | Grid, weights, class histogram, hazard statistics. |
| `data/work/<name>_f_*.tif` | The four individual factor scores — check these when a result looks wrong. |

Everything is EPSG:4326 GeoTIFF; open it in QGIS.

---

## Choosing an inventory

Step 3 is only as good as the landslide data behind it. Measured, same pipeline:

| Inventory | Points | How mapped | AUC | Verdict |
|---|---|---|---|---|
| NASA GLC (screened) | 295 in HKH at ≤1 km | media reports | — | usable **only after filtering by `location_accuracy`**; 2/3 of records are 5 km+ |
| Roback Gorkha, Nepal | 24,794 | satellite, earthquake | **0.701** | best for `--trigger earthquake` |
| Far-West Nepal | 26,348 | satellite, monsoon | 0.563 | best for rainfall — see note |
| Southern Sikkim, India | 255 | satellite | — | small; regional check |

All are downloaded by step 2. Two things that look like paradoxes but are not:

- **The best AUC is not the best inventory.** Far-West Nepal scores lowest yet
  is the *only* one where all four factors come out positive with soil moisture
  dominant — exactly what rainfall-triggered failure should look like. Its low
  score comes from containing 2.4 landslides/km²: the terrain is saturated, so
  background points land on ground that also failed and the two classes stop
  being distinguishable. That is a sampling artefact, not a bad model.
- **A factor forced to zero is a warning about your inventory,** not a
  discovery about geology. COOLR makes lithology look protective, which is
  reporter geography, not rock.

Match the inventory's trigger to the run: earthquake inventories for
earthquakes, monsoon inventories for rainfall.

---

## Climate scenarios

```bash
python -m giri_landslide.cli step4-susceptibility --name ssp585 \
    --climate ssp585 --climate-period 2061-2080
```

Swaps the wetness factor to downscaled CMIP6 projections (default model
IPSL-CM6A-LR, as in the manuscript). Scenarios `ssp126|ssp245|ssp370|ssp585`.

Only susceptibility responds — the trigger return period keeps its present-day
meaning, because terrain takes centuries to adapt. So this captures a change in
**background wetness**, not a change in storm frequency.

For a fair comparison run the baseline at the same precipitation resolution
(`--climate current --worldclim-res 2.5m`); details and measured deltas in
[`docs/RUNNING_LOCALLY.md`](docs/RUNNING_LOCALLY.md).

---

## Data

| Role | Dataset | Resolution | How |
|---|---|---|---|
| Slope | Copernicus GLO-90 DEM | 90 m | automatic |
| Vegetation | ESA WorldCover 2021 | 10 m | automatic |
| Lithology | GLiM (1.2 M polygons) | vector | automatic (1.1 GB, once) |
| Wetness | WorldClim v2.1 | 1 km | automatic (1.0 GB, once) |
| Future wetness | WorldClim CMIP6 | 4.6 km | automatic |
| Inventories | Gorkha, Far-West Nepal, Sikkim, COOLR | points/polygons | automatic |
| Earthquake PGA | GEM seismic hazard map | — | **manual**, or use `--pga` |

`python -m giri_landslide.cli info` prints sources and licences.

**Why 90 m and not 30 m:** the slope table was calibrated on ~90 m statistics.
Switching to a 30 m DEM makes the same hillside measurably steeper and inflates
the high-susceptibility area **13×** — false precision. If you need 30 m, refit
the slope table first with `step3-calibrate --fit-slope-breaks`.

---

## What is missing

Honest status for anyone planning to rely on this.

1. **Risk.** No exposure layer (roads, buildings, population), so no expected
   losses. This is the largest gap.
2. **The rainfall hazard matrix is a placeholder.** The manuscript publishes it
   only as a figure, so the numbers in `config.py` are illustrative. The
   *earthquake* matrix is transcribed exactly. Until this is calibrated,
   **relative patterns are meaningful; absolute rainfall probabilities are
   not.**
3. **Real PGA and ERA5 soil moisture** fall back to uniform values unless you
   supply them.
4. **No independent validation.** Calibration AUC is not validation; the model
   has not been back-tested against a held-out event.
5. **No whole-HKH tiling driver.** Region-wide at 90 m is ~1.5 gigapixels and
   must currently be run as manual tiles.

---

## Layout

```
giri_landslide/
  datasets.py       registry of every dataset + cache/availability checks
  sources.py        downloaders (all cache-first)
  inventory.py      landslide inventories: load, download, sample
  grid.py           the reference grid, warping, tiled processing
  factors.py        inputs  -> the four 0-5 factor scores
  susceptibility.py factors -> the 5-class susceptibility map      (step 4)
  triggers.py       storm/earthquake -> 5 severity classes         (step 5)
  hazard.py         susceptibility x trigger -> probability        (step 5)
  calibrate.py      fit weights and the slope table to an inventory (step 3)
  pipeline.py       orchestration
  cli.py            the six steps
config.py           every table, weight and threshold, in one place
docs/RUNNING_LOCALLY.md   the long-form guide
examples/           ready-made configs, numbered to match the steps
tests/              15 offline tests
```

All model constants live in `giri_landslide/config.py` — reclassification
tables, weights, trigger thresholds, hazard matrices. Change the model there.

```bash
python -m pytest tests/ -q       # 15 passing, no network needed
```

## Licence

MIT. Each dataset keeps its own licence (Copernicus, ESA, WorldClim, GLiM, GEM,
NASA, Zenodo records) — cite them when you publish.
