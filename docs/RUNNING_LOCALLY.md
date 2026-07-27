# Running the HKH landslide model locally

A step-by-step guide to installing and running the model on your own machine,
from an offline smoke test through to a 90 m production run over the Hindu Kush
Himalaya with calibrated weights.

---

## 0. What you need

| | Requirement |
|---|---|
| **Python** | 3.9 or newer (3.11+ recommended) |
| **RAM** | 4 GB is enough — the model streams data in tiles, it never loads a whole region into memory |
| **Disk** | ~5 GB with the robust defaults (full GLiM 2.3 GB + WorldClim 30s 2.0 GB, both one-off and cached); ~0.5 GB if you opt down with `--glim-grid --worldclim-res 10m` |
| **Network** | Only for the download steps; the offline demo needs none |
| **OS** | Linux, macOS or Windows (WSL recommended on Windows) |

The heavy dependency is **rasterio** (it bundles GDAL). If `pip install rasterio`
fails to build on your system, use conda instead:

```bash
conda create -n hkh python=3.11 -c conda-forge rasterio fiona numpy matplotlib requests pytest
conda activate hkh
```

---

## 1. Install

```bash
git clone <your-repo-url> Himalayas_LandslideRisk
cd Himalayas_LandslideRisk

./setup.sh                 # creates .venv, installs deps, runs the tests
#   or, to also pre-fetch the ~2.2 GB robust datasets up front:
# ./setup.sh --with-glim
```

Doing it manually instead:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/ -q         # expect: 12 passed
```

**Activate the environment in every new shell** — `source .venv/bin/activate`.

---

## 2. Offline smoke test (no downloads)

Confirm the install works before spending bandwidth:

```bash
./run_demo.sh
```

This fabricates plausible synthetic terrain over a Himalayan AOI and runs the
complete chain for both triggers plus a calibration. Takes well under a minute.

Check `outputs/hkh_demo_rainfall_quicklook.png` — you should see a two-panel
image: susceptibility classes on the left, hazard probability on the right.

> Everything in this step is synthetic. It proves the *code* works, not the
> science. Real conclusions require step 3 onward.

---

## 3. First real-data run (~5 minutes, ~120 MB)

```bash
python -m giri_landslide.cli run --mode download \
    --config examples/01_hkh_quickstart.json
```

This covers a 1.0° × 0.8° window over the Himachal Pradesh mountain front at
250 m. It **deliberately opts down** from the robust defaults so your first run
is quick:

| Dataset | Size |
|---|---|
| Copernicus GLO-90 DEM (2 tiles) | ~10 MB |
| ESA WorldCover 2021 (1 tile) | ~100 MB |
| WorldClim v2.1 monthly precipitation (10′) | ~7 MB |
| GLiM 0.5° lithology grid | <1 MB |

Downloads are cached in `data/raw/` and reused by every later run.

Expected output:

```
[giri] grid                   400x320 px @ 0.0025 deg (rainfall, weights=multiplicative)
[giri] download:dem           copernicus90
...
[giri] hazard                 applying hazard matrix
[giri] done                   outputs in outputs
```

Open `outputs/hkh_quickstart_quicklook.png`. Over this AOI you should see the
flat Punjab plains in the southwest as class 1 and the steep ranges to the
northeast as class 4–5, with the mountain front sharply delineated.

---

## 4. Robust production run at 90 m (~2.2 GB first time)

3 arc-seconds (~90 m) is the resolution used in the GIRI manuscript. This config
uses the **package defaults**, which are the robust ones:

```bash
python -m giri_landslide.cli run --mode download \
    --config examples/02_hkh_90m_rainfall.json
```

On the first run this additionally downloads, once, and then caches forever:

| Dataset | Size | Why it matters |
|---|---|---|
| **Full GLiM geodatabase** (1,235,259 polygons) | 1.14 GB | the 0.5° grid is one class per ~55 km cell — effectively uniform at hillslope scale |
| **WorldClim 30s** (~1 km) | 1.03 GB | coarser products smear orographic rainfall gradients across whole ranges |

Both are fetched automatically. To pre-fetch them explicitly:

```bash
python -m giri_landslide.cli download --bbox 76.0 30.5 77.0 31.3
```

GLiM extracts to `data/raw/glim/LiMW_GIS 2015.gdb`, needs `fiona`, and ships in
Eckert IV (`ESRI:54012`) — the rasteriser reprojects it to your grid for you.

If you are bandwidth- or disk-constrained, opt down explicitly:

```bash
python -m giri_landslide.cli run --mode download --name lean \
    --bbox 76.0 30.5 77.0 31.3 --res 0.0008333 \
    --glim-grid --worldclim-res 10m
```

Earthquake scenario over the same AOI:

```bash
python -m giri_landslide.cli run --mode download \
    --config examples/03_hkh_90m_earthquake.json
```

**Scaling guidance.** Cost grows with the square of the resolution:

| AOI | Resolution | Grid | Rough runtime |
|---|---|---|---|
| 1° × 0.8° | 250 m | 400 × 320 | seconds |
| 1° × 0.8° | 90 m | 1200 × 960 | 1–2 min |
| 5° × 4° | 90 m | 6000 × 4800 | ~30 min |
| Whole HKH | 90 m | ~54000 × 27600 | many hours; run it in tiles |

RAM stays flat regardless — only disk and time grow. Lower `block_size` (e.g.
512) if you are memory constrained.

### How fine can you actually go?

`--res` accepts any value, but the *meaningful* resolution is set by the inputs:

| Factor | Source | Native resolution | Note |
|---|---|---|---|
| Slope | Copernicus GLO-90 | **92.8 m** | `--dem-source copernicus30` gives 30 m (~8× the download) |
| Vegetation | ESA WorldCover | **9.3 m** | downsampled to your grid; never the constraint |
| Lithology | GLiM vector | polygons | compiled from ~1:1M geological maps — sharp edges, coarse content |
| Soil moisture | WorldClim 30s | **927.7 m** | the finest open monthly climatology; a hard floor |

Slope dominates the model, so **the DEM sets the practical ceiling**.

> ### Finer is not automatically better
>
> Slope is scale-dependent: the same hillside measured on a 30 m DEM is steeper
> than on a 90 m DEM. Measured over a Himachal Pradesh AOI:
>
> | | 90 m | 30 m |
> |---|---|---|
> | mean slope | 1.24° | **2.61°** |
> | 95th-percentile slope | 5.43° | **11.81°** |
> | area in susceptibility class ≥ 4 | 0.40 % | **5.32 %** |
>
> That is a **13× jump in "high susceptibility" area** from changing the DEM
> alone — not new information, but the slope table (Table 2, calibrated on ~90 m
> statistics) being applied to a distribution it was not built for.
>
> So: run at 90 m for results consistent with the published model. If you need
> 30 m, **re-calibrate `SLOPE_BREAKS_DEG` in `config.py` against an inventory at
> that resolution first** — otherwise you get false precision with an inflated
> hazard.

---

## 5. Future-climate scenarios

The model can produce hazard maps for **present and future climate**. Adding
`--climate <ssp>` switches the soil-moisture factor from the WorldClim
1970–2000 baseline to downscaled CMIP6 projections, giving a future
susceptibility — and therefore hazard — map:

```bash
# present day (baseline)
python -m giri_landslide.cli run --mode download --name base \
    --bbox 76.0 30.5 77.0 31.3 --res 0.0025 \
    --climate current --worldclim-res 2.5m

# end-of-century, high-emissions
python -m giri_landslide.cli run --mode download \
    --config examples/05_hkh_future_climate.json
```

Scenarios: `ssp126`, `ssp245`, `ssp370`, `ssp585`. Periods: `2021-2040`,
`2041-2060`, `2061-2080`, `2081-2100`. The default model is **IPSL-CM6A-LR**,
the one used in the GIRI manuscript (`--climate-model` to change it).

Measured over the Himachal Pradesh AOI (IPSL-CM6A-LR, 2061–2080):

| Scenario | Mean wettest-month rainfall | Change | Very-High susceptibility area |
|---|---|---|---|
| current | 304.7 mm | — | 861 px |
| ssp126 | 379.2 mm | **+24.4 %** | 1148 px |
| ssp585 | 329.0 mm | **+8.0 %** | 1032 px |

Two things to take from that:

- **Both futures are wetter here, so susceptibility rises.** That is the
  climate-change signal the model is designed to capture.
- **SSP1-2.6 comes out wetter than SSP5-8.5 *at this location*.** That is not a
  bug: monsoon rainfall does not respond monotonically to forcing, and it gets
  spatially redistributed — the manuscript notes the same counterintuitive
  behaviour. Never assume a higher-emissions scenario means more rain in a
  particular valley; check the map.

> **Compare like with like.** The CMIP6 files are 2.5′ (~4.6 km) while the
> current-climate default is 30″ (~1 km). Run your baseline with
> `--worldclim-res 2.5m` so a present-vs-future difference reflects climate,
> not resampling. The 30″ CMIP6 files exist but are ~22 GB *each*.

### Mapping the change directly

`compare` differences two susceptibility runs — the manuscript's Figure 8:

```bash
python -m giri_landslide.cli compare \
    --baseline outputs/base_susceptibility.tif \
    --scenario outputs/hkh_ssp585_2061_2080_susceptibility.tif \
    --name ssp585_vs_present
```

It writes a class-change GeoTIFF (positive = more susceptible), a diverging
quicklook PNG, and a JSON summary with the share of area that moved up or down.
For the Himachal AOI: **+0.43 % of pixels under SSP1-2.6** and **+0.24 % under
SSP5-8.5**, all increases, concentrated on the mountainous northeast — the
plains do not move because flat ground is pinned at class 1.

> Those percentages look small because the soil-moisture class boundaries are
> coarse (125/250/500/1000 mm), so a +24 % rainfall change only tips pixels that
> were already near a boundary. Read the map, not just the headline number: the
> changes cluster exactly where the terrain is already steep.

> **What does not change with scenario.** Only the *susceptibility* side moves.
> The triggering return period keeps its present-day meaning — a "100-year
> storm" is defined against today's climate — because the terrain takes
> centuries to adapt to a new regime. This follows the manuscript. It also means
> the model captures changes in the *background wetness*, not changes in the
> frequency of extreme storms; if storm intensity shifts too, the real change in
> hazard could be larger than shown.

---

## 6. Calibrate the weights against historical landslides

Fine-tune the factor weights to observed landslides instead of using the
defaults:

```bash
python -m giri_landslide.cli calibrate --mode download \
    --config examples/04_hkh_calibrate.json
```

This downloads the **NASA COOLR** inventory for the HKH, samples the factor
rasters at landslide and background points, fits a logistic model, and writes:

- `outputs/hkh_calibrated_calibration.json` — weights, held-out ROC AUC, warnings
- `outputs/hkh_calibrated_calibrated_config.json` — a ready-to-run config

Then run the model with the calibrated weights:

```bash
python -m giri_landslide.cli run --mode download \
    --config outputs/hkh_calibrated_calibrated_config.json
```

### Reading the calibration output honestly

- **CV AUC > 0.8** — good discrimination; weights are usable.
- **CV AUC 0.6–0.7** — weak. Usually too few or too clustered inventory points.
- **A `!` warning line** — read it. Two matter most:
  - *"uninformative (near-constant) factors excluded"* — that layer does not
    vary over your AOI (e.g. the 0.5° GLiM grid). Get the full GLiM (step 4).
  - *"negatively associated with mapped landslides"* — almost always **spatial
    reporting bias**: citizen-science landslide reports cluster near roads and
    towns, so presence points can be *less* steep than random background. The
    model already density-matches the background to reduce this, but a small,
    clustered inventory can still produce a meaningless fit.

**Use a wide AOI with many inventory points.** COOLR has ~22,000 points across
the HKH but they are very unevenly distributed — thousands in Myanmar and
Bangladesh, only ~176 in the Indian Himalaya. Calibrating on a few hundred
clustered points will not give trustworthy weights.

### Which inventory to use — measured comparison

Three inventories were run through the identical pipeline over Himalayan AOIs:

| Inventory | Points | How mapped | CV AUC | Fitted weights (slope/litho/veg/moisture) | Clamped |
|---|---|---|---|---|---|
| [NASA GLC / COOLR](https://landslides.nasa.gov) (Myanmar) | 8,634 | media reports | 0.640 | 0.78 / 0.00 / 0.00 / 3.22 | litho, veg |
| [Roback Gorkha](https://www.sciencebase.gov/catalog/item/582c74fbe4b04d580bd377e8) (Nepal) | 24,794 | satellite, earthquake | **0.701** | 1.28 / 0.00 / 0.43 / 2.28 | litho |
| [Far-West Nepal](https://doi.org/10.5281/zenodo.4290100) (Zenodo) | 26,348 | satellite, multi-temporal monsoon | 0.563 | **0.38 / 0.54 / 0.56 / 2.52** | **none** |

Read this carefully — **the highest AUC is not the best inventory.**

- **GLC/COOLR is media-report-derived.** A landslide enters it if someone wrote
  about it, which requires roads and settlements nearby. Lithology *and*
  vegetation come out negatively correlated: an artefact of where reporters are,
  not of where slopes fail. Fine for visual validation, poor for calibration.
- **Roback (Gorkha) scores best** and gives the cleanest slope curve, but it is a
  single earthquake, so its weights describe seismic triggering.
- **Far-West Nepal is the only run where all four factors come out positive**,
  with soil moisture dominant — exactly what theory predicts for rainfall
  triggering. Yet it has the *lowest* AUC.

### Why a good inventory can score a bad AUC

The presence/background factor contrast explains it:

| Inventory | Total presence-vs-background contrast |
|---|---|
| GLC/COOLR | 0.384 |
| Roback Gorkha | 0.369 |
| Far-West Nepal | **0.106** |

Far-West Nepal contains **2.4 landslides per km²** — the terrain is *saturated*.
Background points drawn from the same small AOI land on ground that has also
failed, or is equally failure-prone, so presence and background become
statistically near-identical and AUC collapses toward 0.5. That is a **sampling
artefact, not a bad model**.

**Practical guidance:**

1. Prefer **systematically mapped** (satellite) inventories over reported ones.
2. Match the inventory's **trigger** to the model run — earthquake inventories
   for `--trigger earthquake`, monsoon inventories for rainfall.
3. On a saturated inventory, **do not judge the model by AUC**. Use the
   frequency-ratio slope fit below, which stayed physically sensible in all
   three runs even where AUC did not.
4. Treat a factor clamped to 0 as a **red flag about the inventory**, not a
   finding about the terrain.

### Fitting the slope classes too (needed for non-90 m runs)

By default the slope table is the manuscript's Table 2, calibrated on ~90 m
slope statistics. Section 4 showed that reusing it on a 30 m DEM inflates the
high-susceptibility area 13×. `--fit-slope-breaks` derives the table from your
own inventory instead, at whatever DEM resolution you are running:

```bash
python -m giri_landslide.cli calibrate --mode download \
    --config examples/04_hkh_calibrate.json \
    --dem-source copernicus30 --res 0.00027778 \
    --fit-slope-breaks
```

It uses the **frequency-ratio** method: for each slope bin, compare the share of
landslides against the share of terrain. Bins where landslides are
over-represented score high. This reproduces the manuscript's non-monotonic
shape *only if your data shows it* — very steep terrain that has already shed
its regolith drops out on its own rather than by assumption. Slopes below 6°
are always pinned to 0, which is a physical constraint, not a fitted one.

The fitted table is written into the calibrated config as `slope_breaks` and the
per-bin frequency ratios land in the calibration report so you can sanity-check
the curve. **This is the step that makes a 30 m run defensible.**

---

## 7. Where everything lands

```
data/raw/      downloaded source data (cached, safe to delete to reclaim space)
data/work/     per-run intermediate rasters — inspect these to debug a stage
outputs/       final products
```

For a run named `X`:

| File | Meaning |
|---|---|
| `X_susceptibility.tif` | 5-class susceptibility (1 = Very Low … 5 = Very High) |
| `X_hazard_probability.tif` | probability of a damaging landslide per scenario event |
| `X_summary.json` | grid info, weights, class histogram, hazard stats |
| `X_quicklook.png` | two-panel visual check |

Intermediates in `data/work/` (`X_f_slope.tif`, `X_f_litho.tif`, `X_f_veg.tif`,
`X_f_soil.tif`, `X_susc_index.tif`, …) let you verify any single stage. Open the
GeoTIFFs in QGIS — they are standard EPSG:4326 rasters.

---

## 8. Customising a run

Copy an example config and edit it, or override on the command line:

```bash
python -m giri_landslide.cli run --mode download \
    --name my_run --bbox 85.0 27.0 86.0 28.0 --res 0.0008333 \
    --trigger rainfall --return-period 200
```

Useful flags: `--pga` (earthquake scenario, g), `--return-period` (rainfall,
years), `--weight-mode {multiplicative,exponent}`,
`--classification {fixed,quantile}`, `--block` (tile size), `--inventory`.

All model constants — the reclassification tables, factor weights, trigger
thresholds and hazard matrices — live in `giri_landslide/config.py`. Edit them
there to re-calibrate the model itself.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `rasterio` fails to install | Use the conda route in section 0 |
| `ModuleNotFoundError: giri_landslide` | Run from the repo root, or `pip install -e .` |
| `AOI ... is outside the Hindu Kush Himalaya region` | Your `--bbox` is outside `(60E,16N)–(105E,39N)`. Widen `region_bbox` in a config to model elsewhere |
| `No Copernicus DEM tiles found` | AOI is all ocean, or the network is blocked. Check the bbox order: `W S E N` |
| `TIFFReadDirectory: Cannot handle zero number of tiles` | A previous run left a truncated file — delete `data/work/<name>_*` and re-run |
| Lithology looks uniform | You passed `--glim-grid`; drop it to use the full GLiM (section 4) |
| Calibration returns odd weights | Read section 6; usually inventory bias or too few points |
| Run is slow | Coarsen `--res`, shrink the AOI, or tile it |

Re-running is cheap: downloads are cached and every intermediate is written to
disk, so you can stop a run, change one table, and re-run only what you need.
