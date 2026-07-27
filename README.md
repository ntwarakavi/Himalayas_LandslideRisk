# GIRI Landslide Hazard Model (open-source, local, modular)

An end-to-end, **open-source** Python implementation of the global landslide
susceptibility and scenario-based hazard methodology developed by the Norwegian
Geotechnical Institute for the **Global Infrastructure Resilience Index (GIRI)**
of the Coalition for Disaster Resilient Infrastructure (CDRI):

> Palau, R. M., Nadim, F., Paulsen, E., Storrøsten, E. (2023).
> *A new model for global landslide susceptibility assessment and scenario-based
> hazard assessment.* NGI / GIRI.
> [manuscript PDF](https://giri.unepgrid.ch/sites/default/files/2023-06/20230615-NGI_manuscript_GIRI_landlside_hazard_model.pdf)

The pipeline runs **piece by piece on a modest local computer**: it downloads
only the open-data tiles that intersect your area of interest (AOI) and
processes everything in memory-bounded blocks, so a laptop can produce a
90 m-resolution hazard map for a region without ever loading a global array.

**Regional scope.** This build is **restricted to the Hindu Kush Himalaya (HKH)**
as defined by ICIMOD — the mountain arc spanning **Afghanistan, Pakistan, India,
Nepal, Bhutan, Bangladesh, China and Myanmar**, region bounds
`(60°E, 16°N) – (105°E, 39°N)`. Any AOI is automatically clipped to this region,
and historical inventories are filtered to it (bbox + country list). Change
`region_bbox` / `HKH_COUNTRIES` in a config to adjust.

**Data-driven weight calibration.** The factor weights can be **fine-tuned
against a historical Himalayan landslide inventory** (NASA Global Landslide
Catalog / COOLR, or your own) by logistic regression — see
[Calibrating the weights](#calibrating-the-weights-with-historical-himalayan-data) below.

Every run writes a two-panel quicklook (`outputs/*_quicklook.png`) — left:
5-class susceptibility (green→red); right: scenario landslide probability. On
real HKH data the model resolves the mountain front sharply: plains and river
valleys as class 1, steep hillslopes as class 4–5.

---

## What the model does

```
 DEM ───► slope ───► slope factor  (Sr)          ┐
 GLiM ──────────────► lithology factor (Sl)      │   S = Π wᵢ·f(Sᵢ)      5-class
 Land cover ────────► vegetation factor (Sv)     ├─►  weighted product ─► susceptibility
 Precip / VWC ──────► soil-moisture factor (Sp)  ┘        (Eq. 1)          (1..5)
                                                                              │
 Rainfall 24 h  ─► normalise ─► Gumbel return period ─► rainfall class 1..5   │
   or                                                                         ├─► hazard
 Earthquake PGA ──────────────────────────────► seismic class 1..5           │    matrix
                                                                              ▼
                                            landslide probability per scenario event
```

Every reclassification table, weight, threshold and hazard matrix is transcribed
from the manuscript into [`giri_landslide/config.py`](giri_landslide/config.py)
and can be re-calibrated there or via a JSON config.

## Quick start

```bash
./setup.sh          # venv + dependencies + tests
source .venv/bin/activate

./run_demo.sh       # offline smoke test, no downloads

# first real-data run (fast, ~120 MB)
python -m giri_landslide.cli run --mode download \
    --config examples/01_hkh_quickstart.json
```

**→ Full step-by-step guide: [`docs/RUNNING_LOCALLY.md`](docs/RUNNING_LOCALLY.md)**

### Example configs

| Config | What it does |
|---|---|
| `examples/01_hkh_quickstart.json` | fast first run, 250 m, small downloads |
| `examples/02_hkh_90m_rainfall.json` | **robust production run**, 90 m, full GLiM |
| `examples/03_hkh_90m_earthquake.json` | earthquake scenario, 90 m, PGA 0.35 g |
| `examples/04_hkh_calibrate.json` | weight calibration vs. the COOLR inventory |
| `examples/05_hkh_future_climate.json` | **future-climate** hazard (CMIP6 SSP scenarios) |

## Robust-by-default configuration

The defaults are chosen for **scientific robustness, not the smallest download**.
Each is a deliberate trade-off:

| Default | Why | Opt down with |
|---|---|---|
| **90 m grid** (3 arc-sec) | the manuscript's native resolution | `--res 0.0025` |
| **Copernicus GLO-90 DEM** — *not* GLO-30 | the slope table (Table 2) was calibrated on ~90 m slope statistics; a finer DEM yields systematically steeper slopes and would silently bias every class | `--dem-source copernicus30` (re-calibrate `SLOPE_BREAKS_DEG` too) |
| **Full GLiM geodatabase** (1.2M polygons, 1.1 GB) | the 0.5° grid is a single class per ~55 km cell — effectively uniform at hillslope scale | `--glim-grid` |
| **WorldClim 30s** (~1 km, 1 GB) | coarser products smear orographic rainfall gradients across whole ranges | `--worldclim-res 10m` |
| **5-fold cross-validated calibration** | a single hold-out split is noisy and often optimistic on small, clustered inventories | — |
| **Density-matched background sampling** | controls the accessibility bias of citizen-science inventories | — |

First robust run downloads ~2.2 GB, then caches everything in `data/raw/`.

This writes to `./outputs/`:

| file | meaning |
|------|---------|
| `*_susceptibility.tif` | 5-class susceptibility map (1 = Very Low … 5 = Very High) |
| `*_hazard_probability.tif` | probability of a significant landslide per scenario event |
| `*_trigger_class.tif` | rainfall / seismic trigger class |
| `*_summary.json` | run metadata + class histogram + hazard stats |
| `*_quicklook.png` | two-panel visual check |

## Climate scenarios

The model runs for **present and future climate**. `--climate <ssp>` swaps the
soil-moisture factor from the WorldClim 1970–2000 baseline to downscaled CMIP6
projections (default model **IPSL-CM6A-LR**, as in the manuscript):

```bash
python -m giri_landslide.cli run --mode download \
    --config examples/05_hkh_future_climate.json      # ssp585, 2061-2080
```

Scenarios `ssp126|ssp245|ssp370|ssp585`; periods 2021-2040 … 2081-2100. Only the
*susceptibility* side changes — the triggering return period keeps its
present-day meaning, following the manuscript. See
[`docs/RUNNING_LOCALLY.md`](docs/RUNNING_LOCALLY.md) §5 for measured
present-vs-future deltas and the like-for-like comparison caveat.

## Open datasets used

| Factor | Default dataset (no login) | Resolution | Access |
|--------|----------------------------|-----------|--------|
| Slope | **Copernicus GLO-90 DEM** (or GLO-30) | 90 m / 30 m | AWS Open Data, auto |
| Vegetation | **ESA WorldCover 2021 v200** | 10 m | AWS Open Data, auto |
| Soil moisture (rainfall) | **WorldClim v2.1** monthly precip → max-monthly proxy | 30″ (~1 km) | direct download, **auto** |
| Soil moisture (future) | **WorldClim CMIP6** downscaled SSP projections (IPSL-CM6A-LR) | 2.5′ (~4.6 km) | direct download, **auto** |
| Soil moisture (earthquake) | ERA5 volumetric water content | 0.25° | supply via `vwc_path` |
| Lithology | **GLiM** 0.5° global grid (Hartmann & Moosdorf 2012) | 0.5° | PANGAEA, **auto** |
| Lithology (high-res) | **GLiM** full vector, 1 235 259 polygons | polygons | download¹, `glim_path` |
| Landslide inventory | **NASA COOLR / GLC** point catalogue | points | ArcGIS FeatureServer, **auto** |
| Earthquake trigger | GEM/GSHAP PGA (475-yr) | — | supply via `trigger_path`, or use `--pga` scenario |

¹ **Lithology resolution.** The full GLiM geodatabase (1.14 GB) is the
**default** and is downloaded automatically on first use; the 0.5° grid is a
fallback for `--glim-grid`. GLiM ships in Eckert IV (`ESRI:54012`) and is
reprojected to the model grid for you (needs `fiona`).

The manuscript's exact DEM (MERIT) and rainfall (W5E5 / ISIMIP) sources require
registration — the openly downloadable equivalents above are used by default and
can be swapped for the originals via `*_path` config options. The global PGA
layer still needs a manual download (or use a `--pga` scenario).

### Resolution

The model grid is set by `--res` (degrees). Use `0.0008333` (**3 arc-seconds,
~90 m** — the manuscript's resolution) for production runs over the HKH, and a
coarser value (e.g. `0.0025`) to prototype. Note the effective resolution of
each *factor* differs: slope 90 m (Copernicus DEM), vegetation 10 m (WorldCover,
downsampled), lithology polygon-level (full GLiM) or 0.5° (GLiM grid), and soil
moisture ~10′ (WorldClim). Slope and land cover therefore carry most of the
fine-scale signal.

## Modules (run any stage on its own)

| Module | Responsibility |
|--------|----------------|
| `config.py` | all calibration tables, weights, hazard matrices, run config |
| `grid.py` | reference grid, warp-to-grid, **block/tile iteration**, reclassify |
| `sources.py` | AOI-tiled downloaders (DEM, land cover, precip) + GLiM rasteriser |
| `demo.py` | synthetic input generator for offline runs/tests |
| `factors.py` | slope (Horn), lithology, land cover, soil-moisture → factors |
| `susceptibility.py` | weighted product (Eq. 1) → 5-class susceptibility |
| `triggers.py` | rainfall (Gumbel return period) / earthquake (PGA) classes |
| `hazard.py` | susceptibility × trigger → probability via hazard matrix |
| `inventory.py` | load/download/synthesize a landslide inventory; sampling |
| `calibrate.py` | logistic-regression weight calibration + ROC AUC |
| `pipeline.py` | orchestration; writes every intermediate for resume/inspect |
| `cli.py` | `run` / `download` / `calibrate` / `info` commands |

Because each stage reads and writes grid-aligned GeoTIFFs, you can stop after any
step, inspect the intermediate rasters in `data/work/`, tweak a table in
`config.py`, and re-run only the downstream stages.

## Calibrating the weights with historical Himalayan data

The manuscript calibrates the factor weights against landslide inventories by
expert judgment. This build adds an **automated, data-driven calibration** that
tunes the weights to observed landslides in the Himalayan region.

```bash
# Offline demonstration (synthetic Himalayan inventory) — proves the workflow:
python -m giri_landslide.cli calibrate --mode demo \
    --bbox 83.0 27.5 85.0 29.0 --res 0.004

# Real data: supply the NASA Global Landslide Catalog (or your own inventory),
# a CSV with latitude/longitude columns or a GeoJSON of points:
python -m giri_landslide.cli calibrate --mode download \
    --config examples/04_hkh_calibrate.json \
    --inventory data/raw/inventory/nasa_glc.csv

# Then run the model with the calibrated weights it wrote:
python -m giri_landslide.cli run --mode download \
    --config outputs/hkh_calibrated_calibrated_config.json
```

**How it works** (`inventory.py` + `calibrate.py`):

1. Load *presence* points (historical landslides), clip to the Himalayan region
   and filter by country (`Pakistan, India, Nepal, Bhutan, …`).
2. Draw *background* (pseudo-absence) points across the AOI.
3. Sample the four factor rasters at every point.
4. Fit a **logistic regression** of presence on `log(factor + 1)`. Because the
   index is combined in **exponent form** `log S = Σ wᵢ·log(fᵢ+1)`, the fitted
   coefficients *are* the calibrated factor weights.
5. Report **held-out ROC AUC** and write a ready-to-run calibrated config
   (`weight_mode = exponent`, `classification = quantile`).

> **Why exponent mode matters:** in the pure multiplicative form
> `S = Π wᵢ·fᵢ`, the weights are just a global scalar and *do not change which
> pixels rank as more susceptible*. Calibration therefore uses the exponent form,
> where weights genuinely control each factor's influence. Class breaks are then
> taken as equal-area quantiles of `S` (scale-independent).

The demo calibration recovers slope as the dominant factor with AUC ≈ 0.99
against its synthetic inventory; real inventories yield region-specific weights.

> **Getting the inventory:** the live NASA endpoints move periodically. If the
> built-in downloader fails, export the point catalogue from
> <https://landslides.nasa.gov> and pass it with `--inventory path.csv`
> (`python -m giri_landslide.cli info` prints the pointer). The loader
> auto-detects `latitude`/`longitude` columns and a `country_name` field.

## Methodology notes & calibration

- **Slope factor** (Table 2) is non-monotonic: it rises to a maximum at
  30–36° then falls, because very steep terrain sheds sediment and tends to be
  bare/hard rock.
- **Susceptibility class breaks** and the **factor weights** are internal expert
  calibrations in the manuscript; the defaults here are reasonable and clearly
  flagged in `config.py` — recalibrate against a landslide inventory (e.g. the
  NASA COLOR database) for operational use.
- The **earthquake hazard matrix** (Fig. 4) is transcribed exactly. The
  **rainfall hazard matrix** (Fig. 3) is published only as a figure whose numbers
  are not machine-readable, so the default `RAINFALL_MATRIX` is an illustrative
  diagonal calibration with the same structure — replace it with your calibrated
  values (paper target ≈ 400 000 significant rainfall-induced landslides/yr
  globally).
- For **future-climate** rainfall triggers, normalise with today's μ and σ (as
  the manuscript does) by supplying present-climate statistics.

## Tests

```bash
python -m pytest tests/ -q      # or: PYTHONPATH=. python tests/test_model.py
```

Tests are fully offline (synthetic demo) and cover the reclassification tables,
the Gumbel return-period mapping, the hazard-matrix corners, and a full
end-to-end run for both triggers.

## License

MIT. The referenced datasets carry their own licenses (Copernicus, ESA
WorldCover, WorldClim, GLiM, GEM) — cite and comply with each when publishing.
