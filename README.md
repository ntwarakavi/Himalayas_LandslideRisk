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

**Regional scope.** This build is **restricted to the South Asia Himalayan arc**
— northern Pakistan, the Indian Himalaya, Nepal and Bhutan (and adjacent ranges),
region bounds `(71°E, 26°N) – (98°E, 37°N)`. Any AOI is automatically clipped to
this region, and historical inventories are filtered to it. Change
`region_bbox` in a config to widen it.

**Data-driven weight calibration.** The factor weights can be **fine-tuned
against a historical Himalayan landslide inventory** (NASA Global Landslide
Catalog / COOLR, or your own) by logistic regression — see
[Calibrating the weights](#calibrating-the-weights-with-historical-himalayan-data).

Running `./run_demo.sh` produces a two-panel quicklook
(`outputs/*_quicklook.png`) — left: 5-class susceptibility (green→red);
right: scenario landslide probability. Over the real central-Nepal AOI the model
correctly maps river valleys as low susceptibility and steep hillslopes as high.

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

## Quick start (no downloads)

```bash
pip install -r requirements.txt
./run_demo.sh                       # synthetic data, fully offline
```

This writes to `./outputs/`:

| file | meaning |
|------|---------|
| `*_susceptibility.tif` | 5-class susceptibility map (1 = Very Low … 5 = Very High) |
| `*_hazard_probability.tif` | probability of a significant landslide per scenario event |
| `*_trigger_class.tif` | rainfall / seismic trigger class |
| `*_summary.json` | run metadata + class histogram + hazard stats |
| `*_quicklook.png` | two-panel visual check |

## Run on real open data

```bash
# central Nepal, 90 m, 100-yr rainfall scenario
python -m giri_landslide.cli run --mode download \
    --bbox 84.0 28.0 84.6 28.5 --res 0.0008333 \
    --trigger rainfall --return-period 100

# earthquake scenario from a config file
python -m giri_landslide.cli run --mode download \
    --config examples/himalayas_earthquake.json
```

Start with a **small AOI and a coarse `--res` (e.g. 0.0025)** to test, then
refine. Downloads are cached under `data/raw/` and reused across runs.

## Open datasets used

| Factor | Default dataset (no login) | Resolution | Access |
|--------|----------------------------|-----------|--------|
| Slope | **Copernicus GLO-90 DEM** (or GLO-30) | 90 m / 30 m | AWS Open Data, auto |
| Vegetation | **ESA WorldCover 2021 v200** | 10 m | AWS Open Data, auto |
| Soil moisture (rainfall) | **WorldClim v2.1** monthly precip → max-monthly proxy | ~10′ | direct download, auto |
| Soil moisture (earthquake) | ERA5 volumetric water content | 0.25° | supply via `vwc_path` |
| Lithology | **GLiM** (Hartmann & Moosdorf 2012) | polygons | supply via `glim_path`¹ |
| Earthquake trigger | GEM/GSHAP PGA (475-yr) | — | supply via `trigger_path`, or use `--pga` scenario |

¹ GLiM and the global PGA layer are distributed from portals that need a
one-click/registered download, so the pipeline points you to the source
(`python -m giri_landslide.cli info`) and **degrades gracefully** if they are
absent (uniform lithology `Sl = 2`; use a uniform `--pga` scenario). The
manuscript's exact DEM (MERIT) and rainfall (W5E5 / ISIMIP) sources also require
registration — the openly downloadable equivalents above are used by default and
can be swapped for the originals via `*_path` config options.

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
    --config examples/himalaya_calibrate.json \
    --inventory data/raw/inventory/nasa_glc.csv

# Then run the model with the calibrated weights it wrote:
python -m giri_landslide.cli run --mode download \
    --config outputs/himalaya_calibrated_calibrated_config.json
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
