#!/usr/bin/env bash
# Run the full landslide model on synthetic data - no downloads, no network.
# Produces a susceptibility map, a scenario hazard-probability map, a summary
# JSON and a two-panel quicklook PNG in ./outputs.
set -euo pipefail
cd "$(dirname "$0")"

echo ">> Rainfall-triggered scenario (100-yr return period)"
python -m giri_landslide.cli run --mode demo --name himalaya_demo \
    --bbox 83.0 27.5 85.0 29.0 --res 0.005 \
    --trigger rainfall --return-period 100

echo
echo ">> Earthquake-triggered scenario (PGA = 0.35 g)"
python -m giri_landslide.cli run --mode demo --name himalaya_eq \
    --bbox 83.0 27.5 85.0 29.0 --res 0.005 \
    --trigger earthquake --pga 0.35

echo
echo "Done. See ./outputs/*.tif and ./outputs/*_quicklook.png"
