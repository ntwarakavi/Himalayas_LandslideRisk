#!/usr/bin/env bash
# Offline smoke test of the full HKH landslide model - no downloads, no network.
#
# Generates synthetic but physically plausible inputs over a Himalayan AOI and
# runs the complete chain (factors -> susceptibility -> trigger -> hazard) for
# both triggering mechanisms. Use this to confirm the install works before
# committing to real downloads.
#
# Outputs (in ./outputs): susceptibility + hazard GeoTIFFs, a summary JSON and
# a two-panel quicklook PNG per scenario.
set -euo pipefail
cd "$(dirname "$0")"

echo ">> [1/3] Rainfall-triggered scenario (100-yr return period)"
python -m giri_landslide.cli run --mode demo --name hkh_demo_rainfall \
    --bbox 83.0 27.5 85.0 29.0 --res 0.005 \
    --trigger rainfall --return-period 100

echo
echo ">> [2/3] Earthquake-triggered scenario (PGA = 0.35 g)"
python -m giri_landslide.cli run --mode demo --name hkh_demo_earthquake \
    --bbox 83.0 27.5 85.0 29.0 --res 0.005 \
    --trigger earthquake --pga 0.35

echo
echo ">> [3/3] Weight calibration against a synthetic inventory"
python -m giri_landslide.cli calibrate --mode demo --name hkh_demo_calibration \
    --bbox 83.0 27.5 85.0 29.0 --res 0.004

cat <<'EOF'

Offline demo complete. Inspect:
    outputs/hkh_demo_*_quicklook.png     visual check
    outputs/hkh_demo_*_summary.json      class histogram + hazard stats
    outputs/hkh_demo_calibration*.json   fitted weights + held-out AUC

Everything above used synthetic inputs. For real data see:
    docs/RUNNING_LOCALLY.md
EOF
