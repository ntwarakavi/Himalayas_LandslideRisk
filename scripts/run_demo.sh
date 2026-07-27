#!/usr/bin/env bash
# Offline smoke test - synthetic data, no downloads, under a minute.
# Walks the same steps as a real run so you can see the sequence.
set -euo pipefail
cd "$(dirname "$0")/.."
AOI="--bbox 83.0 27.5 85.0 29.0"

echo ">> STEP 1  what data do we have?"
python -m giri_landslide.cli step1-check --offline | tail -5

echo; echo ">> STEP 3  calibrate weights (synthetic inventory)"
python -m giri_landslide.cli step3-calibrate --mode demo --name demo_cal \
    $AOI --res 0.004 | tail -18

echo; echo ">> STEP 4  susceptibility"
python -m giri_landslide.cli step4-susceptibility --mode demo --name demo \
    $AOI --res 0.005 | tail -8

echo; echo ">> STEP 5  hazard, 100-year storm"
python -m giri_landslide.cli step5-hazard --mode demo --name demo \
    $AOI --res 0.005 --return-period 100 | tail -6

echo; echo ">> STEP 4+5  earthquake scenario, 0.35 g"
EQ="--mode demo --name demo_eq $AOI --res 0.005 --trigger earthquake"
python -m giri_landslide.cli step4-susceptibility $EQ >/dev/null
python -m giri_landslide.cli step5-hazard $EQ --pga 0.35 | tail -5

cat <<'EOF'

Done. All synthetic - it proves the code works, not the science.
  outputs/demo_quicklook.png     susceptibility + hazard
  outputs/demo_cal_calibration.json  fitted weights and AUC

Real data:  python -m giri_landslide.cli step1-check
EOF
