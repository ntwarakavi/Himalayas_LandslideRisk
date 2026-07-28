#!/usr/bin/env bash
# Offline smoke test - synthetic data, no downloads, under a minute.
# Walks the same steps as a real run so you can see the sequence.
set -euo pipefail
cd "$(dirname "$0")/.."
AOI="--bbox 83.0 27.5 83.6 28.1"

echo ">> STEP 1  what data do we have?"
python -m giri_landslide.cli step1-check --offline | tail -5

echo; echo ">> STEP 4  stability: flow routing, then failure probability"
python -m giri_landslide.cli step4-stability --mode demo --name demo \
    $AOI --res 0.002 | tail -10

echo; echo ">> STEP 5  hazard, 100-year storm"
python -m giri_landslide.cli step5-hazard --mode demo --name demo \
    $AOI --res 0.002 --return-period 100 | tail -6

echo; echo ">> STEP 5  earthquake scenario, 0.35 g"
python -m giri_landslide.cli step5-hazard --mode demo --name demo \
    $AOI --res 0.002 --trigger earthquake --pga 0.35 | tail -6

echo; echo ">> STEP 7  what did the shaking change?"
python -m giri_landslide.cli step7-compare --name demo_change \
    --baseline outputs/demo_susceptibility_prob.tif \
    --scenario outputs/demo_hazard_pga0.35_prob.tif | tail -6

cat <<'EOF'

Done. All synthetic - it proves the code works, not the science.
  outputs/demo_susceptibility_prob.tif      probability of failure
  outputs/demo_susceptibility_class.tif     SINMAP stability classes
  outputs/demo_critical_acceleration.tif    Newmark yield coefficient (g)
  outputs/demo_change_quicklook.png         what 0.35 g does to the map

Real data:  python -m giri_landslide.cli step1-check
EOF
