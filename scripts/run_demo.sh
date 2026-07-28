#!/usr/bin/env bash
# Offline smoke test - synthetic data, no downloads, about a minute.
# Walks all four phases so you can see the sequence before spending bandwidth.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--mode demo --name demo --bbox 83.0 27.5 83.2 27.7 --res 0.002"

echo "=============== PHASE 1  SET UP ==============="
python -m giri_landslide.cli step1-check --offline | tail -5

echo
echo "=============== PHASE 3  PRODUCE =============="
echo ">> step5  susceptibility: flow routing, then failure probability"
python -m giri_landslide.cli step5-susceptibility $COMMON | tail -10

echo
echo ">> step6  hazard: every rainfall and earthquake scenario"
python -m giri_landslide.cli step6-hazard $COMMON --all | tail -10

echo
echo ">> step7  climate: present day against two futures"
python -m giri_landslide.cli step7-climate $COMMON \
    --scenarios current ssp245:2061-2080 ssp585:2081-2100 | tail -10

echo
echo "=============== PHASE 4  PACKAGE =============="
python -m giri_landslide.cli step8-package --name demo | tail -12

cat <<'EOF'

Done. All synthetic - this proves the code runs, not the science. The demo
terrain sits well away from failure, so the climate scenarios move the
probability only slightly; what they do move is the recharge field, which you
can check directly:

    data/work/demo_recharge_current.tif            median 1.00 by definition
    data/work/demo_recharge_ssp585_2081-2100.tif   median 1.20

Products:
  outputs/demo_susceptibility_prob.tif      probability of failure  <- the product
  outputs/demo_susceptibility_class.tif     SINMAP classes 1-6 (a legend)
  outputs/demo_critical_acceleration.tif    Newmark yield coefficient (g)
  outputs/demo_hazard_*_prob.tif            one raster per trigger scenario
  outputs/demo_climate_*_change.tif         future minus present day
  outputs/demo_manifest.json                what everything is, and its provenance

Phase 2 (calibration and validation) is skipped here because it needs a real
landslide inventory. On real data it is not optional:

    python -m giri_landslide.cli step2-download --config configs/02_calibrate_gorkha.json
    python -m giri_landslide.cli step3-fit      --config configs/02_calibrate_gorkha.json
    python -m giri_landslide.cli step4-validate --name gorkha --inventory <another inventory>
EOF
