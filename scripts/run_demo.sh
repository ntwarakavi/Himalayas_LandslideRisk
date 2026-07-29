#!/usr/bin/env bash
# H-SIM offline smoke test - synthetic data, no downloads, ~2 minutes.
# Walks the whole sequence so you can see it before spending bandwidth.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--mode demo --name demo --bbox 83.0 27.5 83.2 27.7 --res 0.002"

echo "=============== 1-2  GET THE DATA ==============="
python -m h_sim.cli step1-check --offline | tail -5

echo
echo "=============== 4-9  PRODUCE ==================="
echo ">> step5  susceptibility: flow routing, then failure probability"
python -m h_sim.cli step5-susceptibility $COMMON | tail -10

echo
echo ">> step6  hazard: every rainfall and earthquake scenario"
python -m h_sim.cli step6-hazard $COMMON --all | tail -10

echo
echo ">> step7  climate: present day against two futures"
python -m h_sim.cli step7-climate $COMMON \
    --scenarios current ssp245:2041-2060 ssp585:2041-2060 | tail -10

echo
echo ">> step10 risk: what the map means for towns and roads, now and later"
python -m h_sim.cli step10-risk $COMMON \
    --risk-climate current ssp245:2021-2040 ssp585:2041-2060 | tail -24

echo
echo ">> step11 map: one HTML page of the lot"
python -m h_sim.cli step11-map --name demo \
    --bbox 83.0 27.5 83.2 27.7 --res 0.002 | tail -6

echo
echo "=============== 11   PACKAGE ==================="
python -m h_sim.cli step8-package --name demo | tail -12

cat <<'EOF'

Done. All synthetic - this proves the code runs, not the science.

About a sixth of this terrain sits above failure probability 0.5, so the map has
a real range in it. The climate scenarios still shift the probability only
slightly, because a 20% wetting moves few cells across FS = 1 once the wetness
term is already capped on the convergent ground. What the scenarios definitely
do move is the recharge field, which you can check directly:

    data/work/demo_recharge_current.tif            median 1.00 by definition
    data/work/demo_recharge_ssp585_2041-2060.tif   median 1.10

Products:
  outputs/demo_susceptibility_prob.tif      probability of failure  <- the product
  outputs/demo_susceptibility_class.tif     SINMAP classes 1-6 (a legend)
  outputs/demo_critical_acceleration.tif    Newmark yield coefficient (g)
  outputs/demo_hazard_*_prob.tif            one raster per trigger scenario
  outputs/demo_climate_*_change.tif         future minus present day
  outputs/demo_risk_settlements.json        every settlement, every scenario
  outputs/demo_risk_roads.json              every 500 m segment, every scenario
  outputs/demo_webmap/index.html            open it; no web server needed
  outputs/demo_manifest.json                what everything is, and its provenance

Steps 3 and 5 - the fit and the validation - are skipped here because both need
a real landslide inventory. On real data they are not optional:

    python -m h_sim.cli step2-download --config configs/01_calibrate.json
    python -m h_sim.cli step3-fit      --config configs/01_calibrate.json
    python -m h_sim.cli step4-validate --name gorkha --inventory <another inventory>
EOF
