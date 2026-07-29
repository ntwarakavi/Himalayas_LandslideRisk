#!/usr/bin/env bash
# H-SIM offline smoke test - synthetic data, no downloads, ~2 minutes.
#
# The real workflow (steps 5-9) sweeps every province in the region. This walks
# the same stages over ONE synthetic catchment using the area-* commands, which
# is what those commands are for: seeing the machinery work without spending
# bandwidth or days. It proves the code runs, not the science.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--mode demo --name demo --bbox 83.0 27.5 83.2 27.7 --res 0.002"

echo "=============== 1-2  GET THE DATA ==============="
python -m h_sim.cli step1-check --offline | tail -5

echo
echo "=============== 5-9  PRODUCE (one catchment) ===="
echo ">> step 5  susceptibility: flow routing, then failure probability"
python -m h_sim.cli area-susceptibility $COMMON | tail -10

echo
echo ">> trigger scenarios: every rainfall return period and PGA"
python -m h_sim.cli area-hazard $COMMON --all | tail -10

echo
echo ">> step 6  climate: present day against two futures"
python -m h_sim.cli area-climate $COMMON \
    --scenarios current ssp245:2041-2060 ssp585:2041-2060 | tail -10

echo
echo ">> steps 7-8  settlements and roads, now and later"
python -m h_sim.cli area-risk $COMMON \
    --risk-climate current ssp245:2021-2040 ssp585:2041-2060 | tail -24

echo
echo ">> step 9  the web app"
python -m h_sim.cli area-map --name demo \
    --bbox 83.0 27.5 83.2 27.7 --res 0.002 | tail -6

echo
echo "=============== PACKAGE ========================="
python -m h_sim.cli package --name demo | tail -12

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
    python -m h_sim.cli step3-calibrate      --config configs/01_calibrate.json
    python -m h_sim.cli step4-validate --name gorkha --inventory <another inventory>
EOF
