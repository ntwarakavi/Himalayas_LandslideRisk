#!/usr/bin/env bash
# One-off local setup for H-SIM.
#
#   ./setup.sh              install deps into .venv and verify
#   ./setup.sh --with-data  also pre-fetch the quickstart datasets (~120 MB)
#
set -euo pipefail
cd "$(dirname "$0")"

WITH_DATA=0
[[ "${1:-}" == "--with-data" ]] && WITH_DATA=1

echo "==> Python version"
python3 --version

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies"
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt

# Editable install, so `python -m h_sim.cli` works from any directory
# rather than only from the repository root.
echo "==> Installing the package (editable)"
python -m pip install -e . --no-deps -q

echo "==> Verifying the install"
python - <<'PY'
import numpy, rasterio, requests
print(f"  numpy    {numpy.__version__}")
print(f"  rasterio {rasterio.__version__}")
try:
    import fiona
    print(f"  fiona    {fiona.__version__}  (vector inventories and GLiM enabled)")
except ImportError:
    print("  fiona    MISSING -> CSV/GeoJSON inventories only, no GLiM vector")
try:
    import matplotlib
    print(f"  mpl      {matplotlib.__version__}  (quicklook PNGs enabled)")
except ImportError:
    print("  mpl      MISSING -> no quicklook PNGs (GeoTIFFs still written)")
PY

echo "==> Running the offline test suite"
python -m pytest tests/ -q

if [[ "$WITH_DATA" == "1" ]]; then
    echo "==> Pre-fetching the calibration datasets (~1.5 GB, one-off)"
    python -m h_sim.cli step2-download --config configs/01_calibrate.json
fi

cat <<'EOF'

Setup complete.

Activate the environment in every new shell:
    source .venv/bin/activate

Next steps:
    ./scripts/run_demo.sh                                          # offline

    # what the region-wide run would cost, before committing to it
    python -m h_sim.cli step9-region --dry-run --config configs/02_hkh_region.json

    # fit the soil parameters once, then check they travel
    python -m h_sim.cli step2-download --config configs/01_calibrate.json
    python -m h_sim.cli step3-fit      --config configs/01_calibrate.json
    python -m h_sim.cli step4-validate --build --name gorkha --inventory <path>

    # the product: every mountain province in the Hindu Kush Himalaya
    python -m h_sim.cli step9-region --config configs/02_hkh_region.json --everything

See docs/RUNNING_LOCALLY.md for the full walkthrough.
EOF
