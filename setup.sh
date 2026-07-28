#!/usr/bin/env bash
# One-off local setup for the HKH landslide model.
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
    echo "==> Pre-fetching the quickstart datasets (~120 MB, one-off)"
    python -m giri_landslide.cli step2-download --config configs/01_quickstart.json
fi

cat <<'EOF'

Setup complete.

Activate the environment in every new shell:
    source .venv/bin/activate

Next steps:
    ./scripts/run_demo.sh                        # offline, no downloads

    # phase 1 + 3: fetch data and build a map with generic parameters
    python -m giri_landslide.cli step2-download        --config configs/01_quickstart.json
    python -m giri_landslide.cli step5-susceptibility  --config configs/01_quickstart.json

    # phase 2: calibrate to real landslides, then validate on another inventory
    python -m giri_landslide.cli step3-fit      --config configs/02_calibrate_gorkha.json
    python -m giri_landslide.cli step4-validate --name gorkha --inventory <path>

See docs/RUNNING_LOCALLY.md for the full walkthrough.
EOF
