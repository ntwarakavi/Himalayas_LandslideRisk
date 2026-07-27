#!/usr/bin/env bash
# One-off local setup for the HKH landslide model.
#
#   ./setup.sh              install deps into .venv and verify
#   ./setup.sh --with-glim  also pre-fetch the robust datasets (~2.2 GB)
#
set -euo pipefail
cd "$(dirname "$0")"

WITH_GLIM=0
[[ "${1:-}" == "--with-glim" ]] && WITH_GLIM=1

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
    print(f"  fiona    {fiona.__version__}  (full-resolution GLiM enabled)")
except ImportError:
    print("  fiona    MISSING -> only the GLiM 0.5-degree grid will be usable")
try:
    import matplotlib
    print(f"  mpl      {matplotlib.__version__}  (quicklook PNGs enabled)")
except ImportError:
    print("  mpl      MISSING -> no quicklook PNGs (GeoTIFFs still written)")
PY

echo "==> Running the offline test suite"
python -m pytest tests/ -q

if [[ "$WITH_GLIM" == "1" ]]; then
    echo "==> Pre-fetching the robust default datasets (~2.2 GB, one-off)"
    echo "    full GLiM geodatabase + WorldClim 30s + DEM/land-cover tiles"
    python -m giri_landslide.cli download --bbox 76.0 30.5 77.0 31.3
fi

cat <<'EOF'

Setup complete.

Activate the environment in every new shell:
    source .venv/bin/activate

Next steps:
    ./scripts/run_demo.sh                                  # offline, no downloads
    python -m giri_landslide.cli step4-susceptibility --mode download \
        --config configs/01_hkh_quickstart.json   # first real-data run

See docs/RUNNING_LOCALLY.md for the full walkthrough.
EOF
