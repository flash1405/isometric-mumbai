#!/usr/bin/env bash
# v5: dense painted-car traffic, seam harmony, soft city edge, bridges, vivid zoomed-out levels.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*" | tee -a out/overnight.log; }
VERSION=${VERSION:-v5}
log "$VERSION: lifting cars + lanes"
rm -rf site/public/traffic/cars out/mumbai/quads_v2_clean
uv run --with opencv-python-headless python pipeline/lift_cars.py mumbai --src quads_v2
log "$VERSION: seam harmony"
uv run --with scipy python pipeline/harmonize.py mumbai quads_v2_clean
log "$VERSION: export"
QUADS_SUBDIR=quads_v2_clean uv run --with scipy python pipeline/export_city.py mumbai --workers 10
uv run python scripts/og_image.py mumbai
log "$VERSION: upload"
uv run python scripts/deploy.py tiles --version "$VERSION"
uv run python scripts/deploy.py site --version "$VERSION" --preview
log "SHIPPED $VERSION to preview"
