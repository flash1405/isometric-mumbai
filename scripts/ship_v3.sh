#!/usr/bin/env bash
# Final publish: traffic network -> lift the painted cars off the roads -> export the clean art
# (restricted sites as woodland) -> places, preview image -> upload tiles -> deploy site.
#   VERSION=v3 TARGET=preview bash scripts/ship_v3.sh mumbai
set -euo pipefail
cd "$(dirname "$0")/.."
CITY=${1:-mumbai}; VERSION=${VERSION:-v3}; TARGET=${TARGET:-preview}; SRC=${SRC:-quads_v2}
log() { echo "[$(date +%H:%M)] $*" | tee -a out/overnight.log; }
log "ship $VERSION: traffic network"
uv run python pipeline/traffic.py "$CITY"
log "lifting painted cars"
rm -rf "out/$CITY/${SRC}_clean"
uv run --with opencv-python-headless python pipeline/lift_cars.py "$CITY" --src "$SRC"
log "export"
QUADS_SUBDIR="${SRC}_clean" uv run python pipeline/export_city.py "$CITY" --workers 10
uv run python pipeline/places.py
uv run python scripts/og_image.py "$CITY"
log "upload"
uv run python scripts/deploy.py tiles --version "$VERSION"
if [ "$TARGET" = "production" ]; then uv run python scripts/deploy.py site --version "$VERSION"
else uv run python scripts/deploy.py site --version "$VERSION" --preview; fi
log "SHIPPED $VERSION to $TARGET"
