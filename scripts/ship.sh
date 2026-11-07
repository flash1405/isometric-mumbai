#!/usr/bin/env bash
# Finish and publish: palette, tiles, places, traffic, preview image, upload, deploy.
#   VERSION=v1 TARGET=preview|production bash scripts/ship.sh mumbai
set -euo pipefail
cd "$(dirname "$0")/.."
CITY=${1:-mumbai}; VERSION=${VERSION:-v1}; TARGET=${TARGET:-preview}
[ "${PIXEL:-0}" = "1" ] && uv run python pipeline/pixelart.py palette "$CITY"
uv run python pipeline/export_city.py "$CITY" --workers 10 $([ "${PIXEL:-0}" = "1" ] && echo --pixel) $([ -n "${REGION:-}" ] && echo --region $REGION)
uv run python pipeline/places.py
uv run python pipeline/traffic.py "$CITY"
uv run python scripts/og_image.py "$CITY"
uv run python scripts/deploy.py tiles --version "$VERSION"
if [ "$TARGET" = "production" ]; then uv run python scripts/deploy.py site --version "$VERSION"
else uv run python scripts/deploy.py site --version "$VERSION" --preview; fi
echo "SHIPPED $VERSION to $TARGET"
