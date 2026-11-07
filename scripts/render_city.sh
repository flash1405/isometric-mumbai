#!/usr/bin/env bash
# Render a city (or one region of it) into the global render store.
#   bash scripts/render_city.sh mumbai [region]
# Steps (all resumable): plan list of ortho tiles -> Google top-down tiles -> per-chunk
# mosaic + buildings -> per-chunk hybrid iso renders (out/<city>/renders/).
set -uo pipefail
cd "$(dirname "$0")/.."
CITY=$1; REGION=${2:-}
RFLAG=""; [ -n "$REGION" ] && RFLAG="--region=$REGION"
O=out/$CITY
PAGES=${PAGES:-2}

echo "== ortho tiles $(date +%H:%M:%S)"
uv run python pipeline/city.py ortho-todo "$CITY" $RFLAG
npm run render -- "$CITY" --stage=ortho --config-file=$O/ortho_config.json --tiles=$O/ortho_todo.json --pages=$PAGES 2>&1 | grep -E "error|tiles for|Done"

for c in $(uv run python pipeline/city.py chunks "$CITY" $RFLAG | awk '{print $1}'); do
  D=$O/chunks/$c
  M="$D/.rendered-${REGION:-all}"; if [ -f "$M" ]; then continue; fi
  echo "== $c $(date +%H:%M:%S)"
  uv run python pipeline/city.py chunk "$CITY" "$c" $RFLAG || { echo "FAIL prep $c"; continue; }
  npm run render -- "$CITY" --stage=hybrid --config-file=$D/config.json --data-dir=$D \
    --out-dir=$O/renders --tiles=$D/tiles.json --force 2>&1 | grep -E "error|tiles for|Done" \
    && touch "$M" || echo "FAIL render $c"
done
echo "RENDER DONE $(date +%H:%M:%S)"
