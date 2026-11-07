#!/usr/bin/env bash
# Mid-run speed-up: bring up iso-fleet next to the running B200, restart generation on all
# endpoints (finished quadrants are skipped), then shut iso-fleet down and ship to preview.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*" | tee -a out/overnight.log; }
until grep -q "endpoints -> out/endpoints" out/fleet2.log; do sleep 15; done
log "boost: $(grep endpoints out/fleet2.log | tail -1)"
pkill -f "scripts/overnight.sh"; pkill -f "generate_city.py mumbai"; sleep 3
export INFERENCE_URLS="$(cat out/endpoints.txt),$(cat out/endpoints_fleet.txt)"
log "boost: generating on $(echo $INFERENCE_URLS | tr , '\n' | wc -l) endpoints"
for pass in 1 2 3; do  # re-run until nothing is left (finished quadrants are skipped)
  uv run python pipeline/generate_city.py mumbai --per-endpoint 2 >> out/mumbai/gen_full.log 2>&1
  [ "$(ls out/mumbai/quads | wc -l)" -ge 17680 ] && break
  log "pass $pass left gaps ($(ls out/mumbai/quads | wc -l) quadrants), going again"
done
log "generation finished: $(ls out/mumbai/quads | wc -l) quadrants"
sed -n '/shutting down the rented GPU machine/,$p' scripts/overnight.sh > out/tail.sh
source out/tail.sh
