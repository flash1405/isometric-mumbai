#!/usr/bin/env bash
# Generate targets for each training area as soon as build_training_areas.sh marks it OK.
cd "$(dirname "$0")/.."
LOG=out/train/build.log
while true; do
  for a in $(grep -E "^OK train/" "$LOG" | sed 's#^OK train/##'); do
    [ -f "out/train/$a/.flat" ] && continue
    n_r=$(ls out/train/$a/renders_google/tile_*.png 2>/dev/null | wc -l)
    n_t=$(ls out/train/$a/targets/tile_*.png 2>/dev/null | wc -l)
    if [ "$n_t" -lt "$n_r" ]; then uv run python pipeline/targets.py "$a" 2>&1 | grep -vE "^\s+[0-9]+/" ; echo "TARGETS $a done ($n_r tiles)"; fi
  done
  if grep -q "ALL DONE" "$LOG"; then
    pending=0
    for a in $(grep -E "^OK train/" "$LOG" | sed 's#^OK train/##'); do
      [ -f "out/train/$a/.flat" ] && continue
      [ "$(ls out/train/$a/targets/tile_*.png 2>/dev/null | wc -l)" -lt "$(ls out/train/$a/renders_google/tile_*.png | wc -l)" ] && pending=1
    done
    [ $pending -eq 0 ] && { echo "TARGETS ALL DONE"; exit 0; }
  fi
  sleep 60
done
