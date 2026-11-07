#!/usr/bin/env bash
# Render every training area both ways (hybrid input + real Google 3D target).
#   bash scripts/build_training_areas.sh [area ...]     (default: all config/train/*.json)
# Each step skips work already on disk, so the script can be re-run after a failure.
set -uo pipefail
cd "$(dirname "$0")/.."
areas=("$@")
[ ${#areas[@]} -eq 0 ] && areas=($(ls config/train/*.json | xargs -n1 basename | sed 's/.json$//'))
for a in "${areas[@]}"; do
  A="train/$a"; O="out/$A"
  echo "=== $A $(date +%H:%M:%S)"
  if [ ! -f "$O/ortho_meta.json" ] || [ ! -f "$O/ortho_tiles/depth_0_0.png" ]; then
    npm run render -- "$A" --stage=ortho 2>&1 | grep -E "error|Done" || { echo "FAIL ortho $A"; continue; }
  fi
  uv run python pipeline/prepare.py "$A" 2>&1 | tail -1 || { echo "FAIL prepare $A"; continue; }
  uv run python pipeline/align.py "$A" || { echo "FAIL align $A"; continue; }
  npm run render -- "$A" --stage=google 2>&1 | grep -E "error|Done" || { echo "FAIL google $A"; continue; }
  npm run render -- "$A" --stage=hybrid --force 2>&1 | grep -E "error|Done" || { echo "FAIL hybrid $A"; continue; }
  echo "OK $A"
done
echo "ALL DONE $(date +%H:%M:%S)"
