#!/usr/bin/env bash
# Keep topping up the teacher set while the city re-render finishes, until N pairs exist.
cd "$(dirname "$0")/.."
N=${N:-1500}
while true; do
  uv run python pipeline/collect.py mumbai --n $N --ipm 18 --workers 24
  have=$(ls out/mumbai/teacher/*_out.png 2>/dev/null | wc -l)
  [ "$have" -ge "$N" ] && { echo "ALL $have PAIRS"; break; }
  sleep 120
done
