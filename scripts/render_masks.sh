#!/usr/bin/env bash
# Re-render chunks whose tiles have no building silhouette mask yet.
#   bash scripts/render_masks.sh mumbai [region]
cd "$(dirname "$0")/.."
CITY=$1; RFLAG=""; [ -n "${2:-}" ] && RFLAG="--region=$2"
O=out/$CITY
for c in $(uv run python pipeline/city.py chunks "$CITY" $RFLAG | awk '{print $1}'); do
  D=$O/chunks/$c
  [ -f "$D/tiles.json" ] || continue
  need=$(python3 -c "import json,os,sys; t=json.load(open('$D/tiles.json')); print(sum(not os.path.exists(f'$O/renders/mask_{x}_{y}.png') for x,y in t))")
  [ "$need" = "0" ] && continue
  npm run render -- "$CITY" --stage=hybrid --config-file=$D/config.json --data-dir=$D \
    --out-dir=$O/renders --tiles=$D/tiles.json --force 2>&1 | grep -E "error" || true
  echo "masks $c ($need)"
done
echo "MASKS DONE"
