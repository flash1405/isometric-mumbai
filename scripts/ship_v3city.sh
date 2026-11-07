#!/usr/bin/env bash
# v3: accurate buildings + real tower heights + empty roads, painted by the v3 model, with the
# 3D-sprite traffic and trains snapped to the painted roads/tracks. Ends live on sample.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*" | tee -a out/overnight.log; }
B200=150.136.218.176
OUTD=/home/ubuntu/im/train/output/isometric-mumbai-klein-v3
LORA=out/loras/isometric-mumbai-klein-v3.safetensors
VERSION=${VERSION:-v8}

log "v3: waiting for training"
until ssh -o BatchMode=yes ubuntu@$B200 "test -f $OUTD/isometric-mumbai-klein-v3.safetensors && ! pgrep -f \"[r]un.py\" >/dev/null"; do sleep 30; done
scp -q ubuntu@$B200:$OUTD/isometric-mumbai-klein-v3.safetensors $LORA
log "v3: model trained, bringing up painters"
pkill -f "lambda_gpu.py tunnels" || true
for spec in "magpie-b200 4" "iso-gen1 2" "iso-gen2 1" "iso-gen3 1"; do
  set -- $spec
  bash scripts/gen_node.sh "$1" "$2" "$LORA" > out/gen_node_$1.log 2>&1 &
done
wait
cat out/gen_node_*.log | grep -E "ready|error" | tee -a out/overnight.log
READY=$(grep -l ready out/gen_node_*.log | sed 's/.*gen_node_//; s/.log//' | paste -sd, -)
LAMBDA_FLEET=$READY nohup uv run python pipeline/lambda_gpu.py tunnels > out/tunnels_v3.log 2>&1 &
until grep -q "endpoints -> out/endpoints.txt" out/tunnels_v3.log; do sleep 3; done
sleep 5
log "v3: $(grep endpoints out/tunnels_v3.log | tail -1)"

# quick look for the user: render | v3 on two held-out tiles
uv run python - <<'PY' || true
import base64, httpx, sys
from io import BytesIO
from PIL import Image
sys.path.insert(0, "pipeline")
from generate import PROMPT
url = open("out/endpoints.txt").read().split(",")[0]
W = Image.new("RGB", (1536, 1024))
for i in range(2):
  src = Image.open(f"out/v3_samples/holdout_{i}.png").convert("RGB")
  b = BytesIO(); src.save(b, "PNG")
  r = httpx.post(url, json={"image_b64": base64.b64encode(b.getvalue()).decode(), "prompt": PROMPT, "seed": 7, "steps": 4, "guidance_scale": 1.0}, timeout=600)
  out = Image.open(BytesIO(base64.b64decode(r.json()["image_b64"]))).convert("RGB")
  for j, im in enumerate([src, out, Image.open(f"out/v3_samples/holdout_{i}_teacher.png")]):
    W.paste(im.convert("RGB").resize((512, 512)), (j * 512, i * 512))
W.save("out/shots/v3_compare.png")
print("compare -> out/shots/v3_compare.png")
PY

log "v3: painting the city"
export QUADS_SUBDIR=quads_v3 GEN_STEPS=4 GEN_GUIDANCE=1.0
for pass in 1 2 3; do
  uv run python pipeline/generate_city.py mumbai --per-endpoint 2 >> out/mumbai/gen_v3.log 2>&1
  n=$(ls out/mumbai/quads_v3 | wc -l)
  log "v3 pass $pass: $n quadrants"
  [ "$n" -ge 17890 ] && break
done
unset QUADS_SUBDIR GEN_STEPS GEN_GUIDANCE

log "v3: releasing rented GPUs"
uv run python - <<'PY'
import json, os, urllib.request
for l in open(".env"):
  if "=" in l and not l.startswith("#"):
    k, v = l.strip().split("=", 1); os.environ.setdefault(k, v)
def api(path, body=None):
  r = urllib.request.Request("https://cloud.lambda.ai/api/v1/" + path, data=json.dumps(body).encode() if body else None,
      headers={"Authorization": "Bearer " + os.environ["LAMBDA_API_KEY"], "User-Agent": "isometric-mumbai/0.1", "Content-Type": "application/json"},
      method="POST" if body else "GET")
  return json.load(urllib.request.urlopen(r, timeout=60))
ids = [i["id"] for i in api("instances")["data"] if i["name"].startswith("iso-gen")]
print("terminate", ids, api("instance-operations/terminate", {"instance_ids": ids}) if ids else "none")
PY

set -e
log "v3: traffic data on the new art"
uv run python pipeline/traffic.py mumbai
uv run --with opencv-python-headless python pipeline/road_snap.py mumbai --src quads_v3
uv run --with opencv-python-headless python pipeline/rail_snap.py mumbai --src quads_v3
uv run python scripts/vehicles/atlas.py
uv run python scripts/vehicles/train_atlas.py
log "v3: seams + export"
uv run --with scipy python pipeline/harmonize.py mumbai quads_v3
QUADS_SUBDIR=quads_v3 uv run --with scipy python pipeline/export_city.py mumbai --workers 10
uv run python scripts/og_image.py mumbai
log "v3: upload"
uv run python scripts/deploy.py tiles --version $VERSION
uv run python scripts/deploy.py site --version $VERSION --preview
log "SHIPPED $VERSION (v3 city) to preview"
