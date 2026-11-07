#!/usr/bin/env bash
# After training: fetch the LoRA, bring up the fleet, paint the whole city, ship to preview,
# shut the rented GPUs down. Every step logs to out/overnight.log.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*"; }
B200=$(grep '^TRAIN_HOST=' .env | cut -d= -f2)
JOB=isometric-mumbai-teacher-v1
OUTD=/home/ubuntu/isometric-mumbai/train/output/$JOB

log "waiting for training to finish"
until ssh -o BatchMode=yes $B200 "ls $OUTD/$JOB.safetensors >/dev/null 2>&1 && ! pgrep -f '[r]un.py .*train.yaml' >/dev/null"; do sleep 60; done
mkdir -p out/loras && scp -q $B200:$OUTD/$JOB.safetensors out/loras/ && log "LoRA downloaded"
scp -q "$B200:$OUTD/samples/*" out/loras/ 2>/dev/null || true

log "waiting for the hunted GPU machine (max 20 min after training)"
FLEET="magpie-b200"
for i in $(seq 1 40); do
  if grep -q "PREPARED" out/gpu_hunter.out; then FLEET="magpie-b200 iso-fleet"; break; fi
  sleep 30
done
log "fleet: $FLEET"

# fresh start for the map: the earlier sample used the old model and old renders
[ -d out/mumbai/quads ] && mv out/mumbai/quads "out/mumbai/quads_old_$(date +%s)"

log "starting servers on: $FLEET"
LORA=out/loras/$JOB.safetensors bash scripts/fleet_up.sh $FLEET > out/fleet.log 2>&1 &
until grep -q "endpoints -> out/endpoints.txt" out/fleet.log; do sleep 15; done
log "$(grep endpoints out/fleet.log | tail -1)"

log "generating the city"
uv run python pipeline/generate_city.py mumbai --per-endpoint 2 > out/mumbai/gen_full.log 2>&1
log "generation finished: $(ls out/mumbai/quads | wc -l) quadrants"

log "shutting down the rented GPU machine"
uv run python - <<'PY'
import json, os, urllib.request
for l in open('.env'):
    if '=' in l and not l.startswith('#'): k, v = l.strip().split('=', 1); os.environ.setdefault(k, v)
def api(path, body=None):
    r = urllib.request.Request('https://cloud.lambda.ai/api/v1/' + path, data=json.dumps(body).encode() if body else None,
        headers={'Authorization': 'Bearer ' + os.environ['LAMBDA_API_KEY'], 'User-Agent': 'isometric-mumbai/0.1', 'Content-Type': 'application/json'},
        method='POST' if body else 'GET')
    return json.load(urllib.request.urlopen(r, timeout=60))
ids = [i['id'] for i in api('instances')['data'] if i['name'].startswith('iso-fleet')]
print('terminate', ids, api('instance-operations/terminate', {'instance_ids': ids}) if ids else 'none')
PY

log "shipping to preview"
VERSION=v1 TARGET=preview bash scripts/ship.sh mumbai && log "LIVE on sample.isometricmumbai.com"
