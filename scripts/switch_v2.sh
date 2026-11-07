#!/usr/bin/env bash
# Switch the fleet from v1 (Qwen LoRA) to v2 (FLUX.2 klein 4B LoRA), repaint the city into
# quads_v2/ (v1 stays in quads/ as a fallback), shut the rented machines down and ship to preview.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M)] $*" | tee -a out/overnight.log; }
LORA=out/loras/isometric-mumbai-klein-v2.safetensors
MODEL=black-forest-labs/FLUX.2-klein-4B
declare -a HOSTS=("magpie-b200:4" "iso-fleet:2")   # name:replicas per GPU (klein is small)
pkill -f boost_resume.sh; pkill -f "generate_city.py mumbai"; pkill -f "lambda_gpu.py tunnels"; pkill -f "ssh .*-L 18200"
for hr in "${HOSTS[@]}"; do
  name=${hr%%:*}; R=${hr##*:}
  ip=$(uv run python pipeline/lambda_gpu.py status | awk -v n=$name '$1==n {print $3}')
  log "v2: $name ($ip) x$R per GPU"
  ssh -o BatchMode=yes ubuntu@$ip 'mkdir -p ~/loras/klein'
  scp -q $LORA ubuntu@$ip:~/loras/klein/ && scp -q pipeline/server.py ubuntu@$ip:~/isometric-mumbai/pipeline/
  ssh -o BatchMode=yes ubuntu@$ip "R=$R MODEL=$MODEL W=$(basename $LORA) bash -s" <<'R'
cd ~/isometric-mumbai
HF_HUB_ENABLE_HF_TRANSFER=1 .venv-gpu/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" >/dev/null
pkill -f "[u]vicorn server" ; sleep 3; rm -f ~/isometric-server-*.log
cd pipeline; NG=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NG - 1))); do for r in $(seq 0 $((R - 1))); do port=$((8000 + g * R + r))
  CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 MODEL_ID=$MODEL LORA_DIR=$HOME/loras/klein LORA_WEIGHT=$W \
    nohup ../.venv-gpu/bin/uvicorn server:app --host 127.0.0.1 --port $port > ~/isometric-server-$port.log 2>&1 &
done; done
echo "started $((NG * R)) v2 servers"
R
done
for hr in "${HOSTS[@]}"; do
  name=${hr%%:*}; ip=$(uv run python pipeline/lambda_gpu.py status | awk -v n=$name '$1==n {print $3}')
  until ssh -o BatchMode=yes ubuntu@$ip 'n=$(ls ~/isometric-server-*.log | wc -l); r=$(grep -l "model ready" ~/isometric-server-*.log | wc -l); [ "$n" = "$r" ]'; do sleep 10; done
  log "v2: $name ready"
done
LAMBDA_FLEET=magpie-b200,iso-fleet nohup uv run python pipeline/lambda_gpu.py tunnels > out/tunnels_v2.log 2>&1 &
until grep -q "endpoints -> out/endpoints.txt" out/tunnels_v2.log; do sleep 3; done
log "v2: $(grep endpoints out/tunnels_v2.log | tail -1)"
export QUADS_SUBDIR=quads_v2 GEN_STEPS=4 GEN_GUIDANCE=1.0
for pass in 1 2 3; do
  uv run python pipeline/generate_city.py mumbai --per-endpoint 2 >> out/mumbai/gen_v2.log 2>&1
  [ "$(ls out/mumbai/quads_v2 | wc -l)" -ge 17680 ] && break
  log "v2 pass $pass left gaps ($(ls out/mumbai/quads_v2 | wc -l))"
done
log "v2 generation finished: $(ls out/mumbai/quads_v2 | wc -l) quadrants"
sed -n '/shutting down the rented GPU machine/,$p' scripts/overnight.sh | sed 's/VERSION=v1/VERSION=v2/' > out/tail_v2.sh
source out/tail_v2.sh
