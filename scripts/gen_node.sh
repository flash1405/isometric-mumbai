#!/usr/bin/env bash
# Prepare one GPU box to serve the v3 (FLUX.2 klein 4B) painter, and start R servers per GPU.
#   bash scripts/gen_node.sh <lambda-instance-name> <replicas-per-gpu> <lora.safetensors>
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=$1; R=${2:-1}; LORA=$3
until ip=$(uv run python pipeline/lambda_gpu.py status | awk -v n=$NAME '$1==n && $4=="active" {print $3}') && [ -n "$ip" ] \
  && ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 ubuntu@$ip true 2>/dev/null; do sleep 15; done
T=ubuntu@$ip
ssh -o BatchMode=yes $T 'mkdir -p ~/isometric-mumbai/pipeline ~/loras/klein'
scp -q pipeline/server.py $T:~/isometric-mumbai/pipeline/ && scp -q "$LORA" $T:~/loras/klein/
ssh -o BatchMode=yes $T "R=$R W=$(basename $LORA) bash -s" <<'REMOTE'
set -e
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
cd ~/isometric-mumbai
if [ ! -x .venv-gpu/bin/python ]; then
  uv venv -q -p 3.11 .venv-gpu
  uv pip install -q -p .venv-gpu torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q -p .venv-gpu diffusers transformers accelerate peft pillow fastapi uvicorn "huggingface_hub[hf_transfer]" sentencepiece
fi
.venv-gpu/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.2-klein-4B')" >/dev/null 2>&1
pkill -f "[u]vicorn server" || true; sleep 2; rm -f ~/isometric-server-*.log
cd pipeline; NG=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NG - 1))); do for r in $(seq 0 $((R - 1))); do port=$((8000 + g * R + r))
  CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 MODEL_ID=black-forest-labs/FLUX.2-klein-4B LORA_DIR=$HOME/loras/klein LORA_WEIGHT=$W \
    nohup ../.venv-gpu/bin/uvicorn server:app --host 127.0.0.1 --port $port > ~/isometric-server-$port.log 2>&1 &
done; done
REMOTE
until ssh -o BatchMode=yes $T 'n=$(ls ~/isometric-server-*.log | wc -l); r=$(grep -l "model ready" ~/isometric-server-*.log | wc -l); [ "$n" = "$r" ] && [ "$n" -gt 0 ]'; do
  ssh -o BatchMode=yes $T 'grep -l "Error\|Traceback" ~/isometric-server-*.log' >/dev/null 2>&1 && { echo "$NAME: server error"; ssh $T 'tail -3 ~/isometric-server-*.log'; exit 1; }
  sleep 15
done
echo "$NAME ready ($ip)"
