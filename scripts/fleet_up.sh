#!/usr/bin/env bash
# Put the trained LoRA on every fleet machine, start one server per GPU (2 on the B200), tunnel.
#   LORA=out/loras/isometric-mumbai-teacher-v1.safetensors bash scripts/fleet_up.sh <instance-name> ...
set -uo pipefail
cd "$(dirname "$0")/.."
LORA=${LORA:-out/loras/isometric-mumbai-teacher-v1.safetensors}
W=$(basename "$LORA")
for name in "$@"; do
  ip=$(uv run python pipeline/lambda_gpu.py status | awk -v n="$name" '$1==n {print $3}')
  [ -z "$ip" ] && { echo "no instance $name"; continue; }
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new ubuntu@$ip 'mkdir -p ~/loras/mumbai ~/isometric-mumbai/pipeline'
  scp -q "$LORA" ubuntu@$ip:~/loras/mumbai/ && scp -q pipeline/server.py ubuntu@$ip:~/isometric-mumbai/pipeline/
  R=1; [[ "$name" == *b200* ]] && R=2
  REPLICAS=$R LORA_DIR=/home/ubuntu/loras/mumbai LORA_WEIGHT=$W LAMBDA_INSTANCE=$name \
    uv run python pipeline/lambda_gpu.py up 2>&1 | tail -1
done
# wait until every server reports ready, then tunnel all of them
for name in "$@"; do
  ip=$(uv run python pipeline/lambda_gpu.py status | awk -v n="$name" '$1==n {print $3}')
  until ssh -o BatchMode=yes ubuntu@$ip 'n=$(ls ~/isometric-server-*.log | wc -l); r=$(grep -l "model ready" ~/isometric-server-*.log | wc -l); [ "$n" = "$r" ]'; do sleep 15; done
  echo "$name ready"
done
LAMBDA_FLEET=$(IFS=,; echo "$*") uv run python pipeline/lambda_gpu.py tunnels
