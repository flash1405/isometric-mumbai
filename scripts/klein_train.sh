#!/usr/bin/env bash
# v2 student: FLUX.2 [klein] 4B edit LoRA on the teacher pairs, on a single 24 GB+ GPU.
#   bash scripts/klein_train.sh <lambda-instance-name>      (logs: ~/klein-train.log on the box)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:-iso-train}
until ip=$(uv run python pipeline/lambda_gpu.py status | awk -v n=$NAME '$1==n && $4=="active" {print $3}') && [ -n "$ip" ] \
  && ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 ubuntu@$ip true 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M)] $NAME up at $ip"
T=ubuntu@$ip
ssh $T 'mkdir -p ~/im/pipeline ~/im/train'
scp -q pipeline/build_dataset.py pipeline/build_teacher_dataset.py $T:~/im/pipeline/
echo "[$(date +%H:%M)] copying teacher pairs"
PAIRS=${PAIRS:-out/mumbai/teacher_jpg}
ssh $T "rm -rf ~/im/teacher_jpg ~/im/train/dataset"
rsync -az $PAIRS/ $T:~/im/teacher_jpg/
echo "[$(date +%H:%M)] installing"
ssh $T 'bash -s' <<'R'
set -e
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH=$HOME/.local/bin:$PATH
cd ~/im
[ -d ai-toolkit ] || git clone -q https://github.com/ostris/ai-toolkit.git
[ -x .venv/bin/python ] || uv venv -q --python 3.11 .venv
. .venv/bin/activate
uv pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -q torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -q -r ai-toolkit/requirements.txt hf_transfer
cd pipeline && python build_teacher_dataset.py mumbai --src ~/im/teacher_jpg --out ~/im/train/dataset --variants 3 --shift 2 --ext jpg
R
uv run python - "$T" <<'PY'
import subprocess, sys
sys.path.insert(0, "pipeline")
from build_dataset import PROMPT
import os; cfg = open(os.environ.get("KLEIN_CONFIG", "pipeline/train_klein.yaml")).read()
for k, v in {"out_dir": "/home/ubuntu/im/train/output", "dataset": "/home/ubuntu/im/train/dataset", "prompt": PROMPT,
             "holdout_a": "/home/ubuntu/im/train/dataset/holdout/holdout_0.png",
             "holdout_b": "/home/ubuntu/im/train/dataset/holdout/holdout_1.png"}.items():
  cfg = cfg.replace("{" + k + "}", v)
subprocess.run(["ssh", sys.argv[1], "cat > ~/im/train/train.yaml"], input=cfg.encode(), check=True)
PY
ssh $T 'cd ~/im/ai-toolkit && HF_HUB_ENABLE_HF_TRANSFER=1 nohup ../.venv/bin/python run.py ~/im/train/train.yaml > ~/klein-train.log 2>&1 & echo started'
echo "[$(date +%H:%M)] training started on $NAME"
