"""Drive the inference server on a Lambda Cloud GPU instance.

  uv run python pipeline/lambda_gpu.py status   # list instances (uses LAMBDA_API_KEY)
  uv run python pipeline/lambda_gpu.py up       # install deps, fetch weights, start server
  uv run python pipeline/lambda_gpu.py tunnel   # forward 127.0.0.1:18000 -> instance:8000
  uv run python pipeline/lambda_gpu.py logs
  uv run python pipeline/lambda_gpu.py tunnels        # all fleet servers -> 127.0.0.1:18000+ (out/endpoints.txt)
  uv run python pipeline/lambda_gpu.py train          # upload dataset/, stop server, fine-tune
  uv run python pipeline/lambda_gpu.py train-logs
  uv run python pipeline/lambda_gpu.py serve-lora <remote dir> <file.safetensors>

The instance is picked by LAMBDA_INSTANCE (name, id or IP) or, failing that, the only running
instance. SSH uses your normal ~/.ssh config/keys as user `ubuntu` (override: LAMBDA_SSH_USER).
"""

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE_DIR = "~/isometric-mumbai"

SETUP = r"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/isometric-mumbai && cd ~/isometric-mumbai
if [ ! -x .venv-gpu/bin/python ]; then
  uv venv -q -p 3.11 .venv-gpu
  # Blackwell (B200) needs CUDA 12.8+ wheels; they also run on Hopper/Ampere
  uv pip install -q -p .venv-gpu torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q -p .venv-gpu diffusers transformers accelerate peft pillow fastapi uvicorn \
    "huggingface_hub[hf_transfer]"
fi
LORA=~/loras/cannoneyed-dark-copper-flea
if [ ! -s $LORA/model.safetensors ]; then
  # public file on Oxen hub (the oxen client insists on an API key, plain HTTP does not)
  mkdir -p $LORA
  curl -fsSL -o $LORA/model.safetensors \
    "https://hub.oxen.ai/api/repos/cannoneyed/isometric-nyc-v02/file/55b9eaff9e93ceae2549c85ba2f10b4f/models/cannoneyed-dark-copper-flea/model.safetensors"
fi
# base weights (~57 GB) once, before any server starts
HF_HUB_ENABLE_HF_TRANSFER=1 .venv-gpu/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen-Image-Edit')" >/dev/null
[ "${NO_SERVE:-0}" = "1" ] && { echo "prepared (no server started)"; exit 0; }
pkill -f "uvicorn server:app" || true
sleep 2
cd pipeline
NG=$(nvidia-smi -L | wc -l); R=${REPLICAS:-1}
for g in $(seq 0 $((NG - 1))); do
  for r in $(seq 0 $((R - 1))); do
    port=$((8000 + g * R + r))
    CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 nohup ../.venv-gpu/bin/uvicorn server:app --host 127.0.0.1 --port $port \
      > ~/isometric-server-$port.log 2>&1 &
  done
done
echo "started $((NG * R)) server(s) on $NG GPU(s)"
"""


def load_env():
  env = ROOT / ".env"
  if env.exists():
    for line in env.read_text().splitlines():
      if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def instances():
  key = os.environ.get("LAMBDA_API_KEY")
  if not key:
    sys.exit("Set LAMBDA_API_KEY in .env")
  err = None
  for base in ("https://cloud.lambda.ai", "https://cloud.lambdalabs.com"):
    req = urllib.request.Request(
      f"{base}/api/v1/instances",
      # the API's firewall rejects urllib's default User-Agent
      headers={"Authorization": f"Bearer {key}", "User-Agent": "isometric-mumbai/0.1"},
    )
    try:
      with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"]
    except Exception as e:  # noqa: BLE001
      err = e
  sys.exit(f"Lambda API error: {err}")


def pick():
  want = os.environ.get("LAMBDA_INSTANCE")
  live = [i for i in instances() if i.get("status") == "active"]
  if want:
    live = [i for i in live if want in (i.get("name"), i.get("id"), i.get("ip"))]
  if len(live) != 1:
    sys.exit(f"Found {len(live)} matching active instances; set LAMBDA_INSTANCE. Run `status`.")
  return live[0]


def ssh_target(inst):
  return f"{os.environ.get('LAMBDA_SSH_USER', 'ubuntu')}@{inst['ip']}"


SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30"]


TRAIN_REMOTE = f"{REMOTE_DIR}/train"


def train(target):
  """Build the dataset (locally, or on the GPU box with REMOTE_BUILD=1), stop the inference
  server, and start training."""
  rsync = ["rsync", "-az", "--delete", "-e", "ssh " + " ".join(SSH_OPTS)]
  subprocess.run(["ssh", *SSH_OPTS, target, f"mkdir -p {TRAIN_REMOTE}/samples"], check=True)
  sys.path.insert(0, str(ROOT / "pipeline"))
  from build_dataset import PROMPT

  if os.environ.get("REMOTE_BUILD") == "1":  # pairs already uploaded to ~/isometric-mumbai/teacher_jpg
    build = (f"cd {REMOTE_DIR} && .venv-train/bin/python pipeline/build_teacher_dataset.py mumbai "
             f"--src {REMOTE_DIR}/teacher_jpg --out {TRAIN_REMOTE}/dataset --variants 3")
    if os.environ.get("SKIP_BUILD") != "1":
      subprocess.run(["ssh", *SSH_OPTS, target, build], check=True)
    remote_samples = [f"{TRAIN_REMOTE}/dataset/holdout/holdout_{i}.png" for i in range(2)]
  else:
    ds = ROOT / "dataset"
    if not (ds / "images").exists():
      sys.exit("No dataset/; run a build_*dataset.py first")
    subprocess.run([*rsync, f"{ds}/", f"{target}:{TRAIN_REMOTE}/dataset/"], check=True)
    remote_samples = [f"{TRAIN_REMOTE}/dataset/holdout/{p.name}" for p in sorted((ds / "holdout").glob("holdout_[0-9].png"))[:2]]
  remote_samples = [p.replace("~", "/home/ubuntu") for p in remote_samples]
  cfg = (ROOT / "pipeline" / "train_lora.yaml").read_text()
  for k, v in {
    "out_dir": "/home/ubuntu/isometric-mumbai/train/output",
    "dataset": "/home/ubuntu/isometric-mumbai/train/dataset",
    "prompt": PROMPT,
    "holdout_a": remote_samples[0] if remote_samples else "",
    "holdout_b": remote_samples[1] if len(remote_samples) > 1 else "",
  }.items():
    cfg = cfg.replace("{" + k + "}", v)
  subprocess.run(["ssh", *SSH_OPTS, target, f"cat > {TRAIN_REMOTE}/train.yaml"], input=cfg.encode(), check=True)
  start = f"""
pkill -f "uvicorn server:app" || true
cd {REMOTE_DIR}/ai-toolkit
HF_HOME=$HOME/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=1 nohup ../.venv-train/bin/python run.py {TRAIN_REMOTE}/train.yaml \\
  > ~/isometric-train.log 2>&1 &
echo "training started"
"""
  subprocess.run(["ssh", *SSH_OPTS, target, "bash -s"], input=start.encode(), check=True)


def tunnels():
  """Tunnel every inference server on every fleet machine (LAMBDA_FLEET=name,name) to
  local ports 18000+, and write out/endpoints.txt for generate_city (INFERENCE_URLS)."""
  names = [n for n in os.environ.get("LAMBDA_FLEET", "").split(",") if n]
  live = [i for i in instances() if i.get("status") == "active" and (not names or i.get("name") in names)]
  procs, urls, local = [], [], int(os.environ.get("TUNNEL_BASE", "18000"))
  for inst in live:
    t = ssh_target(inst)
    r = subprocess.run(["ssh", *SSH_OPTS, t, "ls ~/isometric-server-*.log 2>/dev/null"], capture_output=True, text=True)
    ports = sorted(int(x.rsplit("-", 1)[1].split(".")[0]) for x in r.stdout.split())
    if not ports:
      continue
    fw = []
    for p in ports:
      fw += ["-L", f"{local}:127.0.0.1:{p}"]
      urls.append(f"http://127.0.0.1:{local}/edit_b64")
      local += 1
    procs.append((inst["name"], ["ssh", *SSH_OPTS, "-o", "ServerAliveCountMax=6", "-o", "ExitOnForwardFailure=yes", "-N", *fw, t]))
    print(f"{inst['name']}: {len(ports)} server(s)")
  (ROOT / "out" / os.environ.get("ENDPOINTS_FILE", "endpoints.txt")).write_text(",".join(urls))
  print(f"{len(urls)} endpoints -> out/endpoints.txt (Ctrl-C to close)", flush=True)
  import threading
  import time

  def keep(name, cmd):  # reconnect forever: a dropped tunnel must not stall generation
    while True:
      subprocess.run(cmd)
      print(f"tunnel to {name} dropped; reconnecting", flush=True)
      time.sleep(3)

  ts = [threading.Thread(target=keep, args=p, daemon=True) for p in procs]
  for t_ in ts:
    t_.start()
  for t_ in ts:
    t_.join()


def main():
  load_env()
  cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
  if cmd == "status":
    for i in instances():
      print(i.get("name"), i.get("id"), i.get("ip"), i.get("status"), i["instance_type"]["name"])
    return
  if cmd == "tunnels":
    tunnels()
    return
  inst = pick()
  target = ssh_target(inst)
  if cmd == "up":
    subprocess.run(["ssh", *SSH_OPTS, target, f"mkdir -p {REMOTE_DIR}/pipeline"], check=True)
    subprocess.run(
      ["rsync", "-az", "-e", "ssh " + " ".join(SSH_OPTS), str(ROOT / "pipeline" / "server.py"),
       f"{target}:{REMOTE_DIR}/pipeline/"],
      check=True,
    )
    envs = " ".join(f"{k}={os.environ[k]}" for k in ("LORA_DIR", "LORA_WEIGHT", "REPLICAS", "NO_SERVE") if os.environ.get(k))
    subprocess.run(["ssh", *SSH_OPTS, target, f"{envs} bash -s"], input=SETUP.encode(), check=True)
  elif cmd == "tunnel":
    print(f"127.0.0.1:18000 -> {inst['ip']}:8000 (Ctrl-C to stop)")
    subprocess.run(["ssh", *SSH_OPTS, "-N", "-L", "18000:127.0.0.1:8000", target])
  elif cmd == "logs":
    subprocess.run(["ssh", *SSH_OPTS, target, "tail -n 5 ~/isometric-server-*.log"])
  elif cmd == "train":
    train(target)
  elif cmd == "train-logs":
    subprocess.run(["ssh", *SSH_OPTS, target, "tail -c 3000 ~/isometric-train.log"])
  elif cmd == "serve-lora":
    # restart the inference server on a trained LoRA: serve-lora <dir> <file.safetensors>
    lora_dir, weight = sys.argv[2], sys.argv[3]
    subprocess.run(["ssh", *SSH_OPTS, target, f"LORA_DIR={lora_dir} LORA_WEIGHT={weight} bash -s"],
                   input=SETUP.encode(), check=True)
  else:
    sys.exit(__doc__)


if __name__ == "__main__":
  main()
