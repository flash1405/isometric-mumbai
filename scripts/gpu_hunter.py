"""Find and launch a big Lambda GPU machine shortly before the full-city run needs it.

Every POLL seconds: log capacity for the candidate machines. Once training on the B200
passes TRIGGER (fraction of steps) -- or the file out/launch_gpus exists -- launch the
first candidate that Lambda accepts, then prepare it (deps + model weights, no server).

  uv run python scripts/gpu_hunter.py            (runs until one machine is launched + prepared)
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = ["gpu_8x_h100_sxm5", "gpu_8x_a100_80gb_sxm4", "gpu_8x_b200_sxm6", "gpu_4x_h100_sxm5", "gpu_8x_a100"]
SKIP = {"gpu_8x_a100"}  # 40 GB cards: too small for the model in bf16
POLL, TRIGGER, NAME = 120, 0.6, "iso-fleet"
TRAIN_HOST = None  # set from .env: TRAIN_HOST=ubuntu@<ip of the training GPU box>


def env():
  for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
      k, v = line.split("=", 1)
      os.environ.setdefault(k.strip(), v.strip())


def api(path, body=None):
  req = urllib.request.Request(
    f"https://cloud.lambda.ai/api/v1/{path}",
    data=json.dumps(body).encode() if body else None,
    headers={"Authorization": f"Bearer {os.environ['LAMBDA_API_KEY']}", "User-Agent": "isometric-mumbai/0.1",
             "Content-Type": "application/json"},
    method="POST" if body else "GET",
  )
  try:
    with urllib.request.urlopen(req, timeout=60) as r:
      return json.load(r)
  except urllib.error.HTTPError as e:
    return json.loads(e.read() or b"{}")


def training_progress():
  r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", os.environ["TRAIN_HOST"],
                      "tail -c 2000 ~/isometric-train.log 2>/dev/null | tr '\\r' '\\n' | grep -a 'mumbai-teacher' | grep -a -o ' [0-9][0-9]*/[0-9][0-9]* ' | tail -1; pgrep -f '[r]un.py .*train.yaml' >/dev/null && echo RUN"],
                     capture_output=True, text=True)
  m = re.search(r" (\d+)/(\d{3,}) ", r.stdout)  # only the new job's step counter, e.g. " 1200/3000 "
  running = "RUN" in r.stdout
  return (int(m.group(1)) / int(m.group(2)) if m and running else 0.0), running


def main():
  env()
  log = ROOT / "out" / "gpu_hunter.log"
  while True:
    types = api("instance-types").get("data", {})
    avail = {k: [r["name"] for r in v["regions_with_capacity_available"]] for k, v in types.items()
             if k in CANDIDATES and v["regions_with_capacity_available"]}
    prog, running = training_progress()
    line = f"{time.strftime('%H:%M')} training {prog:.0%}{' (running)' if running else ''} | available: {avail or 'none'}"
    print(line, flush=True)
    with log.open("a") as f:
      f.write(line + "\n")
    if prog >= TRIGGER or (ROOT / "out" / "launch_gpus").exists():
      for k in CANDIDATES:
        if k in SKIP or k not in avail:
          continue
        for region in avail[k]:
          res = api("instance-operations/launch", {"region_name": region, "instance_type_name": k,
                                                   "ssh_key_names": [os.environ["LAMBDA_SSH_KEY_NAME"]], "quantity": 1, "name": NAME})
          if res.get("data", {}).get("instance_ids"):
            print(f"LAUNCHED {k} in {region}: {res['data']['instance_ids'][0]}", flush=True)
            # wait for boot + SSH, then prepare (deps + weights, no server yet)
            while True:
              inst = [i for i in api("instances").get("data", []) if i.get("name") == NAME]
              if inst and inst[0].get("status") == "active" and inst[0].get("ip"):
                ip = inst[0]["ip"]
                if subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                                   "-o", "ConnectTimeout=10", f"ubuntu@{ip}", "true"]).returncode == 0:
                  break
              time.sleep(20)
            subprocess.run(["uv", "run", "python", "pipeline/lambda_gpu.py", "up"], cwd=ROOT,
                           env=dict(os.environ, LAMBDA_INSTANCE=NAME, NO_SERVE="1"))
            print(f"PREPARED {NAME} at {ip}", flush=True)
            return
          print(f"  launch {k}/{region} refused: {res.get('error', {}).get('message', res)}", flush=True)
    time.sleep(POLL)


if __name__ == "__main__":
  main()
