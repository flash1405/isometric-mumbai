"""Collect teacher pairs (hybrid render -> finished pixel art) for training the Mumbai model.

Picks a varied set of render tiles (skips open sea and near-empty tiles), sends each to
the teacher (pipeline/teacher.py) at the account's rate limit, and stores pairs in
out/<city>/teacher/. Resumable; waits out rate limits / access errors.

  uv run python pipeline/collect.py mumbai --n 1500 --ipm 18
"""

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teacher  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def done_tiles(city):
  """Render tiles from chunks re-rendered with the current geometry (marker .rendered-all)."""
  ok = set()
  for d in (ROOT / "out" / city / "chunks").iterdir():
    if (d / ".rendered-all").exists() and (d / "tiles.json").exists():
      ok |= {tuple(t) for t in json.loads((d / "tiles.json").read_text())}
  return ok


def pick(city, n, seed=0, exclude=()):
  rdir = ROOT / "out" / city / "renders"
  ready = done_tiles(city)
  cands = []
  for m in rdir.glob("mask_*.png"):
    tx, ty = map(int, m.stem.split("_")[1:])
    if (tx, ty) not in ready or (tx, ty) in exclude:
      continue
    t = rdir / f"tile_{tx}_{ty}.png"
    if not t.exists():
      continue
    bf = (np.asarray(Image.open(m).convert("L").resize((64, 64))) > 128).mean()  # building fraction
    if bf < 0.06:
      continue
    small = np.asarray(Image.open(t).convert("RGB").resize((64, 64))).astype(np.float32)
    if (small.sum(-1) < 30).mean() > 0.05:  # black (outside render)
      continue
    cands.append((tx, ty, bf))
  rng = random.Random(seed)
  # stratify by building density so dense cores don't crowd out parks, coast and suburbs
  cands.sort(key=lambda c: c[2])
  bins = np.array_split(np.arange(len(cands)), 5)
  out = []
  for b in bins:
    idx = list(b)
    rng.shuffle(idx)
    out += [cands[i][:2] for i in idx[: n // 5 + 1]]
  return out[:n], len(cands)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("city")
  ap.add_argument("--n", type=int, default=1500)
  ap.add_argument("--ipm", type=float, default=18, help="images per minute (stay under the limit)")
  ap.add_argument("--workers", type=int, default=6)
  ap.add_argument("--out", default="teacher", help="pairs folder under out/<city>/")
  ap.add_argument("--faithful", action="store_true", help="v3 accuracy-first teacher prompt")
  args = ap.parse_args()
  teacher.env()
  out = ROOT / "out" / args.city / args.out
  out.mkdir(parents=True, exist_ok=True)
  plan_f = out / "plan.json"
  tiles = [tuple(t) for t in json.loads(plan_f.read_text())] if plan_f.exists() else []
  if len(tiles) < args.n:  # (re)fill from chunks rendered since the last run
    more, total = pick(args.city, args.n - len(tiles), exclude=set(tiles))
    tiles += [tuple(t) for t in more]
    plan_f.write_text(json.dumps(tiles))
    print(f"plan: {len(tiles)} tiles (+{len(more)} from {total} candidates)", flush=True)
  todo = [t for t in tiles if not (out / f"{t[0]}_{t[1]}_out.png").exists()]
  print(f"{len(todo)} to collect ({len(tiles) - len(todo)} done)", flush=True)

  gap = 60.0 / args.ipm
  lock = threading.Lock()
  state = {"next": time.time(), "done": 0, "t0": time.time()}

  def slot():
    with lock:
      now = time.time()
      wait = max(0.0, state["next"] - now)
      state["next"] = max(now, state["next"]) + gap
    time.sleep(wait)

  def one(t):
    tx, ty = t
    src = Image.open(ROOT / "out" / args.city / "renders" / f"tile_{tx}_{ty}.png").convert("RGB")
    while True:
      slot()
      try:
        res = teacher.restyle(src, prompt=teacher.PROMPT_FAITHFUL if args.faithful else teacher.PROMPT, tries=1)
        break
      except Exception as e:  # noqa: BLE001  (rate limit / access not yet enabled)
        print(f"  waiting: {str(e)[:120]}", flush=True)
        time.sleep(60)
        teacher.env(override=True)  # pick up a new key / org id from .env
    src.save(out / f"{tx}_{ty}_in.png")
    res.save(out / f"{tx}_{ty}_out.png")
    with lock:
      state["done"] += 1
      d = state["done"]
    if d % 20 == 0:
      el = time.time() - state["t0"]
      print(f"  {d}/{len(todo)}  eta {el / d * (len(todo) - d) / 60:.0f} min", flush=True)

  with ThreadPoolExecutor(args.workers) as ex:
    list(ex.map(one, todo))
  print("COLLECT DONE", flush=True)


if __name__ == "__main__":
  main()
