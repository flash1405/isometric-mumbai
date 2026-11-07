"""City-scale generation on a global quadrant store (resumable, multi-GPU).

Store:  out/<city>/renders/tile_<tx>_<ty>.png   1024px hybrid renders (2x2 quadrants)
        out/<city>/quads/<qx>_<qy>.png         512px generated quadrants (the map itself)

Schedule (same as generate.py, on the global grid): cell (i, j) = quadrants 3i..3i+2 x 3j..3j+2
  phase 1  block (3i..3i+1, 3j..3j+1)               no context
  phase 2  column 3i+2 rows 3j..3j+1                art on both sides (block i and block i+1)
  phase 3  row 3j+2 cols 3i..3i+1                   art above and below
  phase 4  corner (3i+2, 3j+2)                      art all round
A region is finished cell by cell: phase 1 also runs for the right/lower/diagonal neighbour
blocks, so later runs never paint a block next to finished art without context.

  uv run python pipeline/generate_city.py mumbai [--region south] [--phases 1,2,3,4]
  INFERENCE_URLS=http://127.0.0.1:18000/edit_b64,http://127.0.0.1:18001/edit_b64  (multi-GPU)
"""

import argparse
import itertools
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import city  # noqa: E402
from generate import call_model, clean_border, load_env  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q, T, H = 512, 1024, 256


class Store:
  def __init__(self, name):
    self.dir = ROOT / "out" / name
    self.quads = self.dir / os.environ.get("QUADS_SUBDIR", "quads")
    self.quads.mkdir(exist_ok=True)
    self.lock = threading.Lock()

  def qpath(self, qx, qy):
    return self.quads / f"{qx}_{qy}.png"

  def has(self, qx, qy):
    return self.qpath(qx, qy).exists()

  def gen(self, qx, qy):
    p = self.qpath(qx, qy)
    return Image.open(p).convert("RGB") if p.exists() else None

  @lru_cache(maxsize=256)
  def render_tile(self, tx, ty):
    p = self.dir / "renders" / f"tile_{tx}_{ty}.png"
    return Image.open(p).convert("RGB") if p.exists() else None

  def render(self, qx, qy):
    t = self.render_tile(qx // 2, qy // 2)
    if t is None:
      return None
    ox, oy = (qx % 2) * Q, (qy % 2) * Q
    return t.crop((ox, oy, ox + Q, oy + Q))

  def window(self, wx, wy, infill, kind):
    """1024px window at global px (wx, wy): kind='gen' or 'render' quadrant content."""
    img = Image.new("RGB", (T, T))
    for qy in range(wy // Q, (wy + T - 1) // Q + 1):
      for qx in range(wx // Q, (wx + T - 1) // Q + 1):
        src = self.render(qx, qy) if (kind == "render" or (qx, qy) in infill) else self.gen(qx, qy)
        if src is not None:
          img.paste(src, (qx * Q - wx, qy * Q - wy))
    return img


GPT_INFILL = (
  "This is a tile of an isometric pixel-art map of Mumbai. Everything outside the thin red rectangle is "
  "finished art. Inside the rectangle is an unfinished 3D render. Repaint ONLY the inside of the rectangle "
  "in exactly the same pixel-art style, palette and lighting as the finished art around it, keeping every "
  "building, road, tree and shoreline where the render puts them and continuing them seamlessly across the "
  "rectangle edge. Remove the red line. Do not change anything outside the rectangle."
)


def gpt_paint(plain, tmpl, box):
  """GPT Image backend: full tiles get the teacher redraw of the bare render; strips/corners get an
  infill edit with the red box, and only the box is kept (context stays pixel-exact)."""
  import teacher
  if box == (0, 0, T, T):
    return teacher.restyle(plain, tries=6)
  out = teacher.restyle(tmpl, prompt=GPT_INFILL, tries=6)
  res = plain.copy()
  res.paste(out.crop(box), box[:2])
  return clean_border(res, box)


def jobs_for(pl, active, targets):
  """Ordered phase -> list of (wx, wy, box, infill_quads) for the target quadrants."""
  cols, rows = pl["cols"], pl["rows"]
  cells = {(qx // 3, qy // 3) for qx, qy in targets}
  p1cells = set()
  for i, j in cells:
    p1cells |= {(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)}

  def act(qs):
    return any(0 <= qx < cols and 0 <= qy < rows and active[qy, qx] for qx, qy in qs)

  ph = {1: [], 2: [], 3: [], 4: []}
  for i, j in sorted(p1cells, key=lambda c: (c[1], c[0])):
    qs = [(3 * i + a, 3 * j + b) for a in (0, 1) for b in (0, 1)]
    if act(qs) and 3 * i + 1 < cols and 3 * j + 1 < rows:
      ph[1].append((3 * i * Q, 3 * j * Q, (0, 0, T, T), qs))
  for i, j in sorted(cells, key=lambda c: (c[1], c[0])):
    gx, gy = 3 * i + 2, 3 * j + 2
    qs = [(gx, 3 * j), (gx, 3 * j + 1)]
    if gx < cols and act(qs):
      ph[2].append((gx * Q - H, 3 * j * Q, (H, 0, H + Q, T), qs))
    qs = [(3 * i, gy), (3 * i + 1, gy)]
    if gy < rows and act(qs):
      ph[3].append((3 * i * Q, gy * Q - H, (0, H, T, H + Q), qs))
    qs = [(gx, gy)]
    if gx < cols and gy < rows and act(qs):
      ph[4].append((gx * Q - H, gy * Q - H, (H, H, H + Q, H + Q), qs))
  return ph


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("name")
  ap.add_argument("--region")
  ap.add_argument("--phases", default="1,2,3,4")
  ap.add_argument("--per-endpoint", type=int, default=2, help="concurrent requests per GPU")
  ap.add_argument("--seed", type=int, default=7)
  args = ap.parse_args()
  load_env()
  ep = ROOT / "out" / "endpoints.txt"
  urls = [u for u in (os.environ.get("INFERENCE_URLS") or (ep.read_text().strip() if ep.exists() else "") or
                      os.environ.get("INFERENCE_URL") or "http://127.0.0.1:18000/edit_b64").split(",") if u]
  print(f"{len(urls)} inference endpoint(s)")

  cfg, pl, active = city.load(args.name)
  targets = city.region_quads(cfg, pl, active, args.region)
  store = Store(args.name)
  # the hybrid renders must cover every job's infill (cells may reach outside the region)
  ph = jobs_for(pl, active, targets)
  cycle = itertools.cycle(urls)
  url_lock = threading.Lock()

  def next_url():
    with url_lock:
      return next(cycle)

  for p in [int(x) for x in args.phases.split(",")]:
    todo = [j for j in ph[p] if not all(store.has(*q) for q in j[3])]
    print(f"phase {p}: {len(todo)} jobs ({len(ph[p]) - len(todo)} already done)", flush=True)
    t0, done, skipped = time.time(), 0, 0

    def run(job, p=p, url=None):
      nonlocal done, skipped
      wx, wy, box, qs = job
      if any(store.render(*q) is None for q in qs):
        skipped += 1
        return
      missing = [q for q in qs if not store.has(*q)]
      if 0 < len(missing) < len(qs):  # extend an existing map (e.g. bridges): keep painted quads as context
        xs = [q[0] * Q - wx for q in missing]
        ys = [q[1] * Q - wy for q in missing]
        box = (min(xs), min(ys), max(xs) + Q, max(ys) + Q)
        qs = missing
      tmpl = store.window(wx, wy, set(qs), "gen")
      d = ImageDraw.Draw(tmpl)
      x0, y0, x1, y1 = box
      for i in range(2):
        d.rectangle([x0 + i, y0 + i, x1 - 1 - i, y1 - 1 - i], outline=(255, 0, 0))
      if os.environ.get("BACKEND") == "gpt":
        try:
          out = gpt_paint(store.window(wx, wy, set(qs), "gen"), tmpl, box)
        except Exception as e:  # noqa: BLE001
          print(f"  gpt failed at {wx},{wy}: {e}", flush=True)
          return
      else:
        out = clean_border(call_model(url or next_url(), tmpl, args.seed + wx * 7 + wy * 13), box)
      for qx, qy in qs:
        ox, oy = qx * Q - wx, qy * Q - wy
        out.crop((ox, oy, ox + Q, oy + Q)).save(store.qpath(qx, qy))
      done += 1
      if done % 25 == 0:
        el = time.time() - t0
        print(f"  p{p} {done}/{len(todo)}  {el / done:.2f}s/job  eta {el / done * (len(todo) - done) / 60:.0f} min", flush=True)

    # pull-based: every endpoint slot takes the next job as soon as it is free, so fast GPUs
    # (a B200) do more of the work instead of waiting their turn behind slow ones (an A10)
    import queue
    jq = queue.Queue()
    for j in todo:
      jq.put(j)

    def worker(url):
      while True:
        try:
          j = jq.get_nowait()
        except queue.Empty:
          return
        run(j, url=url)

    ths = [threading.Thread(target=worker, args=(u,)) for u in urls for _ in range(args.per_endpoint)]
    for t in ths:
      t.start()
    for t in ths:
      t.join()
    print(f"phase {p} done: {done} generated, {skipped} skipped (no render) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
  main()
