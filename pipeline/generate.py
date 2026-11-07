"""Turn an area's renders into pixel art with a seam-free, 4-phase infill schedule.

The grid is made of 512px quadrants. With cols, rows = 2 (mod 3):
  phase 1  2x2 blocks at (3i, 3j)            no neighbours yet, nothing to match
  phase 2  1-quadrant vertical gaps           256px of finished art on left and right
  phase 3  1-quadrant horizontal gaps         256px of finished art above and below
  phase 4  the gap intersections              finished art on all four sides
Jobs inside a phase never touch each other, so each phase runs in parallel.
Each job sends a 1024px template: finished art as context, the raw render inside a
red-bordered infill box (the input format the isometric-nyc LoRA was trained on).

  uv run python pipeline/generate.py fort [--phases 1,2,3,4] [--workers 4]
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
T = 1024  # template size
PROMPT = (
  "Fill in the outlined section with the missing pixels corresponding to the "
  "<isometric nyc pixel art> style, removing the border and exactly following the "
  "shape/style/structure of the surrounding image (if present)."
)


def load_env():
  env = ROOT / ".env"
  if env.exists():
    for line in env.read_text().splitlines():
      if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def stitch_renders(name, cfg):
  q = cfg["quadrant_px"]
  w, h = cfg["cols"] * q, cfg["rows"] * q
  canvas = Image.new("RGB", (w, h))
  rdir = ROOT / "out" / name / "renders"
  for ty in range((cfg["rows"] + 1) // 2):
    for tx in range((cfg["cols"] + 1) // 2):
      f = rdir / f"tile_{tx}_{ty}.png"
      if not f.exists():
        sys.exit(f"missing render {f}; run `npm run render -- {name}` first")
      canvas.paste(Image.open(f).convert("RGB"), (tx * 2 * q, ty * 2 * q))
  canvas.save(ROOT / "out" / name / "render.png")
  return canvas


def plan(cfg):
  """Return {phase: [(window_x, window_y, (ix0, iy0, ix1, iy1))]} in grid pixels."""
  q, cols, rows = cfg["quadrant_px"], cfg["cols"], cfg["rows"]
  assert q * 2 == T
  assert cols % 3 == 2 and rows % 3 == 2, "cols and rows must be 2 mod 3 (e.g. 8, 11, 14)"
  bxs, bys = range(0, cols, 3), range(0, rows, 3)
  gxs, gys = range(2, cols, 3), range(2, rows, 3)
  h = q // 2
  phases = {1: [], 2: [], 3: [], 4: []}
  for by in bys:
    for bx in bxs:
      phases[1].append((bx * q, by * q, (0, 0, T, T)))
  for by in bys:
    for gx in gxs:
      phases[2].append((gx * q - h, by * q, (h, 0, h + q, T)))
  for gy in gys:
    for bx in bxs:
      phases[3].append((bx * q, gy * q - h, (0, h, T, h + q)))
  for gy in gys:
    for gx in gxs:
      phases[4].append((gx * q - h, gy * q - h, (h, h, h + q, h + q)))
  return phases


def build_template(render, gen, wx, wy, box):
  tmpl = Image.new("RGB", (T, T))
  tmpl.paste(gen.crop((wx, wy, wx + T, wy + T)))
  x0, y0, x1, y1 = box
  tmpl.paste(render.crop((wx + x0, wy + y0, wx + x1, wy + y1)), (x0, y0))
  draw = ImageDraw.Draw(tmpl)
  for i in range(2):
    draw.rectangle([x0 + i, y0 + i, x1 - 1 - i, y1 - 1 - i], outline=(255, 0, 0))
  return tmpl


def clean_border(img, box, band=3):
  """Rebuild the band where the red infill border was drawn from the pixels just inside it.
  The model occasionally leaves a faint light line there."""
  import numpy as np

  a = np.asarray(img).copy()
  x0, y0, x1, y1 = box
  a[y0 : y0 + band, x0:x1] = a[y0 + band, x0:x1]
  a[y1 - band : y1, x0:x1] = a[y1 - band - 1, x0:x1]
  a[y0:y1, x0 : x0 + band] = a[y0:y1, x0 + band][:, None]
  a[y0:y1, x1 - band : x1] = a[y0:y1, x1 - band - 1][:, None]
  return Image.fromarray(a)


def call_model(url, img, seed):
  buf = BytesIO()
  img.save(buf, format="PNG")
  payload = {"image_b64": base64.b64encode(buf.getvalue()).decode(), "prompt": PROMPT, "seed": seed}
  if os.environ.get("GEN_STEPS"):  # e.g. 4 for the distilled Klein v2 model
    payload["steps"] = int(os.environ["GEN_STEPS"])
  if os.environ.get("GEN_GUIDANCE"):
    payload["guidance_scale"] = float(os.environ["GEN_GUIDANCE"])
  attempt = 0
  while True:  # never give up: a dropped network / sleeping laptop just pauses the run
    attempt += 1
    try:
      r = httpx.post(url, json=payload, timeout=900)
      r.raise_for_status()
      out = Image.open(BytesIO(base64.b64decode(r.json()["image_b64"]))).convert("RGB")
      return out if out.size == (T, T) else out.resize((T, T), Image.Resampling.LANCZOS)
    except Exception as e:  # noqa: BLE001
      print(f"    retry {attempt}: {e}", flush=True)
      time.sleep(min(30, 5 * attempt))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("name", nargs="?", default="fort")
  ap.add_argument("--phases", default="1,2,3,4")
  ap.add_argument("--workers", type=int, default=2)
  ap.add_argument("--seed", type=int, default=7)
  args = ap.parse_args()
  load_env()
  # default: Lambda server reached via `pipeline/lambda_gpu.py tunnel`
  url = os.environ.get("INFERENCE_URL") or "http://127.0.0.1:18000/edit_b64"
  try:
    httpx.get(url.rsplit("/", 1)[0] + "/health", timeout=5)
  except Exception:  # noqa: BLE001
    if "127.0.0.1:18000" in url:
      sys.exit(
        "No server at 127.0.0.1:18000. Start the tunnel (`pipeline/lambda_gpu.py tunnel`), "
        "or the model is still loading (`pipeline/lambda_gpu.py logs`)."
      )

  cfg = json.loads((ROOT / "config" / f"{args.name}.json").read_text())
  out = ROOT / "out" / args.name
  (out / "jobs").mkdir(parents=True, exist_ok=True)
  render = stitch_renders(args.name, cfg)
  gen_path = out / "generated.png"
  gen = Image.open(gen_path).convert("RGB") if gen_path.exists() else Image.new("RGB", render.size)

  phases = plan(cfg)
  for p in [int(x) for x in args.phases.split(",")]:
    jobs = phases[p]
    print(f"phase {p}: {len(jobs)} jobs")
    t0 = time.time()

    def run(job, p=p):
      wx, wy, box = job
      tag = f"p{p}_{wx}_{wy}"
      done = out / "jobs" / f"{tag}_out.png"
      if done.exists():
        return job, Image.open(done).convert("RGB")
      tmpl = build_template(render, gen, wx, wy, box)
      tmpl.save(out / "jobs" / f"{tag}_in.png")
      result = call_model(url, tmpl, args.seed + wx * 7 + wy * 13)
      result.save(done)
      print(f"  {tag} ok")
      return job, result

    with ThreadPoolExecutor(args.workers) as pool:
      results = list(pool.map(run, jobs))
    for (wx, wy, box), result in results:
      x0, y0, x1, y1 = box
      gen.paste(clean_border(result, box).crop(box), (wx + x0, wy + y0))
    gen.save(gen_path)
    print(f"phase {p} done in {time.time() - t0:.0f}s -> {gen_path}")


if __name__ == "__main__":
  main()
