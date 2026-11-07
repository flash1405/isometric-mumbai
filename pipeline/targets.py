"""Make training targets: real-3D Google renders -> isometric-nyc pixel art.

For each training area, every renders_google tile is restyled as one full-tile infill
(the same call phase 1 of generate.py makes), giving pixel art with true geometry.

  uv run python pipeline/targets.py [area ...]      (default: every area with renders_google)
"""

import os
import zlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import has_real_3d  # noqa: E402
from generate import T, call_model, load_env  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def template(render):
  img = render.convert("RGB").resize((T, T))
  d = ImageDraw.Draw(img)
  for i in range(2):
    d.rectangle([i, i, T - 1 - i, T - 1 - i], outline=(255, 0, 0))
  return img


def main():
  load_env()
  url = os.environ.get("INFERENCE_URL") or "http://127.0.0.1:18000/edit_b64"
  base = ROOT / "out" / "train"
  areas = sys.argv[1:] or sorted(p.parent.name for p in base.glob("*/renders_google"))
  jobs = []
  for a in areas:
    if not has_real_3d(base / a):
      print(f"skip {a}: Google mesh is flat here")
      (base / a / ".flat").touch()
      continue
    src, dst = base / a / "renders_google", base / a / "targets"
    dst.mkdir(exist_ok=True)
    for f in sorted(src.glob("tile_*.png")):
      if not (dst / f.name).exists():
        jobs.append((f, dst / f.name))
  print(f"{len(jobs)} target tiles to generate across {len(areas)} areas")

  def run(job):
    f, out = job
    seed = zlib.crc32(f.as_posix().encode())  # deterministic per tile
    call_model(url, template(Image.open(f)), seed).save(out)
    return out

  with ThreadPoolExecutor(2) as pool:
    for i, out in enumerate(pool.map(run, jobs), 1):
      if i % 20 == 0 or i == len(jobs):
        print(f"  {i}/{len(jobs)} {out.parent.parent.name}/{out.name}", flush=True)


if __name__ == "__main__":
  main()
