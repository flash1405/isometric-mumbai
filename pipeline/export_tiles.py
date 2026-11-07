"""Export a generated map as a web tile pyramid for the public site (site/).

Writes  site/public/tiles/<z>/<x>/<y>.webp  (512px tiles, z = maxZoom is native resolution)
and     site/public/meta.json               (size, zooms, and the lat/lon <-> pixel transform)

The transform lets the site place search results and labels: a ground point (lat, lon)
maps to local metres (x east, y north) around the area centre, then onto the isometric
image plane using the camera's right/up vectors.

  uv run python pipeline/export_tiles.py fort [--layer generated|render]
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import to_local  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TILE = 512
Image.MAX_IMAGE_PIXELS = None


def camera_basis(cfg):
  az = math.radians(cfg["camera_azimuth_degrees"])
  el = math.radians(cfg["camera_elevation_degrees"])
  right = (math.cos(az), -math.sin(az))
  # screen-up for a ground point (z = 0): horizontal part of the camera's up vector
  up = (math.sin(az) * math.sin(-el), math.cos(az) * math.sin(-el))
  return right, up


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("name")
  ap.add_argument("--layer", default="generated")
  ap.add_argument("--out", default=str(ROOT / "site" / "public"))
  args = ap.parse_args()

  cfg = json.loads((ROOT / "config" / f"{args.name}.json").read_text())
  img = Image.open(ROOT / "out" / args.name / f"{args.layer}.png").convert("RGB")
  w, h = img.size
  max_z = math.ceil(math.log2(max(w, h) / TILE))
  out = Path(args.out)
  tiles = out / "tiles"
  if tiles.exists():
    shutil.rmtree(tiles)

  level = img
  for z in range(max_z, -1, -1):
    lw, lh = level.size
    n = 0
    for ty in range(math.ceil(lh / TILE)):
      for tx in range(math.ceil(lw / TILE)):
        t = level.crop((tx * TILE, ty * TILE, (tx + 1) * TILE, (ty + 1) * TILE))
        d = tiles / str(z) / str(tx)
        d.mkdir(parents=True, exist_ok=True)
        t.save(d / f"{ty}.webp", quality=88, method=4)
        n += 1
    print(f"z{z}: {lw}x{lh}px, {n} tiles")
    if z:
      level = level.resize((max(1, math.ceil(lw / 2)), max(1, math.ceil(lh / 2))), Image.Resampling.LANCZOS)

  _, kx, ky = to_local(cfg["lat"], cfg["lon"])
  right, up = camera_basis(cfg)
  meta = {
    "name": args.name,
    "width": w,
    "height": h,
    "tileSize": TILE,
    "maxZoom": max_z,
    "geo": {
      "lat0": cfg["lat"], "lon0": cfg["lon"], "kx": kx, "ky": ky,
      "right": right, "up": up, "mpp": cfg["meters_per_pixel"],
    },
  }
  (out / "meta.json").write_text(json.dumps(meta, indent=1))
  print(f"-> {tiles} (maxZoom {max_z}), {out / 'meta.json'}")


if __name__ == "__main__":
  main()
