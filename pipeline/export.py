"""Export render.png / generated.png as Deep Zoom (DZI) pyramids for the viewer.

  uv run python pipeline/export.py fort
"""

import json
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TILE, OVERLAP = 510, 1


def write_dzi(img: Image.Image, dest: Path):
  w, h = img.size
  files = dest.with_name(dest.stem + "_files")
  max_level = math.ceil(math.log2(max(w, h)))
  for level in range(max_level, -1, -1):
    scale = 2 ** (max_level - level)
    lw, lh = max(1, math.ceil(w / scale)), max(1, math.ceil(h / scale))
    limg = img if scale == 1 else img.resize((lw, lh), Image.Resampling.LANCZOS)
    ldir = files / str(level)
    ldir.mkdir(parents=True, exist_ok=True)
    for ty in range(math.ceil(lh / TILE)):
      for tx in range(math.ceil(lw / TILE)):
        x0, y0 = max(0, tx * TILE - OVERLAP), max(0, ty * TILE - OVERLAP)
        x1, y1 = min(lw, (tx + 1) * TILE + OVERLAP), min(lh, (ty + 1) * TILE + OVERLAP)
        limg.crop((x0, y0, x1, y1)).save(ldir / f"{tx}_{ty}.webp", quality=90)
  dest.write_text(
    f'<?xml version="1.0" encoding="UTF-8"?><Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
    f'Format="webp" Overlap="{OVERLAP}" TileSize="{TILE}"><Size Width="{w}" Height="{h}"/></Image>'
  )


def main():
  name = sys.argv[1] if len(sys.argv) > 1 else "fort"
  src = ROOT / "out" / name
  dest = ROOT / "viewer" / "public" / name
  dest.mkdir(parents=True, exist_ok=True)
  layers = []
  for layer in ("generated", "render"):
    f = src / f"{layer}.png"
    if f.exists():
      print("exporting", layer)
      write_dzi(Image.open(f).convert("RGB"), dest / f"{layer}.dzi")
      layers.append(layer)
  (dest / "layers.json").write_text(json.dumps(layers))
  print("open with: npm run viewer  ->  http://localhost:5174/?area=" + name)


if __name__ == "__main__":
  main()
