"""Downsample the 4x vehicle renders into crisp pixel-art sprites + atlas for site/traffic.js.

  uv run python scripts/vehicles/atlas.py  -> site/public/traffic/cars/library.png + library.json
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
raw = ROOT / "out" / "vehicles" / "raw"
idx = json.loads((raw / "index.json").read_text())
SS = 4
RES = 2  # sprites stored at 2x map resolution: crisp on retina, drawn at half size
sprites = []
for e in idx:
  im = Image.open(raw / f"{e['i']}.png").convert("RGBA")
  im = im.resize((im.width * RES // SS, im.height * RES // SS), Image.Resampling.BOX)
  a = np.asarray(im).astype(np.float32)
  alpha = a[..., 3]
  rgb = a[..., :3] / np.maximum(alpha[..., None] / 255, 1e-3)  # un-premultiply the edge pixels
  solid = alpha > 110  # crisp pixel-art edge
  # contact shadow: the silhouette pushed away from the sun (light from the left), underneath
  sh = np.zeros_like(solid)
  for d in range(1, 2 * RES + 1):
    sh[d:, d:] |= solid[:-d, :-d]
  out = np.zeros(a.shape, np.uint8)
  out[sh & ~solid] = (18, 22, 28, 95)
  out[solid, :3] = np.clip(rgb[solid], 0, 255)
  out[solid, 3] = 255
  # 1 px darker outline on the shadow side keeps shapes readable on asphalt
  edge = solid & ~np.roll(solid, -1, 1) | solid & ~np.roll(solid, -1, 0)
  out[edge, :3] = (out[edge, :3] * 0.72).astype(np.uint8)
  ys, xs = np.nonzero(out[..., 3])
  if not len(xs):
    continue
  # crop around the vehicle; keep the anchor = canvas centre (vehicle centre on the ground)
  cx, cy = im.width // 2, im.height // 2
  x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
  spr = Image.fromarray(out[y0:y1, x0:x1], "RGBA")
  spr = ImageEnhance.Color(spr).enhance(1.1)
  sprites.append((e["cls"], e["head"], spr, cx - x0, cy - y0))
W, x, y, rowh = 1024, 0, 0, 0
pos = []
for cls, head, spr, ox, oy in sprites:
  if x + spr.width > W:
    x, y, rowh = 0, y + rowh + 1, 0
  pos.append((x, y)); x += spr.width + 1; rowh = max(rowh, spr.height)
atlas = Image.new("RGBA", (W, y + rowh + 1), (0, 0, 0, 0))
rows = []
for (cls, head, spr, ox, oy), (ax, ay) in zip(sprites, pos):
  atlas.paste(spr, (ax, ay))
  rows.append([cls, round(head, 4), spr.width, spr.height, ax, ay, int(ox), int(oy), RES])
out = ROOT / "site" / "public" / "traffic" / "cars"
out.mkdir(parents=True, exist_ok=True)
atlas.save(out / "library.png", optimize=True)
(out / "library.json").write_text(json.dumps(rows, separators=(",", ":")))
print(len(rows), "sprites, atlas", atlas.size)
# preview: every 8th heading of each variant, on asphalt
prev = Image.new("RGBA", (9 * 70, 17 * 70), (74, 76, 82, 255))
for n, (cls, head, spr, ox, oy) in enumerate(sprites):
  v, k = divmod(n, 72)
  if k % 8 == 0:
    prev.alpha_composite(spr.resize((spr.width * 2, spr.height * 2), Image.NEAREST), ((k // 8) * 70 + 35 - ox * 2 // 2, v * 70 + 35 - oy * 2 // 2))
prev.save(ROOT / "out" / "shots" / "vehicles_preview.png")
