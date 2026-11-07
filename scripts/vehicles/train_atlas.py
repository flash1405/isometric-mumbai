"""Coach sprites (rendered at 2x map resolution) -> site/public/traffic/rails/trains.png + .json"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
raw = ROOT / "out" / "vehicles" / "trains"
idx = json.loads((raw / "index.json").read_text())
sprites = []
for e in idx:
  im = Image.open(raw / f"{e['i']}.png").convert("RGBA")
  a = np.asarray(im).astype(np.float32)
  al = a[..., 3]
  rgb = a[..., :3] / np.maximum(al[..., None] / 255, 1e-3)
  solid = al > 110
  sh = np.zeros_like(solid)
  for d in range(1, 5):
    sh[d:, d:] |= solid[:-d, :-d]
  out = np.zeros(a.shape, np.uint8)
  out[sh & ~solid] = (18, 22, 28, 90)
  out[solid, :3] = np.clip(rgb[solid], 0, 255); out[solid, 3] = 255
  ys, xs = np.nonzero(out[..., 3])
  x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
  sprites.append((e["cls"], e["head"], Image.fromarray(out[y0:y1, x0:x1], "RGBA"), im.width // 2 - x0, im.height // 2 - y0))
W, x, y, rowh, pos = 2048, 0, 0, 0, []
for c in sprites:
  spr = c[2]
  if x + spr.width > W: x, y, rowh = 0, y + rowh + 1, 0
  pos.append((x, y)); x += spr.width + 1; rowh = max(rowh, spr.height)
atlas = Image.new("RGBA", (W, y + rowh + 1), (0, 0, 0, 0))
rows = []
for (cls, head, spr, ox, oy), (ax, ay) in zip(sprites, pos):
  atlas.paste(spr, (ax, ay)); rows.append([cls, round(head, 4), spr.width, spr.height, ax, ay, int(ox), int(oy), 2])
o = ROOT / "site" / "public" / "traffic" / "rails"; o.mkdir(parents=True, exist_ok=True)
atlas.save(o / "trains.png", optimize=True); (o / "trains.json").write_text(json.dumps(rows, separators=(",", ":")))
print(len(rows), "coach sprites", atlas.size)
prev = Image.new("RGBA", (6 * 150, 4 * 110), (110, 92, 78, 255))
for n, (cls, head, spr, ox, oy) in enumerate(sprites):
  v, k = divmod(n, 72)
  if k % 12 == 0: prev.alpha_composite(spr, ((k // 12) * 150 + 75 - ox, v * 110 + 55 - oy))
prev.save(ROOT / "out" / "shots" / "trains_preview.png")
