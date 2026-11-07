"""Export the city quadrant store as the site's tile pyramid.

  z = maxZoom : one 512px tile per quadrant (generated art; sea quadrants get water)
  z < maxZoom : 2x2 children merged and halved
Output: site/public/tiles/<z>/<x>/<y>.webp + site/public/meta.json (same format as
export_tiles.py, plus `bounds` = the generated area in image pixels).

  uv run python pipeline/export_city.py mumbai [--region south] [--workers 8]
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import city  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q = 512
SITE = ROOT / "site" / "public"


def water_tile(seed=0):
  """Calm water in the model's palette: base colour with faint pixel-art ripples."""
  rng = np.random.default_rng(seed)
  base = np.array([52, 92, 120], np.float32)
  img = np.tile(base, (Q, Q, 1))
  for _ in range(260):
    x, y, w = rng.integers(0, Q), rng.integers(0, Q), rng.integers(6, 22)
    img[y : y + 2, x : x + w] += rng.uniform(8, 18)
  return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def save(img, z, x, y):
  d = SITE / "tiles" / str(z) / str(x)
  d.mkdir(parents=True, exist_ok=True)
  img.save(d / f"{y}.webp", quality=86, method=4)


_PAL = None


BG = (13, 18, 32)  # site background: land outside the city fades into it


_ROADS = None


def road_band(qx, qy):
  """Main roads (incl. bridges like the Sea Link) as a mask, so the water pass never paints over them."""
  global _ROADS
  if _ROADS is None:
    from collections import defaultdict
    import lift_cars
    _, lines = lift_cars.load_lines()
    _ROADS = defaultdict(list)
    for ln in lines:
      for cx in range(int(ln[4][:, 0].min() // 2048), int(ln[4][:, 0].max() // 2048) + 1):
        for cy in range(int(ln[4][:, 1].min() // 2048), int(ln[4][:, 1].max() // 2048) + 1):
          _ROADS[(cx, cy)].append(ln)
  import lift_cars
  from PIL import ImageDraw
  x0, y0 = qx * Q, qy * Q
  m, hit = Image.new("L", (Q, Q), 0), False
  d = ImageDraw.Draw(m)
  for ln in _ROADS.get((x0 // 2048, y0 // 2048), []):
    pts = ln[4]
    if pts[:, 0].max() < x0 - 60 or pts[:, 0].min() > x0 + Q + 60 or pts[:, 1].max() < y0 - 60 or pts[:, 1].min() > y0 + Q + 60:
      continue
    w = int(lift_cars.WIDTH_M[ln[2]] / 0.293 * 1.25)
    d.line([tuple(q) for q in (pts - [x0, y0])], fill=255, width=w, joint="curve")
    hit = True
  return m if hit else None


def building_mask(name, qx, qy):
  p = ROOT / "out" / name / "renders" / f"mask_{qx // 2}_{qy // 2}.png"
  if not p.exists():
    return None
  from PIL import ImageFilter

  ox, oy = (qx % 2) * Q, (qy % 2) * Q
  m = Image.open(p).convert("L").crop((ox, oy, ox + Q, oy + Q)).point(lambda v: 255 if v > 128 else 0)
  return m.filter(ImageFilter.MaxFilter(5))


_MIL = None


def military_mask(name, qx, qy):
  """Restricted sites (blurred in the source renders, so the model paints mush there): a soft
  mask in quadrant pixels, filled with painted woodland instead."""
  global _MIL
  if _MIL is None:
    cfg, pl, _ = city.load(name)
    (rx, ry), (ux, uy) = city.basis(cfg)
    mpp, W, H = cfg["meters_per_pixel"], pl["width"], pl["height"]
    _MIL = []
    for poly in city.mask_polys_enu(cfg):
      for part in getattr(poly, "geoms", [poly]):
        xy = np.array(part.exterior.coords)
        px = np.stack([W / 2 + (xy[:, 0] * rx + xy[:, 1] * ry) / mpp, H / 2 - (xy[:, 0] * ux + xy[:, 1] * uy) / mpp], 1)
        _MIL.append(px)
  x0, y0 = qx * Q, qy * Q
  hit = [p for p in _MIL if p[:, 0].max() > x0 - 20 and p[:, 0].min() < x0 + Q + 20 and p[:, 1].max() > y0 - 20 and p[:, 1].min() < y0 + Q + 20]
  if not hit:
    return None
  from PIL import ImageDraw, ImageFilter
  m = Image.new("L", (Q, Q), 0)
  d = ImageDraw.Draw(m)
  for p in hit:
    d.polygon([tuple(q) for q in (p - [x0, y0])], fill=255)
  return m.filter(ImageFilter.GaussianBlur(28))


FOREST_BLOCK = (126, 50, 6)  # a fully wooded 6x6 block of painted quadrants in Sanjay Gandhi National Park


def forest_texture(name, qdir, qx, qy):
  """Painted woodland for restricted sites: the matching quadrant of a contiguous forest block,
  so neighbouring quadrants join seamlessly (seams only every 6 quadrants)."""
  bx, by, n = FOREST_BLOCK
  p = Path(qdir) / f"{bx + qx % n}_{by + qy % n}.png"
  return Image.open(p).convert("RGB") if p.exists() else Image.new("RGB", (Q, Q), (52, 96, 48))


_HARM = None


class _Both:
  def __init__(self, a, b):
    self.a, self.b = a, b

  def apply(self, img, qx, qy):
    img = self.a.apply(img, qx, qy) if self.a else img
    return self.b.apply(img, qx, qy) if self.b else img


def harmony(qdir):
  """Seam-free colour (pipeline/harmonize.py): per-quad gains + hems, then the vegetation
  low-frequency pass, when they have been built for this art."""
  global _HARM
  if _HARM is None:
    import harmonize
    npz = Path(qdir).parent / f"{Path(qdir).name}_harmony.npz"
    lf = Path(qdir).parent / f"{Path(qdir).name}_lowfreq.png"
    a = harmonize.Harmony(npz) if npz.exists() else None
    b = harmonize.LowFreq(lf) if lf.exists() else None
    _HARM = _Both(a, b) if (a or b) else False
  return _HARM


def edge_alpha(patch):
  """Soft, rounded city edge instead of the square staircase of quadrants: `patch` is the 5x5
  neighbourhood (1 = painted or sea, 0 = unpainted land); returns an alpha for the centre quad.
  Wide enough (~1.5 quads) to turn the staircase into a smooth curve even when zoomed out."""
  from PIL import ImageFilter
  n = len(patch)
  small = Image.fromarray((np.asarray(patch, np.float32) * 255).astype(np.uint8), "L")
  big = small.resize((n * Q // 4, n * Q // 4), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(70))
  big = big.resize((n * Q, n * Q), Image.Resampling.BILINEAR)
  c = n // 2
  a = np.asarray(big, np.float32)[c * Q : (c + 1) * Q, c * Q : (c + 1) * Q] / 255
  return np.clip((a - 0.5) / 0.4, 0, 1) ** 1.5  # 0 exactly at the boundary, rounded inwards


def halo_alpha(patch):
  """Smooth per-pixel version of a 5x5 grid of quadrant values (no threshold)."""
  from PIL import ImageFilter
  n = len(patch)
  small = Image.fromarray((np.asarray(patch, np.float32) * 255).astype(np.uint8), "L")
  big = small.resize((n * Q // 4, n * Q // 4), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(40))
  big = big.resize((n * Q, n * Q), Image.Resampling.BILINEAR)
  c = n // 2
  return np.asarray(big, np.float32)[c * Q : (c + 1) * Q, c * Q : (c + 1) * Q] / 255


def base_tile(args):
  global _PAL
  qx, qy, qdir, gen_ok, maxz, pal_path, name, patch, halo = args
  import water
  from PIL import ImageChops

  p = Path(qdir) / f"{qx}_{qy}.png"
  wm = water.mask(name, qx, qy)
  if p.exists():
    img = Image.open(p).convert("RGB")
    h = harmony(qdir)
    if h:
      img = h.apply(img, qx, qy)
    if pal_path:  # pixel-art finishing pass (pipeline/pixelart.py)
      import pixelart
      if _PAL is None:
        _PAL = np.load(pal_path)
      img = pixelart.pixelate(img, _PAL)
    mm = military_mask(name, qx, qy)
    if mm is not None:
      # woodland fill, colour-matched to what surrounds it in this quadrant so it does not read as a patch
      f = np.asarray(forest_texture(name, qdir, qx, qy), np.float32)
      a = np.asarray(img, np.float32)
      w = np.asarray(mm, np.float32) / 255
      out_px = w < 0.05
      if out_px.sum() > 2000:
        f = f + (a[out_px].mean(0) - f.mean((0, 1))) * 0.6
      f = np.clip(f, 0, 255)
      img = Image.fromarray((a * (1 - w[..., None]) + f * w[..., None]).astype(np.uint8))
    if wm is not None:  # one consistent sea; keep buildings that lean over it, and bridges
      bm = building_mask(name, qx, qy)
      if bm is not None:
        wm = ImageChops.subtract(wm, bm)
      rb = road_band(qx, qy)
      if rb is not None:
        wm = ImageChops.subtract(wm, rb)
      img.paste(water.texture(qx, qy), (0, 0), wm)
    if patch is not None:  # fade into the background where the painted city meets unpainted land
      al = edge_alpha(patch)[..., None]
      img = Image.fromarray((np.asarray(img, np.float32) * al + np.array(BG, np.float32) * (1 - al)).astype(np.uint8))
    save(img, maxz, qx, qy)
    return 1
  if True:  # outside the city or not generated yet: sea near the city (fading out), background elsewhere
    img = Image.new("RGB", (Q, Q), BG)
    if wm is not None and halo is not None:
      fade = halo_alpha(halo)  # per-pixel, smooth across quadrants
      wm = Image.fromarray((np.asarray(wm, np.float32) * fade).astype(np.uint8))
      img.paste(water.texture(qx, qy), (0, 0), wm)
    save(img, maxz, qx, qy)
    return 2 if not gen_ok else 0


def parent_tile(args):
  z, x, y = args
  img, any_ = Image.new("RGB", (2 * Q, 2 * Q), BG), False
  for a in (0, 1):
    for b in (0, 1):
      p = SITE / "tiles" / str(z + 1) / str(2 * x + a) / f"{2 * y + b}.webp"
      if p.exists():
        img.paste(Image.open(p).convert("RGB"), (a * Q, b * Q))
        any_ = True
  if any_:
    out = img.resize((Q, Q), Image.Resampling.LANCZOS)
    if 4 <= z <= 7:  # downsampling averages the art into mud: restore colour and crispness a little per
      from PIL import ImageEnhance, ImageFilter  # level (compounds to ~+26% colour, +8% contrast by z4)
      out = ImageEnhance.Color(out).enhance(1.06)
      out = ImageEnhance.Contrast(out).enhance(1.02)
      out = out.filter(ImageFilter.UnsharpMask(radius=1.0, percent=45, threshold=2))
    save(out, z, x, y)
  return any_


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("name")
  ap.add_argument("--region")
  ap.add_argument("--workers", type=int, default=8)
  ap.add_argument("--pixel", action="store_true", help="apply the palette/2px pixel-art finishing pass")
  args = ap.parse_args()
  cfg, pl, active = city.load(args.name)
  qdir = ROOT / "out" / args.name / os.environ.get("QUADS_SUBDIR", "quads")
  done = {tuple(map(int, p.stem.split("_"))) for p in qdir.glob("*.png")}
  if not done:
    sys.exit("no generated quadrants yet")
  xs, ys = zip(*done)
  if args.region:  # show only the finished region (+ the sea around it)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
  else:
    x0, x1, y0, y1 = 0, pl["cols"] - 1, 0, pl["rows"] - 1
  maxz = math.ceil(math.log2(max(pl["cols"], pl["rows"])))
  pal_path = None
  if args.pixel:
    pal_path = str(ROOT / "out" / args.name / "palette.npy")
    if not Path(pal_path).exists():
      sys.exit(f"no palette: uv run python pipeline/pixelart.py palette {args.name}")
  tiles_dir = SITE / "tiles"
  if tiles_dir.exists():
    import shutil
    shutil.rmtree(tiles_dir)

  import water
  painted = np.zeros_like(active)
  for qx_, qy_ in done:
    if 0 <= qy_ < painted.shape[0] and 0 <= qx_ < painted.shape[1]:
      painted[qy_, qx_] = True
  sea = {}

  def solid(x, y):  # painted, or sea (a coast keeps its water, no fade)
    if not (0 <= y < painted.shape[0] and 0 <= x < painted.shape[1]):
      return 0.0
    if painted[y, x]:
      return 1.0
    if (x, y) not in sea:
      m = water.mask(args.name, x, y)
      sea[(x, y)] = m is not None and np.asarray(m).mean() > 128
    return 1.0 if sea[(x, y)] else 0.0

  # sea halo: how far each unpainted quadrant is from the painted city (full sea within ~600 m,
  # gone by ~2.1 km), so the zoomed-out map is the city with soft sea around it, not stray creeks
  from scipy.ndimage import distance_transform_edt
  dist = distance_transform_edt(~painted)
  near = np.clip(1 - (dist - 4) / 10, 0, 1)

  def halo_patch(x, y):
    return [[float(near[min(max(y + j, 0), near.shape[0] - 1), min(max(x + i, 0), near.shape[1] - 1)]) for i in range(-2, 3)] for j in range(-2, 3)]

  jobs = []
  for qy in range(y0, y1 + 1):
    for qx in range(x0, x1 + 1):
      patch = None
      if painted[qy, qx]:
        p5 = [[solid(qx + i, qy + j) for i in range(-2, 3)] for j in range(-2, 3)]
        if min(min(r) for r in p5) < 1:
          patch = p5
      # sea where the plan says inactive; leave unfinished land blank (sample exports)
      halo = halo_patch(qx, qy) if not painted[qy, qx] and near[qy, qx] < 1 else ([[1.0] * 5] * 5 if not painted[qy, qx] else None)
      jobs.append((qx, qy, str(qdir), bool(active[qy, qx]), maxz, pal_path, args.name, patch, halo))
  with ProcessPoolExecutor(args.workers) as ex:
    res = list(ex.map(base_tile, jobs, chunksize=64))
  print(f"z{maxz}: {res.count(1)} art, {res.count(2)} sea, {res.count(0)} unfinished")

  bx0, bx1, by0, by1 = x0, x1, y0, y1
  for z in range(maxz - 1, -1, -1):
    bx0, bx1, by0, by1 = bx0 // 2, bx1 // 2, by0 // 2, by1 // 2
    jobs = [(z, x, y) for y in range(by0, by1 + 1) for x in range(bx0, bx1 + 1)]
    with ProcessPoolExecutor(args.workers) as ex:
      n = sum(ex.map(parent_tile, jobs, chunksize=16))
    print(f"z{z}: {n} tiles")

  sys.path.insert(0, str(ROOT / "pipeline"))
  (rx, ry), (ux, uy) = city.basis(cfg)
  meta = {
    "name": args.name, "width": pl["width"], "height": pl["height"], "tileSize": Q, "maxZoom": maxz,
    "bounds": [x0 * Q, y0 * Q, (x1 + 1) * Q, (y1 + 1) * Q],
    "geo": {"lat0": cfg["lat"], "lon0": cfg["lon"], "frame": "enu", "right": [rx, ry], "up": [ux, uy],
            "mpp": cfg["meters_per_pixel"]},
  }
  (SITE / "meta.json").write_text(json.dumps(meta, indent=1))
  print("meta ->", SITE / "meta.json")


if __name__ == "__main__":
  main()
