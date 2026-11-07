"""Parked cars on side streets, baked into the art.

Main roads are painted empty and carry the moving traffic (site/traffic.js); side streets get
standing / parked vehicles drawn from the same rendered vehicle set, so the whole city feels
lived-in without frozen cars on the roads where traffic moves.

For every residential / service / unclassified street: walk it, measure the painted
carriageway edges (like road_snap.py), and park a car just inside an edge at random spacing,
facing along the street. Spots where a building stands in front of the street (silhouette pass
of the 3D renders) are skipped. Writes a copy of the art: out/<city>/<src>_parked/.

  uv run --with opencv-python-headless python pipeline/parked.py mumbai --src quads_v3 [--quads x0:x1,y0:y1]
"""

import argparse
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import city  # noqa: E402
from road_snap import Pix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q, K = 512, 0.7071067811865476
MINOR = ["residential", "unclassified", "service", "living_street"]
LIB = ROOT / "site" / "public" / "traffic" / "cars"


def minor_lines(cfg, pl):
  import duckdb
  enu = city.enu_fn(cfg["lat"], cfg["lon"])
  (rx, ry), (ux, uy) = city.basis(cfg)
  mpp, W, H = cfg["meters_per_pixel"], pl["width"], pl["height"]
  cache = ROOT / "out" / cfg["name"] / "minor_roads.json"
  if cache.exists():
    return [np.array(l) for l in json.loads(cache.read_text())]
  con = duckdb.connect()
  con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2'; SET geometry_always_xy = true;")
  lo, la = cfg["lon"], cfg["lat"]
  rows = con.sql(
    f"""SELECT ST_AsGeoJSON(geometry) FROM read_parquet('{city.OVERTURE}/theme=transportation/type=segment/*', hive_partitioning=1)
        WHERE bbox.xmax > {lo - 0.25} AND bbox.xmin < {lo + 0.25} AND bbox.ymax > {la - 0.3} AND bbox.ymin < {la + 0.3}
          AND subtype = 'road' AND class IN ({",".join(f"'{c}'" for c in MINOR)})""").fetchall()
  out = []
  for (gj,) in rows:
    g = json.loads(gj)
    if g["type"] != "LineString" or len(g["coordinates"]) < 2:
      continue
    a = np.array(g["coordinates"])
    x, y = enu(a[:, 1], a[:, 0])
    px = np.stack([W / 2 + (x * rx + y * ry) / mpp, H / 2 - (x * ux + y * uy) / mpp], 1)
    out.append(np.round(px, 1))
  cache.write_text(json.dumps([l.tolist() for l in out]))
  return out


def library():
  rows = json.loads((LIB / "library.json").read_text())
  atlas = Image.open(LIB / "library.png").convert("RGBA")
  by = defaultdict(list)
  for cls, head, w, h, ax, ay, ox, oy, res in rows:
    if cls in ("bus", "tempo", "bike"):
      continue  # parked: cars, taxis and autos
    spr = atlas.crop((ax, ay, ax + w, ay + h))
    if res != 1:  # bake at map resolution
      spr = spr.resize((max(1, round(w / res)), max(1, round(h / res))), Image.Resampling.BOX)
      ox, oy = ox / res, oy / res
    by[cls].append((head, spr, ox, oy))
  return by


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("city")
  ap.add_argument("--src", default="quads_v3")
  ap.add_argument("--quads")
  ap.add_argument("--density", type=float, default=0.32, help="chance of a parked car per ~7 m of kerb")
  args = ap.parse_args()
  rng = random.Random(1)
  cfg, pl, _ = city.load(args.city)
  mpp = cfg["meters_per_pixel"]
  src = ROOT / "out" / args.city / args.src
  dst = ROOT / "out" / args.city / f"{args.src}_parked"
  if dst.exists():
    shutil.rmtree(dst)
  shutil.copytree(src, dst)
  pix = Pix(src)
  lib = library()
  classes = ["white", "white", "silver", "taxi", "taxi", "red", "blue", "dark", "auto", "auto"]
  box = None
  if args.quads:
    (a, b), (c, e) = [map(int, s.split(":")) for s in args.quads.split(",")]
    box = (a * Q, c * Q, b * Q, e * Q)
  masks = {}

  def building_at(x, y):
    tx, ty = int(x // 1024), int(y // 1024)
    if (tx, ty) not in masks:
      p = ROOT / "out" / args.city / "renders" / f"mask_{tx}_{ty}.png"
      masks[(tx, ty)] = np.asarray(Image.open(p).convert("L")) > 128 if p.exists() else None
    m = masks[(tx, ty)]
    return m is not None and m[int(y) % 1024, int(x) % 1024]

  import water
  wcache = {}

  def on_water(x, y):  # streets mapped through harbours / blurred restricted areas: no cars on the sea
    q = (int(x // Q), int(y // Q))
    if q not in wcache:
      m = water.mask(args.city, *q)
      wcache[q] = np.asarray(m) if m is not None else None
    m = wcache[q]
    return m is not None and m[int(y) % Q, int(x) % Q] > 60

  stamps = defaultdict(list)  # quad -> [(sprite, x, y)]
  step = 7 / mpp
  for pts in minor_lines(cfg, pl):
    if box and not ((pts[:, 0] > box[0]) & (pts[:, 0] < box[2]) & (pts[:, 1] > box[1]) & (pts[:, 1] < box[3])).any():
      continue
    seg = np.diff(pts, axis=0)
    gl = np.hypot(seg[:, 0], seg[:, 1] / K)
    cum = np.r_[0, np.cumsum(gl)]
    if cum[-1] < 40:
      continue
    d = np.arange(step / 2, cum[-1], step)
    x, y = np.interp(d, cum, pts[:, 0]), np.interp(d, cum, pts[:, 1])
    si = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
    gx, gy = seg[si, 0] / gl[si].clip(1e-6), -seg[si, 1] / K / gl[si].clip(1e-6)
    offs = np.arange(-20, 20.1, 1.0)
    road = pix.asphalt(x[:, None] - gy[:, None] * offs[None, :], y[:, None] - gx[:, None] * offs[None, :] * K)
    z = len(offs) // 2
    for k in range(len(d)):
      if not road[k, z - 2 : z + 3].all():
        continue
      r = road[k]
      lo = hi = z
      while lo > 0 and r[lo - 1]:
        lo -= 1
      while hi < len(offs) - 1 and r[hi + 1]:
        hi += 1
      width = (offs[hi] - offs[lo]) * mpp
      if width < 4.5 or width > 14:  # too narrow to park, or not really a side street
        continue
      for side in (-1, 1):
        if rng.random() > args.density:
          continue
        off = (offs[hi] - 1.1 / mpp) if side > 0 else (offs[lo] + 1.1 / mpp)
        px, py = x[k] - gy[k] * off, y[k] - gx[k] * off * K
        if box and not (box[0] <= px < box[2] and box[1] <= py < box[3]):
          continue
        if building_at(px, py) or on_water(px, py):
          continue
        head = math.atan2(gy[k], gx[k]) + (math.pi if rng.random() < 0.5 else 0)
        head = (head + math.pi) % (2 * math.pi) - math.pi
        pool = lib.get(rng.choice(classes)) or lib["white"]
        h, spr, ox, oy = min(pool, key=lambda t: min(abs(t[0] - head), 2 * math.pi - abs(t[0] - head)))
        sx, sy = int(round(px - ox)), int(round(py - oy))
        for qx in range(sx // Q, (sx + spr.width) // Q + 1):
          for qy in range(sy // Q, (sy + spr.height) // Q + 1):
            stamps[(qx, qy)].append((spr, sx - qx * Q, sy - qy * Q, py))
  n = 0
  for (qx, qy), items in stamps.items():
    p = dst / f"{qx}_{qy}.png"
    if not p.exists():
      continue
    im = Image.open(p).convert("RGBA")
    for spr, ox, oy, _ in sorted(items, key=lambda t: t[3]):  # far first
      im.alpha_composite(spr, (ox, oy)) if 0 <= ox and 0 <= oy and ox + spr.width <= Q and oy + spr.height <= Q else im.paste(spr, (ox, oy), spr)
      n += 1
    im.convert("RGB").save(p)
  print(f"{n} parked vehicles stamped into {len(stamps)} quadrants -> {dst}")


if __name__ == "__main__":
  main()
