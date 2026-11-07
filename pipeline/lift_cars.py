"""Lift the painted cars off the main roads so the site can drive them (site/traffic.js).

For every quadrant: rasterise the main-road network (from site/public/traffic, built by
traffic.py) into a road band, estimate the asphalt colour, find car-sized blobs that differ
from it (lane markings are too thin and get removed by an opening), cut each blob out as an
RGBA sprite, inpaint the road under it, and record where it sits on which road line.

  out/<city>/<quads_dir>_clean/<qx>_<qy>.png     quadrants with the cars removed
  site/public/traffic/cars/<bin>.png + .json     sprite atlas + [line, d, off, dir, w, h, sx, sy, ox, oy]

  uv run --with opencv-python-headless python pipeline/lift_cars.py mumbai [--src quads_v2] [--quads 26:34,168:176]
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
  import cv2
except ImportError:  # only needed for lifting; export_city just reads the road lines
  cv2 = None
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import city  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q = 512
WIDTH_M = {"motorway": 22, "trunk": 20, "primary": 15, "secondary": 11, "tertiary": 8}  # carriageway width
TRAFFIC = ROOT / "site" / "public" / "traffic"


def load_lines():
  """Main-road lines from the traffic bins: (bin, index-in-bin, cls, oneway, pts[n,2])."""
  idx = json.loads((TRAFFIC / "index.json").read_text())
  out = []
  for b in idx["bins"]:
    d = json.loads((TRAFFIC / f"{b}.json").read_text())
    for i, arr in enumerate(d.get("r", [])):
      cls = idx["classes"][arr[0] & 31]
      if cls not in WIDTH_M:
        continue
      pts = [[arr[1], arr[2]]]
      for k in range(3, len(arr), 2):
        pts.append([pts[-1][0] + arr[k], pts[-1][1] + arr[k + 1]])
      out.append((b, i, cls, arr[0] >= 32, np.array(pts, float)))
  return idx, out


def project(pt, pts, K):
  """Nearest point on polyline: (ground distance along, signed ground left offset, ground dir)."""
  best = None
  g = 0.0
  for a, b in zip(pts[:-1], pts[1:]):
    seg = b - a
    L2 = seg @ seg or 1e-9
    t = max(0.0, min(1.0, ((pt - a) @ seg) / L2))
    p = a + t * seg
    dist = np.hypot(*(pt - p))
    gl = math.hypot(seg[0], seg[1] / K)
    if best is None or dist < best[0]:
      gx, gy = seg[0] / gl, -seg[1] / K / gl  # ground dir (x right, y screen-up)
      dx, dy = pt[0] - p[0], -(pt[1] - p[1]) / K  # ground offset
      left = -gy * dx + gx * dy  # component along ground-left (-gy, gx)
      best = (dist, g + t * gl, left, (gx, gy))
    g += gl
  return best


def work(args):
  qx, qy, src, dst, lines, mpp, K = args
  p = src / f"{qx}_{qy}.png"
  if not p.exists():
    return []
  img = Image.open(p).convert("RGB")
  x0, y0 = qx * Q, qy * Q
  band = Image.new("L", (Q, Q), 0)
  dr = ImageDraw.Draw(band)
  near = []
  for ln in lines:
    pts = ln[4]
    if pts[:, 0].max() < x0 - 40 or pts[:, 0].min() > x0 + Q + 40 or pts[:, 1].max() < y0 - 40 or pts[:, 1].min() > y0 + Q + 40:
      continue
    w = max(4, int(WIDTH_M[ln[2]] / mpp * 0.9))
    dr.line([tuple(q) for q in (pts - [x0, y0])], fill=255, width=w, joint="curve")
    near.append(ln)
  a = np.asarray(img).astype(np.int16)
  bm = np.asarray(band) > 0
  if bm.sum() < 200:
    img.save(dst / p.name)
    return []
  # asphalt: low-saturation dark/mid grey inside the band
  hsv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV)
  asphalt = bm & (hsv[..., 1] < 45) & (hsv[..., 2] > 35) & (hsv[..., 2] < 150)
  if asphalt.sum() < 100:
    img.save(dst / p.name)
    return []
  base = np.median(a[asphalt], axis=0)
  diff = np.abs(a - base).sum(2)
  cand = (bm & (diff > 70)).astype(np.uint8)
  # lane markings / kerbs are thin: an opening removes them, cars (>= ~4 px across) survive
  cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
  n, lab, stats, cent = cv2.connectedComponentsWithStats(cand, 8)
  rgba = np.dstack([np.asarray(img), np.zeros((Q, Q), np.uint8)])
  hole = np.zeros((Q, Q), np.uint8)
  cars = []
  for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area < 35 or area > 320 or max(w, h) > 26 or min(w, h) < 6 or max(w, h) / min(w, h) > 2.6:
      continue
    if area / (w * h) < 0.5:  # cars are compact; kerbs, markings and crossings are not
      continue
    m = (lab[y : y + h, x : x + w] == i).astype(np.uint8)
    px = np.asarray(img)[y : y + h, x : x + w][m > 0]
    ph = hsv[y : y + h, x : x + w][m > 0]
    if ((ph[:, 0] > 30) & (ph[:, 0] < 90) & (ph[:, 1] > 60)).mean() > 0.25:
      continue  # green: trees / planters
    # a car stands on asphalt: most of the ring around it must be road surface
    y0r, y1r, x0r, x1r = max(0, y - 4), min(Q, y + h + 4), max(0, x - 4), min(Q, x + w + 4)
    ring = (lab[y0r:y1r, x0r:x1r] != i)
    if asphalt[y0r:y1r, x0r:x1r][ring].mean() < 0.55:
      continue
    # stripes (kerb ladders, zebra crossings, hatching) flip light/dark many times
    g = cv2.cvtColor(np.asarray(img)[y : y + h, x : x + w], cv2.COLOR_RGB2GRAY).astype(float)
    flips = []
    for prof in (g.mean(0), g.mean(1)):
      if len(prof) > 3:
        sgn = np.sign(prof - np.median(prof))
        flips.append(np.count_nonzero(np.diff(sgn[sgn != 0])) / len(prof))
    if flips and max(flips) > 0.34:
      continue
    core = np.pad(m, 3)
    m = cv2.dilate(core, np.ones((5, 5), np.uint8))  # cleanup area: outline + soft shadow
    X0, Y0 = x - 3, y - 3
    if X0 < 0 or Y0 < 0 or X0 + m.shape[1] > Q or Y0 + m.shape[0] > Q:
      continue  # touching the quadrant edge: leave it painted
    spr = rgba[Y0 : Y0 + m.shape[0], X0 : X0 + m.shape[1]].copy()
    # sprite alpha is tight: the car itself (+1 px outline) and its own darker shadow, no asphalt box
    win = diff[Y0 : Y0 + m.shape[0], X0 : X0 + m.shape[1]]
    dark = (a[Y0 : Y0 + m.shape[0], X0 : X0 + m.shape[1]].sum(2) < base.sum() - 45)
    tight = cv2.dilate(core, np.ones((3, 3), np.uint8)).astype(bool) | (dark & m.astype(bool) & (win > 35))
    spr[..., 3] = tight.astype(np.uint8) * 255
    c = np.array([x0 + cent[i][0], y0 + cent[i][1]])
    best = None
    for ln in near:
      r = project(c, ln[4], K)
      if best is None or r[0] < best[0][0]:
        best = (r, ln)
    (dist, d, left, gdir), ln = best
    half = WIDTH_M[ln[2]] / mpp / 2
    if dist > half * 1.15:
      continue
    if abs(left) > half * 0.95 or (not ln[3] and abs(left) < 1.2 / mpp):
      continue  # on the kerb line, or in the median of a two-way road
    direction = 1 if ln[3] or left >= 0 else -1  # left-hand traffic: left of the line = forward
    hole[Y0 : Y0 + m.shape[0], X0 : X0 + m.shape[1]] |= m
    cars.append({"bin": ln[0], "line": ln[1], "d": round(d, 1), "off": round(left, 1), "dir": direction,
                 "head": math.atan2(gdir[1] * direction, gdir[0] * direction),
                 "x": int(x0 + X0), "y": int(y0 + Y0), "sprite": Image.fromarray(spr, "RGBA")})
  if cars:  # fill with the road's own surface: median asphalt + inpainted structure (markings)
    out = cv2.inpaint(np.asarray(img), hole, 6, cv2.INPAINT_TELEA)
  else:
    out = np.asarray(img)
  Image.fromarray(out).save(dst / p.name)
  return cars


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("city")
  ap.add_argument("--src", default="quads_v2")
  ap.add_argument("--quads", help="qx0:qx1,qy0:qy1 (test area)")
  ap.add_argument("--workers", type=int, default=10)
  args = ap.parse_args()
  cfg, pl, active = city.load(args.city)
  mpp = cfg["meters_per_pixel"]
  idx, lines = load_lines()
  K = 0.7071067811865476
  src = ROOT / "out" / args.city / args.src
  dst = ROOT / "out" / args.city / f"{args.src}_clean"
  dst.mkdir(exist_ok=True)
  if args.quads:
    (a, b), (c, d) = [map(int, s.split(":")) for s in args.quads.split(",")]
    qs = [(x, y) for x in range(a, b) for y in range(c, d)]
  else:
    qs = [tuple(map(int, p.stem.split("_"))) for p in src.glob("*.png")]
  # lines per coarse cell so each worker only gets nearby lines
  cell = defaultdict(list)
  for ln in lines:
    for cx in range(int(ln[4][:, 0].min() // 2048), int(ln[4][:, 0].max() // 2048) + 1):
      for cy in range(int(ln[4][:, 1].min() // 2048), int(ln[4][:, 1].max() // 2048) + 1):
        cell[(cx, cy)].append(ln)
  jobs = [(qx, qy, src, dst, cell.get((qx * Q // 2048, qy * Q // 2048), []), mpp, K) for qx, qy in qs]
  bins = defaultdict(list)
  with ProcessPoolExecutor(args.workers) as ex:
    for cars in ex.map(work, jobs, chunksize=16):
      for c in cars:
        bins[c["bin"]].append(c)
  # each car may only drive where the painted road actually is: walk its path (at its lane
  # offset) both ways from where it was painted, and stop at the first stretch of non-road.
  cache = {}

  def pix(x, y):
    qx, qy = int(x // Q), int(y // Q)
    if (qx, qy) not in cache:
      p = dst / f"{qx}_{qy}.png"
      if not p.exists():
        p = src / f"{qx}_{qy}.png"
      cache[(qx, qy)] = cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2HSV) if p.exists() else None
    t = cache[(qx, qy)]
    if t is None:
      return False
    h, s_, v = t[int(y) % Q, int(x) % Q]
    return (s_ < 60 and 30 < v < 170) or (v > 190 and s_ < 40) or (15 < h < 35 and s_ > 90 and v > 150)  # asphalt / white or yellow paint

  line_pts = {(ln[0], ln[1]): ln[4] for ln in lines}

  def pos(pts, d, off):
    g = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
      seg = b - a
      gl = math.hypot(seg[0], seg[1] / K)
      if g + gl >= d or b is pts[-1]:
        t = min(1.0, max(0.0, (d - g) / (gl or 1)))
        gx, gy = seg[0] / (gl or 1), -seg[1] / K / (gl or 1)
        return a[0] + seg[0] * t - gy * off, a[1] + seg[1] * t - gx * off * K
      g += gl
    return pts[-1]

  def glen(pts):
    return sum(math.hypot(*(b - a) * [1, 1 / K]) for a, b in zip(pts[:-1], pts[1:]))

  for b, cars in bins.items():
    keep = []
    for c in cars:
      pts = line_pts[(c["bin"], c["line"])]
      L = glen(pts)
      lo = hi = c["d"]
      for sign in (-1, 1):
        miss, d = 0, c["d"]
        while 0 <= d <= L:
          d += sign * 2
          x, y = pos(pts, d, c["off"])
          miss = 0 if pix(x, y) else miss + 1
          if miss > 4:
            break
        edge = d - sign * 12
        if sign < 0:
          lo = min(c["d"], max(0, edge))
        else:
          hi = max(c["d"], min(L, edge))
      if hi - lo >= 60:  # needs a real stretch of road to drive
        c["lo"], c["hi"] = round(lo, 1), round(hi, 1)
        keep.append(c)
    bins[b] = keep

  library(bins, dst, src, lines, K, mpp)
  return


def colour_class(spr):
  a = np.asarray(spr)
  m = a[..., 3] > 0
  hsv = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2HSV)[m]
  h, s_, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
  body = ~((s_ < 50) & (v > 40) & (v < 165))  # drop the asphalt margin around the car
  if body.sum() < 8:
    return "other"
  h, s_, v = h[body], s_[body], v[body]
  ys, xs = np.nonzero(a[..., 3] > 0)
  if max(spr.size) - 10 > 24:  # sprite = car + 5 px margin each side
    return "bus"
  if ((h >= 15) & (h <= 35) & (s_ > 90) & (v > 110)).mean() > 0.18:
    return "taxi"
  if (((h < 9) | (h > 170)) & (s_ > 90)).mean() > 0.18:
    return "red"
  if ((h >= 95) & (h <= 130) & (s_ > 70)).mean() > 0.18:
    return "blue"
  if ((v > 175) & (s_ < 50)).mean() > 0.25:
    return "white"
  if (v < 70).mean() > 0.4:
    return "dark"
  return "other"


LANE_M = 3.2
LANES = {"motorway": 3, "trunk": 3, "primary": 2, "secondary": 2, "tertiary": 1}


def library(bins, dst, src, lines, K, mpp):
  """Dense traffic data: a library of painted car sprites (by colour type and facing angle) and,
  for every main-road lane, the stretches where painted asphalt actually runs under it."""
  out = TRAFFIC / "cars"
  out.mkdir(parents=True, exist_ok=True)
  # 1) sprite library: up to 5 sprites per type per 10 deg heading
  allc = [c for cs in bins.values() for c in cs]
  good = []
  for c in allc:
    c["cls"] = colour_class(c["sprite"])
    a = np.asarray(c["sprite"])
    al = a[..., 3] > 0
    n = int(al.sum())
    ys, xs = np.nonzero(al)
    if c["cls"] == "other" or not (45 <= n <= 260) or not len(xs):
      continue
    fill = n / ((np.ptp(xs) + 1) * (np.ptp(ys) + 1))
    if fill < 0.42:  # shards, kerbs, crossings
      continue
    hsv = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2HSV)[al]
    if c["cls"] == "blue" and np.median(hsv[:, 1]) < 120:
      continue  # bluish asphalt, not a blue car
    good.append(c)
  allc = good
  pick = defaultdict(list)
  for c in sorted(allc, key=lambda c: -np.asarray(c["sprite"])[..., 3].astype(bool).sum()):
    k = (c["cls"], int(((math.degrees(c["head"]) + 360) % 360) // 10))
    if len(pick[k]) < 5:
      pick[k].append(c)
  lib = [c for cs in pick.values() for c in cs]
  W, x, y, rowh = 1024, 0, 0, 0
  for c in lib:
    w, h = c["sprite"].size
    if x + w > W:
      x, y, rowh = 0, y + rowh + 1, 0
    c["ax"], c["ay"] = x, y
    x += w + 1
    rowh = max(rowh, h)
  atlas = Image.new("RGBA", (W, y + rowh + 1), (0, 0, 0, 0))
  for c in lib:
    atlas.paste(c["sprite"], (c["ax"], c["ay"]))
  atlas.save(out / "library.png", optimize=True)
  (out / "library.json").write_text(json.dumps(
    [[c["cls"], round(c["head"], 3), *c["sprite"].size, c["ax"], c["ay"]] for c in lib], separators=(",", ":")))
  from collections import Counter
  print(f"sprite library: {len(lib)} sprites {dict(Counter(c['cls'] for c in lib))}")

  # 2) visible lane stretches, vectorised per quadrant
  hsv_cache = {}

  def hsv(qx, qy):
    if (qx, qy) not in hsv_cache:
      p = dst / f"{qx}_{qy}.png"
      hsv_cache[(qx, qy)] = cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2HSV) if p.exists() else None
    return hsv_cache[(qx, qy)]

  def road(px, py):
    ok = np.zeros(len(px), bool)
    qx, qy = (px // Q).astype(int), (py // Q).astype(int)
    for key in set(zip(qx.tolist(), qy.tolist())):
      t = hsv(*key)
      if t is None:
        continue
      m = (qx == key[0]) & (qy == key[1])
      hh = t[(py[m] % Q).astype(int), (px[m] % Q).astype(int)].astype(int)
      h, s_, v = hh[:, 0], hh[:, 1], hh[:, 2]
      ok[m] = (s_ < 55) & (v > 35) & (v < 160)  # asphalt only: pavements and promenades are lighter
    return ok

  lane_px = LANE_M / mpp
  per_bin = defaultdict(list)
  n_lanes = 0
  for b, i, cls, oneway, pts in lines:
    seg = np.diff(pts, axis=0)
    gl = np.hypot(seg[:, 0], seg[:, 1] / K)
    cum = np.r_[0, np.cumsum(gl)]
    L = cum[-1]
    if L < 60:
      continue
    d = np.arange(0, L, 3.0)
    x = np.interp(d, cum, pts[:, 0])
    y = np.interp(d, cum, pts[:, 1])
    si = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
    gx, gy = seg[si, 0] / gl[si].clip(1e-6), -seg[si, 1] / K / gl[si].clip(1e-6)
    n = LANES[cls]
    lanes = [((k - (n - 1) / 2) * lane_px, 1) for k in range(n)] if oneway else \
            [((k + 0.5) * lane_px, 1) for k in range(n)] + [(-(k + 0.5) * lane_px, -1) for k in range(n)]
    for off, direction in lanes:
      ok = road(x - gy * off, y - gx * off * K)
      # close gaps up to 4 samples (12 px), then keep runs of >= 60 px
      okc = ok.copy()
      gap = 0
      for j in range(len(ok)):
        if ok[j]:
          if 0 < gap <= 28:  # ~25 m: tree shadows, crossings and markings do not end a lane
            okc[j - gap : j] = True
          gap = 0
        else:
          gap += 1
      edges = np.flatnonzero(np.diff(np.r_[0, okc.astype(int), 0]))
      for s0, s1 in zip(edges[::2], edges[1::2]):
        lo, hi = d[s0] + 6, d[s1 - 1] - 6
        if hi - lo >= 60:
          per_bin[b].append([i, round(off, 1), direction, round(lo, 1), round(hi, 1)])
          n_lanes += 1
  for b, rows in per_bin.items():
    (out / f"{b}.lanes.json").write_text(json.dumps(rows, separators=(",", ":")))
  print(f"visible lane stretches: {n_lanes} in {len(per_bin)} bins")


def _unused_atlases(bins, out):
  # sprite atlases per bin
  out = TRAFFIC / "cars"
  out.mkdir(parents=True, exist_ok=True)
  total = 0
  for b, cars in bins.items():
    if not cars:
      continue
    W, x, y, rowh = 1024, 0, 0, 0
    for c in cars:
      w, h = c["sprite"].size
      if x + w > W:
        x, y, rowh = 0, y + rowh + 1, 0
      c["ax"], c["ay"] = x, y
      x += w + 1
      rowh = max(rowh, h)
    atlas = Image.new("RGBA", (W, y + rowh + 1), (0, 0, 0, 0))
    for c in cars:
      atlas.paste(c["sprite"], (c["ax"], c["ay"]))
    atlas.save(out / f"{b}.png", optimize=True)
    rows = [[c["line"], c["d"], c["off"], c["dir"], *c["sprite"].size, c["ax"], c["ay"], c["x"], c["y"], c["lo"], c["hi"]] for c in cars]
    (out / f"{b}.json").write_text(json.dumps(rows, separators=(",", ":")))
    total += len(cars)
  print(f"{total} cars lifted from {len(qs)} quadrants into {len(bins)} bins -> {out}; clean quads -> {dst}")


if __name__ == "__main__":
  main()
