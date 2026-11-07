"""Lanes that sit on the painted roads, for the site's traffic.

The model draws roads a little off the OpenStreetMap centrelines (and medians, footpaths and
promenades come out grey too), so lane offsets measured from the map line put cars off the
carriageway. For every main road line: sample perpendicular profiles every STEP px, find the run
of painted asphalt that contains (or is nearest to) the mapped line, and store its left/right
edges. The site places lanes inside those edges, so a vehicle is always on painted asphalt.
Stretches without a carriageway are not driven.

  site/public/traffic/cars/<bin>.roads.json   [[line, lo, step, [left edges], [right edges]], ...]
     (edges are ground-left offsets in image px from the mapped line; lo/step in ground px)

  uv run --with opencv-python-headless python pipeline/road_snap.py mumbai [--src quads_v3] [--debug x0:x1,y0:y1]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
TRAFFIC = ROOT / "site" / "public" / "traffic"
Q, K, STEP, REACH = 512, 0.7071067811865476, 9.0, 75
MAIN = {"motorway", "trunk", "primary", "secondary", "tertiary"}
MAX_W = {"motorway": 40, "trunk": 34, "primary": 26, "secondary": 18, "tertiary": 13}  # carriageway width cap, m


def road_lines():
  idx = json.loads((TRAFFIC / "index.json").read_text())
  out = []
  for b in idx["bins"]:
    d = json.loads((TRAFFIC / f"{b}.json").read_text())
    for i, arr in enumerate(d.get("r", [])):
      cls = idx["classes"][arr[0] & 31]
      if cls not in MAIN:
        continue
      pts = [[arr[1], arr[2]]]
      for k in range(3, len(arr), 2):
        pts.append([pts[-1][0] + arr[k], pts[-1][1] + arr[k + 1]])
      out.append((b, i, cls, arr[0] >= 32, np.array(pts, float)))
  return out


class Pix:
  def __init__(self, qdir):
    self.qdir, self.cache = Path(qdir), {}

  def asphalt(self, px, py):
    ok = np.zeros(px.shape, bool)
    qx, qy = (px // Q).astype(int), (py // Q).astype(int)
    for key in set(zip(qx.ravel().tolist(), qy.ravel().tolist())):
      if key not in self.cache:
        p = self.qdir / f"{key[0]}_{key[1]}.png"
        self.cache[key] = cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2HSV) if p.exists() else None
      t = self.cache[key]
      if t is None:
        continue
      m = (qx == key[0]) & (qy == key[1])
      hsv = t[(py[m] % Q).astype(int), (px[m] % Q).astype(int)].astype(int)
      h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
      # asphalt (incl. shade) and the paint on it (white / yellow lane lines), not pavements
      ok[m] = ((s < 55) & (v > 22) & (v < 150)) | ((v > 200) & (s < 35)) | ((h > 15) & (h < 35) & (s > 110) & (v > 150))
    return ok


def snap(cls, pts, pix, mpp):
  seg = np.diff(pts, axis=0)
  gl = np.hypot(seg[:, 0], seg[:, 1] / K)
  cum = np.r_[0, np.cumsum(gl)]
  if cum[-1] < 80:
    return []
  d = np.arange(0, cum[-1], STEP)
  x, y = np.interp(d, cum, pts[:, 0]), np.interp(d, cum, pts[:, 1])
  si = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
  gx, gy = seg[si, 0] / gl[si].clip(1e-6), -seg[si, 1] / K / gl[si].clip(1e-6)
  offs = np.arange(-REACH, REACH + 0.1, 1.0)
  # majority over 3 samples along the line, so a lane marking or a crossing does not split the road
  vote = np.zeros((len(d), len(offs)))
  for a in (-3.0, 0.0, 3.0):
    px = (x + gx * a)[:, None] - gy[:, None] * offs[None, :]
    py = (y - gy * a * K)[:, None] - gx[:, None] * offs[None, :] * K
    vote += pix.asphalt(px, py)
  road = vote >= 2
  zero = len(offs) // 2
  maxw = MAX_W[cls] / mpp
  L, R = np.full(len(d), np.nan), np.full(len(d), np.nan)
  for k in range(len(d)):
    r = road[k]
    if not r[max(0, zero - 12) : zero + 13].any():
      continue
    # the asphalt run that contains the mapped line, or the nearest one within ~3.5 m
    edges = np.flatnonzero(np.diff(np.r_[0, r.astype(int), 0]))
    runs = [(e0, e1) for e0, e1 in zip(edges[::2], edges[1::2]) if e1 - e0 >= 4]
    if not runs:
      continue
    e0, e1 = min(runs, key=lambda q: 0 if q[0] <= zero < q[1] else min(abs(q[0] - zero), abs(q[1] - zero)))
    lo_, hi_ = offs[e0], offs[e1 - 1]
    if hi_ - lo_ > maxw:  # a plaza / junction box: keep a road-width band around the line
      c = np.clip(0, lo_ + maxw / 2, hi_ - maxw / 2)
      lo_, hi_ = c - maxw / 2, c + maxw / 2
    L[k], R[k] = lo_, hi_
  ok = ~np.isnan(L)
  # smooth: centre line median over ~140 px, width = median over ~270 px; edges rebuilt from those
  C, Wd = (L + R) / 2, R - L
  def rmed(a, h):
    o = a.copy()
    for k in range(len(a)):
      w = a[max(0, k - h) : k + h + 1]
      w = w[~np.isnan(w)]
      if len(w) >= max(3, h // 2):
        o[k] = np.median(w)
    return o
  C, Wd = rmed(rmed(C, 7), 4), rmed(Wd, 15)
  L, R = C - Wd / 2, C + Wd / 2
  okc = ok.copy()
  for k in range(1, len(d) - 2):
    if not ok[k] and (ok[k - 1] and (ok[k + 1] or ok[k + 2])):
      okc[k] = True
      L[k] = L[k - 1] if np.isnan(L[k]) else L[k]
      R[k] = R[k - 1] if np.isnan(R[k]) else R[k]
  edges = np.flatnonzero(np.diff(np.r_[0, okc.astype(int), 0]))
  out = []
  for e0, e1 in zip(edges[::2], edges[1::2]):
    if (e1 - e0) * STEP >= 70 and not np.isnan(L[e0:e1]).any():
      out.append((float(d[e0]), [round(float(v), 1) for v in L[e0:e1]], [round(float(v), 1) for v in R[e0:e1]], x[e0:e1], y[e0:e1], gx[e0:e1], gy[e0:e1]))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("city")
  ap.add_argument("--src", default="quads_v2_clean")
  ap.add_argument("--debug")
  args = ap.parse_args()
  cfg = json.loads((ROOT / "config" / f"{args.city}.json").read_text())
  mpp = cfg["meters_per_pixel"]
  pix = Pix(ROOT / "out" / args.city / args.src)
  lines = road_lines()
  if args.debug:
    (a, b), (c, e) = [map(int, s.split(":")) for s in args.debug.split(",")]
    box = (a * Q, c * Q, b * Q, e * Q)
    lines = [ln for ln in lines if ((ln[4][:, 0] > box[0]) & (ln[4][:, 0] < box[2]) & (ln[4][:, 1] > box[1]) & (ln[4][:, 1] < box[3])).any()]
  per_bin, dbg = defaultdict(list), []
  n = 0
  for bn, i, cls, oneway, pts in lines:
    for lo, Ls, Rs, x, y, gx, gy in snap(cls, pts, pix, mpp):
      per_bin[bn].append([i, round(lo, 1), STEP, Ls, Rs])
      n += 1
      if args.debug:
        for eg in (np.array(Ls), np.array(Rs)):
          dbg.append(np.stack([x - gy * eg, y - gx * eg * K], 1))
  if args.debug:
    W = Image.new("RGB", (box[2] - box[0], box[3] - box[1]))
    for qx in range(a, b):
      for qy in range(c, e):
        p = pix.qdir / f"{qx}_{qy}.png"
        if p.exists():
          W.paste(Image.open(p), ((qx - a) * Q, (qy - c) * Q))
    dr = ImageDraw.Draw(W)
    for ln in lines:
      dr.line([tuple(p - box[:2]) for p in ln[4]], fill=(255, 0, 255), width=1)
    for pl in dbg:
      dr.line([tuple(p - box[:2]) for p in pl], fill=(0, 255, 120), width=2)
    W.save(ROOT / "out" / "shots" / "road_debug.png")
    print(f"{n} stretches; magenta = OSM line, green = painted carriageway edges")
    return
  out = TRAFFIC / "cars"
  out.mkdir(parents=True, exist_ok=True)
  for f in out.glob("*.roads.json"):
    f.unlink()
  for bn, rows in per_bin.items():
    (out / f"{bn}.roads.json").write_text(json.dumps(rows, separators=(",", ":")))
  print(f"{n} carriageway stretches in {len(per_bin)} bins")


if __name__ == "__main__":
  main()
