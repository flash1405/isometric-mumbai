"""Find the painted railway track bed along each suburban rail line, for the site's trains.

The model draws tracks a little off the OpenStreetMap lines, so trains drawn on the raw lines
float beside the tracks. For every rail line, sample perpendicular profiles, classify ballast
pixels (brown gravel / sleepers), take the widest run of track bed, and record its centre and
half-width; keep only stretches where painted track actually exists.

  site/public/traffic/rails/<bin>.json   [[line, lo, step, [ground-left shifts...]], ...]  (drawn stretches)

  uv run --with opencv-python-headless python pipeline/rail_snap.py mumbai [--src quads_v2_clean] [--debug x0:x1,y0:y1]
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
Q, K, STEP, REACH = 512, 0.7071067811865476, 9.0, 60


def rail_lines():
  idx = json.loads((TRAFFIC / "index.json").read_text())
  out = []
  for b in idx["bins"]:
    d = json.loads((TRAFFIC / f"{b}.json").read_text())
    for i, arr in enumerate(d.get("t", [])):
      if idx["classes"][arr[0] & 31] != "rail":
        continue
      pts = [[arr[1], arr[2]]]
      for k in range(3, len(arr), 2):
        pts.append([pts[-1][0] + arr[k], pts[-1][1] + arr[k + 1]])
      out.append((b, i, np.array(pts, float)))
  return out


class Pix:
  def __init__(self, qdir):
    self.qdir, self.cache = Path(qdir), {}

  def ballast(self, px, py):
    """Track-bed pixels: brown gravel and sleepers (plus the grey rails between them)."""
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
      ok[m] = ((h <= 22) | (h >= 170)) & (s >= 30) & (s <= 150) & (v >= 55) & (v <= 175)
    return ok


def snap(pts, pix):
  seg = np.diff(pts, axis=0)
  gl = np.hypot(seg[:, 0], seg[:, 1] / K)
  cum = np.r_[0, np.cumsum(gl)]
  if cum[-1] < 150:
    return []
  d = np.arange(0, cum[-1], STEP)
  x, y = np.interp(d, cum, pts[:, 0]), np.interp(d, cum, pts[:, 1])
  si = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
  gx, gy = seg[si, 0] / gl[si].clip(1e-6), -seg[si, 1] / K / gl[si].clip(1e-6)
  offs = np.arange(-REACH, REACH + 0.1, 1.5)
  # ground-left offset o -> screen (x - gy*o, y - gx*o*K); 3 samples along the line per offset
  score = np.zeros((len(d), len(offs)))
  for a in (-3.0, 0.0, 3.0):
    px = (x - gy * 0 + gx * a)[:, None] - gy[:, None] * offs[None, :]
    py = (y + gy * a * K * -1)[:, None] - gx[:, None] * offs[None, :] * K
    score += pix.ballast(px, py)
  score /= 3
  # keep each mapped track where it is if it is on painted track bed; otherwise nudge it onto
  # the nearest bed within 6 m; samples with no bed nearby are not drawn
  bed = score > 0.45
  zero = len(offs) // 2
  shift, ok = np.zeros(len(d)), np.zeros(len(d), bool)
  near = np.abs(offs) <= 20
  for k in range(len(d)):
    if bed[k, max(0, zero - 2) : zero + 3].any():
      ok[k] = True
      continue
    cand = np.flatnonzero(bed[k] & near)
    if len(cand):
      o = offs[cand[np.argmin(np.abs(offs[cand]))]]
      shift[k], ok[k] = o + 3 * np.sign(o), True
  sm = shift.copy()
  for k in range(len(d)):  # slow-varying: median over ~140 px
    w = ok[max(0, k - 7) : k + 8]
    if w.any():
      sm[k] = np.median(shift[max(0, k - 7) : k + 8][w])
  okc = np.convolve(ok, np.ones(5), "same") >= 3
  edges = np.flatnonzero(np.diff(np.r_[0, okc.astype(int), 0]))
  out = []
  for e0, e1 in zip(edges[::2], edges[1::2]):
    if (e1 - e0) * STEP >= 120:
      out.append((float(d[e0]), [round(v, 1) for v in sm[e0:e1]], None, x, y, gx, gy, e0, e1))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("city")
  ap.add_argument("--src", default="quads_v2_clean")
  ap.add_argument("--debug", help="qx0:qx1,qy0:qy1: draw the snapped track centre over the art")
  args = ap.parse_args()
  pix = Pix(ROOT / "out" / args.city / args.src)
  per_bin = defaultdict(list)
  dbg = []
  lines = rail_lines()
  if args.debug:
    (a, b), (c, e) = [map(int, s.split(":")) for s in args.debug.split(",")]
    box = (a * Q, c * Q, b * Q, e * Q)
    lines = [ln for ln in lines if ((ln[2][:, 0] > box[0]) & (ln[2][:, 0] < box[2]) & (ln[2][:, 1] > box[1]) & (ln[2][:, 1] < box[3])).any()]
  n = 0
  for bn, i, pts in lines:
    for lo, cs, hs, x, y, gx, gy, e0, e1 in snap(pts, pix):
      per_bin[bn].append([i, round(lo, 1), STEP, cs])
      n += 1
      dbg.append(np.stack([x[e0:e1] - gy[e0:e1] * np.array(cs), y[e0:e1] - gx[e0:e1] * np.array(cs) * K], 1))
  if args.debug:
    W = Image.new("RGB", (box[2] - box[0], box[3] - box[1]))
    for qx in range(a, b):
      for qy in range(c, e):
        p = pix.qdir / f"{qx}_{qy}.png"
        if p.exists():
          W.paste(Image.open(p), ((qx - a) * Q, (qy - c) * Q))
    dr = ImageDraw.Draw(W)
    for ln in lines:
      dr.line([tuple(p - box[:2]) for p in ln[2]], fill=(255, 0, 255), width=2)
    for pl in dbg:
      dr.line([tuple(p - box[:2]) for p in pl], fill=(0, 255, 255), width=3)
    W.save(ROOT / "out" / "shots" / "rail_debug.png")
    print(f"{n} snapped stretches; magenta = OSM line, cyan = snapped track centre -> out/shots/rail_debug.png")
    return
  out = TRAFFIC / "rails"
  out.mkdir(parents=True, exist_ok=True)
  for b, rows in per_bin.items():
    (out / f"{b}.json").write_text(json.dumps(rows, separators=(",", ":")))
  print(f"{n} snapped track stretches in {len(per_bin)} bins -> {out}")


if __name__ == "__main__":
  main()
