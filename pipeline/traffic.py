"""Road / rail / ferry network for the site's simulated traffic (site/traffic.js).

Overture transportation segments -> polylines in global image pixels (ground plane),
merged into longer chains where segments meet end to end, simplified, and binned into
square cells so the browser only loads what is on screen.

  site/public/traffic/index.json          {cell, classes, bins: ["x_y", ...]}
  site/public/traffic/<x>_<y>.json        {"r": [[cls, x0, y0, dx, dy, ...], ...], "t": [...], "f": [...]}

  uv run python pipeline/traffic.py mumbai
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import city  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CELL = 8192  # px per bin side (~2.4 km)
ROADS = ["motorway", "trunk", "primary", "secondary", "tertiary"]  # main roads only: they read clearly in the art
OCC_DOWN = 4  # occlusion mask resolution: 1 px per 4 image px
RAIL = {"standard_gauge": "rail", "unknown": "rail", "subway": "metro", "monorail": "mono"}


def simplify(pts, tol=3.0):
  """Douglas-Peucker on an (n, 2) array."""
  if len(pts) < 3:
    return pts
  a, b = pts[0], pts[-1]
  ab = b - a
  n = np.hypot(*ab) or 1.0
  v = pts[1:-1] - a
  d = np.abs(ab[0] * v[:, 1] - ab[1] * v[:, 0]) / n
  i = int(np.argmax(d))
  if d[i] > tol:
    return np.vstack([simplify(pts[: i + 2], tol)[:-1], simplify(pts[i + 1 :], tol)])
  return np.vstack([a, b])


def chain(lines):
  """Merge polylines that meet end to end (only where exactly two lines meet) into longer runs."""
  key = lambda p: (round(p[0] / 2), round(p[1] / 2))
  ends = defaultdict(list)
  for i, l in enumerate(lines):
    ends[key(l[0])].append(i)
    ends[key(l[-1])].append(i)
  used, out = set(), []
  for i in range(len(lines)):
    if i in used:
      continue
    used.add(i)
    cur = [tuple(p) for p in lines[i]]
    for forward in (True, False):
      while True:
        tip = key(cur[-1] if forward else cur[0])
        cand = [j for j in ends[tip] if j not in used]
        if len(ends[tip]) != 2 or not cand:
          break
        j = cand[0]
        used.add(j)
        seg = [tuple(p) for p in lines[j]]
        if forward:  # orient so it starts at the tip, then append
          if key(seg[0]) != tip:
            seg = seg[::-1]
          cur = cur + seg[1:]
        else:  # orient so it ends at the tip, then prepend
          if key(seg[-1]) != tip:
            seg = seg[::-1]
          cur = seg[:-1] + cur
    out.append(np.array(cur, np.float64))
  return out


def _occ_bin(args):
  """Building silhouettes (from the hybrid renders' mask pass) for one bin, downsampled: vehicles
  are erased wherever a building stands in front of the road in the isometric view."""
  from PIL import Image
  rdir, b, out = args
  bx, by = map(int, b.split("_"))
  n = CELL // 1024
  t = 1024 // OCC_DOWN
  img = Image.new("L", (n * t, n * t), 0)
  any_ = False
  for j in range(n):
    for i in range(n):
      p = rdir / f"mask_{bx * n + i}_{by * n + j}.png"
      if p.exists():
        a = Image.open(p).convert("L").resize((t, t), Image.Resampling.BOX)
        img.paste(a, (i * t, j * t))
        any_ = True
  if not any_:
    return
  a = np.asarray(img)
  a = np.where(a > 110, 255, 0).astype(np.uint8)
  la = Image.merge("LA", (Image.new("L", img.size, 0), Image.fromarray(a)))
  la.save(out / f"{b}.png", optimize=True)


def occlusion(name, bins, out):
  from concurrent.futures import ProcessPoolExecutor
  out.mkdir(parents=True, exist_ok=True)
  rdir = ROOT / "out" / name / "renders"
  with ProcessPoolExecutor(8) as ex:
    list(ex.map(_occ_bin, [(rdir, b, out) for b in bins]))
  size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
  print(f"occlusion masks: {len(list(out.iterdir()))} bins, {size:.1f} MB")


def main():
  import duckdb

  name = sys.argv[1]
  cfg, pl, active = city.load(name)
  enu = city.enu_fn(cfg["lat"], cfg["lon"])
  (rx, ry), (ux, uy) = city.basis(cfg)
  mpp = cfg["meters_per_pixel"]
  W, H = pl["width"], pl["height"]

  def to_px(lon, lat):
    x, y = enu(lat, lon)
    return np.stack([W / 2 + (x * rx + y * ry) / mpp, H / 2 - (x * ux + y * uy) / mpp], 1)

  con = duckdb.connect()
  con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2'; SET geometry_always_xy = true;")
  lo, la = cfg["lon"], cfg["lat"]
  rows = con.sql(
    f"""SELECT subtype, class, ST_AsGeoJSON(geometry), CAST(to_json(access_restrictions) AS VARCHAR) FROM read_parquet('{city.OVERTURE}/theme=transportation/type=segment/*', hive_partitioning=1)
        WHERE bbox.xmax > {lo - 0.25} AND bbox.xmin < {lo + 0.25} AND bbox.ymax > {la - 0.3} AND bbox.ymin < {la + 0.3}
          AND ((subtype = 'road' AND class IN ({",".join(f"'{c}'" for c in ROADS)}))
               OR (subtype = 'rail' AND class IN ({",".join(f"'{c}'" for c in RAIL)})) OR subtype = 'water')"""
  ).fetchall()
  groups = defaultdict(list)  # (kind, cls) -> polylines
  for sub, cls, gj, acc in rows:
    g = json.loads(gj)
    oneway = sub == "road" and acc is not None and '"backward"' in acc and '"denied"' in acc
    if g["type"] != "LineString" or len(g["coordinates"]) < 2:
      continue
    a = np.array(g["coordinates"])
    px = to_px(a[:, 0], a[:, 1])
    kind = "r" if sub == "road" else ("t" if sub == "rail" else "f")
    c = cls if sub == "road" else (RAIL.get(cls, "rail") if sub == "rail" else "ferry")
    groups[(kind, c, oneway)].append(px)

  classes = sorted({c for _, c, _ in groups})
  bins = defaultdict(lambda: {"r": [], "t": [], "f": []})
  n_lines = n_pts = 0
  for (kind, c, oneway), lines in groups.items():
    # never chain one-way segments: chaining may reverse them
    merged = lines if oneway else chain(lines)
    for l in merged:
      l = simplify(l, 2.5 if kind == "r" else 4.0)
      if len(l) < 2 or np.hypot(*(l[-1] - l[0])) < 20 and len(l) == 2:
        continue
      # keep lines that touch the generated map (active quadrants)
      qx, qy = (l[:, 0] // 512).astype(int), (l[:, 1] // 512).astype(int)
      ok = (qx >= 0) & (qx < pl["cols"]) & (qy >= 0) & (qy < pl["rows"])
      if not ok.any() or not active[qy[ok], qx[ok]].any():
        continue
      li = np.round(l).astype(int)
      mid = li[len(li) // 2]
      b = f"{mid[0] // CELL}_{mid[1] // CELL}"
      flat = [classes.index(c) + (32 if oneway else 0), int(li[0, 0]), int(li[0, 1])] + np.diff(li, axis=0).ravel().tolist()
      bins[b][kind].append(flat)
      n_lines += 1
      n_pts += len(li)

  out = ROOT / "site" / "public" / "traffic"
  if out.exists():
    import shutil
    shutil.rmtree(out)
  out.mkdir(parents=True)
  for b, d in bins.items():
    (out / f"{b}.json").write_text(json.dumps(d, separators=(",", ":")))
  (out / "index.json").write_text(json.dumps({"cell": CELL, "classes": classes, "bins": sorted(bins), "occ": OCC_DOWN}))
  occlusion(name, sorted(bins), out / "occ")
  size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
  print(f"{n_lines} lines, {n_pts} points in {len(bins)} bins, {size:.1f} MB -> {out}")


if __name__ == "__main__":
  main()
