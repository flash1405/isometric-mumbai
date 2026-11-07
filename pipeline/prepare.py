"""Build the inputs for the hybrid renderer.

  1. ortho.jpg      stitched top-down Google render (from `npm run render -- <area> --stage=ortho`)
  2. buildings.json GlobalBuildingAtlas LoD1 footprints + heights, in local metres

Local frame: origin at the config's lat/lon, x = east, y = north (metres).
The ortho image is north-up and centred on the origin, so pixel (u, v) <-> (x, y) is
x = (u - S/2) * mpp, y = (S/2 - v) * mpp.

  uv run python pipeline/prepare.py fort
"""

import json
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GBA = "https://data.source.coop/tge-labs/globalbuildingatlas-lod1/{tile}.parquet"
Image.MAX_IMAGE_PIXELS = None


def gba_tile(lat, lon):
  # files are named {e|w}{lon0}_n{lat1}_{e|w}{lon1}_n{lat0} on a 5 degree grid (northern hemisphere)
  lon0, lat0 = int(math.floor(lon / 5) * 5), int(math.floor(lat / 5) * 5)
  ew = lambda v: f"w{-v:03d}" if v < 0 else f"e{v:03d}"
  return f"{ew(lon0)}_n{lat0 + 5:02d}_{ew(lon0 + 5)}_n{lat0:02d}"


def to_local(lat0, lon0):
  p = math.radians(lat0)  # metres per degree on WGS84 at this latitude
  kx = 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p) + 0.118 * math.cos(5 * p)
  ky = 111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p) - 0.0023 * math.cos(6 * p)
  return lambda lon, lat: ((lon - lon0) * kx, (lat - lat0) * ky), kx, ky


def stitch_ortho(name, cfg):
  q, n = cfg["quadrant_px"], cfg["ortho"]["quadrants"]
  tiles = (n + 1) // 2
  size = n * q
  img = Image.new("RGB", (size, size))
  d = ROOT / "out" / name / "ortho_tiles"
  for ty in range(tiles):
    for tx in range(tiles):
      img.paste(Image.open(d / f"tile_{tx}_{ty}.png").convert("RGB"), (tx * 2 * q, ty * 2 * q))
  out = ROOT / "out" / name / "ortho.jpg"
  img.save(out, quality=92)
  print(f"ortho {size}x{size}px -> {out}")
  return size


def stitch_dsm(name, cfg):
  """Surface height above local ground (metres) from the ortho pass's depth tiles, or None."""
  import numpy as np
  from numpy.lib.stride_tricks import sliding_window_view

  d = ROOT / "out" / name / "ortho_tiles"
  if not (d / "depth_0_0.png").exists():
    return None
  q, n = cfg["quadrant_px"], cfg["ortho"]["quadrants"]
  size = n * q
  dist = np.full((size, size), np.nan, np.float32)
  for ty in range((n + 1) // 2):
    for tx in range((n + 1) // 2):
      a = np.asarray(Image.open(d / f"depth_{tx}_{ty}.png").convert("RGB")).astype(np.int64)
      mm = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
      h, w = min(1024, size - ty * 1024), min(1024, size - tx * 1024)
      dist[ty * 1024 : ty * 1024 + h, tx * 1024 : tx * 1024 + w] = mm[:h, :w] / 1000.0
  # anti-aliased edges blend the RGB-encoded depth into garbage values: clip to the bulk
  valid = (dist > 1) & (dist < 16000)
  lo, hi = np.nanpercentile(np.where(valid, dist, np.nan)[::4, ::4], [0.5, 99.5])
  surf = np.where(valid & (dist >= lo) & (dist <= hi), -dist, np.nan)
  # local ground: robust low surface on a 2.4 m grid, then 5th percentile in ~60 m windows
  k = 8
  m = size // k * k
  blocks = surf[:m, :m].reshape(m // k, k, m // k, k).transpose(0, 2, 1, 3).reshape(m // k, m // k, k * k)
  small = np.nanpercentile(blocks, 20, axis=2)
  small = np.where(np.isnan(small), np.nanmedian(small), small)
  win = 25
  pad = np.pad(small, win // 2, mode="edge")
  ground = np.percentile(sliding_window_view(pad, (win, win)).reshape(small.shape[0], small.shape[1], -1), 5, axis=2)
  ground = np.kron(ground, np.ones((k, k), np.float32))
  out = np.full_like(surf, np.nan)
  out[:m, :m] = surf[:m, :m] - ground
  return out


def dsm_height(dsm, ring, size, mpp):
  import numpy as np
  from PIL import ImageDraw

  xs = [x / mpp + size / 2 for x, _ in ring]
  ys = [size / 2 - y / mpp for _, y in ring]
  x0, x1 = int(max(0, min(xs))), int(min(size - 1, max(xs)))
  y0, y1 = int(max(0, min(ys))), int(min(size - 1, max(ys)))
  if x1 - x0 < 2 or y1 - y0 < 2:
    return None
  m = Image.new("L", (x1 - x0 + 1, y1 - y0 + 1))
  ImageDraw.Draw(m).polygon([(x - x0, y - y0) for x, y in zip(xs, ys)], fill=1)
  vals = dsm[y0 : y1 + 1, x0 : x1 + 1][np.asarray(m).astype(bool)]
  vals = vals[~np.isnan(vals)]
  return float(np.percentile(vals, 60)) if len(vals) >= 10 else None  # typical, not peak, height


def fetch_buildings(name, cfg, extent_m):
  import duckdb

  lat0, lon0 = cfg["lat"], cfg["lon"]
  proj, kx, ky = to_local(lat0, lon0)
  half = extent_m / 2
  lon_min, lon_max = lon0 - half / kx, lon0 + half / kx
  lat_min, lat_max = lat0 - half / ky, lat0 + half / ky
  b = cfg.get("buildings", {})
  scale, hmin = b.get("height_scale", 1.0), b.get("min_height", 3.0)

  con = duckdb.connect()
  con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET geometry_always_xy = true;")
  url = GBA.format(tile=gba_tile(lat0, lon0))
  rows = con.sql(
    f"""SELECT height, source, ST_AsGeoJSON(geometry) FROM read_parquet('{url}')
        WHERE bbox.xmax >= {lon_min} AND bbox.xmin <= {lon_max}
          AND bbox.ymax >= {lat_min} AND bbox.ymin <= {lat_max}"""
  ).fetchall()

  dsm = stitch_dsm(name, cfg) if cfg.get("heights_from") == "dsm" else None
  size, mpp = cfg["ortho"]["quadrants"] * cfg["quadrant_px"], cfg["ortho"]["meters_per_pixel"]
  out = []
  for height, source, gj in rows:
    geom = json.loads(gj)
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    h = max(hmin, (height if height and height > 0 else hmin) * scale)
    for poly in polys:
      rings = [[[round(c, 2) for c in proj(lon, lat)] for lon, lat in ring[:-1]] for ring in poly]
      if len(rings[0]) >= 3:
        hb = h
        if dsm is not None:  # true height from the real 3D mesh where available
          hd = dsm_height(dsm, rings[0], size, mpp)
          hb = max(hmin, hd) if hd is not None else h
        out.append({"h": round(hb, 1), "src": source, "rings": rings})

  for o in cfg.get("landmarks", []):  # optional manual height overrides
    px, py = proj(o["lon"], o["lat"])
    for bld in out:
      if point_in_ring(px, py, bld["rings"][0]):
        bld["h"] = o["height"]

  path = ROOT / "out" / name / "buildings.json"
  path.write_text(json.dumps(out, separators=(",", ":")))
  hs = sorted(b["h"] for b in out)
  print(f"{len(out)} buildings (median {hs[len(hs) // 2]} m, max {hs[-1]} m) -> {path}")


def point_in_ring(x, y, ring):
  inside = False
  j = len(ring) - 1
  for i in range(len(ring)):
    xi, yi = ring[i]
    xj, yj = ring[j]
    if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
      inside = not inside
    j = i
  return inside


def main():
  name = sys.argv[1] if len(sys.argv) > 1 else "fort"
  cfg = json.loads((ROOT / "config" / f"{name}.json").read_text())
  size = stitch_ortho(name, cfg)
  fetch_buildings(name, cfg, size * cfg["ortho"]["meters_per_pixel"])


if __name__ == "__main__":
  main()
