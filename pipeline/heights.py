"""Better building heights for a city: merge the best available open sources.

Priority (first that applies wins):
  1. OSM `height` / `building:levels` (via Overture buildings)     towers, landmarks
  2. hand-checked Wikidata towers (config `height_overrides`)      named supertalls
  3. Google Open Buildings 2.5D Temporal (2023, ~4 m, cap 100 m)   everything mid-rise
  4. GlobalBuildingAtlas x height_scale                             fallback

  uv run python pipeline/heights.py build mumbai     # cache sources in out/mumbai/heights/
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import city  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OB_BUCKET = "https://storage.googleapis.com/open-buildings-temporal-data/v1"
FLOOR_M = 3.3


def hdir(name):
  d = ROOT / "out" / name / "heights"
  d.mkdir(parents=True, exist_ok=True)
  return d


def bbox(cfg):
  return cfg["lon"] - 0.25, cfg["lat"] - 0.3, cfg["lon"] + 0.25, cfg["lat"] + 0.3


def build_osm(name, cfg):
  import duckdb

  x0, y0, x1, y1 = bbox(cfg)
  con = duckdb.connect()
  con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2'; SET geometry_always_xy = true;")
  out = hdir(name) / "osm.parquet"
  con.sql(f"""COPY (SELECT height, num_floors, ST_X(ST_Centroid(geometry)) lon, ST_Y(ST_Centroid(geometry)) lat
    FROM read_parquet('{city.OVERTURE}/theme=buildings/type=building/*', hive_partitioning=1)
    WHERE bbox.xmin BETWEEN {x0} AND {x1} AND bbox.ymin BETWEEN {y0} AND {y1}
      AND (height IS NOT NULL OR num_floors IS NOT NULL)) TO '{out}' (FORMAT parquet)""")
  print("osm heights:", con.sql(f"SELECT count(*) FROM read_parquet('{out}')").fetchall()[0][0])


def build_ob25d(name, cfg, factor=7):
  """Mosaic the 2023 Open Buildings 2.5D height + presence rasters over the city (UTM)."""
  import httpx
  import rasterio
  from pyproj import Transformer
  from rasterio.enums import Resampling

  x0, y0, x1, y1 = bbox(cfg)
  zone = int((cfg["lon"] + 180) // 6) + 1
  epsg = 32600 + zone
  t = Transformer.from_crs(4326, epsg, always_xy=True)
  ex = [t.transform(x, y) for x, y in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]]
  ux0, ux1 = min(e[0] for e in ex), max(e[0] for e in ex)
  uy0, uy1 = min(e[1] for e in ex), max(e[1] for e in ex)
  res = 0.5 * factor
  W, H = int((ux1 - ux0) / res) + 1, int((uy1 - uy0) / res) + 1
  height = np.zeros((H, W), np.float32)
  presence = np.zeros((H, W), np.float32)
  listing = httpx.get(f"https://storage.googleapis.com/storage/v1/b/open-buildings-temporal-data/o",
                      params={"prefix": "v1/manifests/", "maxResults": 5000}, timeout=60).json()
  mans = [i["name"] for i in listing["items"] if f"EPSG_{epsg}_2023" in i["name"]]
  n = 0
  for mname in mans:
    man = httpx.get(f"https://storage.googleapis.com/open-buildings-temporal-data/{mname}", timeout=120).json()
    pre = man["uriPrefix"].replace("gs://", "https://storage.googleapis.com/")
    for src in man["tilesets"][0]["sources"]:
      a = src["affineTransform"]
      tw = src["dimensions"]["width"] * a["scaleX"]
      tx0, ty1 = a["translateX"], a["translateY"]
      if tx0 > ux1 or tx0 + tw < ux0 or ty1 < uy0 or ty1 - tw > uy1:
        continue
      with rasterio.open("/vsicurl/" + pre + src["uris"][0]) as ds:
        ow = ds.width // factor
        hgt = ds.read(2, out_shape=(ow, ow), resampling=Resampling.average)
        prs = ds.read(3, out_shape=(ow, ow), resampling=Resampling.average)
      # paste into the mosaic
      cx = int(round((tx0 - ux0) / res))
      cy = int(round((uy1 - ty1) / res))
      sx0, sy0 = max(0, -cx), max(0, -cy)
      dx0, dy0 = max(0, cx), max(0, cy)
      w = min(ow - sx0, W - dx0)
      h = min(ow - sy0, H - dy0)
      if w > 0 and h > 0:
        height[dy0 : dy0 + h, dx0 : dx0 + w] = np.maximum(height[dy0 : dy0 + h, dx0 : dx0 + w], hgt[sy0 : sy0 + h, sx0 : sx0 + w])
        presence[dy0 : dy0 + h, dx0 : dx0 + w] = np.maximum(presence[dy0 : dy0 + h, dx0 : dx0 + w], prs[sy0 : sy0 + h, sx0 : sx0 + w])
        n += 1
        print(f"  tile {src['uris'][0]} pasted", flush=True)
  d = hdir(name)
  np.save(d / "ob25d_height.npy", height.astype(np.float16))
  np.save(d / "ob25d_presence.npy", presence.astype(np.float16))
  (d / "ob25d.json").write_text(json.dumps({"epsg": epsg, "x0": ux0, "y1": uy1, "res": res, "w": W, "h": H}))
  print(f"open buildings 2.5D: {n} tiles -> {W}x{H} @ {res} m")


class Heights:
  """Per-building height lookup used by city.chunk_prep."""

  def __init__(self, name, cfg):
    import duckdb
    from pyproj import Transformer
    from shapely import STRtree
    from shapely.geometry import Point

    d = hdir(name)
    self.cfg = cfg
    b = cfg.get("buildings", {})
    self.scale, self.hmin = b.get("height_scale", 1.0), b.get("min_height", 3.0)
    rows = duckdb.sql(f"SELECT height, num_floors, lon, lat FROM read_parquet('{d / 'osm.parquet'}')").fetchall()
    self.osm_h = [h if h else (f * FLOOR_M + 2 if f else None) for h, f, _, _ in rows]
    self.osm_pts = [Point(lo, la) for _, _, lo, la in rows]
    self.osm_tree = STRtree(self.osm_pts)
    ov = cfg.get("height_overrides", [])
    self.ov_pts = [Point(o["lon"], o["lat"]) for o in ov]
    self.ov_h = [o["height"] for o in ov]
    self.ov_tree = STRtree(self.ov_pts) if ov else None
    meta = json.loads((d / "ob25d.json").read_text())
    self.ob = meta
    self.ob_h = np.load(d / "ob25d_height.npy").astype(np.float32)
    self.ob_p = np.load(d / "ob25d_presence.npy").astype(np.float32)
    self.utm = Transformer.from_crs(4326, meta["epsg"], always_xy=True)
    # curated skyscrapers (Wikipedia list of tallest buildings in Mumbai, with coordinates):
    # no open raster source gets Mumbai's 150-300 m towers right (Open Buildings caps at 100 m)
    tw = json.loads((ROOT / "out" / name / "towers.json").read_text()) if (ROOT / "out" / name / "towers.json").exists() else []
    self.tw_pts = [Point(t["lon"], t["lat"]) for t in tw]
    self.tw_h = [t["height"] for t in tw]
    self.tw_tree = STRtree(self.tw_pts) if tw else None
    self.stats = {"tower": 0, "osm": 0, "wikidata": 0, "ob25d": 0, "gba": 0}

  def height(self, poly_ll, gba_h):
    """poly_ll: shapely Polygon in lon/lat."""
    from PIL import Image, ImageDraw

    for tree, pts, hs, key in [(self.ov_tree, self.ov_pts, self.ov_h, "wikidata"), (self.osm_tree, self.osm_pts, self.osm_h, "osm")]:
      if tree is None:
        continue
      hits = [i for i in tree.query(poly_ll) if hs[i] and poly_ll.contains(pts[i])]
      if hits:
        self.stats[key] += 1
        return max(self.hmin, max(hs[i] for i in hits))
    # Open Buildings 2.5D: high percentile of confident pixels inside the footprint
    m = self.ob
    xs, ys = self.utm.transform(*np.array(poly_ll.exterior.coords).T)
    cx, cy = (np.asarray(xs) - m["x0"]) / m["res"], (m["y1"] - np.asarray(ys)) / m["res"]
    c0, c1 = int(max(0, math.floor(cx.min()))), int(min(m["w"] - 1, math.ceil(cx.max())))
    r0, r1 = int(max(0, math.floor(cy.min()))), int(min(m["h"] - 1, math.ceil(cy.max())))
    if c1 >= c0 and r1 >= r0:
      if c1 - c0 < 2 or r1 - r0 < 2:  # small footprint: sample the centre pixel
        r, c = int(np.clip(cy.mean(), 0, m["h"] - 1)), int(np.clip(cx.mean(), 0, m["w"] - 1))
        vals = self.ob_h[r, c][None] if self.ob_p[r, c] > 0.5 else np.array([])
      else:
        mk = Image.new("L", (c1 - c0 + 1, r1 - r0 + 1))
        ImageDraw.Draw(mk).polygon([(x - c0, y - r0) for x, y in zip(cx, cy)], fill=1)
        mk = np.asarray(mk).astype(bool) & (self.ob_p[r0 : r1 + 1, c0 : c1 + 1] > 0.5)
        vals = self.ob_h[r0 : r1 + 1, c0 : c1 + 1][mk]
      if len(vals):
        self.stats["ob25d"] += 1
        return max(self.hmin, float(np.percentile(vals, 85)))
    self.stats["gba"] += 1
    return max(self.hmin, (gba_h if gba_h and gba_h > 0 else self.hmin) * self.scale)


if __name__ == "__main__":
  cmd, name = sys.argv[1], sys.argv[2]
  cfg = json.loads((ROOT / "config" / f"{name}.json").read_text())
  if cmd == "build":
    build_osm(name, cfg)
    build_ob25d(name, cfg)
