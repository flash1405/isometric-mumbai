"""Consistent sea: Overture ocean/river polygons -> water mask per quadrant + one texture.

The model paints water slightly differently in every tile, which shows as a patchwork.
At export, water pixels (ground plane, z = 0) are replaced by a single seamless texture
sampled at global image coordinates; land keeps the generated art.

  uv run python pipeline/water.py build mumbai     -> out/mumbai/water_iso.wkb
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import city  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q = 512
BASE = np.array([12, 74, 86], np.float32)  # matches the teacher model's deep teal sea
_TEX = None
_TREE = None


def build(name):
  import duckdb
  from shapely import wkb
  from shapely.affinity import affine_transform
  from shapely.geometry import box
  from shapely.ops import unary_union

  cfg, pl, _ = city.load(name)
  con = duckdb.connect()
  con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2'; SET geometry_always_xy = true;")
  lo, la = cfg["lon"], cfg["lat"]
  env = f"ST_MakeEnvelope({lo - 0.3}, {la - 0.35}, {lo + 0.3}, {la + 0.35})"
  rows = con.sql(
    f"""SELECT ST_AsWKB(ST_Intersection(geometry, {env})) FROM read_parquet('{city.OVERTURE}/theme=base/type=water/*', hive_partitioning=1)
        WHERE bbox.xmax > {lo - 0.3} AND bbox.xmin < {lo + 0.3} AND bbox.ymax > {la - 0.35} AND bbox.ymin < {la + 0.35}
          AND subtype IN ('ocean', 'river') """
  ).fetchall()
  geoms = [wkb.loads(bytes(r[0])) for r in rows if r[0]]
  water = unary_union([g for g in geoms if not g.is_empty]).buffer(0)
  # lon/lat -> ENU -> global iso pixels; ENU is ~affine over the city, fit it exactly at 3 points
  enu = city.enu_fn(la, lo)
  (rx, ry), (ux, uy) = city.basis(cfg)
  mpp = cfg["meters_per_pixel"]
  pts = np.array([[lo, la], [lo + 0.2, la], [lo, la + 0.2]])
  x, y = enu(pts[:, 1], pts[:, 0])
  px = pl["width"] / 2 + (x * rx + y * ry) / mpp
  py = pl["height"] / 2 - (x * ux + y * uy) / mpp
  A = np.c_[pts, np.ones(3)]
  cx, cy = np.linalg.solve(A, px), np.linalg.solve(A, py)
  iso = affine_transform(water, [cx[0], cx[1], cy[0], cy[1], cx[2], cy[2]])
  iso = iso.intersection(box(0, 0, pl["width"], pl["height"]))
  (ROOT / "out" / name / "water_iso.wkb").write_bytes(iso.wkb)
  print(f"water polygons -> out/{name}/water_iso.wkb ({iso.area / (Q * Q):.0f} quadrants of water)")


def _load(name):
  global _TREE
  if _TREE is None:
    from shapely import STRtree, wkb

    g = wkb.loads((ROOT / "out" / name / "water_iso.wkb").read_bytes())
    parts = list(getattr(g, "geoms", [g]))
    _TREE = (STRtree(parts), parts)
  return _TREE


def mask(name, qx, qy):
  """Water mask (L, 255 = water) for quadrant (qx, qy), or None if no water."""
  from shapely.geometry import box

  tree, parts = _load(name)
  b = box(qx * Q, qy * Q, (qx + 1) * Q, (qy + 1) * Q)
  hits = [parts[i] for i in tree.query(b)]
  if not hits:
    return None
  m = Image.new("L", (Q, Q))
  d = ImageDraw.Draw(m)
  for h in hits:
    c = h.intersection(b)
    for p in getattr(c, "geoms", [c]):
      if p.is_empty or p.geom_type != "Polygon":
        continue
      d.polygon([(x - qx * Q, y - qy * Q) for x, y in p.exterior.coords], fill=255)
      for hole in p.interiors:
        d.polygon([(x - qx * Q, y - qy * Q) for x, y in hole.coords], fill=0)
  return m if m.getbbox() else None


def texture(qx, qy):
  """Seamless water texture for a quadrant: a tileable 2048px pattern at global coords."""
  global _TEX
  if _TEX is None:
    rng = np.random.default_rng(1405)
    t = np.tile(BASE, (2048, 2048, 1))
    for _ in range(9000):
      x, y, w = rng.integers(0, 2048), rng.integers(0, 2048), rng.integers(6, 26)
      xs = np.arange(x, x + w) % 2048
      t[y % 2048, xs] += rng.uniform(10, 22)  # light ripple
      t[(y + 1) % 2048, xs] += rng.uniform(4, 10)
    _TEX = np.clip(t, 0, 255).astype(np.uint8)
  ox, oy = (qx * Q) % 2048, (qy * Q) % 2048
  return Image.fromarray(_TEX[oy : oy + Q, ox : ox + Q])


if __name__ == "__main__":
  if sys.argv[1] == "build":
    build(sys.argv[2])
