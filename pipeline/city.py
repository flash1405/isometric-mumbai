"""City-scale planning for the full map (one global grid, chunked work).

Frame: exact WGS84 tangent plane (ENU metres) at the city centre, the same plane the
Google top-down camera sees, so building footprints and imagery agree everywhere.
Grid:  the isometric image is cols x rows quadrants of 512px, centred on the city centre
       (same convention as render/hybrid.js placeCamera). cols, rows = 2 (mod 3).
Work:  - ortho tiles  : north-up top-down Google tiles (1024px @ 0.3 m), rendered once
       - chunks       : 12x12-quadrant blocks; each gets an ortho mosaic + buildings and
                        renders its own iso tiles into the global store
       - active quads : quadrants touching the city boundary (+ buffer); the rest is sea

  uv run python pipeline/city.py plan mumbai            -> out/mumbai/plan.json, active.png
  uv run python pipeline/city.py region mumbai <name>   -> print sub-region stats
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OVERTURE = "s3://overturemaps-us-west-2/release/2026-10-21.0"
Q = 512  # quadrant px
OT = 1024  # ortho tile px
CHUNK = 12  # quadrants per chunk side (keeps each chunk's ortho mosaic ~10k px)

# --- exact WGS84 lat/lon -> ENU (tangent plane at lat0, lon0) ------------------------
A, F = 6378137.0, 1 / 298.257223563
E2 = F * (2 - F)


def ecef(lat, lon, h=0.0):
  p, l = np.radians(lat), np.radians(lon)
  n = A / np.sqrt(1 - E2 * np.sin(p) ** 2)
  return (n + h) * np.cos(p) * np.cos(l), (n + h) * np.cos(p) * np.sin(l), (n * (1 - E2) + h) * np.sin(p)


def enu_fn(lat0, lon0):
  x0, y0, z0 = ecef(lat0, lon0)
  p, l = math.radians(lat0), math.radians(lon0)
  e = np.array([-math.sin(l), math.cos(l), 0.0])
  n = np.array([-math.sin(p) * math.cos(l), -math.sin(p) * math.sin(l), math.cos(p)])

  def f(lat, lon):
    x, y, z = ecef(np.asarray(lat, float), np.asarray(lon, float))
    d = np.stack([x - x0, y - y0, z - z0], axis=-1)
    return d @ e, d @ n

  return f


def inv_enu_fn(lat0, lon0):
  """Approximate inverse (good to cm over tens of km after 2 Newton steps)."""
  fwd = enu_fn(lat0, lon0)
  p = math.radians(lat0)
  ky = 111132.92 - 559.82 * math.cos(2 * p)
  kx = 111412.84 * math.cos(p)

  def f(x, y):
    lat, lon = lat0 + np.asarray(y) / ky, lon0 + np.asarray(x) / kx
    for _ in range(3):
      ex, ny = fwd(lat, lon)
      lat, lon = lat + (np.asarray(y) - ny) / ky, lon + (np.asarray(x) - ex) / kx
    return lat, lon

  return f


def basis(cfg):
  az, el = math.radians(cfg["camera_azimuth_degrees"]), math.radians(cfg["camera_elevation_degrees"])
  return (math.cos(az), -math.sin(az)), (math.sin(az) * math.sin(-el), math.cos(az) * math.sin(-el))


def boundary(cfg):
  """City boundary polygons (lon, lat rings) from Overture division areas."""
  import duckdb
  from shapely.geometry import Point
  from shapely import wkb
  from shapely.ops import unary_union

  con = duckdb.connect()
  con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2';")
  names = ",".join(f"'{n}'" for n in cfg["boundary"]["names"])
  rows = con.sql(
    f"""SELECT ST_AsWKB(geometry) FROM read_parquet('{OVERTURE}/theme=divisions/type=division_area/*', hive_partitioning=1)
        WHERE names.primary IN ({names}) AND subtype = '{cfg['boundary']['subtype']}'
          AND bbox.xmin > {cfg['lon'] - 1} AND bbox.xmax < {cfg['lon'] + 1}"""
  ).fetchall()
  return unary_union([wkb.loads(bytes(r[0])) for r in rows])


def plan(name):
  cfg_path = ROOT / "config" / f"{name}.json"
  cfg = json.loads(cfg_path.read_text())
  out = ROOT / "out" / name
  out.mkdir(parents=True, exist_ok=True)
  mpp, ompp = cfg["meters_per_pixel"], cfg["ortho"]["meters_per_pixel"]
  enu = enu_fn(cfg["lat"], cfg["lon"])
  (rx, ry), (ux, uy) = basis(cfg)

  geom = boundary(cfg)
  buf_m = cfg["boundary"].get("buffer_m", 300)
  polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
  # rings in ENU and in iso pixels (relative to centre)
  rings_iso = []
  for poly in polys:
    lon, lat = np.array(poly.exterior.coords).T
    x, y = enu(lat, lon)
    rings_iso.append(np.stack([(x * rx + y * ry) / mpp, -(x * ux + y * uy) / mpp], axis=1))
  allp = np.concatenate(rings_iso)
  pad = buf_m / mpp + 2 * Q
  half_w = max(abs(allp[:, 0].min()), abs(allp[:, 0].max())) + pad
  half_h = max(abs(allp[:, 1].min()), abs(allp[:, 1].max())) + pad
  cols = math.ceil(2 * half_w / Q)
  rows = math.ceil(2 * half_h / Q)
  cols += (2 - cols % 3) % 3  # 2 (mod 3) for the generation schedule
  rows += (2 - rows % 3) % 3
  cols += (cols // 2 % 2)  # keep an even number of render tiles across the centre
  while cols % 3 != 2:
    cols += 1
  W, H = cols * Q, rows * Q

  # active quadrants: rasterise boundary at 1px = 1 quadrant, dilate by the buffer
  m = Image.new("L", (cols * 4, rows * 4))
  d = ImageDraw.Draw(m)
  for r in rings_iso:
    d.polygon([((px + W / 2) / Q * 4, (py + H / 2) / Q * 4) for px, py in r], fill=255)
  grow = int(math.ceil(buf_m / mpp / Q * 4)) * 2 + 1
  m = m.filter(ImageFilter.MaxFilter(grow if grow % 2 else grow + 1))
  active = np.asarray(m.resize((cols, rows), Image.Resampling.BOX)) > 0
  Image.fromarray((active * 255).astype(np.uint8)).save(out / "active.png")

  # ortho tiles: global north-up grid centred on the city centre; mark those under
  # active quadrants' ground footprints (+ margin for tall buildings' roofs)
  margin = cfg["ortho"].get("margin_m", 300)
  det = rx * uy - ry * ux

  def iso_to_enu(px, py):  # pixel (relative to centre) on the ground plane -> ENU
    a, b = px * mpp, -py * mpp
    return (a * uy - b * ry) / det, (b * rx - a * ux) / det

  qy, qx = np.nonzero(active)
  ext = []
  for cx, cy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    ext.append(iso_to_enu((qx + cx) * Q - W / 2, (qy + cy) * Q - H / 2))
  xs = np.concatenate([e[0] for e in ext])
  ys = np.concatenate([e[1] for e in ext])
  otm = OT * ompp
  ox_n = 2 * math.ceil((max(abs(xs.min()), abs(xs.max())) + margin) / otm)
  oy_n = 2 * math.ceil((max(abs(ys.min()), abs(ys.max())) + margin) / otm)
  omask = np.zeros((oy_n, ox_n), bool)
  for i in range(len(qx)):
    x4 = [e[0][i] for e in ext]
    y4 = [e[1][i] for e in ext]
    tx0 = int(math.floor((min(x4) - margin) / otm + ox_n / 2))
    tx1 = int(math.floor((max(x4) + margin) / otm + ox_n / 2))
    ty0 = int(math.floor((oy_n / 2) - (max(y4) + margin) / otm))
    ty1 = int(math.floor((oy_n / 2) - (min(y4) - margin) / otm))
    omask[max(0, ty0) : ty1 + 1, max(0, tx0) : tx1 + 1] = True

  plan = {
    "name": name, "cols": cols, "rows": rows, "width": W, "height": H,
    "active_quads": int(active.sum()), "total_quads": cols * rows,
    "ortho_grid": [ox_n, oy_n], "ortho_tiles": int(omask.sum()),
    "chunks": [],
  }
  for cy in range(math.ceil(rows / CHUNK)):
    for cx in range(math.ceil(cols / CHUNK)):
      blk = active[cy * CHUNK : (cy + 1) * CHUNK, cx * CHUNK : (cx + 1) * CHUNK]
      if blk.any():
        plan["chunks"].append({"id": f"c{cx:02d}_{cy:02d}", "cx": cx, "cy": cy, "active": int(blk.sum())})
  np.save(out / "ortho_mask.npy", omask)
  (out / "plan.json").write_text(json.dumps(plan, indent=1))
  area_km2 = plan["active_quads"] * (Q * mpp) * (Q * mpp / math.sin(math.radians(-cfg["camera_elevation_degrees"]))) / 1e6
  print(json.dumps({k: v for k, v in plan.items() if k != "chunks"}))
  print(f"{len(plan['chunks'])} chunks; active ground ~{area_km2:.0f} km2; image {W}x{H}px")
  return plan


# --- chunks -----------------------------------------------------------------------------

def load(name):
  cfg = json.loads((ROOT / "config" / f"{name}.json").read_text())
  pl = json.loads((ROOT / "out" / name / "plan.json").read_text())
  active = np.asarray(Image.open(ROOT / "out" / name / "active.png")) > 0
  extra = ROOT / "out" / name / "extra_quads.json"  # e.g. bridges over the sea (Sea Link)
  if extra.exists():
    active = active.copy()
    for qx, qy in json.loads(extra.read_text()):
      active[qy, qx] = True
  return cfg, pl, active


def iso_to_enu_fn(cfg):
  (rx, ry), (ux, uy) = basis(cfg)
  det, mpp = rx * uy - ry * ux, cfg["meters_per_pixel"]

  def f(px, py):
    a, b = np.asarray(px) * mpp, -np.asarray(py) * mpp
    return (a * uy - b * ry) / det, (b * rx - a * ux) / det

  return f


def region_quads(cfg, pl, active, region):
  """Active quadrants whose ground centre lies in the named lat/lon box (None = all)."""
  qy, qx = np.nonzero(active)
  if not region:
    return set(zip(qx.tolist(), qy.tolist()))
  if region == "extra" or region.startswith("file:"):  # a list of quadrants, e.g. ones to repaint
    f = "extra_quads" if region == "extra" else region[5:]
    return {tuple(q) for q in json.loads((ROOT / "out" / cfg["name"] / f"{f}.json").read_text())}
  r = cfg["regions"][region]
  x, y = iso_to_enu_fn(cfg)((qx + 0.5) * Q - pl["width"] / 2, (qy + 0.5) * Q - pl["height"] / 2)
  lat, lon = inv_enu_fn(cfg["lat"], cfg["lon"])(x, y)
  ok = (lat >= r["lat"][0]) & (lat <= r["lat"][1]) & (lon >= r["lon"][0]) & (lon <= r["lon"][1])
  return set(zip(qx[ok].tolist(), qy[ok].tolist()))


def chunk_tiles(chunk, quads):
  """Render tiles (2x2 quadrants, global indices) of a chunk that contain a wanted quadrant."""
  qx0, qy0 = chunk["cx"] * CHUNK, chunk["cy"] * CHUNK
  out = []
  for ty in range(qy0 // 2, (qy0 + CHUNK) // 2):
    for tx in range(qx0 // 2, (qx0 + CHUNK) // 2):
      if any((2 * tx + i, 2 * ty + j) in quads for i in (0, 1) for j in (0, 1)):
        out.append([tx, ty])
  return out


def chunk_ortho_range(cfg, pl, tiles):
  """Ortho tile index range covering the ground under these render tiles (+ margin)."""
  f = iso_to_enu_fn(cfg)
  margin = cfg["ortho"].get("margin_m", 300)
  otm = OT * cfg["ortho"]["meters_per_pixel"]
  nx, ny = pl["ortho_grid"]
  t = np.array(tiles)
  xs, ys = [], []
  for cx, cy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    x, y = f((t[:, 0] + cx) * 2 * Q - pl["width"] / 2, (t[:, 1] + cy) * 2 * Q - pl["height"] / 2)
    xs.append(x)
    ys.append(y)
  xs, ys = np.concatenate(xs), np.concatenate(ys)
  tx0 = max(0, int(math.floor((xs.min() - margin) / otm + nx / 2)))
  tx1 = min(nx - 1, int(math.floor((xs.max() + margin) / otm + nx / 2)))
  ty0 = max(0, int(math.floor(ny / 2 - (ys.max() + margin) / otm)))
  ty1 = min(ny - 1, int(math.floor(ny / 2 - (ys.min() - margin) / otm)))
  return tx0, tx1, ty0, ty1


def with_neighbour_cells(quads, active):
  """Region quads grown to whole 3x3 schedule cells plus their right/lower/diagonal
  neighbour cells: everything generation touches when finishing the region."""
  rows, cols = active.shape
  cells = {(qx // 3, qy // 3) for qx, qy in quads}
  cells |= {(i + a, j + b) for i, j in cells for a in (0, 1) for b in (0, 1)}
  out = set()
  for i, j in cells:
    for a in range(3):
      for b in range(3):
        qx, qy = 3 * i + a, 3 * j + b
        if qx < cols and qy < rows and active[qy, qx]:
          out.add((qx, qy))
  return out


def chunks_for(name, region=None):
  cfg, pl, active = load(name)
  quads = region_quads(cfg, pl, active, region)
  if region:
    quads = with_neighbour_cells(quads, active)
  out = []
  for c in pl["chunks"]:
    tiles = chunk_tiles(c, quads)
    if tiles:
      out.append((c, tiles))
  return cfg, pl, quads, out


def ortho_todo(name, region=None):
  """Write the list of ortho tiles the region's chunks need (+ the ortho render config)."""
  cfg, pl, _, chs = chunks_for(name, region)
  need = set()
  for c, tiles in chs:
    tx0, tx1, ty0, ty1 = chunk_ortho_range(cfg, pl, tiles)
    need |= {(x, y) for x in range(tx0, tx1 + 1) for y in range(ty0, ty1 + 1)}
  out = ROOT / "out" / name
  (out / "ortho_todo.json").write_text(json.dumps(sorted([list(t) for t in need], key=lambda t: (t[1], t[0]))))
  oc = dict(cfg, ortho_grid=pl["ortho_grid"])
  (out / "ortho_config.json").write_text(json.dumps(oc, indent=1))
  print(f"{len(chs)} chunks, {len(need)} ortho tiles -> {out / 'ortho_todo.json'}")


def chunk_prep(name, cid, region=None):
  """Stitch the chunk's ortho mosaic, fetch its buildings, write its render config."""
  import duckdb
  from shapely.geometry import Point

  cfg, pl, quads, chs = chunks_for(name, region)
  c, tiles = next((c, t) for c, t in chs if c["id"] == cid)
  out = ROOT / "out" / name
  cdir = out / "chunks" / cid
  cdir.mkdir(parents=True, exist_ok=True)
  ompp = cfg["ortho"]["meters_per_pixel"]
  otm = OT * ompp
  nx, ny = pl["ortho_grid"]
  tx0, tx1, ty0, ty1 = chunk_ortho_range(cfg, pl, tiles)
  mos = Image.new("RGB", ((tx1 - tx0 + 1) * OT, (ty1 - ty0 + 1) * OT))
  missing = 0
  for ty in range(ty0, ty1 + 1):
    for tx in range(tx0, tx1 + 1):
      f = out / "ortho_tiles" / f"tile_{tx}_{ty}.png"
      if f.exists():
        mos.paste(Image.open(f).convert("RGB"), ((tx - tx0) * OT, (ty - ty0) * OT))
      else:
        missing += 1
  rect = {"x0": (tx0 - nx / 2) * otm, "y1": (ny / 2 - ty0) * otm, "w": mos.width * ompp, "h": mos.height * ompp}
  apply_masks(cfg, mos, rect, ompp)
  mos.save(cdir / "ortho.jpg", quality=92)

  # buildings (GlobalBuildingAtlas) inside the rect, in the city ENU frame
  inv = inv_enu_fn(cfg["lat"], cfg["lon"])
  lats, lons = inv(np.array([rect["x0"], rect["x0"] + rect["w"]]), np.array([rect["y1"] - rect["h"], rect["y1"]]))
  sys.path.insert(0, str(ROOT / "pipeline"))
  from prepare import GBA, gba_tile

  con = duckdb.connect()
  con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET geometry_always_xy = true;")
  local = ROOT / "out" / name / "gba.parquet"  # city-wide local copy (see `city.py gba`)
  src = str(local) if local.exists() else GBA.format(tile=gba_tile(cfg["lat"], cfg["lon"]))
  rows = con.sql(
    f"""SELECT height, ST_AsGeoJSON(geometry) FROM read_parquet('{src}')
        WHERE bbox.xmax >= {lons.min()} AND bbox.xmin <= {lons.max()} AND bbox.ymax >= {lats.min()} AND bbox.ymin <= {lats.max()}"""
  ).fetchall()
  enu = enu_fn(cfg["lat"], cfg["lon"])
  b = cfg.get("buildings", {})
  scale, hmin = b.get("height_scale", 1.0), b.get("min_height", 3.0)
  masks = mask_polys_enu(cfg)
  heights = None
  if (ROOT / "out" / name / "heights" / "ob25d.json").exists():  # merged OSM / Wikidata / 2.5D heights
    from heights import Heights
    heights = Heights(name, cfg)
  from shapely.geometry import Polygon as _P
  blds = []
  for height, gj in rows:
    g = json.loads(gj)
    for poly in g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]:
      rings = []
      for ring in poly:
        a = np.array(ring[:-1])
        x, y = enu(a[:, 1], a[:, 0])
        rings.append(np.round(np.stack([x, y], 1), 2).tolist())
      if len(rings[0]) < 3:
        continue
      cxm, cym = np.mean(rings[0], axis=0)
      if masks and any(m.contains(Point(cxm, cym)) for m in masks):
        continue
      if heights is not None and len(poly[0]) >= 4:
        h = heights.height(_P(poly[0]), height)
      else:
        h = max(hmin, (height if height and height > 0 else hmin) * scale)
      blds.append({"h": round(h, 1), "rings": rings})
  # listed towers with no footprint at all (GlobalBuildingAtlas misses some new ones): add one
  tw_f = ROOT / "out" / name / "towers.json"
  if tw_f.exists():
    from shapely import STRtree as _T
    polys = [_P(b["rings"][0]) for b in blds]
    tree = _T(polys) if polys else None
    added = 0
    for t in json.loads(tw_f.read_text()):
      x, y = enu(t["lat"], t["lon"])
      x, y = float(x), float(y)
      if not (rect["x0"] <= x <= rect["x0"] + rect["w"] and rect["y1"] - rect["h"] <= y <= rect["y1"]):
        continue
      pt = Point(x, y)
      # exactly one footprint per listed tower: the one it stands in, else the nearest within 15 m
      cand = [(polys[i].distance(pt), i) for i in tree.query(pt.buffer(15)) if polys[i].area > 120] if tree is not None else []
      if cand:
        d_, i = min(cand)
        if polys[i].area <= 2200:  # a tower-sized footprint: raise it
          blds[i]["h"] = max(blds[i]["h"], t["height"])
          continue
        # a whole complex / podium outline: keep it, put the tower on it at its own position
      s_ = 16.0  # 32 x 32 m tower footprint
      blds.append({"h": t["height"], "rings": [[[x - s_, y - s_], [x + s_, y - s_], [x + s_, y + s_], [x - s_, y + s_]]]})
      added += 1
    if added:
      print(f"  + {added} tower footprints missing from the atlas")
  (cdir / "buildings.json").write_text(json.dumps(blds, separators=(",", ":")))
  ccfg = dict(cfg, cols=pl["cols"], rows=pl["rows"], ortho_rect=rect)
  (cdir / "config.json").write_text(json.dumps(ccfg))
  (cdir / "tiles.json").write_text(json.dumps(tiles))
  print(f"{cid}: {len(tiles)} render tiles, ortho {mos.width}x{mos.height}px ({missing} missing), {len(blds)} buildings"
        + (f", heights {heights.stats}" if heights else ""))


# --- sensitive-site masks -------------------------------------------------------------

def military_polys(cfg):
  """Military land-use polygons from Overture (OSM landuse=military etc.), cached per city."""
  cache = ROOT / "out" / cfg["name"] / "military.json"
  if cache.exists():
    return json.loads(cache.read_text())
  import duckdb

  b = cfg["boundary"]
  con = duckdb.connect()
  con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2'; SET geometry_always_xy = true;")
  rows = con.sql(
    f"""SELECT names.primary, ST_AsGeoJSON(geometry) FROM read_parquet('{OVERTURE}/theme=base/type=land_use/*', hive_partitioning=1)
        WHERE bbox.xmin BETWEEN {cfg['lon'] - 0.4} AND {cfg['lon'] + 0.4} AND bbox.ymin BETWEEN {cfg['lat'] - 0.4} AND {cfg['lat'] + 0.4}
          AND subtype = 'military'"""
  ).fetchall()
  out = []
  for name, gj in rows:
    g = json.loads(gj)
    for poly in g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]] if g["type"] == "Polygon" else []:
      out.append({"name": name or "military", "polygon": [[lat, lon] for lon, lat in poly[0]]})
  cache.write_text(json.dumps(out))
  return out


def mask_polys_enu(cfg):
  """Sensitive sites: Overture military areas (+ manual masks from the config), buffered."""
  from shapely.geometry import Polygon

  enu = enu_fn(cfg["lat"], cfg["lon"])
  masks = list(cfg.get("masks", []))
  if cfg.get("mask_military", True) and cfg.get("boundary"):
    masks += military_polys(cfg)
  out = []
  for m in masks:
    a = np.array(m["polygon"])  # [[lat, lon], ...]
    if len(a) < 3:
      continue
    x, y = enu(a[:, 0], a[:, 1])
    p = Polygon(np.stack([x, y], 1)).buffer(cfg.get("mask_buffer_m", 25))
    if p.is_valid and not p.is_empty:
      out.append(p)
  return out


def apply_masks(cfg, img, rect, mpp, block_m=30, blur_m=12):
  """Hide sensitive sites in the ortho mosaic: coarse 30 m mosaic + blur inside the mask.
  Keeps land vs water (so the model doesn't paint land in the harbour) but no detail."""
  polys = mask_polys_enu(cfg)
  if not polys:
    return
  from PIL import ImageFilter

  m = Image.new("L", img.size)
  d = ImageDraw.Draw(m)
  hit = False
  for p in polys:
    for part in getattr(p, "geoms", [p]):
      pts = [((x - rect["x0"]) / mpp, (rect["y1"] - y) / mpp) for x, y in part.exterior.coords]
      xs, ys = [q[0] for q in pts], [q[1] for q in pts]
      if max(xs) < 0 or min(xs) > img.width or max(ys) < 0 or min(ys) > img.height:
        continue
      d.polygon(pts, fill=255)
      hit = True
  if not hit:
    return
  k = max(1, int(block_m / mpp))
  coarse = img.resize((max(1, img.width // k), max(1, img.height // k)), Image.Resampling.BOX)
  coarse = coarse.resize(img.size, Image.Resampling.NEAREST).filter(ImageFilter.GaussianBlur(blur_m / mpp))
  img.paste(coarse, (0, 0), m.filter(ImageFilter.GaussianBlur(3)))

if __name__ == "__main__":
  cmd, name = sys.argv[1], sys.argv[2]
  region = next((a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("--region=")), None)
  if cmd == "plan":
    plan(name)
  elif cmd == "ortho-todo":
    ortho_todo(name, region)
  elif cmd == "gba":  # download the city's GlobalBuildingAtlas rows once
    import duckdb
    sys.path.insert(0, str(ROOT / "pipeline"))
    from prepare import GBA, gba_tile
    cfg = json.loads((ROOT / "config" / f"{name}.json").read_text())
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    out = ROOT / "out" / name / "gba.parquet"
    con.sql(f"""COPY (SELECT * FROM read_parquet('{GBA.format(tile=gba_tile(cfg['lat'], cfg['lon']))}')
      WHERE bbox.xmin BETWEEN {cfg['lon'] - 0.25} AND {cfg['lon'] + 0.25} AND bbox.ymin BETWEEN {cfg['lat'] - 0.3} AND {cfg['lat'] + 0.3})
      TO '{out}' (FORMAT parquet)""")
    print("->", out, con.sql(f"SELECT count(*) FROM read_parquet('{out}')").fetchall())
  elif cmd == "chunks":
    for c, t in chunks_for(name, region)[3]:
      print(c["id"], len(t))
  elif cmd == "chunk":
    chunk_prep(name, sys.argv[3], next((a.split("=", 1)[1] for a in sys.argv[4:] if a.startswith("--region=")), None))
