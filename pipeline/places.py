"""Build site/public/places.json: searchable names and map labels from Overture Maps.

  areas      neighbourhoods / localities (Overture divisions) -> fading map labels + search
  landmarks  notable places (stations, museums, monuments, parks, stadiums...) -> search

Only places that fall inside the exported image are kept. Coordinates are stored as
image pixels (col, row) so the site needs no projection maths for them.

  uv run python pipeline/places.py         (reads site/public/meta.json)
"""

import json
import math
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
OVERTURE = "s3://overturemaps-us-west-2/release/2026-10-21.0"
LANDMARK_CATS = {
  "historic_site", "monument", "museum", "art_gallery", "train_station", "stadium_arena",
  "beach", "park", "courthouse", "movie_theater", "theatre_venue", "event_venue",
  "hindu_place_of_worship", "christian_place_of_worship", "muslim_place_of_worship",
  "college_university", "hospital", "library", "embassy", "police_station",
}
# Famous places search must always find (Overture's POIs are noisy for Mumbai).
CURATED = ROOT / "config" / "landmarks_mumbai.json"


MAJOR = {n.lower() for n in [
  "Colaba", "Fort", "Churchgate", "Nariman Point", "Cuffe Parade", "Girgaon", "Malabar Hill", "Byculla", "Mazgaon",
  "Tardeo", "Worli", "Lower Parel", "Parel", "Prabhadevi", "Dadar", "Matunga", "Mahim", "Dharavi", "Sion", "Wadala",
  "Sewri", "Bandra", "Bandra West", "Bandra East", "Khar", "Khar West", "Santacruz", "Santacruz East", "Juhu", "Vile Parle",
  "Andheri", "Andheri East", "Andheri West", "Versova", "Jogeshwari", "Goregaon", "Malad", "Kandivali", "Borivali",
  "Dahisar", "Gorai", "Powai", "Ghatkopar", "Vikhroli", "Kanjurmarg", "Bhandup", "Mulund", "Kurla", "Chembur",
  "Trombay", "Mankhurd", "Govandi", "Bandra Kurla Complex", "Marine Lines", "Grant Road", "Kalbadevi", "Chakala",
  "Marol", "Saki Naka", "Aarey", "Madh Island", "Dharavi", "Nariman Point", "Marine Drive",
]}


def main():
  meta = json.loads((ROOT / "site" / "public" / "meta.json").read_text())
  g = meta["geo"]
  W, H, mpp = meta["width"], meta["height"], g["mpp"]
  (rx, ry), (ux, uy) = g["right"], g["up"]

  if g.get("frame") == "enu":  # city maps: exact tangent-plane frame (pipeline/city.py)
    import sys
    sys.path.insert(0, str(ROOT / "pipeline"))
    import city
    enu = city.enu_fn(g["lat0"], g["lon0"])
    inv = city.inv_enu_fn(g["lat0"], g["lon0"])
    to_xy = lambda lat, lon: tuple(float(v) for v in enu(lat, lon))
    to_ll = lambda x, y: tuple(float(v) for v in inv(x, y))
  else:
    to_xy = lambda lat, lon: ((lon - g["lon0"]) * g["kx"], (lat - g["lat0"]) * g["ky"])
    to_ll = lambda x, y: (g["lat0"] + y / g["ky"], g["lon0"] + x / g["kx"])

  def to_px(lat, lon):
    x, y = to_xy(lat, lon)
    return W / 2 + (x * rx + y * ry) / mpp, H / 2 - (x * ux + y * uy) / mpp

  # ground bbox covering the image: invert the 2x2 transform at the four corners
  det = rx * uy - ry * ux
  corners = []
  bx0, by0, bx1, by1 = meta.get("bounds", [0, 0, W, H])
  for px, py in [(bx0, by0), (bx1, by0), (bx0, by1), (bx1, by1)]:
    a, b = (px - W / 2) * mpp, (H / 2 - py) * mpp  # a = p.right, b = p.up
    x, y = (a * uy - b * ry) / det, (b * rx - a * ux) / det
    corners.append(to_ll(x, y))
  lat_min, lat_max = min(c[0] for c in corners), max(c[0] for c in corners)
  lon_min, lon_max = min(c[1] for c in corners), max(c[1] for c in corners)
  bbox = f"bbox.xmin BETWEEN {lon_min} AND {lon_max} AND bbox.ymin BETWEEN {lat_min} AND {lat_max}"

  con = duckdb.connect()
  con.sql("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial; SET s3_region='us-west-2';")
  out, seen = [], set()

  def add(name, lat, lon, cat, rank):
    if not name:
      return
    col, row = to_px(lat, lon)
    key = name.lower()
    if bx0 <= col < bx1 and by0 <= row < by1 and key not in seen:
      seen.add(key)
      out.append({"name": name, "cat": cat, "col": round(col), "row": round(row), "rank": rank})

  if CURATED.exists():
    for p in json.loads(CURATED.read_text()):
      add(p["name"], p["lat"], p["lon"], "landmark", 0)

  areas = con.sql(f"""
    SELECT names.primary, subtype, (bbox.ymin + bbox.ymax) / 2, (bbox.xmin + bbox.xmax) / 2
    FROM read_parquet('{OVERTURE}/theme=divisions/type=division/*', hive_partitioning=1)
    WHERE {bbox} AND subtype IN ('neighborhood', 'macrohood', 'microhood', 'locality')""").fetchall()
  order = {"locality": 0, "macrohood": 1, "neighborhood": 2, "microhood": 3}
  import re
  for name, sub, lat, lon in sorted(areas, key=lambda r: order.get(r[1], 9)):
    if not name:
      continue
    if name.lower() in MAJOR:
      rank = 0  # the names people actually know: shown even when zoomed right out
    elif re.search(r"\bward\b|colony$|^mumbai$|suburban|district", name, re.I):
      rank = 9  # administrative units: searchable, never drawn as map labels
    else:
      rank = min(order.get(sub, 9) + 1, 3)
    add(name, lat, lon, "area", rank)

  # "Malad West" + "Malad East" -> one "Malad" label for the zoomed-out map, between the two halves
  import re as _re
  halves = {}
  for p in out:
    if p["cat"] == "area":
      base = _re.sub(r"\s+(east|west)$", "", p["name"], flags=_re.I)
      if base != p["name"] and base.lower() in MAJOR:
        halves.setdefault(base, []).append(p)
  for base, ps in halves.items():
    if base.lower() not in seen:
      seen.add(base.lower())
      out.append({"name": base, "cat": "area", "rank": 0,
                  "col": round(sum(q["col"] for q in ps) / len(ps)), "row": round(sum(q["row"] for q in ps) / len(ps))})
    for q in ps:
      q["rank"] = max(q["rank"], 1)

  pois = con.sql(f"""
    SELECT names.primary, basic_category, ST_Y(geometry), ST_X(geometry), confidence
    FROM read_parquet('{OVERTURE}/theme=places/type=place/*', hive_partitioning=1)
    WHERE {bbox} AND confidence >= 0.9 AND basic_category IN ({",".join(f"'{c}'" for c in LANDMARK_CATS)})
    ORDER BY confidence DESC LIMIT 3000""").fetchall()
  for name, cat, lat, lon, conf in pois:
    add(name, lat, lon, "landmark", round(10 + (1 - conf) * 10, 2))

  out.sort(key=lambda p: (p["rank"] != 0, p["cat"] != "area", p["rank"]))
  path = ROOT / "site" / "public" / "places.json"
  path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
  n_area = sum(p["cat"] == "area" for p in out)
  print(f"{len(out)} places ({n_area} areas, {len(out) - n_area} landmarks) -> {path}")
  print("bbox", round(lat_min, 4), round(lon_min, 4), round(lat_max, 4), round(lon_max, 4))


if __name__ == "__main__":
  main()
