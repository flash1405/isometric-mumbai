"""Write config/train/<area>.json for the cities used to teach massing -> real 3D.

Each training area is rendered twice: the hybrid way (Google top-down imagery +
GlobalBuildingAtlas blocks, exactly what Mumbai gets) and the real Google 3D way.
Areas are flat on purpose: the hybrid scene assumes level ground.

  python3 scripts/make_train_configs.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AREAS = [
  ("london-city", 51.5136, -0.0890),
  ("london-westminster", 51.4995, -0.1300),
  ("london-kingscross", 51.5250, -0.1250),
  ("sg-chinatown", 1.2810, 103.8460),
  ("sg-littleindia", 1.3050, 103.8530),
  ("sg-toapayoh", 1.3343, 103.8500),
  ("hk-mongkok", 22.3193, 114.1694),
  ("hk-shamshuipo", 22.3300, 114.1600),
  ("hk-yaumatei", 22.3100, 114.1700),
  ("bkk-silom", 13.7270, 100.5290),
  ("bkk-yaowarat", 13.7400, 100.5100),
  ("kl-chowkit", 3.1600, 101.6980),
  ("kl-bukitbintang", 3.1470, 101.7130),
]

base = json.loads((ROOT / "config" / "fort.json").read_text())
out = ROOT / "config" / "train"
out.mkdir(exist_ok=True)
for name, lat, lon in AREAS:
  path = out / f"{name}.json"
  prev = json.loads(path.read_text()) if path.exists() else {}
  c = {k: v for k, v in base.items() if k not in ("description", "landmarks")}
  c.update(
    name=f"train/{name}",
    lat=lat,
    lon=lon,
    target_height=prev.get("target_height", 0),  # measured per area by pipeline/align.py
    cols=14,
    rows=11,
    ortho={"meters_per_pixel": 0.3, "quadrants": 21},
    heights_from="dsm",  # real 3D exists here: use true heights so the model only learns detail
    buildings={"height_scale": 1.0, "min_height": 3.5},
  )
  path.write_text(json.dumps(c, indent=2))
print(f"wrote {len(AREAS)} configs to {out}")
