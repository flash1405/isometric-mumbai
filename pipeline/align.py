"""Set target_height so the Google 3D camera aims at real ground level.

The hybrid scene puts the ground at z=0 under the area centre. The Google camera aims
at an ellipsoid height (target_height); if that differs from the true ground height the
Google render shifts vertically. The ortho pass's depth tiles measure the surface from a
camera 3000 m above target_height, so ground ellipsoid height at the centre is
  target_height_ortho + 3000 - distance_to_ground.

  uv run python pipeline/align.py train/london-city
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CAM_BACK = 3000.0  # render/main.js backs the camera off 3000 m along its axis


def main():
  name = sys.argv[1]
  cfg_path = ROOT / "config" / f"{name}.json"
  cfg = json.loads(cfg_path.read_text())
  d = ROOT / "out" / name
  meta = d / "ortho_meta.json"
  th_ortho = json.loads(meta.read_text())["target_height"]
  n = cfg["ortho"]["quadrants"]
  tiles = (n + 1) // 2
  c = tiles // 2  # tile containing the centre
  # centre pixel of the whole ortho canvas -> tile + offset
  size = n * cfg["quadrant_px"]
  cx = cy = size // 2
  tx, ty = cx // 1024, cy // 1024
  a = np.asarray(Image.open(d / "ortho_tiles" / f"depth_{tx}_{ty}.png").convert("RGB")).astype(np.int64)
  mm = ((a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]) / 1000.0
  px, py = cx - tx * 1024, cy - ty * 1024
  r = int(60 / cfg["ortho"]["meters_per_pixel"])  # 60 m window around the centre
  win = mm[max(0, py - r) : py + r, max(0, px - r) : px + r]
  win = win[(win > 1) & (win < 16000)]
  ground_dist = np.percentile(win, 98)  # farthest surface = ground
  hg = th_ortho + CAM_BACK - ground_dist
  print(f"{name}: ground at centre = {hg:.2f} m ellipsoid (ortho target {th_ortho}); was {cfg['target_height']}")
  cfg["target_height"] = round(float(hg), 2)
  cfg_path.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
  main()
