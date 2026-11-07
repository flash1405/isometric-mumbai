"""Assemble the fine-tuning dataset: hybrid render (massing + imagery) -> real-3D pixel art.

Each training tile yields samples in the same infill format generate.py uses, so the
fine-tuned model drops into the seam-free schedule unchanged:
  full   whole tile is hybrid render inside a red border           (phase 1)
  vstrip 256px of target art left/right, hybrid strip in the middle (phase 2)
  hstrip 256px of target art above/below                           (phase 3)
  centre target art all round, hybrid square in the middle         (phase 4)

Layout (ai-toolkit style):  dataset/images/<id>.png + <id>.txt,  dataset/control/<id>.png

  uv run python pipeline/build_dataset.py [--variants 2] [--holdout london-city]
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
T, H = 1024, 256
PROMPT = (
  "Fill in the outlined section with the missing pixels corresponding to the "
  "<isometric nyc pixel art> style, removing the border and exactly following the "
  "shape/style/structure of the surrounding image (if present)."
)
BOXES = {
  "full": (0, 0, T, T),
  "vstrip": (H, 0, H + 512, T),
  "hstrip": (0, H, T, H + 512),
  "centre": (H, H, H + 512, H + 512),
}


def control(target, hybrid, box):
  img = target.copy()
  x0, y0, x1, y1 = box
  img.paste(hybrid.crop(box), (x0, y0))
  d = ImageDraw.Draw(img)
  for i in range(2):
    d.rectangle([x0 + i, y0 + i, x1 - 1 - i, y1 - 1 - i], outline=(255, 0, 0))
  return img


def has_real_3d(area_dir):
  """Google's mesh is flat in some cities (Mumbai, Bangkok): their DSM heights collapse.
  Keep an area only if its true (DSM) building heights show real relief."""
  import json

  f = area_dir / "buildings.json"
  if not f.exists():
    return False
  hs = sorted(b["h"] for b in json.loads(f.read_text()))
  return len(hs) > 50 and hs[int(len(hs) * 0.9)] >= 12.0


def usable(img):
  """Skip tiles that are mostly empty (outside coverage) or mostly water."""
  a = np.asarray(img.resize((128, 128)).convert("RGB")).astype(np.float32)
  dark = (a.sum(axis=2) < 30).mean()
  flat = a.std(axis=(0, 1)).mean()
  return dark < 0.05 and flat > 18


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--variants", type=int, default=2, help="infill variants per tile")
  ap.add_argument("--holdout", default="london-city", help="area kept out for evaluation")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()
  rng = random.Random(args.seed)

  out = ROOT / "dataset"
  if out.exists():
    shutil.rmtree(out)
  (out / "images").mkdir(parents=True)
  (out / "control").mkdir()

  n = skipped = 0
  for area in sorted((ROOT / "out" / "train").glob("*/targets")):
    name = area.parent.name
    if name == args.holdout:
      continue
    if not has_real_3d(area.parent):
      print(f"skip {name}: no real 3D in Google's mesh")
      continue
    for tgt_path in sorted(area.glob("tile_*.png")):
      hyb_path = area.parent / "renders" / tgt_path.name
      if not hyb_path.exists():
        continue
      target = Image.open(tgt_path).convert("RGB").resize((T, T))
      hybrid = Image.open(hyb_path).convert("RGB").resize((T, T))
      if not (usable(hybrid) and usable(target)):
        skipped += 1
        continue
      kinds = ["full"] + rng.sample(["vstrip", "hstrip", "centre"], max(0, args.variants - 1))
      for k in kinds:
        sid = f"{name}_{tgt_path.stem}_{k}"
        target.save(out / "images" / f"{sid}.png")
        (out / "images" / f"{sid}.txt").write_text(PROMPT)
        control(target, hybrid, BOXES[k]).save(out / "control" / f"{sid}.png")
        n += 1
  print(f"{n} samples ({skipped} tiles skipped as empty/water), holdout={args.holdout} -> {out}")


if __name__ == "__main__":
  main()
