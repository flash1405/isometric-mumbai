"""Training set for the Mumbai style model from teacher pairs (out/<city>/teacher/).

Target = teacher's finished art. Controls use the same infill formats as generation
(full tile, vertical strip, horizontal strip, centre square): teacher art as context, our
render inside the red box, so the model learns both the style and seamless infill.

  uv run python pipeline/build_teacher_dataset.py mumbai [--variants 3] [--holdout 4]
"""

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

from build_dataset import BOXES, PROMPT, T, control

ROOT = Path(__file__).resolve().parent.parent


def colour_shift(img, rng, strength):
  """Random brightness / contrast / saturation / warmth change of the render only."""
  from PIL import ImageEnhance
  j = 0.12 * strength
  img = ImageEnhance.Brightness(img).enhance(1 + rng.uniform(-j, j))
  img = ImageEnhance.Contrast(img).enhance(1 + rng.uniform(-j, j))
  img = ImageEnhance.Color(img).enhance(1 + rng.uniform(-1.5 * j, 1.5 * j))
  r, g, b = img.split()
  w = rng.uniform(-j, j) * 40
  r = r.point(lambda v: max(0, min(255, int(v + w))))
  b = b.point(lambda v: max(0, min(255, int(v - w))))
  return Image.merge("RGB", (r, g, b))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("city")
  ap.add_argument("--variants", type=int, default=3)
  ap.add_argument("--holdout", type=int, default=4, help="pairs kept out for previews")
  ap.add_argument("--src", help="folder of <id>_in / <id>_out images (default out/<city>/teacher)")
  ap.add_argument("--out", help="dataset folder (default ./dataset)")
  ap.add_argument("--shift", type=int, default=0, help="extra full-tile copies with colour-shifted renders "
                  "(teaches the model not to copy Google's colour quirks; after Isometric SF)")
  ap.add_argument("--ext", default="png", help="png or jpg (jpg keeps big datasets small)")
  args = ap.parse_args()
  rng = random.Random(0)
  src = Path(args.src) if args.src else ROOT / "out" / args.city / "teacher"
  pairs = sorted(p for p in src.glob("*_out.*") if (src / p.name.replace("_out", "_in")).exists())
  rng.shuffle(pairs)
  hold, train = pairs[: args.holdout], pairs[args.holdout :]
  out = Path(args.out) if args.out else ROOT / "dataset"
  if out.exists():
    shutil.rmtree(out)
  (out / "images").mkdir(parents=True)
  (out / "control").mkdir()
  (out / "holdout").mkdir()
  n = 0
  for p in train:
    target = Image.open(p).convert("RGB").resize((T, T))
    render = Image.open(src / p.name.replace("_out", "_in")).convert("RGB").resize((T, T))
    kinds = ["full"] + rng.sample(["vstrip", "hstrip", "centre"], max(0, args.variants - 1))
    for k in kinds:
      sid = f"{p.stem.replace('_out', '')}_{k}"
      target.save(out / "images" / f"{sid}.{args.ext}", quality=95)
      (out / "images" / f"{sid}.txt").write_text(PROMPT)
      control(target, render, BOXES[k]).save(out / "control" / f"{sid}.{args.ext}", quality=95)
      n += 1
    for s_ in range(args.shift):
      sid = f"{p.stem.replace('_out', '')}_shift{s_}"
      target.save(out / "images" / f"{sid}.{args.ext}", quality=95)
      (out / "images" / f"{sid}.txt").write_text(PROMPT)
      control(target, colour_shift(render, rng, 1 + s_), BOXES["full"]).save(out / "control" / f"{sid}.{args.ext}", quality=95)
      n += 1
  for i, p in enumerate(hold):
    render = Image.open(src / p.name.replace("_out", "_in")).convert("RGB").resize((T, T))
    control(render, render, BOXES["full"]).save(out / "holdout" / f"holdout_{i}.png")
    Image.open(p).convert("RGB").save(out / "holdout" / f"holdout_{i}_teacher.png")
  print(f"{n} samples from {len(train)} pairs; {len(hold)} held out -> {out}")


if __name__ == "__main__":
  main()
