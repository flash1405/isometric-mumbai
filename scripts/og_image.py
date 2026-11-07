"""Link-preview image (1200x630) from the exported map: a crop of South Mumbai + title.

  uv run python scripts/og_image.py mumbai      -> site/public/og-image.jpg
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site" / "public"


def main():
  meta = json.loads((SITE / "meta.json").read_text())
  places = json.loads((SITE / "places.json").read_text())
  focus = next((p for p in places if p["name"] == "Marine Drive"), places[0])
  z = meta["maxZoom"] - 3  # each tile covers 8 quadrants
  scale = 2 ** (meta["maxZoom"] - z)
  cx, cy = focus["col"] / scale, focus["row"] / scale
  W, H = 1200, 630
  canvas = Image.new("RGB", (W, H), (13, 18, 32))
  x0, y0 = cx - W / 2, cy - H / 2
  for ty in range(int(y0 // 512), int((y0 + H) // 512) + 1):
    for tx in range(int(x0 // 512), int((x0 + W) // 512) + 1):
      f = SITE / "tiles" / str(z) / str(tx) / f"{ty}.webp"
      if f.exists():
        canvas.paste(Image.open(f).convert("RGB"), (int(tx * 512 - x0), int(ty * 512 - y0)))
  # darken the left side for the title
  shade = Image.new("L", (W, H))
  ImageDraw.Draw(shade).rectangle([0, 0, W, H], fill=0)
  grad = Image.linear_gradient("L").rotate(-90).resize((W, H))  # strongest on the left, behind the title
  canvas = Image.composite(Image.new("RGB", (W, H), (13, 18, 32)), canvas, grad.point(lambda v: int(v * 0.75)))
  d = ImageDraw.Draw(canvas)
  def font(size):
    for f in ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
      if Path(f).exists():
        return ImageFont.truetype(f, size)
    return ImageFont.load_default()
  d.text((64, 230), "isometric", font=font(84), fill=(238, 241, 247))
  d.text((64, 318), "mumbai", font=font(84), fill=(255, 209, 102))
  d.text((68, 425), "the whole city in pixel art", font=font(30), fill=(200, 206, 220))
  canvas.save(SITE / "og-image.jpg", quality=88)
  print("->", SITE / "og-image.jpg")


if __name__ == "__main__":
  main()
