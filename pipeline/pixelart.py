"""Pixel-art finishing pass (same idea as isometric-nyc's unfake post-process).

  1. one shared palette for the whole city (k-means on a sample of generated tiles)
  2. map every pixel to the palette
  3. snap to a `px`-pixel grid: each block takes its dominant palette colour
  4. drop isolated single blocks (majority of the 3x3 neighbourhood), upscale with nearest

Quadrant edges sit on the grid (512 is a multiple of px), so tiles stay seamless.

  uv run python pipeline/pixelart.py palette mumbai [--colors 128]   -> out/mumbai/palette.npy
  uv run python pipeline/pixelart.py demo <in.png> <out.png> <palette.npy>
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def build_palette(images, colors=128, seed=0):
  rng = np.random.default_rng(seed)
  per = max(500, 150000 // max(1, len(images)))
  px = np.concatenate([np.asarray(im.convert("RGB")).reshape(-1, 3)[rng.integers(0, im.width * im.height, per)]
                       for im in images]).astype(np.float32)
  # k-means++ init + Lloyd iterations
  c = px[rng.integers(0, len(px), 1)]
  for _ in range(colors - 1):
    d = ((px[:, None] - c[None]) ** 2).sum(-1).min(1)
    c = np.vstack([c, px[rng.choice(len(px), p=d / d.sum())]])
  for _ in range(20):
    lab = ((px[:, None] - c[None]) ** 2).sum(-1).argmin(1)
    for k in range(colors):
      m = lab == k
      if m.any():
        c[k] = px[m].mean(0)
  return np.round(c).astype(np.uint8)


def nearest(a, pal):
  """Palette index per pixel (chunked to bound memory)."""
  flat = a.reshape(-1, 3).astype(np.int32)
  p = pal.astype(np.int32)
  out = np.empty(len(flat), np.int32)
  for i in range(0, len(flat), 65536):
    out[i : i + 65536] = ((flat[i : i + 65536, None] - p[None]) ** 2).sum(-1).argmin(1)
  return out.reshape(a.shape[:2])


def pixelate(img, pal, px=2, cleanup=True):
  a = np.asarray(img.convert("RGB"))
  h, w = a.shape[:2]
  idx = nearest(a, pal)
  # dominant colour per px*px block
  blocks = idx[: h // px * px, : w // px * px].reshape(h // px, px, w // px, px).transpose(0, 2, 1, 3).reshape(h // px, w // px, px * px)
  n = len(pal)
  if px > 1:  # dominant index per block: count each candidate's matches within the block
    match = (blocks[..., :, None] == blocks[..., None, :]).sum(-1)
    small = np.take_along_axis(blocks, match.argmax(-1)[..., None], -1)[..., 0]
  else:
    small = blocks[:, :, 0]
  if cleanup:  # isolated block -> most common neighbour
    pad = np.pad(small, 1, mode="edge")
    nb = np.stack([pad[1 + dy : 1 + dy + small.shape[0], 1 + dx : 1 + dx + small.shape[1]]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx], -1)
    same = (nb == small[..., None]).sum(-1)
    lonely = same == 0
    if lonely.any():
      cand = nb[lonely]
      m = (cand[:, :, None] == cand[:, None, :]).sum(-1)
      small[lonely] = np.take_along_axis(cand, m.argmax(-1)[:, None], 1)[:, 0]
  out = pal[small].repeat(px, 0).repeat(px, 1)
  res = a.copy()
  res[: out.shape[0], : out.shape[1]] = out
  return Image.fromarray(res)


if __name__ == "__main__":
  cmd = sys.argv[1]
  if cmd == "palette":
    name = sys.argv[2]
    colors = int(sys.argv[sys.argv.index("--colors") + 1]) if "--colors" in sys.argv else 128
    files = sorted((ROOT / "out" / name / "quads").glob("*.png"))
    rng = np.random.default_rng(0)
    pick = [files[i] for i in rng.choice(len(files), min(200, len(files)), replace=False)]
    pal = build_palette([Image.open(f) for f in pick], colors)
    np.save(ROOT / "out" / name / "palette.npy", pal)
    print(f"palette of {len(pal)} colours from {len(pick)} tiles")
  elif cmd == "demo":
    src, dst, pal = sys.argv[2], sys.argv[3], np.load(sys.argv[4])
    pixelate(Image.open(src), pal).save(dst)
