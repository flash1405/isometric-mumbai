"""Seam-free colour across the whole map (used by export_city.py).

Each quadrant was painted by a separate model job, so tone drifts a little from job to job:
in flat areas (forest, mangroves, maidans) that reads as a patchwork of squares. Two passes:

1. global: one RGB offset per quadrant, solved by least squares so that every quadrant's edge
   strips match its neighbours' (lightly pulled to 0 so the city keeps its overall colour), and
   applied as a smooth field interpolated between quadrant centres (no new edges inside a quad);
2. local "hem" (after Isometric SF): the remaining difference along each edge, profiled in
   16 px bins, is split between the two sides and faded out over HEM px.

  uv run python pipeline/harmonize.py mumbai quads_v2_clean    -> out/<city>/<dir>_harmony.npz
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
Q, K, BIN, HEM = 512, 6, 16, 40
CAP = 28  # hem: never shift colour by more than this at an edge (coastlines differ for real)
GCAP = 0.4  # global: log-gain limit per channel (x0.67 .. x1.49)
NB = Q // BIN


def strips(path):
  a = np.asarray(Image.open(path).convert("RGB"), np.float32)
  prof = lambda s: s.reshape(NB, BIN, -1, 3).mean((1, 2))  # noqa: E731  (NB, 3) along the edge
  return np.stack([
    prof(a[:, :K].reshape(Q, K, 3)), prof(a[:, -K:].reshape(Q, K, 3)),  # left, right (along y)
    prof(a[:K].transpose(1, 0, 2)), prof(a[-K:].transpose(1, 0, 2)),  # top, bottom (along x)
  ])


def build(qdir):
  qdir = Path(qdir)
  quads = sorted(tuple(map(int, p.stem.split("_"))) for p in qdir.glob("*.png"))
  with ProcessPoolExecutor(10) as ex:
    S = np.stack(list(ex.map(strips, [qdir / f"{x}_{y}.png" for x, y in quads], chunksize=64)))
  idx = {q: i for i, q in enumerate(quads)}
  pairs = []  # (i, j, axis): j is right of i (axis 0) or below i (axis 1)
  for (x, y), i in idx.items():
    if (x + 1, y) in idx:
      pairs.append((i, idx[(x + 1, y)], 0))
    if (x, y + 1) in idx:
      pairs.append((i, idx[(x, y + 1)], 1))
  from scipy.sparse import coo_matrix
  from scipy.sparse.linalg import lsqr
  n, m = len(quads), len(pairs)
  rows = np.r_[np.repeat(np.arange(m), 2), m + np.arange(n)]
  cols = np.r_[np.array([[i, j] for i, j, _ in pairs]).ravel(), np.arange(n)]
  vals = np.r_[np.tile([1.0, -1.0], m), np.full(n, 0.15)]
  A = coo_matrix((vals, (rows, cols)), shape=(m + n, n)).tocsr()
  off = np.zeros((n, 3), np.float32)
  for c in range(3):
    # o_i - o_j = mean(edge_j) - mean(edge_i)
    L = np.log1p(S[..., c].astype(np.float64))  # gains, not offsets: dark / tinted jobs need scaling
    b = np.array([L[j, 0 if ax == 0 else 2].mean() - L[i, 1 if ax == 0 else 3].mean() for i, j, ax in pairs])
    off[:, c] = lsqr(A, np.r_[b, np.zeros(n)], atol=1e-6, btol=1e-6, iter_lim=4000)[0]
  out = qdir.parent / f"{qdir.name}_harmony.npz"
  np.savez_compressed(out, quads=np.array(quads, np.int32), off=off, strips=S.astype(np.float16))
  print(f"{n} quads, {m} seams; offsets |mean| {np.abs(off).mean():.1f}, max {np.abs(off).max():.1f} -> {out}")
  return out


class Harmony:
  def __init__(self, npz):
    d = np.load(npz)
    self.idx = {tuple(q): i for i, q in enumerate(d["quads"])}
    self.off, self.S = np.clip(d["off"], -GCAP, GCAP), d["strips"].astype(np.float32)

  def apply(self, img, qx, qy):
    i = self.idx.get((qx, qy))
    if i is None:
      return img
    a = np.asarray(img, np.float32)
    g = np.zeros((Q, Q, 3), np.float32)
    off = lambda x, y: self.off[self.idx[(x, y)]] if (x, y) in self.idx else self.off[i]  # noqa: E731
    # 1) offsets interpolated bilinearly between quadrant centres, one quarter at a time
    t = np.abs((np.arange(Q) + 0.5) / Q - 0.5)  # 0 at the centre .. 0.5 at the edge
    H = Q // 2
    for hy, ys in ((-1, slice(0, H)), (1, slice(H, Q))):
      for hx, xs in ((-1, slice(0, H)), (1, slice(H, Q))):
        wx, wy = t[xs][None, :, None], t[ys][:, None, None]
        g[ys, xs] = ((1 - wx) * (1 - wy) * off(qx, qy) + wx * (1 - wy) * off(qx + hx, qy)
                      + (1 - wx) * wy * off(qx, qy + hy) + wx * wy * off(qx + hx, qy + hy))
    a = (a + 1) * np.exp(g) - 1
    # 2) hem: meet the neighbour halfway along each edge, faded over HEM px
    r = np.clip(1 - np.arange(Q) / HEM, 0, 1)
    gain = lambda k: np.exp(self.off[k])  # noqa: E731
    me = (self.S[i] + 1) * gain(i) - 1
    smooth = lambda v: np.stack([np.convolve(np.pad(v[:, c], 24, mode="edge"), np.ones(49) / 49, "valid") for c in range(3)], 1)  # noqa: E731
    for side, (nx, ny, theirs) in enumerate([(qx - 1, qy, 1), (qx + 1, qy, 0), (qx, qy - 1, 3), (qx, qy + 1, 2)]):
      j = self.idx.get((nx, ny))
      if j is None:
        continue
      d = smooth(np.repeat(np.clip((((self.S[j, theirs] + 1) * gain(j) - 1) - me[side]) / 2, -CAP, CAP), BIN, 0))  # (Q, 3) along the edge
      if side == 0:
        a += r[None, :, None] * d[:, None, :]
      elif side == 1:
        a += r[::-1][None, :, None] * d[:, None, :]
      elif side == 2:
        a += r[:, None, None] * d[None, :, :]
      else:
        a += r[::-1][:, None, None] * d[None, :, :]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))



# ---- vegetation patchwork: local low-frequency colour correction -------------------------------
LF = 16  # 1 low-res px = 16 image px (a quadrant is 32x32)


def _small(path):
  return np.asarray(Image.open(path).convert("RGB").resize((Q // LF, Q // LF), Image.Resampling.BOX), np.float32)


def build_lowfreq(qdir):
  """Per-pixel RGB offsets (at 1/16 res) that pull each patch of forest / mangrove towards the
  colour of its wider surroundings: removes quadrant-sized tone patches, leaves detail alone."""
  from scipy.ndimage import gaussian_filter
  qdir = Path(qdir)
  quads = sorted(tuple(map(int, p.stem.split("_"))) for p in qdir.glob("*.png"))
  with ProcessPoolExecutor(10) as ex:
    smalls = list(ex.map(_small, [qdir / f"{x}_{y}.png" for x, y in quads], chunksize=64))
  n = Q // LF
  X, Y = max(q[0] for q in quads) + 1, max(q[1] for q in quads) + 1
  img = np.zeros((Y * n, X * n, 3), np.float32)
  have = np.zeros((Y * n, X * n), np.float32)
  for (x, y), s in zip(quads, smalls):
    img[y * n : (y + 1) * n, x * n : (x + 1) * n] = s
    have[y * n : (y + 1) * n, x * n : (x + 1) * n] = 1
  r, g, b = img[..., 0], img[..., 1], img[..., 2]
  veg = ((g > r + 6) & (g > b) & (have > 0)).astype(np.float32)
  w = np.clip((gaussian_filter(veg, 3) - 0.45) / 0.35, 0, 1)
  hs = gaussian_filter(have, 2) + 1e-3
  hb = gaussian_filter(have, 24) + 1e-3
  corr = np.zeros_like(img)
  for c in range(3):
    small = gaussian_filter(img[..., c] * have, 2) / hs  # ~ 32 px real: patch tone
    big = gaussian_filter(img[..., c] * have, 24) / hb  # ~ 400 px real: surroundings
    corr[..., c] = np.clip(big - small, -30, 30) * w
  out = qdir.parent / f"{qdir.name}_lowfreq.png"
  Image.fromarray(np.clip(corr + 128, 0, 255).astype(np.uint8)).save(out, optimize=True)
  print(f"low-frequency vegetation correction {img.shape[1]}x{img.shape[0]}, mean |corr| {np.abs(corr).mean():.2f} -> {out}")


class LowFreq:
  def __init__(self, png):
    Image.MAX_IMAGE_PIXELS = None
    self.c = np.asarray(Image.open(png), np.int16) - 128

  def apply(self, img, qx, qy):
    n = Q // LF
    y0, x0 = qy * n - 1, qx * n - 1
    if y0 < 0 or x0 < 0 or y0 + n + 2 > self.c.shape[0] or x0 + n + 2 > self.c.shape[1]:
      return img
    patch = self.c[y0 : y0 + n + 2, x0 : x0 + n + 2].astype(np.float32)
    if not np.abs(patch).max():
      return img
    # bilinear upsample with the 1-px margin so neighbouring quadrants agree at the seam
    ch = [np.asarray(Image.fromarray(patch[..., k] + 64).resize(((n + 2) * LF, (n + 2) * LF), Image.Resampling.BILINEAR), np.float32)[LF : LF + Q, LF : LF + Q] - 64 for k in range(3)]
    a = np.asarray(img, np.float32) + np.stack(ch, -1)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


if __name__ == "__main__":
  build(ROOT / "out" / sys.argv[1] / sys.argv[2])
  build_lowfreq(ROOT / "out" / sys.argv[1] / sys.argv[2])
