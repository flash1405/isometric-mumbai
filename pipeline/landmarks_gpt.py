"""Hero landmarks: repaint famous buildings so they are recognisable.

The style model paints every building as a generic Mumbai block (it only sees extruded
footprints), so the Gateway of India came out as a shop building. For each landmark: take the
painted map around it (1024 px window), replace the centre box with the 3D render (position and
size), and ask GPT Image to paint that landmark there, described in words, in the map's style.
Only the box is kept, feathered into the map.

  uv run python pipeline/landmarks_gpt.py mumbai candidates   -> out/<city>/landmarks/<slug>_<k>.png
  uv run python pipeline/landmarks_gpt.py mumbai apply <slug>=<k> ...   (writes into quads_v2)
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import city  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q, T = 512, 1024

# name -> (description, box px). Positions come from config/landmarks_mumbai.json (+ extras here).
LANDMARKS = {
  "Gateway of India": ("the Gateway of India: a yellow-brown basalt triumphal arch about 26 m tall, one large central arch with four ornate corner turrets, standing on a wide stone plaza at the harbour edge", 448),
  "Taj Mahal Palace": ("the Taj Mahal Palace hotel: a grand Indo-Saracenic hotel with a large central red-orange dome and several smaller red domes, cream and grey stone facades with arched windows, the modern glass Taj tower wing beside it", 640),
  "Chhatrapati Shivaji Maharaj Terminus (CSMT)": ("Chhatrapati Shivaji Maharaj Terminus: a Victorian Gothic railway station of cream and sandstone, a big central stone dome with a statue on top, pointed turrets, gables and arched windows, platforms with long roofs behind it", 640),
  "Flora Fountain (Hutatma Chowk)": ("Flora Fountain: an ornate white stone fountain with a statue on top, on a small plaza island in a busy junction surrounded by heritage buildings", 384),
  "Rajabai Clock Tower": ("the Rajabai Clock Tower: a tall slender Gothic clock tower of buff stone, about 85 m high, beside the Gothic buildings of Mumbai University in a garden", 512),
  "Bombay High Court": ("the Bombay High Court: a long Gothic Revival building of grey-blue basalt with pointed arches and a central tower", 576),
  "Churchgate Station": ("Churchgate railway station: a heritage stone station building with a Gothic facade, with long platform roofs and suburban trains behind it", 512),
  "Chhatrapati Shivaji Maharaj Vastu Sangrahalaya": ("the Prince of Wales Museum: an Indo-Saracenic museum with a large white-grey central dome and smaller domes, set in green gardens", 512),
  "Asiatic Society (Town Hall)": ("the Asiatic Society Town Hall: a white neoclassical building with a long portico of Doric columns above a wide flight of steps", 448),
  "BMC Headquarters": ("the BMC Headquarters: a Gothic municipal building of stone with a tall central dome tower, at a road junction", 512),
  "Bombay Stock Exchange": ("the Bombay Stock Exchange: a tall 29-storey brown-grey concrete tower (Phiroze Jeejeebhoy Towers) among older buildings", 448),
  "Regal Cinema": ("Regal Cinema: an Art Deco cinema building in cream with bold vertical lines, at a roundabout", 384),
  "Crawford Market": ("Crawford Market: a Norman Gothic market hall of red and buff stone with a clock tower and big roofs", 512),
  "Haji Ali Dargah": ("the Haji Ali Dargah: a white marble mosque and tomb with white domes and slender white minarets on a small islet in the sea, joined to the shore by a narrow causeway", 512),
  "Siddhivinayak Temple": ("the Siddhivinayak Temple: a Hindu temple with a gold-plated central dome and many smaller domes, on a busy street", 384),
  "Mount Mary Church": ("Mount Mary Church: a Gothic church of grey stone with two tall spires, on a hill with trees", 448),
  "Antilia": ("Antilia: a 27-storey skyscraper of irregular stacked floors with hanging gardens and terraces, much taller than its neighbours", 448),
  "Global Vipassana Pagoda": ("the Global Vipassana Pagoda: a huge golden pagoda dome with a spire, surrounded by smaller golden stupas and lawns", 640),
}
EXTRA = {"Antilia": (18.9682, 72.8095), "Global Vipassana Pagoda": (19.2286, 72.8059)}

PROMPT = (
  "This is a tile of an isometric pixel-art map of Mumbai, seen from above at 45 degrees. Everything outside the "
  "thin red rectangle is finished art. Inside the rectangle is a rough 3D massing render that shows where the "
  "building stands. Repaint ONLY the inside of the rectangle as {desc}. Make it clearly recognisable, at its real "
  "size and position as in the render, in exactly the same pixel-art style, palette, lighting and isometric angle "
  "as the surrounding art, continuing roads and ground seamlessly across the rectangle edge. Remove the red line. "
  "Do not add text or labels."
)


def slug(s):
  return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def positions(name):
  cfg, pl, _ = city.load(name)
  enu = city.enu_fn(cfg["lat"], cfg["lon"])
  (rx, ry), (ux, uy) = city.basis(cfg)
  mpp = cfg["meters_per_pixel"]
  ll = {l["name"]: (l["lat"], l["lon"]) for l in json.loads((ROOT / "config" / "landmarks_mumbai.json").read_text())}
  ll.update(EXTRA)
  out = {}
  for n in LANDMARKS:
    if n in ll:
      x, y = enu(*ll[n])
      out[n] = (pl["width"] / 2 + (x * rx + y * ry) / mpp, pl["height"] / 2 - (x * ux + y * uy) / mpp)
  return out


def window(qdir, wx, wy, kind="quads"):
  img = Image.new("RGB", (T, T))
  for qy in range(wy // Q, (wy + T - 1) // Q + 1):
    for qx in range(wx // Q, (wx + T - 1) // Q + 1):
      if kind == "quads":
        p = qdir / f"{qx}_{qy}.png"
        src = Image.open(p).convert("RGB") if p.exists() else None
      else:
        p = qdir / f"tile_{qx // 2}_{qy // 2}.png"
        src = Image.open(p).convert("RGB").crop(((qx % 2) * Q, (qy % 2) * Q, (qx % 2) * Q + Q, (qy % 2) * Q + Q)) if p.exists() else None
      if src is not None:
        img.paste(src, (qx * Q - wx, qy * Q - wy))
  return img


def candidates(name, n=2):
  import teacher
  teacher.env()
  out = ROOT / "out" / name / os.environ.get("LM_OUT", "landmarks")
  out.mkdir(parents=True, exist_ok=True)
  qdir, rdir = ROOT / "out" / name / os.environ.get("LM_QDIR", "quads_v2"), ROOT / "out" / name / "renders"
  jobs = []
  for lm, (px, py) in positions(name).items():
    desc, box = LANDMARKS[lm]
    wx, wy = int(px) - T // 2, int(py) - T // 2
    art, ren = window(qdir, wx, wy), window(rdir, wx, wy, "render")
    b0 = (T - box) // 2
    tmpl = art.copy()
    tmpl.paste(ren.crop((b0, b0, b0 + box, b0 + box)), (b0, b0))
    d = ImageDraw.Draw(tmpl)
    for i in range(2):
      d.rectangle([b0 + i, b0 + i, b0 + box - 1 - i, b0 + box - 1 - i], outline=(255, 0, 0))
    s = slug(lm)
    art.save(out / f"{s}_before.png")
    json.dump({"name": lm, "wx": wx, "wy": wy, "box": box}, open(out / f"{s}.json", "w"))
    for k in range(n):
      jobs.append((s, k, tmpl, PROMPT.format(desc=desc)))

  def run(j):
    s, k, tmpl, prompt = j
    try:
      teacher.restyle(tmpl, prompt=prompt, tries=4).save(out / f"{s}_{k}.png")
      return s, k, "ok"
    except Exception as e:  # noqa: BLE001
      return s, k, str(e)[:80]

  with ThreadPoolExecutor(16) as ex:
    for r in ex.map(run, jobs):
      print(*r, flush=True)


def blend(art, new, b0, box, ring=48, hem=56, feather=22):
  """Merge GPT's repainted box into the map so the seam disappears:
  1) colour transfer: GPT also redrew the context around the box; matching its colours there to the
     real art gives a per-channel gain/offset for the whole box;
  2) hem: the remaining difference along each edge (smoothed profile) fades out ~hem px into the box;
  3) a soft feathered edge."""
  A = np.asarray(art, np.float32)
  N = np.asarray(new, np.float32).copy()
  x0, x1 = b0, b0 + box
  outer = np.zeros(A.shape[:2], bool)
  outer[max(0, x0 - ring):min(T, x1 + ring), max(0, x0 - ring):min(T, x1 + ring)] = True
  outer[x0:x1, x0:x1] = False
  for c in range(3):
    ma, sa = A[..., c][outer].mean(), A[..., c][outer].std() + 1e-3
    mn, sn = N[..., c][outer].mean(), N[..., c][outer].std() + 1e-3
    N[..., c] = (N[..., c] - mn) * np.clip(sa / sn, 0.8, 1.25) + ma
  # hem: per-position difference across each edge, smoothed, faded inward
  def smooth(v, k=41):
    return np.stack([np.convolve(np.pad(v[:, ch], k // 2, mode="edge"), np.ones(k) / k, "valid") for ch in range(3)], 1)
  r = np.clip(1 - np.arange(box) / hem, 0, 1)
  band = 6
  sides = [  # (art strip just outside, new strip just inside, axis, at start)
    (A[x0:x1, max(0, x0 - band):x0].mean(1), N[x0:x1, x0:x0 + band].mean(1), "x", True),
    (A[x0:x1, x1:x1 + band].mean(1), N[x0:x1, x1 - band:x1].mean(1), "x", False),
    (A[max(0, x0 - band):x0, x0:x1].mean(0), N[x0:x0 + band, x0:x1].mean(0), "y", True),
    (A[x1:x1 + band, x0:x1].mean(0), N[x1 - band:x1, x0:x1].mean(0), "y", False),
  ]
  for out_strip, in_strip, axis, start in sides:
    d = np.clip(smooth(out_strip - in_strip), -40, 40)  # (box, 3)
    rr = r if start else r[::-1]
    if axis == "x":
      N[x0:x1, x0:x1] += d[:, None, :] * rr[None, :, None]
    else:
      N[x0:x1, x0:x1] += d[None, :, :] * rr[:, None, None]
  N = np.clip(N, 0, 255)
  m = Image.new("L", (T, T), 0)
  ImageDraw.Draw(m).rectangle([x0 + feather // 2, x0 + feather // 2, x1 - feather // 2, x1 - feather // 2], fill=255)
  m = np.asarray(m.filter(ImageFilter.GaussianBlur(feather / 2)), np.float32)[..., None] / 255
  return Image.fromarray((A * (1 - m) + N * m).astype(np.uint8))


def apply(name, picks):
  """Feather the chosen candidates' boxes into the painted quadrants (originals backed up)."""
  out = ROOT / "out" / name / os.environ.get("LM_OUT", "landmarks")
  qdir = ROOT / "out" / name / os.environ.get("LM_QDIR", "quads_v2")
  bak = ROOT / "out" / name / f"{qdir.name}_prelandmark"
  bak.mkdir(exist_ok=True)
  for pick in picks:
    s, k = pick.split("=")
    meta = json.load(open(out / f"{s}.json"))
    wx, wy, box = meta["wx"], meta["wy"], meta["box"]
    art = window(qdir, wx, wy)
    new = Image.open(out / f"{s}_{k}.png").convert("RGB").resize((T, T))
    b0 = (T - box) // 2
    art = blend(art, new, b0, box)
    for qy in range(wy // Q, (wy + T - 1) // Q + 1):
      for qx in range(wx // Q, (wx + T - 1) // Q + 1):
        p = qdir / f"{qx}_{qy}.png"
        if not p.exists():
          continue
        if not (bak / p.name).exists():
          Image.open(p).save(bak / p.name)
        ox, oy = qx * Q - wx, qy * Q - wy
        quad = Image.open(p).convert("RGB")
        part = art.crop((max(0, ox), max(0, oy), min(T, ox + Q), min(T, oy + Q)))
        quad.paste(part, (max(0, -ox), max(0, -oy)))
        quad.save(p)
    print("applied", s, k)


if __name__ == "__main__":
  if sys.argv[2] == "candidates":
    candidates(sys.argv[1])
  else:
    apply(sys.argv[1], sys.argv[3:])
