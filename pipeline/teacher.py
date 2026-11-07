"""Teacher: a frontier image model redraws hybrid renders as finished isometric pixel art.

Used to build the training set for our own Mumbai style model (the recipe isometric SF,
tiny.london and isometric Vancouver used). The render fixes geometry (camera, footprints,
heights, roof imagery); the teacher invents believable facades, roofs and depth.

  uv run python pipeline/teacher.py test     # 4 sample tiles -> scratch comparison
"""

import base64
import io
import os
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("TEACHER_MODEL", "gpt-image-2.6-sunburst")
PROMPT = (
  "Redraw this image as a finished isometric pixel-art city illustration, in the style of a "
  "classic high-detail city-builder game (crisp hand-placed pixels, clean outlines, rich but "
  "harmonious colours, soft consistent lighting from the left). Keep EXACTLY the same camera "
  "angle, framing, street layout, coastline and every building's footprint, position and height; "
  "do not add, remove or move buildings or roads. Turn the plain blocks into real Mumbai buildings: "
  "give each one believable facades with windows, balconies and shop fronts at street level, "
  "parapets, water tanks and terraces on flat roofs, terracotta tiles on pitched roofs, and keep "
  "colonial stone buildings, domes and towers where the roof imagery suggests them. Trees become "
  "lush rounded canopies, roads get lane markings and small cars, the sea is calm deep teal. "
  "No text, no labels, no border, no vignette."
)


# v3: accuracy first. The v1/v2 prompt invited domes, terracotta and water tanks, and the student
# learned to invent them everywhere. This one asks for the city as it is.
PROMPT_FAITHFUL = (
  "Redraw this 3D city render as a finished, richly detailed isometric pixel-art illustration in the style of a "
  "high-detail city-builder game (crisp hand-placed pixels, clean outlines, warm harmonious colours, soft light "
  "from the left). Stay FAITHFUL to the geometry: same camera angle and framing; every building exactly where it "
  "is, with the same footprint, height and roof shape; do not add, remove, merge or move any building, road, tree "
  "or water. The plain white walls in the render are only placeholders: paint each building as a real Mumbai "
  "building with its own facade colour (weathered cream, pale yellow, salmon pink, light blue, mint, grey "
  "concrete, a few glass towers), window rows, balconies, window grilles, AC units and shop fronts at street "
  "level. Roofs keep the colours and materials the render shows (grey concrete terraces, blue tarpaulins and "
  "corrugated tin on low-rise roofs, red tiles only where red). Dense low-rise areas stay low and tightly packed; "
  "towers stay tall. Do not invent domes, spires, ornaments or water bodies the render does not show. Trees are "
  "lush rounded canopies. Roads, flyovers and bridges get clean asphalt with lane markings and zebra "
  "crossings but are EMPTY: no cars, buses, rickshaws, bikes or people on them, and no trains on the railway "
  "tracks (moving traffic is added separately). No text, no labels, no border."
)


def env(override=False):
  """Load .env (override=True re-reads it, so a newly pasted key takes effect)."""
  f = ROOT / ".env"
  for line in f.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
      k, v = line.split("=", 1)
      if override:
        os.environ[k.strip()] = v.strip()
      else:
        os.environ.setdefault(k.strip(), v.strip())


_KEYS, _DEAD, _KI = [], set(), [0]


def _key():
  """Round-robin over working OpenAI keys (TEACHER_KEYS = names of env vars; each key has its own
  image rate limit, so more keys = faster). Keys that answer 401 are dropped."""
  if not _KEYS:
    names = os.environ.get("TEACHER_KEYS", "OPENAI_API_KEY_3,OPENAI_API_KEY_2,OPENAI_API_KEY").split(",")
    _KEYS.extend(dict.fromkeys(os.environ[n] for n in names if os.environ.get(n)))
  live = [k for k in _KEYS if k not in _DEAD] or _KEYS
  _KI[0] += 1
  return live[_KI[0] % len(live)]


def restyle(img, prompt=PROMPT, quality="high", tries=4):
  buf = io.BytesIO()
  img.convert("RGB").save(buf, format="PNG")
  for attempt in range(tries):
    key = _key()
    try:
      r = httpx.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {key}",
                 **({"OpenAI-Organization": os.environ["OPENAI_ORG_ID"]} if os.environ.get("OPENAI_ORG_ID") else {})},
        data={"model": MODEL, "prompt": prompt, "size": "1024x1024", "quality": quality, "n": "1"},
        files={"image[]": ("render.png", buf.getvalue(), "image/png")},
        timeout=300,
      )
      if r.status_code == 401:
        _DEAD.add(key)
      if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
      out = Image.open(io.BytesIO(base64.b64decode(r.json()["data"][0]["b64_json"]))).convert("RGB")
      return out.resize((1024, 1024), Image.Resampling.LANCZOS) if out.size != (1024, 1024) else out
    except Exception as e:  # noqa: BLE001
      print(f"  teacher retry {attempt + 1}: {e}", flush=True)
      time.sleep(5 * (attempt + 1))
  raise RuntimeError("teacher failed")


if __name__ == "__main__":
  env()
  if sys.argv[1] == "test":
    sys.path.insert(0, str(ROOT / "pipeline"))
    import city
    from concurrent.futures import ThreadPoolExecutor

    cfg, pl, _ = city.load("mumbai")
    enu = city.enu_fn(cfg["lat"], cfg["lon"])
    (rx, ry), (ux, uy) = city.basis(cfg)
    spots = {"fort": (18.9325, 72.8335), "girgaon": (18.9560, 72.8180),
             "marine_drive": (18.9440, 72.8235), "malabar_hill": (18.9550, 72.8030)}
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "out" / "teacher_test"
    out.mkdir(parents=True, exist_ok=True)

    def one(item):
      name, (la, lo) = item
      x, y = enu(la, lo)
      px = pl["width"] / 2 + (x * rx + y * ry) / cfg["meters_per_pixel"]
      py = pl["height"] / 2 - (x * ux + y * uy) / cfg["meters_per_pixel"]
      t = ROOT / "out" / "mumbai" / "renders" / f"tile_{int(px // 1024)}_{int(py // 1024)}.png"
      src = Image.open(t).convert("RGB")
      t0 = time.time()
      res = restyle(src)
      src.save(out / f"{name}_in.png")
      res.save(out / f"{name}_out.png")
      return f"{name}: {time.time() - t0:.0f}s"

    with ThreadPoolExecutor(4) as ex:
      for line in ex.map(one, spots.items()):
        print(line, flush=True)
