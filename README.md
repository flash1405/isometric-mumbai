# Isometric Mumbai

All of Greater Mumbai — about 560 km², Colaba to Dahisar — as one explorable isometric
pixel-art map, with simulated taxis, BEST buses, autos and local trains running on the
city's real roads and tracks.

**Live:** https://isometricmumbai.com · made by [@flash1405](https://x.com/flash1405)

This repo is the whole pipeline: render a city from open data and satellite imagery, train a
style model, paint the city tile by tile without seams, and serve it as a zoomable map. It is
written so you can point it at another city.

---

## Why Mumbai needed a different pipeline

Isometric NYC, London, SF and Vancouver were drawn from **Google Photorealistic 3D Tiles**.
In Mumbai (and Bangkok, Kuala Lumpur, most of India) that mesh is essentially flat — a
satellite photo draped on terrain. So this project builds its own 3D city:

| Layer | Source |
|---|---|
| Ground and roof imagery | Google 3D Tiles rendered straight down (one top-down "ortho" pass) |
| Building footprints | [GlobalBuildingAtlas](https://github.com/zhu-xlab/GlobalBuildingAtlas) (TU Munich, 2025) |
| Building heights | merged, best first: OSM `height`/`building:levels` (via Overture) → hand-checked Wikidata towers → [Google Open Buildings 2.5D](https://sites.research.google/gr/open-buildings/temporal) (2023, ~4 m) → GlobalBuildingAtlas; plus the 254 completed 150 m+ towers from Wikipedia's *List of tallest buildings in Mumbai* (with coordinates), since no open raster gets them right (Open Buildings stops at 100 m, GlobalBuildingAtlas at 143 m) |
| Coastline, creeks | Overture `water` (ocean, river) |
| Roads, rail, ferries | Overture `transportation` (OpenStreetMap) |
| Names, search | Overture divisions + places, plus a short curated landmark list |
| Sensitive sites | Overture military land use + BARC: pixelated and blurred before generation |

The footprints are extruded to their heights in a three.js scene (`render/hybrid.js`) with
the satellite photo on roofs and ground, and rendered with the same isometric camera the NYC
project used (azimuth −15°, elevation −45°, 0.293 m per pixel).

## How the art is made

1. **Teacher.** GPT Image redraws ~700 varied Mumbai render tiles as finished isometric pixel
   art with an *accuracy-first* prompt: every building keeps its footprint, height and roof;
   facades get real Mumbai colours; low-rise stays low-rise; nothing is invented; **roads and
   tracks are painted empty** so that all traffic on the map is the moving kind
   (`pipeline/teacher.py` `PROMPT_FAITHFUL`, `pipeline/collect.py --faithful`).
2. **Student.** A rank-32 LoRA on [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
   (Apache 2.0) learns render → art in the same "infill" template format the generator uses,
   plus colour-shifted copies of each render (after [Isometric SF](https://www.justinchavez.com/blog/isometric-sf/))
   (`pipeline/build_teacher_dataset.py --shift 2`, `pipeline/train_klein*.yaml`,
   `scripts/klein_train.sh`, [ai-toolkit](https://github.com/ostris/ai-toolkit)). About 1 hour on one B200;
   it paints a 1024 px tile in 4 steps (~2.5 s).
3. **Seam-free generation.** The city is a global grid of 512 px quadrants. A 4-phase
   schedule paints 2×2 blocks first, then the gaps between them with finished art on both
   sides, so tiles never meet without context (`pipeline/generate_city.py`). Jobs within a
   phase are independent and run across any number of GPUs (`scripts/gen_node.sh`).
4. **Finishing** (`pipeline/export_city.py`): per-quadrant colour gains solved city-wide by
   least squares + a feathered "hem" at every seam + a low-frequency pass that evens out tone
   patches in forest and mangrove (`pipeline/harmonize.py`); one consistent sea from coastline
   polygons that keeps bridges and buildings leaning over water; restricted sites as painted
   woodland; a soft, rounded edge where the city meets unpainted land; a WebP tile pyramid with
   a little extra colour and sharpness on the zoomed-out levels.
5. **Landmarks** (`pipeline/landmarks_gpt.py`): the student paints generic buildings, so ~20
   famous ones (Gateway of India, CSMT, Haji Ali, the Taj…) are repainted in place by GPT Image
   with a description and the surrounding art as context, then blended in.

### What didn't work (so you don't have to try it)

* **Borrowing the isometric-nyc LoRA as is.** Fine for a demo, but on synthetic block renders
  it copies the boxes and window grids: flat and repetitive.
* **Learning "real 3D" from cities that have it.** We rendered London, Singapore and Hong Kong
  both ways (blocks vs Google's real mesh) and trained massing → 3D. The targets were too
  photographic and the model produced soft, melted buildings. A frontier teacher on the
  target city itself was far better.
* **Google's mesh for heights in Mumbai.** Its surface is smoothed to ~30 m; per-building
  heights can't be recovered from it.
* **Open image editors as teacher.** Qwen-Image-Edit-2511 draws lovely art but invents
  harbours, seafronts and roads that aren't there.
* **A "stylised" teacher prompt.** Asking for domes, terracotta and water tanks "where the
  roofs suggest them" made the student invent them everywhere and turn tin-roof low-rise into
  beige apartment blocks. Ask for the city as it is.
* **Lifting the painted cars and animating them.** They come out incomplete and change shape
  when they turn. Paint roads empty and drive rendered sprites instead.

## The site

`site/` is a static Leaflet app (no build step): shareable `/v/<zoom>/<x>/<y>` views,
search, "surprise me", neighbourhood names by importance, landmark labels, and a canvas
traffic layer (`site/traffic.js`):

* **Vehicles** are rendered from small 3D models in the map's own camera and scale
  (`scripts/vehicles/`): kaali-peeli taxis, BEST buses, autos, hatchbacks, sedans, SUVs,
  tempos and bikes, each at 72 headings, at 2× resolution.
* They drive on the left **inside the painted carriageways**: every main road's asphalt edges
  are measured on the art every ~3 m (`pipeline/road_snap.py`) and lanes are placed between
  them. Cars follow each other, queue at signalled junctions, turn onto connecting roads and
  respawn off-screen; buildings in front of a road hide them (silhouettes from the renders).
* **Mumbai locals**: 12-coach Western/Central rakes, also 3D-rendered, on the painted track bed
  (`pipeline/rail_snap.py`), only where track is visible.
* Only the on-screen part of the city is simulated: 60 fps on a mid-range phone.

Hosting: Cloudflare Worker (static assets) for the site, R2 for ~5 GB of WebP tiles
(`scripts/deploy.py`, versioned so a deploy never mixes old and new data).

---

## Run it for your city

Requirements: macOS/Linux with Chrome, Node 20+, [uv](https://docs.astral.sh/uv/), a CUDA
GPU box reachable over SSH (we used Lambda: one B200 + 8× A100 for the final run), and keys
for Google Maps (Map Tiles API) and an image model for the teacher.

```bash
npm install && uv sync
cp .env.example .env                   # GOOGLE_MAPS_API_KEY, OPENAI_API_KEY, LAMBDA_API_KEY, ...
```

1. **Configure** `config/<city>.json`: centre, boundary (Overture division names), regions,
   masks. See `config/mumbai.json`.
2. **Plan the grid:** `uv run python pipeline/city.py plan <city>`
3. **Data:** `uv run python pipeline/city.py gba <city>` (buildings),
   `uv run python pipeline/heights.py build <city>`, `uv run python pipeline/water.py build <city>`
4. **Render** (Chrome on your machine, free Google tier): `bash scripts/render_city.sh <city> [region]`
5. **Teacher pairs:** `N=800 bash scripts/collect_loop.sh`
6. **Train:** `uv run python pipeline/build_teacher_dataset.py <city>` then
   `uv run python pipeline/lambda_gpu.py train`
7. **Generate:** bring up painters (`bash scripts/gen_node.sh <lambda-instance> <replicas> <lora>`,
   `uv run python pipeline/lambda_gpu.py tunnels`) then
   `QUADS_SUBDIR=quads_v3 GEN_STEPS=4 GEN_GUIDANCE=1.0 uv run python pipeline/generate_city.py <city>`
8. **Traffic data on the art:** `pipeline/traffic.py`, `pipeline/road_snap.py`, `pipeline/rail_snap.py`,
   `node scripts/vehicles/render.mjs && uv run python scripts/vehicles/atlas.py` (and the train equivalents)
9. **Export & publish:** `pipeline/harmonize.py`, `pipeline/export_city.py`, `pipeline/places.py`,
   `scripts/deploy.py tiles|site` — or the whole tail in one go: `bash scripts/ship_v3city.sh`

Or skip steps 5–6 and use the published Mumbai LoRA (Hugging Face, see below) as your style.

### Rough cost for Mumbai (563 km²)

| Item | Cost |
|---|---|
| Google Map Tiles API (≈50 sessions) | $0 (free tier) |
| Teacher images (~800 × GPT Image, high) | ~$100–150 |
| LoRA training (≈1 h on one B200) | your GPU time |
| Generation (~8,000 calls; 4-step klein ≈1 h on a B200 + a few small GPUs) | ~$10–30 |
| Hosting (Cloudflare Worker + R2) | ~$0 |

## Model

The Mumbai style LoRAs are on Hugging Face ([Flash1405/isometric-mumbai](https://huggingface.co/Flash1405/isometric-mumbai))
with a model card describing inputs, prompt and training data: FLUX.2 [klein] 4B (recommended)
and the first Qwen-Image-Edit version. Their training targets were produced with GPT Image;
check OpenAI's terms for your use.

## Credits & licences

* Code: MIT (this repo). Camera conventions and the infill template format follow
  [isometric-nyc](https://github.com/cannoneyed/isometric-nyc) by Andy Coenen (MIT).
* Imagery & 3D: Google. The repo contains no Google imagery; render your own with your key.
* Buildings: GlobalBuildingAtlas (CC BY 4.0 / ODbL). Heights also from Google Open Buildings
  2.5D Temporal (CC BY 4.0) and OpenStreetMap.
* Names, roads, rail, water, land use: Overture Maps Foundation, © OpenStreetMap contributors (ODbL).
* Base models: FLUX.2 [klein] 4B and Qwen-Image-Edit (both Apache 2.0). Viewer: Leaflet (BSD-2).
* Tower heights: Wikipedia, *List of tallest buildings in Mumbai* (CC BY-SA 4.0).
* Inspired by Isometric NYC, tiny.london, Isometric SF and Isometric Vancouver.
