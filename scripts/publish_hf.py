"""Publish the trained Mumbai style LoRA to Hugging Face with a model card.

Needs HF_TOKEN (write) and HF_REPO (e.g. flash1405/isometric-mumbai) in .env, and the LoRA
file locally (default: out/loras/isometric-mumbai-teacher-v1.safetensors).

  uv run --with huggingface_hub python scripts/publish_hf.py [path/to/lora.safetensors] [--samples dir]
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from build_dataset import PROMPT  # noqa: E402

CARD = """---
license: apache-2.0
base_model: Qwen/Qwen-Image-Edit
library_name: diffusers
pipeline_tag: image-to-image
tags: [lora, qwen-image-edit, isometric, pixel-art, map, mumbai]
---

# Isometric Mumbai — style LoRA

Turns an isometric "block" render of a city (extruded building footprints with satellite
imagery on roofs and ground, camera azimuth −15°, elevation −45°, ~0.29 m/px) into finished
isometric pixel art, in the infill-template format used by the
[isometric-mumbai](https://github.com/{gh}) pipeline. It painted all of
[isometricmumbai.com](https://isometricmumbai.com).

## Use

```python
import torch
from diffusers import QwenImageEditPipeline
from PIL import Image

pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("{repo}", weight_name="{weight}")
template = Image.open("template.png")  # 1024x1024: finished art as context, render inside a 2px red box
out = pipe(image=template, prompt=PROMPT, true_cfg_scale=2.0, num_inference_steps=14, guidance_scale=3.0).images[0]
```

Prompt:

> {prompt}

The red box marks the region to paint; anything outside it is context the output should
continue seamlessly. For a full tile, draw the box around the whole image.

## Training

* Base: Qwen/Qwen-Image-Edit (Apache 2.0), LoRA rank 16, 3000 steps, lr 1e-4, bf16 ([ai-toolkit](https://github.com/ostris/ai-toolkit)).
* Data: ~800 Mumbai render tiles (footprints: GlobalBuildingAtlas; heights: OpenStreetMap,
  Google Open Buildings 2.5D; imagery: Google) paired with redraws by GPT Image
  (OpenAI), each used as full-tile and strip/centre infill examples.
* Provenance note: the training targets were generated with an OpenAI model. You own your
  outputs under OpenAI's terms, but check those terms for your use of this LoRA.

## Credits

Pipeline and conventions build on [isometric-nyc](https://github.com/cannoneyed/isometric-nyc)
by Andy Coenen (MIT). Made by [@flash1405](https://x.com/flash1405).
"""


def main():
  for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
      k, v = line.split("=", 1)
      os.environ.setdefault(k.strip(), v.strip())
  from huggingface_hub import HfApi

  lora = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ROOT / "out" / "loras" / "isometric-mumbai-teacher-v1.safetensors"
  repo = os.environ.get("HF_REPO", "Flash1405/isometric-mumbai")
  api = HfApi(token=os.environ.get("HF_TOKEN"))  # falls back to `hf auth login`
  api.create_repo(repo, repo_type="model", exist_ok=True)
  card = CARD.replace("{repo}", repo).replace("{weight}", lora.name).replace("{prompt}", PROMPT).replace(
    "{gh}", os.environ.get("GITHUB_REPO", "flash1405/isometric-mumbai"))
  (ROOT / "out" / "MODEL_CARD.md").write_text(card)
  api.upload_file(path_or_fileobj=str(ROOT / "out" / "MODEL_CARD.md"), path_in_repo="README.md", repo_id=repo)
  api.upload_file(path_or_fileobj=str(lora), path_in_repo=lora.name, repo_id=repo)
  if "--samples" in sys.argv:
    d = Path(sys.argv[sys.argv.index("--samples") + 1])
    api.upload_folder(folder_path=str(d), path_in_repo="samples", repo_id=repo)
  print(f"https://huggingface.co/{repo}")


if __name__ == "__main__":
  main()
