"""Qwen-Image-Edit + isometric-nyc LoRA inference endpoint on Modal.

Adapted from cannoneyed/isometric-nyc inference/server.py (MIT, (c) Andy Coenen).
Changes: base-model weights are cached on the volume so later cold starts skip the
~40GB download, and the LoRA is fetched from Oxen by a one-off Modal function.

  uv run modal run pipeline/modal_server.py::fetch_lora     # once
  uv run modal deploy pipeline/modal_server.py              # prints the edit_b64 URL
"""

import base64
import os
import random
from io import BytesIO

import modal
from pydantic import BaseModel

LORA_MODEL_ID = os.environ.get("LORA_MODEL_ID", "cannoneyed-dark-copper-flea")
OXEN_REPO = "cannoneyed/isometric-nyc-v02"
OXEN_REVISION = "55b9eaff9e93ceae2549c85ba2f10b4f"
GPU = os.environ.get("MODAL_GPU", "H100")

image = (
  modal.Image.debian_slim(python_version="3.11")
  .pip_install(
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "pillow",
    "fastapi[standard]",
    "huggingface_hub[hf_transfer]",
    "oxenai",
  )
  .env({"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("isometric-mumbai")
volume = modal.Volume.from_name("isometric-mumbai-vol", create_if_missing=True)


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def fetch_lora():
  """Download the public isometric-nyc LoRA from Oxen into the volume."""
  from oxen import RemoteRepo

  dest = f"/data/loras/{LORA_MODEL_ID}"
  os.makedirs(dest, exist_ok=True)
  repo = RemoteRepo(OXEN_REPO, revision=OXEN_REVISION)
  # final checkpoint only (the folder also holds intermediate checkpoints + optimizer state)
  repo.download(f"models/{LORA_MODEL_ID}/model.safetensors", dest)
  volume.commit()
  print("files:", os.listdir(dest))


class EditRequest(BaseModel):
  image_b64: str
  prompt: str
  negative_prompt: str | None = None
  true_cfg_scale: float = 2.0
  steps: int = 14
  guidance_scale: float = 3.0
  seed: int | None = None


@app.cls(
  image=image,
  gpu=GPU,
  volumes={"/data": volume},
  scaledown_window=120,  # shut down 2 min after the last request
  timeout=900,
  max_containers=4,
)
class ImageEditor:
  @modal.enter()
  def setup(self):
    import torch
    from diffusers import QwenImageEditPipeline

    self.pipe = QwenImageEditPipeline.from_pretrained(
      "Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16
    )
    volume.commit()  # persist the HF cache after the first download
    lora_path = f"/data/loras/{LORA_MODEL_ID}"
    self.pipe.load_lora_weights(lora_path, weight_name="model.safetensors", adapter_name="iso")
    self.pipe.set_adapters(["iso"], adapter_weights=[1.0])
    self.pipe.to("cuda")

  @modal.fastapi_endpoint(method="POST")
  def edit_b64(self, req: EditRequest):
    import torch
    from PIL import Image

    img = Image.open(BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    with torch.inference_mode():
      out = self.pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        true_cfg_scale=req.true_cfg_scale,
        image=img,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        generator=torch.manual_seed(seed),
      ).images[0]
    buf = BytesIO()
    out.save(buf, format="PNG")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode(), "seed": seed}
