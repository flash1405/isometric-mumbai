"""Standalone Qwen-Image-Edit + isometric-nyc LoRA server (any CUDA box, e.g. a Lambda B200).

Same request/response as pipeline/modal_server.py (adapted from cannoneyed/isometric-nyc, MIT).
Started on the GPU box by `pipeline/lambda_gpu.py up`; listens on 127.0.0.1:8000 and is
reached through an SSH tunnel, so nothing is exposed publicly.
"""

import base64
import os
import random
import threading
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI
from pydantic import BaseModel

LORA_DIR = os.environ.get("LORA_DIR", os.path.expanduser("~/loras/cannoneyed-dark-copper-flea"))
LORA_WEIGHT = os.environ.get("LORA_WEIGHT", "model.safetensors")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit")  # e.g. Qwen/Qwen-Image-Edit-2511
USE_LORA = os.environ.get("USE_LORA", "1") == "1"
state = {}
lock = threading.Lock()  # one generation at a time per process


@asynccontextmanager
async def lifespan(app):
  import torch
  from diffusers import QwenImageEditPipeline

  cls = QwenImageEditPipeline
  if "klein" in MODEL_ID.lower():  # v2 student: FLUX.2 [klein] 4B (Apache 2.0)
    from diffusers import Flux2KleinPipeline as cls
  elif MODEL_ID.endswith(("2509", "2511")):  # newer "Plus" editors use a different pipeline class
    from diffusers import QwenImageEditPlusPipeline as cls
  pipe = cls.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
  if USE_LORA:
    pipe.load_lora_weights(LORA_DIR, weight_name=LORA_WEIGHT, adapter_name="iso")
    pipe.set_adapters(["iso"], adapter_weights=[1.0])
  print(f"model {MODEL_ID} lora={USE_LORA and LORA_WEIGHT}", flush=True)
  pipe.to("cuda")
  state["pipe"] = pipe
  print("model ready", flush=True)
  yield


app = FastAPI(lifespan=lifespan)


class EditRequest(BaseModel):
  image_b64: str
  prompt: str
  negative_prompt: str | None = None
  true_cfg_scale: float = 2.0
  steps: int = 14
  model_kwargs: dict | None = None
  guidance_scale: float = 3.0
  seed: int | None = None


@app.get("/health")
def health():
  return {"ready": "pipe" in state}


@app.post("/edit_b64")
def edit_b64(req: EditRequest):
  import torch
  from PIL import Image

  img = Image.open(BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
  seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
  kw = dict(prompt=req.prompt, num_inference_steps=req.steps, guidance_scale=req.guidance_scale,
            generator=torch.manual_seed(seed), **(req.model_kwargs or {}))
  if "klein" in MODEL_ID.lower():
    kw.update(image=img, height=img.height, width=img.width)
  else:
    kw.update(negative_prompt=req.negative_prompt or " ", true_cfg_scale=req.true_cfg_scale,
              image=[img] if MODEL_ID.endswith(("2509", "2511")) else img)
  with lock, torch.inference_mode():
    out = state["pipe"](**kw).images[0]
  buf = BytesIO()
  out.save(buf, format="PNG")
  return {"image_b64": base64.b64encode(buf.getvalue()).decode(), "seed": seed}
