"""Deploy isometricmumbai.com: site -> Cloudflare Worker (static assets), tiles -> R2.

Needs in .env:
  CF_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY   (R2 API token, Object Read & Write)
  R2_BUCKET=isometric-mumbai-tiles
  TILES_PUBLIC_BASE=https://tiles.isometricmumbai.com     (R2 custom domain)
and `npx wrangler login` once.

  uv run python scripts/deploy.py tiles [--version v1]            # upload site/public/tiles/** to R2
  uv run python scripts/deploy.py site  [--version v1] [--preview] # build deploy/dist and deploy
     --preview -> worker "isometricmumbai-preview" on preview.isometricmumbai.com
     default   -> worker "isometricmumbai" on isometricmumbai.com + www
"""

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"


def env():
  f = ROOT / ".env"
  if f.exists():
    for line in f.read_text().splitlines():
      if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def upload_tiles(version):
  import boto3
  from botocore.config import Config

  s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{os.environ['CF_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    # R2 + recent botocore: default CRC checksums use an unseekable chunked stream, so any retry
    # fails with UnseekableStreamError. Only checksum when required, and send plain bytes.
    config=Config(max_pool_connections=64, retries={"max_attempts": 8, "mode": "adaptive"},
                  request_checksum_calculation="when_required", response_checksum_validation="when_required"),
    region_name="auto",
  )
  bucket = os.environ["R2_BUCKET"]
  files = [p for p in (SITE / "public" / "tiles").rglob("*.webp")]
  print(f"uploading {len(files)} tiles to r2://{bucket}/{version}/")

  def put(p):
    key = f"{version}/{p.relative_to(SITE / 'public' / 'tiles').as_posix()}"
    body = p.read_bytes()
    for attempt in range(6):
      try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="image/webp",
                      CacheControl="public, max-age=31536000, immutable")
        return
      except Exception:  # noqa: BLE001 - network blip: back off and retry the whole object
        import time
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"upload failed: {key}")

  done = 0
  with ThreadPoolExecutor(48) as ex:
    for _ in ex.map(put, files):
      done += 1
      if done % 2000 == 0:
        print(f"  {done}/{len(files)}", flush=True)
  print("tiles uploaded")


def build_site(version):
  dist = ROOT / "deploy" / "dist"
  if dist.exists():
    shutil.rmtree(dist)
  dist.mkdir(parents=True)
  for f in ["index.html", "app.js", "style.css", "traffic.js"]:
    shutil.copy(SITE / f, dist / f)
  for p in (SITE / "public").iterdir():
    if p.name == "tiles":
      continue
    (shutil.copytree if p.is_dir() else shutil.copy)(p, dist / p.name)
  cfg = (SITE / "config.js").read_text()
  base = os.environ["TILES_PUBLIC_BASE"].rstrip("/") + f"/{version}/"
  cfg = cfg.replace('tilesBase: "/tiles/"', f'tilesBase: "{base}"').replace('tilesVersion: "dev"', f'tilesVersion: "{version}"')
  (dist / "config.js").write_text(cfg)
  # open the connection to the tile host while the page is still loading
  host = "/".join(base.split("/")[:3])
  html = (dist / "index.html").read_text()
  if "preconnect" not in html:
    html = html.replace("<head>", f'<head>\n<link rel="preconnect" href="{host}" crossorigin>', 1)
  (dist / "index.html").write_text(html)
  # caching: data is versioned (?v=<version>) so it can be cached long; html/js/config revalidate
  (dist / "_headers").write_text(
    "/traffic/*\n  Cache-Control: public, max-age=604800\n"
    "/vendor/*\n  Cache-Control: public, max-age=31536000, immutable\n"
    "/places.json\n  Cache-Control: public, max-age=3600\n"
    "/og-image.jpg\n  Cache-Control: public, max-age=86400\n"
    "/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: strict-origin-when-cross-origin\n"
  )
  n = sum(1 for _ in dist.rglob("*") if _.is_file())
  print(f"dist/ built ({n} files), tiles from {base}")
  return dist


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("what", choices=["tiles", "site"])
  ap.add_argument("--version", default="v1")
  ap.add_argument("--preview", action="store_true")
  args = ap.parse_args()
  env()
  if args.what == "tiles":
    upload_tiles(args.version)
  else:
    build_site(args.version)
    name = "isometricmumbai-preview" if args.preview else "isometricmumbai"
    cfg = {
      "name": name,
      "compatibility_date": "2026-10-01",
      "assets": {"directory": "./dist", "not_found_handling": "single-page-application"},
      "workers_dev": False,
    }
    if args.preview:
      cfg["routes"] = [{"pattern": "preview.isometricmumbai.com", "custom_domain": True},
                       {"pattern": "sample.isometricmumbai.com", "custom_domain": True}]
    else:
      cfg["routes"] = [{"pattern": "isometricmumbai.com", "custom_domain": True}]
      if os.environ.get("WITH_WWW", "1") == "1":
        cfg["routes"].append({"pattern": "www.isometricmumbai.com", "custom_domain": True})
    (ROOT / "deploy" / "wrangler.json").write_text(json.dumps(cfg, indent=2))
    e = dict(os.environ, CLOUDFLARE_ACCOUNT_ID=os.environ["CF_ACCOUNT_ID"])
    r = subprocess.run(["npx", "-y", "wrangler@4.158.0", "deploy", "--config", "wrangler.json"], cwd=ROOT / "deploy", env=e)
    sys.exit(r.returncode)


if __name__ == "__main__":
  main()
