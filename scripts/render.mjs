// Render an area's 1024px tiles in one headless Chrome session.
//
//   npm run render -- fort --stage=ortho    top-down Google render  -> out/fort/ortho_tiles/
//   npm run render -- fort --stage=dsm      top-down surface height -> out/fort/dsm_tiles/
//   npm run render -- fort --stage=hybrid   ortho + extruded buildings -> out/fort/renders/
//   npm run render -- fort --stage=google   isometric Google render  -> out/fort/renders_google/
//
// Options: --headed, --only=tx,ty, --force (re-render existing tiles)
// Google stages use one billable 3D Tiles root request per run; hybrid uses none.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { chromium } from "playwright";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const flag = (k) => args.find((a) => a.startsWith(`--${k}=`))?.split("=")[1];
const name = args.find((a) => !a.startsWith("--")) || "fort";
const stage = flag("stage") || "hybrid";
const headed = args.includes("--headed");
const force = args.includes("--force");
const only = flag("only");
// city-scale options: explicit config file, data dir, output dir, tile list, parallel pages
const configFile = flag("config-file") || path.join(root, "config", `${name}.json`);
const tilesFile = flag("tiles");
const pages = parseInt(flag("pages") || "1");

const base = JSON.parse(fs.readFileSync(configFile, "utf8"));
const areaDir = flag("data-dir") || path.join(root, "out", name);

let config = base, page_ = "index.html", outDir = path.join(areaDir, "renders");
if (stage === "ortho" || stage === "dsm") {
  // straight down, north-up, centred on the area; covers the iso view plus margin
  config = {
    ...base,
    camera_azimuth_degrees: 0,
    camera_elevation_degrees: -90,
    meters_per_pixel: base.ortho.meters_per_pixel,
    cols: base.ortho_grid ? base.ortho_grid[0] * 2 : base.ortho.quadrants,
    rows: base.ortho_grid ? base.ortho_grid[1] * 2 : base.ortho.quadrants,
  };
  if (stage === "dsm") config.output = "depth";
  if (stage === "ortho" && base.heights_from === "dsm") config.with_depth = true;
  outDir = path.join(areaDir, stage === "dsm" ? "dsm_tiles" : "ortho_tiles");
} else if (stage === "hybrid") {
  page_ = "hybrid.html";
  config = { ...base, with_mask: true };
  for (const f of ["ortho.jpg", "buildings.json"])
    if (!fs.existsSync(path.join(areaDir, f)))
      throw new Error(`missing out/${name}/${f}: run the ortho stage, then pipeline/prepare.py`);
} else if (stage === "google") {
  outDir = path.join(areaDir, "renders_google");
} else throw new Error(`unknown stage ${stage}`);
if (flag("out-dir")) outDir = flag("out-dir");

const apiKey = process.env.GOOGLE_MAPS_API_KEY;
if (stage !== "hybrid" && !apiKey) throw new Error("Set GOOGLE_MAPS_API_KEY in .env");

const tilesX = Math.ceil(config.cols / 2);
const tilesY = Math.ceil(config.rows / 2);
fs.mkdirSync(outDir, { recursive: true });

const server = await createServer({
  root: path.join(root, "render"),
  server: { port: 5173, strictPort: false },
  logLevel: "error",
});
await server.listen();
const url = server.resolvedUrls.local[0] + page_;

const browser = await chromium.launch({
  channel: "chrome", // installed Google Chrome; headless still gets the Metal GPU
  headless: !headed,
  args: ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"],
});
async function openPage() {
  const page = await browser.newPage({ viewport: { width: 1100, height: 1100 } });
  page.on("console", (m) => m.type() === "error" && console.error("[page]", m.text()));
  page.on("pageerror", (e) => console.error("[page error]", e.message));
  await page.route("**/data/*", (route) => {
    const f = path.join(areaDir, path.basename(new URL(route.request().url()).pathname));
    return fs.existsSync(f) ? route.fulfill({ path: f }) : route.fulfill({ status: 404 });
  });
  await page.addInitScript((c) => (window.__RENDER_CONFIG__ = c), { apiKey, config });
  await page.goto(url);
  await page.waitForFunction("window.RENDERER_READY === true", null, { timeout: 300000 });
  return page;
}

let todo;
if (only) todo = [only.split(",").map(Number)];
else if (tilesFile) todo = JSON.parse(fs.readFileSync(tilesFile, "utf8"));
else {
  todo = [];
  for (let ty = 0; ty < tilesY; ty++) for (let tx = 0; tx < tilesX; tx++) todo.push([tx, ty]);
}
if (!only && !force) todo = todo.filter(([tx, ty]) => !fs.existsSync(path.join(outDir, `tile_${tx}_${ty}.png`)));

console.log(`[${stage}] ${todo.length} tiles for "${name}" -> ${outDir} (${pages} page${pages > 1 ? "s" : ""})`);
const t0 = Date.now();
let done = 0;
async function worker(queue) {
  const page = await openPage();
  const stats = await page.evaluate(() => window.SCENE_STATS);
  if (stats && pages === 1) console.log("scene:", stats);
  while (queue.length) {
    const [tx, ty] = queue.shift();
    if (!only && !force && fs.existsSync(path.join(outDir, `tile_${tx}_${ty}.png`))) continue; // another worker did it
    const { ok, png, depth, mask } = await page.evaluate(([x, y]) => window.renderTile(x, y), [tx, ty]);
    if (mask) fs.writeFileSync(path.join(outDir, `mask_${tx}_${ty}.png`), Buffer.from(mask.split(",")[1], "base64"));
    fs.writeFileSync(path.join(outDir, `tile_${tx}_${ty}.png`), Buffer.from(png.split(",")[1], "base64"));
    if (depth) fs.writeFileSync(path.join(outDir, `depth_${tx}_${ty}.png`), Buffer.from(depth.split(",")[1], "base64"));
    done++;
    if (!ok) console.log(`  tile ${tx},${ty} (timed out waiting for tiles, saved anyway)`);
    if (done % 50 === 0) console.log(`  ${done}/${todo.length} (${((Date.now() - t0) / done / 1000).toFixed(2)} s/tile)`);
  }
}
const queue = [...todo];
await Promise.all(Array.from({ length: Math.min(pages, Math.max(1, todo.length)) }, () => worker(queue)));
console.log(`Done in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
// record the camera target the ortho pass used (pipeline/align.py needs it)
if (stage === "ortho") fs.writeFileSync(path.join(areaDir, "ortho_meta.json"), JSON.stringify({ target_height: config.target_height ?? 0 }));

await browser.close();
await server.close();
