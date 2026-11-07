// Orthographic "isometric" renderer for Google Photorealistic 3D Tiles.
// Camera setup follows cannoneyed/isometric-nyc (MIT). Differences: a single page
// session renders every tile (one billable root-tileset request), and tiles are
// produced by translating one fixed camera in its own image plane, so neighbouring
// renders are pixel-aligned by construction.
//
// Driven by scripts/render.mjs via window.renderTile(tx, ty).

import { TilesRenderer, WGS84_ELLIPSOID, CAMERA_FRAME } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin,
  GLTFExtensionsPlugin,
  TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import { Matrix4, MathUtils, OrthographicCamera, Scene, ShaderMaterial, WebGLRenderer } from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

// Injected by scripts/render.mjs (config JSON + API key).
const { apiKey: API_KEY, config: cfg } = window.__RENDER_CONFIG__;

const TILE_PX = cfg.quadrant_px * 2; // one render = 2x2 quadrants
const MPP = cfg.meters_per_pixel;
const GRID_W = cfg.cols * cfg.quadrant_px;
const GRID_H = cfg.rows * cfg.quadrant_px;
const hud = document.getElementById("hud");

const renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(TILE_PX, TILE_PX);
renderer.setClearColor(0x000000);
document.body.appendChild(renderer.domElement);

const scene = new Scene();
const half = (TILE_PX * MPP) / 2;
const camera = new OrthographicCamera(-half, half, half, -half, 1, 20000);

const tiles = new TilesRenderer();
tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: API_KEY, autoRefreshToken: true }));
tiles.registerPlugin(new TileCompressionPlugin());
tiles.registerPlugin(
  new GLTFExtensionsPlugin({
    dracoLoader: new DRACOLoader().setDecoderPath(
      "https://www.gstatic.com/draco/versioned/decoders/1.5.7/"
    ),
  })
);
if (cfg.error_target) tiles.errorTarget = cfg.error_target;
tiles.group.rotation.x = -Math.PI / 2; // Z-up -> Y-up
scene.add(tiles.group);
tiles.setCamera(camera);
tiles.setResolutionFromRenderer(camera, renderer);

let tilesetLoaded = false;
tiles.addEventListener("load-tile-set", () => (tilesetLoaded = true));

// Base camera pose: looking at (lat, lon) from the configured azimuth/elevation.
function baseCameraMatrix() {
  tiles.group.updateMatrixWorld(true);
  const m = new Matrix4();
  WGS84_ELLIPSOID.getObjectFrame(
    cfg.lat * MathUtils.DEG2RAD,
    cfg.lon * MathUtils.DEG2RAD,
    cfg.target_height ?? 0,
    cfg.camera_azimuth_degrees * MathUtils.DEG2RAD,
    cfg.camera_elevation_degrees * MathUtils.DEG2RAD,
    0,
    m,
    CAMERA_FRAME
  );
  m.multiply(new Matrix4().makeTranslation(0, 0, 3000)); // back off along view axis
  m.premultiply(tiles.group.matrixWorld);
  return m;
}

// Place the camera so this render covers grid pixels
// [tx*TILE_PX, (tx+1)*TILE_PX) x [ty*TILE_PX, (ty+1)*TILE_PX).
function placeCamera(tx, ty) {
  baseCameraMatrix().decompose(camera.position, camera.quaternion, camera.scale);
  const cx = tx * TILE_PX + TILE_PX / 2 - GRID_W / 2;
  const cy = ty * TILE_PX + TILE_PX / 2 - GRID_H / 2;
  camera.translateX(cx * MPP);
  camera.translateY(-cy * MPP); // image y grows downward
  camera.updateMatrixWorld(true);
}

let frame = 0;
function tick() {
  frame++;
  camera.updateMatrixWorld();
  tiles.setCamera(camera);
  tiles.setResolutionFromRenderer(camera, renderer);
  tiles.update();
  renderer.render(scene, camera);
  const s = tiles.stats || {};
  hud.textContent = `dl ${s.downloading ?? "?"} parse ${s.parsing ?? "?"} frame ${frame}`;
  requestAnimationFrame(tick);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitStable(stableMs = 1500, timeoutMs = 90000) {
  const t0 = performance.now();
  let stableSince = 0;
  while (performance.now() - t0 < timeoutMs) {
    const s = tiles.stats || {};
    const idle = tilesetLoaded && (s.downloading ?? 0) === 0 && (s.parsing ?? 0) === 0;
    if (idle) {
      if (!stableSince) stableSince = performance.now();
      if (performance.now() - stableSince > stableMs) return true;
    } else stableSince = 0;
    await sleep(100);
  }
  return false;
}

// Depth mode (cfg.output === "depth"): encode distance from the camera plane in
// millimetres as 24-bit RGB. With a top-down camera this is a surface-height map (DSM).
const depthMaterial = new ShaderMaterial({
  vertexShader: `varying float vDist;
    void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vDist = -mv.z; gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `varying float vDist;
    void main() {
      float mm = clamp(vDist * 1000.0, 0.0, 16777215.0);
      float r = floor(mm / 65536.0); float g = floor((mm - r * 65536.0) / 256.0); float b = mm - r * 65536.0 - g * 256.0;
      gl_FragColor = vec4(r / 255.0, g / 255.0, b / 255.0, 1.0);
    }`,
});

window.renderTile = async (tx, ty) => {
  placeCamera(tx, ty);
  const startFrame = frame;
  while (frame - startFrame < 5) await sleep(16); // let LOD traversal see the new pose
  const ok = await waitStable();
  if (cfg.output === "depth") scene.overrideMaterial = depthMaterial;
  renderer.render(scene, camera);
  const png = renderer.domElement.toDataURL("image/png");
  scene.overrideMaterial = null;
  let depth = null;
  if (cfg.with_depth) { // same tiles, second pass: surface-distance map
    scene.overrideMaterial = depthMaterial;
    renderer.render(scene, camera);
    depth = renderer.domElement.toDataURL("image/png");
    scene.overrideMaterial = null;
  }
  return { ok, png, depth };
};

window.RENDERER_READY = true;
placeCamera(Math.floor(cfg.cols / 4), Math.floor(cfg.rows / 4));
tick();
