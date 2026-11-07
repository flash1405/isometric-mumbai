// Hybrid renderer for cities without photorealistic 3D coverage (e.g. Mumbai).
// Ground and roofs: a top-down Google render (ortho.jpg, from the ortho stage).
// Walls: GlobalBuildingAtlas LoD1 footprints extruded to their heights, with a
// procedural window pattern and sun shadows. Same camera + tile grid as main.js.
//
// Driven by scripts/render.mjs (--stage=hybrid) via window.renderTile(tx, ty).
// Data files are served by the driver under /data/.

import {
  AmbientLight,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  DirectionalLight,
  DoubleSide,
  Mesh,
  MeshBasicMaterial,
  MeshLambertMaterial,
  OrthographicCamera,
  PCFSoftShadowMap,
  PlaneGeometry,
  RepeatWrapping,
  Scene,
  Shape,
  ShapeGeometry,
  ShapeUtils,
  ShadowMaterial,
  SRGBColorSpace,
  TextureLoader,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

const { config: cfg } = window.__RENDER_CONFIG__;
const TILE_PX = cfg.quadrant_px * 2;
const MPP = cfg.meters_per_pixel;
const GRID_W = cfg.cols * cfg.quadrant_px;
const GRID_H = cfg.rows * cfg.quadrant_px;
// Ortho imagery rectangle in local metres (x east, y north). City chunks pass an explicit
// rect; single areas use a square centred on the origin.
const EXTENT = cfg.ortho.quadrants ? cfg.ortho.quadrants * cfg.quadrant_px * cfg.ortho.meters_per_pixel : 0;
const RECT = cfg.ortho_rect || { x0: -EXTENT / 2, y1: EXTENT / 2, w: EXTENT, h: EXTENT };
const RCX = RECT.x0 + RECT.w / 2, RCY = RECT.y1 - RECT.h / 2;
const D2R = Math.PI / 180;

const renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(TILE_PX, TILE_PX);
renderer.setClearColor(0x000000);
renderer.outputColorSpace = SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new Scene();

// --- ground: ortho imagery -------------------------------------------------
const ortho = await new TextureLoader().loadAsync("/data/ortho.jpg");
ortho.colorSpace = SRGBColorSpace;
ortho.anisotropy = renderer.capabilities.getMaxAnisotropy();
// Imagery is drawn unlit (exact colours); sun shadows go on a transparent overlay.
const shadowMat = new ShadowMaterial({ opacity: 0.32 });
const ground = new Mesh(new PlaneGeometry(RECT.w, RECT.h), new MeshBasicMaterial({ map: ortho }));
const groundShadow = new Mesh(new PlaneGeometry(RECT.w, RECT.h), shadowMat);
ground.position.set(RCX, RCY, 0);
groundShadow.position.set(RCX, RCY, 0);
groundShadow.position.z = 0.05; // (x, y set above)
groundShadow.receiveShadow = true;
scene.add(ground, groundShadow);

// small CPU copy of the ortho for per-building wall colours
const SAMPLE = 1024;
const sampleCanvas = document.createElement("canvas");
sampleCanvas.width = sampleCanvas.height = SAMPLE;
const sctx = sampleCanvas.getContext("2d", { willReadFrequently: true });
sctx.drawImage(ortho.image, 0, 0, SAMPLE, SAMPLE);
const samplePx = sctx.getImageData(0, 0, SAMPLE, SAMPLE).data;
function orthoColor(x, y) {
  const u = Math.floor(((x - RECT.x0) / RECT.w) * SAMPLE);
  const v = Math.floor(((RECT.y1 - y) / RECT.h) * SAMPLE);
  let r = 0, g = 0, b = 0, n = 0;
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) {
      const uu = Math.min(SAMPLE - 1, Math.max(0, u + dx));
      const vv = Math.min(SAMPLE - 1, Math.max(0, v + dy));
      const i = (vv * SAMPLE + uu) * 4;
      r += samplePx[i]; g += samplePx[i + 1]; b += samplePx[i + 2]; n++;
    }
  return new Color(r / n / 255, g / n / 255, b / n / 255);
}

// --- buildings --------------------------------------------------------------
const buildings = await (await fetch("/data/buildings.json")).json();
const FACADE = new Color(0.86, 0.82, 0.74); // light stone/plaster, typical of the Fort area
const FLOOR_H = 3.2, BAY_W = 3.4;
const roofGeoms = [], wallGeoms = [];

for (const b of buildings) {
  const rings = b.rings.map((r) => r.map(([x, y]) => new Vector2(x, y)));
  let outer = rings[0];
  if (ShapeUtils.isClockWise(outer)) outer = outer.slice().reverse();
  const holes = rings.slice(1).map((h) => (ShapeUtils.isClockWise(h) ? h : h.slice().reverse()));
  const shape = new Shape(outer);
  for (const h of holes) shape.holes.push(new Shape(h));

  // roof: flat cap textured from the ortho in world space
  const roof = new ShapeGeometry(shape);
  const pos = roof.attributes.position;
  const uv = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i);
    pos.setZ(i, b.h);
    uv[i * 2] = (x - RECT.x0) / RECT.w;
    uv[i * 2 + 1] = 1 - (RECT.y1 - y) / RECT.h;
  }
  roof.setAttribute("uv", new BufferAttribute(uv, 2));
  roof.computeVertexNormals();
  roofGeoms.push(roof);

  // walls: one quad per edge; colour = facade tint pulled toward the roof colour
  let cx = 0, cy = 0;
  for (const p of outer) { cx += p.x; cy += p.y; }
  const tint = FACADE.clone().lerp(orthoColor(cx / outer.length, cy / outer.length), 0.5);
  const ringsForWalls = [outer, ...holes.map((h) => h.slice().reverse())];
  const P = [], N = [], U = [], C = [];
  for (const ring of ringsForWalls) {
    let s = 0;
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i], c = ring[(i + 1) % ring.length];
      const dx = c.x - a.x, dy = c.y - a.y, len = Math.hypot(dx, dy);
      if (len < 0.05) continue;
      const nx = dy / len, ny = -dx / len; // outward for CCW rings
      const s1 = s + len;
      const quad = [
        [a.x, a.y, 0, s], [c.x, c.y, 0, s1], [c.x, c.y, b.h, s1],
        [a.x, a.y, 0, s], [c.x, c.y, b.h, s1], [a.x, a.y, b.h, s],
      ];
      for (const [x, y, z, ss] of quad) {
        P.push(x, y, z); N.push(nx, ny, 0);
        U.push(ss / BAY_W, z / FLOOR_H);
        C.push(tint.r, tint.g, tint.b);
      }
      s = s1;
    }
  }
  const walls = new BufferGeometry();
  walls.setAttribute("position", new BufferAttribute(new Float32Array(P), 3));
  walls.setAttribute("normal", new BufferAttribute(new Float32Array(N), 3));
  walls.setAttribute("uv", new BufferAttribute(new Float32Array(U), 2));
  walls.setAttribute("color", new BufferAttribute(new Float32Array(C), 3));
  wallGeoms.push(walls);
}

// window pattern: one bay x one floor per texture repeat
function windowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = "#ffffff";
  g.fillRect(0, 0, 64, 64);
  g.fillStyle = "#b9bcc2"; // soft windows: a hint of floors, not a hard grid to copy
  g.fillRect(20, 18, 24, 22); // window
  g.fillStyle = "#e6e2da";
  g.fillRect(0, 56, 64, 8); // floor band
  const t = new CanvasTexture(c);
  t.wrapS = t.wrapT = RepeatWrapping;
  t.colorSpace = SRGBColorSpace;
  return t;
}

// chunks over sea/forest can have no buildings at all
const roofGeom = roofGeoms.length ? mergeGeometries(roofGeoms) : new BufferGeometry();
const roofMesh = new Mesh(roofGeom, new MeshBasicMaterial({ map: ortho }));
const roofShadow = new Mesh(roofGeom, shadowMat);
roofShadow.position.z = 0.05;
roofShadow.receiveShadow = true;
const wallMesh = new Mesh(
  wallGeoms.length ? mergeGeometries(wallGeoms) : new BufferGeometry(),
  new MeshLambertMaterial({ map: windowTexture(), vertexColors: true, side: DoubleSide })
);
roofMesh.castShadow = wallMesh.castShadow = true;
scene.add(roofMesh, roofShadow, wallMesh);

// --- lighting -------------------------------------------------------------------
// Ground/roofs are unlit imagery; the lights only shade walls and cast shadows.
// (three >= r155 Lambert divides by PI, hence the PI factors.)
scene.add(new AmbientLight(0xffffff, Math.PI * (cfg.light?.ambient ?? 0.7)));
const sunAz = (cfg.sun?.azimuth_degrees ?? 135) * D2R;
const sunEl = (cfg.sun?.elevation_degrees ?? 55) * D2R;
const sunDir = new Vector3(Math.sin(sunAz) * Math.cos(sunEl), Math.cos(sunAz) * Math.cos(sunEl), Math.sin(sunEl));
const sun = new DirectionalLight(0xfff4e0, Math.PI * (cfg.light?.sun ?? 0.6));
sun.position.copy(sunDir).multiplyScalar(3000).add(new Vector3(RCX, RCY, 0));
sun.target.position.set(RCX, RCY, 0);
sun.castShadow = true;
const maxShadow = Math.min(cfg.shadow_map_px ?? 8192, renderer.capabilities.maxTextureSize);
sun.shadow.mapSize.set(maxShadow, maxShadow);
const half = Math.max(RECT.w, RECT.h) / 2 + 50;
Object.assign(sun.shadow.camera, { left: -half, right: half, top: half, bottom: -half, near: 1, far: 8000 });
sun.shadow.camera.updateProjectionMatrix();
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.3;
scene.add(sun, sun.target);

// --- camera ------------------------------------------------------------------------
// Same pose as the Google renderer: heading = azimuth (clockwise from north),
// pitched down by |elevation|. x = east, y = north, z = up.
const az = cfg.camera_azimuth_degrees * D2R;
const el = cfg.camera_elevation_degrees * D2R;
const forward = new Vector3(Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el));
const halfView = (TILE_PX * MPP) / 2;
const camera = new OrthographicCamera(-halfView, halfView, halfView, -halfView, 1, 10000);
camera.up.set(0, 0, 1);

// Aim at the ground point under the tile centre and back off along the view axis.
// (Sliding one camera across a big grid would push it below the ground: the image
// plane is tilted, so moving "down" the screen also lowers the camera.)
const RIGHT_H = [Math.cos(az), -Math.sin(az)];
const UP_H = [Math.sin(az) * Math.sin(-el), Math.cos(az) * Math.sin(-el)];
const DET = RIGHT_H[0] * UP_H[1] - RIGHT_H[1] * UP_H[0];
function placeCamera(tx, ty) {
  const cx = tx * TILE_PX + TILE_PX / 2 - GRID_W / 2;
  const cy = ty * TILE_PX + TILE_PX / 2 - GRID_H / 2;
  const a = cx * MPP, b = -cy * MPP; // a = g.right, b = g.up for the ground point g
  const gx = (a * UP_H[1] - b * RIGHT_H[1]) / DET;
  const gy = (b * RIGHT_H[0] - a * UP_H[0]) / DET;
  const g = new Vector3(gx, gy, 0);
  camera.position.copy(forward).multiplyScalar(-4000).add(g);
  camera.lookAt(g);
  camera.updateMatrixWorld(true);
}

// Building silhouette pass: buildings white on black (used to keep tall buildings that
// lean over the sea when the exporter repaints water).
const maskMat = new MeshBasicMaterial({ color: 0xffffff });
function silhouette() {
  const vis = [ground.visible, groundShadow.visible, roofShadow.visible];
  ground.visible = groundShadow.visible = roofShadow.visible = false;
  const sm = renderer.shadowMap.enabled;
  renderer.shadowMap.enabled = false;
  scene.overrideMaterial = maskMat;
  renderer.render(scene, camera);
  const url = renderer.domElement.toDataURL("image/png");
  scene.overrideMaterial = null;
  renderer.shadowMap.enabled = sm;
  [ground.visible, groundShadow.visible, roofShadow.visible] = vis;
  return url;
}

window.renderTile = async (tx, ty) => {
  placeCamera(tx, ty);
  renderer.render(scene, camera);
  const png = renderer.domElement.toDataURL("image/png");
  return { ok: true, png, mask: cfg.with_mask ? silhouette() : null };
};
window.SCENE_STATS = { buildings: buildings.length, shadowMap: maxShadow };
window.RENDERER_READY = true;
