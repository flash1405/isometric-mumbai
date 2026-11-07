// Frame-perfect zoom-out video of the live site (for sharing).
//   node scripts/video/zoomout.mjs [url] [outdir]  -> outdir/frame_00000.jpg ...; then encode.py
// The page's clock is ours: requestAnimationFrame / performance.now are stepped exactly 1/30 s
// per frame, and every frame waits until all map tiles have loaded before it is captured.
import { chromium } from "playwright";
import fs from "node:fs";

const URL = process.argv[2] || "https://sample.isometricmumbai.com/";
const OUT = process.argv[3] || "out/video/frames";
const W = 1920, H = 1080, FPS = 30;
const HOLD_IN = 3.2, ZOOM = 12.5, HOLD_OUT = 3.5; // seconds
fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });

const b = await chromium.launch({ channel: "chrome" });
const p = await b.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
await p.addInitScript(() => {
  let t = 0; const q = [];
  performance.now = () => t;
  window.requestAnimationFrame = (cb) => { q.push(cb); return q.length; };
  window.cancelAnimationFrame = () => {};
  window.__step = (ms) => { t += ms; const c = q.splice(0); for (const f of c) { try { f(t); } catch (e) { /* keep going */ } } };
  try { localStorage.setItem("im_intro", "1"); localStorage.setItem("im_night", "0"); } catch (e) { /* ignore */ }
});
await p.goto(URL, { waitUntil: "domcontentloaded" });
await p.waitForFunction(() => window.api && window.imTraffic, null, { polling: 100, timeout: 60000 });
await p.addStyleTag({ content: `#top,#rail,#autumn,#credits,#chip,#madeby,#writeup,#attr,.leaflet-control-container,#loading,#toast,.toast,#intro{display:none!important}
  #title{position:fixed;inset:0;display:grid;place-items:center;z-index:5000;pointer-events:none;opacity:0}
  #title div{text-align:center;color:#fff;font:800 92px/1 "Inter",system-ui,sans-serif;letter-spacing:-.02em;text-shadow:0 4px 30px rgba(0,0,0,.6)}
  #title div b{color:#f7c948} #title div small{display:block;margin-top:22px;font:600 36px system-ui,sans-serif;opacity:.9;letter-spacing:0}` });
await p.evaluate(() => {
  const d = document.createElement("div"); d.id = "title";
  d.innerHTML = "<div>isometric <b>mumbai</b><small>isometricmumbai.com</small></div>";
  document.body.appendChild(d);
});

const setup = await p.evaluate(() => {
  const { map, places, meta } = window.api;
  map.options.zoomSnap = 0;
  const md = places.find((x) => x.name === "Marine Drive");
  const [bx0, by0, bx1, by1] = meta.bounds;
  return { MAXZ: meta.maxZoom, start: [md.col + 300, md.row + 150], end: [(bx0 + bx1) / 2, (by0 + by1) / 2 - 2500],
           endZ: meta.maxZoom - Math.log2(Math.max((bx1 - bx0) / 1820, (by1 - by0) / 1000)) };
});
const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const n1 = Math.round(HOLD_IN * FPS), n2 = Math.round(ZOOM * FPS), n3 = Math.round(HOLD_OUT * FPS);
const startZ = setup.MAXZ;

async function view(cx, cy, z) {
  await p.evaluate(([cx, cy, z]) => {
    const { map, meta } = window.api;
    map.setView(map.unproject([cx, cy], meta.maxZoom), z, { animate: false });
  }, [cx, cy, z]);
}
async function settle() {
  for (let i = 0; i < 300; i++) {
    const pending = await p.evaluate(() => { window.__step(0); return document.querySelectorAll(".leaflet-tile:not(.leaflet-tile-loaded)").length; });
    if (!pending) return;
    await p.waitForTimeout(25);
  }
}
let f = 0;
async function frame(titleOpacity = 0) {
  await settle();
  await p.evaluate((o) => { window.__step(1000 / 30); document.getElementById("title").style.opacity = o; }, titleOpacity);
  await p.screenshot({ path: `${OUT}/frame_${String(f++).padStart(5, "0")}.jpg`, type: "jpeg", quality: 93 });
}

// warm up: load the start view and the traffic around it
await view(...setup.start, startZ);
for (let i = 0; i < 90; i++) { await settle(); await p.evaluate(() => window.__step(1000 / 30)); }
for (let i = 0; i < n1; i++) await frame();
for (let i = 1; i <= n2; i++) {
  const t = ease(i / n2);
  const z = startZ + (setup.endZ - startZ) * t;
  const cx = setup.start[0] + (setup.end[0] - setup.start[0]) * t, cy = setup.start[1] + (setup.end[1] - setup.start[1]) * t;
  await view(cx, cy, z);
  await frame();
  if (i % 60 === 0) console.log(`zoom ${i}/${n2}`);
}
for (let i = 0; i < n3; i++) await frame(Math.min(1, i / (FPS * 1.0)));
console.log(`${f} frames -> ${OUT}`);
await b.close();
