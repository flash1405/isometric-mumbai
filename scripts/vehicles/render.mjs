// Render the vehicle sprite sheet: node scripts/vehicles/render.mjs -> out/vehicles/raw/*.png + index.json
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright";
const root = path.resolve("scripts/vehicles"), three = path.resolve("node_modules/three/build");
const srv = http.createServer((req, res) => {
  const f = req.url === "/" ? path.join(root, "vehicles.html") : req.url.startsWith("/three") ? path.join(three, req.url) : null;
  if (!f || !fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "Content-Type": f.endsWith(".html") ? "text/html" : "text/javascript" });
  fs.createReadStream(f).pipe(res);
}).listen(5871);
const b = await chromium.launch({ channel: "chrome", args: ["--use-angle=metal"] });
const p = await b.newPage();
p.on("pageerror", (e) => console.log("pageerror", e.message));
await p.goto("http://127.0.0.1:5871/");
await p.waitForFunction(() => window.ready === true);
const sprites = await p.evaluate(() => window.renderAll());
const out = path.resolve("out/vehicles/raw"); fs.mkdirSync(out, { recursive: true });
const index = sprites.map((s, i) => {
  fs.writeFileSync(path.join(out, `${i}.png`), Buffer.from(s.url.split(",")[1], "base64"));
  return { i, cls: s.cls, vi: s.vi, head: s.head };
});
fs.writeFileSync(path.join(out, "index.json"), JSON.stringify(index));
console.log(`${index.length} sprites -> ${out}`);
await b.close(); srv.close();
