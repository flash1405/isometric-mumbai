import http from "node:http"; import fs from "node:fs"; import path from "node:path"; import { chromium } from "playwright";
const root = path.resolve("scripts/vehicles"), three = path.resolve("node_modules/three/build");
const srv = http.createServer((req, res) => {
  const f = req.url === "/" ? path.join(root, "vehicles.html") : req.url.startsWith("/three") ? path.join(three, req.url) : null;
  if (!f || !fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "Content-Type": f.endsWith(".html") ? "text/html" : "text/javascript" }); fs.createReadStream(f).pipe(res);
}).listen(5872);
const b = await chromium.launch({ channel: "chrome" }); const p = await b.newPage();
p.on("pageerror", (e) => console.log("pageerror", e.message));
await p.goto("http://127.0.0.1:5872/"); await p.waitForFunction(() => window.ready === true);
const sp = await p.evaluate(() => window.renderTrains());
const out = path.resolve("out/vehicles/trains"); fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(path.join(out, "index.json"), JSON.stringify(sp.map((s, i) => { fs.writeFileSync(path.join(out, `${i}.png`), Buffer.from(s.url.split(",")[1], "base64")); return { i, cls: s.cls, head: s.head }; })));
console.log(sp.length, "coach sprites"); await b.close(); srv.close();
