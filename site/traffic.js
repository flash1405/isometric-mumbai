// Simulated city life on the real network: kaali-peeli taxis, BEST buses, autos and cars
// driving on the left in lanes, Mumbai locals on the real tracks, ferries on real routes.
// Movement is simulated (no real trips); the network is OpenStreetMap via Overture.
//
// Vehicles are drawn as small isometric boxes in the same camera as the art (ground
// foreshortened by sin(elevation), heights by cos(elevation)), and erased wherever a
// building stands in front of the road, using the building silhouettes from the 3D
// renders (/traffic/occ/<bin>.png). Data: /traffic/index.json + bins from pipeline/traffic.py.
function startTraffic() {
  const api = window.api;
  const { map, meta } = api;
  const MAXZ = meta.maxZoom;
  const MIN_Z = MAXZ - 1.5; // vehicles appear when zoomed in this far (further out they are dots)
  const DENSITY = matchMedia("(pointer: coarse)").matches ? 0.6 : 1; // phones: fewer vehicles
  const TIME = 2.0; // sim speed-up so the city feels alive
  const M = 1 / meta.geo.mpp; // metres -> image px
  const KMH = (1000 / 3600) * M; // km/h -> image px / s
  const K = Math.hypot(meta.geo.up[0], meta.geo.up[1]) || Math.SQRT1_2; // ground foreshortening (sin elev)
  const V = Math.sqrt(1 - K * K); // vertical scale (cos elev)
  const ART = 1.3; // the art draws vehicles a little larger than life
  const BASE = window.IM_CONFIG.trafficBase || "/traffic/";
  const VQ = `?v=${window.IM_CONFIG.tilesVersion || "dev"}`; // versioned: long-cached, never mixed across deploys

  // ---- canvas overlay --------------------------------------------------------------
  const canvas = document.createElement("canvas");
  canvas.className = "traffic-layer";
  map.getContainer().appendChild(canvas);
  const ctx = canvas.getContext("2d");
  let dpr = 1;
  function resize() {
    const s = map.getSize();
    dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = s.x * dpr;
    canvas.height = s.y * dpr;
    canvas.style.width = s.x + "px";
    canvas.style.height = s.y + "px";
  }
  resize();
  map.on("resize", resize);
  map.on("zoomstart", () => (canvas.style.opacity = 0));
  map.on("zoomend", () => (canvas.style.opacity = 1));

  // ---- network ------------------------------------------------------------------------
  let index = null;
  const bins = new Map(); // key -> {vehicles, occ}
  const loading = new Set();
  const ends = new Map(); // endpoint key -> [{line, atStart}]
  const ROAD = {
    motorway: { kmh: 60, gap: 70, lanes: 3 }, trunk: { kmh: 48, gap: 75, lanes: 3 },
    primary: { kmh: 38, gap: 85, lanes: 2 }, secondary: { kmh: 32, gap: 120, lanes: 2 },
    tertiary: { kmh: 26, gap: 200, lanes: 1 },
  };
  const LANE = 3.3 * M;
  const ekey = (x, y) => `${Math.round(x / 4)},${Math.round(y / 4)}`;

  function decode(arr, kind) {
    const cls = index.classes[arr[0] & 31];
    const oneway = arr[0] >= 32;
    const pts = [[arr[1], arr[2]]];
    for (let i = 3; i < arr.length; i += 2) {
      const p = pts[pts.length - 1];
      pts.push([p[0] + arr[i], p[1] + arr[i + 1]]);
    }
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    // ground length (screen px undo foreshortening) for realistic speeds
    const gcum = [0];
    for (let i = 1; i < pts.length; i++) gcum.push(gcum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], (pts[i][1] - pts[i - 1][1]) / K));
    return { kind, cls, oneway, pts, cum: gcum, len: gcum[gcum.length - 1], spec: ROAD[cls] };
  }

  function at(line, d) {
    // position + unit ground direction (x right, y screen-up) at ground distance d
    const { pts, cum } = line;
    d = Math.max(0, Math.min(line.len, d));
    let lo = 0, hi = cum.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= d) lo = m; else hi = m; }
    const seg = cum[hi] - cum[lo] || 1;
    const t = (d - cum[lo]) / seg;
    const a = pts[lo], b = pts[hi];
    const gx = (b[0] - a[0]) / seg, gy = -(b[1] - a[1]) / K / seg;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, gx, gy];
  }

  const SPRITES = !location.search.includes("boxes");
  const LIB = { byCls: {}, atlas: new Image() };
  LIB.ready = fetch(`${BASE}cars/library.json${VQ}`).then((r) => r.json()).then((rows) => {
    const mk = ([cls, head, w, h, ax, ay, ox, oy, res]) => ({ head, w, h, ax, ay, ox: ox ?? w / 2, oy: oy ?? h / 2, res: res || 1 });
    for (const r of rows) (LIB.byCls[r[0]] ||= []).push(mk(r));
    LIB.all = rows.map(mk);
    return new Promise((ok) => { LIB.atlas.onload = ok; LIB.atlas.src = `${BASE}cars/library.png${VQ}`; });
  }).catch(() => {});
  function laneOff(st, d) {
    const g = st.geo;
    if (!g) return st.off;
    const f = Math.max(0, Math.min(g.L.length - 1.001, (d - g.lo) / g.step)), k = f | 0, t = f - k;
    const L = g.L[k] + (g.L[k + 1] - g.L[k]) * t, R = g.R[k] + (g.R[k + 1] - g.R[k]) * t;
    const C = (L + R) / 2;
    return st.frac >= 0 ? C + st.frac * (R - C) * 0.92 : C + st.frac * (C - L) * 0.92;
  }
  const ENTRIES = new Map(); // junction key -> lane stretches starting there
  const LANES = new Map(); // line|offset|dir -> its visible stretches (a lane broken by gaps)
  const JUNC = new Map(); // junction key -> number of lanes arriving (signals where >= 3 roads meet)
  function newCar(st, d) {
    const major = ["motorway", "trunk", "primary"].includes(st.line.cls), fast = st.line.cls === "motorway";
    const r = Math.random();
    const cls = r < 0.26 ? "taxi" : r < 0.42 ? (fast ? "white" : "auto") : r < 0.6 ? "white" : r < 0.7 ? "silver" : r < 0.76 ? "red"
      : r < 0.8 ? "blue" : r < 0.86 ? "dark" : r < 0.93 ? "bike" : r < 0.965 ? "tempo" : r < 0.985 && major ? "bus" : "white";
    const slow = cls === "bus" || cls === "tempo" ? 0.75 : cls === "auto" ? 0.8 : 1;
    return { sprite: true, st, line: st.line, off: laneOff(st, d), dir: st.dir, lo: st.lo, hi: st.hi, d, cls, seed: Math.random(),
      fade: 1, v0: st.line.spec.kmh * KMH * rnd(0.8, 1.05) * slow };
  }
  // signals: each junction alternates two phases (roughly N-S vs E-W approaches), with an all-red gap
  const PERIOD = 16;
  function green(key, head) {
    if ((JUNC.get(key) || 0) < 4) return true; // simple joins and bends: no signal
    let h = 0; for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
    const t = (performance.now() / 1000 + (h & 1023) / 64) % (2 * PERIOD);
    if (t % PERIOD > PERIOD - 2.5) return false; // all-red / amber
    const phase = t < PERIOD ? 0 : 1;
    const ns = Math.abs(Math.sin(head)) > Math.abs(Math.cos(head)) ? 0 : 1;
    return phase === ns;
  }
  const TLIB = { byCls: {}, atlas: new Image() };
  TLIB.ready = fetch(`${BASE}rails/trains.json${VQ}`).then((r) => r.json()).then((rows) => {
    for (const [cls, head, w, h, ax, ay, ox, oy, res] of rows) (TLIB.byCls[cls] ||= []).push({ head, w, h, ax, ay, ox, oy, res });
    TLIB.atlas.src = `${BASE}rails/trains.png${VQ}`;
  }).catch(() => {});
  function coachSprite(cls, head) {
    const pool = TLIB.byCls[cls] || [];
    let best = null, bd = 9;
    for (const sp of pool) { let dd = Math.abs(sp.head - head); if (dd > Math.PI) dd = 2 * Math.PI - dd; if (dd < bd) { bd = dd; best = sp; } }
    return best;
  }
  let NIGHT = false;
  function lamp(x, y, r, rgb, a) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(${rgb},${a})`); g.addColorStop(1, `rgba(${rgb},0)`);
    const op = ctx.globalCompositeOperation, f = ctx.filter; ctx.globalCompositeOperation = "lighter"; ctx.filter = "none";
    ctx.fillStyle = g; ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
    ctx.globalCompositeOperation = op; ctx.filter = f;
  }
  // ---- night: street lamps along the measured carriageways, lit windows on the buildings ----
  function streetLamps(b, o, s, size) {
    if (!b.lamps) {
      b.lamps = [];
      const seen = new Set();
      for (const st of b.stretches || []) {
        const g = st.geo;
        if (!g || seen.has(g)) continue;
        seen.add(g);
        const every = Math.max(1, Math.round((30 * M) / g.step));
        for (let k = 0; k < g.L.length; k += every) {
          const d = g.lo + k * g.step;
          const [x, y, gx, gy] = at(st.line, d);
          for (const e of [g.L[k] - 1.5 * M, g.R[k] + 1.5 * M]) b.lamps.push(x - gy * e, y - gx * e * K);
        }
      }
    }
    const L = b.lamps, r = Math.max(4, 12 * s);
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < L.length; i += 2) {
      const sx = o.x + L[i] * s, sy = o.y + L[i + 1] * s;
      if (sx < -20 || sy < -20 || sx > size.x + 20 || sy > size.y + 20) continue;
      const gr = ctx.createRadialGradient(sx, sy, 0, sx, sy, r);
      gr.addColorStop(0, "rgba(255,200,120,0.85)"); gr.addColorStop(0.3, "rgba(255,165,70,0.35)"); gr.addColorStop(1, "rgba(255,150,60,0)");
      ctx.fillStyle = gr; ctx.fillRect(sx - r, sy - r, 2 * r, 2 * r);
    }
    ctx.globalCompositeOperation = "source-over";
  }
  function litWindows(bx, by, b, o, s, size) {
    if (!b.occ) return;
    if (!b.winLayer) {
      // one light layer per area, drawn once: warm (and the odd TV-blue) windows wherever buildings stand
      const w = b.occ.width, h = b.occ.height;
      const c = document.createElement("canvas"); c.width = w; c.height = h;
      const g = c.getContext("2d", { willReadFrequently: true }); g.drawImage(b.occ, 0, 0);
      const a = g.getImageData(0, 0, w, h).data;
      const L = document.createElement("canvas"); L.width = w; L.height = h;
      const lg = L.getContext("2d"), img = lg.createImageData(w, h), d = img.data;
      let seed = (bx * 73856093) ^ (by * 19349663) || 1;
      const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        if (a[i + 3] < 200 || rnd() > 0.095) continue; // ~1 in 10 building pixels has a lit window
        const blue = rnd() < 0.08, warm = 170 + ((rnd() * 60) | 0);
        d[i] = blue ? 160 : 255; d[i + 1] = blue ? 195 : warm; d[i + 2] = blue ? 255 : 110; d[i + 3] = 150 + ((rnd() * 105) | 0);
      }
      lg.putImageData(img, 0, 0);
      b.winLayer = L;
    }
    const c = index.cell;
    ctx.globalCompositeOperation = "lighter";
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(b.winLayer, o.x + bx * c * s, o.y + by * c * s, c * s, c * s);
    ctx.globalCompositeOperation = "source-over";
  }
  function spriteFor(veh, head) {
    const hb = Math.round(head / 0.087); // 5 degree bins
    if (veh.hb === hb && veh.spr) return veh.spr;
    const pool = LIB.byCls[veh.cls] || LIB.all || [];
    let best = 9, cands = [];
    for (const sp of pool) {
      let dd = Math.abs(sp.head - head); if (dd > Math.PI) dd = 2 * Math.PI - dd;
      if (dd < best - 0.06) { best = dd; cands = [sp]; } else if (dd < best + 0.06) cands.push(sp);
    }
    if (best > 0.35 && pool !== LIB.all) { veh.cls = "white"; return spriteFor(veh, head); } // no good angle in this colour
    cands.sort((a, b) => a.ax - b.ax || a.ay - b.ay);
    veh.hb = hb;
    veh.spr = cands.length ? cands[Math.floor(veh.seed * cands.length)] : null;
    return veh.spr;
  }
  const TRAINS = !location.search.includes("notrains"); // trains run only on painted track (rail_snap.py)
  function place(line, d, off, dir) {
    // screen position at ground distance d, shifted by ground-left offset `off`; plus heading angle
    const [x, y, gx, gy] = at(line, d);
    return [x + -gy * off, y - gx * off * K, Math.atan2(gy * dir, gx * dir)];
  }
  const rnd = (a, b) => a + Math.random() * (b - a);
  const pick = (arr) => arr[(Math.random() * arr.length) | 0];
  const CAR = ["#f2f1ec", "#f2f1ec", "#c4c8ce", "#9aa0a8", "#3a3f47", "#a8302c", "#2f5590", "#e6dfcc", "#5b6b4e"];

  function vehicleType(line) {
    const r = Math.random(), big = line.cls === "motorway" || line.cls === "trunk" || line.cls === "primary";
    if (r < 0.27) return "taxi";
    if (r < 0.45) return line.cls === "motorway" ? "car" : "auto";
    if (r < 0.52 && big) return "bus";
    if (r < 0.62) return "bike";
    return "car";
  }

  function makeVehicle(line, d) {
    const type = vehicleType(line);
    const spec = line.spec;
    const lanes = spec.lanes;
    const dir = line.oneway ? 1 : Math.random() < 0.5 ? 1 : -1;
    const lane = type === "bus" || type === "auto" ? lanes - 1 : (Math.random() * lanes) | 0; // slow ones keep left
    return {
      line, d: d ?? rnd(0, line.len), dir, lane, type, fade: 0,
      v0: spec.kmh * KMH * rnd(0.8, 1.15) * (type === "bus" ? 0.75 : type === "auto" ? 0.8 : type === "bike" ? 1.05 : 1),
      colour: type === "car" ? pick(CAR) : null,
    };
  }

  function makeTrain(line, n) {
    const spec = { rail: { cars: 12, kmh: 55 }, metro: { cars: 6, kmh: 45 }, mono: { cars: 4, kmh: 30 } }[line.cls] || { cars: 12, kmh: 55 };
    const trains = [];
    for (let i = 0; i < n; i++) trains.push({ line, train: true, cars: spec.cars, d: rnd(0, line.len), dir: Math.random() < 0.5 ? 1 : -1, v0: spec.kmh * KMH, livery: pick(["wr", "cr"]), fade: 0 });
    return trains;
  }

  async function loadBin(key) {
    if (bins.has(key) || loading.has(key)) return;
    loading.add(key);
    try {
      const d = await fetch(`${BASE}${key}.json${VQ}`).then((r) => r.json());
      const vehicles = [], roads = [], byIdx = [], roadsHere = [];
      const rails = new Map(); // rail line index -> stretches where painted track exists
      if (TRAINS) {
        try {
          for (const [li, lo, step, sh] of await fetch(`${BASE}rails/${key}.json${VQ}`).then((r) => (r.ok ? r.json() : []))) {
            if (!rails.has(li)) rails.set(li, []);
            rails.get(li).push({ lo, hi: lo + step * (sh.length - 1), step, sh });
          }
        } catch (e) { /* no track here */ }
      }
      let ti = 0;
      for (const kind of ["r", "t", "f"]) {
        for (const arr of d[kind] || []) {
          const line = decode(arr, kind);
          if (kind === "r") byIdx.push(line);
          if (kind === "t") line.ti = ti++;
          if (line.len < 20) continue;
          if (kind === "r") {
            if (!line.spec) continue;
            roads.push(line);
            for (const [p, atStart] of [[line.pts[0], true], [line.pts[line.pts.length - 1], false]]) {
              const k = ekey(p[0], p[1]);
              if (!ends.has(k)) ends.set(k, []);
              ends.get(k).push({ line, atStart });
            }
            if (!SPRITES) {
              const n = Math.floor((line.len / line.spec.gap) * (line.oneway ? 0.6 : 1) * line.spec.lanes / 1.6 + Math.random());
              for (let i = 0; i < n; i++) vehicles.push(makeVehicle(line));
            }
          } else if (kind === "t") {
            const st = rails.get(line.ti);
            if (TRAINS && st && line.cls === "rail") {
              line.stretches = st;
              vehicles.push(...makeTrain(line, Math.max(1, Math.round(line.len / 5000))));
            }
          } else {
            vehicles.push({ line, ferry: true, d: rnd(0, line.len), dir: Math.random() < 0.5 ? 1 : -1, v0: 18 * KMH, fade: 0 });
          }
        }
      }
      if (SPRITES) {
        // dense traffic in every lane stretch where painted asphalt actually runs, drawn with
        // the city's own painted cars (lifted off the map by pipeline/lift_cars.py)
        try {
          await LIB.ready;
          // carriageways measured on the painted map (pipeline/road_snap.py): lanes sit inside them
          const rows = await fetch(`${BASE}cars/${key}.roads.json${VQ}`).then((r) => (r.ok ? r.json() : []));
          for (const [li, lo, step, Ls, Rs] of rows) {
            const line = byIdx[li];
            if (!line || !line.spec) continue;
            const hi = lo + step * (Ls.length - 1);
            const geo = { lo, step, L: Ls, R: Rs };
            const wM = (Rs[Rs.length >> 1] - Ls[Ls.length >> 1]) / M; // carriageway width, metres
            const total = Math.max(1, Math.min(8, Math.round(wM / 3.3)));
            const dirs = line.oneway ? [[1, total]] : [[1, Math.max(1, total >> 1)], [-1, Math.max(1, total >> 1)]];
            const A = line.pts[0], B = line.pts[line.pts.length - 1];
            const nearStart = lo < 70, nearEnd = line.len - hi < 70;
            for (const [dir, nd] of dirs) {
              for (let li2 = 0; li2 < nd; li2++) {
                // signed position across the carriageway: 0 = centre line, +1 = left edge, -1 = right edge
                const frac = line.oneway ? ((li2 + 0.5) / nd) * 2 - 1 : dir * ((li2 + 0.5) / nd);
                const st = { line, dir, lo, hi, geo, frac, off: 0 };
                st.exitKey = dir > 0 ? (nearEnd ? ekey(B[0], B[1]) : null) : nearStart ? ekey(A[0], A[1]) : null;
                st.entryKey = dir > 0 ? (nearStart ? ekey(A[0], A[1]) : null) : nearEnd ? ekey(B[0], B[1]) : null;
                if (st.entryKey) (ENTRIES.get(st.entryKey) || ENTRIES.set(st.entryKey, []).get(st.entryKey)).push(st);
                if (st.exitKey) JUNC.set(st.exitKey, (JUNC.get(st.exitKey) || 0) + 1);
                st.lk = `${li}|${dir}|${li2}|${nd}`;
                (LANES.get(st.lk) || LANES.set(st.lk, []).get(st.lk)).push(st);
                roadsHere.push(st);
                const gap = { motorway: 60, trunk: 64, primary: 72, secondary: 95, tertiary: 140 }[line.cls] || 140;
                const n = Math.floor(((hi - lo) / gap) * rnd(0.35, 0.55) * DENSITY + Math.random() * 0.6);
                for (let k = 0; k < n; k++) vehicles.push(newCar(st, rnd(lo, hi)));
              }
            }
          }
        } catch (e) { /* no lanes in this bin */ }
      }
      const bin = { vehicles, roads, occ: null, stretches: roadsHere };
      const occ = new Image();
      occ.onload = () => (bin.occ = occ);
      occ.src = `${BASE}occ/${key}.png${VQ}`;
      for (const v of vehicles) v.binRef = bin;
      bins.set(key, bin);
    } catch (e) { /* missing bin: nothing to draw there */ }
    loading.delete(key);
  }

  function visibleBins() {
    const b = map.getBounds();
    const p0 = map.project(b.getNorthWest(), MAXZ), p1 = map.project(b.getSouthEast(), MAXZ);
    const c = index.cell, out = [];
    for (let y = Math.floor(Math.min(p0.y, p1.y) / c) - 1; y <= Math.floor(Math.max(p0.y, p1.y) / c) + 1; y++)
      for (let x = Math.floor(Math.min(p0.x, p1.x) / c) - 1; x <= Math.floor(Math.max(p0.x, p1.x) / c) + 1; x++) out.push(`${x}_${y}`);
    return out.filter((k) => index.bins.includes(k));
  }

  function refresh() {
    if (!index || !enabled || map.getZoom() < MIN_Z) return;
    const want = new Set(visibleBins());
    for (const k of want) loadBin(k);
    for (const [k, b] of [...bins]) if (!want.has(k)) {
      bins.delete(k);
      for (const l of b.roads) for (const p of [l.pts[0], l.pts[l.pts.length - 1]]) {
        const key = ekey(p[0], p[1]);
        const arr = (ends.get(key) || []).filter((o) => o.line !== l);
        if (arr.length) ends.set(key, arr); else ends.delete(key);
      }
    }
  }
  map.on("moveend zoomend", refresh);

  // ---- simulation ------------------------------------------------------------------------
  const vlen = (veh) => veh.sprite ? (veh.cls === "bus" ? 52 : veh.cls === "tempo" ? 28 : veh.cls === "bike" ? 10 : 20) : (veh.type === "bus" ? 12 : veh.type === "bike" ? 2 : veh.type === "auto" ? 2.7 : 4.3) * M * ART;

  function follow(vehs) {
    // keep a gap to the vehicle ahead in the same lane: queues instead of overlaps
    const groups = new Map();
    for (const v of vehs) {
      if (v.train || v.ferry) { v.v = v.v0; continue; }
      const k = v.sprite ? v.st : v.line.pts[0][0] + "," + v.line.pts[0][1] + "," + v.dir + "," + v.lane;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(v);
    }
    for (const [k, g] of groups) {
      if (typeof k === "object") k._cars = g;
      g.sort((a, b) => b.d * b.dir - a.d * a.dir);
      for (let i = 0; i < g.length; i++) {
        const v = g[i];
        v.v = v.v0;
        if (i === 0) continue;
        const lead = g[i - 1];
        const gap = (lead.d - v.d) * v.dir - (vlen(lead) + vlen(v)) / 2;
        const safe = 2.2 * M * ART;
        if (gap < safe) v.v = Math.min(v.v0, lead.v * Math.max(0, gap / safe));
        else if (gap < safe * 4) v.v = Math.min(v.v0, lead.v + (gap - safe) * 0.6);
      }
    }
  }

  function step(veh, dt) {
    if (veh.fade < 1) veh.fade = Math.min(1, veh.fade + dt * 2);
    veh.d += veh.dir * veh.v * dt * TIME;
    if (veh.sprite) {
      veh.off = laneOff(veh.st, veh.d);
      const toEnd = veh.dir > 0 ? veh.hi - veh.d : veh.d - veh.lo;
      const st = veh.st;
      if (st.exitKey && toEnd < 70 && !veh.go) {
        const head = place(veh.line, veh.d, veh.off, veh.dir)[2];
        if (!green(st.exitKey, head)) {
          // stop line ~8 m before the junction, so approaches queue on their own roads
          if (toEnd < 30) { veh.d -= veh.dir * veh.v * dt * TIME; veh.v = 0; }
          return;
        }
        veh.go = true; // passed the light
      }
      if (toEnd > 0) { if (toEnd < 14 && !st.exitKey) veh.fade = Math.max(0, toEnd / 14); return; }
      veh.go = false;
      // the same lane carries on after a short gap: keep driving, no fade
      const nxt = (LANES.get(st.lk) || []).filter((o) => o !== st && (veh.dir > 0 ? o.lo >= st.hi - 1 && o.lo - st.hi < 260 : o.hi <= st.lo + 1 && st.lo - o.hi < 260))
        .sort((a, b) => (veh.dir > 0 ? a.lo - b.lo : b.hi - a.hi))[0];
      if (nxt) {
        // drive on through the gap (no jump): the car's range now runs on to the end of the next piece
        Object.assign(veh, { st: nxt, lo: veh.dir > 0 ? veh.d : nxt.lo, hi: veh.dir > 0 ? nxt.hi : veh.d });
        return;
      }
      // continue across the junction onto a connecting lane (straight on preferred, turns allowed)
      const opts = (st.exitKey && ENTRIES.get(st.exitKey)) || [];
      if (opts.length) {
        const h0 = place(veh.line, veh.dir > 0 ? veh.hi : veh.lo, laneOff(veh.st, veh.dir > 0 ? veh.hi : veh.lo), veh.dir)[2];
        const scored = opts.filter((o) => o.line !== veh.line).map((o) => {
          const h1 = place(o.line, o.dir > 0 ? o.lo : o.hi, laneOff(o, o.dir > 0 ? o.lo : o.hi), o.dir)[2];
          let dd = Math.abs(h1 - h0); if (dd > Math.PI) dd = 2 * Math.PI - dd;
          return [dd, o];
        }).filter(([dd]) => dd < 1.75);
        // only into a lane with room at its entry
        const free = (o) => !(o._cars || []).some((c) => Math.abs(c.d - (o.dir > 0 ? o.lo : o.hi)) < 24);
        const room = scored.filter(([, o]) => free(o));
        if (scored.length && !room.length) { veh.d -= veh.dir * veh.v * dt * TIME; veh.v = 0; veh.go = true; return; } // wait
        if (room.length) {
          room.sort((a, b) => a[0] - b[0]);
          const o = Math.random() < 0.7 ? room[0][1] : room[(Math.random() * room.length) | 0][1];
          Object.assign(veh, { st: o, line: o.line, dir: o.dir, lo: o.lo, hi: o.hi,
            d: o.dir > 0 ? o.lo : o.hi, v0: o.line.spec.kmh * KMH * rnd(0.75, 1.1) });
          return;
        }
      }
      // dead end of the visible road: reappear at the start of a random lane nearby
      const pool = (veh.binRef && veh.binRef.stretches) || [st];
      let o = pool[(Math.random() * pool.length) | 0];
      for (let t = 0; t < 12; t++) { // prefer a lane start that is off-screen
        const c = pool[(Math.random() * pool.length) | 0];
        const p0 = place(c.line, c.dir > 0 ? c.lo : c.hi, laneOff(c, c.dir > 0 ? c.lo : c.hi), c.dir);
        const pt = map.latLngToContainerPoint(map.unproject([p0[0], p0[1]], MAXZ)), sz = map.getSize();
        if (pt.x < -30 || pt.y < -30 || pt.x > sz.x + 30 || pt.y > sz.y + 30) { o = c; break; }
      }
      Object.assign(veh, { st: o, line: o.line, dir: o.dir, lo: o.lo, hi: o.hi,
        d: o.dir > 0 ? o.lo : o.hi, fade: 0, v0: o.line.spec.kmh * KMH * rnd(0.75, 1.1) });
      return;
    }
    if (veh.d >= 0 && veh.d <= veh.line.len) return;
    const atEnd = veh.d > veh.line.len;
    if (veh.sprite) {
      const p = atEnd ? veh.line.pts[veh.line.pts.length - 1] : veh.line.pts[0];
      const opts = (ends.get(ekey(p[0], p[1])) || []).filter((o) => o.line !== veh.line && (!o.line.oneway || o.atStart));
      for (const o of opts.sort(() => Math.random() - 0.5)) {
        const dir = o.atStart ? 1 : -1, d = o.atStart ? 0 : o.line.len;
        const hd = place(o.line, d, veh.off, dir)[2];
        let diff = Math.abs(hd - veh.h0); if (diff > Math.PI) diff = 2 * Math.PI - diff;
        if (diff < 0.45) { veh.line = o.line; veh.dir = dir; veh.d = d; return; }
      }
      // nowhere to go with this heading: reappear further back on its own road
      veh.d = veh.dir > 0 ? rnd(0, veh.line.len * 0.3) : veh.line.len - rnd(0, veh.line.len * 0.3);
      veh.fade = 0;
      return;
    }
    if (!veh.train && !veh.ferry) {
      const p = atEnd ? veh.line.pts[veh.line.pts.length - 1] : veh.line.pts[0];
      // continue onto a connecting road we are allowed to enter (one-ways only from their start)
      const opts = (ends.get(ekey(p[0], p[1])) || []).filter((o) => o.line !== veh.line && (!o.line.oneway || o.atStart));
      if (opts.length) {
        // prefer roads of the same or bigger class: traffic stays on the arteries
        const rank = (c) => ["tertiary", "secondary", "primary", "trunk", "motorway"].indexOf(c);
        const keep = opts.filter((o) => rank(o.line.cls) >= rank(veh.line.cls));
        const o = pick(keep.length && Math.random() < 0.8 ? keep : opts);
        veh.line = o.line;
        veh.dir = o.atStart ? 1 : -1;
        veh.d = o.atStart ? 0 : o.line.len;
        veh.lane = Math.min(veh.lane, o.line.spec.lanes - 1);
        veh.v0 = o.line.spec.kmh * KMH * rnd(0.8, 1.15) * (veh.type === "bus" ? 0.75 : veh.type === "auto" ? 0.8 : 1);
        return;
      }
      if (veh.line.oneway) { // end of a one-way with nowhere to go: reappear somewhere else on it
        veh.d = rnd(0, veh.line.len * 0.5);
        veh.fade = 0;
        return;
      }
    }
    veh.dir *= -1; // dead end / end of line: turn around
    veh.d = atEnd ? veh.line.len : 0;
  }

  // ---- drawing ------------------------------------------------------------------------------
  // A box of ground size len x wid (px, along x across), height h, base height z, centred at
  // (sx, sy) on screen, heading (fx, fy) in ground coords. Only faces turned to the camera.
  function shade(hex, f) {
    const n = parseInt(hex.slice(1), 16);
    const r = Math.min(255, ((n >> 16) & 255) * f) | 0, g = Math.min(255, ((n >> 8) & 255) * f) | 0, b = Math.min(255, (n & 255) * f) | 0;
    return `rgb(${r},${g},${b})`;
  }
  function boxPath(pts) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.fill();
  }
  function box(sx, sy, fx, fy, len, wid, h, z, top, side, s, alpha) {
    // ground (along, across) -> screen: along = (fx, -fy*K), left = (-fy, -fx*K)
    const ax = fx * s, ay = -fy * K * s, lx = -fy * s, ly = -fx * K * s;
    const hz = V * s;
    const c = [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => [sx + (a * len) / 2 * ax + (b * wid) / 2 * lx, sy + (a * len) / 2 * ay + (b * wid) / 2 * ly]);
    const lift = (p, hh) => [p[0], p[1] - hh * hz];
    ctx.globalAlpha = alpha;
    // faces: (c0,c1) front (+along), (c1,c2) right, (c2,c3) back, (c3,c0) left; normal ground y-component
    const faces = [[0, 1, fy], [1, 2, -(-fx)], [2, 3, -fy], [3, 0, -fx]];
    // right normal = -left = (fy, -fx) -> y = -fx ; left normal y = fx
    faces[1][2] = -fx; faces[3][2] = fx;
    for (const [i, j, ny] of faces) {
      if (ny >= 0) continue; // faces away from the camera (camera looks toward +y)
      ctx.fillStyle = side;
      boxPath([lift(c[i], z), lift(c[j], z), lift(c[j], z + h), lift(c[i], z + h)]);
    }
    ctx.fillStyle = top;
    boxPath(c.map((p) => lift(p, z + h)));
    ctx.globalAlpha = 1;
  }
  function shadow(sx, sy, fx, fy, len, wid, s, alpha) {
    const ax = fx * s, ay = -fy * K * s, lx = -fy * s, ly = -fx * K * s;
    ctx.globalAlpha = 0.28 * alpha;
    ctx.fillStyle = "#000";
    const o = 0.9 * s;
    boxPath([[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => [sx + o + (a * len) / 2 * ax + (b * wid) / 2 * lx, sy + o * 0.6 + (a * len) / 2 * ay + (b * wid) / 2 * ly]));
    ctx.globalAlpha = 1;
  }

  const LIVERY = { wr: ["#e9e4d8", "#7d2a35"], cr: ["#dfe2e6", "#d9a521"], metro: ["#e9eef5", "#2f6fb5"], mono: ["#f1d7e2", "#c2457a"] };
  const GLASS = "#3b4757";

  function drawVehicle(veh, sx, sy, fx, fy, s) {
    const a = veh.fade, m = M * ART;
    switch (veh.type) {
      case "taxi": // kaali-peeli: black body, yellow top
        shadow(sx, sy, fx, fy, 4.3 * m, 1.75 * m, s, a);
        box(sx, sy, fx, fy, 4.3 * m, 1.75 * m, 0.75 * m, 0.15 * m, "#1d1d1d", "#141414", s, a);
        box(sx, sy, fx, fy, 2.2 * m, 1.6 * m, 0.55 * m, 0.9 * m, "#f2c200", GLASS, s, a);
        break;
      case "auto": // yellow body, black canopy
        shadow(sx, sy, fx, fy, 2.7 * m, 1.35 * m, s, a);
        box(sx, sy, fx, fy, 2.7 * m, 1.35 * m, 0.6 * m, 0.15 * m, "#e8b818", "#a88410", s, a);
        box(sx, sy, fx, fy, 2.1 * m, 1.3 * m, 0.7 * m, 0.75 * m, "#1f1f1f", "#2a2a2a", s, a);
        break;
      case "bus": // BEST: red body, cream window band
        shadow(sx, sy, fx, fy, 12 * m, 2.5 * m, s, a);
        box(sx, sy, fx, fy, 12 * m, 2.5 * m, 1.3 * m, 0.25 * m, "#b8312c", "#8f2421", s, a);
        box(sx, sy, fx, fy, 12 * m, 2.5 * m, 0.9 * m, 1.55 * m, "#c23a33", GLASS, s, a);
        box(sx, sy, fx, fy, 11.6 * m, 2.3 * m, 0.08 * m, 2.45 * m, "#e7e1d3", "#e7e1d3", s, a);
        break;
      case "bike":
        shadow(sx, sy, fx, fy, 2 * m, 0.7 * m, s, a);
        box(sx, sy, fx, fy, 2 * m, 0.6 * m, 1.2 * m, 0.1 * m, "#2b2d33", "#1c1d21", s, a);
        break;
      default: {
        shadow(sx, sy, fx, fy, 4.3 * m, 1.75 * m, s, a);
        box(sx, sy, fx, fy, 4.3 * m, 1.75 * m, 0.75 * m, 0.15 * m, veh.colour, shade(veh.colour, 0.72), s, a);
        box(sx, sy, fx, fy, 2.2 * m, 1.6 * m, 0.55 * m, 0.9 * m, shade(veh.colour, 0.95), GLASS, s, a);
      }
    }
  }

  function draw(vehs, o, s, W, H) {
    for (const veh of vehs) {
      if (veh.train) {
        // Mumbai local: 12 coaches (3D-rendered sprites), only where painted track exists
        if (!TLIB.atlas.complete) continue;
        const carL = 21 * M;
        for (let c = veh.cars - 1; c >= 0; c--) {
          const d = veh.d - veh.dir * c * (carL + 0.6 * M);
          // the track stretch under this coach, or the nearest one within ~35 m (short gaps)
          let st = null, gapd = 120;
          for (const q of veh.line.stretches) {
            const g = d < q.lo ? q.lo - d : d > q.hi ? d - q.hi : 0;
            if (g < gapd) { gapd = g; st = q; }
          }
          if (!st || d < 0 || d > veh.line.len) continue;
          const off = st.med ??= [...st.sh].sort((a, b) => a - b)[st.sh.length >> 1]; // one steady offset per stretch
          const [x, y, gx, gy] = at(veh.line, d);
          const px = x - gy * off, py = y - gx * off * K;
          const sx = o.x + px * s, sy = o.y + py * s;
          if (sx < -160 || sy < -160 || sx > W + 160 || sy > H + 160) continue;
          const sp = coachSprite(veh.livery + (c % 4 === 1 ? "_m" : ""), Math.atan2(gy * veh.dir, gx * veh.dir));
          if (!sp) continue;
          const kk = s / sp.res;
          ctx.globalAlpha = veh.fade;
          ctx.imageSmoothingEnabled = s * dpr < 1.9;
          ctx.drawImage(TLIB.atlas, sp.ax, sp.ay, sp.w, sp.h, Math.round((sx - sp.ox * kk) * 2) / 2, Math.round((sy - sp.oy * kk) * 2) / 2, sp.w * kk, sp.h * kk);
          if (NIGHT) { // lit windows along the coach
            const ux = gx * veh.dir, uy = -gy * veh.dir * K;
            for (const t of [-0.35, -0.12, 0.12, 0.35]) lamp(sx + ux * t * carL * s, sy + uy * t * carL * s, 5 * s, "255,210,130", 0.22);
          }
          ctx.globalAlpha = 1;
        }
        continue;
      }
      if (veh.sprite) {
        if (!LIB.atlas.complete) continue;
        const p = place(veh.line, veh.d, veh.off, veh.dir);
        const sp = spriteFor(veh, p[2]);
        if (!sp) continue;
        const sx = o.x + p[0] * s, sy = o.y + p[1] * s;
        if (sx < -40 || sy < -40 || sx > W + 40 || sy > H + 40 || veh.fade <= 0.02) continue;
        ctx.globalAlpha = veh.fade;
        ctx.imageSmoothingEnabled = s * dpr < 1.9;
        const k = s / sp.res; // 2x sprites drawn at half size: full detail on retina screens
        ctx.drawImage(LIB.atlas, sp.ax, sp.ay, sp.w, sp.h, Math.round((sx - sp.ox * k) * 2) / 2, Math.round((sy - sp.oy * k) * 2) / 2, sp.w * k, sp.h * k);
        if (NIGHT) {
          // screen direction of travel: ground (cos h, sin h) -> screen (cos h, -sin h * K)
          const fx = Math.cos(p[2]), fy = -Math.sin(p[2]) * K, half = (veh.cls === "bus" ? 6 : 2.3) * M * 1.3 * s;
          lamp(sx + fx * half, sy + fy * half, 7 * s, "255,236,180", 0.55 * veh.fade);
          lamp(sx - fx * half, sy - fy * half, 3.5 * s, "255,60,40", 0.5 * veh.fade);
        }
        ctx.globalAlpha = 1;
        continue;
      }
      const [x, y, gx0, gy0] = at(veh.line, veh.d);
      const gx = gx0 * veh.dir, gy = gy0 * veh.dir;
      let sx = o.x + x * s, sy = o.y + y * s;
      if (!veh.ferry) {
        // drive on the left: offset toward the left of travel (ground left = (-gy, gx))
        const n = veh.line.spec.lanes;
        const off = veh.line.oneway ? (veh.lane - (n - 1) / 2) * LANE : (veh.lane + 0.5) * LANE;
        const lx = -gy * off, ly = gx * off; // ground
        sx += lx * s;
        sy += -ly * K * s;
      }
      if (sx < -60 || sy < -60 || sx > W + 60 || sy > H + 60) continue;
      if (veh.ferry) {
        shadow(sx, sy, gx, gy, 32 * M, 8 * M, s, veh.fade);
        box(sx, sy, gx, gy, 32 * M, 8 * M, 2 * M, 0, "#f3f1ea", "#2c5d8f", s, veh.fade);
        box(sx, sy, gx, gy, 16 * M, 6 * M, 2.2 * M, 2 * M, "#f3f1ea", "#d6d2c6", s, veh.fade);
      } else drawVehicle(veh, sx, sy, gx, gy, s);
    }
  }

  let last = performance.now(), enabled = true;
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const z = map.getZoom();
    if (enabled && z >= MIN_Z && !document.hidden) {
      const s = Math.pow(2, z - MAXZ);
      NIGHT = document.body.classList.contains("night");
      ctx.filter = NIGHT ? "brightness(0.5) saturate(0.8)" : "none"; // vehicles dim with the city; their lamps stay bright
      const o = map.latLngToContainerPoint(map.unproject([0, 0], MAXZ));
      const size = map.getSize();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // only the map cells actually on screen are simulated and drawn (others wait, preloaded)
      const c = index.cell, m = 600;
      const vx0 = (-o.x) / s - m, vy0 = (-o.y) / s - m, vx1 = (size.x - o.x) / s + m, vy1 = (size.y - o.y) / s + m;
      const active = [];
      for (const [k, b] of bins) {
        const [bx, by] = k.split("_").map(Number);
        if (bx * c < vx1 && (bx + 1) * c > vx0 && by * c < vy1 && (by + 1) * c > vy0) active.push([bx, by, b]);
      }
      const all = [];
      for (const [, , b] of active) for (const v of b.vehicles) all.push(v);
      follow(all);
      for (const v of all) step(v, dt);
      // far (top of screen) first so nearer vehicles overlap them
      for (const v of all) v._y = v.sprite ? place(v.line, v.d, v.off, v.dir)[1] : at(v.line, v.d)[1];
      all.sort((a, b) => a._y - b._y);
      draw(all, o, s, size.x, size.y);
      // erase vehicles wherever a building stands in front of the road
      ctx.filter = "none";
      if (NIGHT) for (const [bx, by, b] of active) streetLamps(b, o, s, size);
      const dbg = location.search.includes("occdebug");
      ctx.globalCompositeOperation = dbg ? "source-over" : "destination-out";
      if (dbg) ctx.globalAlpha = 0.45;
      ctx.imageSmoothingEnabled = true;
      for (const [bx, by, b] of active) {
        if (b.occ) ctx.drawImage(b.occ, o.x + bx * c * s, o.y + by * c * s, c * s, c * s);
      }
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      if (NIGHT) for (const [bx, by, b] of active) litWindows(bx, by, b, o, s, size);
    }
    requestAnimationFrame(frame);
  }

  fetch(`${BASE}index.json${VQ}`).then((r) => r.json()).then((idx) => { index = idx; refresh(); requestAnimationFrame(frame); });

  // toggle (T)
  window.imTraffic = {
    toggle() { enabled = !enabled; if (enabled) refresh(); return enabled; },
    get enabled() { return enabled; },
    pickFollow() {
      const c = map.getSize().divideBy(2), all = [];
      for (const b of bins.values()) for (const v of b.vehicles) all.push(v);
      const scr = (v) => {
        const o = map.latLngToContainerPoint(map.unproject([0, 0], MAXZ)), s = Math.pow(2, map.getZoom() - MAXZ);
        let x, y;
        if (v.train) {
          const st = v.line.stretches && v.line.stretches.find((q) => v.d >= q.lo && v.d <= q.hi);
          if (!st) return null;
          [x, y] = at(v.line, v.d);
        } else if (v.sprite) [x, y] = place(v.line, v.d, v.off, v.dir); else return null;
        return { x: o.x + x * s, y: o.y + y * s };
      };
      const near = (v) => { const p = scr(v); return p ? Math.hypot(p.x - c.x, p.y - c.y) : 1e9; };
      const pick = (f) => all.filter(f).map((v) => [near(v), v]).filter(([d]) => d < 2500).sort((a, b) => a[0] - b[0])[0];
      const hit = pick((v) => v.train) || pick((v) => v.sprite && v.cls === "taxi") || pick((v) => v.sprite);
      if (!hit) return null;
      const v = hit[1];
      return { kind: v.train ? "train" : v.cls === "taxi" ? "kaali-peeli taxi" : v.cls === "auto" ? "rickshaw" : "car", pos: () => scr(v) };
    },
    get stats() { let n = 0; for (const b of bins.values()) n += b.vehicles.length; return { bins: bins.size, vehicles: n, index: !!index }; },
  };
}
if (window.api) startTraffic();
else addEventListener("im-ready", startTraffic, { once: true });
