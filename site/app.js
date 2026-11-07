// isometric mumbai: map viewer (Leaflet, image-pixel CRS).
(async function () {
  const C = window.IM_CONFIG;
  const [meta, places] = await Promise.all([
    fetch("/meta.json").then((r) => r.json()),
    fetch("/places.json").then((r) => r.json()).catch(() => []),
  ]);
  const MAXZ = meta.maxZoom;
  const [bx0, by0, bx1, by1] = meta.bounds || [0, 0, meta.width, meta.height];

  const map = L.map("map", {
    crs: L.CRS.Simple,
    zoomControl: false,
    attributionControl: false,
    zoomSnap: 0.25,
    zoomDelta: 0.5,
    wheelPxPerZoomLevel: 90,
    maxZoom: MAXZ + 2,
    inertia: true,
    bounceAtZoomLimits: false,
  });
  const ll = (col, row) => map.unproject([col, row], MAXZ);
  const px = (latlng) => map.project(latlng, MAXZ);
  const bounds = L.latLngBounds(ll(bx0, by1), ll(bx1, by0));
  // fully zoomed out = the whole city, neatly framed (not the empty sea and land around it)
  const [cx0, cy0, cx1, cy1] = meta.city || meta.bounds || [bx0, by0, bx1, by1];
  const cityB = L.latLngBounds(ll(cx0, cy1), ll(cx1, cy0));
  map.setMaxBounds(cityB.pad(0.25));
  const fitZ = map.getBoundsZoom(cityB, false, [24, 24]);
  map.setMinZoom(Math.max(0, Math.floor(fitZ * 4) / 4));

  const layer = L.tileLayer(`${C.tilesBase}{z}/{x}/{y}.webp?v=${C.tilesVersion}`, {
    tileSize: meta.tileSize,
    maxNativeZoom: MAXZ,
    maxZoom: MAXZ + 2,
    minZoom: 0,
    noWrap: true,
    bounds,
    keepBuffer: 4,
    crossOrigin: "anonymous", // postcards draw the tiles into a canvas
    updateWhenZooming: false,
    updateWhenIdle: false, // keep loading while panning (Leaflet waits for idle on phones by default)
  }).addTo(map);
  layer.once("load", ready);
  setTimeout(ready, 5000);
  function ready() { document.body.classList.add("ready"); }

  // ---- URL state: /v/<zoom>/<x>/<y>, x/y = view centre in 1024px tile units ----
  function parseView() {
    const m = location.pathname.match(/^\/v\/([\d.]+)\/([\d.-]+)\/([\d.-]+)/);
    if (!m) return null;
    return { z: +m[1], c: ll(+m[2] * 1024, +m[3] * 1024) };
  }
  function writeView() {
    const p = px(map.getCenter());
    const path = `/v/${+map.getZoom().toFixed(2)}/${(p.x / 1024).toFixed(1)}/${(p.y / 1024).toFixed(1)}`;
    if (location.pathname !== path) history.replaceState(null, "", path + location.search);
  }
  const fixed = parseView();
  const opening = places.find((p) => p.name === (C.opening || "Marine Drive")) || places.find((p) => p.rank === 0);
  if (fixed) map.setView(fixed.c, fixed.z);
  else if (opening) map.setView(ll(opening.col, opening.row), MAXZ - 1.5);
  else map.fitBounds(bounds);
  map.on("moveend", writeView);
  map.on("zoomend", () => document.body.classList.toggle("pixelated", map.getZoom() > MAXZ));

  // ---- search ----
  const norm = (s) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
  places.forEach((p, i) => { p.id = i; p.key = norm(p.name); });
  const q = document.getElementById("q"), results = document.getElementById("results"), chip = document.getElementById("chip");
  let hits = [], sel = -1, pin = null;

  let extra = [], extraLoading = null;
  function loadExtra() {
    return (extraLoading ??= fetch(`/search.json?v=${C.tilesVersion}`).then((r) => (r.ok ? r.json() : []))
      .then((rows) => { extra = rows.map(([name, kind, col, row]) => ({ name, cat: "place", kind, col, row, rank: 30, key: norm(name) }));
        if (q.value) { hits = search(q.value); sel = hits.length ? 0 : -1; renderResults(); } })
      .catch(() => {}));
  }
  q.addEventListener("focus", loadExtra, { once: true });
  function search(text) {
    const t = norm(text.trim());
    if (!t) return [];
    const scored = [];
    for (const p of places.concat(extra)) {
      const i = p.key.indexOf(t);
      if (i < 0) continue;
      const wordStart = i === 0 || /[\s(,-]/.test(p.key[i - 1]);
      scored.push([(i === 0 ? 0 : wordStart ? 1 : 2) * 100 + (p.rank === 0 ? 0 : p.cat === "area" ? 10 : 20) + p.key.length / 100, p]);
    }
    return scored.sort((a, b) => a[0] - b[0]).slice(0, 8).map((s) => s[1]);
  }
  function renderResults() {
    results.innerHTML = "";
    hits.forEach((p, i) => {
      const li = document.createElement("li");
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", i === sel);
      li.innerHTML = `<span></span><span class="k">${p.kind || (p.cat === "area" ? "area" : "place")}</span>`;
      li.firstChild.textContent = p.name;
      li.onmousedown = (e) => { e.preventDefault(); go(p); };
      results.appendChild(li);
    });
    results.classList.toggle("on", hits.length > 0);
  }
  q.addEventListener("input", () => { hits = search(q.value); sel = hits.length ? 0 : -1; renderResults(); });
  q.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { sel = Math.min(hits.length - 1, sel + 1); renderResults(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); renderResults(); e.preventDefault(); }
    else if (e.key === "Enter" && hits[sel]) go(hits[sel]);
    else if (e.key === "Escape") clear();
  });
  q.addEventListener("blur", () => setTimeout(() => results.classList.remove("on"), 100));

  function go(p) {
    results.classList.remove("on");
    q.value = p.name;
    q.blur();
    const target = ll(p.col, p.row);
    map.flyTo(target, p.cat === "area" ? MAXZ - 2 : p.kind === "street" ? MAXZ - 1 : MAXZ - 0.5, { duration: 1.2 });
    if (pin) map.removeLayer(pin);
    pin = L.marker(target, { icon: L.divIcon({ className: "pin", html: "<span></span>", iconSize: [0, 0] }), keyboard: false }).addTo(map);
    chip.textContent = p.name;
    chip.classList.add("on");
  }
  function clear() {
    q.value = ""; hits = []; renderResults();
    if (pin) { map.removeLayer(pin); pin = null; }
    chip.classList.remove("on");
  }

  // ---- surprise me / share ----
  const landmarks = places.filter((p) => p.cat === "landmark");
  function surprise() { if (landmarks.length) go(landmarks[Math.floor(Math.random() * landmarks.length)]); }
  document.getElementById("rand").onclick = surprise;
  const toast = document.getElementById("toast");
  function say(t) { toast.textContent = t; toast.classList.add("on"); clearTimeout(say.t); say.t = setTimeout(() => toast.classList.remove("on"), 1800); }
  document.getElementById("share").onclick = async () => {
    writeView();
    const url = location.href, title = chip.classList.contains("on") ? `${chip.textContent} · isometric mumbai` : "isometric mumbai";
    if (navigator.share && matchMedia("(pointer: coarse)").matches) {
      try { await navigator.share({ title, url }); return; } catch (e) { /* cancelled */ }
    }
    try { await navigator.clipboard.writeText(url); say("Link copied"); } catch (e) { say(url); }
  };
  addEventListener("keydown", (e) => {
    if (document.activeElement === q) return;
    if (e.key === "/") { e.preventDefault(); q.focus(); }
    else if (e.key === "r" || e.key === "R") surprise();
    else if (e.key === "Escape") clear();
  });

  // ---- area names that fade as you zoom in ----
  // most important first, so the big names win when labels would collide
  const areas = places.filter((p) => p.cat === "area").sort((a, b) => a.rank - b.rank);
  const live = new Map();
  const Z_IN = 1.5, Z_OUT = MAXZ - 1.25, FADE = 0.75;
  // zoomed out: only the big neighbourhoods; more detail appears as you zoom in
  const maxRank = (z) => (z < 4.25 ? 0 : z < 5.25 ? 1 : 2);
  function labels() {
    const z = map.getZoom(), size = map.getSize(), taken = [], want = new Map();
    if (z >= Z_IN && z < Z_OUT + FADE) {
      const mr = maxRank(z);
      for (const p of areas) {
        if (p.rank > mr) continue;
        const pt = map.latLngToContainerPoint(ll(p.col, p.row));
        const w = p.name.length * 10 + 40, h = 34; // padded: labels keep some air between them
        const box = [pt.x - w / 2, pt.y - h / 2, pt.x + w / 2, pt.y + h / 2];
        if (box[0] < 4 || box[2] > size.x - 4 || box[1] < 80 || box[3] > size.y - 40) continue;
        if (taken.some((t) => box[0] < t[2] && box[2] > t[0] && box[1] < t[3] && box[3] > t[1])) continue;
        taken.push(box);
        want.set(p.id, Math.max(0, Math.min(1, (Z_OUT + FADE - z) / FADE)));
      }
    }
    for (const [id, m] of live) if (!want.has(id)) { map.removeLayer(m); live.delete(id); }
    for (const [id, o] of want) {
      let m = live.get(id);
      if (!m) {
        const p = places[id];
        m = L.marker(ll(p.col, p.row), { interactive: false, keyboard: false,
          icon: L.divIcon({ className: "area-label", html: "<span></span>", iconSize: [0, 0] }) }).addTo(map);
        m.getElement().firstChild.textContent = p.name;
        live.set(id, m);
      }
      m.getElement().style.setProperty("--o", o.toFixed(2));
    }
  }
  map.on("zoomend moveend", labels);
  labels();

  // ---- footer ----
  const mb = document.getElementById("madeby");
  if (C.madeBy && C.madeBy.handle) { mb.href = C.madeBy.url; mb.innerHTML = `made by <b></b>`; mb.querySelector("b").textContent = "@" + C.madeBy.handle; }
  document.getElementById("writeup").href = C.writeup;
  document.getElementById("attr").innerHTML = C.attribution;
  // ---- landmark labels when zoomed in (hand-picked famous places first) ----
  const famous = places.filter((p) => p.cat === "landmark" && p.rank === 0);
  const lmLive = new Map();
  function landmarkLabels() {
    const z = map.getZoom(), size = map.getSize(), taken = [];
    const show = z >= MAXZ - 1.5;
    const want = new Set();
    if (show) {
      for (const p of famous) {
        const pt = map.latLngToContainerPoint(ll(p.col, p.row));
        const w = p.name.length * 7 + 20, box = [pt.x - w / 2, pt.y - 34, pt.x + w / 2, pt.y - 10];
        if (box[0] < 0 || box[2] > size.x || box[1] < 70 || box[3] > size.y - 30) continue;
        if (taken.some((t) => box[0] < t[2] && box[2] > t[0] && box[1] < t[3] && box[3] > t[1])) continue;
        taken.push(box);
        want.add(p.id);
      }
    }
    for (const [id, m] of lmLive) if (!want.has(id)) { map.removeLayer(m); lmLive.delete(id); }
    for (const id of want) {
      if (lmLive.has(id)) continue;
      const p = places[id];
      const m = L.marker(ll(p.col, p.row), { interactive: false, keyboard: false,
        icon: L.divIcon({ className: "lm-label", html: "<span></span>", iconSize: [0, 0] }) }).addTo(map);
      m.getElement().firstChild.textContent = p.name;
      lmLive.set(id, m);
    }
  }
  map.on("zoomend moveend", landmarkLabels);
  landmarkLabels();

  // ---- first-visit intro ----
  const intro = document.getElementById("intro");
  let seen = false;
  try { seen = localStorage.getItem("im_intro") === "1"; } catch (e) { /* storage blocked */ }
  if (!seen && !fixed) intro.hidden = false;
  document.getElementById("intro-how").href = C.writeup;
  const closeIntro = () => { intro.hidden = true; try { localStorage.setItem("im_intro", "1"); } catch (e) { /* ignore */ } };
  document.getElementById("explore").onclick = closeIntro;
  map.on("dragstart zoomstart", () => { if (!intro.hidden) closeIntro(); });
  addEventListener("keydown", (e) => { if (e.key === "Escape" && !intro.hidden) closeIntro(); });

  window.api = { map, ll, px, places, meta };
  dispatchEvent(new Event("im-ready"));

  // traffic toggle
  const tb = document.getElementById("traffic");
  const syncT = () => tb.classList.toggle("off", !(window.imTraffic && window.imTraffic.enabled));
  tb.onclick = () => { window.imTraffic && window.imTraffic.toggle(); syncT(); say(window.imTraffic && window.imTraffic.enabled ? "Traffic on" : "Traffic off"); };
  addEventListener("keydown", (e) => {
    if (document.activeElement === q) return;
    if (e.key === "t" || e.key === "T") tb.click();
    if (e.key === "p" || e.key === "P") document.getElementById("postcard").click();
    if (e.key === "+" || e.key === "=") map.zoomIn(0.5);
    if (e.key === "-" || e.key === "_") map.zoomOut(0.5);
    if (e.key === "0") document.getElementById("home").click();
    if (e.key === "?") { const it = document.getElementById("intro"); it.hidden = !it.hidden; }
  });

  // ---- the city's name, when fully zoomed out ----
  const ctitle = document.getElementById("citytitle");
  const fadeTitle = () => ctitle.style.setProperty("--o", Math.max(0, Math.min(1, (map.getMinZoom() + 1.2 - map.getZoom()) / 0.8)).toFixed(2));
  map.on("zoomend", fadeTitle);
  fadeTitle();

  // ---- map controls ----
  document.getElementById("zin").onclick = () => map.zoomIn(0.5);
  document.getElementById("zout").onclick = () => map.zoomOut(0.5);
  document.getElementById("home").onclick = () => map.flyToBounds(cityB, { padding: [24, 24], duration: 1.1 });

  // ---- where am I: the neighbourhood under the middle of the screen ----
  const hoods = places.filter((p) => p.cat === "area" && p.rank <= 2);
  function here() {
    if (pin) return;
    const z = map.getZoom();
    if (z < 4.5) { chip.classList.remove("on"); return; }
    const c = px(map.getCenter());
    let best = null, bd = 4500 * 4500;
    for (const p of hoods) { const d = (p.col - c.x) ** 2 + (p.row - c.y) ** 2; if (d < bd) { bd = d; best = p; } }
    if (best) { chip.textContent = best.name; chip.classList.add("on"); } else chip.classList.remove("on");
  }
  map.on("moveend", here);
  here();

  // ---- postcard: the current view, framed and captioned, as an image ----
  document.getElementById("postcard").onclick = async () => {
    const size = map.getSize(), k = Math.min(2, window.devicePixelRatio || 1);
    const W = Math.round(size.x * k), H = Math.round(size.y * k), band = Math.round(86 * k);
    const cv = document.createElement("canvas"); cv.width = W; cv.height = H + band;
    const g = cv.getContext("2d");
    g.fillStyle = "#0d1220"; g.fillRect(0, 0, cv.width, cv.height);
    const box = map.getContainer().getBoundingClientRect();
    const night = document.body.classList.contains("night");
    if (night) g.filter = "url(#nightf) contrast(1.08)";
    try {
      for (const img of map.getContainer().querySelectorAll("img.leaflet-tile-loaded")) {
        const r = img.getBoundingClientRect();
        if (r.right < box.left || r.bottom < box.top || r.left > box.right || r.top > box.bottom) continue;
        g.drawImage(img, (r.left - box.left) * k, (r.top - box.top) * k, r.width * k, r.height * k);
      }
    } catch (e) { say("Postcards need the map to finish loading"); return; }
    g.filter = "none";
    const tc = map.getContainer().querySelector(".traffic-layer");
    if (tc && tc.style.opacity !== "0") g.drawImage(tc, 0, 0, W, H);
    // caption band
    g.fillStyle = "#0d1220"; g.fillRect(0, H, W, band);
    g.fillStyle = "#ffd166"; g.fillRect(0, H, W, Math.max(2, 3 * k));
    const place = pin ? chip.textContent : (chip.classList.contains("on") ? chip.textContent : "Mumbai");
    g.fillStyle = "#eef1f7"; g.font = `800 ${30 * k}px Inter, system-ui, sans-serif`; g.textBaseline = "middle";
    g.fillText(place, 24 * k, H + band / 2 - 10 * k);
    g.fillStyle = "#9aa3b8"; g.font = `500 ${15 * k}px Inter, system-ui, sans-serif`;
    g.fillText("isometric mumbai · isometricmumbai.com", 24 * k, H + band / 2 + 20 * k);
    let blob;
    try { blob = await new Promise((ok) => cv.toBlob(ok, "image/png")); } catch (e) { blob = null; }
    if (!blob) { say("Couldn't make a postcard here"); return; }
    const file = new File([blob], `mumbai-${place.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.png`, { type: "image/png" });
    writeView();
    if (navigator.canShare && navigator.canShare({ files: [file] }) && matchMedia("(pointer: coarse)").matches) {
      try { await navigator.share({ files: [file], title: `${place} · isometric mumbai`, url: location.href }); return; } catch (e) { /* cancelled */ }
    }
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = file.name;
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    say("Postcard saved");
  };

  // (night mode removed; clear any saved preference so nobody is stuck in it)
  try { localStorage.removeItem("im_night"); } catch (e) { /* ignore */ }
})();
