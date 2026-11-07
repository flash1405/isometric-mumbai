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
  map.setMaxBounds(bounds.pad(0.15));
  const fitZ = map.getBoundsZoom(bounds);
  map.setMinZoom(Math.max(0, Math.floor(fitZ) - 1));

  const layer = L.tileLayer(`${C.tilesBase}{z}/{x}/{y}.webp?v=${C.tilesVersion}`, {
    tileSize: meta.tileSize,
    maxNativeZoom: MAXZ,
    maxZoom: MAXZ + 2,
    minZoom: 0,
    noWrap: true,
    bounds,
    keepBuffer: 4,
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

  function search(text) {
    const t = norm(text.trim());
    if (!t) return [];
    const scored = [];
    for (const p of places) {
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
      li.innerHTML = `<span></span><span class="k">${p.cat === "area" ? "area" : "place"}</span>`;
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
    map.flyTo(target, p.cat === "area" ? MAXZ - 2 : MAXZ - 0.5, { duration: 1.2 });
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
    if (e.key === "n" || e.key === "N") nb.click();
    if (e.key === "f" || e.key === "F") fb.click();
    if (e.key === "Escape") { stopTour(); stopFollow(); }
  });

  // ---- night ----
  const nb = document.getElementById("night");
  const setNight = (on) => { document.body.classList.toggle("night", on); nb.classList.toggle("on", on); try { localStorage.setItem("im_night", on ? "1" : "0"); } catch (e) { /* ignore */ } };
  try { if (localStorage.getItem("im_night") === "1") setNight(true); } catch (e) { /* storage blocked */ }
  nb.onclick = () => { const on = !document.body.classList.contains("night"); setNight(on); say(on ? "Night" : "Day"); };

  // ---- ride along: the camera follows a local train (or a taxi) until you take over ----
  const fb = document.getElementById("follow");
  let follow = null;
  function stopFollow() { if (!follow) return; cancelAnimationFrame(follow.raf); follow = null; fb.classList.remove("on"); }
  fb.onclick = () => {
    if (follow) { stopFollow(); return; }
    if (map.getZoom() < MAXZ - 1.5) map.setZoom(MAXZ - 0.5, { animate: false });
    const t = window.imTraffic && window.imTraffic.pickFollow();
    if (!t) { say("Nothing moving here; try a busier spot"); return; }
    say(t.kind === "train" ? "Riding a Mumbai local" : "Following a " + t.kind);
    fb.classList.add("on");
    follow = { t };
    const tick = () => {
      if (!follow) return;
      const p = follow.t.pos();
      if (p) {
        const c = map.getSize().divideBy(2);
        const dx = p.x - c.x, dy = p.y - c.y;
        if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) map.panBy([dx * 0.12, dy * 0.12], { animate: false });
      }
      follow.raf = requestAnimationFrame(tick);
    };
    tick();
  };
  map.on("dragstart", stopFollow);

  // ---- city tour ----
  const TOUR = [
    ["Gateway of India", "Built in 1924 on the Apollo Bunder waterfront, facing the harbour."],
    ["Taj Mahal Palace", "The red-domed hotel of 1903, beside the Gateway."],
    ["Chhatrapati Shivaji Maharaj Terminus (CSMT)", "Victorian Gothic terminus of the Central line, a UNESCO site."],
    ["Marine Drive", "The Queen's Necklace: 3.6 km of seafront from Nariman Point to Chowpatty."],
    ["Haji Ali Dargah", "A mosque and tomb on an islet, reached by a causeway at low tide."],
    ["Mahalaxmi Racecourse", "Turf, grandstands and the supertalls of Mahalaxmi rising behind."],
    ["Bandra-Worli Sea Link", "The cable-stayed bridge across Mahim Bay, opened in 2009."],
    ["Bandstand, Bandra", "The promenade of the stars, below Mount Mary and Land's End."],
    ["Bandra Kurla Complex", "The glass towers of Mumbai's newer business district."],
    ["Juhu Beach", "Bhel puri, sunsets and the long beach of the western suburbs."],
  ];
  const cap = document.getElementById("caption");
  let tour = null;
  function stopTour() { if (!tour) return; clearTimeout(tour.t); tour = null; cap.hidden = true; document.getElementById("tour").classList.remove("on"); }
  document.getElementById("caption-stop").onclick = stopTour;
  document.getElementById("tour").onclick = () => {
    if (tour) { stopTour(); return; }
    stopFollow();
    const stops = TOUR.map(([n, d]) => [places.find((p) => p.name === n), n.replace(/ \(.*\)$/, ""), d]).filter((x) => x[0]);
    tour = { i: 0 };
    document.getElementById("tour").classList.add("on");
    const next = () => {
      if (!tour) return;
      const [p, name, desc] = stops[tour.i % stops.length];
      tour.i++;
      cap.hidden = false; cap.querySelector("b").textContent = name; cap.querySelector("span").textContent = desc;
      map.flyTo(ll(p.col, p.row), MAXZ - 1, { duration: 3.2 });
      tour.t = setTimeout(next, 9000);
    };
    next();
  };
  map.on("dragstart", stopTour);
})();
