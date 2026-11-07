// Tiny events endpoint for the map (everything else is served as static assets).
// POST /api/e {sid, e, v, z} -> one row in D1. No cookies, no IPs, no personal data stored.
const EVENTS = new Set(["view", "search", "place", "postcard", "share", "surprise", "traffic"]);

export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    if (url.pathname === "/api/e") {
      if (req.method !== "POST") return new Response(null, { status: 405 });
      try {
        const b = await req.json();
        const e = String(b.e || "");
        if (EVENTS.has(e) && env.DB) {
          const cf = req.cf || {};
          const ref = (() => { try { return b.r ? new URL(String(b.r)).hostname.slice(0, 80) : ""; } catch (x) { return ""; } })();
          const ua = /Mobi|Android|iPhone/i.test(req.headers.get("user-agent") || "") ? "mobile" : "desktop";
          ctx.waitUntil(env.DB.prepare("INSERT INTO ev (ts, host, sid, e, v, z, country, city, ref, ua) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
            .bind(Date.now(), url.hostname, String(b.sid || "").slice(0, 16), e, String(b.v || "").slice(0, 80),
                  Number(b.z) || null, String(cf.country || ""), String(cf.city || "").slice(0, 60), ref, ua).run());
        }
      } catch (x) { /* ignore malformed */ }
      return new Response(null, { status: 204, headers: { "cache-control": "no-store" } });
    }
    return env.ASSETS.fetch(req);
  },
};
