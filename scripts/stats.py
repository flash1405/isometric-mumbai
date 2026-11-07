"""Map analytics report (events written by site/worker.js into D1).
  uv run python scripts/stats.py [hours=24]"""
import json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for l in (ROOT / ".env").read_text().splitlines():
  if "=" in l and not l.startswith("#"):
    k, v = l.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
H = float(sys.argv[1]) if len(sys.argv) > 1 else 24
since = f"(strftime('%s','now') - {int(H * 3600)}) * 1000"

def q(sql):
  r = subprocess.run(["npx", "-y", "wrangler@4.158.0", "d1", "execute", "isometricmumbai-analytics", "--remote", "--json", "--command", sql],
                     capture_output=True, text=True, env=dict(os.environ, CLOUDFLARE_ACCOUNT_ID=os.environ["CF_ACCOUNT_ID"]), cwd=ROOT)
  out = r.stdout[r.stdout.find("["):]  # skip any npm notices printed before the JSON
  return json.loads(out)[0]["results"]

def top(title, sql, n=10):
  rows = q(sql + f" LIMIT {n}")
  print(f"\n{title}")
  for r in rows:
    vals = list(r.values())
    print(f"  {vals[-1]:>6}  {' · '.join(str(v) for v in vals[:-1] if v)}")

w = f"WHERE ts > {since} AND host LIKE '%isometricmumbai.com'"
s = q(f"SELECT COUNT(DISTINCT sid) visitors, SUM(e='view') views, SUM(e='search') searches, SUM(e='postcard') postcards, SUM(e='share') shares, SUM(e='surprise') surprises FROM ev {w}")[0]
print(f"Last {H:g} h: {s['visitors']} visitors · {s['views']} page views · {s['searches']} searches · {s['postcards']} postcards · {s['shares']} shares · {s['surprises']} surprise-me")
top("Countries", f"SELECT country, COUNT(DISTINCT sid) n FROM ev {w} GROUP BY country ORDER BY n DESC")
top("Cities", f"SELECT city, country, COUNT(DISTINCT sid) n FROM ev {w} AND city != '' GROUP BY city, country ORDER BY n DESC")
top("Most searched", f"SELECT v, COUNT(*) n FROM ev {w} AND e='search' GROUP BY v ORDER BY n DESC", 15)
top("Most explored neighbourhoods", f"SELECT v, COUNT(DISTINCT sid) n FROM ev {w} AND e='place' GROUP BY v ORDER BY n DESC", 15)
top("Referrers", f"SELECT CASE WHEN ref = '' THEN 'direct / app' ELSE ref END r, COUNT(*) n FROM ev {w} AND e='view' GROUP BY r ORDER BY n DESC")
top("Devices", f"SELECT ua, COUNT(DISTINCT sid) n FROM ev {w} GROUP BY ua ORDER BY n DESC")
