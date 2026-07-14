"""
Idea 8 from Fable's exploration-tooling brainstorm (2026-07-09): per-prompt-family novelty
curves across generations. Novelty is scored against the growing "already explored" set, so a
prompt family whose novelty has flattened to the noise floor is exhausted and should be retired
from the explore pool; one whose novelty still climbs is still yielding new territory. Turns
"which prompts to keep" from a feel-based call into a chart you can read off.

Run after scored_manifest.json exists: python3 -m clawmarks.build.novelty_decay
"""
import json
import re
from collections import defaultdict

from clawmarks.shared_ui import nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, INFOTIP_CSS, info_btn, json_script


def compute_data(sweep_dir):
    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    def generation_of(tag):
        m = re.match(r"(?:r2_)?gen(\d+)_", tag)
        return int(m.group(1)) if m else 0

    by_prompt = defaultdict(lambda: defaultdict(list))
    for m in manifest:
        gen = generation_of(m["tag"])
        by_prompt[m["prompt_name"]][gen].append(m["novelty"])

    series = []
    for prompt_name, gens in by_prompt.items():
        if len(gens) < 2:
            continue  # only appeared in one generation, no decay/growth to plot
        points = sorted((g, sum(v) / len(v), len(v)) for g, v in gens.items())
        total_n = sum(p[2] for p in points)
        last3 = [p[1] for p in points[-3:]]
        first3 = [p[1] for p in points[:3]]
        trend = (sum(last3) / len(last3)) - (sum(first3) / len(first3))
        series.append({
            "prompt_name": prompt_name,
            "points": [{"gen": g, "novelty": round(v, 4), "n": n} for g, v, n in points],
            "total_n": total_n,
            "trend": round(trend, 4),
        })

    series.sort(key=lambda s: s["trend"])
    return {"series": series}


def render_html(data):
    series = data["series"]

    if not series:
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS novelty decay watchlist</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; }}
body {{ background:#0b0b0d; color:#eaeaee; font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
h1 {{ font-size:18px; }}
p {{ color:#9a9aa4; max-width:640px; font-size:13px; line-height:1.7; }}
a.navlink {{ color:#7c9eff; font-size:12.5px; text-decoration:none; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
</style></head><body>
{nav_bar_html('novelty_decay.html')}
<h1>Novelty decay watchlist</h1>
<p>No prompt family in this dataset has appeared in 2+ generations yet, so there's no decay curve
to plot (placeholder page). This chart tracks each prompt's mean novelty generation over
generation, to flag prompts that have stopped yielding new territory; a single-generation seed
run has nothing to compare across.</p>
<p>Once a second generation runs against this sweep, reload this page and it will show one
sparkline per prompt family that has appeared more than once, sorted worst-trending first.</p>
</body></html>"""

    data_json = json_script(series)

    trend_tip = info_btn(
        "The trend is the average novelty of a prompt's last 3 generations minus the average of its "
        "first 3. Below -0.01 counts as declining, above +0.01 as still rising, and anything in "
        "between as flat. That 0.01 cutoff is a rough rule of thumb, not a statistically derived "
        "noise floor, so treat borderline cases as worth a second look rather than a firm verdict."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS novelty decay watchlist</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee;
  --text-dim:#9a9aa4; --up:#5ec98a; --down:#e0605e; --flat:#9a9aa4; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
h1 {{ font-size:18px; margin:0 0 4px; }}
p.sub {{ color:var(--text-dim); max-width:760px; font-size:13px; line-height:1.6; }}
a.navlink {{ color:#7c9eff; font-size:12.5px; text-decoration:none; }}
#list {{ display:flex; flex-direction:column; gap:10px; margin-top:20px; max-width:920px; }}
.row {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:10px 14px;
  display:flex; align-items:center; gap:16px; }}
.row .name {{ width:220px; font-size:13px; flex-shrink:0; }}
.row .name .n {{ color:var(--text-dim); font-size:11px; display:block; }}
.row svg {{ flex:1; height:44px; }}
.trendtag {{ font-size:11px; padding:2px 8px; border-radius:5px; width:70px; text-align:center; flex-shrink:0; }}
.trendtag.down {{ background:rgba(224,96,94,0.15); color:var(--down); }}
.trendtag.up {{ background:rgba(94,201,138,0.15); color:var(--up); }}
.trendtag.flat {{ background:rgba(154,154,164,0.15); color:var(--flat); }}
@media (max-width: 640px) {{
  .row {{ flex-wrap:wrap; gap:8px; padding:10px; }}
  .row .name {{ width:100%; }}
  .row svg {{ flex:1 1 100%; order:3; }}
}}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('novelty_decay.html')}
<h1>Novelty decay watchlist{trend_tip}</h1>
<p class="sub">Mean novelty per generation, one line per prompt family that has appeared in 2+
generations, sorted worst-trending first. A flat or falling line means that prompt has stopped
yielding new territory against the growing "already explored" set and is a candidate to retire
from the explore pool; a rising line means it's still working.</p>

<div id="list"></div>

<script>
// json_script() only protects this declaration from a </script> breakout; it does not
// HTML-escape decoded string values. Every SERIES field written into innerHTML below must
// go through escHtml() first.
function escHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

const SERIES = {data_json};
const list = document.getElementById('list');

function sparkline(points) {{
  const w = 100, h = 40, pad = 3;
  const novelties = points.map(p => p.novelty);
  const lo = Math.min(...novelties), hi = Math.max(...novelties);
  const range = (hi - lo) || 0.01;
  const xs = points.map((p, i) => pad + i * (w - 2 * pad) / Math.max(1, points.length - 1));
  const ys = points.map(p => h - pad - ((p.novelty - lo) / range) * (h - 2 * pad));
  const path = xs.map((x, i) => `${{i === 0 ? 'M' : 'L'}}${{x.toFixed(1)}},${{ys[i].toFixed(1)}}`).join(' ');
  const dots = xs.map((x, i) => `<circle cx="${{x.toFixed(1)}}" cy="${{ys[i].toFixed(1)}}" r="2" fill="#7c9eff"><title>gen ${{points[i].gen}}: novelty ${{points[i].novelty}} (n=${{points[i].n}})</title></circle>`).join('');
  return `<svg viewBox="0 0 ${{w}} ${{h}}" preserveAspectRatio="none" style="width:100%;height:100%;">
    <path d="${{path}}" fill="none" stroke="#7c9eff" stroke-width="1.5"/>${{dots}}</svg>`;
}}

list.innerHTML = SERIES.map(s => {{
  const cls = s.trend < -0.01 ? 'down' : (s.trend > 0.01 ? 'up' : 'flat');
  const label = s.trend < -0.01 ? 'declining' : (s.trend > 0.01 ? 'still rising' : 'flat');
  return `<div class="row">
    <div class="name">${{escHtml(s.prompt_name)}}<span class="n">${{s.total_n}} images, ${{s.points.length}} gens</span></div>
    ${{sparkline(s.points)}}
    <div class="trendtag ${{cls}}">${{label}}</div>
  </div>`;
}}).join('');
</script>
<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    return html
