"""
Ideas 1, 2, and 6 from Fable's exploration-tooling brainstorm (2026-07-09), built into a single
page: an interactive UMAP scatter of the full embedding space (real training images as gold
stars, every generated image as a dot), a generation slider/play control that ghosts earlier
generations to show whether the search is finding new territory or re-treading old ground, and
a "nearest real image" bar chart (mode-collapse check: if the population only ever anchors to a
handful of the 31 real training images, faithfulness is being measured against a sliver of the
style, not the whole thing).

Depends on solution_map.py's compute_data(), which does the actual DINOv2 re-embedding and UMAP
fit; this module only lays out the already-computed points. compute_data(sweep_dir, deps) takes
solution_map's result via `deps["solution-map"]`, served live by curation_server.py through
LiveCache's depends_on=["solution-map"] mechanism, not a standalone build step.
"""
from collections import Counter

from clawmarks.shared_ui import (
    BTN_CSS,
    DARK_TOKENS,
    DINO_TIP,
    INFOTIP_CSS,
    MOBILE_BASE_CSS,
    TOPNAV_CSS,
    info_btn,
    json_script,
    nav_bar_html,
)


def compute_data(sweep_dir, deps):
    solution_map_data = deps["solution-map"]["solution_map_data"]
    points = solution_map_data["points"]
    real_points = solution_map_data["real_points"]
    max_gen = max((p["gen"] for p in points), default=0)
    real_anchor_counts = Counter(p["nearest_real"] for p in points)
    return {
        "points": points,
        "real_points": real_points,
        "max_gen": max_gen,
        "real_anchor_counts": sorted(real_anchor_counts.items(), key=lambda kv: -kv[1]),
    }


def render_html(data, active_expedition=None, active_leg=None, running=None):
    points = data["points"]
    real_points = data["real_points"]
    max_gen = data["max_gen"]
    faith_values = sorted(p["faith"] for p in points)
    faith_min = faith_values[0] if faith_values else 0
    faith_median = faith_values[len(faith_values) // 2] if faith_values else 0
    faith_max = faith_values[-1] if faith_values else 0

    umap_tip = info_btn(
        "UMAP takes the high-dimensional embedding (the numeric fingerprint DINOv2 assigns each "
        "image, hundreds of numbers long) and squashes it down to the 2D layout you see here, trying "
        "to keep images that were close together in that fingerprint space close together on screen. "
        "Distance on this map is a rough stand-in for visual similarity: nearby dots tend to look "
        "alike, and a gap between clusters means those images don't resemble each other much, even if "
        "their faithfulness/novelty scores look similar."
    )
    mode_collapse_tip = info_btn(
        "If almost every generated image's nearest real image is the same one or two training "
        "photos, the search has collapsed onto a narrow slice of the style rather than exploring all "
        "of it. This bar chart counts, for each real training photo, how many generated images anchor "
        "to it as their closest match."
    )
    dino_tip = info_btn(DINO_TIP)
    play_tip = info_btn(
        "Play the generation history to watch the search add images over time. Earlier dots fade "
        "so the newest generation stays visible."
    )

    real_anchor_json = json_script(data["real_anchor_counts"])

    points_json = json_script(points)
    real_json = json_script(real_points)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS solution map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{DARK_TOKENS}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:20px; }}
h1 {{ font-size:18px; margin:0 0 4px; }}
h2 {{ font-size:14px; color:var(--text-dim); font-weight:600; margin:36px 0 8px; }}
p.sub {{ color:var(--text-dim); max-width:760px; font-size:13px; line-height:1.6; }}
a.navlink {{ color:#7c9eff; font-size:12.5px; text-decoration:none; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
{BTN_CSS}
#bar {{ display:flex; gap:16px; align-items:center; margin-top:16px; font-size:12.5px; color:var(--text-dim); flex-wrap:wrap; }}
#bar select, #bar input[type=range] {{ background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:6px; }}
#bar input[type=range] {{ width:260px; }}
#wrap {{ display:flex; gap:20px; margin-top:14px; flex-wrap:wrap; }}
#canvasWrap {{ position:relative; max-width:100%; }}
canvas {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; cursor:crosshair;
  max-width:100%; height:auto; display:block; }}
#panel {{ width:260px; flex-shrink:0; }}
#panel img {{ width:100%; border-radius:8px; display:none; }}
#panel .info {{ font-size:12px; color:var(--text-dim); line-height:1.7; margin-top:10px; }}
#panel .info b {{ color:var(--text); }}
#panel .realWrap {{ margin-top:10px; }}
#panel .realWrap .caption {{ font-size:11px; color:var(--text-dim); margin-top:4px; }}
#panel .realWrap img {{ border:1px solid var(--pick); }}
#mapLegend {{ position:absolute; left:12px; bottom:12px; background:rgba(11,11,13,0.82); border:1px solid var(--border);
  border-radius:6px; padding:7px 9px; color:var(--text-dim); font-size:11px; line-height:1.55; pointer-events:none; }}
button.playbtn {{ background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:4px 12px; cursor:pointer; }}
#anchorChart {{ display:flex; flex-direction:column; gap:4px; max-width:640px; }}
.abar {{ display:flex; align-items:center; gap:8px; font-size:11.5px; cursor:pointer; }}
.abar .label {{ width:170px; color:var(--text-dim); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.abar .track {{ flex:1; background:var(--panel); border-radius:3px; height:12px; overflow:hidden; }}
.abar .fill {{ background:#7c9eff; height:100%; }}
.abar.selected .label {{ color:var(--text); font-weight:600; }}
.abar .count {{ color:var(--text-dim); width:36px; text-align:right; }}
@media (max-width: 640px) {{
  #wrap {{ flex-direction:column; }}
  #panel {{ width:100%; }}
  #bar input[type=range] {{ width:150px; flex:1; }}
  .abar .label {{ width:110px; }}
}}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('map.html', active_expedition=active_expedition, active_leg=active_leg, running=running)}
<h1>Solution map{umap_tip}</h1>
<p class="sub">UMAP projection of the full DINOv2{dino_tip} embedding space: every generated image plus the
31 real training images (gold stars), not just the two faithfulness/novelty scalars. Distance
here approximates visual similarity, so clusters are genuinely similar images and empty regions
between the real-image cluster and the generated cloud are territory the search hasn't reached
in embedding space, whether or not the faithfulness/novelty grid shows it as "explored."</p>

<div id="bar">
  <label>Generation &le; <input type="range" id="genSlider" min="0" max="{max_gen}" value="{max_gen}"></label>
  <span id="genLabel"></span>
  <button class="playbtn" id="playBtn">&#9654; play</button>{play_tip}
  <label>Color by <select id="colorMode">
    <option value="gen">generation (ghosted)</option>
    <option value="type">prompt type</option>
    <option value="pick">picked</option>
  </select></label>
  <label><input type="checkbox" id="anchorFilterActive" disabled> filtering by nearest real image (click a bar below)</label>
</div>

<div id="wrap">
  <div id="canvasWrap"><canvas id="cv" width="720" height="560"></canvas>
    <div id="mapLegend">&#9733; real training photo<br>&bull; generated<br><span style="color:#f5c542">&#9679; gold dot = picked winner</span></div>
  </div>
  <div id="panel">
    <img id="panelImg">
    <div class="info" id="panelInfo">Hover or tap a point for details.</div>
    <div class="realWrap" id="realWrap" style="display:none;">
      <img id="realImg">
      <div class="caption" id="realCaption"></div>
    </div>
  </div>
</div>

<h2>Nearest real training image (mode-collapse check){mode_collapse_tip}</h2>
<p class="sub">Which of the 31 real training images each generated image anchors closest to. A
population piling onto a handful of bars means the search is faithful to a narrow slice of the
real style, not the whole training set; click a bar to highlight those images on the map above.</p>
<div id="anchorChart"></div>

<script>
// json_script() only protects this declaration from a <\\/script> breakout; it does not
// HTML-escape decoded string values. Every POINTS/REAL field written into innerHTML below
// must go through escHtml() first.
function escHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

const POINTS = {points_json};
const REAL = {real_json};
const ANCHOR_COUNTS = {real_anchor_json};
const MAX_GEN = {max_gen};
let picks = {{}};
fetch('/api/favorites').then(r => r.json()).then(favorites => {{
  picks = {{}};
  Object.keys(favorites).forEach(tag => {{ picks[tag] = true; }});
  draw();
}}).catch(() => {{ draw(); }});

const xs = POINTS.map(p => p.x).concat(REAL.map(p => p.x));
const ys = POINTS.map(p => p.y).concat(REAL.map(p => p.y));
const xMin = Math.min(...xs), xMax = Math.max(...xs);
const yMin = Math.min(...ys), yMax = Math.max(...ys);

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const W = cv.width, H = cv.height, PAD = 20;

function toPx(x, y) {{
  return [
    PAD + (x - xMin) / (xMax - xMin) * (W - 2 * PAD),
    H - PAD - (y - yMin) / (yMax - yMin) * (H - 2 * PAD),
  ];
}}

function genColor(gen, maxGenShown) {{
  if (gen === maxGenShown) return '#7c9eff';
  const age = maxGenShown - gen;
  const t = Math.min(1, age / 15);
  const alpha = 0.55 - t * 0.4;
  return `rgba(154,154,164,${{Math.max(0.08, alpha).toFixed(2)}})`;
}}
const TYPE_COLOR = {{ style: '#5ec98a', conflict: '#e0a25e' }};

let anchorFilter = null;
let pinned = null;

function draw() {{
  ctx.clearRect(0, 0, W, H);
  const genMax = parseInt(document.getElementById('genSlider').value);
  const mode = document.getElementById('colorMode').value;
  const visible = POINTS.filter(p => p.gen <= genMax);

  visible.forEach(p => {{
    const [px, py] = toPx(p.x, p.y);
    let color;
    if (anchorFilter && p.nearest_real !== anchorFilter) {{
      color = 'rgba(154,154,164,0.06)';
    }} else if (mode === 'gen') {{
      color = genColor(p.gen, genMax);
    }} else if (mode === 'type') {{
      color = TYPE_COLOR[p.prompt_type] || '#9a9aa4';
    }} else if (mode === 'pick') {{
      color = picks[p.tag] ? '#f5c542' : 'rgba(154,154,164,0.25)';
    }}
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, picks[p.tag] ? 3.2 : 2.2, 0, 2 * Math.PI);
    ctx.fill();
  }});

  REAL.forEach(r => {{
    const [px, py] = toPx(r.x, r.y);
    ctx.fillStyle = '#f5c542';
    ctx.font = '11px sans-serif';
    ctx.fillText('\\u2605', px - 5, py + 4);
  }});

  document.getElementById('genLabel').textContent = `gen 0-${{genMax}} (${{visible.length}} images)`;
}}

function nearestPoint(mx, my, radius) {{
  const genMax = parseInt(document.getElementById('genSlider').value);
  let best = null, bestD = radius || 14;
  POINTS.filter(p => p.gen <= genMax).forEach(p => {{
    const [px, py] = toPx(p.x, p.y);
    const d = Math.hypot(px - mx, py - my);
    if (d < bestD) {{ bestD = d; best = p; }}
  }});
  return best;
}}

function showInfo(p) {{
  const img = document.getElementById('panelImg');
  const info = document.getElementById('panelInfo');
  img.src = p.thumb;
  img.style.display = 'block';
  img.style.cursor = 'pointer';
  img.onclick = () => Lightbox.open(p.tag);
  info.innerHTML = `<b>${{escHtml(p.tag)}}</b><br>gen ${{p.gen}} | ${{escHtml(p.category)}}<br>`
    + `type=${{escHtml(p.prompt_type)}} | prompt=${{escHtml(p.prompt_name)}}<br>`
    + `style match to your real art's average=${{p.faith}} (range {faith_min:.2f}-{faith_max:.2f}, median {faith_median:.2f} this sweep)<br>`
    + `novelty=${{p.novelty}}<br>`
    + `closest single training photo: ${{escHtml(p.nearest_real)}} (sim ${{p.nearest_real_sim}})`
    + (picks[p.tag] ? '<br><b style="color:#f5c542">picked winner</b>' : '');

  const realWrap = document.getElementById('realWrap');
  const realImg = document.getElementById('realImg');
  const realCaption = document.getElementById('realCaption');
  mountProgressive(realImg, '/real_thumbs/' + encodeURIComponent(p.nearest_real),
    '/real/' + encodeURIComponent(p.nearest_real));
  realCaption.textContent = `Closest single training photo (sim ${{p.nearest_real_sim}})`;
  realWrap.style.display = 'block';
}}

function eventToCanvasCoords(e) {{
  const rect = cv.getBoundingClientRect();
  const t = e.touches && e.touches[0] ? e.touches[0] : e;
  const scaleX = W / rect.width, scaleY = H / rect.height;
  return [(t.clientX - rect.left) * scaleX, (t.clientY - rect.top) * scaleY];
}}
cv.addEventListener('mousemove', e => {{
  if (pinned) return;
  const [mx, my] = eventToCanvasCoords(e);
  const p = nearestPoint(mx, my);
  if (p) showInfo(p);
}});
cv.addEventListener('click', e => {{
  const [mx, my] = eventToCanvasCoords(e);
  const p = nearestPoint(mx, my);
  if (p) {{ pinned = p; showInfo(p); }} else {{ pinned = null; }}
}});
// Touch has no hover state, so a tap has to both find and pin the nearest point in one go,
// instead of relying on mousemove to populate the panel before a click ever fires.
cv.addEventListener('touchstart', e => {{
  e.preventDefault();
  const [mx, my] = eventToCanvasCoords(e);
  const p = nearestPoint(mx, my, 26);
  if (p) {{ pinned = p; showInfo(p); }}
}}, {{passive: false}});

document.getElementById('genSlider').addEventListener('input', draw);
document.getElementById('colorMode').addEventListener('input', draw);

let playing = false, playTimer = null;
document.getElementById('playBtn').addEventListener('click', () => {{
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? '\\u23f8 pause' : '\\u25b6 play';
  if (playing) {{
    const slider = document.getElementById('genSlider');
    playTimer = setInterval(() => {{
      let v = parseInt(slider.value) + 1;
      if (v > MAX_GEN) v = 0;
      slider.value = v;
      draw();
    }}, 350);
  }} else {{
    clearInterval(playTimer);
  }}
}});

const anchorChart = document.getElementById('anchorChart');
const maxCount = ANCHOR_COUNTS.length ? ANCHOR_COUNTS[0][1] : 1;
anchorChart.innerHTML = ANCHOR_COUNTS.map(([name, count]) => `
  <div class="abar" data-name="${{escHtml(name)}}">
    <div class="label">${{escHtml(name)}}</div>
    <div class="track"><div class="fill" style="width:${{(count / maxCount * 100).toFixed(1)}}%"></div></div>
    <div class="count">${{count}}</div>
  </div>`).join('');
anchorChart.querySelectorAll('.abar').forEach(el => {{
  el.addEventListener('click', () => {{
    const name = el.dataset.name;
    if (anchorFilter === name) {{
      anchorFilter = null;
      el.classList.remove('selected');
    }} else {{
      anchorChart.querySelectorAll('.abar').forEach(x => x.classList.remove('selected'));
      anchorFilter = name;
      el.classList.add('selected');
    }}
    document.getElementById('anchorFilterActive').checked = !!anchorFilter;
    draw();
  }});
}});

draw();
</script>
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    return html
