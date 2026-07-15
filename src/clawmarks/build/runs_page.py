"""
Generates runs.html: launch, monitor, and stop an overnight search run (search/driver.py)
from the browser instead of an SSH session, per
docs/superpowers/specs/2026-07-12-overnight-search-launch-design.md. Bakes in no data at
render time; every dynamic piece is a live fetch against curation_server.py:
  GET  /api/expeditions                          -> [{name, legs: [...]}] to populate the pickers
  GET  /api/searchrun/status                      -> {running, pid, expedition, leg, started_at,
                                                        out_dir} | {running: false}
  GET  /api/searchrun/report?expedition=&leg=     -> novelty trajectory, plateau count, spend,
                                                        pick rate, explore/exploit split
  POST /api/searchrun/launch  body: {expedition, leg} -> backs up out_dir, verifies,
                                                           balance-checks, launches driver.py
                                                           detached
  POST /api/searchrun/stop            -> SIGTERM, SIGKILL after a grace period

Served live at /runs.html by curation_server.py.
"""
from clawmarks.shared_ui import nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS


def render_html():
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS search runs</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee;
  --text-dim:#9a9aa4; --up:#5ec98a; --down:#e0605e; --accent:#7c9eff; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
h1 {{ font-size:18px; margin:0 0 4px; }}
p.sub {{ color:var(--text-dim); max-width:760px; font-size:13px; line-height:1.6; }}
a.navlink {{ color:var(--accent); font-size:12.5px; text-decoration:none; }}
.panel {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:16px; margin-top:16px; max-width:760px; }}
.row {{ display:flex; align-items:center; gap:12px; margin-bottom:10px; }}
select, button {{ font-size:13px; padding:6px 12px; border-radius:6px; border:1px solid var(--border);
  background:#1f1f24; color:var(--text); }}
button {{ cursor:pointer; }}
button.primary {{ background:var(--accent); color:#0b0b0d; border-color:var(--accent); font-weight:600; }}
button.danger {{ background:var(--down); color:#0b0b0d; border-color:var(--down); font-weight:600; }}
button:disabled {{ opacity:0.4; cursor:not-allowed; }}
#launchError {{ color:var(--down); font-size:12.5px; margin-top:8px; }}
.statgrid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(140px, 1fr)); gap:10px; margin-top:10px; }}
.stat {{ background:#1f1f24; border-radius:6px; padding:10px 12px; }}
.stat .label {{ color:var(--text-dim); font-size:11px; text-transform:uppercase; letter-spacing:0.03em; }}
.stat .value {{ font-size:18px; margin-top:2px; }}
#sparkwrap {{ margin-top:14px; }}
#spark {{ width:100%; height:60px; }}
.catrow {{ display:flex; justify-content:space-between; font-size:12.5px; padding:4px 0;
  border-bottom:1px solid var(--border); }}
.catrow:last-child {{ border-bottom:none; }}
.idle {{ color:var(--text-dim); }}
.live {{ color:var(--up); }}
</style></head><body>

{nav_bar_html('runs.html')}
<h1>Search runs</h1>
<p class="sub">Launch an overnight search round from here instead of SSHing in. Every launch
backs up the round's out_dir first and refuses to start if that backup can't be verified by file
count, checks the RunPod balance floor once up front, and refuses a second launch while one is
already running.</p>

<div class="panel">
  <div class="row">
    <label for="expedition">Expedition</label>
    <select id="expedition"></select>
    <label for="leg">Leg</label>
    <select id="leg"></select>
    <button id="launchBtn" class="primary">Back up and launch</button>
    <button id="stopBtn" class="danger" disabled>Stop</button>
  </div>
  <div id="launchError"></div>
  <div class="row"><span id="statusLine" class="idle">Not running.</span></div>
</div>

<div class="panel">
  <h2 style="font-size:14px;margin:0 0 4px;">Per-run report</h2>
  <div class="statgrid">
    <div class="stat"><div class="label">Generation</div><div class="value" id="statGen">-</div></div>
    <div class="stat"><div class="label">Plateau count</div><div class="value" id="statPlateau">-</div></div>
    <div class="stat"><div class="label">Total images</div><div class="value" id="statImages">-</div></div>
    <div class="stat"><div class="label">Spend</div><div class="value" id="statSpend">-</div></div>
  </div>
  <div id="sparkwrap"><div class="label" style="color:var(--text-dim);font-size:11px;">Novelty trajectory</div>
    <svg id="spark" viewBox="0 0 100 40" preserveAspectRatio="none"></svg>
  </div>
  <div id="categoryBreakdown" style="margin-top:10px;"></div>
</div>

<script>
function escHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

const launchBtn = document.getElementById('launchBtn');
const stopBtn = document.getElementById('stopBtn');
const expeditionSel = document.getElementById('expedition');
const legSel = document.getElementById('leg');
const statusLine = document.getElementById('statusLine');
const launchError = document.getElementById('launchError');
let expeditionsData = [];

function populateLegs() {{
  const exp = expeditionsData.find(e => e.name === expeditionSel.value);
  const legs = exp ? exp.legs : [];
  legSel.innerHTML = legs.map(l => `<option value="${{escHtml(l)}}">${{escHtml(l)}}</option>`).join('');
}}

function loadExpeditions() {{
  return fetch('/api/expeditions').then(r => r.json()).then(d => {{
    expeditionsData = d.expeditions || [];
    expeditionSel.innerHTML = expeditionsData.map(e =>
      `<option value="${{escHtml(e.name)}}">${{escHtml(e.name)}}</option>`).join('');
    populateLegs();
  }});
}}

expeditionSel.addEventListener('change', () => {{ populateLegs(); refreshReport(); }});

function renderSpark(points) {{
  const svg = document.getElementById('spark');
  if (!points.length) {{ svg.innerHTML = ''; return; }}
  const w = 100, h = 40, pad = 3;
  const lo = Math.min(...points), hi = Math.max(...points);
  const range = (hi - lo) || 0.01;
  const xs = points.map((_, i) => pad + i * (w - 2 * pad) / Math.max(1, points.length - 1));
  const ys = points.map(v => h - pad - ((v - lo) / range) * (h - 2 * pad));
  const path = xs.map((x, i) => `${{i === 0 ? 'M' : 'L'}}${{x.toFixed(1)}},${{ys[i].toFixed(1)}}`).join(' ');
  svg.innerHTML = `<path d="${{path}}" fill="none" stroke="#7c9eff" stroke-width="1.5"/>`;
}}

function refreshReport() {{
  if (!expeditionSel.value || !legSel.value) return;
  const params = new URLSearchParams({{expedition: expeditionSel.value, leg: legSel.value}});
  fetch('/api/searchrun/report?' + params).then(r => r.json()).then(d => {{
    document.getElementById('statGen').textContent = d.generation;
    document.getElementById('statPlateau').textContent = d.plateau_count;
    document.getElementById('statImages').textContent = d.total_images;
    document.getElementById('statSpend').textContent =
      (typeof d.spend === 'number') ? ('$' + d.spend.toFixed(2)) : (d.start_balance != null ? ('started $' + d.start_balance.toFixed(2)) : '-');
    renderSpark(d.novelty_trajectory);
    const cats = Object.keys(d.explore_exploit_split);
    document.getElementById('categoryBreakdown').innerHTML = cats.length ? cats.map(cat =>
      `<div class="catrow"><span>${{escHtml(cat)}}</span><span>${{escHtml(d.explore_exploit_split[cat])}} images, ` +
      `${{escHtml((d.pick_rate_by_category[cat] * 100).toFixed(0))}}% picked</span></div>`
    ).join('') : '<span style="color:var(--text-dim);font-size:12.5px;">No images scored for this leg yet.</span>';
  }});
}}

function refreshStatus() {{
  fetch('/api/searchrun/status').then(r => r.json()).then(d => {{
    if (d.running) {{
      statusLine.className = 'live';
      statusLine.textContent = `Running ${{escHtml(d.expedition)}}/${{escHtml(d.leg)}} (pid ${{d.pid}}), started ${{new Date(d.started_at * 1000).toLocaleString()}}.`;
      launchBtn.disabled = true;
      stopBtn.disabled = false;
    }} else {{
      statusLine.className = 'idle';
      statusLine.textContent = 'Not running.';
      launchBtn.disabled = false;
      stopBtn.disabled = true;
    }}
  }});
}}

launchBtn.addEventListener('click', () => {{
  launchError.textContent = '';
  if (!expeditionSel.value || !legSel.value) {{
    launchError.textContent = 'pick an expedition and leg first';
    return;
  }}
  launchBtn.disabled = true;
  launchBtn.textContent = 'Backing up and launching...';
  fetch('/api/searchrun/launch', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{expedition: expeditionSel.value, leg: legSel.value}}),
  }}).then(async r => {{
    const d = await r.json();
    if (!r.ok) {{ launchError.textContent = d.error || 'launch failed'; }}
    launchBtn.textContent = 'Back up and launch';
    refreshStatus();
  }}).catch(e => {{
    launchError.textContent = String(e);
    launchBtn.textContent = 'Back up and launch';
    launchBtn.disabled = false;
  }});
}});

stopBtn.addEventListener('click', () => {{
  stopBtn.disabled = true;
  fetch('/api/searchrun/stop', {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: '{{}}'}})
    .then(() => refreshStatus());
}});

legSel.addEventListener('change', refreshReport);

loadExpeditions().then(() => {{ refreshStatus(); refreshReport(); }});
setInterval(() => {{ refreshStatus(); refreshReport(); }}, 4000);
</script>
<script src="scrollnav.js"></script>
</body></html>"""
