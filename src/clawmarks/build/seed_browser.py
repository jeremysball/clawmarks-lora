"""
Candidate seed browser: view and grow the pool of subject/texture descriptions "explore" jobs
draw from. The search driver (search/driver.py) already calls out to GPT-5.5 for fresh
subjects on plateau; this page exposes the same mechanism (via curation_server.py's
/api/seeds/generate) so the pool can be reviewed and topped up between runs, not just mid-run.

Unlike the other tool pages, this one bakes in no data at build time: the seed pool lives on the
server (the active leg's out_dir/seed_pool.json, shared with search/driver.py) and grows over
time, so the page fetches
/api/seeds live instead. Run this script once to (re)write the static shell; no rebuild needed
after that, since new seeds show up via the API without a rebuild.

Run: python3 -m clawmarks.build.seed_browser
"""
from clawmarks.shared_ui import BTN_CSS, DARK_TOKENS, INFOTIP_CSS, MOBILE_BASE_CSS, TOPNAV_CSS, info_btn, nav_bar_html


def render_html(active_expedition=None, active_leg=None, running=None):
    seeds_tip = info_btn(
        "A candidate seed is a short subject/texture description (e.g. \"empty parking garage at "
        "night, one flickering light\") that an \"explore\" job can draw when building a fresh, "
        "unrelated image, as opposed to mutating an existing one. The pool starts from a small "
        "hand-written fallback list, and grows either automatically (the search driver asks GPT-5.5 "
        "for more once novelty plateaus) or on demand from this page. Adding seeds here doesn't "
        "change any run already in progress: a run reads the pool it started with, so new seeds are "
        "picked up by the next run, not retroactively."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS candidate seeds</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{DARK_TOKENS}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--text); margin:0; padding:0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
}}
{MOBILE_BASE_CSS}
{TOPNAV_CSS}
{BTN_CSS}
{INFOTIP_CSS}
main {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size:18px; margin:0 0 6px; display:flex; align-items:center; gap:8px; }}
p.sub {{ color:var(--text-dim); font-size:13px; line-height:1.6; margin:0 0 20px; }}
#genPanel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px;
  padding:14px 16px; margin-bottom:20px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
#genPanel label {{ font-size:12.5px; color:var(--text-dim); display:flex; gap:6px; align-items:center; }}
#genPanel input[type=number] {{ width:60px; background:var(--panel-2); color:var(--text);
  border:1px solid var(--border); border-radius:6px; padding:5px 8px; font-size:12.5px; }}
#genBtn {{ background:#2a4a7c; border:1px solid #3a5a8c; color:#cddcff; border-radius:7px;
  padding:8px 16px; font-size:13px; cursor:pointer; }}
#genBtn:disabled {{ opacity:0.5; cursor:default; }}
#genStatus {{ font-size:12px; color:var(--text-dim); flex-basis:100%; min-height:1.4em; }}
#genStatus.err {{ color:#e0605e; }}
#count {{ color:var(--text-faint); font-size:12.5px; margin-bottom:10px; }}
#seedList {{ display:flex; flex-direction:column; gap:6px; }}
.seed {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:10px 14px; font-size:13.5px; line-height:1.5; display:flex; justify-content:space-between;
  align-items:center; gap:12px; }}
.seed .text {{ flex:1; }}
.seed .src {{ font-size:10.5px; color:var(--text-faint); white-space:nowrap; text-transform:uppercase;
  letter-spacing:0.03em; }}
.seed.new {{ border-color: var(--accent); background: rgba(124,158,255,0.08); }}
@media (max-width: 640px) {{
  main {{ padding:12px; }}
  #genPanel {{ padding:12px; }}
  .seed {{ flex-direction:column; align-items:flex-start; gap:4px; }}
}}
</style></head><body>

{nav_bar_html('seeds.html', active_expedition=active_expedition, active_leg=active_leg, running=running)}

<main>
<h1>Candidate seeds{seeds_tip}</h1>
<p class="sub">The subject/texture pool "explore" jobs draw fresh combinations from. View what's
already in the pool, or ask GPT-5.5 for more right now instead of waiting for the next run to
plateau and escalate on its own.</p>

<div id="genPanel">
  <label>Generate <input type="number" id="genN" value="20" min="1" max="40"> new seeds</label>
  <button id="genBtn">Generate</button>
  <div id="genStatus"></div>
</div>

<div id="count"></div>
<div id="seedList"></div>
</main>

<script>
// GPT-generated seed text is untrusted and is written into innerHTML below; it must go through
// escHtml() first, or a seed containing e.g. "<img src=x onerror=...>" executes on render.
function escHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

function fmtSource(s) {{
  return s.replace('gpt5.5', 'GPT-5.5').replace('-', ' ');
}}

function render(seeds) {{
  const entries = Object.entries(seeds).sort((a, b) => (b[1].created_at || '').localeCompare(a[1].created_at || ''));
  if (entries.length === 0) {{
    document.getElementById('count').textContent = '';
    document.getElementById('seedList').innerHTML =
      '<p class="sub" style="margin:0;">No candidate seeds yet for this dataset. Use "Generate" ' +
      'above to ask GPT-5.5 for some, or wait for a search run to top up the pool on plateau.</p>';
    return;
  }}
  document.getElementById('count').textContent = entries.length + ' candidate seeds';
  document.getElementById('seedList').innerHTML = entries.map(([text, meta]) => `
    <div class="seed" data-text="${{escHtml(text)}}">
      <div class="text">${{escHtml(text)}}</div>
      <div class="src">${{escHtml(fmtSource(meta.source || ''))}}</div>
    </div>`).join('');
}}

function load() {{
  fetch('/api/seeds').then(r => r.json()).then(render).catch(() => {{
    document.getElementById('count').textContent = 'failed to load seeds';
  }});
}}

document.getElementById('genBtn').onclick = function() {{
  const n = parseInt(document.getElementById('genN').value, 10) || 20;
  const btn = document.getElementById('genBtn');
  const status = document.getElementById('genStatus');
  btn.disabled = true;
  status.classList.remove('err');
  status.textContent = 'Asking GPT-5.5 for ' + n + ' new seeds... up to a few minutes.';

  fetch('/api/seeds/generate', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{n}})}})
    .then(r => r.json().then(data => ({{ok: r.ok, data}})))
    .then(({{ok, data}}) => {{
      btn.disabled = false;
      if (!ok || data.error) {{
        status.classList.add('err');
        status.textContent = data.error || 'generation failed';
        return;
      }}
      status.textContent = `Added ${{data.added.length}} new seeds (${{data.count}} total).`;
      fetch('/api/seeds').then(r => r.json()).then(seeds => {{
        render(seeds);
        data.added.forEach(text => {{
          const el = document.querySelector(`.seed[data-text="${{text.replace(/"/g, '&quot;')}}"]`);
          if (el) el.classList.add('new');
        }});
      }});
    }})
    .catch(e => {{
      btn.disabled = false;
      status.classList.add('err');
      status.textContent = 'request failed: ' + e;
    }});
}};

load();
</script>
<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    return html
