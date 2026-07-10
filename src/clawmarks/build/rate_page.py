"""
Generates rate.html: a full-screen, swipe-driven yes/no rating page. Unlike every other
build/*.py generator, this page bakes in no per-image data at build time. It fetches
GET /api/rate/next itself and POSTs to /api/rate, both served by curation_server.py, so the page
never goes stale between rebuilds. Rebuilding only matters if this file itself changes.

Served live at /rate.html by curation_server.py.
"""
from clawmarks.shared_ui import nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, INFOTIP_CSS, info_btn


def render_html():
    rate_tip = info_btn(
        "Rating trains the preference classifier: yes/no on as many images as you can stand "
        "to look at. Yes-rated images immediately take over the search's exploit pool (the same "
        "role picking used to play); once enough ratings exist, a model trained on them takes "
        "over ranking automatically."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS rate</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee;
  --text-dim:#9a9aa4; --yes:#5ec98a; --no:#e0605e; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px;
  display:flex; flex-direction:column; align-items:center; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
h1 {{ font-size:18px; margin:0 0 4px; align-self:flex-start; }}
p.sub {{ color:var(--text-dim); max-width:640px; font-size:13px; line-height:1.6; align-self:flex-start; }}
#stage {{ margin-top:20px; width:100%; max-width:640px; display:flex; flex-direction:column; align-items:center; }}
#imgwrap {{ position:relative; max-width:100%; max-height:78vh; overflow:hidden; touch-action:none;
  display:flex; align-items:center; justify-content:center; }}
#img {{ max-width:100%; max-height:78vh; border-radius:10px; box-shadow:0 20px 60px rgba(0,0,0,0.6);
  user-select:none; -webkit-user-drag:none; }}
#swipe-overlay {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  font-size:48px; font-weight:800; letter-spacing:0.08em; opacity:0; pointer-events:none; border-radius:10px; }}
#swipe-overlay.yes {{ color:var(--yes); background:rgba(94,201,138,0.12); }}
#swipe-overlay.no {{ color:var(--no); background:rgba(224,96,94,0.12); }}
#meta {{ color:var(--text-dim); font-size:12.5px; margin-top:10px; text-align:center; }}
#count {{ color:var(--text-dim); font-size:12px; margin-top:14px; }}
#done {{ color:var(--text-dim); font-size:14px; margin-top:40px; text-align:center; }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('rate.html')}
<h1>Rate{rate_tip}</h1>
<p class="sub">Swipe left for no, right for yes (or &larr;/&rarr;, n/y on a keyboard). Double-click
or double-tap an image to zoom to full resolution; drag to look around while zoomed.</p>

<div id="stage">
  <div id="imgwrap">
    <img id="img" style="display:none;">
    <div id="swipe-overlay"></div>
  </div>
  <div id="meta"></div>
  <div id="done" style="display:none;">Nothing left to rate right now &mdash; every image in the pool has been rated or favorited.</div>
</div>
<div id="count"></div>

<script>
let current = null;
let ratedThisSession = 0;

function loadNext() {{
  fetch('/api/rate/next').then(r => r.json()).then(d => {{
    if (d.done) {{
      current = null;
      document.getElementById('img').style.display = 'none';
      document.getElementById('done').style.display = 'block';
      return;
    }}
    current = d;
    const img = document.getElementById('img');
    img.style.transform = 'translateX(0px)';
    img.src = d.file;
    img.style.display = 'block';
    document.getElementById('swipe-overlay').style.opacity = 0;
    document.getElementById('meta').textContent =
      `${{d.prompt_name}} | faith=${{d.faith}} novelty=${{d.novelty}}`;
  }});
}}

function rate(label) {{
  if (!current) return;
  const tag = current.tag;
  fetch('/api/rate', {{method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{tag, label}})}})
    .then(r => r.json())
    .then(() => {{
      ratedThisSession++;
      document.getElementById('count').textContent = `${{ratedThisSession}} rated this session`;
      loadNext();
    }});
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft' || e.key === 'n' || e.key === 'N') rate('no');
  if (e.key === 'ArrowRight' || e.key === 'y' || e.key === 'Y') rate('yes');
}});

loadNext();
</script>
<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    return html
