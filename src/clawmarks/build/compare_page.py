"""
Generates compare.html: a head-to-head comparison page. Shows two images side by side; tapping
or clicking one picks it as the winner, feeding search/preference_pairwise_model.py. The old
yes/no rating interface is gone. This page bakes in no per-image data at build time: it fetches
GET /api/compare/next itself and POSTs to
/api/compare, both served by curation_server.py, so the page never goes stale between rebuilds.

Served live at /compare.html by curation_server.py.
"""
from clawmarks.shared_ui import nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, INFOTIP_CSS, info_btn


def render_html():
    compare_tip = info_btn(
        "Trains the preference model by comparison: pick whichever of the two images you "
        "prefer, as many times as you can stand. Early comparisons are sampled to spread across "
        "the faithfulness/novelty grid; once 50+ comparisons exist, the model itself starts "
        "picking which pairs are most useful to compare next."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS compare</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee;
  --text-dim:#9a9aa4; --pick:#7c9eff; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px;
  display:flex; flex-direction:column; align-items:center; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
h1 {{ font-size:18px; margin:0 0 4px; align-self:flex-start; }}
p.sub {{ color:var(--text-dim); max-width:640px; font-size:13px; line-height:1.6; align-self:flex-start; }}
#stage {{ margin-top:20px; width:100%; max-width:1100px; display:flex; flex-direction:column; align-items:center; }}
#pair {{ display:flex; gap:16px; width:100%; justify-content:center; flex-wrap:wrap; }}
.pane {{ position:relative; flex:1 1 420px; max-width:520px; cursor:pointer; border-radius:10px;
  border:2px solid transparent; transition:border-color .12s ease; }}
.pane:hover {{ border-color:var(--pick); }}
.pane img {{ display:block; width:100%; max-height:70vh; object-fit:contain; border-radius:8px;
  background:var(--panel); user-select:none; -webkit-user-drag:none; }}
.zoom-icon {{ position:absolute; top:8px; right:8px; width:30px; height:30px; border-radius:50%;
  background:rgba(20,20,24,0.7); border:1px solid rgba(255,255,255,0.2); color:#eaeaee;
  font-size:15px; display:flex; align-items:center; justify-content:center; cursor:zoom-in; z-index:2; }}
.zoom-icon:hover {{ background:rgba(124,158,255,0.35); }}
#meta {{ color:var(--text-dim); font-size:12.5px; margin-top:10px; text-align:center; display:flex; gap:24px; }}
#count {{ color:var(--text-dim); font-size:12px; margin-top:14px; }}
#done {{ color:var(--text-dim); font-size:14px; margin-top:40px; text-align:center; }}
#zoom-overlay {{ position:fixed; inset:0; background:rgba(8,8,10,0.94); backdrop-filter:blur(6px);
  display:none; align-items:center; justify-content:center; z-index:1000; cursor:grab; overflow:hidden; }}
#zoom-overlay.open {{ display:flex; }}
#zoom-overlay img {{ max-width:none; max-height:none; user-select:none; -webkit-user-drag:none; }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('compare.html')}
<h1>Compare{compare_tip}</h1>
<p class="sub">Tap or click the image you prefer (or press &larr;/&rarr;). Tap the magnifier in
a corner to inspect that image at full resolution; tap again to close.</p>

<div id="stage">
  <div id="pair">
    <div class="pane" id="pane1" data-side="1">
      <img id="img1" style="display:none;">
      <div class="zoom-icon" id="zoom1">&#128269;</div>
    </div>
    <div class="pane" id="pane2" data-side="2">
      <img id="img2" style="display:none;">
      <div class="zoom-icon" id="zoom2">&#128269;</div>
    </div>
  </div>
  <div id="meta"></div>
  <div id="done" style="display:none;">Nothing left to compare right now &mdash; the pool doesn't have enough images left to form a new pair.</div>
</div>
<div id="count"></div>

<div id="zoom-overlay">
  <img id="zoom-img">
</div>

<script>
let current = null;
let comparedThisSession = 0;

function loadNext() {{
  fetch('/api/compare/next').then(r => {{
    if (!r.ok) throw new Error('Could not load the next comparison');
    return r.json();
  }}).then(d => {{
    if (d.done) {{
      current = null;
      document.getElementById('pair').style.display = 'none';
      document.getElementById('done').style.display = 'block';
      return;
    }}
    current = d;
    document.getElementById('pair').style.display = 'flex';
    document.getElementById('done').style.display = 'none';
    const img1 = document.getElementById('img1');
    const img2 = document.getElementById('img2');
    img1.src = d.img1.file; img1.style.display = 'block';
    img2.src = d.img2.file; img2.style.display = 'block';
    document.getElementById('meta').innerHTML =
      `<span>${{d.img1.prompt_name}} | faith=${{d.img1.faith}} novelty=${{d.img1.novelty}}</span>` +
      `<span>${{d.img2.prompt_name}} | faith=${{d.img2.faith}} novelty=${{d.img2.novelty}}</span>`;
  }}).catch(() => {{
    document.getElementById('done').textContent =
      "Couldn't reach the server. Check your connection and try again.";
    document.getElementById('done').style.display = 'block';
  }});
}}

function choose(side) {{
  if (!current) return;
  const winner = side === 1 ? current.img1.tag : current.img2.tag;
  const loser = side === 1 ? current.img2.tag : current.img1.tag;
  fetch('/api/compare', {{method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{winner, loser}})}})
    .then(r => {{
      if (!r.ok) throw new Error('Could not save the comparison');
      return r.json();
    }})
    .then(() => {{
      comparedThisSession++;
      document.getElementById('count').textContent = `${{comparedThisSession}} compared this session`;
      loadNext();
    }}).catch(() => {{
      document.getElementById('done').textContent =
        "Couldn't reach the server. Check your connection and try again.";
      document.getElementById('done').style.display = 'block';
    }});
}}

document.getElementById('pane1').addEventListener('click', () => choose(1));
document.getElementById('pane2').addEventListener('click', () => choose(2));

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') choose(1);
  if (e.key === 'ArrowRight') choose(2);
}});

// --- zoom overlay: opens on a zoom-icon tap, closes on any tap, drag to pan while open ---

let zoomOpen = false;
let panX = 0, panY = 0, dragging = false, dragMoved = false, dragStartX = 0, dragStartY = 0;

function clampOffset(offset, wrapSize, imgSize) {{
  if (imgSize <= wrapSize) return (wrapSize - imgSize) / 2;
  return Math.min(0, Math.max(wrapSize - imgSize, offset));
}}

function openZoom(side, e) {{
  e.stopPropagation();
  if (!current) return;
  const src = side === 1 ? current.img1.file : current.img2.file;
  const overlay = document.getElementById('zoom-overlay');
  const zimg = document.getElementById('zoom-img');
  zimg.src = src;
  panX = 0; panY = 0;
  zimg.style.transform = 'translate(0px, 0px)';
  overlay.classList.add('open');
  zoomOpen = true;
}}

function closeZoom() {{
  document.getElementById('zoom-overlay').classList.remove('open');
  zoomOpen = false;
}}

document.getElementById('zoom1').addEventListener('click', e => openZoom(1, e));
document.getElementById('zoom2').addEventListener('click', e => openZoom(2, e));

const overlayEl = document.getElementById('zoom-overlay');
overlayEl.addEventListener('mousedown', e => {{
  dragging = true; dragMoved = false;
  dragStartX = e.clientX - panX; dragStartY = e.clientY - panY;
}});
document.addEventListener('mousemove', e => {{
  if (!dragging) return;
  dragMoved = true;
  const zimg = document.getElementById('zoom-img');
  panX = clampOffset(e.clientX - dragStartX, overlayEl.clientWidth, zimg.naturalWidth);
  panY = clampOffset(e.clientY - dragStartY, overlayEl.clientHeight, zimg.naturalHeight);
  zimg.style.transform = `translate(${{panX}}px, ${{panY}}px)`;
}});
document.addEventListener('mouseup', () => {{
  if (dragging && !dragMoved) closeZoom();
  dragging = false;
}});
overlayEl.addEventListener('touchstart', e => {{
  const touch = e.touches[0];
  dragging = true; dragMoved = false;
  dragStartX = touch.clientX - panX; dragStartY = touch.clientY - panY;
}});
overlayEl.addEventListener('touchmove', e => {{
  if (!dragging) return;
  e.preventDefault();
  dragMoved = true;
  const touch = e.touches[0];
  const zimg = document.getElementById('zoom-img');
  panX = clampOffset(touch.clientX - dragStartX, overlayEl.clientWidth, zimg.naturalWidth);
  panY = clampOffset(touch.clientY - dragStartY, overlayEl.clientHeight, zimg.naturalHeight);
  zimg.style.transform = `translate(${{panX}}px, ${{panY}}px)`;
}}, {{passive: false}});
overlayEl.addEventListener('touchend', () => {{
  if (dragging && !dragMoved) closeZoom();
  dragging = false;
}});

loadNext();
</script>
<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    return html
