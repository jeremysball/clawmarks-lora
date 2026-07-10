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
#imgwrap.zoomed {{ height:78vh; width:100%; cursor:grab; }}
#imgwrap.zoomed #img {{ max-width:none; max-height:none; border-radius:0; box-shadow:none; }}
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
let zoomed = false;
let panOffsetX = 0, panOffsetY = 0;

function loadNext() {{
  fetch('/api/rate/next').then(r => r.json()).then(d => {{
    if (d.done) {{
      current = null;
      document.getElementById('img').style.display = 'none';
      document.getElementById('done').style.display = 'block';
      return;
    }}
    current = d;
    resetZoom();
    const img = document.getElementById('img');
    img.style.transition = '';
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

// --- zoom ---

function clampOffset(offset, wrapSize, imgSize) {{
  if (imgSize <= wrapSize) return (wrapSize - imgSize) / 2;
  return Math.min(0, Math.max(wrapSize - imgSize, offset));
}}

function resetZoom() {{
  zoomed = false;
  panOffsetX = 0;
  panOffsetY = 0;
  document.getElementById('imgwrap').classList.remove('zoomed');
  document.getElementById('img').style.transform = 'translate(0px, 0px)';
}}

function zoomIn(clientX, clientY) {{
  const img = document.getElementById('img');
  const wrap = document.getElementById('imgwrap');
  const rect = img.getBoundingClientRect();
  const fracX = (clientX - rect.left) / rect.width;
  const fracY = (clientY - rect.top) / rect.height;
  wrap.classList.add('zoomed');
  zoomed = true;
  const targetX = fracX * img.naturalWidth;
  const targetY = fracY * img.naturalHeight;
  panOffsetX = clampOffset(wrap.clientWidth / 2 - targetX, wrap.clientWidth, img.naturalWidth);
  panOffsetY = clampOffset(wrap.clientHeight / 2 - targetY, wrap.clientHeight, img.naturalHeight);
  img.style.transform = `translate(${{panOffsetX}}px, ${{panOffsetY}}px)`;
}}

const imgwrapEl = document.getElementById('imgwrap');

imgwrapEl.addEventListener('dblclick', e => {{
  if (!current) return;
  if (zoomed) {{
    resetZoom();
  }} else {{
    zoomIn(e.clientX, e.clientY);
  }}
}});

// mouse panning while zoomed (desktop)
let mouseDown = false, mouseStartX = 0, mouseStartY = 0;
imgwrapEl.addEventListener('mousedown', e => {{
  if (!zoomed) return;
  mouseDown = true;
  mouseStartX = e.clientX;
  mouseStartY = e.clientY;
}});
document.addEventListener('mousemove', e => {{
  if (!mouseDown || !zoomed) return;
  const img = document.getElementById('img');
  const wrap = document.getElementById('imgwrap');
  const newX = clampOffset(panOffsetX + (e.clientX - mouseStartX), wrap.clientWidth, img.naturalWidth);
  const newY = clampOffset(panOffsetY + (e.clientY - mouseStartY), wrap.clientHeight, img.naturalHeight);
  img.style.transform = `translate(${{newX}}px, ${{newY}}px)`;
}});
document.addEventListener('mouseup', e => {{
  if (!mouseDown) return;
  mouseDown = false;
  if (!zoomed) return;
  const img = document.getElementById('img');
  const wrap = document.getElementById('imgwrap');
  panOffsetX = clampOffset(panOffsetX + (e.clientX - mouseStartX), wrap.clientWidth, img.naturalWidth);
  panOffsetY = clampOffset(panOffsetY + (e.clientY - mouseStartY), wrap.clientHeight, img.naturalHeight);
}});

// --- touch: swipe-to-vote when not zoomed, pan when zoomed ---

const DEADZONE_PX = 10;
const SWIPE_THRESHOLD_FRACTION = 0.25;
let touchActive = false, touchClassified = null; // null | 'swipe' | 'pan' | 'ignore'
let touchStartX = 0, touchStartY = 0, touchDX = 0, touchDY = 0;

function updateSwipeVisual(dx) {{
  const img = document.getElementById('img');
  const overlay = document.getElementById('swipe-overlay');
  img.style.transition = '';
  img.style.transform = `translateX(${{dx}}px)`;
  const width = img.getBoundingClientRect().width || 1;
  const frac = Math.min(1, Math.abs(dx) / (width * SWIPE_THRESHOLD_FRACTION));
  overlay.className = dx > 0 ? 'yes' : 'no';
  overlay.style.opacity = frac;
}}

function finishSwipe(dx) {{
  const img = document.getElementById('img');
  const overlay = document.getElementById('swipe-overlay');
  const width = img.getBoundingClientRect().width || 1;
  const threshold = width * SWIPE_THRESHOLD_FRACTION;
  if (Math.abs(dx) >= threshold) {{
    const label = dx > 0 ? 'yes' : 'no';
    img.style.transition = 'transform 0.2s ease';
    img.style.transform = `translateX(${{dx > 0 ? width * 1.2 : -width * 1.2}}px)`;
    overlay.style.opacity = 0;
    setTimeout(() => rate(label), 180);
  }} else {{
    img.style.transition = 'transform 0.2s ease';
    img.style.transform = 'translateX(0px)';
    overlay.style.opacity = 0;
  }}
}}

imgwrapEl.addEventListener('touchstart', e => {{
  if (!current || e.touches.length !== 1) return;
  touchActive = true;
  touchClassified = null;
  touchStartX = e.touches[0].clientX;
  touchStartY = e.touches[0].clientY;
  touchDX = 0;
  touchDY = 0;
}});

imgwrapEl.addEventListener('touchmove', e => {{
  if (!touchActive) return;
  const t = e.touches[0];
  touchDX = t.clientX - touchStartX;
  touchDY = t.clientY - touchStartY;

  if (touchClassified === null) {{
    if (Math.abs(touchDX) < DEADZONE_PX && Math.abs(touchDY) < DEADZONE_PX) return;
    if (zoomed) {{
      touchClassified = 'pan';
    }} else if (Math.abs(touchDX) > Math.abs(touchDY)) {{
      touchClassified = 'swipe';
    }} else {{
      touchClassified = 'ignore';
    }}
  }}

  if (touchClassified === 'ignore') return;
  e.preventDefault();

  if (touchClassified === 'swipe') {{
    updateSwipeVisual(touchDX);
  }} else if (touchClassified === 'pan') {{
    const img = document.getElementById('img');
    const wrap = document.getElementById('imgwrap');
    const newX = clampOffset(panOffsetX + touchDX, wrap.clientWidth, img.naturalWidth);
    const newY = clampOffset(panOffsetY + touchDY, wrap.clientHeight, img.naturalHeight);
    img.style.transform = `translate(${{newX}}px, ${{newY}}px)`;
  }}
}}, {{passive: false}});

imgwrapEl.addEventListener('touchend', () => {{
  if (!touchActive) return;
  touchActive = false;
  if (touchClassified === 'swipe') {{
    finishSwipe(touchDX);
  }} else if (touchClassified === 'pan') {{
    const img = document.getElementById('img');
    const wrap = document.getElementById('imgwrap');
    panOffsetX = clampOffset(panOffsetX + touchDX, wrap.clientWidth, img.naturalWidth);
    panOffsetY = clampOffset(panOffsetY + touchDY, wrap.clientHeight, img.naturalHeight);
  }}
  touchClassified = null;
}});

loadNext();
</script>
<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    return html
