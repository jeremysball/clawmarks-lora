"""
Generates rate.html: a full-screen, keyboard-driven yes/no rating page. Unlike every other
build/*.py generator, this page bakes in no per-image data at build time — it fetches
GET /api/rate/next itself and POSTs to /api/rate, both served by curation_server.py, so the page
never goes stale between rebuilds. Rebuilding only matters if this file itself changes.

Run with: python3 -m clawmarks.build.rate_page (or `clawmarks build rate`)
"""
from clawmarks.config import SWEEP_DIR
from clawmarks.shared_ui import (
    nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, write_scrollnav_asset, write_infotip_asset,
    INFOTIP_CSS, info_btn,
)


def main(argv=None):
    write_scrollnav_asset(SWEEP_DIR)
    write_infotip_asset(SWEEP_DIR)

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
#img {{ max-width:100%; max-height:60vh; border-radius:10px; box-shadow:0 20px 60px rgba(0,0,0,0.6); }}
#meta {{ color:var(--text-dim); font-size:12.5px; margin-top:10px; text-align:center; }}
#buttons {{ display:flex; gap:16px; margin-top:18px; }}
#buttons button {{ font-size:16px; padding:14px 34px; border-radius:10px; cursor:pointer; border:1px solid var(--border); background:var(--panel); color:var(--text); }}
#buttons .no {{ border-color:var(--no); color:var(--no); }}
#buttons .yes {{ border-color:var(--yes); color:var(--yes); }}
#count {{ color:var(--text-dim); font-size:12px; margin-top:14px; }}
#done {{ color:var(--text-dim); font-size:14px; margin-top:40px; text-align:center; }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('rate.html')}
<h1>Rate{rate_tip}</h1>
<p class="sub">Yes or no, as fast as you can go. Keyboard: &larr; or n = no, &rarr; or y = yes.</p>

<div id="stage">
  <img id="img" style="display:none;">
  <div id="meta"></div>
  <div id="buttons" style="display:none;">
    <button class="no" onclick="rate('no')">&larr; no</button>
    <button class="yes" onclick="rate('yes')">yes &rarr;</button>
  </div>
  <div id="done" style="display:none;">Nothing left to rate right now &mdash; every image in the pool has been rated or favorited.</div>
</div>
<div id="count"></div>

<script>
let current = null;
let ratedThisSession = 0;

function loadNext() {{
  document.getElementById('buttons').style.display = 'none';
  fetch('/api/rate/next').then(r => r.json()).then(d => {{
    if (d.done) {{
      current = null;
      document.getElementById('img').style.display = 'none';
      document.getElementById('done').style.display = 'block';
      return;
    }}
    current = d;
    const img = document.getElementById('img');
    img.src = d.thumb;
    img.style.display = 'block';
    document.getElementById('meta').textContent =
      `${{d.prompt_name}} | faith=${{d.faith}} novelty=${{d.novelty}}`;
    document.getElementById('buttons').style.display = 'flex';
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

    with open(f"{SWEEP_DIR}/rate.html", "w") as f:
        f.write(html)

    print(f"wrote {SWEEP_DIR}/rate.html", flush=True)


if __name__ == "__main__":
    main()
