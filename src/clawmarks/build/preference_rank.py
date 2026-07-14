"""
Ranks every embedded image by the trained pairwise model's predicted preference score, highest
first, so the model's judgment can be eyeballed against the user's own taste before Stage 5b
lets it steer anything live. Requires search/preference_pairwise_model.py to have already
produced notes/uncanny_sweep/preference_pairwise_model.joblib (needs 50+ comparisons — see
search/preference_pairwise_model.py's MIN_COMPARISONS). See
docs/superpowers/specs/2026-07-11-head-to-head-preference-design.md.

Served live at /preference_rank.html by curation_server.py.
"""
import json
import os

import joblib

from clawmarks.search import embed_cache
from clawmarks.search.manifest_index import index_by_tag, item_summary
from clawmarks.search.preference_pairwise_model import MODEL_FILE, score
from clawmarks.shared_ui import INFOTIP_CSS, MOBILE_BASE_CSS, TOPNAV_CSS, info_btn, nav_bar_html, json_script


def build_ranked_items(by_tag, tags, scores, sweep_dir, limit=500):
    ranked = sorted(
        ((t, s) for t, s in zip(tags, scores) if t in by_tag),
        key=lambda pair: -pair[1],
    )[:limit]
    items = []
    for tag, pref_score in ranked:
        summary = item_summary(by_tag[tag], sweep_dir)
        summary["predicted_preference"] = round(float(pref_score), 4)
        items.append(summary)
    return items


def compute_data(sweep_dir):
    if not os.path.exists(MODEL_FILE):
        return {"has_model": False}

    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)
    by_tag = index_by_tag(manifest)

    tags, embeddings = embed_cache.load_cache(embed_cache.EMBEDDINGS_FILE)
    model = joblib.load(MODEL_FILE)
    scores = score(model, embeddings)
    items = build_ranked_items(by_tag, tags, scores, sweep_dir)

    return {"has_model": True, "items": items}


def render_html(data):
    if not data["has_model"]:
        return (f"<!doctype html><html><body>no trained model at {MODEL_FILE}; run `python -m "
                f"clawmarks.search.preference_pairwise_model` first (needs 50+ comparisons)</body></html>")

    items = data["items"]

    rank_tip = info_btn(
        "Sorted by the trained preference model's predicted score, highest first: the model "
        "learned this ranking from your head-to-head comparisons, not a yes/no judgment. This "
        "view exists to sanity-check the model before it's allowed to steer the live search: "
        "does the top of this list actually look like things you like?"
    )
    data_json = json_script(items)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS predicted preference</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee; --text-dim:#9a9aa4; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
h1 {{ font-size:18px; margin:0 0 4px; }}
p.sub {{ color:var(--text-dim); max-width:760px; font-size:13px; line-height:1.6; }}
#grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap:10px; margin-top:20px; }}
.cell {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
.cell img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; cursor:pointer; }}
.cell .meta {{ padding:6px 8px; font-size:11px; color:var(--text-dim); }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('preference_rank.html')}
<h1>Predicted preference{rank_tip}</h1>
<p class="sub">Top {len(items)} images by predicted preference score, highest first.</p>
<div id="grid"></div>
<script>
// json_script() only protects this declaration from a </script> breakout; it does not
// HTML-escape decoded string values. Every ITEMS field written into innerHTML/an attribute
// below must go through escHtml() first.
function escHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

const ITEMS = {data_json};

// Click handler looks the item up by its trusted array index instead of interpolating a tag
// string into the onclick attribute, so an attacker-controlled tag can't break out of the JS
// string literal there.
function openItem(i) {{
  Lightbox.open(ITEMS[i].tag);
}}

document.getElementById('grid').innerHTML = ITEMS.map((it, i) => `
  <div class="cell">
    <img src="${{escHtml(it.thumb)}}" loading="lazy" data-tag="${{escHtml(it.tag)}}" onclick="openItem(${{i}})">
    <div class="meta">p=${{it.predicted_preference}} | f=${{it.faith}} n=${{it.novelty}}</div>
  </div>`).join('');
</script>
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    return html
