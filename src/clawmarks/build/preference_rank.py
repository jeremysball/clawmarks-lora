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
from pathlib import Path

import joblib

from clawmarks.search import embed_cache, preference_pairwise_model
from clawmarks.search.manifest_index import index_by_tag, item_summary
from clawmarks.search.preference_pairwise_model import score
from clawmarks.shared_ui import (
    CONTROL_CSS,
    INFOTIP_CSS,
    MOBILE_BASE_CSS,
    SULFUR_CSS,
    SULFUR_FONT_CSS,
    TOPNAV_CSS,
    info_btn,
    json_script,
    nav_bar_html,
)
from clawmarks.workspace_context import WorkspaceContext, generated_image_url


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
    sweep_dir = Path(sweep_dir)
    model_path = preference_pairwise_model.model_file(sweep_dir)
    if not os.path.exists(model_path):
        return {"has_model": False, "model_file": model_path}

    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)
    by_tag = index_by_tag(manifest)

    tags, embeddings = embed_cache.load_cache(embed_cache.embeddings_file(sweep_dir))
    model = joblib.load(model_path)
    scores = score(model, embeddings)
    items = build_ranked_items(by_tag, tags, scores, sweep_dir)

    return {"has_model": True, "items": items}


def render_html(
    data, active_expedition=None, active_leg=None, running=None,
    context: WorkspaceContext | None = None,
    focus=None,
):
    focus = focus or (context.focus if context is not None else None)
    if not data["has_model"]:
        return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
body {{ margin:0; padding:24px; }}
h1 {{ font-size:22px; margin:24px 0 4px; letter-spacing:0.02em; text-transform:uppercase; }}
p.empty {{ color:var(--text-soft); max-width:640px; font-size:13px; line-height:1.7;
  padding:14px 0; border-bottom:1px solid var(--rule); }}
</style>
</head><body>
{nav_bar_html('preference_rank.html', active_expedition=active_expedition, active_leg=active_leg, running=running, focus=focus)}
<h1>Predicted preference</h1>
<p class="empty">No trained model at <code>{data["model_file"]}</code>. Run <code>python -m clawmarks.search.preference_pairwise_model</code> first. It needs 50 or more comparisons.</p>
<script src="scrollnav.js"></script>
<script src="/shared-ui.js"></script>
</body></html>"""

    items = data["items"]
    if context is not None:
        items = [
            {
                **item,
                "thumb": generated_image_url(item["tag"], context, thumbnail=True),
                "file": generated_image_url(item["tag"], context),
            }
            for item in items
        ]

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
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
body {{ margin:0; padding:24px; }}
h1 {{ font-size:22px; margin:24px 0 4px; letter-spacing:0.02em; text-transform:uppercase; }}
p.sub {{ color:var(--text-soft); max-width:760px; font-size:13px; line-height:1.6;
  padding-bottom:14px; border-bottom:1px solid var(--rule); }}
#list {{ display:flex; flex-direction:column; gap:0; margin-top:8px; max-width:960px; }}
.row {{ display:flex; align-items:center; gap:16px; padding:12px 0; border-bottom:1px solid var(--rule); }}
.row:last-child {{ border-bottom:none; }}
.row img {{ width:72px; height:72px; object-fit:cover; flex-shrink:0; cursor:pointer;
  border:1px solid var(--ink); }}
.row .meta {{ flex:1; font:12.5px/1.4 var(--font-mono); color:var(--text-soft); }}
.row .meta b {{ color:var(--ink); font-weight:600; }}
.row .review {{ display:flex; gap:6px; flex-shrink:0; }}
.row .review button {{ font:600 11.5px/1 var(--font-body); padding:6px 10px; cursor:pointer;
  background:var(--paper); color:var(--ink); border:1px solid var(--ink); }}
.row .review button:hover {{ background:var(--sulfur); }}
.row .review button.selected {{ background:var(--sulfur); color:var(--ink); }}
#review-controls {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap;
  font-size:12.5px; color:var(--text-soft); margin-top:14px; padding:8px 0;
  border-bottom:1px solid var(--rule); }}
.flag-error {{ color:#a84820; min-height:1em; font-size:12px; }}
@media (max-width: 700px) {{
  .row {{ flex-wrap:wrap; gap:10px; padding:10px 0; }}
  .row img {{ width:64px; height:64px; }}
  .row .meta {{ flex:1 1 100%; order:3; }}
  .row .review {{ order:2; }}
}}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('preference_rank.html', active_expedition=active_expedition, active_leg=active_leg, running=running, focus=focus)}
<h1>Predicted preference{rank_tip}</h1>
<p class="sub">Top {len(items)} images by predicted preference score, highest first.</p>
<div id="review-controls"><label><input id="reviewMode" type="checkbox"> Review top, middle, and bottom</label><span id="reviewCount"></span><span id="flagError" class="flag-error" role="alert" aria-live="polite"></span></div>
<div id="list"></div>
<script>
// json_script() only protects this declaration from a <\\/script> breakout; it does not
// HTML-escape decoded string values. Every ITEMS field written into innerHTML/an attribute
// below must go through escHtml() first.
function escHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

const ITEMS = {data_json};
let reviewMode = false;
let flags = {{}};
const SCOPE_QUERY = new URLSearchParams(window.location.search);

function reviewIndexes() {{
  const middle = Math.floor(ITEMS.length / 2);
  return new Set([...Array(Math.min(20, ITEMS.length)).keys(),
    ...Array.from({{length: Math.min(10, ITEMS.length)}}, (_, i) => Math.max(0, middle - 5) + i),
    ...Array.from({{length: Math.min(10, ITEMS.length)}}, (_, i) => Math.max(0, ITEMS.length - 10) + i)]);
}}

// Click handler looks the item up by its trusted array index instead of interpolating a tag
// string into the onclick attribute, so an attacker-controlled tag can't break out of the JS
// string literal there.
function openItem(i) {{
  Lightbox.open(ITEMS[i].tag);
}}

function flagSelected(tag, flag) {{
  return flags[tag]?.flag === flag ? 'selected' : '';
}}

function saveFlag(tag, flag) {{
  fetch('/api/preference_rank/flag', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{tag, flag, expedition:SCOPE_QUERY.get('expedition'), leg:SCOPE_QUERY.get('leg')}})}})
    .then(r => {{
      if (!r.ok) throw new Error('flag save failed');
      return r.json();
    }})
    .then(result => {{
      if (result.ok !== true) throw new Error('flag save failed');
      flags[tag] = {{flag: flag, flagged_at: flags[tag]?.flagged_at ?? null}};
      document.getElementById('flagError').textContent = '';
      render();
    }})
    .catch(() => {{
      document.getElementById('flagError').textContent = 'Could not save this flag.';
    }});
}}

function render() {{
  const indexes = reviewIndexes();
  const visible = reviewMode ? ITEMS.map((it, i) => [it, i]).filter(([, i]) => indexes.has(i)) : ITEMS.map((it, i) => [it, i]);
  document.getElementById('reviewCount').textContent = reviewMode ? `${{visible.length}} representative images` : '';
  document.getElementById('list').innerHTML = visible.map(([it, i]) => `
   <div class="row">
     <img src="${{escHtml(it.thumb)}}" loading="lazy" data-tag="${{escHtml(it.tag)}}" onclick="openItem(${{i}})">
    <div class="meta"><b>Rank #${{i + 1}}</b> | p=${{it.predicted_preference}} | f=${{it.faith}} n=${{it.novelty}}</div>
     ${{reviewMode ? `<div class="review"><button class="flag-button ${{flagSelected(it.tag, 'matches')}}" aria-pressed="${{flags[it.tag]?.flag === 'matches'}}" data-review-index="${{i}}" data-flag="matches">matches my taste</button><button class="flag-button ${{flagSelected(it.tag, 'questionable')}}" aria-pressed="${{flags[it.tag]?.flag === 'questionable'}}" data-review-index="${{i}}" data-flag="questionable">questionable</button></div>` : ''}}
   </div>`).join('');
  document.querySelectorAll('[data-review-index]').forEach(button => button.addEventListener('click', () => {{
    saveFlag(ITEMS[Number(button.dataset.reviewIndex)].tag, button.dataset.flag);
  }}));
}}

const flagsUrl = new URL('/api/preference_rank/flags', window.location.origin);
['expedition', 'leg'].forEach(key => {{ if (SCOPE_QUERY.has(key)) flagsUrl.searchParams.set(key, SCOPE_QUERY.get(key)); }});
fetch(flagsUrl).then(r => r.ok ? r.json() : {{}}).then(data => {{ flags = data; render(); }}).catch(render);
document.getElementById('reviewMode').addEventListener('change', e => {{ reviewMode = e.target.checked; render(); }});
</script>
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
<script src="/shared-ui.js"></script>
</body></html>"""

    return html
