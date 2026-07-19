"""
Builds a scan-through gallery for the full uncanny-frontier image set: every image as a real
file reference (not embedded base64, which would make a 3000+ image page unusably huge), with
client-side sort/filter/search, a click-to-enlarge lightbox with "show similar" browsing
(build_similarity_index.py's precomputed nearest-neighbor lists), and a pick button that
marks an image as a human-judged winner via curation_server.py's API. Meant for actually
scrolling through the whole batch by eye and guiding the next search run, unlike
notes/uncanny_sweep/gallery.html's binned descriptor grid (capped at 12 thumbnails per bin,
built for the faithfulness/novelty map shape, not for browsing or picking).

compute_data() depends on similarity_index's compute_data() (DEPENDS_ON = ["similarity"]) for
"show similar" neighbor lists; render_html() is a pure function of the returned items list, used
for both scan.html and scan_data.json (the shared data source lightbox.js fetches on every tool
page, not just scan.html's own grid).
"""
import json
import os
import re

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


def round_of(tag):
    """Which search round produced this tag. Only round 1 (no prefix) and round 2 (`r2_`
    prefix) exist so far; extend this if a round 3 driver ever ships."""
    return 2 if tag.startswith("r2_") else 1


def generation_of(tag):
    """Generation number *within* its round. Combined with round_of() below into a single
    sortable integer, since generation numbers restart at 0 each round: a bare gen3_ from round 1
    and an r2_gen3_ from round 2 both parse to 3 here, so sorting on this alone put round 2's
    early generations in the middle of round 1's, instead of after all of round 1."""
    m = re.match(r"(?:r2_)?gen(\d+)_", tag)
    return int(m.group(1)) if m else 0


def sortable_generation(tag):
    """Round-aware sort key: round dominates, generation breaks ties within a round."""
    return round_of(tag) * 100_000 + generation_of(tag)


def compute_data(sweep_dir, deps):
    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    similarity = deps.get("similarity", {})

    items = []
    tag_to_index = {}
    for i, m in enumerate(manifest):
        tag_to_index[m["tag"]] = i
        thumb_path = f"thumbs/{m['tag']}.jpg"
        has_thumb = os.path.exists(f"{sweep_dir}/{thumb_path}")
        items.append({
            "file": os.path.basename(m["file"]),
            "thumb": thumb_path if has_thumb else os.path.basename(m["file"]),
            "tag": m["tag"],
            "gen": generation_of(m["tag"]),
            "sort_gen": sortable_generation(m["tag"]),
            "category": m["category"],
            "prompt_name": m["prompt_name"],
            "prompt_type": m["prompt_type"],
            "prompt": m["prompt"],
            "strength": m["strength"],
            "cfg": m["cfg"],
            "seed": m["seed"],
            "steps": m["steps"],
            "sampler": m["sampler"],
            "negative": m["negative"],
            "faith": round(m["centroid_sim"], 4),
            "novelty": round(m["novelty"], 4),
            "sim": [],
        })

    if similarity:
        for tag, neighbor_tags in similarity.items():
            if tag in tag_to_index:
                items[tag_to_index[tag]]["sim"] = [t for t in neighbor_tags if t in tag_to_index]

    return items


def render_html(
    items, active_expedition=None, active_leg=None, context: WorkspaceContext | None = None,
    focus=None,
):
    focus = focus or (context.focus if context is not None else None)
    render_items = items
    if context is not None:
        render_items = [
            {
                **item,
                "thumb": generated_image_url(item["tag"], context, thumbnail=True),
                "file": generated_image_url(item["tag"], context),
            }
            for item in items
        ]
    data_json = json_script(render_items)

    faith_tip = info_btn("faithfulness")
    novelty_tip = info_btn("novelty")
    picked_only_tip = info_btn(
        "\"Picked\" images are ones a human has flagged as winners in the lightbox. The next search "
        "generation prefers picked images as starting points for new variations, ahead of the "
        "algorithm's own novelty ranking."
    )
    favorited_only_tip = info_btn(
        "\"Favorited\" images are bookmarked for reference (e.g. for a writeup) but have no effect on "
        "the search, unlike picking."
    )
    category_tip = info_btn("map_elites_cell")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS uncanny scan</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
#bar {{
  position:sticky; top:0; z-index:10; background:var(--paper); border-bottom:2px solid var(--ink);
  padding:12px 20px; display:flex; gap:18px; flex-wrap:wrap; align-items:center; font-size:13px;
  transition: transform .18s ease;
}}
#bar.navhidden {{ transform: translateY(-100%); }}
#bar h1 {{
  font-size:18px; font-weight:800; letter-spacing:0.06em; margin:0; color:var(--ink);
  white-space:nowrap; text-transform:uppercase; font-family:var(--font-display);
}}
#bar h1 span {{ color: var(--text-soft); font-weight:400; text-transform:none;
  letter-spacing:0.02em; font-family:var(--font-body); font-size:14px; }}
#bar label {{ display:flex; gap:7px; align-items:center; color:var(--text-soft); font-size:12.5px; }}
#bar select, #bar input[type=text], #bar input[type=number] {{
  background:var(--paper); color:var(--ink); border:1px solid var(--ink);
  padding:5px 9px; font-size:12.5px; outline:none; transition: border-color .15s;
}}
#bar select:hover, #bar input:hover {{ border-color:var(--text-soft); }}
#bar select:focus, #bar input:focus {{ border-color: var(--ink); box-shadow:2px 2px 0 var(--ink); }}
#bar input[type=number] {{ width:64px; }}
#bar input[type=text] {{ width:170px; }}
#bar input[type=checkbox] {{ accent-color: var(--pick); width:14px; height:14px; }}
#count {{ color: var(--text-soft); margin-left:auto; font-variant-numeric: tabular-nums;
  font-family:var(--font-mono); }}

#grid {{
  display:grid; grid-template-columns: repeat(auto-fill, minmax(158px, 1fr));
  gap:8px; padding:14px;
}}
@media (max-width: 700px) {{
  #grid {{ grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap:6px; padding:10px; }}
  #bar {{ padding:10px 12px; gap:10px; }}
  #bar input[type=text] {{ width:120px; }}
}}
.thumb.raised-readout {{
  position:relative; background:var(--paper); cursor:pointer; aspect-ratio:1;
  overflow:hidden; transition: box-shadow .12s ease;
}}
.thumb.raised-readout:hover {{ box-shadow:4px 4px 0 var(--ink),
  inset 1px 1px 0 var(--paper),
  inset -1px -1px 0 var(--paper-deep); z-index:1; }}
.thumb.raised-readout img {{ width:100%; height:100%; object-fit:cover; display:block; }}
.thumb .meta {{
  position:absolute; bottom:0; left:0; right:0;
  background: linear-gradient(to top, rgba(17,18,15,0.85), rgba(17,18,15,0));
  font-size:9.5px; padding:12px 6px 5px; color:var(--paper); white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; opacity:0; transition: opacity .12s ease; font-family:var(--font-mono);
}}
.thumb:hover .meta {{ opacity:1; }}
.thumb .pickbadge {{
  position:absolute; top:5px; right:5px; font-size:14px; color: var(--pick);
  text-shadow:0 1px 3px rgba(17,18,15,0.8);
}}
.thumb.style-b {{ box-shadow: inset 0 0 0 2px #5ec98a, 3px 3px 0 var(--ink),
  inset 1px 1px 0 var(--paper), inset -1px -1px 0 var(--paper-deep); }}
.thumb.conflict-b {{ box-shadow: inset 0 0 0 2px #e0a25e, 3px 3px 0 var(--ink),
  inset 1px 1px 0 var(--paper), inset -1px -1px 0 var(--paper-deep); }}
.thumb.picked {{ box-shadow: inset 0 0 0 2.5px var(--pick), 3px 3px 0 var(--ink),
  inset 1px 1px 0 var(--paper), inset -1px -1px 0 var(--paper-deep); }}
.thumb.picked:hover {{ box-shadow: inset 0 0 0 2.5px var(--pick), 4px 4px 0 var(--ink),
  inset 1px 1px 0 var(--paper), inset -1px -1px 0 var(--paper-deep); z-index:1; }}
.thumb .favbadge {{
  position:absolute; top:5px; left:5px; font-size:14px; color: #e0609a;
  text-shadow:0 1px 3px rgba(17,18,15,0.8);
}}
.thumb.favorited {{ box-shadow: inset 0 0 0 2.5px #e0609a, 3px 3px 0 var(--ink),
  inset 1px 1px 0 var(--paper), inset -1px -1px 0 var(--paper-deep); }}
.thumb.favorited.picked {{ box-shadow: inset 0 0 0 2.5px var(--pick),
  inset 0 0 0 5px #e0609a, 3px 3px 0 var(--ink),
  inset 1px 1px 0 var(--paper), inset -1px -1px 0 var(--paper-deep); }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('scan.html', active_expedition, active_leg, focus=focus)}
<div id="bar">
   <h1>CLAWMARKS <span>uncanny scan</span></h1>
   <span style="color:var(--text-soft);font-size:13px;margin-right:auto">Browse and curate AI-generated artwork from this LoRA search.</span>
  <label>Sort{novelty_tip} <select id="sortKey">
    <option value="novelty_desc">How new or different (high to low)</option>
    <option value="faith_desc">Similarity to real art (high to low)</option>
    <option value="faith_asc">Similarity to real art (low to high)</option>
    <option value="gen_desc">Generation (newest first)</option>
    <option value="gen_asc">Generation (oldest first)</option>
    <option value="prompt_asc">Prompt (A to Z)</option>
    <option value="prompt_desc">Prompt (Z to A)</option>
  </select></label>
  <label>Type <select id="typeFilter">
    <option value="">all</option>
    <option value="style">style</option>
    <option value="conflict">conflict</option>
  </select></label>
  <label>Category{category_tip} <select id="catFilter">
    <option value="">all</option>
    <option value="grid">grid (fixed sweep)</option>
    <option value="negtrigger">negtrigger</option>
    <option value="truncated">truncated</option>
    <option value="allnight_exploit">allnight exploit</option>
    <option value="allnight_explore">allnight explore</option>
    <option value="r2_exploit">round2 exploit</option>
    <option value="r2_explore">round2 explore</option>
  </select></label>
  <label>Prompt <select id="promptFilter"><option value="">all</option></select></label>
  <label>Similarity to real art &gt;= <input type="number" id="faithMin" step="0.05" value=""></label>
  <label>Similarity to real art &lt;= <input type="number" id="faithMax" step="0.05" value="">{faith_tip}</label>
  <label>Search <input type="text" id="search" placeholder="prompt name..."></label>
  <label><input type="checkbox" id="pickedOnly"> picked only{picked_only_tip}</label>
  <label><input type="checkbox" id="favoritedOnly"> favorited only{favorited_only_tip}</label>
  <span id="count"></span>
</div>

<div id="grid"></div>

<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
<script src="/shared-ui.js"></script>
<script>
// json_script() (see shared_ui.py) only protects this declaration from breaking out of the
// <script> tag; it does not HTML-escape the decoded string values. Every place a DATA field
// (model-generated prompt/tag/category text) is written into innerHTML or an HTML attribute
// below must go through escHtml() first, or a value containing e.g. "<img src=x onerror=...>"
// executes when the thumbnail renders.
function escHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

const DATA = {data_json};
let view = DATA.slice();
let picks = {{}};
let favorites = {{}};
const SCOPE_QUERY = new URLSearchParams(window.location.search);
function scopedApi(path) {{
  const url = new URL(path, window.location.origin);
  ['expedition', 'leg'].forEach(key => {{ if (SCOPE_QUERY.has(key)) url.searchParams.set(key, SCOPE_QUERY.get(key)); }});
  return url.toString();
}}

fetch(scopedApi('/api/favorites')).then(r => r.json()).then(f => {{
  favorites = f;
  picks = {{}};
  Object.keys(favorites).forEach(tag => {{ picks[tag] = true; }});
}}).catch(() => {{}}).then(applyFilters);

(function populatePromptFilter() {{
  const names = Array.from(new Set(DATA.map(d => d.prompt_name))).sort();
  const sel = document.getElementById('promptFilter');
  sel.innerHTML = '<option value="">all</option>' +
    names.map(n => `<option value="${{escHtml(n)}}">${{escHtml(n)}}</option>`).join('');
}})();

// Every control that shapes `view` gets mirrored into the URL's query string, so a filtered/
// sorted view survives a reload or a browser-back navigation instead of resetting to defaults.
const FILTER_IDS = ['sortKey', 'typeFilter', 'catFilter', 'promptFilter', 'faithMin', 'faithMax',
  'search', 'pickedOnly', 'favoritedOnly'];

function syncStateToUrl() {{
  const params = new URLSearchParams();
  ['expedition', 'leg', 'focus_id'].forEach(key => {{
    if (SCOPE_QUERY.has(key)) params.set(key, SCOPE_QUERY.get(key));
  }});
  FILTER_IDS.forEach(id => {{
    const el = document.getElementById(id);
    const val = el.type === 'checkbox' ? (el.checked ? '1' : '') : el.value;
    if (val) params.set(id, val);
  }});
  const qs = params.toString();
  history.replaceState(null, '', qs ? `${{location.pathname}}?${{qs}}` : location.pathname);
}}

function syncStateFromUrl() {{
  const params = new URLSearchParams(location.search);
  FILTER_IDS.forEach(id => {{
    if (!params.has(id)) return;
    const el = document.getElementById(id);
    if (el.type === 'checkbox') el.checked = params.get(id) === '1';
    else el.value = params.get(id);
  }});
}}

function applyFilters() {{
  const type = document.getElementById('typeFilter').value;
  const cat = document.getElementById('catFilter').value;
  const prompt = document.getElementById('promptFilter').value;
  const fmin = parseFloat(document.getElementById('faithMin').value);
  const fmax = parseFloat(document.getElementById('faithMax').value);
  const q = document.getElementById('search').value.trim().toLowerCase();
  const sortKey = document.getElementById('sortKey').value;
  const pickedOnly = document.getElementById('pickedOnly').checked;
  const favoritedOnly = document.getElementById('favoritedOnly').checked;

  view = DATA.filter(d => {{
    if (type && d.prompt_type !== type) return false;
    if (cat && d.category !== cat) return false;
    if (prompt && d.prompt_name !== prompt) return false;
    if (!isNaN(fmin) && d.faith < fmin) return false;
    if (!isNaN(fmax) && d.faith > fmax) return false;
    if (q && !d.prompt_name.toLowerCase().includes(q)) return false;
    if (pickedOnly && !picks[d.tag]) return false;
    if (favoritedOnly && !favorites[d.tag]) return false;
    return true;
  }});

  view.sort((a, b) => {{
    switch (sortKey) {{
      case 'novelty_desc': return b.novelty - a.novelty;
      case 'faith_desc': return b.faith - a.faith;
      case 'faith_asc': return a.faith - b.faith;
      case 'gen_desc': return b.sort_gen - a.sort_gen;
      case 'gen_asc': return a.sort_gen - b.sort_gen;
      case 'prompt_asc': return a.prompt_name.localeCompare(b.prompt_name);
      case 'prompt_desc': return b.prompt_name.localeCompare(a.prompt_name);
    }}
  }});
  syncStateToUrl();
  withViewTransition(render);
}}

// Rendering all matching thumbnails in one innerHTML write is what made the page lag on every
// filter keystroke: up to 3672 <img> tags parsed/laid out at once, and on a slow connection that
// also fires a burst of thumbnail requests all at once. Instead render in chunks and grow the
// grid as the user actually scrolls near the bottom (a sentinel + IntersectionObserver), so a
// filter change repaints only a page's worth of thumbnails, not the whole result set.
const PAGE_SIZE = 150;
let shown = 0;
let sentinelObserver = null;

// Wraps a full-grid rebuild so the browser animates matching thumbnails (by
// view-transition-name) sliding from their old position to their new one, instead of snapping
// instantly. Falls back to calling fn() directly on browsers without View Transitions support.
function withViewTransition(fn) {{
  if (document.startViewTransition) document.startViewTransition(fn);
  else fn();
}}

function render() {{
  shown = 0;
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  document.getElementById('count').textContent =
    view.length + ' / ' + DATA.length + ' images | ' + Object.keys(picks).length + ' picked | ' +
    Object.keys(favorites).length + ' favorited';
  renderMore();
}}

function vtName(tag) {{
  return 'vt-' + tag.replace(/[^a-zA-Z0-9_-]/g, '_');
}}

// Click handler looks the item up by its trusted array index (i) instead of interpolating
// d.tag as a string into the onclick attribute, so an attacker-controlled tag value can't break
// out of the JS string literal there.
function openThumb(i) {{
  Lightbox.open(view[i].tag, view.map(v => v.tag));
}}

function thumbHtml(d, i) {{
  const cls = [
    d.prompt_type + '-b',
    picks[d.tag] ? 'picked' : '',
    favorites[d.tag] ? 'favorited' : '',
  ].join(' ');
  return `
    <div class="thumb raised-readout ${{cls}}" style="view-transition-name: ${{vtName(d.tag)}}"
         onclick="openThumb(${{i}})" data-i="${{i}}">
      <img loading="lazy" decoding="async" src="${{escHtml(d.thumb)}}" data-tag="${{escHtml(d.tag)}}">
      ${{picks[d.tag] ? '<div class="pickbadge">&#9733;</div>' : ''}}
      ${{favorites[d.tag] ? '<div class="favbadge">&#9829;</div>' : ''}}
      <div class="meta">${{escHtml(d.prompt_name)}}</div>
    </div>`;
}}

function renderMore() {{
  const grid = document.getElementById('grid');
  const old = document.getElementById('sentinel');
  if (old) old.remove();
  const next = view.slice(shown, shown + PAGE_SIZE);
  grid.insertAdjacentHTML('beforeend', next.map((d, j) => thumbHtml(d, shown + j)).join(''));
  shown += next.length;
  if (shown < view.length) {{
    const sentinel = document.createElement('div');
    sentinel.id = 'sentinel';
    sentinel.style.gridColumn = '1 / -1';
    sentinel.style.height = '1px';
    grid.appendChild(sentinel);
    if (!sentinelObserver) {{
      sentinelObserver = new IntersectionObserver(entries => {{
        if (entries.some(e => e.isIntersecting)) renderMore();
      }}, {{rootMargin: '600px'}});
    }}
    sentinelObserver.observe(sentinel);
  }}
}}

function debounce(fn, ms) {{
  let t;
  return (...args) => {{ clearTimeout(t); t = setTimeout(() => fn(...args), ms); }};
}}
const debouncedApplyFilters = debounce(applyFilters, 250);

['sortKey', 'typeFilter', 'catFilter', 'promptFilter', 'pickedOnly', 'favoritedOnly'].forEach(id =>
  document.getElementById(id).addEventListener('input', applyFilters));
['faithMin', 'faithMax', 'search'].forEach(id =>
  document.getElementById(id).addEventListener('input', debouncedApplyFilters));

document.addEventListener('lightbox:favorite', e => {{
  if (e.detail.favorited) favorites[e.detail.tag] = true; else delete favorites[e.detail.tag];
  withViewTransition(render);
}});

window.addEventListener('popstate', () => {{
  syncStateFromUrl();
  applyFilters();
}});

syncStateFromUrl();
applyFilters();
</script>
</body></html>"""

    return html
