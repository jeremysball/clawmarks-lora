"""
Builds a scan-through gallery for the full uncanny-frontier image set: every image as a real
file reference (not embedded base64, which would make a 3000+ image page unusably huge), with
client-side sort/filter/search, a click-to-enlarge lightbox with "show similar" browsing
(build_similarity_index.py's precomputed nearest-neighbor lists), and a pick button that
marks an image as a human-judged winner via curation_server.py's API. Meant for actually
scrolling through the whole batch by eye and guiding the next search run, unlike
notes/uncanny_sweep/gallery.html's binned descriptor grid (capped at 12 thumbnails per bin,
built for the faithfulness/novelty map shape, not for browsing or picking).

Run after scored_manifest.json (and, for "show similar," similarity.json) exist:
    python3 -m clawmarks.build.scan_gallery
Then open via clawmarks serve (NOT plain http.server, which can't accept picks).
"""
import json, os, re, sys

from clawmarks.config import SWEEP_DIR
from clawmarks.shared_ui import (
    write_lightbox_asset, write_scrollnav_asset, write_infotip_asset,
    MOBILE_BASE_CSS, INFOTIP_CSS, info_btn,
)


def generation_of(tag):
    m = re.match(r"(?:r2_)?gen(\d+)_", tag)
    return int(m.group(1)) if m else 0


def main(argv=None):
    with open(f"{SWEEP_DIR}/scored_manifest.json") as f:
        manifest = json.load(f)

    similarity = {}
    sim_path = f"{SWEEP_DIR}/similarity.json"
    if os.path.exists(sim_path):
        with open(sim_path) as f:
            similarity = json.load(f)

    items = []
    tag_to_index = {}
    for i, m in enumerate(manifest):
        tag_to_index[m["tag"]] = i
        thumb_path = f"thumbs/{m['tag']}.jpg"
        has_thumb = os.path.exists(f"{SWEEP_DIR}/{thumb_path}")
        items.append({
            "file": os.path.basename(m["file"]),
            "thumb": thumb_path if has_thumb else os.path.basename(m["file"]),
            "tag": m["tag"],
            "gen": generation_of(m["tag"]),
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

    data_json = json.dumps(items)

    # scan_data.json is the single shared data source lightbox.js fetches, so every tool page's
    # lightbox (not just scan.html's own grid) has full metadata and "show similar" data for any tag.
    with open(f"{SWEEP_DIR}/scan_data.json", "w") as f:
        json.dump(items, f)
    write_lightbox_asset(SWEEP_DIR)
    write_scrollnav_asset(SWEEP_DIR)
    write_infotip_asset(SWEEP_DIR)

    faith_tip = info_btn(
        "Faithfulness measures how close an image stays to the original training photos, on a scale "
        "from 0 (no resemblance) to 1 (near-identical). It's a cosine similarity between the image's "
        "DINOv2 embedding and the centroid (average position) of the real training images in that "
        "embedding space, not a human judgment of quality."
    )
    novelty_tip = info_btn(
        "Novelty measures how different an image is from everything already explored: the real "
        "training photos, plus (in round 2 onward) every image a prior generation already produced. "
        "It's 1 minus the highest similarity to anything in that reference set, so a novelty of 1 "
        "means nothing seen so far looks like it, and 0 means it's a near-duplicate of something "
        "already found."
    )
    picked_only_tip = info_btn(
        "\"Picked\" images are ones a human has flagged as winners in the lightbox. The next search "
        "generation prefers picked images as starting points for new variations, ahead of the "
        "algorithm's own novelty ranking."
    )
    favorited_only_tip = info_btn(
        "\"Favorited\" images are bookmarked for reference (e.g. for a writeup) but have no effect on "
        "the search, unlike picking."
    )
    category_tip = info_btn(
        "This is a MAP-Elites search: it keeps a grid of faithfulness x novelty bins and tries to "
        "fill every bin with a good example, mapping the whole space instead of hill-climbing "
        "toward one 'best' image. Each generation makes new images two ways. Explore jobs draw a "
        "fresh random subject/texture combination unrelated to anything made before: this is how "
        "the search finds new territory. Exploit jobs nudge an existing strong image's "
        "strength/cfg/seed slightly, hoping a small step nearby does even better: this is how the "
        "search refines what's already working. 'allnight'/'r2' mark which run (round 1 vs round 2) "
        "an image came from; "
        "'grid' and 'negtrigger'/'truncated' are earlier fixed-parameter sweeps, not part of the "
        "generational search at all."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS uncanny scan</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  color-scheme: dark;
  --bg: #0b0b0d; --panel: #16161a; --panel-2: #1d1d22; --border: #2a2a30;
  --text: #eaeaee; --text-dim: #9a9aa4; --text-faint: #6a6a74;
  --accent: #7c9eff; --style: #5ec98a; --conflict: #e0a25e; --pick: #f5c542;
  --radius: 10px;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--text); margin:0; padding:0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
  -webkit-font-smoothing: antialiased;
}}
{MOBILE_BASE_CSS}
#bar {{
  position:sticky; top:0; z-index:10; background: rgba(22,22,26,0.92); backdrop-filter: blur(10px);
  border-bottom:1px solid var(--border); padding:12px 20px; display:flex; gap:18px;
  flex-wrap:wrap; align-items:center; font-size:13px; transition: transform .18s ease;
}}
#bar.navhidden {{ transform: translateY(-100%); }}
#bar h1 {{
  font-size:14px; font-weight:600; letter-spacing:0.01em; margin:0; color:var(--text);
  white-space:nowrap;
}}
#bar h1 span {{ color: var(--text-faint); font-weight:400; }}
#bar label {{ display:flex; gap:7px; align-items:center; color:var(--text-dim); font-size:12.5px; }}
#bar select, #bar input[type=text], #bar input[type=number] {{
  background: var(--panel-2); color: var(--text); border:1px solid var(--border);
  border-radius:6px; padding:5px 9px; font-size:12.5px; outline:none; transition: border-color .15s;
}}
#bar select:hover, #bar input:hover {{ border-color:#3a3a42; }}
#bar select:focus, #bar input:focus {{ border-color: var(--accent); }}
#bar input[type=number] {{ width:64px; }}
#bar input[type=text] {{ width:170px; }}
#bar input[type=checkbox] {{ accent-color: var(--pick); width:14px; height:14px; }}
#count {{ color: var(--text-faint); margin-left:auto; font-variant-numeric: tabular-nums; }}

#grid {{
  display:grid; grid-template-columns: repeat(auto-fill, minmax(158px, 1fr));
  gap:5px; padding:12px;
}}
@media (max-width: 640px) {{
  #grid {{ grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap:3px; padding:8px; }}
  #bar {{ padding:10px 12px; gap:10px; }}
  #bar input[type=text] {{ width:120px; }}
}}
.thumb {{
  position:relative; background: var(--panel); cursor:pointer; aspect-ratio:1;
  overflow:hidden; border-radius:6px; transition: transform .12s ease, box-shadow .12s ease;
}}
.thumb:hover {{ transform: translateY(-2px) scale(1.015); box-shadow: 0 6px 18px rgba(0,0,0,0.45); z-index:1; }}
.thumb img {{ width:100%; height:100%; object-fit:cover; display:block; }}
.thumb .meta {{
  position:absolute; bottom:0; left:0; right:0;
  background: linear-gradient(to top, rgba(0,0,0,0.85), rgba(0,0,0,0));
  font-size:9.5px; padding:12px 6px 5px; color:#dcdce2; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; opacity:0; transition: opacity .12s ease;
}}
.thumb:hover .meta {{ opacity:1; }}
.thumb .pickbadge {{
  position:absolute; top:5px; right:5px; font-size:14px; color: var(--pick);
  text-shadow:0 1px 3px rgba(0,0,0,0.8);
}}
.thumb.style-b {{ box-shadow: inset 0 0 0 2px rgba(94,201,138,0.55); }}
.thumb.conflict-b {{ box-shadow: inset 0 0 0 2px rgba(224,162,94,0.55); }}
.thumb.picked {{ box-shadow: inset 0 0 0 2.5px var(--pick); }}
.thumb.picked:hover {{ box-shadow: inset 0 0 0 2.5px var(--pick), 0 6px 18px rgba(0,0,0,0.45); }}
.thumb .favbadge {{
  position:absolute; top:5px; left:5px; font-size:14px; color: #e0609a;
  text-shadow:0 1px 3px rgba(0,0,0,0.8);
}}
.thumb.favorited {{ box-shadow: inset 0 0 0 2.5px #e0609a; }}
.thumb.favorited.picked {{ box-shadow: inset 0 0 0 2.5px var(--pick), inset 0 0 0 5px #e0609a; }}
{INFOTIP_CSS}
</style></head><body>

<div id="bar" data-autohide>
  <h1>CLAWMARKS <span>uncanny scan</span></h1>
  <label>More tools <select id="toolNav" onchange="if(this.value) location.href=this.value;">
    <option value="">jump to...</option>
    <option value="explore.html">all tools (hub)</option>
    <option value="map.html">solution map (UMAP)</option>
    <option value="coverage.html">coverage / void map</option>
    <option value="archive.html">elite archive</option>
    <option value="redundancy.html">redundancy clusters</option>
    <option value="novelty_decay.html">novelty decay watchlist</option>
    <option value="lineage.html">lineage tree</option>
    <option value="seeds.html">candidate seeds</option>
    <option value="gallery.html">binned atlas (original)</option>
  </select></label>
  <label>Sort{novelty_tip} <select id="sortKey">
    <option value="novelty_desc">Novelty (high to low)</option>
    <option value="faith_desc">Faithfulness (high to low)</option>
    <option value="faith_asc">Faithfulness (low to high)</option>
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
  <label>Faith &gt;= <input type="number" id="faithMin" step="0.05" value=""></label>
  <label>Faith &lt;= <input type="number" id="faithMax" step="0.05" value="">{faith_tip}</label>
  <label>Search <input type="text" id="search" placeholder="prompt name..."></label>
  <label><input type="checkbox" id="pickedOnly"> picked only{picked_only_tip}</label>
  <label><input type="checkbox" id="favoritedOnly"> favorited only{favorited_only_tip}</label>
  <span id="count"></span>
</div>

<div id="grid"></div>

<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
<script>
const DATA = {data_json};
let view = DATA.slice();
let picks = {{}};
let favorites = {{}};

Promise.all([
  fetch('/api/picks').then(r => r.json()).then(p => {{ picks = p; }}).catch(() => {{}}),
  fetch('/api/favorites').then(r => r.json()).then(f => {{ favorites = f; }}).catch(() => {{}}),
]).then(render);

(function populatePromptFilter() {{
  const names = Array.from(new Set(DATA.map(d => d.prompt_name))).sort();
  const sel = document.getElementById('promptFilter');
  sel.innerHTML = '<option value="">all</option>' +
    names.map(n => `<option value="${{n}}">${{n}}</option>`).join('');
}})();

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
      case 'gen_desc': return b.gen - a.gen;
      case 'gen_asc': return a.gen - b.gen;
      case 'prompt_asc': return a.prompt_name.localeCompare(b.prompt_name);
      case 'prompt_desc': return b.prompt_name.localeCompare(a.prompt_name);
    }}
  }});
  render();
}}

// Rendering all matching thumbnails in one innerHTML write is what made the page lag on every
// filter keystroke: up to 3672 <img> tags parsed/laid out at once, and on a slow connection that
// also fires a burst of thumbnail requests all at once. Instead render in chunks and grow the
// grid as the user actually scrolls near the bottom (a sentinel + IntersectionObserver), so a
// filter change repaints only a page's worth of thumbnails, not the whole result set.
const PAGE_SIZE = 150;
let shown = 0;
let sentinelObserver = null;

function render() {{
  shown = 0;
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  document.getElementById('count').textContent =
    view.length + ' / ' + DATA.length + ' images | ' + Object.keys(picks).length + ' picked | ' +
    Object.keys(favorites).length + ' favorited';
  renderMore();
}}

function thumbHtml(d, i) {{
  const cls = [
    d.prompt_type + '-b',
    picks[d.tag] ? 'picked' : '',
    favorites[d.tag] ? 'favorited' : '',
  ].join(' ');
  return `
    <div class="thumb ${{cls}}" onclick="Lightbox.open('${{d.tag}}', view.map(v=>v.tag))" data-i="${{i}}">
      <img loading="lazy" decoding="async" src="${{d.thumb}}" data-tag="${{d.tag}}">
      ${{picks[d.tag] ? '<div class="pickbadge">&#9733;</div>' : ''}}
      ${{favorites[d.tag] ? '<div class="favbadge">&#9829;</div>' : ''}}
      <div class="meta">f=${{d.faith}} n=${{d.novelty}} ${{d.prompt_name}}</div>
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

document.addEventListener('lightbox:pick', e => {{
  if (e.detail.picked) picks[e.detail.tag] = true; else delete picks[e.detail.tag];
  render();
}});
document.addEventListener('lightbox:favorite', e => {{
  if (e.detail.favorited) favorites[e.detail.tag] = true; else delete favorites[e.detail.tag];
  render();
}});

applyFilters();
</script>
</body></html>"""

    with open(f"{SWEEP_DIR}/scan.html", "w") as f:
        f.write(html)

    print(f"wrote {SWEEP_DIR}/scan.html ({len(items)} images, "
          f"{sum(1 for it in items if it['sim'])} with similarity data)")


if __name__ == "__main__":
    main()
