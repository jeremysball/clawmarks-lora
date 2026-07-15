"""
Idea 5 from Fable's exploration-tooling brainstorm (2026-07-09): the actual MAP-Elites archive,
one image per occupied cell instead of gallery.html's up-to-12-per-cell atlas. This is the
whitepaper-ready artifact, and clicking through the archive to override an elite by hand (via
the existing pick API) is a faster human-curation loop than scrolling all 3392 images in
scan.html.

Elite selection per cell: a favorited image (notes/uncanny_sweep/user_favorites.json) wins if
one exists in that cell, since a person's judgment substitutes for the coherence/quality scorer
this project doesn't have (lab_notebook.md Section 3b). This used to be driven by binary image
ratings, but yes/no ratings were replaced by head-to-head comparisons (see
docs/superpowers/specs/2026-07-11-head-to-head-preference-design.md), which have no per-image
manual-override signal of their own, so favoriting fills that role instead. Otherwise falls back
to highest novelty in the cell, matching the ranking the search itself uses to build its
automated "elites" list.

Run after scored_manifest.json exists: python3 -m clawmarks.build.elite_archive
"""
import json
import os
from pathlib import Path

from clawmarks.search import preference_pairwise_model
from clawmarks.search.manifest_index import item_summary
from clawmarks.shared_ui import nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, INFOTIP_CSS, info_btn, json_script

N_BINS = 4  # matches gallery.html's display grid


def elite_sort_key(m, predicted_scores):
    """Sort key for ranking a cell's candidates, most-preferred first (caller sorts ascending
    on this value). Falls back to novelty when no predicted-preference scores exist at all
    (Stage 5a behavior); once scores exist, a tag missing its own score (e.g. an image added to
    the manifest after the embedding cache was last synced) is treated as neutral (0.0) rather
    than assumed bad, so a sync gap doesn't quietly bury an otherwise-good image."""
    if predicted_scores:
        return -predicted_scores.get(m["tag"], 0.0)
    return -m["novelty"]


def build_item_summary(m, sweep_dir, predicted_scores):
    summary = item_summary(m, sweep_dir)
    if m["tag"] in predicted_scores:
        summary["predicted_preference"] = round(float(predicted_scores[m["tag"]]), 4)
    return summary


def compute_data(sweep_dir, use_predicted_preference=False):
    sweep_dir = Path(sweep_dir)
    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    picks = {}
    favorites_path = f"{sweep_dir}/user_favorites.json"
    if os.path.exists(favorites_path):
        with open(favorites_path) as f:
            picks = json.load(f)

    predicted_scores = {}
    model_path = preference_pairwise_model.model_file(sweep_dir)
    if use_predicted_preference and os.path.exists(model_path):
        import joblib

        from clawmarks.search import embed_cache
        from clawmarks.search.preference_pairwise_model import score as pairwise_score

        tags, embeddings = embed_cache.load_cache(embed_cache.embeddings_file(sweep_dir))
        model = joblib.load(model_path)
        scores = pairwise_score(model, embeddings)
        predicted_scores = dict(zip(tags, scores))

    faith_vals = sorted(m["centroid_sim"] for m in manifest)
    novelty_vals = sorted(m["novelty"] for m in manifest)

    def bin_edges(vals, n):
        return [vals[int(i * len(vals) / n)] for i in range(1, n)]

    faith_edges = bin_edges(faith_vals, N_BINS)
    novelty_edges = bin_edges(novelty_vals, N_BINS)

    def bin_ranges(vals, edges):
        """[lo, hi] value span for each of the N_BINS bins, so the archive can label what range
        of faithfulness/novelty a cell actually covers. Bin 0 starts at the data minimum, the
        last bin ends at the maximum, and the interior boundaries are the quantile edges."""
        lo_bounds = [vals[0]] + edges
        hi_bounds = edges + [vals[-1]]
        return [[round(lo, 3), round(hi, 3)] for lo, hi in zip(lo_bounds, hi_bounds)]

    faith_bins = bin_ranges(faith_vals, faith_edges)
    novelty_bins = bin_ranges(novelty_vals, novelty_edges)

    def bin_of(val, edges):
        for i, e in enumerate(edges):
            if val <= e:
                return i
        return len(edges)

    grid = {}
    for m in manifest:
        fb = bin_of(m["centroid_sim"], faith_edges)
        nb = bin_of(m["novelty"], novelty_edges)
        grid.setdefault((fb, nb), []).append(m)

    cells = []
    n_human = 0
    for fb in range(N_BINS):
        for nb in range(N_BINS):
            items = grid.get((fb, nb), [])
            if not items:
                continue
            picked_here = [m for m in items if m["tag"] in picks]
            if picked_here:
                n_human += 1
            cells.append({
                "fb": fb, "nb": nb, "n": len(items),
                "items": [build_item_summary(m, sweep_dir, predicted_scores)
                          for m in sorted(items, key=lambda m: elite_sort_key(m, predicted_scores))],
            })

    cells.sort(key=lambda c: (c["fb"], c["nb"]))
    return {"cells": cells, "n_human": n_human,
            "faith_bins": faith_bins, "novelty_bins": novelty_bins}


def render_html(data):
    cells = data["cells"]
    data_json = json_script(cells)
    faith_bins_json = json_script(data.get("faith_bins", []))
    novelty_bins_json = json_script(data.get("novelty_bins", []))

    elite_tip = info_btn(
        "MAP-Elites is a search strategy that keeps a grid of bins (here, faithfulness x novelty) "
        "and remembers only the single best image found so far for each bin, rather than every image "
        "ever generated. Each cell below is that bin's current champion: a human pick if one exists, "
        "otherwise the highest-novelty image the automated search found there."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS elite archive</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee;
  --text-dim:#9a9aa4; --pick:#f5c542; --style:#5ec98a; --conflict:#e0a25e; --predicted:#7c9eff; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
h1 {{ font-size:18px; margin:0 0 4px; }}
p.sub {{ color:var(--text-dim); max-width:760px; font-size:13px; line-height:1.6; }}
a.navlink {{ color:#7c9eff; font-size:12.5px; text-decoration:none; }}
#grid {{ display:grid; grid-template-columns: repeat({N_BINS}, 1fr); gap:10px; margin-top:20px; max-width:900px; }}
.cell {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
.cell img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; cursor:pointer; }}
.cell .meta {{ padding:8px 10px; font-size:11px; color:var(--text-dim); line-height:1.6; }}
.cell .meta b {{ color:var(--text); }}
.cell .meta .bin {{ display:block; margin-top:5px; padding-top:5px; border-top:1px solid var(--border);
  color:var(--text); font-size:10.5px; }}
.cell.human {{ box-shadow:0 0 0 2px var(--pick); }}
.cell.predicted {{ box-shadow:0 0 0 2px var(--predicted); }}
.badge {{ display:inline-block; padding:1px 6px; border-radius:4px; font-size:10px; margin-left:4px; }}
.badge.human {{ background:rgba(245,197,66,0.18); color:var(--pick); }}
.badge.predicted {{ background:rgba(124,158,255,0.18); color:var(--predicted); }}
.badge.auto {{ background:rgba(154,154,164,0.15); color:var(--text-dim); }}
.cell .viewall {{ display:block; width:100%; background:var(--panel-2,#1d1d22); color:var(--text);
  border:1px solid var(--border); border-top:none; border-radius:0 0 10px 10px; padding:6px;
  font-size:11px; cursor:pointer; }}
.cell .viewall:hover {{ color:#7c9eff; }}
@media (max-width: 640px) {{
  #grid {{ grid-template-columns: repeat(2, 1fr); gap:8px; }}
  .cell .meta {{ font-size:10px; padding:6px 8px; }}
}}

#modal {{ position:fixed; inset:0; background:rgba(8,8,10,0.94); backdrop-filter:blur(6px);
  display:none; z-index:100; padding:30px; overflow-y:auto; }}
#modal.open {{ display:block; }}
#modal .close {{ position:fixed; top:16px; right:22px; font-size:26px; cursor:pointer; color:var(--text-dim); }}
#modal .close:hover {{ color:var(--text); }}
#modal h2 {{ font-size:15px; margin:0 0 6px; color:var(--text); }}
#modal p.hint {{ color:var(--text-dim); font-size:12px; margin:0 0 14px; max-width:640px; }}
#modalGrid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:8px; max-width:1200px; }}
#modalGrid .item {{ background:var(--panel); border-radius:8px; overflow:hidden; cursor:pointer; }}
#modalGrid .item.human {{ box-shadow:0 0 0 2px var(--pick); }}
#modalGrid img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; }}
#modalGrid .meta {{ font-size:10px; color:var(--text-dim); padding:5px 6px; line-height:1.5; }}
@media (max-width: 640px) {{
  #modal {{ padding:14px; }}
  #modalGrid {{ grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); }}
}}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('archive.html')}
<h1>Elite archive{elite_tip}</h1>
<p class="sub">One image per occupied cell of the faithfulness x novelty grid: the actual
MAP-Elites archive, not the full population. Gold-bordered cells are favorited winners;
blue-bordered cells (only when this page is built with --use-predicted-preference) are the
trained model's top pick for that cell; others fall back to the highest-novelty image the
automated search found. The DINOv2 scorer only ranks faithfulness and novelty, not aesthetic
quality, so it can't tell which image in a cell is the better picture: click "view all" to browse
every candidate in a cell and pick a different one by hand. Each cell's bin label spells out the
faithfulness and novelty range that defines it; the four bins per axis are population quartiles,
so every bin holds a similar share of the images rather than an equal slice of the value range.</p>

<div id="grid"></div>

<div id="modal">
  <span class="close" onclick="closeModal()">&times;</span>
  <h2 id="modalTitle"></h2>
  <p class="hint">Click an image to pick it as this cell's elite (or unpick the current one). The
  grid above updates immediately, no rebuild needed.</p>
  <div id="modalGrid"></div>
</div>

<script>
// json_script() only protects this declaration from a </script> breakout; it does not
// HTML-escape decoded string values. Every CELLS field written into innerHTML/an attribute
// below must go through escHtml() first.
function escHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

const CELLS = {data_json};
let picks = {{}};
const FAITH_BINS = {faith_bins_json};
const NOVELTY_BINS = {novelty_bins_json};
const N_BINS = {N_BINS};

// display novelty descending within faith rows, faith ascending row order, to roughly mirror gallery.html
CELLS.sort((a, b) => a.fb - b.fb || b.nb - a.nb);

function binRange(arr, idx) {{
  const r = arr[idx];
  return r ? `${{r[0]}}–${{r[1]}}` : '?';
}}

function eliteFor(c) {{
  const pickedHere = c.items.filter(it => picks[it.tag]);
  if (pickedHere.length) return {{ item: pickedHere[0], source: 'favorited' }};
  if (c.items[0].predicted_preference !== undefined) return {{ item: c.items[0], source: 'predicted preference' }};
  return {{ item: c.items[0], source: 'highest novelty' }};  // items pre-sorted by elite_sort_key
}}

// Click handlers look items up by their trusted array index instead of interpolating a
// tag string into the onclick attribute, so an attacker-controlled tag can't break out of
// the JS string literal there.
function openElite(i) {{
  Lightbox.open(eliteFor(CELLS[i]).item.tag);
}}

function openModalItem(i, j) {{
  Lightbox.open(CELLS[i].items[j].tag);
}}

function render() {{
  const grid = document.getElementById('grid');
  grid.innerHTML = CELLS.map((c, i) => {{
    const {{ item: elite, source }} = eliteFor(c);
    const human = source === 'favorited';
    const predicted = source === 'predicted preference';
    const badgeClass = human ? 'human' : (predicted ? 'predicted' : 'auto');
    const cellClass = human ? 'human' : (predicted ? 'predicted' : '');
    return `
    <div class="cell ${{cellClass}}">
      <img src="${{escHtml(elite.thumb)}}" loading="lazy" data-tag="${{escHtml(elite.tag)}}" onclick="openElite(${{i}})">
      <div class="meta">
        <b>${{escHtml(elite.prompt_name)}}</b> <span class="badge ${{badgeClass}}">${{source}}</span><br>
        faith=${{elite.faith}} novelty=${{elite.novelty}}<br>
        n=${{c.n}} in cell | s=${{elite.strength}} cfg=${{elite.cfg}}
        <span class="bin">bin faith ${{c.fb + 1}}/${{N_BINS}} (${{binRange(FAITH_BINS, c.fb)}}) ·
        novelty ${{c.nb + 1}}/${{N_BINS}} (${{binRange(NOVELTY_BINS, c.nb)}})</span>
      </div>
      <button class="viewall" onclick="openModal(${{i}})">view all ${{c.n}} in this cell</button>
    </div>`;
  }}).join('');
}}

function openModal(i) {{
  const c = CELLS[i];
  document.getElementById('modalTitle').textContent = `${{c.n}} images in this cell`;
  document.getElementById('modalGrid').innerHTML = c.items.map((it, j) => `
    <div class="item ${{picks[it.tag] ? 'human' : ''}}" title="${{escHtml(it.tag)}}" onclick="openModalItem(${{i}}, ${{j}})">
      <img src="${{escHtml(it.thumb)}}" loading="lazy" data-tag="${{escHtml(it.tag)}}">
      <div class="meta">${{escHtml(it.prompt_name)}}<br>f=${{it.faith}} n=${{it.novelty}}</div>
    </div>`).join('');
  document.getElementById('modal').classList.add('open');
}}
function closeModal() {{
  document.getElementById('modal').classList.remove('open');
}}
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') closeModal();
}});

fetch('/api/favorites').then(r => r.json()).then(favorites => {{
  picks = {{}};
  Object.keys(favorites).forEach(tag => {{ picks[tag] = true; }});
  render();
}}).catch(() => {{ render(); }});
</script>
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    return html
