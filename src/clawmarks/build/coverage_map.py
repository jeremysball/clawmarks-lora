"""
Idea 3 from Fable's exploration-tooling brainstorm (2026-07-09): a coverage/void heatmap on the
faithfulness x novelty plane. gallery.html already bins images into this plane but is built to
show *what's* in each cell (up to 12 thumbnails); this instead makes the empty cells salient,
since MAP-Elites is fundamentally about filling bins, and an unfilled cell next to a
high-novelty occupied one is the shortlist for "where should the next generation explore."

Uses a finer grid (8x8) than gallery.html's 4x4 display grid, and colors cells by image count
(density), with "frontier" cells (empty, but 4-adjacent to a cell at or above the median
occupied-cell count) called out separately from ordinary empty cells.

Run after scored_manifest.json exists: python3 -m clawmarks.build.coverage_map
"""
import json
import os

from clawmarks.shared_ui import (
    BTN_CSS,
    DARK_TOKENS,
    DINO_TIP,
    INFOTIP_CSS,
    MOBILE_BASE_CSS,
    TOPNAV_CSS,
    info_btn,
    json_script,
    nav_bar_html,
)

N_BINS = 8


def compute_data(sweep_dir):
    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    faith_vals = sorted(m["centroid_sim"] for m in manifest)
    novelty_vals = sorted(m["novelty"] for m in manifest)

    def bin_edges(vals, n):
        return [vals[int(i * len(vals) / n)] for i in range(1, n)]

    faith_edges = bin_edges(faith_vals, N_BINS)
    novelty_edges = bin_edges(novelty_vals, N_BINS)

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

    counts = {k: len(v) for k, v in grid.items()}
    occupied_counts = sorted(counts.values())
    median_count = occupied_counts[len(occupied_counts) // 2] if occupied_counts else 0
    max_count = max(occupied_counts) if occupied_counts else 1

    frontier = set()
    for fb in range(N_BINS):
        for nb in range(N_BINS):
            if (fb, nb) in counts:
                continue
            neighbors = [(fb + 1, nb), (fb - 1, nb), (fb, nb + 1), (fb, nb - 1)]
            if any(counts.get(n, 0) >= median_count and counts.get(n, 0) > 0 for n in neighbors):
                frontier.add((fb, nb))

    def item_summary(m):
        return {
            "tag": m["tag"],
            "thumb": (f"thumbs/{m['tag']}.jpg" if os.path.exists(f"{sweep_dir}/thumbs/{m['tag']}.jpg")
                      else os.path.basename(m["file"])),
            "faith": round(m["centroid_sim"], 4),
            "novelty": round(m["novelty"], 4),
            "prompt_name": m["prompt_name"],
        }

    cells_json = []
    for fb in range(N_BINS):
        for nb in range(N_BINS):
            items = sorted(grid.get((fb, nb), []), key=lambda m: -m["novelty"])
            n = len(items)
            best = items[0] if items else None
            cells_json.append({
                "fb": fb, "nb": nb, "count": n,
                "frontier": (fb, nb) in frontier,
                "faith_lo": round(faith_edges[fb - 1], 3) if fb > 0 else None,
                "faith_hi": round(faith_edges[fb], 3) if fb < len(faith_edges) else None,
                "novelty_lo": round(novelty_edges[nb - 1], 3) if nb > 0 else None,
                "novelty_hi": round(novelty_edges[nb], 3) if nb < len(novelty_edges) else None,
                "thumb": (f"thumbs/{best['tag']}.jpg" if best and os.path.exists(f"{sweep_dir}/thumbs/{best['tag']}.jpg")
                          else (os.path.basename(best["file"]) if best else None)),
                "best_tag": best["tag"] if best else None,
                "items": [item_summary(m) for m in items],
            })

    return {"cells": cells_json, "median_count": median_count, "max_count": max_count}


def _fmt_range(lo, hi, decimals=2):
    lo_s = f"{lo:.{decimals}f}" if lo is not None else "0"
    hi_s = f"{hi:.{decimals}f}" if hi is not None else "1"
    return f"{lo_s}-{hi_s}"


def top_frontier_cells(data, n=3):
    """Picks the `n` most attractive frontier cells (empty, but bordering well-populated
    territory) for the generation cockpit's target-cell picker. "Most attractive" means densest
    adjacent territory: the total image count summed across the cell's occupied 4-neighbors,
    since a frontier cell bordering many images is more reachable than one bordering few. Each
    result is shaped for the picker's card UI, including a representative thumbnail/faith/novelty
    pulled from the single best (highest-novelty) neighboring image, not the empty cell itself."""
    cells = data["cells"]
    by_coord = {(c["fb"], c["nb"]): c for c in cells}
    frontier = [c for c in cells if c["frontier"]]

    shaped = []
    for c in frontier:
        fb, nb = c["fb"], c["nb"]
        neighbor_coords = [(fb + 1, nb), (fb - 1, nb), (fb, nb + 1), (fb, nb - 1)]
        neighbors = [by_coord[nc] for nc in neighbor_coords if nc in by_coord and by_coord[nc]["count"] > 0]
        if not neighbors:
            continue
        adjacent = sum(nb_cell["count"] for nb_cell in neighbors)
        # Each cell's "items" is already sorted descending by novelty (see compute_data), so
        # items[0] is that cell's own best; the single best across all neighbors is the max of
        # those per-cell bests, not just the item from the most populous neighbor.
        neighbor_bests = [nb_cell["items"][0] for nb_cell in neighbors if nb_cell["items"]]
        best_item = max(neighbor_bests, key=lambda item: item["novelty"]) if neighbor_bests else None
        shaped.append({
            "fb": fb, "nb": nb,
            "range": f"Faith {_fmt_range(c['faith_lo'], c['faith_hi'])}, "
                     f"novelty {_fmt_range(c['novelty_lo'], c['novelty_hi'])}",
            "adjacent": adjacent,
            "thumb": best_item["thumb"] if best_item else None,
            "near_faith": best_item["faith"] if best_item else None,
            "near_novelty": best_item["novelty"] if best_item else None,
            "faith_lo": c["faith_lo"], "faith_hi": c["faith_hi"],
            "novelty_lo": c["novelty_lo"], "novelty_hi": c["novelty_hi"],
        })

    shaped.sort(key=lambda s: -s["adjacent"])
    return shaped[:n]


def neighbor_tags(data, fb, nb):
    """Tags of every item in the occupied 4-neighbors of cell (fb, nb). A frontier/gap cell is
    empty by definition (see top_frontier_cells), so its own items are always []; the 4-neighbor
    territory is the closest 'nearby work' the cockpit's evidence panel can show for a gap-mission
    target cell."""
    by_coord = {(c["fb"], c["nb"]): c for c in data["cells"]}
    neighbor_coords = [(fb + 1, nb), (fb - 1, nb), (fb, nb + 1), (fb, nb - 1)]
    tags = set()
    for nc in neighbor_coords:
        cell = by_coord.get(nc)
        if cell and cell["count"] > 0:
            tags.update(item["tag"] for item in cell["items"])
    return tags


def render_html(data, active_expedition=None, active_leg=None, running=None):
    cells_json = data["cells"]
    median_count = data.get("median_count", 0)
    max_count = data["max_count"]
    data_json = json_script(cells_json)

    axes_tip = info_btn(
        "Faithfulness (x-axis) measures how close an image stays to the original training photos, "
        "from 0 (no resemblance) to 1 (near-identical). Novelty (y-axis) measures how different an "
        "image is from everything already explored, so 1 means nothing found so far looks like it. "
        "Every image lands in exactly one cell of this grid based on those two scores; a frontier "
        "cell is an empty one next to a well-populated one, a promising, reachable gap the search "
        "hasn't filled yet. The grid uses quantile bins, so each axis is split into ranges with "
        "roughly equal numbers of images. The median frontier gate only highlights empty cells "
        "beside occupied cells at or above the median count."
    )
    dino_tip = info_btn(DINO_TIP)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS coverage map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{DARK_TOKENS}
:root {{ --frontier:#f5c542; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
{BTN_CSS}
h1 {{ font-size:18px; margin:0 0 4px; }}
p.sub {{ color:var(--text-dim); max-width:760px; font-size:13px; line-height:1.6; }}
#wrap {{ display:flex; gap:24px; margin-top:20px; flex-wrap:wrap; }}
#grid {{ display:grid; grid-template-columns: repeat({N_BINS}, 84px); grid-template-rows: repeat({N_BINS}, 84px); gap:3px; }}
@media (max-width: 640px) {{
  #grid {{ grid-template-columns: repeat({N_BINS}, minmax(30px, 1fr)); grid-template-rows: repeat({N_BINS}, minmax(30px, 1fr));
    max-width: calc(100vw - 20px); }}
  .cell {{ font-size:9px; }}
  #wrap {{ flex-direction:column; gap:16px; }}
  #panel {{ width:100%; }}
}}
.cell {{ position:relative; border-radius:4px; cursor:pointer; display:flex; align-items:center; justify-content:center;
  font-size:11px; color:#0b0b0d; font-weight:600; }}
.cell.frontier {{ outline:2px solid var(--frontier); outline-offset:-2px; }}
.cell.empty {{ background:var(--panel); color:var(--text-dim); font-weight:400; }}
#axisY {{ writing-mode:vertical-rl; transform:rotate(180deg); color:var(--text-dim); font-size:12px; text-align:center; }}
#axisX {{ color:var(--text-dim); font-size:12px; text-align:center; margin-top:4px; }}
#legend {{ display:flex; align-items:center; gap:8px; font-size:12px; color:var(--text-dim); margin-top:14px; }}
#legend .swatch {{ width:16px; height:16px; border-radius:3px; }}
#panel {{ width:280px; }}
#panel img {{ width:100%; border-radius:8px; display:none; cursor:pointer; }}
#panel .info {{ font-size:12px; color:var(--text-dim); line-height:1.7; margin-top:10px; }}
#panel .info b {{ color:var(--text); }}
#panel .viewall {{ display:none; margin-top:10px; background:var(--panel); color:var(--text);
  border:1px solid var(--border); border-radius:7px; padding:7px 14px; font-size:12.5px; cursor:pointer; }}
#panel .viewall:hover {{ border-color:#4a4a54; }}
#panel .cockpit-link {{ display:none; margin-top:8px; color:var(--accent,#7c9eff); font-size:12.5px; }}
a.navlink {{ color:#7c9eff; font-size:12.5px; text-decoration:none; }}

#modal {{ position:fixed; inset:0; background:rgba(8,8,10,0.94); backdrop-filter:blur(6px);
  display:none; z-index:100; padding:30px; overflow-y:auto; }}
#modal.open {{ display:block; }}
#modal .close {{ position:fixed; top:16px; right:22px; font-size:26px; cursor:pointer; color:var(--text-dim); }}
#modal .close:hover {{ color:var(--text); }}
#modal h2 {{ font-size:15px; margin:0 0 14px; color:var(--text); }}
#modalGrid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:8px; max-width:1200px; }}
#modalGrid .item {{ background:var(--panel); border-radius:8px; overflow:hidden; }}
#modalGrid img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; }}
#modalGrid .meta {{ font-size:10px; color:var(--text-dim); padding:5px 6px; line-height:1.5; }}
@media (max-width: 640px) {{
  #modal {{ padding:14px; }}
  #modalGrid {{ grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); }}
}}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('coverage.html', active_expedition=active_expedition, active_leg=active_leg, running=running)}
<h1>Coverage / void map{axes_tip}</h1>
<p class="sub">Same DINOv2{dino_tip}-based faithfulness (x) x novelty (y) plane as gallery.html, but at a finer {N_BINS}x{N_BINS}
grid and colored by image count instead of showing thumbnails per cell. Gold-outlined cells are
empty but sit next to a cell at or above the median occupied-cell count: the shortlist of gaps
worth targeting, rather than gaps that are empty because nothing in that region is reachable at
all. Click a cell to preview its top image, or "view all" to see every image in that cell.</p>

<div id="wrap">
  <div style="display:flex; gap:8px;">
    <div id="axisY">novelty &rarr;</div>
    <div>
      <div id="grid"></div>
      <div id="axisX">faithfulness &rarr;</div>
    </div>
  </div>
  <div id="panel">
    <img id="panelImg">
    <div class="info" id="panelInfo">Click a cell to see its highest-novelty image.</div>
    <button class="viewall" id="viewAllBtn">view all images in this cell</button>
    <a class="cockpit-link" id="cockpitLink" href="cockpit.html">Target this gap in cockpit</a>
  </div>
</div>
<div id="legend"></div>

<div id="modal">
  <span class="close" onclick="closeModal()">&times;</span>
  <h2 id="modalTitle"></h2>
  <div id="modalGrid"></div>
</div>

<script>
// json_script() only protects this declaration from a <\\/script> breakout; it does not
// HTML-escape decoded string values. Every CELLS field written into innerHTML/an attribute
// below must go through escHtml() first.
function escHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

const CELLS = {data_json};
const N_BINS = {N_BINS};
const MEDIAN_COUNT = {median_count};
const MAX_COUNT = {max_count};

function colorFor(count) {{
  if (count === 0) return null;
  const t = Math.min(1, count / MAX_COUNT);
  const r = Math.round(28 + t * (124 - 28));
  const g = Math.round(30 + t * (158 - 30));
  const b = Math.round(40 + t * (255 - 40));
  return `rgb(${{r}},${{g}},${{b}})`;
}}

const grid = document.getElementById('grid');
// render with novelty (nb) increasing upward -> reverse row order
const byCoord = {{}};
CELLS.forEach(c => byCoord[`${{c.fb}},${{c.nb}}`] = c);

for (let nb = N_BINS - 1; nb >= 0; nb--) {{
  for (let fb = 0; fb < N_BINS; fb++) {{
    const c = byCoord[`${{fb}},${{nb}}`];
    const div = document.createElement('div');
    const color = colorFor(c.count);
    div.className = 'cell' + (c.count === 0 ? ' empty' : '') + (c.frontier ? ' frontier' : '');
    if (color) div.style.background = color;
    div.textContent = c.count || (c.frontier ? '\\u2605' : '');
    div.title = `faith [${{c.faith_lo}}, ${{c.faith_hi}}) x novelty [${{c.novelty_lo}}, ${{c.novelty_hi}}) | n=${{c.count}}`;
    div.onclick = () => showCell(c);
    grid.appendChild(div);
  }}
}}

let currentCell = null;

function showCell(c) {{
  currentCell = c;
  const img = document.getElementById('panelImg');
  const info = document.getElementById('panelInfo');
  const viewAllBtn = document.getElementById('viewAllBtn');
  if (c.thumb) {{
    img.src = c.thumb;
    img.style.display = 'block';
    img.onclick = () => Lightbox.open(c.best_tag);
  }} else {{
    img.style.display = 'none';
  }}
  info.innerHTML = `<b>faith</b> [${{c.faith_lo}}, ${{c.faith_hi}})<br>`
    + `<b>novelty</b> [${{c.novelty_lo}}, ${{c.novelty_hi}})<br>`
    + `<b>count</b> ${{c.count}}${{c.frontier ? ' &mdash; frontier cell' : ''}}<br>`
    + (c.best_tag ? `<b>top image</b> ${{escHtml(c.best_tag)}}` : 'no images in this cell yet');
  viewAllBtn.style.display = c.count > 0 ? 'block' : 'none';
  document.getElementById('cockpitLink').style.display = c.frontier ? 'inline-block' : 'none';
}}

document.getElementById('viewAllBtn').addEventListener('click', () => {{
  if (currentCell) openModal(currentCell);
}});

// Click handler looks the item up by its trusted array index instead of interpolating a tag
// string into the onclick attribute, so an attacker-controlled tag can't break out of the JS
// string literal there.
function openModalItem(j) {{
  Lightbox.open(currentCell.items[j].tag);
}}

function openModal(c) {{
  document.getElementById('modalTitle').textContent =
    `${{c.count}} images | faith [${{c.faith_lo}}, ${{c.faith_hi}}) x novelty [${{c.novelty_lo}}, ${{c.novelty_hi}})`;
  document.getElementById('modalGrid').innerHTML = c.items.map((it, j) => `
    <div class="item">
      <img src="${{escHtml(it.thumb)}}" loading="lazy" data-tag="${{escHtml(it.tag)}}" title="${{escHtml(it.tag)}}" style="cursor:pointer"
           onclick="openModalItem(${{j}})">
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

const legend = document.getElementById('legend');
legend.innerHTML = `<span class="swatch" style="background:${{colorFor(1)}}"></span> 1`
  + `<span style="margin-left:6px;">median ${{MEDIAN_COUNT}}</span>`
  + `<span class="swatch" style="background:${{colorFor(MAX_COUNT)}}; margin-left:12px;"></span> max ${{MAX_COUNT}}`
  + `<span class="swatch" style="background:transparent; outline:2px solid #f5c542; margin-left:12px;"></span> frontier (empty, adjacent to a dense cell)`;
</script>
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    return html
