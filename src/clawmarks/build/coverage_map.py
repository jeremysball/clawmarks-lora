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
import html as html_lib
import json
import math
import os

from clawmarks.shared_ui import (
    CONTROL_CSS,
    DINO_TIP,
    INFOTIP_CSS,
    MOBILE_BASE_CSS,
    SULFUR_CSS,
    SULFUR_FONT_CSS,
    TOPNAV_CSS,
    info_btn,
    json_script,
    nav_bar_html,
    scoped_href,
)
from clawmarks.workspace_context import WorkspaceContext, generated_image_url
from clawmarks.durable_records import sha256_json

N_BINS = 8
METRIC_DOMAINS = {
    "faithfulness": [-1.0, 1.0],
    "novelty": [0.0, 2.0],
}


def compute_data(sweep_dir):
    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    faith_vals = sorted(m["centroid_sim"] for m in manifest)
    novelty_vals = sorted(m["novelty"] for m in manifest)

    def bin_edges(vals, n, domain):
        if not vals:
            lo, hi = domain
            step = (hi - lo) / n
            return [lo + i * step for i in range(1, n)]
        return [vals[int(i * len(vals) / n)] for i in range(1, n)]

    faith_edges = [METRIC_DOMAINS["faithfulness"][0], *bin_edges(faith_vals, N_BINS, METRIC_DOMAINS["faithfulness"]), METRIC_DOMAINS["faithfulness"][1]]
    novelty_edges = [METRIC_DOMAINS["novelty"][0], *bin_edges(novelty_vals, N_BINS, METRIC_DOMAINS["novelty"]), METRIC_DOMAINS["novelty"][1]]

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

    def cell_ranges(fb, nb):
        return {
            "faith_lo": faith_edges[fb], "faith_hi": faith_edges[fb + 1],
            "novelty_lo": novelty_edges[nb], "novelty_hi": novelty_edges[nb + 1],
        }

    def actionable(ranges):
        return all(math.isfinite(ranges[key]) for key in ranges) and (
            ranges["faith_lo"] < ranges["faith_hi"]
            and ranges["novelty_lo"] < ranges["novelty_hi"]
        )

    frontier = set()
    for fb in range(N_BINS):
        for nb in range(N_BINS):
            if (fb, nb) in counts:
                continue
            if not actionable(cell_ranges(fb, nb)):
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
            ranges = cell_ranges(fb, nb)
            cells_json.append({
                "fb": fb, "nb": nb, "count": n,
                "frontier": (fb, nb) in frontier and actionable(ranges),
                **ranges,
                "thumb": (f"thumbs/{best['tag']}.jpg" if best and os.path.exists(f"{sweep_dir}/thumbs/{best['tag']}.jpg")
                          else (os.path.basename(best["file"]) if best else None)),
                "best_tag": best["tag"] if best else None,
                "items": [item_summary(m) for m in items],
            })

    canonical_cell_ranges = [
        {key: cell[key] for key in ("fb", "nb", "faith_lo", "faith_hi", "novelty_lo", "novelty_hi")}
        for cell in cells_json
    ]
    return {
        "cells": cells_json, "median_count": median_count, "max_count": max_count,
        "metric_domains": METRIC_DOMAINS,
        "binning_version": sha256_json({"cells": canonical_cell_ranges, "domains": METRIC_DOMAINS}),
        "real_anchor_tags": sorted({m["nearest_real"] for m in manifest if m.get("nearest_real")}),
    }


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


def render_html(
    data, active_expedition=None, active_leg=None, running=None,
    context: WorkspaceContext | None = None,
    focus=None,
):
    focus = focus or (context.focus if context is not None else None)
    cells_json = data["cells"]
    if context is not None:
        cells_json = [
            {
                **cell,
                "thumb": (
                    generated_image_url(cell["best_tag"], context, thumbnail=True)
                    if cell.get("best_tag") else cell.get("thumb")
                ),
                "items": [
                    {
                        **item,
                        "thumb": generated_image_url(
                            item["tag"], context, thumbnail=True
                        ),
                    }
                    for item in cell["items"]
                ],
            }
            for cell in cells_json
        ]
    median_count = data.get("median_count", 0)
    max_count = data["max_count"]
    data_json = json_script(cells_json)
    context_json = json.dumps({"expedition": active_expedition, "leg": active_leg})
    anchor_options = "".join(
        f'<option value="{html_lib.escape(tag, quote=True)}">{html_lib.escape(tag)}</option>'
        for tag in data.get("real_anchor_tags", [])
    )

    axes_tip = info_btn(
        "Faithfulness (x-axis) measures how close an image stays to the original training photos, "
        "from 0 (no resemblance) to 1 (near-identical). Novelty (y-axis) measures how different an "
        "image is from everything already explored, so 1 means nothing found so far looks like it. "
        "Every image lands in exactly one cell of this grid based on those two scores; a frontier "
        "cell is an empty one next to a well-populated one, a reachable gap the search "
        "hasn't filled yet. The grid uses quantile bins, so each axis is split into ranges with "
        "roughly equal numbers of images. The median frontier gate only highlights empty cells "
        "beside occupied cells at or above the median count."
    )
    dino_tip = info_btn(DINO_TIP)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS coverage map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
body {{ margin:0; padding:22px 26px; }}
h1 {{ font-size:22px; margin:24px 0 4px; letter-spacing:0.02em; text-transform:uppercase; }}
p.sub {{ color:var(--text-soft); max-width:760px; font-size:13px; line-height:1.6; border-bottom:1px solid var(--rule); padding-bottom:14px; }}
#wrap {{ display:flex; gap:24px; margin-top:20px; flex-wrap:wrap; }}
#grid {{ display:grid; grid-template-columns: repeat({N_BINS}, 84px); grid-template-rows: repeat({N_BINS}, 84px); gap:3px; }}
@media (max-width: 640px) {{
  #grid {{ grid-template-columns: repeat({N_BINS}, minmax(30px, 1fr)); grid-template-rows: repeat({N_BINS}, minmax(30px, 1fr));
    max-width: calc(100vw - 20px); }}
  .cell {{ font-size:9px; }}
  #wrap {{ flex-direction:column; gap:16px; }}
  #panel {{ width:100%; }}
}}
.cell {{ position:relative; cursor:pointer; display:flex; align-items:center; justify-content:center;
  font-size:11px; color:var(--ink); font-weight:600; border:1px solid var(--rule); }}
.cell.frontier {{
  border:2px solid var(--ink);
  background-image:repeating-linear-gradient(45deg,var(--sulfur) 0 6px,transparent 6px 12px);
  background-color:var(--paper);
  color:var(--ink);
  font:800 14px/1 var(--font-display);
}}
.cell.frontier::after {{
  content:"F";
  position:absolute; top:2px; right:4px; font:600 10px/1 var(--font-mono);
  color:var(--ink); background:var(--sulfur); padding:2px 4px; border:1px solid var(--ink);
}}
.cell.empty {{ background:var(--paper); color:var(--text-soft); font-weight:400; border:1px solid var(--rule); }}
#axisY {{ writing-mode:vertical-rl; transform:rotate(180deg); color:var(--text-soft); font-size:12px; text-align:center; }}
#axisX {{ color:var(--text-soft); font-size:12px; text-align:center; margin-top:4px; }}
#legend {{ display:flex; align-items:center; gap:8px; font-size:12px; color:var(--text-soft); margin-top:14px; border-top:1px solid var(--rule); padding-top:10px; }}
#legend .swatch {{ width:16px; height:16px; border:1px solid var(--rule); }}
#legend .swatch.frontier-swatch {{
  background-image:repeating-linear-gradient(45deg,var(--sulfur) 0 4px,transparent 4px 8px);
  background-color:var(--paper);
  border:2px solid var(--ink);
}}
#panel {{ width:280px; }}
#panel img {{ width:100%; display:none; cursor:pointer; border:1px solid var(--ink); }}
#panel .info {{ font-size:12px; color:var(--text-soft); line-height:1.7; margin-top:10px; border-top:1px solid var(--rule); padding-top:10px; }}
#panel .info b {{ color:var(--ink); }}
#panel .viewall {{ display:none; margin-top:10px; background:var(--paper); color:var(--ink);
  border:1px solid var(--ink); padding:7px 14px; font-size:12.5px; cursor:pointer; font:600 13px/1 var(--font-body); box-shadow:2px 2px 0 var(--ink); }}
#panel .viewall:hover {{ box-shadow:3px 3px 0 var(--ink); }}
#panel .viewall:active {{ transform:translate(2px,2px); box-shadow:none; }}
#panel .cockpit-link {{ display:none; margin-top:8px; color:var(--ink); font-size:12.5px; text-decoration:underline; }}
a.navlink {{ color:var(--ink); font-size:12.5px; text-decoration:underline; }}

#modal {{ position:fixed; inset:0; background:rgba(8,8,10,0.94); backdrop-filter:blur(6px);
  display:none; z-index:100; padding:30px; overflow-y:auto; }}
#modal.open {{ display:block; }}
#modal .close {{ position:fixed; top:16px; right:22px; font-size:26px; cursor:pointer; color:var(--guide-ink); }}
#modal .close:hover {{ color:var(--paper); }}
#modal h2 {{ font-size:15px; margin:0 0 14px; color:var(--guide-ink); }}
#modalGrid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:8px; max-width:1200px; }}
#modalGrid .item {{ background:var(--paper); border:1px solid var(--rule); overflow:hidden; }}
#modalGrid img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; }}
#modalGrid .meta {{ font-size:10px; color:var(--text-soft); padding:5px 6px; line-height:1.5; }}
@media (max-width: 640px) {{
  #modal {{ padding:14px; }}
  #modalGrid {{ grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); }}
}}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('coverage.html', active_expedition=active_expedition, active_leg=active_leg, running=running, focus=focus)}
<h1>Coverage / void map{axes_tip}</h1>
<p class="sub">Same DINOv2{dino_tip}-based faithfulness (x) x novelty (y) plane as gallery.html, but at a finer {N_BINS}x{N_BINS}
grid and colored by image count instead of showing thumbnails per cell. Gold-outlined cells are
empty but sit next to a cell at or above the median occupied-cell count: the shortlist of gaps
 worth examining, rather than gaps that are empty because nothing in that region is reachable at
all. Click a cell to preview its top image, or "view all" to see every image in that cell.</p>

<div id="wrap">
  <div style="display:flex; gap:8px;">
    <div id="axisY">novelty &rarr;</div>
    <div>
      <div id="grid" role="grid" aria-label="Coverage frontier"></div>
      <div id="axisX">faithfulness &rarr;</div>
    </div>
  </div>
  <div id="panel">
    <img id="panelImg">
     <div class="info" id="panelInfo">Click a cell to see its highest-novelty image.</div>
     <button class="viewall" id="viewAllBtn">view all images in this cell</button>
     <label for="realAnchor">REAL-ART ANCHOR
       <select id="realAnchor"><option value="">none selected</option>{anchor_options}</select>
     </label>
     <label for="focusLabel">FOCUS LABEL
       <input id="focusLabel" type="text" placeholder="Name this frontier">
     </label>
     <label for="focusQuestion">QUESTION
       <textarea id="focusQuestion" placeholder="What do you want to learn from this frontier?"></textarea>
     </label>
     <button id="createCoverageFocus" class="raised-control" type="button" disabled>Create Focus</button>
     <div id="selectionStatus" role="status" aria-live="polite"></div>
     <a class="cockpit-link" id="cockpitLink" href="{scoped_href('/cockpit.html', active_expedition, active_leg, focus)}">Target this gap in cockpit</a>
   </div>
 </div>
 <table id="coverageValues" aria-label="Coverage values">
   <caption>Coverage values</caption>
   <thead><tr><th>Faithfulness</th><th>Novelty</th><th>Count</th><th>State</th></tr></thead>
   <tbody id="coverageValuesBody"></tbody>
 </table>
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
const CONTEXT = {context_json};
const DATA = {json.dumps({"metric_domains": data.get("metric_domains", METRIC_DOMAINS), "binning_version": data.get("binning_version", "")})};

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
    const div = document.createElement(c.frontier ? 'button' : 'div');
    const color = colorFor(c.count);
    div.className = 'cell' + (c.count === 0 ? ' empty' : '') + (c.frontier ? ' frontier' : '');
    if (!c.frontier) div.setAttribute('role', 'gridcell');
    div.type = 'button';
    div.setAttribute('aria-rowindex', String(N_BINS - nb));
    div.setAttribute('aria-colindex', String(fb + 1));
    div.setAttribute('aria-label',
      (c.frontier ? 'frontier cell, ' : 'cell, ') +
      `faith [${{c.faith_lo}}, ${{c.faith_hi}}) novelty [${{c.novelty_lo}}, ${{c.novelty_hi}}) count ${{c.count}}`);
    if (color) div.style.background = color;
    div.textContent = c.count || '';
    div.title = `faith [${{c.faith_lo}}, ${{c.faith_hi}}) x novelty [${{c.novelty_lo}}, ${{c.novelty_hi}}) | n=${{c.count}}${{c.frontier ? ' (frontier)' : ''}}`;
    div.onclick = () => showCell(c);
    grid.appendChild(div);
  }}
}}

const valuesBody = document.getElementById('coverageValuesBody');
CELLS.forEach(c => {{
  const row = document.createElement('tr');
  row.innerHTML = `<td>${{c.faith_lo}} to ${{c.faith_hi}}</td><td>${{c.novelty_lo}} to ${{c.novelty_hi}}</td>`
    + `<td>${{c.count}}</td><td>${{c.frontier ? 'frontier' : (c.count ? 'occupied' : 'empty')}}</td>`;
  row.onclick = () => showCell(c);
  row.tabIndex = 0;
  row.addEventListener('keydown', e => {{
    if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); showCell(c); }}
  }});
  valuesBody.appendChild(row);
}});

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
  document.getElementById('createCoverageFocus').disabled = !c.frontier || !CONTEXT.expedition || !CONTEXT.leg;
}}

function context_url(path, created_context) {{
  const params = new URLSearchParams();
  if (created_context.expedition) params.set('expedition', created_context.expedition);
  if (created_context.leg) params.set('leg', created_context.leg);
  if (created_context.focus_id) params.set('focus_id', created_context.focus_id);
  return `${{path}}?${{params.toString()}}`;
}}

function adjacentTags(c) {{
  const tags = new Set();
  [[c.fb + 1, c.nb], [c.fb - 1, c.nb], [c.fb, c.nb + 1], [c.fb, c.nb - 1]].forEach(([fb, nb]) => {{
    const neighbor = byCoord[`${{fb}},${{nb}}`];
    if (neighbor && neighbor.count > 0) neighbor.items.forEach(item => tags.add(item.tag));
  }});
  return Array.from(tags);
}}

async function createCoverageFocus() {{
  const status = document.getElementById('selectionStatus');
  if (!currentCell || !currentCell.frontier) return;
  const anchor = document.getElementById('realAnchor').value;
  const payload = {{
    scope: {{expedition: CONTEXT.expedition, leg: CONTEXT.leg}},
    label: document.getElementById('focusLabel').value,
    source: {{
      view: 'coverage', kind: 'coverage_frontier',
      score_ranges: {{
        faithfulness: [currentCell.faith_lo, currentCell.faith_hi],
        novelty: [currentCell.novelty_lo, currentCell.novelty_hi],
      }},
      adjacent_member_tags: adjacentTags(currentCell),
      real_anchor_tags: anchor ? [anchor] : [],
      coverage_hint: {{row: currentCell.nb, column: currentCell.fb,
        domains: DATA.metric_domains, binning_version: DATA.binning_version}},
    }},
    question: document.getElementById('focusQuestion').value,
    observation: '', hypothesis_text: '', test_contract: null,
  }};
  status.textContent = 'Creating Focus...';
  try {{
    const response = await fetch('/api/foci', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify(payload),
    }});
    const created = await response.json();
    if (!response.ok) throw new Error(created.error || 'Focus creation failed');
    window.location.href = context_url('/explore.html', {{...CONTEXT, focus_id: created.focus_id}});
  }} catch (error) {{
    status.textContent = error.message;
  }}
}}

document.getElementById('viewAllBtn').addEventListener('click', () => {{
  if (currentCell) openModal(currentCell);
}});
document.getElementById('createCoverageFocus').addEventListener('click', createCoverageFocus);

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
  + `<span class="swatch frontier-swatch" style="margin-left:12px;"></span> F frontier (empty, adjacent to a dense cell)`;
</script>
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
<script src="/shared-ui.js"></script>
</body></html>"""

    return html
