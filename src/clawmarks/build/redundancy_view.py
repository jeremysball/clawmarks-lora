"""
Idea 4 from Fable's exploration-tooling brainstorm (2026-07-09): a redundancy/duplicate-cluster
view. With 3000+ generated images, a meaningful fraction are likely near-copies of each other
(same subject/settings, different seed noise), which inflates how much of the map actually
looks "covered." This clusters images by DINOv2 cosine similarity at an adjustable threshold
(connected components over the precomputed top-16 nearest-neighbor edges) so you can see the
population's true effective size and which "different" bins are actually duplicates.

Clustering happens client-side in JS (union-find over ~3400 nodes / ~54k edges is instant), so
one threshold slider can be dragged live instead of needing a rebuild per threshold.

Depends on solution_map.py's compute_data(), which includes top-16 neighbors WITH cosine scores
(the separate similarity_index.py only stores neighbor identity, not the score, which isn't
enough to threshold on). compute_data(sweep_dir, deps) takes solution_map's result via
`deps["solution-map"]`, served live by curation_server.py through LiveCache's
depends_on=["solution-map"] mechanism, not a standalone build step.
"""
import json
import math
import os

from clawmarks.shared_ui import nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, INFOTIP_CSS, info_btn, json_script


def compute_data(sweep_dir, deps):
    sim_scored = deps["solution-map"]["similarity_scored"]

    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    by_tag = {m["tag"]: m for m in manifest}

    thumbs = {}
    for tag in sim_scored:
        m = by_tag.get(tag)
        if not m:
            continue
        thumbs[tag] = f"thumbs/{tag}.jpg" if os.path.exists(f"{sweep_dir}/thumbs/{tag}.jpg") else os.path.basename(m["file"])

    meta = {t: {"prompt_name": m["prompt_name"], "novelty": round(m["novelty"], 4),
                "faith": round(m["centroid_sim"], 4)} for t, m in by_tag.items() if t in sim_scored}

    return {"sim_scored": sim_scored, "thumbs": thumbs, "meta": meta}


def render_html(data):
    sim_scored = data["sim_scored"]
    thumbs = data["thumbs"]
    meta = data["meta"]

    cluster_tip = info_btn(
        "A cluster here is a connected component: if A is similar enough to B, and B is similar "
        "enough to C, then A, B, and C all land in one cluster, even if A and C aren't directly "
        "similar to each other. So a cluster can be a chain of gradual drift rather than a tight "
        "group of near-duplicates. Read a big cluster as 'this region is redundant,' not as 'every "
        "pair in it looks alike.'"
    )

    # Size the slider to the data, not a fixed near-duplicate range. A diverse single-round seed
    # run's closest pairs can sit near cosine 0.78, below the old hardcoded 0.80 minimum, so every
    # slider position produced an empty graph and the page looked broken. Span the actual edge
    # range (padded to a 0.05 grid) and default where only the tightest ~5% of edges survive, so
    # the strongest clusters always show without merging the whole population.
    all_scores = sorted(s for lst in sim_scored.values() for _, s in lst)
    if all_scores:
        slider_min = math.floor(all_scores[0] * 20) / 20
        slider_max = math.ceil(all_scores[-1] * 20) / 20
        if slider_max - slider_min < 0.1:
            slider_max = slider_min + 0.1
        default_thresh = round(all_scores[int(len(all_scores) * 0.95)], 3)
        default_thresh = min(max(default_thresh, slider_min), slider_max)
    else:
        slider_min, slider_max, default_thresh = 0.80, 0.99, 0.93

    edges_json = json_script(sim_scored)
    thumbs_json = json_script(thumbs)
    meta_json = json_script(meta)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS redundancy clusters</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee; --text-dim:#9a9aa4; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
h1 {{ font-size:18px; margin:0 0 4px; }}
p.sub {{ color:var(--text-dim); max-width:760px; font-size:13px; line-height:1.6; }}
a.navlink {{ color:#7c9eff; font-size:12.5px; text-decoration:none; }}
#bar {{ display:flex; gap:14px; align-items:center; margin:16px 0; font-size:12.5px; color:var(--text-dim); flex-wrap:wrap; }}
#bar input[type=range] {{ width:280px; }}
#summary {{ font-size:13px; color:var(--text-dim); margin-bottom:14px; }}
#summary b {{ color:var(--text); }}
#clusters {{ display:flex; flex-direction:column; gap:10px; }}
.cluster {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:8px 10px; }}
.cluster .head {{ font-size:11.5px; color:var(--text-dim); margin-bottom:6px; }}
.cluster .strip {{ display:flex; gap:5px; overflow-x:auto; }}
.cluster .strip img {{ width:64px; height:64px; object-fit:cover; border-radius:5px; flex-shrink:0; }}
@media (max-width: 640px) {{
  #bar input[type=range] {{ width:100%; flex:1; }}
  .cluster .strip img {{ width:52px; height:52px; }}
}}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('redundancy.html')}
<h1>Redundancy clusters{cluster_tip}</h1>
<p class="sub">Connected components over each image's top-16 DINOv2 nearest neighbors, using
only edges at or above the similarity threshold below. Higher threshold = stricter "near-
duplicate," lower threshold = looser "similar family." This tells you the effective diversity
of the population, not just its raw count.</p>

<div id="bar">
  <label>Similarity threshold &ge; <input type="range" id="thresh" min="{slider_min:.3f}" max="{slider_max:.3f}" step="0.005" value="{default_thresh:.3f}"></label>
  <span id="threshLabel"></span>
</div>
<div id="summary"></div>
<div id="clusters"></div>

<script>
// json_script() only protects this declaration from a </script> breakout; it does not
// HTML-escape decoded string values. Every EDGES/THUMBS/META field written into innerHTML/an
// attribute below must go through escHtml() first.
function escHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
}}

const EDGES = {edges_json};
const THUMBS = {thumbs_json};
const META = {meta_json};
const TAGS = Object.keys(EDGES);
const idx = {{}};
TAGS.forEach((t, i) => idx[t] = i);
let renderedClusters = [];

// Click handler looks the tag up by trusted (cluster, item) indices instead of interpolating
// a tag string into the onclick attribute, so an attacker-controlled tag can't break out of
// the JS string literal there.
function openClusterItem(gi, ti) {{
  Lightbox.open(renderedClusters[gi][ti]);
}}

function cluster(threshold) {{
  const parent = TAGS.map((_, i) => i);
  function find(x) {{ while (parent[x] !== x) {{ parent[x] = parent[parent[x]]; x = parent[x]; }} return x; }}
  function union(a, b) {{ const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; }}
  TAGS.forEach(t => {{
    (EDGES[t] || []).forEach(([nbr, score]) => {{
      if (score >= threshold && idx[nbr] !== undefined) union(idx[t], idx[nbr]);
    }});
  }});
  const groups = {{}};
  TAGS.forEach((t, i) => {{
    const r = find(i);
    (groups[r] = groups[r] || []).push(t);
  }});
  return Object.values(groups);
}}

function render() {{
  const threshold = parseFloat(document.getElementById('thresh').value);
  document.getElementById('threshLabel').textContent = threshold.toFixed(3);
  const groups = cluster(threshold);
  const multi = groups.filter(g => g.length > 1).sort((a, b) => b.length - a.length);
  const singletons = groups.length - multi.length;
  document.getElementById('summary').innerHTML =
    `<b>${{groups.length}}</b> effective clusters out of ${{TAGS.length}} images `
    + `(<b>${{singletons}}</b> singletons, <b>${{multi.length}}</b> multi-image clusters, `
    + `largest = <b>${{multi.length ? multi[0].length : 0}}</b> images)`;
  renderedClusters = multi.slice(0, 60);
  document.getElementById('clusters').innerHTML = renderedClusters.map((g, gi) => {{
    const rep = g.reduce((best, t) => (META[t] && (!best || META[t].novelty > META[best].novelty)) ? t : best, null);
    return `<div class="cluster">
      <div class="head">${{g.length}} images | representative: ${{escHtml(rep)}} (${{META[rep] ? escHtml(META[rep].prompt_name) : ''}})</div>
      <div class="strip">${{g.map((t, ti) => `<img loading="lazy" src="${{escHtml(THUMBS[t] || '')}}" data-tag="${{escHtml(t)}}" title="${{escHtml(t)}}" style="cursor:pointer" onclick="openClusterItem(${{gi}}, ${{ti}})">`).join('')}}</div>
    </div>`;
  }}).join('') + (multi.length > 60 ? `<p style="color:#9a9aa4;font-size:12px;">...and ${{multi.length - 60}} more clusters not shown</p>` : '');
}}

document.getElementById('thresh').addEventListener('input', render);
render();
</script>
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    return html
