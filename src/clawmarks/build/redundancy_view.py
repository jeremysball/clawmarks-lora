"""
Idea 4 from Fable's exploration-tooling brainstorm (2026-07-09): a redundancy/duplicate-cluster
view. With 3000+ generated images, a meaningful fraction are likely near-copies of each other
(same subject/settings, different seed noise), which inflates how much of the map actually
looks "covered." This clusters images by DINOv2 cosine similarity at an adjustable threshold
(connected components over the precomputed top-16 nearest-neighbor edges) so you can see the
population's true effective size and which "different" bins are actually duplicates.

Clustering happens client-side in JS (union-find over ~3400 nodes / ~54k edges is instant), so
one threshold slider can be dragged live instead of needing a rebuild per threshold.

Depends on build_solution_map.py's similarity_scored.json (top-16 neighbors WITH cosine
scores; the original build_similarity_index.py only stores neighbor identity, not the score,
which isn't enough to threshold on).

Run after similarity_scored.json exists: python3 -m clawmarks.build.redundancy_view
"""
import json, os, sys

from clawmarks.config import SWEEP_DIR
from clawmarks.shared_ui import (
    nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, write_lightbox_asset, write_scrollnav_asset,
    write_infotip_asset, INFOTIP_CSS, info_btn,
)


def main(argv=None):
    write_lightbox_asset(SWEEP_DIR)
    write_scrollnav_asset(SWEEP_DIR)
    write_infotip_asset(SWEEP_DIR)

    cluster_tip = info_btn(
        "A cluster here is a connected component: if A is similar enough to B, and B is similar "
        "enough to C, then A, B, and C all land in one cluster, even if A and C aren't directly "
        "similar to each other. So a cluster can be a chain of gradual drift rather than a tight "
        "group of near-duplicates. Read a big cluster as 'this region is redundant,' not as 'every "
        "pair in it looks alike.'"
    )

    with open(f"{SWEEP_DIR}/similarity_scored.json") as f:
        sim_scored = json.load(f)

    with open(f"{SWEEP_DIR}/scored_manifest.json") as f:
        manifest = json.load(f)

    by_tag = {m["tag"]: m for m in manifest}

    thumbs = {}
    for tag in sim_scored:
        m = by_tag.get(tag)
        if not m:
            continue
        thumbs[tag] = f"thumbs/{tag}.jpg" if os.path.exists(f"{SWEEP_DIR}/thumbs/{tag}.jpg") else os.path.basename(m["file"])

    edges_json = json.dumps(sim_scored)
    thumbs_json = json.dumps(thumbs)
    meta_json = json.dumps({t: {"prompt_name": m["prompt_name"], "novelty": round(m["novelty"], 4),
                                 "faith": round(m["centroid_sim"], 4)} for t, m in by_tag.items() if t in sim_scored})

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
  <label>Similarity threshold &ge; <input type="range" id="thresh" min="0.80" max="0.99" step="0.005" value="0.93"></label>
  <span id="threshLabel"></span>
</div>
<div id="summary"></div>
<div id="clusters"></div>

<script>
const EDGES = {edges_json};
const THUMBS = {thumbs_json};
const META = {meta_json};
const TAGS = Object.keys(EDGES);
const idx = {{}};
TAGS.forEach((t, i) => idx[t] = i);

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
  document.getElementById('clusters').innerHTML = multi.slice(0, 60).map(g => {{
    const rep = g.reduce((best, t) => (META[t] && (!best || META[t].novelty > META[best].novelty)) ? t : best, null);
    return `<div class="cluster">
      <div class="head">${{g.length}} images | representative: ${{rep}} (${{META[rep] ? META[rep].prompt_name : ''}})</div>
      <div class="strip">${{g.map(t => `<img loading="lazy" src="${{THUMBS[t] || ''}}" data-tag="${{t}}" title="${{t}}" style="cursor:pointer" onclick="Lightbox.open('${{t}}')">`).join('')}}</div>
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

    with open(f"{SWEEP_DIR}/redundancy.html", "w") as f:
        f.write(html)

    print(f"wrote {SWEEP_DIR}/redundancy.html ({len(sim_scored)} images with similarity edges)", flush=True)


if __name__ == "__main__":
    main()
