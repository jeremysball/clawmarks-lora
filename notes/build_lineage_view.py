"""
Idea 7 from Fable's exploration-tooling brainstorm (2026-07-09): a lineage tree for exploit
chains, showing whether mutating near a parent image actually improves faithfulness/novelty or
just wobbles. This requires a "parent_tag" field on generated images, which round 1's driver
never recorded and round 2's driver (notes/run_uncanny_allnight2.py) only started recording
after a code patch landed mid-run; the currently-running round-2 process has the old function
in memory and won't tag parents until it's restarted. So this script degrades gracefully: if no
image in scored_manifest.json has a parent_tag yet, it writes an explanatory placeholder page
instead of an empty tree.

Run after scored_manifest.json exists: python3 notes/build_lineage_view.py
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from shared_ui import nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, write_lightbox_asset, write_scrollnav_asset

SWEEP_DIR = "/workspace/trent-with-smart-prompts/notes/uncanny_sweep"
write_lightbox_asset(SWEEP_DIR)
write_scrollnav_asset(SWEEP_DIR)

with open(f"{SWEEP_DIR}/scored_manifest.json") as f:
    manifest = json.load(f)

by_tag = {m["tag"]: m for m in manifest}
has_lineage = any(m.get("parent_tag") for m in manifest)

PLACEHOLDER = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS lineage tree</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; }}
body {{ background:#0b0b0d; color:#eaeaee; font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
h1 {{ font-size:18px; }}
p {{ color:#9a9aa4; max-width:640px; font-size:13px; line-height:1.7; }}
a.navlink {{ color:#7c9eff; font-size:12.5px; text-decoration:none; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
</style></head><body>
{nav_bar_html('lineage.html')}
<h1>Lineage tree</h1>
<p>No image in this dataset carries a <code>parent_tag</code> yet, so there's nothing to draw a
tree from. Round 1's driver never recorded which parent an exploit step mutated near, and while
round 2's driver was patched on 2026-07-09 to record it, the patch only takes effect the next
time that process is (re)started, not for a run already in progress.</p>
<p>Once round 2 restarts (or a future round runs) with the patch active, re-run
<code>notes/build_lineage_view.py</code> and this page will render exploit chains: each parent
image and the children it spawned, with faithfulness/novelty deltas at each step, to show
whether exploiting actually improves on its parent or just wobbles.</p>
</body></html>"""

if not has_lineage:
    with open(f"{SWEEP_DIR}/lineage.html", "w") as f:
        f.write(PLACEHOLDER)
    print(f"wrote {SWEEP_DIR}/lineage.html (placeholder: no parent_tag data exists yet)", flush=True)
else:
    children_by_parent = {}
    roots = []
    for m in manifest:
        p = m.get("parent_tag")
        if p and p in by_tag:
            children_by_parent.setdefault(p, []).append(m["tag"])
        elif p:
            roots.append(m["tag"])  # parent existed but isn't in this dataset (e.g. a user pick)

    def node_html(tag, depth=0):
        m = by_tag[tag]
        children = children_by_parent.get(tag, [])
        child_html = "".join(node_html(c, depth + 1) for c in children)
        return (f'<li><div class="node" onclick="Lightbox.open(\'{tag}\')"><b>{tag}</b> faith={m["centroid_sim"]:.3f} '
                f'novelty={m["novelty"]:.3f}</div>{"<ul>" + child_html + "</ul>" if children else ""}</li>')

    parent_tags_present = set(children_by_parent.keys()) | set(roots)
    top_level = [t for t in by_tag if t not in {c for cs in children_by_parent.values() for c in cs}]
    tree_html = "<ul>" + "".join(node_html(t) for t in top_level if t in children_by_parent) + "</ul>"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS lineage tree</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; }}
body {{ background:#0b0b0d; color:#eaeaee; font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
ul {{ list-style:none; padding-left:20px; border-left:1px solid #2a2a30; }}
.node {{ font-size:12.5px; padding:3px 0; color:#9a9aa4; cursor:pointer; }}
.node b {{ color:#eaeaee; }}
a.navlink {{ color:#7c9eff; font-size:12.5px; text-decoration:none; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
</style></head><body>
{nav_bar_html('lineage.html')}
<h1>Lineage tree</h1>
{tree_html}
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
</body></html>"""
    with open(f"{SWEEP_DIR}/lineage.html", "w") as f:
        f.write(html)
    print(f"wrote {SWEEP_DIR}/lineage.html ({len(children_by_parent)} parent nodes)", flush=True)
