"""
Idea 7 from Fable's exploration-tooling brainstorm (2026-07-09): a lineage tree for exploit
chains, showing whether mutating near a parent image actually improves faithfulness/novelty or
just wobbles. This requires a "parent_tag" field on generated images, which round 1's driver
never recorded and round 2's driver (search/driver.py) only started recording after a code
patch landed mid-run. So this script degrades gracefully: if no image in scored_manifest.json
has a parent_tag yet, it writes an explanatory placeholder page instead of an empty tree.

Run after scored_manifest.json exists: python3 -m clawmarks.build.lineage_view
"""
import html
import json

from clawmarks.shared_ui import (
    CONTROL_CSS,
    INFOTIP_CSS,
    MOBILE_BASE_CSS,
    SULFUR_CSS,
    SULFUR_FONT_CSS,
    TOPNAV_CSS,
    nav_bar_html,
    scoped_href,
)


def compute_data(sweep_dir):
    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    by_tag = {m["tag"]: m for m in manifest}
    has_lineage = any(m.get("parent_tag") for m in manifest)

    if not has_lineage:
        return {"has_lineage": False}

    children_by_parent = {}
    for m in manifest:
        p = m.get("parent_tag")
        if p and p in by_tag:
            children_by_parent.setdefault(p, []).append(m["tag"])

    return {"has_lineage": True, "by_tag": by_tag, "children_by_parent": children_by_parent}


def render_html(data, active_expedition=None, active_leg=None, running=None, focus=None):
    if not data["has_lineage"]:
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS lineage tree</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
body {{ margin:0; padding:24px; }}
h1 {{ font-size:18px; }}
p {{ color:var(--text-soft); max-width:640px; font-size:13px; line-height:1.7; }}
a.navlink {{ color:var(--ink); font-size:12.5px; text-decoration:underline; }}
{INFOTIP_CSS}
</style></head><body>
{nav_bar_html('lineage.html', active_expedition=active_expedition, active_leg=active_leg, running=running, focus=focus)}
<h1>Lineage tree</h1>
<p>No image in this dataset carries a <code>parent_tag</code> yet, so there's nothing to draw a
tree from (placeholder page). Round 1's driver never recorded which parent an exploit step mutated near, and while
round 2's driver was patched on 2026-07-09 to record it, the patch only takes effect the next
time that process is (re)started, not for a run already in progress.</p>
<p>Once round 2 restarts (or a future round runs) with the patch active, reload this page and it
will render exploit chains: each parent image and the children it spawned, with faithfulness/novelty
deltas at each step, to show whether exploiting actually improves on its parent or just wobbles.</p>
<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
<script src="/shared-ui.js"></script>
</body></html>"""

    by_tag = data["by_tag"]
    children_by_parent = data["children_by_parent"]

    def node_html(tag, depth=0):
        m = by_tag[tag]
        children = children_by_parent.get(tag, [])
        child_html = "".join(node_html(c, depth + 1) for c in children)
        # tag is written into an HTML attribute and text node; escaping it here (rather than
        # embedding it as a JS string literal in an inline onclick, as before) means a single
        # html.escape() covers the whole sink instead of needing separate JS-string and
        # HTML-attribute escaping layers. The click handler reads it back via data-tag below.
        safe_tag = html.escape(tag, quote=True)
        return (f'<li><div class="node" data-tag="{safe_tag}"><b>{safe_tag}</b> faith={m["centroid_sim"]:.3f} '
                f'novelty={m["novelty"]:.3f}</div>{"<ul>" + child_html + "</ul>" if children else ""}</li>')

    top_level = [t for t in by_tag if t not in {c for cs in children_by_parent.values() for c in cs}]
    tree_html = "<ul>" + "".join(node_html(t) for t in top_level if t in children_by_parent) + "</ul>"

    page_html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS lineage tree</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
body {{ margin:0; padding:24px; }}
ul {{ list-style:none; padding-left:20px; border-left:1px solid var(--rule); }}
ul li > .node {{ padding:6px 0; border-bottom:1px solid var(--rule); }}
ul li:last-child > .node {{ border-bottom:none; }}
.node {{ font-size:12.5px; color:var(--text-soft); cursor:pointer; }}
.node b {{ color:var(--ink); }}
a.navlink {{ color:var(--ink); font-size:12.5px; text-decoration:underline; }}
{INFOTIP_CSS}
</style></head><body>
{nav_bar_html('lineage.html', active_expedition=active_expedition, active_leg=active_leg, running=running, focus=focus)}
<h1>Lineage tree</h1>
<p><a class="navlink" href="{scoped_href('/cockpit.html', active_expedition, active_leg, focus)}">Continue this lineage in cockpit</a></p>
{tree_html}
<script>
document.querySelectorAll('.node[data-tag]').forEach(el => {{
  el.addEventListener('click', () => Lightbox.open(el.dataset.tag));
}});
</script>
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
<script src="/shared-ui.js"></script>
</body></html>"""
    return page_html
