"""
Landing page linking every exploration/curation tool built for the uncanny-frontier search, so
there's one page to bookmark instead of remembering file names. Static, no data dependencies of
its own.

Run: python3 -m clawmarks.build.explore_hub
"""

import os, sys

from clawmarks.config import SWEEP_DIR
from clawmarks.shared_ui import MOBILE_BASE_CSS, INFOTIP_CSS, info_btn, write_infotip_asset

TOOLS = [
    ("scan.html", "Scan gallery", "Every image, sortable/filterable/searchable, with a lightbox, similarity browsing, and the pick-as-winner curation control that feeds round 2."),
    ("gallery.html", "Binned atlas", "The original faithfulness x novelty grid, up to 12 thumbnails per bin."),
    ("map.html", "Solution map", "Interactive UMAP scatter of the full embedding space (real images + every generation), with a generation slider/play control and a nearest-real-image mode-collapse chart."),
    ("coverage.html", "Coverage / void map", "Fine-grained faithfulness x novelty heatmap by image count, with frontier cells (empty but adjacent to dense ones) called out."),
    ("archive.html", "Elite archive", "One image per occupied cell: the actual MAP-Elites archive, human picks preferred over the automated novelty ranking."),
    ("redundancy.html", "Redundancy clusters", "Near-duplicate clustering by DINOv2 similarity at an adjustable threshold, to see the population's true effective diversity."),
    ("novelty_decay.html", "Novelty decay watchlist", "Per-prompt-family novelty over generations, to see which prompts are exhausted vs. still yielding new territory."),
    ("lineage.html", "Lineage tree", "Exploit chains showing whether mutating near a parent actually improves on it. Needs parent-tracking data that only starts accumulating after 2026-07-09."),
    ("seeds.html", "Candidate seeds", "View the subject/texture pool 'explore' jobs draw from, and ask GPT-5.5 for more on demand instead of waiting for a run to plateau and escalate on its own."),
]


def main(argv=None):
    write_infotip_asset(SWEEP_DIR)

    process_tip = info_btn(
        "This is a MAP-Elites search: instead of hill-climbing toward one 'best' image, it keeps a "
        "grid of bins (a faithfulness x novelty archive, see the elite archive) and tries to fill "
        "every bin with a good example, so the whole space gets mapped, not just its peak. Each "
        "generation makes a batch of new images two ways. Explore jobs draw a fresh random "
        "subject/texture combination, unrelated to anything made before: this is how the search "
        "finds genuinely new territory. Exploit jobs take an existing strong image (the current "
        "elite in a bin, or a human pick) and nudge its strength/cfg/seed slightly, hoping a small "
        "step nearby does even better: this is how the search refines what's already working. "
        "The mix between the two (e.g. round 2's 85% explore/15% exploit) is a deliberate dial: more "
        "explore finds new regions faster but refines them less, more exploit polishes known-good "
        "regions but can plateau if there's nothing better nearby to find. A 'generation' is one "
        "batch of jobs; 'plateau' means several generations in a row without the best novelty score "
        "improving, which is when the search either stops, escalates (e.g. asking an LLM for fresh "
        "subject ideas), or hits its budget cap and ends. Human picks made in the lightbox feed "
        "directly into the next generation's exploit pool, ahead of the algorithm's own ranking."
    )

    items_html = "".join(f"""
<a class="tool" href="{path}">
  <div class="name">{name}</div>
  <div class="desc">{desc}</div>
</a>""" for path, name, desc in TOOLS)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS exploration tools</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee; --text-dim:#9a9aa4; --accent:#7c9eff; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:32px; }}
{MOBILE_BASE_CSS}
{INFOTIP_CSS}
h1 {{ font-size:20px; margin:0 0 6px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
h1 .howtip {{ font-size:12.5px; font-weight:400; color:var(--text-dim); display:inline-flex; align-items:center; gap:6px; }}
p.sub {{ color:var(--text-dim); max-width:700px; font-size:13.5px; line-height:1.6; }}
#tools {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap:14px; margin-top:24px; max-width:1100px; }}
.tool {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; text-decoration:none;
  color:var(--text); transition: border-color .15s, transform .15s; display:block; }}
.tool:hover {{ border-color:var(--accent); transform:translateY(-2px); }}
.tool .name {{ font-size:14.5px; font-weight:600; margin-bottom:6px; }}
.tool .desc {{ font-size:12.5px; color:var(--text-dim); line-height:1.55; }}
</style></head><body>

<h1>CLAWMARKS exploration tools <span class="howtip">How does this search work?{process_tip}</span></h1>
<p class="sub">Tools for browsing individual generated images and clusters of them, and for
understanding the shape of the solution space the search is mapping out: where it's dense,
where it's empty but reachable, and how it's moved generation over generation.</p>

<div id="tools">{items_html}</div>

<script src="infotip.js"></script>
</body></html>"""

    with open(f"{SWEEP_DIR}/explore.html", "w") as f:
        f.write(html)

    print(f"wrote {SWEEP_DIR}/explore.html ({len(TOOLS)} tools linked)", flush=True)


if __name__ == "__main__":
    main()
