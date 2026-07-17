"""
Landing page linking every exploration/curation tool built for the uncanny-frontier search, so
there's one page to bookmark instead of remembering file names. Static, no data dependencies of
its own.

Run: python3 -m clawmarks.build.explore_hub
"""

from clawmarks.shared_ui import (
    CONTROL_CSS,
    INFOTIP_CSS,
    MOBILE_BASE_CSS,
    NAV_GROUPS,
    SULFUR_CSS,
    SULFUR_FONT_CSS,
    TOPNAV_CSS,
    info_btn,
    nav_bar_html,
)

# Order mirrors shared_ui.NAV_OPTIONS (minus explore.html, which is this hub) so the home page
# and the jump-to dropdown list the same tools in the same order.
TOOLS = [
    ("cockpit.html", "Generation cockpit", "Pick a mission, target a real coverage gap, draft a prompt against live evidence, queue a trial, and run it as a real generation batch scored into the archive."),
    ("runs.html", "Search runs", "Launch, monitor, and stop an overnight search round from the browser: backs up the round's out_dir and verifies it before launching, checks the RunPod balance floor, and shows a live novelty/plateau/spend report."),
    ("seeds.html", "Candidate seeds", "View the subject/texture pool 'explore' jobs draw from, and ask GPT-5.5 for more on demand instead of waiting for a run to plateau and escalate on its own."),
    ("compare.html", "Compare images (head-to-head)", "Pick the better of two images, over and over. Trains a preference model that learns your taste, ranks the whole pool from it, and steers which pairs to show next."),
    ("scan.html", "Scan gallery", "Every image, sortable/filterable/searchable, with a lightbox, similarity browsing, and the pick-as-winner curation control that feeds round 2."),
    ("archive.html", "Elite archive", "One image per occupied cell: the actual MAP-Elites archive, human picks preferred over the automated novelty ranking."),
    ("map.html", "Solution map", "Interactive UMAP scatter of the full embedding space (real images + every generation), with a generation slider/play control and a nearest-real-image mode-collapse chart."),
    ("coverage.html", "Coverage / void map", "Fine-grained faithfulness x novelty heatmap by image count, with frontier cells (empty but adjacent to dense ones) called out."),
    ("redundancy.html", "Redundancy clusters", "Near-duplicate clustering by DINOv2 similarity at an adjustable threshold, to see the population's true effective diversity."),
    ("novelty_decay.html", "Novelty decay watchlist", "Per-prompt-family novelty over generations, to see which prompts are exhausted vs. still yielding new territory."),
    ("lineage.html", "Lineage tree", "Exploit chains showing whether mutating near a parent actually improves on it. Needs parent-tracking data that only starts accumulating after 2026-07-09."),
    ("preference_status.html", "Preference status", "Training status for the preference model: comparison count, cross-validated accuracy, a permutation-test significance check, and controls to retrain or enable predicted preference."),
    ("preference_rank.html", "Predicted preference", "The trained preference model's ranking of every image, most-preferred first: what it predicts you'd pick, including images you never directly compared."),
]


def render_html(active_expedition=None, active_leg=None, running=None):
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

    # The Explore group in NAV_GROUPS is a quick-access subset of the workflow's five stage
    # destinations; those destinations are also listed in the detailed groups below, so the
    # hub doesn't double-render them and the user isn't pointed at "/" (this very page).
    detailed_groups = [g for g in NAV_GROUPS if g[0] != "Explore"]
    descriptions = {path: (name, desc) for path, name, desc in TOOLS}
    items_html = "".join(f"""
<section class="tool-group"><h2>{group}</h2><div class="tools">
{"".join(f'''<a class="tool raised-readout" href="{path}">
  <div class="name">{descriptions[path][0]}</div>
  <div class="desc">{descriptions[path][1]}</div>
</a>''' for path, _label in group_tools)}
</div></section>""" for group, group_tools in detailed_groups)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS exploration tools</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
{INFOTIP_CSS}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size:22px; margin:0 0 6px; display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  letter-spacing:0.02em; text-transform:uppercase; }}
h1 .howtip {{ font-size:12.5px; font-weight:400; color:var(--text-soft);
  display:inline-flex; align-items:center; gap:6px; text-transform:none;
  letter-spacing:0; font-family:var(--font-body); }}
p.sub {{ color:var(--text-soft); max-width:780px; font-size:13.5px; line-height:1.6;
  margin:0 0 18px; padding-bottom:14px; border-bottom:1px solid var(--rule); }}
.tool-group {{ margin-top:22px; max-width:1100px; }}
.tool-group h2 {{ font:600 13px/1.2 var(--font-display); color:var(--text-soft);
  margin:0 0 10px; text-transform:uppercase; letter-spacing:0.08em; }}
.tools {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap:14px; }}
.tool {{ background:var(--paper); border:1px solid var(--ink); color:var(--ink);
  padding:16px; text-decoration:none; display:block;
  transition: box-shadow .12s ease, transform .12s ease; }}
.tool .name {{ font:600 14.5px/1.3 var(--font-body); margin-bottom:6px; }}
.tool .desc {{ font-size:12.5px; color:var(--text-soft); line-height:1.55; }}
</style></head><body>

{nav_bar_html('explore.html', active_expedition=active_expedition, active_leg=active_leg, running=running)}

<main>
<h1>CLAWMARKS exploration tools <span class="howtip">How does this search work?{process_tip}</span></h1>
<p class="sub">Tools for browsing individual generated images and clusters of them, and for
understanding the shape of the solution space the search is mapping out: where it's dense,
where it's empty but reachable, and how it's moved generation over generation.</p>

<div id="tools">{items_html}</div>
</main>

<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
<script src="/shared-ui.js"></script>
</body></html>"""

    return html
