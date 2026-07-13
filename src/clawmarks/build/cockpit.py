"""
Generates cockpit.html: the generation cockpit, a single-page workbench for picking a mission,
choosing a target coverage cell, drafting a prompt, and queuing a real generation trial. Visual
direction is Fable's paper-craft critique (2026-07-13): light paper/sheet tokens, straight-cut
panels, a vertical-striation motif for active/selected state, no dark theme, no torn-paper/tape.

Unlike most other pages, this page bakes in no data at render time (missions are the only static
content). Every dynamic piece is a live fetch against curation_server.py:
  GET  /api/cockpit/target_cells  -> real frontier cells from build/coverage_map.top_frontier_cells
  GET  /api/cockpit/evidence      -> nearest-prompt neighbors + coverage context for a draft prompt
  GET  /api/cockpit/queue         -> all trials (draft/running/completed/failed)
  POST /api/cockpit/queue         -> append a draft trial
  POST /api/cockpit/queue/<id>/run -> submit a queued trial for real generation
  POST /api/cockpit/autopilot     -> opencode-backed next-trial suggestions

Served live at /cockpit.html by curation_server.py.
"""
import json

from clawmarks.shared_ui import nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, INFOTIP_CSS, info_btn


MISSIONS = {
    "gap": {
        "name": "Fill a coverage gap",
        "title": "Reach a sparse faith x novelty frontier",
        "hyp": "Test whether this prompt direction can reach a sparse, reachable frontier cell.",
        "queue": "Fill gap: coverage frontier",
        "uses_target_picker": True,
    },
    "candidate": {
        "name": "Develop a candidate",
        "title": "Give one candidate subject a first deliberate trial",
        "hyp": "Test whether this unused candidate belongs in the CLAWMARKS style grammar.",
        "queue": "Candidate trial",
        "uses_target_picker": False,
    },
    "lineage": {
        "name": "Continue a lineage",
        "title": "Change one parent intentionally",
        "hyp": "Test whether one deliberate change preserves the parent's strong structure.",
        "queue": "Lineage trial",
        "uses_target_picker": False,
    },
    "freeform": {
        "name": "Freeform",
        "title": "Start a standalone trial",
        "hyp": "",
        "queue": "Freeform trial",
        "uses_target_picker": False,
    },
}


def render_html():
    missions_json_keys = "".join(
        f'<button class="mission striate{" active" if key == "gap" else ""}" data-mission="{key}">'
        f'<span>Mission</span><b>{m["name"]}</b><span>{m["title"]}</span></button>'
        for key, m in MISSIONS.items()
    )
    faith_novelty_tip = info_btn(
        "Faith is how close the model thinks an image is to real reference photos. Novelty is "
        "how different it is from images already made."
    )
    prompt_similarity_tip = info_btn(
        "This compares the draft prompt's wording with prompts already in the manifest by word "
        "overlap. It does not say whether the resulting image will be good."
    )
    coverage_grid_tip = info_btn(
        "This is a small crop of the full faith x novelty map. The outlined cell is this trial's "
        "current target."
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLAWMARKS generation cockpit</title>
<link rel="icon" href="data:,">
<style>
:root{{
  color-scheme:light;
  --paper:#F5F2E9;
  --sheet:#EFE8C2;
  --sheet-deep:color-mix(in srgb, var(--sheet) 78%, #cbbf87);
  --ink:#17150F;
  --ballpoint:#32407F;
  --teal:#4F8A75;
  --red:#AF1F32;
  --line:color-mix(in srgb, var(--ink) 13%, transparent);
  --line-strong:color-mix(in srgb, var(--ink) 24%, transparent);
  --muted:color-mix(in srgb, var(--ink) 55%, transparent);
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,"Segoe UI",Helvetica,Arial,sans-serif;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 var(--sans)}}

.striate{{position:relative;isolation:isolate}}
.striate:before{{content:"";position:absolute;inset:0;z-index:-1;
  background-image:repeating-linear-gradient(90deg,
    color-mix(in srgb, var(--paper) 82%, transparent) 0 2px,
    transparent 2px 5px);
  opacity:.5;pointer-events:none}}

.topnav.cockpit-topnav {{ background:color-mix(in srgb, var(--paper) 92%, transparent) !important;
  border-bottom:1px solid var(--line) !important; }}
.topnav.cockpit-topnav a.navlink {{ color:var(--ballpoint) !important; font-family:var(--mono); }}
.topnav.cockpit-topnav select {{ background:var(--sheet) !important; color:var(--ink) !important;
  border:1px solid var(--line) !important; font-family:var(--mono); }}

main{{max-width:1500px;margin:auto;padding:26px 24px 150px}}
.eyebrow{{color:var(--ballpoint);font:700 10.5px var(--sans);letter-spacing:.14em;text-transform:uppercase}}
h1{{margin:3px 0 0;font:800 26px/1.15 var(--sans);letter-spacing:-.01em}}
.sub{{max-width:680px;margin:6px 0 0;color:var(--muted);line-height:1.55}}

.mission-bar{{display:flex;align-items:stretch;gap:10px;margin-top:22px;padding-bottom:2px}}
.mission{{flex:1;min-height:80px;padding:12px 13px;text-align:left;color:var(--ink);
  background:var(--sheet);border:1px solid var(--line);border-top:3px solid var(--line-strong);
  cursor:pointer;transition:background .15s,border-color .15s}}
.mission:hover{{background:var(--sheet-deep)}}
.mission span{{display:block;color:var(--muted);font-size:11px;line-height:1.3}}
.mission b{{display:block;margin:3px 0 4px;font:800 13.5px var(--sans);letter-spacing:-.01em}}
.mission.active{{background:var(--ballpoint);border-top-color:var(--ballpoint);color:var(--paper)}}
.mission.active.striate:before{{background-image:repeating-linear-gradient(90deg,
    color-mix(in srgb, var(--paper) 60%, transparent) 0 2px, transparent 2px 6px);opacity:.35}}
.mission.active span{{color:color-mix(in srgb, var(--paper) 78%, transparent)}}
.mission.active span:first-child{{color:var(--paper);font-weight:700}}

.workbench{{display:grid;grid-template-columns:minmax(500px,1.25fr) minmax(330px,.75fr);
  gap:3px;margin-top:24px;align-items:start}}
.panel{{background:var(--sheet);border:1px solid var(--line)}}
.recipe{{padding:0 18px 18px}}
.evidence{{background:color-mix(in srgb, var(--sheet) 88%, var(--paper))}}
.panel-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;
  padding:16px 0 12px;border-bottom:2px solid var(--line-strong)}}
.evidence .panel-head,.evidence-head{{padding:16px 17px 12px}}
.section-tag{{margin:0 0 6px;color:var(--ballpoint);font:800 11px var(--sans);
  letter-spacing:.12em;text-transform:uppercase}}
.section-tag.teal{{color:var(--teal)}}
.panel-head h2,.evidence-head h2{{margin:0;font-size:18px;line-height:1.25;font-weight:800}}
.small{{color:var(--muted);font-size:11.5px;line-height:1.45}}

.brief{{display:grid;grid-template-columns:1fr 1fr;grid-auto-flow:row;gap:5px 15px;margin:16px 0 13px}}
.field label,.brief label,.prompt-label{{display:block;margin-bottom:5px;color:var(--muted);
  font:600 11px var(--mono);letter-spacing:.02em}}
.field input,.brief input,textarea,select{{width:100%;padding:8px 9px;background:var(--paper);color:var(--ink);
  border:1px solid var(--line);font:13px var(--mono)}}
.prompt-row{{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:5px}}
.prompt-row .prompt-label{{margin-bottom:0}}
.seed-picker{{max-width:220px;padding:5px 6px;background:var(--paper);color:var(--muted);
  border:1px solid var(--line);font:11px var(--mono)}}
textarea{{min-height:100px;resize:vertical;line-height:1.55}}
input:focus,textarea:focus,select:focus{{outline:2px solid var(--ballpoint);outline-offset:1px;
  border-color:var(--ballpoint)}}

.target-picker{{margin:16px 0 13px}}
.target-picker h3{{margin:0 0 8px;color:var(--muted);font:600 11px var(--mono)}}
.target-cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
.target-card{{position:relative;display:grid;grid-template-columns:34px 1fr 16px;gap:8px;
  min-height:60px;padding:9px 9px 9px 13px;background:var(--paper);
  border:1px solid var(--line);color:var(--ink);text-align:left;cursor:pointer}}
.target-card:before{{position:absolute;top:0;bottom:0;left:0;width:4px;
  background:var(--line-strong);content:""}}
.target-card:hover{{border-color:var(--ballpoint)}}
.target-card.selected{{background:color-mix(in srgb, var(--ballpoint) 9%, var(--paper));
  border-color:var(--ballpoint)}}
.target-card.selected:before{{background:var(--ballpoint)}}
.target-check{{align-self:center;color:transparent;font:800 16px var(--mono)}}
.target-card.selected .target-check{{color:var(--ballpoint)}}
.target-range{{font:700 10.5px var(--mono);line-height:1.35}}
.target-adjacent{{margin-top:3px;color:var(--muted);font-size:10.5px}}
.target-empty{{color:var(--muted);font-size:12px;padding:10px 2px}}

.nearest b{{color:var(--ink);font-weight:700}}
.thumb{{position:relative;display:inline-block;flex:none;width:34px;height:34px;
  background-size:cover;background-position:center;background-color:var(--sheet-deep)}}
.thumb:before{{content:"";position:absolute;inset:3px -3px -3px 3px;z-index:-1;
  background:var(--sheet-deep)}}
.nearest .thumb{{width:40px;height:40px}}

.controls{{display:grid;grid-template-columns:1.1fr .7fr .7fr;gap:14px;margin-top:13px}}
.segmented,.number{{display:flex;height:34px;background:var(--paper);border:1px solid var(--line)}}
.segmented button{{flex:1;background:transparent;border:0;border-right:1px solid var(--line);
  color:var(--muted);font:600 11px var(--mono);cursor:pointer}}
.segmented button.active{{color:var(--paper);background:var(--ballpoint)}}
.number{{align-items:center}}
.number button{{width:28px;height:100%;border:0;background:transparent;color:var(--muted);
  font-size:16px;cursor:pointer}}
.number span{{flex:1;text-align:center;font:700 12px var(--mono)}}
.advanced-toggle{{display:flex;justify-content:space-between;width:100%;margin-top:14px;
  padding:9px 0;background:transparent;border:0;border-top:1px solid var(--line);
  border-bottom:1px solid var(--line);color:var(--muted);cursor:pointer;text-align:left;
  font:600 12px var(--sans)}}
.advanced-toggle b{{color:var(--ink);font-family:var(--mono);font-weight:700}}
.advanced-body{{display:none;grid-template-columns:repeat(3,1fr);gap:12px;padding:13px 0 2px}}
.advanced-body.open{{display:grid}}
.action-row{{display:flex;justify-content:space-between;align-items:center;gap:14px;
  margin-top:18px;padding-top:14px;border-top:2px solid var(--line-strong)}}
.estimate{{color:var(--muted);font-size:11.5px;line-height:1.5}}
.estimate b{{color:var(--ink);font-family:var(--mono)}}
.estimate-note{{margin:5px 0 0;color:var(--muted);font-size:11px}}
.generate{{position:relative;isolation:isolate;padding:11px 18px;background:var(--ballpoint);
  color:var(--paper);border:0;font:800 12.5px var(--sans);letter-spacing:.03em;cursor:pointer}}
.generate:before{{content:"";position:absolute;inset:0;z-index:-1;
  background-image:repeating-linear-gradient(90deg,
    color-mix(in srgb, var(--paper) 55%, transparent) 0 2px, transparent 2px 7px);opacity:.3}}
.generate:hover{{background:color-mix(in srgb, var(--ballpoint) 88%, black)}}
.generate:disabled{{opacity:.5;cursor:default}}

.honest{{margin-bottom:6px;color:var(--teal);font:700 10px var(--sans);letter-spacing:.13em;
  text-transform:uppercase}}
.e-section{{padding:14px 17px;border-top:1px solid var(--line)}}
.e-section h3{{margin:0 0 10px;color:var(--muted);font:700 11px var(--sans);
  letter-spacing:.1em;text-transform:uppercase}}
.nearest{{display:flex;align-items:center;gap:10px;padding:9px 0;border-top:1px solid var(--line)}}
.nearest:first-of-type{{padding-top:0;border:0}}
.meta{{margin-top:2px;color:var(--muted);font:10.5px var(--mono)}}
.status-kept{{color:var(--teal)}}
.status-rejected{{color:var(--red)}}
.status-unrated{{color:var(--muted)}}
.coverage{{display:grid;grid-template-columns:76px 1fr;gap:12px;align-items:center}}
.mini-grid-wrap{{position:relative}}
.grid-axis{{position:absolute;color:var(--muted);font:8px var(--mono);white-space:nowrap}}
.grid-axis.top{{top:-12px;left:3px}}
.grid-axis.side{{top:28px;left:-13px;transform:rotate(-90deg)}}
.mini-grid{{display:grid;grid-template-columns:repeat(4,16px);gap:2px}}
.cell{{width:16px;height:16px;background:var(--paper);border:1px solid var(--line)}}
.coverage p{{margin:0;color:var(--muted);font-size:11.5px;line-height:1.5}}
.coverage b{{color:var(--ink)}}

.drawer{{position:fixed;z-index:20;right:0;bottom:0;left:0;background:var(--sheet);
  border-top:2px solid var(--line-strong);box-shadow:0 -8px 22px rgba(23,21,15,.12)}}
.drawer-inner{{max-width:1500px;margin:auto}}
.drawer-head{{display:flex;align-items:center;gap:6px;padding:9px 24px}}
.drawer-title{{margin-right:8px;color:var(--muted);font:600 11px var(--mono)}}
.tab{{padding:6px 10px;background:transparent;border:0;border-bottom:2px solid transparent;
  color:var(--muted);font:700 12px var(--sans);cursor:pointer}}
.tab.active{{color:var(--ink);border-bottom-color:var(--ballpoint)}}
.drawer-toggle{{margin-left:auto;background:transparent;border:0;color:var(--muted);
  font:600 12px var(--mono);cursor:pointer}}
.drawer-body{{display:none;padding:0 24px 14px}}
.drawer.open .drawer-body{{display:block}}
.tab-pane{{display:none}}
.tab-pane.active{{display:block}}
.trial-row{{display:flex;align-items:center;gap:12px;max-width:840px;padding:10px 0;
  border-top:1px solid var(--line);border-bottom:1px solid var(--line);font-size:11.5px}}
.trial-row b{{font:800 12px var(--sans)}}
.trial-row .status{{font-family:var(--mono);font-weight:700}}
.trial-row .status.draft{{color:var(--muted)}}
.trial-row .status.running{{color:var(--ballpoint)}}
.trial-row .status.completed{{color:var(--teal)}}
.trial-row .status.failed{{color:var(--red)}}
.trial-row button{{margin-left:auto;padding:6px 9px;background:var(--paper);
  color:var(--ink);border:1px solid var(--line-strong);font:600 11px var(--mono);cursor:pointer}}
.trial-row button:disabled{{opacity:.5;cursor:default}}
.empty-note{{color:var(--muted);font-size:12px;padding:10px 0}}
.result-card{{display:inline-flex;align-items:center;gap:9px;margin-right:12px;padding:8px 0;
  font:11px var(--mono);cursor:pointer}}
.result-card .thumb{{width:34px;height:34px}}
.autopilot-note{{max-width:820px;margin:10px 0 5px;color:var(--muted);font-size:11.5px}}
.autopilot-refresh{{padding:6px 10px;background:var(--paper);border:1px solid var(--line-strong);
  color:var(--ink);font:600 11px var(--mono);cursor:pointer}}
.suggestions{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;
  max-width:1050px;padding-top:10px}}
.suggestion{{padding:12px;background:var(--paper);border:1px solid var(--line);
  border-top:3px solid var(--teal)}}
.suggestion h3{{margin:0 0 5px;font:800 13px var(--sans)}}
.suggestion .mission-map{{color:var(--teal);font:700 10px var(--mono)}}
.suggestion textarea{{min-height:78px;margin-top:9px;font-size:10.5px}}
.rationale{{margin:8px 0 0;color:var(--muted);font-size:11px}}
.use-suggestion{{display:block;margin:10px 0 0;padding:7px 10px;background:var(--ballpoint);
  color:var(--paper);border:0;font:700 11px var(--mono);cursor:pointer}}

@media(max-width:900px){{
  .mission-bar{{display:grid;grid-template-columns:repeat(2,1fr)}}
  .workbench{{grid-template-columns:1fr;gap:14px}}
  .evidence{{display:grid;grid-template-columns:1fr 1fr}}
  .evidence-head{{grid-column:1/-1}}
  .suggestions{{grid-template-columns:1fr 1fr}}
}}
@media(max-width:640px){{
  main{{padding:18px 11px 136px}}
  .mission-bar{{gap:8px;margin-top:12px}}
  .mission{{min-height:74px;padding:11px}}
  .mission-bar.mobile-collapsed{{display:none}}
  .mobile-mission-summary{{display:flex;align-items:center;gap:9px;margin-top:14px;
    padding:9px 11px;background:var(--ballpoint);color:var(--paper);
    border-left:4px solid var(--ink);font:600 11px var(--mono)}}
  .mobile-mission-summary b{{flex:1;color:var(--paper);font:800 13px var(--sans)}}
  .mobile-mission-summary button{{padding:5px 8px;background:transparent;
    border:1px solid var(--paper);color:var(--paper);font:600 11px var(--mono);cursor:pointer}}
  .workbench{{gap:20px;margin-top:14px}}
  .recipe{{display:flex;min-width:0;flex-direction:column;padding:0 14px 18px}}
  .recipe .panel-head{{order:1;padding-top:15px}}
  .recipe .target-picker{{order:2;min-width:0;margin-top:13px}}
  .recipe .prompt-label[for="prompt"]{{order:3;margin-top:17px;color:var(--ink);font-weight:700}}
  .recipe #prompt{{order:4;min-height:144px;padding:12px;background:var(--paper);
    border:2px solid var(--ballpoint);font-size:13px}}
  .recipe .action-row{{order:5;display:grid;grid-template-columns:1fr;gap:9px;
    margin-top:12px;padding:12px;background:color-mix(in srgb, var(--ballpoint) 10%, var(--sheet));
    border-top:0}}
  .recipe .generate{{min-height:52px;padding:13px 16px;font-size:14px}}
  .recipe .brief{{order:6;margin-top:20px}}
  .recipe .controls{{order:8}}
  .recipe .advanced-toggle{{order:9}}
  .recipe .advanced-body{{order:10}}
  .target-cards{{display:flex;gap:8px;overflow-x:auto;padding:1px 1px 6px}}
  .target-card{{flex:0 0 176px}}
  .brief{{grid-template-columns:1fr 1fr;gap:9px;padding-top:14px;border-top:1px solid var(--line)}}
  .evidence{{display:block;position:relative}}
  .evidence-head{{padding:13px 14px 11px}}
  .evidence-head .section-tag,.evidence-head .honest,.evidence-head h2,.evidence-head .small{{display:none}}
  .mobile-evidence-summary{{display:flex;align-items:center;gap:8px;color:var(--muted);
    font-size:11px;line-height:1.35}}
  .mobile-evidence-summary b{{color:var(--ink);font:800 11px var(--sans);letter-spacing:.06em}}
  .mobile-evidence-summary span{{flex:1}}
  .mobile-evidence-toggle{{padding:6px 8px;background:transparent;border:1px solid var(--line-strong);
    color:var(--ink);font:600 11px var(--mono);white-space:nowrap;cursor:pointer}}
  .evidence.mobile-collapsed .e-section{{display:none}}
  .evidence:not(.mobile-collapsed) .evidence-head{{border-bottom:1px solid var(--line)}}
  .evidence:not(.mobile-collapsed) .evidence-head .section-tag,
  .evidence:not(.mobile-collapsed) .evidence-head .honest,
  .evidence:not(.mobile-collapsed) .evidence-head h2,
  .evidence:not(.mobile-collapsed) .evidence-head .small{{display:block}}
  .evidence:not(.mobile-collapsed) .mobile-evidence-summary{{margin-top:11px}}
  .drawer-body{{padding:0 10px 12px}}
  .trial-row{{flex-wrap:wrap;align-items:flex-start}}
  .trial-row button{{margin-left:0}}
  .suggestions{{grid-template-columns:1fr;gap:8px}}
}}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
{INFOTIP_CSS}
</style>
</head>
<body>
{nav_bar_html('cockpit.html')}
<main>
<div class="eyebrow">Interactive trial workbench</div>
<h1>Generation cockpit</h1>
<p class="sub">Choose an intent, record a small test, then generate deliberately. Evidence describes nearby work already in the manifest. It does not predict an unmade image.</p>

<section class="mission-bar" aria-label="Choose a mission">{missions_json_keys}</section>

<section class="workbench">
<article class="panel recipe">
<div class="panel-head"><div><div class="section-tag">Recipe</div><div class="eyebrow" id="briefMission">Fill a coverage gap</div>
<h2 id="briefTitle">Reach a sparse faith x novelty frontier</h2></div><span class="small">Draft trial</span></div>
<div class="target-picker" id="targetPicker"><h3>Choose a target coverage cell
{faith_novelty_tip}</h3>
<div class="target-cards" id="targetCards"><div class="target-empty">Loading frontier cells&hellip;</div></div></div>
<div class="brief">
<label for="hypothesis">What are you testing? <span class="small">optional</span></label>
<label id="targetLabel" for="target">Target coverage cell</label>
<input id="hypothesis" value="">
<input id="target" value="" readonly>
</div>
<div class="prompt-row"><label class="prompt-label" for="prompt">Prompt</label>
<select class="seed-picker" id="seedPicker" aria-label="Insert from candidate seed pool">
<option value="">Insert from seed pool&hellip;</option></select></div>
<textarea id="prompt" placeholder="trentbuckle style, ..."></textarea>
<div class="controls"><div><label class="prompt-label">Seed strategy</label>
<div class="segmented" id="seedStrategy"><button class="active" data-seed="random">random</button>
<button data-seed="fixed">fixed</button></div></div>
<div><label class="prompt-label">Batch size</label><div class="number"><button id="minusN" type="button">-</button>
<span id="batchN">4</span><button id="plusN" type="button">+</button></div></div>
<div><label class="prompt-label">LoRA strength</label><div class="number"><button id="minusS" type="button">-</button>
<span id="strengthN">1.00</span><button id="plusS" type="button">+</button></div></div></div>
<button class="advanced-toggle" id="advancedToggle" type="button"><span>Advanced <b id="advancedMeta">ddim / 28 / 7.5</b></span>
<span id="chevron">show</span></button>
<div class="advanced-body" id="advancedBody"><div class="field"><label>Sampler</label>
<select id="sampler"><option value="ddim">ddim</option><option value="dpmpp_2m">dpmpp_2m</option><option value="euler">euler</option></select></div>
<div class="field"><label>Steps</label><input id="steps" value="28" inputmode="numeric"></div>
<div class="field"><label>CFG</label><input id="cfg" value="7.5" inputmode="decimal"></div>
<div class="field"><label>Negative prompt</label><input id="negative" value="low quality, blurry, watermark"></div></div>
<div class="action-row"><div class="estimate"><div><b id="estimateN">4 images</b> queued as a draft</div>
<p class="estimate-note" id="queueStatus">New images will enter the trial record as a draft.</p></div>
<button class="generate striate" id="sendDraft" type="button">Send draft to queue</button></div>
</article>

<aside class="panel evidence"><div class="evidence-head"><div class="section-tag teal">Evidence</div>
<div class="honest">Existing evidence only</div><h2>What is already nearby</h2>
<div class="small">These are prompt-text neighbors and a current coverage cell, not a forecast for this draft.</div></div>
<div class="e-section"><h3>Nearest past prompts
{prompt_similarity_tip}</h3>
<div id="nearestList"><div class="empty-note">Start typing a prompt to see nearby work.</div></div></div>
<div class="e-section"><h3>Target coverage context
{coverage_grid_tip}</h3>
<div class="coverage" id="coverageBox"><div class="empty-note">Select a target cell (Fill a coverage gap mission) to see context.</div></div></div>
</aside>
</section>
</main>

<section class="drawer" id="drawer"><div class="drawer-inner">
<div class="drawer-head"><span class="drawer-title">Trial record</span>
<button class="tab active" data-tab="queue" type="button">Queue <span id="queueCount">0</span></button>
<button class="tab" data-tab="results" type="button">Results</button>
<button class="tab" data-tab="autopilot" type="button">Autopilot</button>
<button class="drawer-toggle" id="drawerToggle" type="button">show queue</button></div>
<div class="drawer-body">
<div class="tab-pane active" id="queuePane"><div class="empty-note">No trials queued yet.</div></div>
<div class="tab-pane" id="resultsPane"><div class="empty-note">No completed trials yet.</div></div>
<div class="tab-pane" id="autopilotPane"><p class="autopilot-note">Suggestions are grounded in real trial history (frontier cells, nearby prompts). They are not a prediction of how a new image will score.
<button class="autopilot-refresh" id="autopilotRefresh" type="button">refresh suggestions</button></p>
<div class="suggestions" id="suggestions"><div class="empty-note">Open this tab to fetch suggestions.</div></div></div>
</div></div></section>

<script>
const MISSIONS = {json.dumps(MISSIONS)};
let current='gap',selectedCell=null,frontierCells=[],n=4,seed='random',strength=1.0,queue=[];
let evidenceReqId=0,pollTimer=null;
const $=id=>document.getElementById(id);

function escapeHtml(s){{return String(s).replace(/[&<>"']/g, c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}

function updateEstimate(){{
  const sampler=$('sampler').value,steps=$('steps').value||'28',cfg=$('cfg').value||'7.5';
  $('estimateN').textContent=`${{n}} image${{n===1?'':'s'}}`;
  $('advancedMeta').textContent=`${{sampler}} / ${{steps}} / ${{cfg}}`;
}}

function renderTargetCards(){{
  const wrap=$('targetCards');
  if(!frontierCells.length){{wrap.innerHTML='<div class="target-empty">No frontier cells found (need scored images first).</div>';return}}
  wrap.innerHTML=frontierCells.map((c,i)=>`<button class="target-card ${{i===selectedCell?'selected':''}}" data-cell="${{i}}" aria-pressed="${{i===selectedCell}}"><span class="thumb" style="background-image:url('${{escapeHtml(c.thumb||'')}}')"></span><span><span class="target-range">${{escapeHtml(c.range)}}</span><span class="target-adjacent">adjacent to ${{c.adjacent}} images</span></span><span class="target-check" aria-hidden="true">&#10003;</span></button>`).join('');
  wrap.querySelectorAll('[data-cell]').forEach(button=>button.onclick=()=>{{selectedCell=Number(button.dataset.cell);applySelectedCell();renderTargetCards()}});
}}

function applySelectedCell(){{
  const c=frontierCells[selectedCell];
  if(!c){{$('target').value='';renderCoverageBox(null);return}}
  $('target').value=c.range;
  renderCoverageBox(c);
  fetchEvidence();
}}

function renderCoverageBox(c){{
  const box=$('coverageBox');
  if(!c){{box.innerHTML='<div class="empty-note">Select a target cell (Fill a coverage gap mission) to see context.</div>';return}}
  const near = (c.near_faith!=null && c.near_novelty!=null) ? `Its nearest occupied neighbor has faith ${{c.near_faith}} and novelty ${{c.near_novelty}}.` : '';
  box.innerHTML=`<div class="mini-grid-wrap"><span class="grid-axis top">novelty</span><span class="grid-axis side">faith</span>
    <div class="mini-grid">${{Array.from({{length:16}},()=>'<i class="cell"></i>').join('')}}</div></div>
    <p><b>Frontier cell: empty</b><br>Adjacent to ${{c.adjacent}} images. ${{near}}</p>`;
}}

function fetchTargetCells(){{
  fetch('/api/cockpit/target_cells').then(r=>r.json()).then(d=>{{
    frontierCells=d.cells||[];
    selectedCell=frontierCells.length?0:null;
    renderTargetCards();
    if(current==='gap')applySelectedCell();
  }}).catch(()=>{{$('targetCards').innerHTML='<div class="target-empty">Could not load frontier cells.</div>'}});
}}

function fetchEvidence(){{
  const myReq=++evidenceReqId;
  const prompt=$('prompt').value.trim();
  const list=$('nearestList');
  if(!prompt){{list.innerHTML='<div class="empty-note">Start typing a prompt to see nearby work.</div>';return}}
  const params=new URLSearchParams({{prompt}});
  const c=frontierCells[selectedCell];
  if(current==='gap' && c) params.set('cell', `${{c.fb}},${{c.nb}}`);
  fetch('/api/cockpit/evidence?'+params.toString()).then(r=>r.json()).then(d=>{{
    if(myReq!==evidenceReqId)return;
    const items=d.nearest||[];
    list.innerHTML = items.length ? items.map(it=>`<div class="nearest"><span class="thumb" style="background-image:url('${{escapeHtml(it.thumb||'')}}')"></span><div><b>${{escapeHtml(it.prompt_name)}}</b>
      <div class="meta">word overlap ${{Math.round(it.similarity*100)}}% &middot; faith ${{it.faith}} &middot; novelty ${{it.novelty}} &middot; <span class="status-${{it.status}}">${{it.status}}</span></div></div></div>`).join('')
      : '<div class="empty-note">No manifest prompts share meaningful wording with this draft yet.</div>';
  }}).catch(()=>{{if(myReq===evidenceReqId)list.innerHTML='<div class="empty-note">Could not load evidence.</div>'}});
}}

let evidenceDebounce=null;
$('prompt').addEventListener('input', ()=>{{clearTimeout(evidenceDebounce);evidenceDebounce=setTimeout(fetchEvidence,400)}});

function fetchSeeds(){{
  fetch('/api/seeds').then(r=>r.json()).then(d=>{{
    const seeds=Object.keys(d).sort();
    $('seedPicker').innerHTML='<option value="">Insert from seed pool&hellip;</option>'
      +seeds.map(s=>`<option value="${{escapeHtml(s)}}">${{escapeHtml(s)}}</option>`).join('');
  }}).catch(()=>{{}});
}}
$('seedPicker').addEventListener('change', ()=>{{
  const val=$('seedPicker').value;
  if(!val)return;
  const ta=$('prompt');
  if(!ta.value.trim()){{
    ta.value=`trentbuckle style, ${{val}}`;
  }} else {{
    const start=ta.selectionStart!=null?ta.selectionStart:ta.value.length;
    const end=ta.selectionEnd!=null?ta.selectionEnd:ta.value.length;
    ta.value=ta.value.slice(0,start)+val+ta.value.slice(end);
  }}
  $('seedPicker').value='';
  ta.dispatchEvent(new Event('input'));
  ta.focus();
}});

function setMission(key){{
  if(!MISSIONS[key])key='freeform';
  current=key;
  const m=MISSIONS[key];
  document.querySelectorAll('.mission').forEach(x=>x.classList.toggle('active',x.dataset.mission===key));
  $('briefMission').textContent=m.name;
  $('briefTitle').textContent=m.title;
  $('hypothesis').value=m.hyp;
  const gap=m.uses_target_picker;
  $('targetPicker').hidden=!gap;
  $('targetLabel').textContent=gap?'Target coverage cell':'Target';
  if(gap){{applySelectedCell()}}else{{$('target').value='freeform, no target cell';renderCoverageBox(null)}}
  updateEstimate();
}}
document.querySelectorAll('.mission').forEach(button=>button.onclick=()=>setMission(button.dataset.mission));

function holdRepeat(id, fn){{
  const el=$(id);
  let delayTimer=null, repeatTimer=null;
  const stop=()=>{{clearTimeout(delayTimer);clearInterval(repeatTimer);delayTimer=null;repeatTimer=null}};
  const start=event=>{{event.preventDefault();stop();fn();delayTimer=setTimeout(()=>{{repeatTimer=setInterval(fn,80)}},400)}};
  el.addEventListener('mousedown',start);
  el.addEventListener('touchstart',start,{{passive:false}});
  ['mouseup','mouseleave','touchend','touchcancel'].forEach(evt=>el.addEventListener(evt,stop));
}}
holdRepeat('minusN',()=>{{n=Math.max(1,n-1);$('batchN').textContent=n;updateEstimate()}});
holdRepeat('plusN',()=>{{n=Math.min(6,n+1);$('batchN').textContent=n;updateEstimate()}});
holdRepeat('minusS',()=>{{strength=Math.max(0.3,Math.round((strength-0.05)*100)/100);$('strengthN').textContent=strength.toFixed(2)}});
holdRepeat('plusS',()=>{{strength=Math.min(2.2,Math.round((strength+0.05)*100)/100);$('strengthN').textContent=strength.toFixed(2)}});
document.querySelectorAll('[data-seed]').forEach(button=>button.onclick=()=>{{seed=button.dataset.seed;document.querySelectorAll('[data-seed]').forEach(x=>x.classList.toggle('active',x===button))}});
['sampler','steps','cfg'].forEach(id=>$(id).addEventListener('input',updateEstimate));
$('advancedToggle').onclick=()=>{{$('advancedBody').classList.toggle('open');$('chevron').textContent=$('advancedBody').classList.contains('open')?'hide':'show'}};

function selectTab(tab){{
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.tab===tab));
  document.querySelectorAll('.tab-pane').forEach(x=>x.classList.toggle('active',x.id===`${{tab}}Pane`));
  $('drawer').classList.add('open');$('drawerToggle').textContent='hide record';
  if(tab==='autopilot' && $('suggestions').dataset.loaded!=='1')loadAutopilot();
}}
document.querySelectorAll('.tab').forEach(button=>button.onclick=()=>selectTab(button.dataset.tab));
$('drawerToggle').onclick=()=>{{$('drawer').classList.toggle('open');$('drawerToggle').textContent=$('drawer').classList.contains('open')?'hide record':'show record'}};

function sendDraft(){{
  const c=frontierCells[selectedCell];
  const body={{
    mission: current, prompt: $('prompt').value.trim(), hypothesis: $('hypothesis').value.trim(),
    target: $('target').value, target_cell: (current==='gap' && c) ? [c.fb, c.nb] : null,
    seed_strategy: seed, n, strength, sampler: $('sampler').value,
    steps: parseInt($('steps').value,10)||28, cfg: parseFloat($('cfg').value)||7.5,
    negative: $('negative').value,
  }};
  if(!body.prompt){{$('queueStatus').textContent='Write a prompt before queuing a trial.';return}}
  $('sendDraft').disabled=true;
  fetch('/api/cockpit/queue', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}})
    .then(r=>r.json().then(data=>({{ok:r.ok,data}})))
    .then(({{ok,data}})=>{{
      $('sendDraft').disabled=false;
      if(!ok||data.error){{$('queueStatus').textContent=data.error||'Could not queue this trial.';return}}
      $('queueStatus').textContent='Draft queued. Open the Queue tab to run it.';
      loadQueue();selectTab('queue');
    }}).catch(()=>{{$('sendDraft').disabled=false;$('queueStatus').textContent='Could not reach the server.'}});
}}
$('sendDraft').onclick=sendDraft;

function renderQueue(){{
  $('queueCount').textContent=queue.filter(t=>t.status!=='completed'&&t.status!=='failed').length;
  const draftsAndRunning=queue.filter(t=>t.status==='draft'||t.status==='running'||t.status==='failed');
  $('queuePane').innerHTML = draftsAndRunning.length ? draftsAndRunning.map(t=>`
    <div class="trial-row"><div><b>${{escapeHtml(t.queue_title||t.mission)}}</b><br>
    <span class="small">${{t.n}} images &middot; ${{escapeHtml(t.seed_strategy)}} seeds &middot; ${{t.strength.toFixed(2)}} strength &middot; ${{escapeHtml(t.sampler)}} / ${{t.steps}} / ${{t.cfg}}</span></div>
    <span class="status ${{t.status}}">${{t.status}}${{t.error?': '+escapeHtml(t.error):''}}</span>
    ${{t.status==='draft'?`<button data-run="${{t.id}}" type="button">Run queued trial</button>`:''}}</div>`).join('')
    : '<div class="empty-note">No trials queued yet.</div>';
  $('queuePane').querySelectorAll('[data-run]').forEach(button=>button.onclick=()=>runTrial(button.dataset.run));

  const completed=queue.filter(t=>t.status==='completed');
  $('resultsPane').innerHTML = completed.length ? completed.map(t=>
    (t.result_tags||[]).map(tag=>`<span class="result-card" data-tag="${{escapeHtml(tag)}}"><span class="thumb" style="background-image:url('thumbs/${{escapeHtml(tag)}}.jpg')"></span><span>${{escapeHtml(tag)}}</span></span>`).join('')
  ).join('') : '<div class="empty-note">No completed trials yet.</div>';
  $('resultsPane').querySelectorAll('[data-tag]').forEach(el=>el.onclick=()=>{{if(window.Lightbox)Lightbox.open(el.dataset.tag)}});

  const anyRunning=queue.some(t=>t.status==='running');
  if(anyRunning && !pollTimer){{pollTimer=setInterval(loadQueue,4000)}}
  if(!anyRunning && pollTimer){{clearInterval(pollTimer);pollTimer=null}}
}}

function loadQueue(){{
  fetch('/api/cockpit/queue').then(r=>r.json()).then(d=>{{queue=d.trials||[];renderQueue()}}).catch(()=>{{}});
}}

function runTrial(id){{
  const button=document.querySelector(`[data-run="${{id}}"]`);
  if(button)button.disabled=true;
  fetch(`/api/cockpit/queue/${{id}}/run`, {{method:'POST'}})
    .then(r=>r.json().then(data=>({{ok:r.ok,data}})))
    .then(({{ok,data}})=>{{
      if(!ok||data.error){{alert(data.error||'Could not start this trial.');if(button)button.disabled=false;return}}
      loadQueue();
    }}).catch(()=>{{alert('Could not reach the server.');if(button)button.disabled=false}});
}}

function loadAutopilot(){{
  const wrap=$('suggestions');
  wrap.innerHTML='<div class="empty-note">Asking for suggestions grounded in current coverage and prompt history&hellip; this can take a minute.</div>';
  fetch('/api/cockpit/autopilot', {{method:'POST'}}).then(r=>r.json()).then(d=>{{
    wrap.dataset.loaded='1';
    const items=d.suggestions||[];
    wrap.innerHTML = items.length ? items.map(s=>`
      <article class="suggestion" data-mission="${{escapeHtml(s.mission)}}" data-cell="${{escapeHtml(s.target_cell!=null?JSON.stringify(s.target_cell):'')}}">
      <h3>${{escapeHtml(s.title)}}</h3><div class="mission-map">${{escapeHtml(s.mission)}}</div>
      <textarea readonly>${{escapeHtml(s.prompt)}}</textarea>
      <p class="rationale">${{escapeHtml(s.rationale)}}</p>
      <button class="use-suggestion" type="button">Use this</button></article>`).join('')
      : `<div class="empty-note">${{escapeHtml(d.error||'No suggestions available yet.')}}</div>`;
    wrap.querySelectorAll('.use-suggestion').forEach(button=>button.onclick=()=>{{
      const card=button.closest('.suggestion');
      setMission(card.dataset.mission);
      const cellJson=card.dataset.cell;
      selectedCell=null;
      if(cellJson){{
        const parsed=JSON.parse(cellJson);
        if(Array.isArray(parsed)&&parsed.length===2){{
          const [fb,nb]=parsed;
          const idx=frontierCells.findIndex(c=>c.fb===fb&&c.nb===nb);
          if(idx!==-1)selectedCell=idx;
        }}
      }}
      renderTargetCards();
      $('prompt').value=card.querySelector('textarea').value;
      $('hypothesis').value=card.querySelector('h3').textContent;
      if(current==='gap')applySelectedCell();
      else fetchEvidence();
      selectTab('queue');
    }});
  }}).catch(()=>{{wrap.innerHTML='<div class="empty-note">Could not reach the server.</div>'}});
}}
$('autopilotRefresh').onclick=loadAutopilot;

(()=>{{
  const mobile=matchMedia('(max-width: 640px)'),bar=document.querySelector('.mission-bar'),summary=document.createElement('div');
  summary.className='mobile-mission-summary';summary.innerHTML='<span>Current mission</span><b></b><button type="button">change</button>';
  bar.after(summary);
  const missionName=summary.querySelector('b'),change=summary.querySelector('button');
  function syncMission(){{missionName.textContent=document.querySelector('.mission.active b')?.textContent||'Choose a mission';bar.classList.toggle('mobile-collapsed',mobile.matches);summary.hidden=!mobile.matches}}
  change.onclick=()=>bar.classList.remove('mobile-collapsed');
  const priorSetMission=setMission;
  setMission=function(key){{priorSetMission(key);if(mobile.matches)bar.classList.add('mobile-collapsed');syncMission()}};
  const evidence=document.querySelector('.evidence'),head=evidence.querySelector('.evidence-head'),summaryEvidence=document.createElement('div');
  summaryEvidence.className='mobile-evidence-summary';
  summaryEvidence.innerHTML='<b>Evidence</b><span>select a target cell or draft a prompt to see nearby work</span><button class="mobile-evidence-toggle" type="button" aria-expanded="false">see context</button>';
  head.append(summaryEvidence);
  const toggle=summaryEvidence.querySelector('button');
  function syncEvidence(){{const compact=mobile.matches&&evidence.classList.contains('mobile-collapsed');toggle.textContent=compact?'see context':'hide context';toggle.setAttribute('aria-expanded',String(!compact))}}
  toggle.onclick=()=>{{evidence.classList.toggle('mobile-collapsed');syncEvidence()}};
  mobile.addEventListener('change',()=>{{evidence.classList.toggle('mobile-collapsed',mobile.matches);syncMission();syncEvidence()}});
  evidence.classList.toggle('mobile-collapsed',mobile.matches);syncMission();syncEvidence();
}})();

document.querySelector('.topnav').classList.add('cockpit-topnav');
setMission('gap');
fetchTargetCells();
fetchSeeds();
loadQueue();
updateEstimate();
</script>
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
</body>
</html>"""
    return html
