"""Render the active research desk and its compact full-tool index."""

from __future__ import annotations

import html
from typing import Any, Iterable

from clawmarks.shared_ui import (
    CONTROL_CSS,
    INFOTIP_CSS,
    MOBILE_BASE_CSS,
    NAV_GROUPS,
    SULFUR_CSS,
    SULFUR_FONT_CSS,
    TOPNAV_CSS,
    nav_bar_html,
)
from clawmarks.workspace_context import WorkspaceContext, context_url, generated_image_url


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

_CONTRACT_FIELDS = ("intention", "evidence_scope", "changed_variable", "held_constant", "expected_move", "evidence_against")


def _scope_from_focus(focus: dict[str, Any] | None) -> WorkspaceContext:
    scope = focus.get("scope", {}) if isinstance(focus, dict) else {}
    return WorkspaceContext(scope.get("expedition"), scope.get("leg"), focus)


def _focus_url(path: str, focus: dict[str, Any] | None) -> str:
    return context_url(path, _scope_from_focus(focus)) if focus else path


def _contract_complete(focus: dict[str, Any] | None) -> bool:
    contract = focus.get("test_contract") if isinstance(focus, dict) else None
    return isinstance(contract, dict) and all(bool(contract.get(field)) for field in _CONTRACT_FIELDS)


def _trial_evaluated(trial: dict[str, Any]) -> bool:
    return bool(trial.get("evaluated") or trial.get("evaluation") or trial.get("judgment"))


def derive_next_decision(focus, trials=()):
    """Return the next concrete workflow action for a Focus, without changing state."""
    if not focus:
        return {"stage": "Orient", "label": "Choose a Focus", "href": "/status.html"}
    if not _contract_complete(focus):
        return {"stage": "Explain", "label": "Edit Focus", "href": _focus_url("/explore.html", focus)}
    focus_id = focus.get("focus_id")
    relevant = [trial for trial in trials if not focus_id or trial.get("focus_id") in (None, focus_id)]
    latest = relevant[-1] if relevant else None
    context = _scope_from_focus(focus)
    if latest is None:
        return {"stage": "Act", "label": "Draft a trial", "href": context_url("/cockpit.html", context)}
    if latest.get("status") in {"draft", "queued", "confirmed", "running", "in_progress"}:
        return {"stage": "Act", "label": "Review trial", "href": context_url("/cockpit.html", context)}
    if not _trial_evaluated(latest):
        return {"stage": "Learn", "label": "Evaluate results", "href": context_url("/runs.html", context)}
    return {"stage": "Learn", "label": "Revise Focus", "href": _focus_url("/explore.html", focus)}


def _source_tags(focus: dict[str, Any]) -> tuple[list[str], list[str]]:
    source = focus.get("source") or {}
    generated = source.get("member_tags") or source.get("adjacent_member_tags") or []
    anchors = source.get("real_anchor_tags") or []
    return [tag for tag in generated if isinstance(tag, str)], [tag for tag in anchors if isinstance(tag, str)]


def _stored_evidence(focus: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evidence = focus.get("evidence") or {}
    records: list[Any] = []
    if isinstance(evidence, dict):
        for key in ("generated_members", "members", "real_anchors", "anchors"):
            value = evidence.get(key, [])
            if isinstance(value, list):
                records.extend(value)
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        tag = record if isinstance(record, str) else record.get("tag") if isinstance(record, dict) else None
        if isinstance(tag, str):
            result[tag] = record if isinstance(record, dict) else {"tag": tag}
    return result


def _evidence_wall(focus: dict[str, Any]) -> list[dict[str, Any]]:
    generated, anchors = _source_tags(focus)
    stored = _stored_evidence(focus)
    chosen = generated[:4]
    anchor = anchors[0] if anchors else None
    if anchor is None and len(generated) > 4:
        chosen.append(generated[4])
    result: list[dict[str, Any]] = []
    for tag, role in [(tag, "generated_member") for tag in chosen] + ([(anchor, "real_anchor")] if anchor else []):
        record = stored.get(tag)
        item: dict[str, Any] = {"tag": tag, "role": role, "missing": record is None}
        if record:
            item.update({key: value for key, value in record.items() if key != "tag"})
            item["missing"] = bool(record.get("missing", False))
        result.append(item)
    return result


def _record_id(record: dict[str, Any], fallback: str) -> str:
    for key in ("record_id", "id", "thread_id", "trial_id", "launch_id", "focus_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def _record_timestamp(record: dict[str, Any]) -> str:
    for key in ("timestamp", "updated_at", "created_at"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def _activity(focus: dict[str, Any] | None, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if focus:
        focus_id = str(focus.get("focus_id", "focus"))
        if focus.get("created_at"):
            events.append({"record_id": f"{focus_id}:created", "timestamp": focus["created_at"], "label": "Focus created"})
        if focus.get("updated_at") and focus.get("updated_at") != focus.get("created_at"):
            events.append({"record_id": f"{focus_id}:updated", "timestamp": focus["updated_at"], "label": "Focus updated"})
    for index, record in enumerate(records):
        if isinstance(record, dict):
            event = dict(record)
            event["record_id"] = _record_id(record, f"record-{index}")
            event["timestamp"] = _record_timestamp(record)
            events.append(event)
    return sorted(events, key=lambda event: (event.get("timestamp", ""), event.get("record_id", "")))


def build_explore_data(context, foci, trials=(), guide_threads=(), launches=()):
    """Shape the leg-wide or Focus-scoped records consumed by :func:`render_html`."""
    focus = context.focus
    if focus is not None:
        focus_id = focus.get("focus_id")
        focus = next((record for record in foci if record.get("focus_id") == focus_id), focus)
    open_foci = [record for record in foci if record.get("status", "open") == "open"]
    selected_trials = [record for record in trials if isinstance(record, dict)]
    if focus is not None:
        focus_id = focus.get("focus_id")
        selected_trials = [record for record in selected_trials if record.get("focus_id") in (None, focus_id)]
    decision = derive_next_decision(focus, selected_trials)
    latest = selected_trials[-1] if selected_trials else None
    return {
        "context": {"expedition": context.expedition, "leg": context.leg},
        "focus": focus,
        "open_foci": open_foci,
        "saved_observations": ([focus["observation"]] if focus and focus.get("observation") else []),
        "evidence": _evidence_wall(focus) if focus else [],
        "activity": _activity(focus, list(guide_threads) + selected_trials + list(launches)),
        "readiness": {"contract_complete": _contract_complete(focus), "has_trial": latest is not None, "trial_status": latest.get("status") if latest else None, "trial_evaluated": _trial_evaluated(latest) if latest else False},
        "next_decision": decision,
        "trials": selected_trials,
    }


def _esc(value: Any) -> str:
    return html.escape(str(value))


def render_html(active_expedition=None, active_leg=None, running=None, data=None, context=None):
    context = context or WorkspaceContext(active_expedition, active_leg)
    data = data or build_explore_data(context, [])
    focus = data.get("focus")
    context = WorkspaceContext(context.expedition, context.leg, focus)
    evidence = "".join(_evidence_html(item, context) for item in data.get("evidence", [])) or '<p class="subtle">No saved evidence in this Focus.</p>'
    observations = "".join(f"<li>{_esc(item)}</li>" for item in data.get("saved_observations", [])) or "<li>No saved observations yet.</li>"
    activity = "".join(f'<li class="light-detent"><time>{_esc(event.get("timestamp", ""))}</time>{_esc(event.get("label") or event.get("status") or event.get("record_id"))}</li>' for event in data.get("activity", [])) or '<li class="light-detent">No activity recorded yet.</li>'
    decision = data["next_decision"]
    desk = _focus_desk(focus, data, context, evidence, observations, activity, decision) if focus else _focus_list(data, context)
    descriptions = {path: (name, desc) for path, name, desc in TOOLS}
    groups = [group for group in NAV_GROUPS if group[0] != "Explore"]
    tool_index = "".join(f'<section><h2>{_esc(group)}</h2><div class="tool-index">' + "".join(f'<a href="{_esc(context_url(path, context))}"><span class="name">{_esc(descriptions[path.lstrip("/")][0])}</span><span class="desc">{html.escape(descriptions[path.lstrip("/")][1], quote=False)}</span></a>' for path, _ in group_tools) + "</div></section>" for group, group_tools in groups)
    search_round_details = '''<details style="margin:16px 0"><summary style="cursor:pointer;font-weight:600">How a search round works</summary><ol style="margin:8px 0 0;padding-left:22px;line-height:1.7"><li><strong>Orient</strong> &mdash; Choose an expedition and leg, then name a visual question worth answering.</li><li><strong>Scout</strong> &mdash; Find a cluster, anchor, or frontier cell worth investigating.</li><li><strong>Explain</strong> &mdash; Separate what you observe from what you interpret, changing one variable at a time.</li><li><strong>Act</strong> &mdash; Turn one Focus revision into a bounded trial with a spend cap.</li><li><strong>Learn</strong> &mdash; Review results against the evidence and record your judgment.</li></ol></details>'''
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>CLAWMARKS research desk</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>{SULFUR_FONT_CSS}{SULFUR_CSS}{CONTROL_CSS}{TOPNAV_CSS}{MOBILE_BASE_CSS}{INFOTIP_CSS}
main{{max-width:1220px;margin:0 auto;padding:22px 24px 52px}}h1{{font:600 clamp(28px,5vw,48px)/1 var(--font-display);margin:0}}.desk-scope{{display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin:18px 0 10px;flex-wrap:wrap}}.desk-scope span{{font:12px var(--font-mono);color:var(--text-soft)}}.subtle,.sub{{color:var(--text-soft)}}.workspace-grid{{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:22px;margin-top:22px}}.focus-title{{font:600 31px/1.05 var(--font-display);margin:0 0 5px}}.question{{border-left:4px solid var(--sulfur);padding:9px 12px;margin:15px 0;background:var(--paper-deep)}}.mounted{{padding:14px}}.evidence-wall{{display:grid;grid-template-columns:repeat(5,minmax(100px,1fr));gap:10px}}.evidence-item{{margin:0;border:1px solid var(--rule);background:var(--paper-deep);min-width:0}}.evidence-item img,.evidence-missing{{width:100%;aspect-ratio:1;object-fit:cover;display:flex;align-items:center;justify-content:center;font:11px var(--font-mono);text-align:center;padding:8px}}.evidence-item figcaption{{padding:6px;font-size:11px;color:var(--text-soft)}}.evidence-item.missing{{border-style:dashed}}.tabs{{display:flex;gap:8px;border-bottom:1px solid var(--rule);margin-top:18px}}.tab{{border:0;background:none;padding:8px 2px;margin-right:12px;font-weight:600;cursor:pointer}}.next-decision{{padding:15px;border:2px solid var(--ink);margin-bottom:16px}}.next-decision h... (line truncated to 2000 chars)
</style></head><body>{nav_bar_html("/", active_expedition=context.expedition, active_leg=context.leg, running=running, focus=focus)}<main><div class="desk-scope"><h1>{_esc(focus.get("label") or "Untitled Focus") if focus else "No Focus selected"}</h1><span>{_esc(context.expedition or "no expedition")} / {_esc(context.leg or "no leg")}</span></div>{search_round_details}{desk}<section aria-label="All tools"><h2>All tools</h2>{tool_index}</section></main><script src="scrollnav.js"></script><script src="infotip.js"></script><script src="/shared-ui.js"></script></body></html>'''


def _evidence_html(item: dict[str, Any], context: WorkspaceContext) -> str:
    tag = _esc(item["tag"])
    body = f'<div class="evidence-missing">missing: {tag}</div>' if item["missing"] else f'<img src="{_esc(generated_image_url(item["tag"], context, thumbnail=True))}" alt="{_esc(item["role"])} {tag}">'
    return f'<figure class="evidence-item {"missing" if item["missing"] else ""}">{body}<figcaption>{_esc(item["role"])} · {tag}</figcaption></figure>'


def _focus_desk(focus, data, context, evidence, observations, activity, decision):
    next_href = _esc(decision["href"])
    return f'''<section class="focus-summary"><p class="question" id="focusQuestion"><strong>Question:</strong> {_esc(focus.get("question") or "State a visual question to begin.")}</p><div class="tabs" role="tablist"><button class="tab" type="button" data-tab="focus" aria-selected="true" aria-controls="focusPanel">Focus</button><button class="tab" type="button" data-tab="observations" aria-selected="false" aria-controls="observationsPanel">Saved Observations</button></div><div id="focusPanel" class="tabPanel"><div class="workspace-grid"><section><h2 class="focus-title">Evidence scope</h2><div class="mounted mounted-evidence evidence-wall" aria-label="Focus evidence">{evidence}</div></section><aside><div class="next-decision raised-readout"><h2>Next Decision</h2><a href="{next_href}"><strong>{_esc(decision["label"])}</strong></a><p>Stage: {_esc(decision["stage"])}</p></div><h2>Activity</h2><ul class="activity">{activity}</ul></aside></div></div><div id="observationsPanel" class="tabPanel" hidden><h2>Saved Observations</h2><ul>{observations}</ul></div></section><script>document.querySelectorAll('[data-tab]').forEach(tab => tab.addEventListener('click', () => {{ document.querySelectorAll('[data-tab]').forEach(item => item.setAttribute('aria-selected', item === tab ? 'true' : 'false')); document.querySelectorAll('.tabPanel').forEach(panel => panel.hidden = panel.id !== tab.getAttribute('aria-controls')); }}));</script>'''


def _focus_list(data, context):
    rows = "".join(f'<li><a href="{_esc(context_url("/explore.html", WorkspaceContext(context.expedition, context.leg, item)))}">{_esc(item.get("label") or item.get("focus_id"))}</a><span>r{_esc(item.get("revision", "?"))}</span></li>' for item in data.get("open_foci", [])) or '<li class="subtle">No open Foci in this expedition and leg.</li>'
    return f'<section><h2>Open Foci</h2><p class="sub">Explore keeps the choice explicit. No Focus is selected automatically.</p><ul class="focus-list" aria-label="Open Foci">{rows}</ul><div class="workflow-actions"><a class="raised-control" href="{_esc(context_url("/map.html", context, False))}">Create from Map</a><a class="raised-control" href="{_esc(context_url("/coverage.html", context, False))}">Create from Coverage</a></div></section>'
