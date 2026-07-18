# tests/test_shared_ui.py
import json

from clawmarks import shared_ui
from clawmarks.shared_ui import (
    NAV_GROUPS, NAV_OPTIONS, _LIGHTBOX_JS, json_script, nav_bar_html,
    billable_badge,
)


def test_nav_options_includes_preference_status_page():
    hrefs = [href for href, _label in NAV_OPTIONS]
    assert "/preference_status.html" in hrefs


def test_nav_options_has_compare_not_rate():
    hrefs = [href for href, _ in NAV_OPTIONS]
    assert "/compare.html" in hrefs
    assert "/rate.html" not in hrefs


def test_nav_bar_html_marks_preference_status_selected_when_current():
    html = nav_bar_html("/preference_status.html")
    assert 'value="/preference_status.html" selected' in html


def test_nav_bar_html_omits_active_leg_without_selection():
    html = nav_bar_html("/scan.html")

    assert 'class="nav-activeleg"' not in html


def test_nav_bar_groups_tools_and_links_active_context_to_home():
    html = nav_bar_html(
        "/compare.html", active_expedition="demo", active_leg="round1"
    )

    assert '<optgroup label="Look at images">' in html
    assert '<optgroup label="Make new images">' in html
    assert '<optgroup label="Understand the search">' in html
    assert '<optgroup label="Preference model">' in html
    assert 'href="/?expedition=demo&amp;leg=round1"' in html
    assert "demo/round1" in html


def test_nav_bar_active_context_button_announces_dialog_and_carries_data():
    """Task 3 replaced the old `<a class="nav-activeleg" href="/">` link with a context
    button that opens the context-switcher dialog. The data attributes and accessible
    relationships are how SHARED_UI_JS knows which (expedition, leg) to mark active and
    which dialog to show; verify they're wired correctly."""
    html = nav_bar_html("/scan.html", active_expedition="demo", active_leg="leg-b")

    assert 'id="contextPicker"' in html
    assert 'data-expedition="demo"' in html
    assert 'data-leg="leg-b"' in html
    assert 'aria-controls="contextDialog"' in html
    assert 'aria-haspopup="dialog"' in html
    assert "demo/leg-b" in html


def test_json_script_escapes_close_script_sequence():
    """Regression test for issue #17: model-generated text embedded in a page's <script> tag via
    raw json.dumps() could contain a literal "</script>", closing the script block early and
    letting whatever follows execute as HTML/JS. json_script must neutralize that."""
    payload = [{"prompt": "a cat </script><script>alert(1)</script>"}]
    escaped = json_script(payload)
    assert "</script>" not in escaped
    assert json.loads(escaped) == payload


def test_json_script_round_trips_normal_data():
    payload = {"tag": "gen1_foo", "novelty": 0.5, "sim": ["a", "b"]}
    assert json.loads(json_script(payload)) == payload


def test_nav_bar_shows_active_leg():
    html = nav_bar_html("/compare.html", active_expedition="uncanny_frontier", active_leg="round2")
    assert "uncanny_frontier" in html
    assert "round2" in html


def test_nav_bar_omits_label_when_no_selection():
    html = nav_bar_html("/compare.html")
    assert "nav-activeleg" not in html


def test_nav_bar_shows_running_indicator():
    html = nav_bar_html("/runs.html", running=("trent_v3_epoch4", "freeform1"))
    assert "RUNNING" in html
    assert "trent_v3_epoch4/freeform1" in html


def test_nav_bar_omits_running_indicator_when_none():
    html = nav_bar_html("/runs.html")
    assert "nav-running" not in html
    assert "RUNNING" not in html


def test_nav_groups_use_plain_task_labels():
    groups = dict(NAV_GROUPS)
    assert tuple(groups) == (
        "Look at images", "Make new images", "Understand the search", "Preference model",
    )
    assert groups["Look at images"] == [
        ("/scan.html", "Browse all images"),
        ("/archive.html", "Best images by area"),
        ("/compare.html", "Choose between two images"),
    ]
    assert ("/runs.html", "Run or monitor a search") in groups["Make new images"]
    assert ("/coverage.html", "Find gaps in the image space") in groups["Understand the search"]


def test_nav_has_no_workflow_stage_group():
    html = nav_bar_html("/")
    assert 'optgroup label="Explore"' not in html
    for label in ("Orient", "Scout", "Explain", "Act", "Learn"):
        assert f">{label}<" not in html


def test_dark_tokens_defines_pick_as_gold():
    from clawmarks import shared_ui

    assert "--pick:#f5c542" in shared_ui.DARK_TOKENS


# ---------------------------------------------------------------------------
# Sulfur Proof foundation: token, font, and depth constants (Task 2)
# ---------------------------------------------------------------------------


def test_sulfur_tokens_and_fonts_are_exact():
    """Brief Step 1 (verbatim): the Sulfur Proof token palette and the bundled-font @font-face
    declarations must match the brief exactly. The `https://` check enforces the plan's "zero
    runtime font requests" Global Constraint: any CDN reference fails this assertion."""
    from clawmarks import shared_ui

    assert "--paper:#C3C5BA" in shared_ui.SULFUR_CSS
    assert "--text-soft:#4D5048" in shared_ui.SULFUR_CSS
    assert "Barlow Condensed" in shared_ui.SULFUR_FONT_CSS
    assert "url('/assets/fonts/" in shared_ui.SULFUR_FONT_CSS
    assert "https://" not in shared_ui.SULFUR_FONT_CSS


def test_depth_uses_hard_shadows_and_reduced_motion():
    """Brief Step 1 (verbatim): CONTROL_CSS must use hard 4px 4px 0 offsets (no blur, per the
    plan's "hard unblurred depth only" Global Constraint), and SULFUR_CSS must honor
    prefers-reduced-motion at the universal level (covers drawer/sheet transitions, focus
    rings, and any later additions)."""
    from clawmarks import shared_ui

    assert "box-shadow:4px 4px 0" in shared_ui.CONTROL_CSS
    assert "@media (prefers-reduced-motion: reduce)" in shared_ui.SULFUR_CSS


def test_sulfur_css_defines_full_approved_palette():
    """The Sulfur Proof design system spec approves exactly eight color tokens with the listed
    hex values; rewriting any of them would break the contrast and meaning the spec is built
    on. Test them all together so a typo can't pass by matching only one or two by accident."""
    from clawmarks import shared_ui

    for token in (
        "--paper:#C3C5BA",
        "--paper-deep:#B3B5A9",
        "--ink:#11120F",
        "--text-soft:#4D5048",
        "--rule:#898D81",
        "--sulfur:#CBD63F",
        "--guide-surface:#20251B",
        "--guide-ink:#ECEFDF",
    ):
        assert token in shared_ui.SULFUR_CSS, f"missing approved token {token}"


def test_sulfur_css_preserves_legacy_aliases():
    """Later page-migration tasks depend on the legacy `--bg`, `--panel`, `--panel-2`,
    `--border`, `--text`, `--text-dim`, `--accent` aliases coexisting alongside the new
    Sulfur tokens so migration can proceed incrementally without a flag day. The brief's own
    SULFUR_CSS block defines all seven inside :root; verify they survive verbatim."""
    from clawmarks import shared_ui

    for alias in (
        "--bg:",
        "--panel:",
        "--panel-2:",
        "--border:",
        "--text:",
        "--text-dim:",
        "--accent:",
    ):
        assert alias in shared_ui.SULFUR_CSS, f"legacy alias {alias} missing from SULFUR_CSS"


def test_sulfur_css_uses_bundled_font_urls_only():
    """Plan Global Constraint: SULFUR_FONT_CSS must reference only bundled /assets/fonts/
    URLs, never a CDN. Reinforced by the brief's `https://` check; this one also confirms the
    actual URL path is the same one Task 1 wired into curation_server.py."""
    from clawmarks import shared_ui

    fonts = (
        "BarlowCondensed-SemiBold.ttf",
        "BarlowCondensed-ExtraBold.ttf",
        "IBMPlexSans-Variable.ttf",
        "IBMPlexMono-Regular.ttf",
        "IBMPlexMono-SemiBold.ttf",
    )
    for name in fonts:
        assert f"/assets/fonts/{name}" in shared_ui.SULFUR_FONT_CSS, (
            f"bundled font {name} not referenced from SULFUR_FONT_CSS"
        )


def test_control_css_defines_all_five_depth_classes():
    """The spec recognizes three approved depth strengths but expresses them through five named
    classes that page migration will reuse. All five must exist in CONTROL_CSS so a page that
    reaches for `.recessed-readout` or `.light-detent` doesn't silently get an unstyled span."""
    from clawmarks import shared_ui

    for cls in (
        ".raised-control",
        ".raised-readout",
        ".mounted-evidence",
        ".light-detent",
        ".recessed-readout",
    ):
        assert f"{cls} {{" in shared_ui.CONTROL_CSS or f"{cls}{{" in shared_ui.CONTROL_CSS, (
            f"CONTROL_CSS is missing class definition for {cls}"
        )


def test_control_css_shadows_are_hard_edged():
    """Plan Global Constraint: "hard unblurred depth only" — every box-shadow inside CONTROL_CSS
    must use 0 blur. A blur radius (the third length in a box-shadow, before the color) is
    prohibited; this test asserts no such blurred shadow is present anywhere in the constant.
    inset shadows are allowed because they describe inner edges, not outer depth."""
    from clawmarks import shared_ui

    import re

    for match in re.finditer(r"box-shadow\s*:\s*([^;]+);", shared_ui.CONTROL_CSS):
        declaration = match.group(1)
        # Each comma-separated layer in a box-shadow list is "<offset-x> <offset-y> [blur]
        # [spread] <color>?". We strip inset layers (whose blur would describe bevel softness,
        # not outer depth) and then check that no outer-shadow layer carries a non-zero blur.
        for layer in declaration.split(","):
            stripped = layer.strip()
            if stripped.startswith("inset"):
                continue
            tokens = stripped.split()
            # offset-x, offset-y, optional blur, optional spread, color(s)
            assert len(tokens) <= 4 or not _looks_like_length(tokens[2]), (
                f"CONTROL_CSS has a hard-edged violation: outer shadow layer {layer!r} "
                "appears to carry a blur radius. Use `4px 4px 0 <color>` form."
            )


def _looks_like_length(token):
    """True if token is a non-zero CSS length like '4px' or '1.5em'. Zero / unitless tokens
    are not blur values."""
    if token in {"0", "0px", "0em", "0rem"}:
        return False
    import re

    return bool(re.fullmatch(r"-?\d+(\.\d+)?(px|em|rem|vh|vw|%)", token))


def test_raised_control_uses_4px_offset_and_1px_inner_edges():
    """`.raised-control` is the depth grammar's default bounded control (paper-colored, not
    the black-fill `.primary-action` component). The brief pins it at `box-shadow:4px 4px 0`
    (its own test asserts this literal string) and a 1px border or inset edge. 1px ink border
    is the natural reading of "1px inner edges" — a deeper notch would make the control feel
    carded, which the spec prohibits."""
    from clawmarks import shared_ui

    css = shared_ui.CONTROL_CSS
    start = css.index(".raised-control {")
    end = css.index("}", start)
    block = css[start:end]
    assert "box-shadow:4px 4px 0" in block
    assert "1px" in block


def test_raised_readout_uses_quiet_3px_shadow():
    """`.raised-readout` is the spec's "Shallow raised readout": 3px hard shadow, visibly
    quieter than `.raised-control`/`mounted-evidence`. The 3px (not 4px) is the
    distinguishing mark."""
    from clawmarks import shared_ui

    css = shared_ui.CONTROL_CSS
    start = css.index(".raised-readout {")
    end = css.index("}", start)
    block = css[start:end]
    assert "box-shadow:3px 3px 0" in block


def test_mounted_evidence_uses_stronger_offset_than_raised_control():
    """`.mounted-evidence` is the spec's "Mounted working piece" for evidence images and
    similar bounded working pieces. It is the strongest depth treatment: 4-5px hard shadow.
    Since `.raised-control` already takes 4px, `.mounted-evidence` must be 5px to be visibly
    stronger (otherwise the two strengths collapse into one)."""
    from clawmarks import shared_ui

    css = shared_ui.CONTROL_CSS
    start = css.index(".mounted-evidence {")
    end = css.index("}", start)
    block = css[start:end]
    assert "box-shadow:5px 5px 0" in block


def test_light_detent_has_no_outer_shadow():
    """`.light-detent` is the spec's "Light detent": a 1px rule and 1-2px reversed inner
    edges with NO outer shadow. If a translation-on-active rule drops the shadow and we
    accidentally added an outer shadow here, the visual difference from `.raised-readout`
    collapses."""
    from clawmarks import shared_ui

    css = shared_ui.CONTROL_CSS
    start = css.index(".light-detent {")
    end = css.index("}", start)
    block = css[start:end]
    # No outer (non-inset) shadow layer. Inset shadows describing reversed inner edges are
    # the whole point of this class.
    import re

    for match in re.finditer(r"box-shadow\s*:\s*([^;]+);", block):
        for layer in match.group(1).split(","):
            assert layer.strip().startswith("inset"), (
                ".light-detent must have no outer box-shadow; got non-inset layer "
                f"{layer!r}"
            )


def test_recessed_readout_has_no_outer_shadow():
    """`.recessed-readout` is the recessed treatment for context receipts, composers, and
    counters. Like `.light-detent`, no outer shadow; distinguished by name and purpose, not
    shadow shape."""
    from clawmarks import shared_ui

    css = shared_ui.CONTROL_CSS
    start = css.index(".recessed-readout {")
    end = css.index("}", start)
    block = css[start:end]
    import re

    for match in re.finditer(r"box-shadow\s*:\s*([^;]+);", block):
        for layer in match.group(1).split(","):
            assert layer.strip().startswith("inset"), (
                ".recessed-readout must have no outer box-shadow; got non-inset layer "
                f"{layer!r}"
            )


def test_raised_states_have_pressed_translate_and_no_outer_shadow():
    """Plan Global Constraint: pressed controls translate by their shadow offset and remove
    the outer shadow. Verify both `:active` and `.pressed` (some pages apply the class instead
    of relying on :active) follow the rule, with per-class offset matching each class's resting
    shadow depth."""
    from clawmarks import shared_ui

    css = shared_ui.CONTROL_CSS
    # (selector -> expected pressed-translate)
    cases = (
        (".raised-control:active", "translate(4px,4px)"),
        (".raised-control.pressed", "translate(4px,4px)"),
        (".raised-readout:active", "translate(3px,3px)"),
        (".raised-readout.pressed", "translate(3px,3px)"),
        (".mounted-evidence:active", "translate(5px,5px)"),
        (".mounted-evidence.pressed", "translate(5px,5px)"),
    )
    for selector, expected_translate in cases:
        start = css.index(selector)
        brace = css.index("{", start)
        end = css.index("}", brace)
        block = css[brace:end]
        assert f"transform:{expected_translate}" in block or f"transform: {expected_translate}" in block, (
            f"{selector} must translate by its shadow offset ({expected_translate}) on press; "
            f"got block {block!r}"
        )
        import re

        for match in re.finditer(r"box-shadow\s*:\s*([^;]+);", block):
            for layer in match.group(1).split(","):
                assert layer.strip().startswith("inset"), (
                    f"{selector} must remove outer box-shadow on press; got {layer!r}"
                )


def test_raised_control_hover_increases_offset():
    """Plan Global Constraint: hover states increase the hard offset by 1px-2px. .raised-control
    starts at 4px 4px 0, so the hover offset should be 5px or 6px."""
    from clawmarks import shared_ui

    css = shared_ui.CONTROL_CSS
    start = css.index(".raised-control:hover")
    brace = css.index("{", start)
    end = css.index("}", brace)
    block = css[brace:end]
    assert "5px 5px 0" in block or "6px 6px 0" in block


def test_primary_action_uses_black_fill_and_sulfur_underline():
    """Spec section "Controls > Primary actions": black fill with paper text, sulfur as a
    bottom registration mark or short underline. This is a distinct component from
    `.raised-control` (the depth-grammar's default control, paper-colored) — Task 2 of the
    original sulfur-proof-shared-shell plan only extracted the Dimensional Grammar section
    into CSS and never built this Controls-section component, so no button on the live site
    could ever look like a primary action. `.primary-action` closes that gap."""
    from clawmarks import shared_ui

    css = shared_ui.CONTROL_CSS
    start = css.index(".primary-action {")
    end = css.index("}", start)
    block = css[start:end]
    assert "background:var(--ink)" in block
    assert "color:var(--paper)" in block
    assert "border-bottom" in block and "var(--sulfur)" in block


def test_primary_action_press_state_flattens_to_sulfur_zero_shadow():
    """Matches the decided Lavish artifact's sulfur-theme `.mark-button:active` rule: pressed
    primary actions translate into their offset and settle to a flat `0 0 0` sulfur shadow,
    distinct from the depth-grammar classes' inset-only press state."""
    from clawmarks import shared_ui

    css = shared_ui.CONTROL_CSS
    start = css.index(".primary-action:active")
    brace = css.index("{", start)
    end = css.index("}", brace)
    block = css[brace:end]
    assert "box-shadow:0 0 0 var(--sulfur)" in block


def test_topnav_css_has_responsive_breakpoint_and_context_label():
    """Task 3's brief asserts both `@media (max-width:700px)` and `.context-label` are present
    in TOPNAV_CSS. This task owns the constant but Task 3 fills in the header-specific rules;
    the breakpoint scaffold and the .context-label class stub belong here so Task 3 extends,
    rather than rewrites, the constant."""
    from clawmarks import shared_ui

    assert "@media (max-width:700px)" in shared_ui.TOPNAV_CSS
    assert ".context-label" in shared_ui.TOPNAV_CSS


def test_mobile_base_css_enforces_44px_touch_targets():
    """Plan Global Constraint + Task 6 verification: at <=700px, buttons/inputs/icon controls
    must provide at least a 44px by 44px touch target. MOBILE_BASE_CSS is the constant that
    enforces this globally so individual pages don't each reinvent the rule."""
    from clawmarks import shared_ui

    assert "min-height:44px" in shared_ui.MOBILE_BASE_CSS
    assert "@media (max-width:700px)" in shared_ui.MOBILE_BASE_CSS


def test_lightbox_undo_flow_honors_recovery_contract():
    undo_start = _LIGHTBOX_JS.index("function showUndoFavorite")
    undo_js = _LIGHTBOX_JS[undo_start:]
    toggle_start = _LIGHTBOX_JS.index("function toggleFavorite")
    toggle_js = _LIGHTBOX_JS[toggle_start:undo_start]

    assert "let undoBtn = null;" in _LIGHTBOX_JS
    assert "if (undoBtn) { undoBtn.remove(); undoBtn = null; }" in undo_js
    assert "const undoBtn =" not in undo_js
    onclick_js = undo_js[undo_js.index("undoBtn.onclick"):]
    assert "document.dispatchEvent(new CustomEvent('lightbox:favorite'" in onclick_js
    assert "if (res.error) throw new Error(res.error);" in toggle_js
    assert "if (res.error) throw new Error(res.error);" in onclick_js


# ---------------------------------------------------------------------------
# Task 3: semantic shared header + context-switcher dialog
# ---------------------------------------------------------------------------


def test_header_names_page_scope_focus_and_guide():
    markup = nav_bar_html(
        "/map.html", "demo", "round1",
        focus={"focus_id": "focus_11111111111111111111111111111111", "label": "Ink anchor", "revision": 3},
    )
    assert "CLAWMARKS" in markup
    assert "Solution map" in markup
    assert "demo/round1" in markup
    assert "Ink anchor" in markup and "r3" in markup
    assert 'id="contextPicker"' in markup
    assert 'id="guideOpen"' in markup
    assert "<dialog" in markup


def test_nav_bar_scopes_every_tool_link_to_focus():
    markup = nav_bar_html(
        "/map.html", "demo", "round1",
        focus={"focus_id": "focus_11111111111111111111111111111111", "label": "Ink anchor", "revision": 3},
    )

    assert 'value="/redundancy.html?expedition=demo&amp;leg=round1&amp;focus_id=focus_11111111111111111111111111111111"' in markup
    assert 'href="/status.html?expedition=demo&amp;leg=round1&amp;focus_id=focus_11111111111111111111111111111111"' in markup


def test_scoped_href_preserves_leg_scope_without_focus():
    assert shared_ui.scoped_href("/map.html", "demo", "round1") == (
        "/map.html?expedition=demo&amp;leg=round1"
    )


def test_narrow_header_keeps_context_and_guide_labels():
    assert "@media (max-width:700px)" in shared_ui.TOPNAV_CSS
    assert ".context-label" in shared_ui.TOPNAV_CSS


# ---------------------------------------------------------------------------
# Leg-picker-into-shell: create expedition/leg from the shared context dialog
# ---------------------------------------------------------------------------


def test_context_dialog_offers_create_expedition_and_create_leg_forms():
    """Approved via Lavish review (2026-07-18): expedition/leg creation, previously only
    available on /status.html, now lives in the shared contextDialog so it's reachable from
    every page, not just the one that used to own the picker."""
    markup = nav_bar_html("/map.html", "demo", "round1")
    assert 'id="contextNewExpToggle"' in markup
    assert 'id="contextNewExpForm"' in markup
    assert 'id="contextNewExpName"' in markup
    assert 'id="contextNewLegToggle"' in markup
    assert 'id="contextNewLegForm"' in markup
    assert 'id="contextNewLegExpedition"' in markup
    assert 'id="contextNewLegName"' in markup


def test_topnav_css_styles_context_create_forms():
    assert ".context-create" in shared_ui.TOPNAV_CSS


def test_shared_ui_js_wires_up_create_expedition_and_create_leg():
    js = shared_ui.SHARED_UI_JS
    assert "contextNewExpToggle" in js
    assert "contextNewLegToggle" in js
    assert "/api/expeditions'" in js or '/api/expeditions"' in js
    assert "/api/legs'" in js or '/api/legs"' in js


# ---------------------------------------------------------------------------
# Task 3: Shared Glossary And Information Controls
# ---------------------------------------------------------------------------


def test_glossary_holds_plain_and_formal_metric_names_together():
    assert shared_ui.GLOSSARY["faithfulness"][0] == "Similarity to real art"
    assert "DINOv2 cosine similarity" in shared_ui.GLOSSARY["faithfulness"][1]
    assert shared_ui.GLOSSARY["novelty"][0] == "How new or different"


def test_info_btn_with_glossary_key_is_accessible_button():
    markup = shared_ui.info_btn("novelty")
    assert markup.startswith('<button type="button"')
    assert '>i</button>' in markup
    assert 'aria-label="More information about How new or different"' in markup
    assert "?" not in markup


def test_info_btn_with_raw_text_fallback_still_works():
    markup = shared_ui.info_btn("Some fallback raw tip text")
    assert markup.startswith('<span class="infobtn"')
    assert 'data-tip="Some fallback raw tip text"' in markup
    assert "?" in markup


def test_info_btn_glossary_key_has_no_question_mark():
    markup = shared_ui.info_btn("faithfulness")
    assert "?" not in markup


def test_glossary_defines_required_entries():
    for key in ("faithfulness", "novelty", "map_elites_cell", "umap", "redundancy"):
        assert key in shared_ui.GLOSSARY
        label, definition = shared_ui.GLOSSARY[key]
        assert label
        assert definition


def test_info_btn_sets_aria_expanded_false():
    markup = shared_ui.info_btn("novelty")
    assert 'aria-expanded="false"' in markup


def test_infotip_js_updates_aria_expanded_and_handles_escape():
    js = shared_ui.INFOTIP_JS
    assert 'setAttribute("aria-expanded"' in js or "setAttribute('aria-expanded'" in js
    assert "Escape" in js or "escape" in js.lower() or "27" in js
    assert ".focus()" in js


def test_mobile_base_css_keeps_overflow_wrap_anywhere():
    assert "overflow-wrap:anywhere" in shared_ui.MOBILE_BASE_CSS
    assert "code, pre { overflow-wrap:anywhere; }" in shared_ui.MOBILE_BASE_CSS


# ---------------------------------------------------------------------------
# Task 4: Billable Action Affordances
# ---------------------------------------------------------------------------


def test_billable_badge_never_invents_an_estimate():
    assert billable_badge() == '<span class="cost-badge">Spends money</span>'
    assert "~$2.00" in billable_badge("~$2.00")


def test_cost_token_in_sulfur_tokens():
    assert "--cost:#5B3A63" in shared_ui.SULFUR_CSS


def test_billable_action_css_class_exists():
    full_css = shared_ui.CONTROL_CSS
    assert ".billable-action" in full_css
    assert ".cost-badge" in full_css


def test_context_create_buttons_are_not_billable():
    markup = nav_bar_html("/map.html", "demo", "round1")
    assert 'class="context-create-btn primary-action"' in markup
    idx = markup.index('context-create-btn primary-action')
    snippet = markup[idx:idx + 60]
    assert 'billable-action' not in snippet


def test_billable_action_css_references_cost_token():
    full_css = shared_ui.CONTROL_CSS
    for cls in (".billable-action", ".cost-badge"):
        start = full_css.index(cls)
        end = full_css.index("}", start)
        block = full_css[start:end]
        assert "var(--cost)" in block, f"{cls} must reference --cost"
