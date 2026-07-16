# tests/test_shared_ui.py
import json

from clawmarks.shared_ui import NAV_OPTIONS, _LIGHTBOX_JS, json_script, nav_bar_html


def test_nav_options_includes_preference_status_page():
    hrefs = [href for href, _label in NAV_OPTIONS]
    assert "preference_status.html" in hrefs


def test_nav_options_has_compare_not_rate():
    hrefs = [href for href, _ in NAV_OPTIONS]
    assert "compare.html" in hrefs
    assert "rate.html" not in hrefs


def test_nav_bar_html_marks_preference_status_selected_when_current():
    html = nav_bar_html("preference_status.html")
    assert 'value="preference_status.html" selected' in html


def test_nav_bar_html_omits_active_leg_without_selection():
    html = nav_bar_html("scan.html")

    assert 'class="nav-activeleg"' not in html


def test_nav_bar_groups_tools_and_links_active_context_to_home():
    html = nav_bar_html(
        "compare.html", active_expedition="demo", active_leg="round1"
    )

    assert '<optgroup label="Generate">' in html
    assert '<optgroup label="Curate">' in html
    assert '<optgroup label="Understand search">' in html
    assert '<optgroup label="Preference model">' in html
    assert 'href="/"' in html
    assert "demo/round1" in html


def test_nav_bar_active_context_badge_uses_served_root_route():
    html = nav_bar_html("scan.html", active_expedition="demo", active_leg="leg-b")

    assert '<a class="nav-activeleg" href="/">demo/leg-b</a>' in html


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
    html = nav_bar_html("compare.html", active_expedition="uncanny_frontier", active_leg="round2")
    assert "uncanny_frontier" in html
    assert "round2" in html


def test_nav_bar_omits_label_when_no_selection():
    html = nav_bar_html("compare.html")
    assert "nav-activeleg" not in html


def test_nav_bar_shows_running_indicator():
    html = nav_bar_html("runs.html", running=("trent_v3_epoch4", "freeform1"))
    assert "RUNNING" in html
    assert "trent_v3_epoch4/freeform1" in html


def test_nav_bar_omits_running_indicator_when_none():
    html = nav_bar_html("runs.html")
    assert "nav-running" not in html
    assert "RUNNING" not in html


def test_dark_tokens_defines_pick_as_gold():
    from clawmarks import shared_ui

    assert "--pick:#f5c542" in shared_ui.DARK_TOKENS


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
