# tests/test_shared_ui.py
import json

from clawmarks.shared_ui import NAV_OPTIONS, json_script, nav_bar_html


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
