from clawmarks.shared_ui import nav_bar_html


FOCUS = {
    "focus_id": "focus_11111111111111111111111111111111",
    "label": "Ink anchor",
    "revision": 3,
}


def test_all_focus_tool_links_preserve_complete_context():
    page = nav_bar_html("map.html", "demo", "round1", focus=FOCUS)
    suffix = "expedition=demo&amp;leg=round1&amp;focus_id=focus_11111111111111111111111111111111"

    assert suffix in page
    assert page.count(suffix) >= 3
