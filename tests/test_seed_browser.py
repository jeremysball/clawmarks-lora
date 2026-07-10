from clawmarks.build import seed_browser


def test_render_html_includes_seed_generation_ui():
    html = seed_browser.render_html()
    assert "/api/seeds" in html
    assert "genBtn" in html
