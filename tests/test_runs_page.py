from clawmarks.build import runs_page


def test_render_html_uses_panel_token_for_secondary_surfaces():
    html = runs_page.render_html()

    assert "select, button { font-size:13px; padding:6px 12px; border-radius:6px; border:1px solid var(--border);\n  background:var(--panel-2);" in html
    assert ".stat { background:var(--panel-2); border-radius:6px; padding:10px 12px; }" in html
