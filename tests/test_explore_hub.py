from clawmarks.build import explore_hub


def test_render_html_lists_every_tool():
    html = explore_hub.render_html()
    for path, label, _desc in explore_hub.TOOLS:
        assert path in html
