# tests/test_rate_page.py
from clawmarks.build import rate_page


def test_render_html_includes_rate_api_calls():
    html = rate_page.render_html()
    assert "/api/rate/next" in html
    assert "/api/rate" in html
