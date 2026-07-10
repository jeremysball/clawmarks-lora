# tests/test_rate_page.py
from clawmarks.build import rate_page


def test_render_html_includes_rate_api_calls():
    html = rate_page.render_html()
    assert "/api/rate/next" in html
    assert "/api/rate" in html


def test_render_html_has_no_tap_buttons():
    html = rate_page.render_html()
    assert "<button" not in html
    assert 'id="buttons"' not in html


def test_render_html_has_imgwrap_and_overlay():
    html = rate_page.render_html()
    assert 'id="imgwrap"' in html
    assert 'id="swipe-overlay"' in html


def test_render_html_has_zoom_machinery():
    html = rate_page.render_html()
    assert "function zoomIn(" in html
    assert "function resetZoom(" in html
    assert "function clampOffset(" in html
    assert "dblclick" in html
    assert "classList.add('zoomed')" in html


def test_render_html_has_touch_swipe_machinery():
    html = rate_page.render_html()
    assert "addEventListener('touchstart'" in html
    assert "addEventListener('touchmove'" in html
    assert "addEventListener('touchend'" in html
    assert "SWIPE_THRESHOLD_FRACTION" in html
    assert "function finishSwipe(" in html
