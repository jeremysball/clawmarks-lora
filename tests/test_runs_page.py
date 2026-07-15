from clawmarks.build import runs_page


def test_completed_report_links_activate_the_selected_leg_before_navigation():
    html = runs_page.render_html()

    assert "function openReportTool(event, path)" in html
    assert "event.preventDefault()" in html
    assert "fetch('/api/active-leg'" in html
    assert "location.href = path" in html
    assert 'onclick="openReportTool(event, \'scan.html\')"' in html
    assert 'onclick="openReportTool(event, \'coverage.html\')"' in html
    assert 'onclick="openReportTool(event, \'novelty_decay.html\')"' in html
