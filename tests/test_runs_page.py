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


def test_render_html_uses_panel_token_for_secondary_surfaces():
    html = runs_page.render_html()

    assert "select, button { font-size:13px; padding:6px 12px; border-radius:6px; border:1px solid var(--border);\n  background:var(--panel-2);" in html
    assert ".stat { background:var(--panel-2); border-radius:6px; padding:10px 12px; }" in html


def test_stop_request_uses_the_last_status_run_identity():
    html = runs_page.render_html()

    assert "let lastStatusPid = null;" in html
    assert "let lastStatusStartTicks = null;" in html
    assert "lastStatusPid = d.pid;" in html
    assert "lastStatusStartTicks = d.start_time_ticks;" in html
    assert "const confirmedPid = lastStatusPid;" in html
    assert "const confirmedStart = lastStatusStartTicks;" in html
    assert "JSON.stringify({pid: confirmedPid, start_time_ticks: confirmedStart})" in html
