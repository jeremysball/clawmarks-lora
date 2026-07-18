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


def test_render_html_uses_sulfur_proof_shell():
    """Task 5 render contract: the page sits on the Sulfur Proof foundation, includes the
    shared header's context-switcher script, ships a semantic <header>, and has no
    prefers-color-scheme: dark branch (Sulfur Proof is the only theme)."""
    html = runs_page.render_html()
    assert "--paper:#C3C5BA" in html
    assert "shared-ui.js" in html
    assert "<header" in html
    assert "prefers-color-scheme: dark" not in html


def test_render_html_uses_sulfur_proof_outcome_first_inline_statistics():
    """Task 5 brief, Step 3 (Runs): 'Runs leads with outcome and inline statistics.' The page
    must put the run status (the outcome: 'Not running.' / 'Running <exp>/<leg> (pid <pid>),
    started <ts>.') as the most prominent element at the top of the page body, and the
    per-run report statistics (generation, plateau count, total images, spend) must render
    as flat inline ruled rows of label:value pairs, not as bordered stat cards. The legacy
    `.stat { background:var(--panel-2); border-radius:6px; padding:10px 12px; }` filled card
    grid is gone; the new statistics are flat text on the paper background, separated by a
    rule line per the same pattern Task 4c used for redundancy/novelty/lineage evidence rows.
    The launch button is the page's primary action so it picks up CONTROL_CSS's
    `.primary-action` black-fill/sulfur-underline class; the stop button is a danger button
    and gets a depth treatment of its own. The legacy `border-radius:6px` on form controls is
    gone, per the no-rounded-card rule."""
    html = runs_page.render_html()

    # The launch button carries the primary-action class (the page's single commit action)
    # -- not a flat var(--panel-2) fill and not a border-radius:6px rounded bar.
    assert "primary-action" in html
    # The legacy filled `.stat` card grid is gone: no border-radius:6px on the stat
    # rounded card, and no background:var(--panel-2) on the .stat selector. The legacy
    # `select, button { ...; border-radius:6px; ... background:var(--panel-2); }` rounded
    # form-control fill is also gone (BTN_CSS is no longer imported).
    assert ".stat { background:var(--panel-2); border-radius:6px; padding:10px 12px; }" not in html
    assert "select, button { font-size:13px; padding:6px 12px; border-radius:6px; border:1px solid var(--border);\n  background:var(--panel-2);" not in html
    # The new statistics are flat ruled rows: the page-local CSS carries
    # `border-bottom:1px solid var(--rule)` for the ruled-row treatment, exactly like the
    # redundancy/novelty/lineage ruled rows in commit 8f36053 and the preference_status /
    # preference_rank evidence rows in commit 94fbe5c.
    assert "border-bottom:1px solid var(--rule)" in html
    # The legacy .panel wrapper (rounded filled card) is gone: no border-radius:8px on
    # the .panel selector.
    assert ".panel { background:var(--panel); border:1px solid var(--border); border-radius:8px;" not in html
    # Outcome (the status line) is the most prominent thing at the top: the statusline
    # container carries a substantial font size so the user sees the outcome first instead
    # of buried below the launch controls. The status line's `Not running.` text is the
    # legacy starting state and still appears.
    assert "Not running." in html
    assert "id=\"statusLine\"" in html
    # The four per-run report values are still rendered (just no longer as bordered cards):
    # generation, plateau count, total images, spend.
    assert 'id="statGen"' in html
    assert 'id="statPlateau"' in html
    assert 'id="statImages"' in html
    assert 'id="statSpend"' in html


def test_render_html_status_is_prominent_at_top_of_page_body():
    """Task 5 brief: 'Runs leads with outcome.' The page body order is: h1 (title) →
    intro paragraph → statusline (the outcome) → launch controls → per-run report. The
    statusline (which holds `Not running.` / `Running X/Y (pid Z), started T.` ) must come
    before the stat values in the rendered HTML body, so the user sees the outcome first
    instead of buried below the launch controls. The page body order is also asserted
    explicitly so a future refactor that reorders the sections fails the test rather than
    silently changing the visual hierarchy."""
    html = runs_page.render_html()
    body = html.split("</header>", 1)[1]
    h1_pos = body.index("<h1>")
    status_pos = body.index("id=\"statusLine\"")
    stat_grid_pos = body.index("id=\"statGen\"")
    assert h1_pos < status_pos < stat_grid_pos


def test_stop_request_uses_the_last_status_run_identity():
    html = runs_page.render_html()

    assert "let lastStatusPid = null;" in html
    assert "let lastStatusStartTicks = null;" in html
    assert "lastStatusPid = d.pid;" in html
    assert "lastStatusStartTicks = d.start_time_ticks;" in html
    assert "const confirmedPid = lastStatusPid;" in html
    assert "const confirmedStart = lastStatusStartTicks;" in html
    assert "JSON.stringify({pid: confirmedPid, start_time_ticks: confirmedStart})" in html
