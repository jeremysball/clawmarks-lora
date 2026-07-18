from clawmarks.build import seed_browser


def test_render_html_includes_seed_generation_ui():
    html = seed_browser.render_html()
    assert "/api/seeds" in html
    assert "genBtn" in html


def test_render_html_scopes_seed_api_calls_to_the_rendered_context():
    html = seed_browser.render_html("demo", "round1", focus={"focus_id": "focus_abc"})

    assert "const CONTEXT =" in html
    assert "function scopedApi(path)" in html
    assert "fetch(scopedApi('/api/seeds'))" in html
    assert "fetch(scopedApi('/api/seeds/generate')" in html
    assert "focus_id" in html


def test_render_html_uses_sulfur_proof_shell():
    """Task 5 render contract: the page sits on the Sulfur Proof foundation, includes the
    shared header's context-switcher script, ships a semantic <header>, and has no
    prefers-color-scheme: dark branch (Sulfur Proof is the only theme)."""
    html = seed_browser.render_html()
    assert "--paper:#C3C5BA" in html
    assert "shared-ui.js" in html
    assert "<header" in html
    assert "prefers-color-scheme: dark" not in html


def test_render_html_marks_generate_as_billable_action():
    html = seed_browser.render_html()
    assert 'id="genBtn" class="primary-action billable-action"' in html
    assert "Spends money" in html


def test_safe_actions_on_seed_page_do_not_have_billable_markup():
    html = seed_browser.render_html()
    assert 'id="genBtn" class="primary-action billable-action"' in html
    for marker in (
            'class="topnav',
            'class="context-label',
            'class="guide-button',
            'class="session-status',
            'class="wordmark',
        ):
        assert marker in html, f"missing {marker}"
        idx = html.index(marker)
        snippet = html[idx:idx + len(marker) + 30]
        assert 'billable-action' not in snippet, f"safe element {marker} has unwanted billable-action"


def test_render_html_drops_legacy_dark_theme_and_rounded_cards():
    """The legacy dark theme (`#cddcff` / `#2a4a7c` / `#3a5a8c` button accents, the
    `rgba(124,158,255,0.08)` translucent .seed.new accent, the `#e0605e` error text) and
    the legacy rounded card treatments (`border-radius:10px` on the `#genPanel`,
    `border-radius:7px` on the `#genBtn`, `border-radius:6px` on the `#genPanel` input,
    `border-radius:8px` on the `.seed` card) are all gone. The new `#genBtn` (the page's
    primary action -- generating new seeds) picks up CONTROL_CSS's `.primary-action`
    black-fill/sulfur-underline class, not a rounded filled card. The page-local CSS carries
    no legacy `background:var(--panel-2)` filled-card backgrounds."""
    html = seed_browser.render_html()

    # The legacy rounded `#genPanel` card is gone.
    assert "#genPanel { background:var(--panel); border:1px solid var(--border); border-radius:10px;" not in html
    # The legacy blue `.genBtn` is gone (it had a 7px radius, a #2a4a7c background, and a
    # #cddcff foreground; the migrated genBtn is the primary-action class with the Sulfur
    # paper/ink/sulfur tokens).
    assert "#genBtn { background:#2a4a7c; border:1px solid #3a5a8c; color:#cddcff; border-radius:7px;" not in html
    # The legacy rounded `.seed` card is gone.
    assert ".seed { background:var(--panel); border:1px solid var(--border); border-radius:8px;" not in html
    # The legacy translucent .seed.new accent is gone (it used rgba(124,158,255,0.08)).
    assert "border-color: var(--accent); background: rgba(124,158,255,0.08);" not in html
    # The generate button is the page's primary action.
    assert "primary-action" in html
    # The legacy filled-card backgrounds are gone.
    assert "background:var(--panel); border:1px solid var(--border);" not in html
    # The legacy system-font stack on the body override is gone (Sulfur's body font comes
    # from SULFUR_CSS, which uses the bundled IBM Plex Sans).
    assert "font-family: -apple-system, BlinkMacSystemFont" not in html
