from clawmarks.build import explore_hub
from clawmarks.workspace_context import WorkspaceContext


def test_render_html_lists_every_tool():
    html = explore_hub.render_html()
    for path, label, _desc in explore_hub.TOOLS:
        assert path in html


def test_hub_groups_tools_into_researcher_workflows():
    html = explore_hub.render_html()

    for heading in ("Look at images", "Make new images", "Understand the search", "Preference model"):
        assert f"<h2>{heading}</h2>" in html


def test_explore_has_details_how_search_round_works_instead_of_workflow_stepper():
    html = explore_hub.render_html()

    assert 'id="workflowStepper"' not in html
    assert 'class="workflow-stage"' not in html
    assert "workflow-card" not in html
    assert "<details" in html
    assert "How a search round works" in html


def test_render_html_uses_sulfur_proof_shell():
    """Task 5 render contract: the page sits on the Sulfur Proof foundation, includes the
    shared header's context-switcher script, ships a semantic <header>, and has no
    prefers-color-scheme: dark branch (Sulfur Proof is the only theme)."""
    html = explore_hub.render_html()
    assert "--paper:#C3C5BA" in html
    assert "shared-ui.js" in html
    assert "<header" in html
    assert "prefers-color-scheme: dark" not in html


def test_render_html_wires_the_shared_topnav_header():
    """Task 5 brief: 'Explore receives only the shell here; the active-desk composition
    belongs to the navigation plan.' The page's only job in this dispatch is to wire in the
    shared shell: import nav_bar_html, call it with 'explore.html' (the canonical name for
    this hub), and embed the Sulfur Proof style constants instead of DARK_TOKENS/BTN_CSS.
    The render_html() signature accepts the same (active_expedition, active_leg, running)
    kwargs as every other migrated page so a future curation_server.py route update can
    pass the active research context through without a second signature change. The
    legacy rounded `.tool { background:var(--panel); border:1px solid var(--border);
    border-radius:10px; ... }` filled card is gone; the new `.tool` card is a bounded
    working piece (a tool card with title + description + link) and carries a CONTROL_CSS
    depth class so it reads as a mounted piece, not as a flat rounded card. The legacy
    system-font stack on the body override is gone (Sulfur's body font comes from
    SULFUR_CSS, which uses the bundled IBM Plex Sans)."""
    html = explore_hub.render_html()

    # The legacy rounded `.tool` filled card is gone.
    assert ".tool { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px;" not in html
    # The new `.tool` card carries a CONTROL_CSS depth class.
    assert "raised-readout" in html or "raised-control" in html
    # The legacy system-font stack on the body override is gone.
    assert "font-family: -apple-system,sans-serif" not in html


def test_render_html_preserves_a_compact_ruled_full_tool_index():
    html = explore_hub.render_html()

    for heading in ("Look at images", "Make new images", "Understand the search", "Preference model"):
        assert f"<h2>{heading}</h2>" in html

    assert 'class="tool-index"' in html
    assert "border-bottom:1px solid var(--rule)" in html

    for path, name, desc in explore_hub.TOOLS:
        name_pos = html.find(f'<span class="name">{name}</span>')
        assert name_pos != -1, f"missing .name for {path}"
        a_open = html.rfind("<a ", 0, name_pos)
        assert a_open != -1, f"no <a> wrapping .name for {path}"
        a_close = html.find(">", a_open)
        a_tag = html[a_open:a_close + 1]
        assert f'href="/{path}"' in a_tag, f"link wrapping {path}'s .name has wrong href: {a_tag!r}"
        assert f'<span class="desc">{desc}</span>' in html

    # The page includes the "How a search round works" onboarding details.
    assert "How a search round works" in html


def test_render_html_accepts_optional_active_leg_kwargs():
    """The page's render_html() signature accepts the same (active_expedition, active_leg,
    running) kwargs as every other migrated page so a future curation_server.py route
    update can pass the active research context through without a second signature change.
    With no kwargs, the page still renders (the existing curation_server.py route calls
    explore_hub.render_html() with no args); with the kwargs, it still renders and embeds
    the same nav-bar header."""
    html_no_args = explore_hub.render_html()
    assert "<header" in html_no_args

    html_with_args = explore_hub.render_html(
        active_expedition="demo", active_leg="cockpit",
        running=("demo", "round1"),
    )
    assert "<header" in html_with_args
    assert 'data-expedition="demo"' in html_with_args
    assert 'data-leg="cockpit"' in html_with_args


def test_next_decision_handles_each_focus_readiness_state():
    focus = {
        "focus_id": "focus_11111111111111111111111111111111",
        "scope": {"expedition": "demo", "leg": "round1"},
        "test_contract": {
            "intention": "test",
            "evidence_scope": "members",
            "changed_variable": "subject",
            "held_constant": ["style"],
            "expected_move": "up",
            "evidence_against": "down",
        },
    }

    cases = [
        (None, [], "Orient", "Choose a Focus"),
        ({**focus, "test_contract": None}, [], "Explain", "Edit Focus"),
        (focus, [], "Act", "Draft a trial"),
        (focus, [{"id": "trial-1", "status": "running"}], "Act", "Review trial"),
        (focus, [{"id": "trial-1", "status": "completed"}], "Learn", "Evaluate results"),
        (
            focus,
            [{"id": "trial-1", "status": "completed", "evaluated": True}],
            "Learn",
            "Revise Focus",
        ),
    ]

    for current_focus, trials, stage, label in cases:
        decision = explore_hub.derive_next_decision(current_focus, trials=trials)
        assert decision["stage"] == stage
        assert decision["label"] == label
        assert decision["href"].startswith("/")


def test_build_explore_data_keeps_scope_and_missing_evidence():
    focus = {
        "focus_id": "focus_11111111111111111111111111111111",
        "label": "Ink anchor",
        "revision": 2,
        "question": "Can the ink family travel?",
        "observation": "The real anchor holds the cluster together.",
        "source": {
            "member_tags": ["generated-a", "missing-generated"],
            "real_anchor_tags": ["real-anchor"],
        },
        "created_at": "2026-07-16T00:00:00Z",
        "updated_at": "2026-07-16T01:00:00Z",
        "test_contract": None,
    }
    context = WorkspaceContext("demo", "round1", focus)
    data = explore_hub.build_explore_data(
        context,
        [focus],
        trials=[{"id": "trial-1", "created_at": "2026-07-16T02:00:00Z"}],
    )

    assert data["focus"]["focus_id"] == focus["focus_id"]
    assert data["saved_observations"] == ["The real anchor holds the cluster together."]
    assert data["evidence"][-1] == {
        "tag": "real-anchor", "role": "real_anchor", "missing": True
    }
    assert data["evidence"][1] == {
        "tag": "missing-generated", "role": "generated_member", "missing": True
    }
    assert data["activity"][-1]["record_id"] == "trial-1"


def test_build_explore_data_lists_open_foci_without_selecting_one():
    foci = [
        {"focus_id": "focus_11111111111111111111111111111111", "status": "open"},
        {"focus_id": "focus_22222222222222222222222222222222", "status": "archived"},
    ]
    data = explore_hub.build_explore_data(WorkspaceContext("demo", "round1"), foci)

    assert data["focus"] is None
    assert [focus["focus_id"] for focus in data["open_foci"]] == [
        "focus_11111111111111111111111111111111"
    ]


def test_build_explore_data_uses_enriched_selected_focus_evidence():
    focus_id = "focus_11111111111111111111111111111111"
    raw_focus = {
        "focus_id": focus_id,
        "scope": {"expedition": "demo", "leg": "round1"},
        "source": {"member_tags": ["present"], "real_anchor_tags": ["anchor"]},
    }
    enriched_focus = {
        **raw_focus,
        "evidence": {
            "generated_members": [{"tag": "present"}],
            "real_anchors": [{"tag": "anchor"}],
        },
    }

    data = explore_hub.build_explore_data(
        WorkspaceContext("demo", "round1", raw_focus), [enriched_focus]
    )

    assert data["focus"] is enriched_focus
    assert data["evidence"] == [
        {"tag": "present", "role": "generated_member", "missing": False},
        {"tag": "anchor", "role": "real_anchor", "missing": False},
    ]


def test_focus_desk_makes_tabs_and_next_decision_actionable():
    focus = {
        "focus_id": "focus_11111111111111111111111111111111",
        "scope": {"expedition": "demo", "leg": "round1"},
        "test_contract": None,
    }
    html = explore_hub.render_html(
        context=WorkspaceContext("demo", "round1", focus),
        data=explore_hub.build_explore_data(
            WorkspaceContext("demo", "round1", focus), [focus]
        ),
    )

    assert 'data-tab="focus"' in html
    assert 'data-tab="observations"' in html
    assert "tabPanel" in html
    assert 'href="/explore.html?expedition=demo&amp;leg=round1&amp;focus_id=' in html


def test_focus_tool_index_preserves_explicit_context():
    focus = {
        "focus_id": "focus_11111111111111111111111111111111",
        "scope": {"expedition": "demo", "leg": "round1"},
        "test_contract": None,
    }
    context = WorkspaceContext("demo", "round1", focus)
    html = explore_hub.render_html(context=context, data=explore_hub.build_explore_data(context, [focus]))

    assert 'href="/cockpit.html?expedition=demo&amp;leg=round1&amp;focus_id=' in html
