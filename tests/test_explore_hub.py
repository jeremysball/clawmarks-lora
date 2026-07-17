from clawmarks.build import explore_hub
from clawmarks.shared_ui import NAV_GROUPS


def test_render_html_lists_every_tool():
    html = explore_hub.render_html()
    for path, label, _desc in explore_hub.TOOLS:
        assert path in html


def test_hub_lists_the_same_tools_as_the_nav_dropdown():
    # The home page and the jump-to dropdown must stay in sync: every navigable tool listed
    # in the dropdown's detailed groups (Generate, Curate, Understand search, Preference
    # model) needs a card here, in the same order. The Explore group in the dropdown is a
    # quick-access subset of those same destinations; rendering it again would double-list
    # every stage page and self-link to "/" (this very hub).
    DETAILED = ("Generate", "Curate", "Understand search", "Preference model")
    nav_tools = [
        href for group, options in NAV_GROUPS if group in DETAILED for href, _ in options
    ]
    hub_tools = [path for path, _, _ in explore_hub.TOOLS]
    assert hub_tools == nav_tools


def test_hub_groups_tools_into_researcher_workflows():
    html = explore_hub.render_html()

    for heading in ("Generate", "Curate", "Understand search", "Preference model"):
        assert f"<h2>{heading}</h2>" in html


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


def test_render_html_preserves_the_tool_card_grid_composition():
    """Task 5 brief: 'a full redesign of Explore's composition is explicitly out of scope for
    this task.' The actual card grid (section headings, .tools grid containers, individual
    .tool card links) must keep its existing structure and order after the migration: same
    group headings (Generate, Curate, Understand search, Preference model), same grid-of-
    cards composition (`.tools` is a grid with auto-fill minmax(260px, 1fr)), same per-tool
    .tool card with a .name and a .desc, same list of tool destinations."""
    html = explore_hub.render_html()

    # Section headings preserved.
    for heading in ("Generate", "Curate", "Understand search", "Preference model"):
        assert f"<h2>{heading}</h2>" in html

    # The .tools grid container is preserved (still a CSS grid, still the same column
    # track width).
    assert 'class="tools"' in html
    assert "grid-template-columns: repeat(auto-fill, minmax(260px, 1fr))" in html

    # Every tool card is still a link, with the same .name and .desc structure. The .tool
    # card may carry an additional CONTROL_CSS depth class (e.g. "raised-readout") on top
    # of "tool"; the test checks for "tool" appearing as a class on the link wrapping each
    # tool's .name to be tolerant of the depth-class addition while still locking in the
    # link structure.
    for path, name, desc in explore_hub.TOOLS:
        # Look for the tool-card link wrapping this specific name+desc. The class on the
        # link must include "tool" (it may also include a depth class like "raised-readout"),
        # and the link's href must point at this tool's path.
        name_pos = html.find(f'<div class="name">{name}</div>')
        assert name_pos != -1, f"missing .name for {path}"
        # The opening <a> tag immediately precedes the <div class="name"> block.
        a_open = html.rfind("<a ", 0, name_pos)
        assert a_open != -1, f"no <a> wrapping .name for {path}"
        a_close = html.find(">", a_open)
        a_tag = html[a_open:a_close + 1]
        assert 'class="tool' in a_tag, f"link wrapping {path}'s .name has no 'tool' class: {a_tag!r}"
        assert f'href="{path}"' in a_tag, f"link wrapping {path}'s .name has wrong href: {a_tag!r}"
        assert f'<div class="desc">{desc}</div>' in html

    # The h1 still carries the page's "How does this search work?" tip link.
    assert "How does this search work?" in html


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
