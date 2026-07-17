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
