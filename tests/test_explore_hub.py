from clawmarks.build import explore_hub
from clawmarks.shared_ui import NAV_OPTIONS


def test_render_html_lists_every_tool():
    html = explore_hub.render_html()
    for path, label, _desc in explore_hub.TOOLS:
        assert path in html


def test_hub_lists_the_same_tools_as_the_nav_dropdown():
    # The home page and the jump-to dropdown must stay in sync: every navigable tool (everything
    # in NAV_OPTIONS except explore.html, which is this hub itself) needs a card here, in the
    # same order. This caught compare/preference_rank/preference_status missing from the hub.
    nav_tools = [href for href, _ in NAV_OPTIONS if href != "explore.html"]
    hub_tools = [path for path, _, _ in explore_hub.TOOLS]
    assert hub_tools == nav_tools


def test_hub_groups_tools_into_researcher_workflows():
    html = explore_hub.render_html()

    for heading in ("Generate", "Curate", "Understand search", "Preference model"):
        assert f"<h2>{heading}</h2>" in html
