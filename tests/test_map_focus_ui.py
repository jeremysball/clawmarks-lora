from clawmarks.build import map_view


def test_map_focus_ui_contains_authoritative_payload_and_keyboard_list():
    data = {
        "points": [{"tag": "gen0_a", "x": 0.1, "y": 0.2, "gen": 0, "prompt_name": "p",
                    "prompt_type": "style", "faith": 0.5, "novelty": 0.5, "category": "seedrun1",
                    "thumb": "thumbs/gen0_a.jpg", "nearest_real": "r0", "nearest_real_sim": 0.9}],
        "real_points": [{"name": "r0", "x": 0.0, "y": 0.0}],
        "max_gen": 0,
        "real_anchor_counts": [("r0", 1)],
        "projection_version": "abc123",
    }
    page = map_view.render_html(data, active_expedition="demo", active_leg="round1")
    assert 'id="focusLabel"' in page
    assert 'id="focusQuestion"' in page
    assert 'type="checkbox"' in page
    assert 'id="createMapFocus"' in page
    assert 'disabled' in page
    assert 'member_tags: Array.from(selectedTags)' in page
    assert "real_anchor_tags" in page
    assert "projection_version: DATA.projection_version" in page
    assert "context_url('/explore.html', created_context)" in page
    assert "lassoMoved" in page
    assert "selectedTags.has(p.tag)" in page
