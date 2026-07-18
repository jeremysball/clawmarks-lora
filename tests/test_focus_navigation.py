from clawmarks.build import (
    cockpit,
    compare_page,
    coverage_map,
    elite_archive,
    lineage_view,
    map_view,
    novelty_decay,
    preference_rank,
    preference_status,
    redundancy_view,
    runs_page,
    scan_gallery,
    seed_browser,
)
from clawmarks.shared_ui import _LIGHTBOX_JS


FOCUS = {
    "focus_id": "focus_11111111111111111111111111111111",
    "label": "Ink anchor",
    "revision": 3,
}


def test_all_focus_tool_links_preserve_complete_context():
    suffix = "expedition=demo&amp;leg=round1&amp;focus_id=focus_11111111111111111111111111111111"
    rendered_pages = [
        scan_gallery.render_html([], "demo", "round1", focus=FOCUS),
        elite_archive.render_html({"cells": [], "faith_bins": [], "novelty_bins": [], "n_human": 0}, "demo", "round1", focus=FOCUS),
        map_view.render_html({"points": [], "real_points": [], "max_gen": 0, "real_anchor_counts": []}, "demo", "round1", focus=FOCUS),
        coverage_map.render_html({"cells": [], "max_count": 0, "real_anchor_tags": []}, "demo", "round1", focus=FOCUS),
        redundancy_view.render_html({"sim_scored": {}, "thumbs": {}, "meta": {}}, "demo", "round1", focus=FOCUS),
        novelty_decay.render_html({"series": []}, "demo", "round1", focus=FOCUS),
        lineage_view.render_html({"has_lineage": False}, "demo", "round1", focus=FOCUS),
        runs_page.render_html("demo", "round1", focus=FOCUS),
        compare_page.render_html("demo", "round1", focus=FOCUS),
        preference_rank.render_html({"has_model": False, "model_file": "model"}, "demo", "round1", focus=FOCUS),
        preference_status.render_html({
            "model_meta": None, "comparisons_gate_message": "", "comparisons_changed_since_train": False,
            "new_comparisons_since_train": 0, "has_model": False, "use_predicted_preference": False,
            "n_usable": 0, "n_comparisons": 0, "min_comparisons": 50,
        }, "demo", "round1", focus=FOCUS),
        seed_browser.render_html("demo", "round1", focus=FOCUS),
        cockpit.render_html(active_expedition="demo", active_leg="round1", focus=FOCUS),
    ]

    assert all(suffix in page for page in rendered_pages)


def test_lightbox_scopes_script_relative_scan_data_url():
    branch = _LIGHTBOX_JS[_LIGHTBOX_JS.index("function dataUrl"):]
    script_branch = branch[branch.index("if (s.src"):branch.index("return url.toString()", branch.index("if (s.src"))]

    assert "searchParams.set('expedition'" in script_branch
    assert "searchParams.set('leg'" in script_branch
