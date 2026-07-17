import json

from clawmarks.build import redundancy_view


def test_compute_data_uses_similarity_scored_from_deps(tmp_path):
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    deps = {"solution-map": {"similarity_scored": {"a": [["a", 0.9]]}, "solution_map_data": {}}}

    data = redundancy_view.compute_data(str(tmp_path), deps)
    assert data["sim_scored"] == {"a": [["a", 0.9]]}
    assert data["thumbs"]["a"] == "a.png"
    assert data["meta"]["a"]["prompt_name"] == "p"


def test_render_html_embeds_edges():
    data = {
        "sim_scored": {"a": [["a", 0.9]]},
        "thumbs": {"a": "a.png"},
        "meta": {"a": {"prompt_name": "p", "novelty": 0.5, "faith": 0.5}},
    }
    html = redundancy_view.render_html(data)
    assert '"a": [["a", 0.9]]' in html


def _thresh_attrs(html):
    import re
    m = re.search(r'id="thresh"[^>]*min="([0-9.]+)"[^>]*max="([0-9.]+)"[^>]*value="([0-9.]+)"', html)
    return tuple(float(x) for x in m.groups())


def test_slider_range_tracks_a_diverse_datasets_edge_distribution():
    # A diverse population's closest pair can sit below the old hardcoded 0.80 slider minimum, which
    # left every slider position empty and the page looking broken. The slider must reach the data:
    # its minimum sits at or below the smallest edge and its default at or below the largest, so at
    # least one edge always survives the default threshold.
    edges = {f"n{i}": [[f"n{i+1}", 0.20 + 0.005 * i]] for i in range(60)}  # all edges in [0.20, 0.50)
    data = {"sim_scored": edges, "thumbs": {}, "meta": {}}
    lo, hi, default = _thresh_attrs(redundancy_view.render_html(data))
    all_scores = [s for lst in edges.values() for _, s in lst]
    assert lo <= min(all_scores)
    assert default <= max(all_scores)
    assert lo <= default <= hi


def test_slider_falls_back_to_a_valid_range_with_no_edges():
    lo, hi, default = _thresh_attrs(redundancy_view.render_html(
        {"sim_scored": {}, "thumbs": {}, "meta": {}}))
    assert lo < hi
    assert lo <= default <= hi


def test_render_html_explains_dinov2_and_similarity_threshold_scale():
    data = {
        "sim_scored": {"a": [["b", 0.71]], "b": [["a", 0.98]]},
        "thumbs": {},
        "meta": {},
    }
    html = redundancy_view.render_html(data)
    assert "DINOv2 is an open vision model" in html
    assert "image-to-image match threshold" in html
    assert "tightest 5% of pairs this sweep" in html
    assert "your pairs span 0.71-0.98" in html
    assert "(highest novelty)" in html


def test_render_html_never_emits_a_literal_closing_script_tag():
    """A literal "</script>" substring anywhere before the real closing tag truncates the
    browser's HTML parse of the whole <script> block early -- everything after it is dropped
    silently, with no console error. This bit six pages via a copy-pasted comment; guard
    against it coming back."""
    data = {"sim_scored": {}, "thumbs": {}, "meta": {}}
    html = redundancy_view.render_html(data)
    script_start = html.index("<script>")
    script_end = html.index("</script>", script_start + len("<script>"))
    body = html[script_start + len("<script>"):script_end]
    assert "</script" not in body


def test_render_html_uses_sulfur_proof_shell():
    """Task 4 render contract: the page sits on the Sulfur Proof foundation, includes the
    shared header's context-switcher script, ships a semantic <header>, and has no
    prefers-color-scheme: dark branch (Sulfur Proof is the only theme)."""
    data = {"sim_scored": {}, "thumbs": {}, "meta": {}}
    html = redundancy_view.render_html(data)
    assert "--paper:#C3C5BA" in html
    assert "shared-ui.js" in html
    assert "<header" in html
    assert "prefers-color-scheme: dark" not in html


def test_render_html_clusters_are_ruled_rows_not_stat_cards():
    """Task 4 brief, Step 3 (Redundancy): the page shows clusters as ruled evidence rows,
    not a grid of rounded bordered stat cards. Each cluster row is flat (no card border, no
    border-radius on the page-local .cluster rule) and is separated from the next by a
    `border-bottom:1px solid var(--rule)` rule line, exactly like a flat prose/data row in
    the brief's Global Constraint. The shared INFOTIP_CSS carve-out (the infopop has
    border-radius:8px) is allowed by the brief, so we check the page-local portion only."""
    data = {"sim_scored": {}, "thumbs": {}, "meta": {}}
    html = redundancy_view.render_html(data)

    # The page-local CSS lives in the <style> block before the first .infobtn rule, which
    # is the start of the shared INFOTIP_CSS.
    infobtn_start = html.index(".infobtn")
    page_local = html[:infobtn_start]

    assert ".cluster" in page_local
    # New ruled-row treatment: each cluster row has a `border-bottom:1px solid var(--rule)`.
    assert "border-bottom:1px solid var(--rule)" in page_local
    # No border-radius in the page-local CSS (only the shared INFOTIP_CSS carve-out may have it).
    assert "border-radius" not in page_local
    # The legacy `--panel` filled-card background on .cluster is gone; the row is flat on paper.
    assert "background:var(--panel)" not in page_local
    # No hex-coded border color on the page-local selectors; everything routes through the
    # shared rule/ink tokens.
    assert "border:1px solid var(--border)" not in page_local
