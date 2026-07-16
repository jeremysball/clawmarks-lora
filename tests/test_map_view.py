from clawmarks.build import map_view


def test_compute_data_reads_from_deps_not_disk(tmp_path):
    deps = {"solution-map": {
        "solution_map_data": {
            "points": [
                {"tag": "a", "x": 0.1, "y": 0.2, "gen": 0, "prompt_name": "p",
                 "prompt_type": "conflict", "faith": 0.5, "novelty": 0.5, "category": "seedrun1",
                 "thumb": "thumbs/a.jpg", "nearest_real": "r0", "nearest_real_sim": 0.9},
            ],
            "real_points": [{"x": 0.0, "y": 0.0}],
        },
        "similarity_scored": {},
    }}
    data = map_view.compute_data(str(tmp_path), deps)
    assert len(data["points"]) == 1
    assert len(data["real_points"]) == 1
    assert data["max_gen"] == 0


def test_render_html_embeds_points():
    data = {
        "points": [{"tag": "a", "x": 0.1, "y": 0.2, "gen": 0, "prompt_name": "p",
                    "prompt_type": "conflict", "faith": 0.5, "novelty": 0.5, "category": "seedrun1",
                    "thumb": "thumbs/a.jpg", "nearest_real": "r0", "nearest_real_sim": 0.9}],
        "real_points": [{"x": 0.0, "y": 0.0}],
        "max_gen": 0,
        "real_anchor_counts": [("r0", 1)],
    }
    html = map_view.render_html(data)
    assert '"tag": "a"' in html


def test_render_html_wires_real_image_hover():
    data = {
        "points": [{"tag": "a", "x": 0.1, "y": 0.2, "gen": 0, "prompt_name": "p",
                    "prompt_type": "conflict", "faith": 0.5, "novelty": 0.5, "category": "seedrun1",
                    "thumb": "thumbs/a.jpg", "nearest_real": "r0", "nearest_real_sim": 0.9}],
        "real_points": [{"x": 0.0, "y": 0.0}],
        "max_gen": 0,
        "real_anchor_counts": [("r0", 1)],
    }
    html = map_view.render_html(data)
    assert 'id="realImg"' in html
    assert "'/real/' + encodeURIComponent(p.nearest_real)" in html


def test_render_html_explains_dinov2_and_calibrates_style_match():
    data = {
        "points": [
            {"tag": "a", "x": 0.1, "y": 0.2, "gen": 0, "prompt_name": "p",
             "prompt_type": "conflict", "faith": 0.5, "novelty": 0.5, "category": "seedrun1",
             "thumb": "thumbs/a.jpg", "nearest_real": "r0", "nearest_real_sim": 0.9},
        ],
        "real_points": [{"x": 0.0, "y": 0.0}],
        "max_gen": 0,
        "real_anchor_counts": [("r0", 1)],
    }
    html = map_view.render_html(data)
    assert "DINOv2 is an open vision model" in html
    assert "style match to your real art's average" in html
    assert "range 0.50-0.50" in html
    assert "closest single training photo" in html
    assert "median" in html
    assert "real training photo" in html
    assert "gold dot = picked winner" in html
    assert "Play the generation history" in html


def test_render_html_never_emits_a_literal_closing_script_tag():
    """A literal "</script>" substring anywhere before the real closing tag truncates the
    browser's HTML parse of the whole <script> block early -- everything after it is dropped
    silently, with no console error. This bit six pages via a copy-pasted comment; guard
    against it coming back."""
    data = {"points": [], "real_points": [], "max_gen": 0, "real_anchor_counts": []}
    html = map_view.render_html(data)
    script_start = html.index("<script>")
    script_end = html.index("</script>", script_start + len("<script>"))
    body = html[script_start + len("<script>"):script_end]
    assert "</script" not in body
