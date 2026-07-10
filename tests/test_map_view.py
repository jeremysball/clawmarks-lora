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
