import json

from clawmarks.build import elite_archive


def test_compute_data_prefers_yes_rated_image_in_cell(tmp_path):
    manifest = [
        {"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.9,
         "prompt_type": "conflict", "strength": 1.0, "cfg": 5.0},
        {"file": "/x/b.png", "tag": "b", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.1,
         "prompt_type": "conflict", "strength": 1.0, "cfg": 5.0},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "user_ratings.json").write_text(json.dumps({"b": {"label": "yes", "rated_at": "x"}}))
    data = elite_archive.compute_data(str(tmp_path))
    html = elite_archive.render_html(data)
    assert '"tag": "b"' in html


def test_compute_data_falls_back_to_novelty_without_ratings(tmp_path):
    manifest = [
        {"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.9,
         "prompt_type": "conflict", "strength": 1.0, "cfg": 5.0},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = elite_archive.compute_data(str(tmp_path))
    assert data["cells"][0]["items"][0]["tag"] == "a"
