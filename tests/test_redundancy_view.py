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
