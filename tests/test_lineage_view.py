import json

from clawmarks.build import lineage_view


def test_compute_data_placeholder_when_no_parent_tags(tmp_path):
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = lineage_view.compute_data(str(tmp_path))
    assert data["has_lineage"] is False
    html = lineage_view.render_html(data)
    assert "placeholder" in html.lower() or "no" in html.lower()


def test_compute_data_builds_tree_when_parent_tags_exist(tmp_path):
    manifest = [
        {"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5},
        {"file": "/x/b.png", "tag": "b", "prompt_name": "p", "centroid_sim": 0.6, "novelty": 0.4, "parent_tag": "a"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = lineage_view.compute_data(str(tmp_path))
    assert data["has_lineage"] is True
    html = lineage_view.render_html(data)
    assert "a" in html and "b" in html
