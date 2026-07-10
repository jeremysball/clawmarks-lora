import json

from clawmarks.build import coverage_map


def test_compute_data_reads_manifest(tmp_path):
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = coverage_map.compute_data(str(tmp_path))
    assert data is not None
    html = coverage_map.render_html(data)
    assert "<html>" in html.lower() or "<!doctype" in html.lower()
