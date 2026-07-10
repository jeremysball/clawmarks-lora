import json

from PIL import Image

from clawmarks.build import uncanny_gallery


def _scored_manifest_fixture(tmp_path):
    a_path = tmp_path / "a.png"
    b_path = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color="red").save(a_path)
    Image.new("RGB", (32, 32), color="blue").save(b_path)
    return [
        {"file": str(a_path), "tag": "a", "centroid_sim": 0.6, "novelty": 0.4,
         "prompt_name": "fox_face", "prompt_type": "conflict", "strength": 1.2, "cfg": 5.0,
         "steps": 28, "sampler": "ddim"},
        {"file": str(b_path), "tag": "b", "centroid_sim": 0.3, "novelty": 0.7,
         "prompt_name": "style_ink", "prompt_type": "style", "strength": 1.5, "cfg": 4.0,
         "steps": 28, "sampler": "ddim"},
    ]


def test_compute_data_bins_manifest_without_importing_torch(tmp_path):
    (tmp_path / "scored_manifest.json").write_text(json.dumps(_scored_manifest_fixture(tmp_path)))
    data = uncanny_gallery.compute_data(str(tmp_path))
    assert len(data["manifest"]) == 2
    assert "grid" in data
    assert data["type_summary"]["conflict"][1] == 1


def test_render_html_produces_gallery_markup(tmp_path):
    (tmp_path / "scored_manifest.json").write_text(json.dumps(_scored_manifest_fixture(tmp_path)))
    data = uncanny_gallery.compute_data(str(tmp_path))
    html = uncanny_gallery.render_html(data)
    assert "CLAWMARKS uncanny frontier atlas" in html
    assert "<html>" in html
