import json

from clawmarks.build import scan_gallery


def test_compute_data_builds_items_with_similarity(tmp_path):
    manifest = [
        {"file": f"/x/{tag}.png", "tag": tag, "category": "seedrun1", "prompt_name": "fox",
         "prompt_type": "conflict", "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1,
         "steps": 28, "sampler": "ddim", "negative": "n", "centroid_sim": 0.5, "novelty": 0.5}
        for tag in ("a", "b", "c")
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    deps = {"similarity": {"a": ["b", "c"]}}

    items = scan_gallery.compute_data(str(tmp_path), deps)
    assert items[0]["tag"] == "a"
    assert items[0]["sim"] == ["b", "c"]


def test_render_html_embeds_data_and_infobtn_tips():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0, "category": "seedrun1",
              "prompt_name": "fox", "prompt_type": "conflict", "prompt": "p", "strength": 1.0,
              "cfg": 5.0, "seed": 1, "steps": 28, "sampler": "ddim", "negative": "n",
              "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert '"tag": "a"' in html
    assert "infobtn" in html
