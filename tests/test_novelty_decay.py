import json

from clawmarks.build import novelty_decay


def test_compute_data_builds_series_across_generations(tmp_path):
    manifest = [
        {"tag": "gen0_a", "prompt_name": "p", "novelty": 0.5},
        {"tag": "gen1_a", "prompt_name": "p", "novelty": 0.4},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = novelty_decay.compute_data(str(tmp_path))
    assert len(data["series"]) == 1
    assert data["series"][0]["prompt_name"] == "p"

    html = novelty_decay.render_html(data)
    assert "<!doctype" in html.lower()


def test_render_html_placeholder_when_no_multi_generation_prompt(tmp_path):
    manifest = [{"tag": "gen0_a", "prompt_name": "p", "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = novelty_decay.compute_data(str(tmp_path))
    assert data["series"] == []

    html = novelty_decay.render_html(data)
    assert "placeholder" in html.lower()
    assert "DINOv2 is an open vision model" in html


def test_render_html_defines_novelty_and_dinov2(tmp_path):
    manifest = [
        {"tag": "gen0_a", "prompt_name": "p", "novelty": 0.5},
        {"tag": "gen1_a", "prompt_name": "p", "novelty": 0.4},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    html = novelty_decay.render_html(novelty_decay.compute_data(str(tmp_path)))
    assert "DINOv2 is an open vision model" in html
    assert "Novelty measures how unlike an image is from the images already explored" in html


def test_render_html_never_emits_a_literal_closing_script_tag(tmp_path):
    """A literal "</script>" substring anywhere before the real closing tag truncates the
    browser's HTML parse of the whole <script> block early -- everything after it is dropped
    silently, with no console error. This bit six pages via a copy-pasted comment; guard
    against it coming back."""
    manifest = [
        {"tag": "gen0_a", "prompt_name": "p", "novelty": 0.5},
        {"tag": "gen1_a", "prompt_name": "p", "novelty": 0.4},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = novelty_decay.compute_data(str(tmp_path))
    html = novelty_decay.render_html(data)
    script_start = html.index("<script>")
    script_end = html.index("</script>", script_start + len("<script>"))
    body = html[script_start + len("<script>"):script_end]
    assert "</script" not in body
