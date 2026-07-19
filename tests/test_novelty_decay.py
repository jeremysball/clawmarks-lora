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


def test_sparkline_tooltip_labels_image_count_plainly(tmp_path):
    manifest = [
        {"tag": "gen0_a", "prompt_name": "p", "novelty": 0.5},
        {"tag": "gen1_a", "prompt_name": "p", "novelty": 0.4},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    html = novelty_decay.render_html(novelty_decay.compute_data(str(tmp_path)))
    assert "images=${points[i].n}" in html
    assert "(n=${points[i].n})" not in html


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


def test_render_html_uses_sulfur_proof_shell(tmp_path):
    """Task 4 render contract: the page sits on the Sulfur Proof foundation, includes the
    shared header's context-switcher script, ships a semantic <header>, and has no
    prefers-color-scheme: dark branch (Sulfur Proof is the only theme)."""
    manifest = [
        {"tag": "gen0_a", "prompt_name": "p", "novelty": 0.5},
        {"tag": "gen1_a", "prompt_name": "p", "novelty": 0.4},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    html = novelty_decay.render_html(novelty_decay.compute_data(str(tmp_path)))
    assert "--paper:#C3C5BA" in html
    assert "shared-ui.js" in html
    assert "<header" in html
    assert "prefers-color-scheme: dark" not in html


def test_render_html_rows_are_ruled_no_stat_card_grid(tmp_path):
    """Task 4 brief, Step 3 (Novelty Decay): the per-prompt rows are flat ruled evidence rows,
    not a grid of rounded bordered stat cards. Each row uses `border-bottom:1px solid var(--rule)`
    to separate from the next, no card border, no border-radius on .row. The summary line
    (counts, percentages, scores) is also flat prose, not a stat-card grid. The trend tag is a
    small inline colored tag, not a stat card."""
    manifest = [
        {"tag": "gen0_a", "prompt_name": "p", "novelty": 0.5},
        {"tag": "gen1_a", "prompt_name": "p", "novelty": 0.4},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    html = novelty_decay.render_html(novelty_decay.compute_data(str(tmp_path)))

    # The page-local CSS lives in the <style> block before the first .infobtn rule, which
    # is the start of the shared INFOTIP_CSS.
    infobtn_start = html.index(".infobtn")
    page_local = html[:infobtn_start]

    assert ".row" in page_local
    # New ruled-row treatment: each row has a `border-bottom:1px solid var(--rule)`.
    assert "border-bottom:1px solid var(--rule)" in page_local
    # No border-radius in the page-local CSS (only the shared INFOTIP_CSS carve-out may have it).
    assert "border-radius" not in page_local
    # The legacy `--panel` filled-card background on .row is gone; the row is flat on paper.
    assert "background:var(--panel)" not in page_local
    # The legacy hex-coded dark theme color is gone.
    assert "color:#eaeaee" not in page_local
    assert "color:#9a9aa4" not in page_local
