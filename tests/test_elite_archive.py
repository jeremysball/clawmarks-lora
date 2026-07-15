# tests/test_elite_archive.py
import json
import re

from clawmarks.build import elite_archive


def test_compute_data_uses_favorited_images_not_user_picks(tmp_path, monkeypatch):
    # Force every image into a single cell, regardless of its faith/novelty values, so the test
    # doesn't depend on how a 2-item manifest happens to quantile-split across N_BINS x N_BINS
    # cells (bin_edges(vals, 1) always returns [], so bin_of always returns 0).
    monkeypatch.setattr(elite_archive, "N_BINS", 1)
    manifest = [
        {"tag": "a", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.9,
         "novelty": 0.1, "strength": 1.0, "cfg": 7.0, "file": "a.png"},
        {"tag": "b", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.9,
         "novelty": 0.9, "strength": 1.0, "cfg": 7.0, "file": "b.png"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    # "a" has lower novelty than "b" but is favorited: it should win the cell despite that,
    # exactly the behavior user_picks.json used to provide before ratings existed.
    (tmp_path / "user_favorites.json").write_text(json.dumps({"a": {"tag": "a", "favorited_at": "t0"}}))
    # a stale user_picks.json should be ignored entirely
    (tmp_path / "user_picks.json").write_text(json.dumps({"b": {"picked_at": "t0"}}))

    data = elite_archive.compute_data(str(tmp_path))
    assert len(data["cells"]) == 1
    assert data["n_human"] == 1

    html = elite_archive.render_html(data)
    match = re.search(r"const CELLS = (\[.+?\]);\nlet picks", html)
    assert match is not None, "could not find 'const CELLS = [...]; let picks' in archive.html"
    cells = json.loads(match.group(1))
    assert len(cells) == 1
    tags_in_cell = {item["tag"] for item in cells[0]["items"]}
    assert tags_in_cell == {"a", "b"}


def test_render_html_uses_panel_token_for_view_all_button():
    html = elite_archive.render_html({"cells": [], "n_human": 0, "faith_bins": [], "novelty_bins": []})

    assert ".cell .viewall { display:block; width:100%; background:var(--panel-2);" in html


def test_render_html_never_emits_a_literal_closing_script_tag():
    """A literal "</script>" substring anywhere before the real closing tag truncates the
    browser's HTML parse of the whole <script> block early -- everything after it (CELLS,
    render(), openModal(), ...) is dropped silently, with no console error, and the leftover
    HTML-shaped text in template literals gets parsed as real (non-functional) DOM instead.
    This bit archive.html and five sibling pages via a copy-pasted comment; guard against it
    coming back."""
    html = elite_archive.render_html({"cells": [], "n_human": 0, "faith_bins": [], "novelty_bins": []})
    script_start = html.index("<script>")
    script_end = html.index("</script>", script_start + len("<script>"))
    body = html[script_start + len("<script>"):script_end]
    assert "</script" not in body
