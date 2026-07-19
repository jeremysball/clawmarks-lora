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


def test_render_html_uses_sulfur_proof_viewall_button():
    """The view-all-in-this-cell button under each archive cell used to be a `--panel-2`
    filled bar with rounded bottom corners. After the Sulfur Proof migration, the button uses
    the Sulfur paper-deep tone (the dark text on light-paper fill the spec calls for on a
    "patterned mark" / annotation button) and square corners, plus a hard offset shadow so it
    still reads as a real control without the rounded treatment."""
    html = elite_archive.render_html({"cells": [], "n_human": 0, "faith_bins": [], "novelty_bins": []})

    assert ".cell .viewall" in html
    assert "var(--paper-deep)" in html
    # The legacy rounded-bottom-corner treatment (border-radius:0 0 10px 10px) must be gone.
    assert "border-radius:0 0 10px 10px" not in html


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


def test_render_html_uses_sulfur_proof_shell():
    """Task 4 render contract: the page sits on the Sulfur Proof foundation, includes the
    shared header's context-switcher script, ships a semantic <header>, and has no
    prefers-color-scheme: dark branch (Sulfur Proof is the only theme)."""
    html = elite_archive.render_html({"cells": [], "n_human": 0, "faith_bins": [], "novelty_bins": []})
    assert "--paper:#C3C5BA" in html
    assert "shared-ui.js" in html
    assert "<header" in html
    assert "prefers-color-scheme: dark" not in html


def test_render_html_uses_mounted_evidence_for_archive_grid_cells():
    """Task 4 brief, Step 1: the rendered archive.html literally contains the substring
    `mounted-evidence` (the per-cell depth class for bounded working evidence on the
    paper background). The class comes from CONTROL_CSS, not the page itself, but the rendered
    HTML still has to mention it so the styling is applied via the global rule."""
    html = elite_archive.render_html({"cells": [], "n_human": 0, "faith_bins": [], "novelty_bins": []})

    assert "mounted-evidence" in html


def test_render_html_uses_plain_metric_labels():
    """No user-facing text uses faith=, f=, n= as unexplained labels."""
    html = elite_archive.render_html({"cells": [], "n_human": 0, "faith_bins": [], "novelty_bins": []})
    assert "faithfulness=${elite.faith} novelty=${elite.novelty}" in html
    assert "faithfulness=${it.faith} novelty=${it.novelty}" in html
    assert "count=${c.n} in cell" in html
    assert "bin faithfulness ${c.fb + 1}" in html
    assert re.search(r'(?<!fulness)faith=', html) is None
    assert 'f=${' not in html
