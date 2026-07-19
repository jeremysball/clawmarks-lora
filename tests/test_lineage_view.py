import json
import re

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
    assert "Continue this lineage in cockpit" in html


def test_render_html_uses_sulfur_proof_shell(tmp_path):
    """Task 4 render contract: the page sits on the Sulfur Proof foundation, includes the
    shared header's context-switcher script, ships a semantic <header>, and has no
    prefers-color-scheme: dark branch (Sulfur Proof is the only theme). The legacy lineage
    page rendered its own `:root { color-scheme: dark }` branch; after migration both the
    placeholder and the full tree pages inherit the shared SULFUR_CSS (color-scheme: light)."""
    placeholder_manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(placeholder_manifest))
    data = lineage_view.compute_data(str(tmp_path))  # no parent_tag -> placeholder
    html_placeholder = lineage_view.render_html(data)
    assert "--paper:#C3C5BA" in html_placeholder
    assert "shared-ui.js" in html_placeholder
    assert "<header" in html_placeholder
    assert "prefers-color-scheme: dark" not in html_placeholder
    assert "color-scheme: dark" not in html_placeholder

    manifest = [
        {"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5},
        {"file": "/x/b.png", "tag": "b", "prompt_name": "p", "centroid_sim": 0.6, "novelty": 0.4, "parent_tag": "a"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    html_tree = lineage_view.render_html(lineage_view.compute_data(str(tmp_path)))
    assert "--paper:#C3C5BA" in html_tree
    assert "shared-ui.js" in html_tree
    assert "<header" in html_tree
    assert "prefers-color-scheme: dark" not in html_tree
    assert "color-scheme: dark" not in html_tree


def test_render_html_tree_nodes_are_ruled_rows_not_stat_cards(tmp_path):
    """Task 4 brief, Step 3 (Lineage): the tree is built from ruled evidence rows (one
    tree node per row, each separated by a `border-bottom:1px solid var(--rule)` rule line
    in the page-local CSS), not a grid of rounded bordered stat cards. The connecting tree
    indent line on `<ul>` is also a thin rule (var(--rule)), not the legacy `#2a2a30` dark
    surface line."""
    manifest = [
        {"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5},
        {"file": "/x/b.png", "tag": "b", "prompt_name": "p", "centroid_sim": 0.6, "novelty": 0.4, "parent_tag": "a"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    html = lineage_view.render_html(lineage_view.compute_data(str(tmp_path)))

    # The page-local CSS lives in the <style> block before the first .infobtn rule, which
    # is the start of the shared INFOTIP_CSS. The shared INFOTIP_CSS carve-out (the infopop
    # uses #2a2a30 and border-radius:8px) is allowed by the brief, so we check only the
    # page-local portion.
    infobtn_start = html.index(".infobtn")
    page_local = html[:infobtn_start]

    assert ".node" in page_local
    # New ruled-row treatment: each tree node has a `border-bottom:1px solid var(--rule)`.
    assert "border-bottom:1px solid var(--rule)" in page_local
    # The legacy dark surface hex `#2a2a30` on the tree-indent line is gone from page-local CSS.
    assert "#2a2a30" not in page_local
    # The legacy `color-scheme: dark` declaration on a page-local `:root` is gone.
    assert "color-scheme: dark" not in page_local
    # No border-radius on a page-local selector (tree nodes are flat, not pill/card).
    assert "border-radius" not in page_local
    # No legacy dark-theme hex text colors in page-local CSS.
    assert "color:#eaeaee" not in page_local
    assert "color:#9a9aa4" not in page_local


def test_render_html_uses_plain_metric_labels(tmp_path):
    """No user-facing text uses faith=, f=, n= as unexplained labels."""
    manifest = [
        {"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5},
        {"file": "/x/b.png", "tag": "b", "prompt_name": "p", "centroid_sim": 0.6, "novelty": 0.4, "parent_tag": "a"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    html = lineage_view.render_html(lineage_view.compute_data(str(tmp_path)))
    assert "faithfulness={m[" in html or "faithfulness=" in html
    assert re.search(r'(?<!fulness)faith=', html) is None
