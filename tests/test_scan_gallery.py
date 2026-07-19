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


def test_sortable_generation_orders_round_2_after_all_of_round_1():
    """Round 2 restarts its own generation numbering at 0, so a naive generation-only sort put
    early round-2 images in the middle of round 1's timeline instead of after all of it. The
    combined sort key must put every round-2 tag after every round-1 tag, regardless of
    generation number within each round."""
    assert scan_gallery.sortable_generation("r2_gen0_x") > scan_gallery.sortable_generation("gen99_x")
    assert scan_gallery.sortable_generation("gen3_x") < scan_gallery.sortable_generation("r2_gen3_x")


def test_compute_data_keeps_display_generation_separate_from_sort_key(tmp_path):
    manifest = [
        {"file": "/x/r2_gen3_a.png", "tag": "r2_gen3_a", "category": "r2_explore",
         "prompt_name": "fox", "prompt_type": "conflict", "prompt": "p", "strength": 1.0,
         "cfg": 5.0, "seed": 1, "steps": 28, "sampler": "ddim", "negative": "n",
         "centroid_sim": 0.5, "novelty": 0.5},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))

    item = scan_gallery.compute_data(str(tmp_path), {})[0]

    assert item["gen"] == 3
    assert item["sort_gen"] == scan_gallery.sortable_generation("r2_gen3_a")


def test_render_html_reapplies_url_filters_after_favorites_load():
    html = scan_gallery.render_html([])

    assert "}).catch(() => {}).then(applyFilters);" in html


def test_render_html_embeds_data_and_infobtn_tips():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0, "category": "seedrun1",
              "prompt_name": "fox", "prompt_type": "conflict", "prompt": "p", "strength": 1.0,
              "cfg": 5.0, "seed": 1, "steps": 28, "sampler": "ddim", "negative": "n",
              "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert '"tag": "a"' in html
    assert "infobtn" in html
    assert 'id="topnav"' in html
    assert 'More tools' not in html


def test_render_html_escapes_close_script_in_model_generated_prompt_text():
    """Regression test for issue #17 (stored XSS): a prompt/tag field containing a literal
    "</script>" used to close the page's own <script> tag early, letting whatever followed
    execute as HTML/JS. render_html must not leak that sequence into the page."""
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0, "category": "seedrun1",
              "prompt_name": "fox", "prompt_type": "conflict",
              "prompt": "a cat </script><script>alert(1)</script>", "strength": 1.0,
              "cfg": 5.0, "seed": 1, "steps": 28, "sampler": "ddim", "negative": "n",
              "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert "</script><script>alert(1)" not in html


def test_render_html_includes_view_transition_helper():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0, "category": "seedrun1",
              "prompt_name": "fox", "prompt_type": "conflict", "prompt": "p", "strength": 1.0,
              "cfg": 5.0, "seed": 1, "steps": 28, "sampler": "ddim", "negative": "n",
              "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert "function withViewTransition(fn)" in html
    assert "document.startViewTransition" in html
    assert "withViewTransition(render)" in html


def test_thumb_html_has_sanitized_view_transition_name():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "gen3_r2/exploit#1", "gen": 3,
              "category": "seedrun1", "prompt_name": "fox", "prompt_type": "conflict", "prompt": "p",
              "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28, "sampler": "ddim",
              "negative": "n", "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert "view-transition-name" in html


def test_render_html_uses_sulfur_proof_shell():
    """Task 4 render contract: the page sits on the Sulfur Proof foundation, includes the
    shared header's context-switcher script, ships a semantic <header>, and has no
    prefers-color-scheme: dark branch (Sulfur Proof is the only theme)."""
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0,
              "category": "seedrun1", "prompt_name": "fox", "prompt_type": "conflict",
              "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28,
              "sampler": "ddim", "negative": "n", "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert "--paper:#C3C5BA" in html
    assert "shared-ui.js" in html
    assert "<header" in html
    assert "prefers-color-scheme: dark" not in html


def test_render_html_includes_homepage_orientation_heading():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0,
              "category": "seedrun1", "prompt_name": "fox", "prompt_type": "conflict",
              "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28,
              "sampler": "ddim", "negative": "n", "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert "Browse and curate AI-generated artwork from this LoRA search." in html


def test_render_html_grid_cells_use_mounted_evidence_depth():
    """Task 4 brief, Step 3 (Scan): thumbnail grid cells are mounted evidence on the paper
    background, not rounded cards. The page-local grid CSS therefore uses
    .mounted-evidence (or .raised-readout) and contains no border-radius in the page-local
    selectors (the shared INFOTIP_CSS carve-out from the brief still applies)."""
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0,
              "category": "seedrun1", "prompt_name": "fox", "prompt_type": "conflict",
              "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28,
              "sampler": "ddim", "negative": "n", "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    # The grid-cell selector must reference the Task 2 depth class. Pick one of the two
    # approved treatments; either satisfies the brief.
    assert (".thumb.mounted-evidence" in html) or (".thumb.raised-readout" in html)


# ---------------------------------------------------------------------------
# Task 3: plain-language glossary labels replace abbreviations
# ---------------------------------------------------------------------------


def test_render_html_uses_plain_language_labels_not_faith_abbreviation():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0,
              "category": "seedrun1", "prompt_name": "fox", "prompt_type": "conflict",
              "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28,
              "sampler": "ddim", "negative": "n", "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert "Faith &gt;=" not in html
    assert "Faith &lt;=" not in html


def test_render_html_thumbnails_no_longer_contain_f_and_n_prefixes():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0,
              "category": "seedrun1", "prompt_name": "fox", "prompt_type": "conflict",
              "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28,
              "sampler": "ddim", "negative": "n", "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert '<div class="meta">' in html
    # Thumbnail overlay should contain only the prompt name, no f=/n= prefix
    meta_idx = html.index('<div class="meta">')
    meta_end = html.index('</div>', meta_idx)
    meta_content = html[meta_idx:meta_end]
    assert 'f=' not in meta_content
    assert 'n=' not in meta_content


def test_render_html_sort_labels_use_plain_language():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0,
              "category": "seedrun1", "prompt_name": "fox", "prompt_type": "conflict",
              "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28,
              "sampler": "ddim", "negative": "n", "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert "How new or different (high to low)" in html
    # Sort labels now use the glossary plain labels, not bare field names
    assert "Faithfulness (high to low)" not in html


def test_render_html_uses_similarity_to_real_art_for_range_labels():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0,
              "category": "seedrun1", "prompt_name": "fox", "prompt_type": "conflict",
              "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28,
              "sampler": "ddim", "negative": "n", "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert "Similarity to real art &gt;=" in html
    assert "Similarity to real art &lt;=" in html


def test_render_html_infobtn_calls_use_glossary_keys():
    """Scan's info_btn calls for faithfulness/novelty/map_elites should be keyed, not raw-text."""
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0,
              "category": "seedrun1", "prompt_name": "fox", "prompt_type": "conflict",
              "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28,
              "sampler": "ddim", "negative": "n", "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert 'info_btn("faithfulness")' not in html  # Python-side, not in rendered HTML
    # The rendered button for faithfulness looks up the glossary, so should not contain the raw
    # definition text in its data-tip:
    assert 'aria-label="More information about Similarity to real art"' in html
    assert 'aria-label="More information about How new or different"' in html
