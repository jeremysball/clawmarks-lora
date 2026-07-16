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
              "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28, "sampler": "ddim", "negative": "n",
              "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert "view-transition-name" in html
