import json
import math

from clawmarks.build import coverage_map


def test_compute_data_reads_manifest(tmp_path):
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = coverage_map.compute_data(str(tmp_path))
    assert data is not None
    html = coverage_map.render_html(data)
    assert "<html>" in html.lower() or "<!doctype" in html.lower()
    assert "Target this gap in cockpit" in html


def test_render_html_explains_dinov2_and_density_scale():
    html = coverage_map.render_html({"cells": [], "max_count": 3})
    assert "DINOv2 is an open vision model" in html
    assert "quantile bins" in html
    assert "median occupied-cell count" in html
    assert "median ${MEDIAN_COUNT}" in html
    assert "max ${MAX_COUNT}" in html


def _cell(fb, nb, count, frontier=False, items=None):
    return {
        "fb": fb, "nb": nb, "count": count, "frontier": frontier,
        "faith_lo": round(fb * 0.1, 3), "faith_hi": round((fb + 1) * 0.1, 3),
        "novelty_lo": round(nb * 0.1, 3), "novelty_hi": round((nb + 1) * 0.1, 3),
        "thumb": f"thumbs/best_{fb}_{nb}.jpg" if count else None,
        "best_tag": f"best_{fb}_{nb}" if count else None,
        "items": items if items is not None else (
            [{"tag": f"best_{fb}_{nb}", "faith": 0.5, "novelty": 0.5, "thumb": "t.jpg", "prompt_name": "p"}]
            if count else []
        ),
    }


def test_top_frontier_cells_picks_densest_adjacent():
    # (1,0) is a frontier cell adjacent only to a count=3 cell.
    # (3,0) is a frontier cell adjacent to a count=10 cell and a count=4 cell (denser).
    cells = [
        _cell(0, 0, count=3), _cell(1, 0, count=0, frontier=True), _cell(2, 0, count=0),
        _cell(3, 0, count=0, frontier=True), _cell(4, 0, count=10), _cell(3, 1, count=4),
    ]
    data = {"cells": cells, "max_count": 10}
    top = coverage_map.top_frontier_cells(data, n=2)
    assert [c["fb"] for c in top] == [3, 1]
    assert top[0]["adjacent"] == 14
    assert top[1]["adjacent"] == 3


def test_top_frontier_cells_shapes_cards_with_neighbor_exemplar():
    cells = [
        _cell(0, 0, count=0, frontier=True),
        _cell(1, 0, count=5, items=[{"tag": "x", "faith": 0.61, "novelty": 0.42, "thumb": "t.jpg", "prompt_name": "p"}]),
    ]
    data = {"cells": cells, "max_count": 5}
    top = coverage_map.top_frontier_cells(data, n=3)
    assert len(top) == 1
    card = top[0]
    assert card["adjacent"] == 5
    assert card["thumb"] == "t.jpg"
    assert card["near_faith"] == 0.61
    assert card["near_novelty"] == 0.42
    assert "Faith" in card["range"] and "novelty" in card["range"]


def test_top_frontier_cells_picks_highest_novelty_item_across_all_neighbors():
    # (1,0) has 10 low-novelty images; (2,0) has 1 high-novelty image. The exemplar should be
    # the single highest-novelty item across every neighbor, not just the item from whichever
    # neighbor happens to have the most images.
    cells = [
        _cell(1, 0, count=0, frontier=True),
        _cell(0, 0, count=10, items=[
            {"tag": "dense", "faith": 0.5, "novelty": 0.1, "thumb": "dense.jpg", "prompt_name": "p"},
        ]),
        _cell(2, 0, count=1, items=[
            {"tag": "sparse", "faith": 0.7, "novelty": 0.9, "thumb": "sparse.jpg", "prompt_name": "p"},
        ]),
    ]
    data = {"cells": cells, "max_count": 10}
    top = coverage_map.top_frontier_cells(data, n=1)
    assert len(top) == 1
    assert top[0]["adjacent"] == 11
    assert top[0]["thumb"] == "sparse.jpg"
    assert top[0]["near_novelty"] == 0.9


def test_top_frontier_cells_skips_frontier_cells_with_no_populated_neighbor():
    # Shouldn't happen given how `frontier` is computed upstream, but defend against it anyway.
    cells = [_cell(0, 0, count=0, frontier=True)]
    data = {"cells": cells, "max_count": 0}
    assert coverage_map.top_frontier_cells(data, n=3) == []


def test_top_frontier_cells_respects_n():
    cells = [
        _cell(0, 0, count=2), _cell(1, 0, count=0, frontier=True),
        _cell(2, 0, count=2), _cell(1, 1, count=0, frontier=True),
        _cell(1, 2, count=2),
    ]
    data = {"cells": cells, "max_count": 2}
    assert len(coverage_map.top_frontier_cells(data, n=1)) == 1


def test_render_html_never_emits_a_literal_closing_script_tag():
    """A literal "</script>" substring anywhere before the real closing tag truncates the
    browser's HTML parse of the whole <script> block early -- everything after it is dropped
    silently, with no console error. This bit six pages via a copy-pasted comment; guard
    against it coming back."""
    data = {"cells": [], "max_count": 0}
    html = coverage_map.render_html(data)
    script_start = html.index("<script>")
    script_end = html.index("</script>", script_start + len("<script>"))
    body = html[script_start + len("<script>"):script_end]
    assert "</script" not in body


def test_render_html_uses_sulfur_proof_shell():
    """Task 4 render contract: the page sits on the Sulfur Proof foundation, includes the
    shared header's context-switcher script, ships a semantic <header>, and has no
    prefers-color-scheme: dark branch (Sulfur Proof is the only theme)."""
    html = coverage_map.render_html({"cells": [], "max_count": 0})
    assert "--paper:#C3C5BA" in html
    assert "shared-ui.js" in html
    assert "<header" in html
    assert "prefers-color-scheme: dark" not in html


def test_render_html_labels_coverage_frontier():
    """Task 4 brief, Step 1: the cell grid is the coverage frontier visualization, so the
    element grouping frontier cells carries aria-label="Coverage frontier" so screen readers
    hear what the canvas shows."""
    html = coverage_map.render_html({"cells": [], "max_count": 0})
    assert 'aria-label="Coverage frontier"' in html


def test_outer_bins_use_declared_metric_domains(tmp_path):
    (tmp_path / "scored_manifest.json").write_text(json.dumps([
        {"tag": "a", "centroid_sim": 0.2, "novelty": 0.7,
         "prompt_name": "p", "file": str(tmp_path / "a.png")}
    ]))
    data = coverage_map.compute_data(str(tmp_path))
    assert min(c["faith_lo"] for c in data["cells"]) == -1.0
    assert max(c["faith_hi"] for c in data["cells"]) == 1.0
    assert min(c["novelty_lo"] for c in data["cells"]) == 0.0
    assert max(c["novelty_hi"] for c in data["cells"]) == 2.0


def test_empty_manifest_keeps_coverage_cells_finite(tmp_path):
    (tmp_path / "scored_manifest.json").write_text("[]")
    data = coverage_map.compute_data(str(tmp_path))
    assert len(data["cells"]) == coverage_map.N_BINS ** 2
    assert all(c["faith_lo"] is not None and c["faith_hi"] is not None for c in data["cells"])
    assert all(c["novelty_lo"] is not None and c["novelty_hi"] is not None for c in data["cells"])


def test_repeated_quantile_edges_never_expose_zero_width_frontier(tmp_path):
    (tmp_path / "scored_manifest.json").write_text(json.dumps([
        {"tag": "a", "centroid_sim": 0.2, "novelty": 0.7,
         "prompt_name": "p", "file": str(tmp_path / "a.png")}
    ]))
    data = coverage_map.compute_data(str(tmp_path))
    assert not [cell for cell in data["cells"] if cell["frontier"]]


def test_every_frontier_has_finite_strict_ranges(tmp_path):
    manifest = []
    for index in range(16):
        manifest.append({
            "tag": f"a{index}", "centroid_sim": 0.05 + index * 0.05,
            "novelty": 0.1 + index * 0.1, "prompt_name": "p",
            "file": str(tmp_path / f"a{index}.png"),
        })
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = coverage_map.compute_data(str(tmp_path))
    frontiers = [cell for cell in data["cells"] if cell["frontier"]]
    assert frontiers
    for cell in frontiers:
        assert all(math.isfinite(cell[key]) for key in (
            "faith_lo", "faith_hi", "novelty_lo", "novelty_hi"
        ))
        assert cell["faith_lo"] < cell["faith_hi"]
        assert cell["novelty_lo"] < cell["novelty_hi"]
