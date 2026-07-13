import json

from clawmarks.build import coverage_map


def test_compute_data_reads_manifest(tmp_path):
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = coverage_map.compute_data(str(tmp_path))
    assert data is not None
    html = coverage_map.render_html(data)
    assert "<html>" in html.lower() or "<!doctype" in html.lower()


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
