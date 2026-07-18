import json
import threading
import urllib.request
from http.server import HTTPServer

import pytest

from clawmarks import config, curation_server as cs
from clawmarks.build import coverage_map


@pytest.fixture
def coverage_server(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", tmp_path / "state" / "active_leg.json")
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    monkeypatch.setattr(cs, "REAL_DIR", str(real_dir))
    monkeypatch.setitem(cs._active_selection, "expedition", None)
    monkeypatch.setitem(cs._active_selection, "leg", None)
    cs._create_expedition({"name": "demo", "textures": [], "fallback_subjects": []})
    leg_dir = config.leg_dir("demo", "round1")
    leg_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(16):
        tag = f"t{index}"
        image_path = leg_dir / f"{tag}.png"
        image_path.write_bytes(b"PNG" + tag.encode())
        records.append({
            "tag": tag, "prompt_name": "p", "prompt_type": "style",
            "centroid_sim": 0.1 + index * 0.05, "novelty": 0.1 + index * 0.05,
            "file": str(image_path),
        })
    (leg_dir / "scored_manifest.json").write_text(json.dumps(records))
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=2)


@pytest.fixture
def data():
    cells = []
    for nb in range(8):
        for fb in range(8):
            cells.append({
                "fb": fb, "nb": nb, "count": 0,
                "frontier": fb == 1 and nb == 1,
                "faith_lo": -1.0 + fb * 0.25, "faith_hi": -1.0 + (fb + 1) * 0.25,
                "novelty_lo": nb * 0.25, "novelty_hi": (nb + 1) * 0.25,
                "thumb": None, "best_tag": None, "items": [],
            })
    return {"cells": cells, "median_count": 1, "max_count": 1}


def test_frontier_is_labeled_and_has_accessible_equivalent(data):
    page = coverage_map.render_html(data, active_expedition="demo", active_leg="round1")
    assert 'aria-label="Coverage frontier"' in page
    assert 'aria-label="Coverage values"' in page
    assert "Create Focus" in page
    assert "promising" not in page.lower()
    assert "createCoverageFocus" in page
    assert "if (!c.frontier) div.setAttribute('role', 'gridcell')" in page
    assert "row.tabIndex = 0" in page
    assert "row.addEventListener('keydown'" in page


def test_coverage_frontier_payload_round_trips_over_http(coverage_server):
    leg_dir = config.leg_dir("demo", "round1")
    computed = coverage_map.compute_data(str(leg_dir))
    page = coverage_map.render_html(computed, active_expedition="demo", active_leg="round1")
    for fragment in (
        "score_ranges:",
        "faithfulness: [currentCell.faith_lo, currentCell.faith_hi]",
        "novelty: [currentCell.novelty_lo, currentCell.novelty_hi]",
        "adjacent_member_tags: adjacentTags(currentCell)",
        "real_anchor_tags: anchor ? [anchor] : []",
        "coverage_hint:",
        "row: currentCell.nb",
        "column: currentCell.fb",
        "domains: DATA.metric_domains",
        "binning_version: DATA.binning_version",
    ):
        assert fragment in page
    cell = next(cell for cell in computed["cells"] if cell["frontier"])
    payload = {
        "scope": {"expedition": "demo", "leg": "round1"},
        "label": "Coverage gap",
        "source": {
            "view": "coverage",
            "kind": "coverage_frontier",
            "score_ranges": {
                "faithfulness": [cell["faith_lo"], cell["faith_hi"]],
                "novelty": [cell["novelty_lo"], cell["novelty_hi"]],
            },
            "adjacent_member_tags": sorted(coverage_map.neighbor_tags(
                computed, cell["fb"], cell["nb"]
            )),
            "real_anchor_tags": [],
            "coverage_hint": {
                "row": cell["nb"],
                "column": cell["fb"],
                "domains": computed["metric_domains"],
                "binning_version": computed["binning_version"],
            },
        },
        "question": "What fills this reachable gap?",
        "observation": "",
        "hypothesis_text": "",
        "test_contract": None,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{coverage_server.server_address[1]}/api/foci",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        assert response.status == 201
