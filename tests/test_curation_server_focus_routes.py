import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from clawmarks import config
from clawmarks import curation_server as cs
from clawmarks.build import coverage_map


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", tmp_path / "state" / "active_leg.json")
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "real.jpg").write_bytes(b"real")
    monkeypatch.setattr(cs, "REAL_DIR", str(real_dir))
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None
    yield


@pytest.fixture
def running_server(tmp_path):
    """Stand up a real Handler instance with a `demo` expedition whose `round1` leg has a
    scored_manifest.json containing 16 images with distinct (centroid_sim, novelty) values
    so coverage_map.compute_data produces frontier cells, and every manifest file path
    exists on disk so FocusStore._validate_manifest_members passes."""
    cs._create_expedition({"name": "demo", "textures": [], "fallback_subjects": []})
    leg_dir = config.leg_dir("demo", "round1")
    leg_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    for index in range(16):
        tag = f"t{index}"
        image_path = leg_dir / f"{tag}.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + tag.encode())
        manifest_records.append({
            "tag": tag,
            "prompt_name": "p",
            "prompt_type": "style",
            "centroid_sim": 0.1 + index * 0.05,
            "novelty": 0.1 + index * 0.05,
            "strength": 1.0,
            "cfg": 7.0,
            "file": str(image_path),
        })
    (leg_dir / "scored_manifest.json").write_text(json.dumps(manifest_records))
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=2)


def _request_json(server, path, method, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_address[1]}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "null")


def post_json(server, path, payload):
    return _request_json(server, path, "POST", payload)


def get_json(server, path):
    return _request_json(server, path, "GET")


def patch_json(server, path, payload):
    return _request_json(server, path, "PATCH", payload)


@pytest.fixture
def focus_payload():
    return {
        "scope": {"expedition": "demo", "leg": "round1"},
        "label": "Ink anchor",
        "source": {
            "view": "map",
            "kind": "map_members",
            "member_tags": ["t0", "t1"],
            "real_anchor_tags": ["real.jpg"],
            "projection_hint": {
                "projection_version": "sha256:abc",
                "polygon": [[0.1, 0.2]],
            },
        },
        "question": "Keep these spaces",
        "observation": "Six clusters.",
        "hypothesis_text": "Marks survive.",
        "test_contract": None,
    }


def test_focus_create_get_update_archive_round_trip(running_server, focus_payload):
    status, created = post_json(running_server, "/api/foci", focus_payload)
    assert status == 201
    focus_id = created["focus_id"]

    status, fetched = get_json(
        running_server, f"/api/foci/{focus_id}?expedition=demo&leg=round1"
    )
    assert status == 200 and fetched == created

    status, updated = patch_json(
        running_server,
        f"/api/foci/{focus_id}?expedition=demo&leg=round1",
        {"expected_revision": 1, "changes": {"observation": "Changed"}},
    )
    assert status == 200 and updated["revision"] == 2

    status, archived = post_json(
        running_server,
        f"/api/foci/{focus_id}/archive?expedition=demo&leg=round1",
        {"expected_revision": 2},
    )
    assert status == 200
    assert archived["status"] == "archived"
    assert archived["revision"] == 3


def test_focus_route_rejects_scope_mismatch_without_using_active_leg(running_server, focus_payload):
    _, created = post_json(running_server, "/api/foci", focus_payload)
    status, body = get_json(
        running_server,
        f"/api/foci/{created['focus_id']}?expedition=demo&leg=other",
    )
    assert status == 404


def test_focus_list_filters_by_status(running_server, focus_payload):
    status, open_focus = post_json(running_server, "/api/foci", focus_payload)
    assert status == 201

    archived_payload = {**focus_payload, "source": {**focus_payload["source"],
                                                    "member_tags": ["t2", "t3"]}}
    _, archived_focus = post_json(running_server, "/api/foci", archived_payload)
    post_json(running_server,
              f"/api/foci/{archived_focus['focus_id']}/archive?expedition=demo&leg=round1",
              {"expected_revision": 1})

    status, body = get_json(running_server, "/api/foci?expedition=demo&leg=round1")
    assert status == 200
    assert {item["focus_id"] for item in body} == {open_focus["focus_id"], archived_focus["focus_id"]}

    status, open_body = get_json(running_server, "/api/foci?expedition=demo&leg=round1&status=open")
    assert status == 200
    assert {item["focus_id"] for item in open_body} == {open_focus["focus_id"]}

    status, archived_body = get_json(running_server, "/api/foci?expedition=demo&leg=round1&status=archived")
    assert status == 200
    assert {item["focus_id"] for item in archived_body} == {archived_focus["focus_id"]}


def test_focus_list_rejects_unsupported_status_filter(running_server):
    status, body = get_json(running_server, "/api/foci?expedition=demo&leg=round1&status=banana")
    assert status == 400
    assert "banana" in body["error"] or "status" in body["error"].lower()


def test_focus_patch_returns_409_with_current_record_on_stale_revision(running_server, focus_payload):
    _, created = post_json(running_server, "/api/foci", focus_payload)

    status, updated = patch_json(
        running_server,
        f"/api/foci/{created['focus_id']}?expedition=demo&leg=round1",
        {"expected_revision": 1, "changes": {"label": "first"}},
    )
    assert status == 200
    assert updated["revision"] == 2

    status, body = patch_json(
        running_server,
        f"/api/foci/{created['focus_id']}?expedition=demo&leg=round1",
        {"expected_revision": 1, "changes": {"label": "stale"}},
    )
    assert status == 409
    assert body["current"]["revision"] == 2
    assert body["current"]["focus_id"] == created["focus_id"]


def test_focus_archive_returns_409_with_current_record_on_stale_revision(running_server, focus_payload):
    _, created = post_json(running_server, "/api/foci", focus_payload)

    status, body = post_json(
        running_server,
        f"/api/foci/{created['focus_id']}/archive?expedition=demo&leg=round1",
        {"expected_revision": 99},
    )
    assert status == 409
    assert body["current"]["revision"] == 1


def test_focus_post_rejects_malformed_json(running_server):
    port = running_server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/foci",
        data=b"{not valid",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_focus_patch_rejects_malformed_json(running_server, focus_payload):
    _, created = post_json(running_server, "/api/foci", focus_payload)
    port = running_server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/foci/{created['focus_id']}?expedition=demo&leg=round1",
        data=b"{not valid",
        method="PATCH",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_focus_get_requires_expedition_and_leg_query_params(running_server):
    status, body = get_json(running_server, "/api/foci")
    assert status == 400
    status, body = get_json(running_server, "/api/foci?expedition=demo")
    assert status == 400


def test_focus_get_one_requires_expedition_and_leg_query_params(running_server, focus_payload):
    _, created = post_json(running_server, "/api/foci", focus_payload)
    status, body = get_json(running_server, f"/api/foci/{created['focus_id']}")
    assert status == 400


def test_focus_patch_requires_expedition_and_leg_query_params(running_server, focus_payload):
    _, created = post_json(running_server, "/api/foci", focus_payload)
    status, body = patch_json(
        running_server,
        f"/api/foci/{created['focus_id']}",
        {"expected_revision": 1, "changes": {"label": "x"}},
    )
    assert status == 400


def test_focus_archive_requires_expedition_and_leg_query_params(running_server, focus_payload):
    _, created = post_json(running_server, "/api/foci", focus_payload)
    status, body = post_json(
        running_server,
        f"/api/foci/{created['focus_id']}/archive",
        {"expected_revision": 1},
    )
    assert status == 400


def test_focus_create_rejects_unknown_member_tag(running_server, focus_payload):
    bad_payload = {**focus_payload,
                   "source": {**focus_payload["source"], "member_tags": ["t0", "ghost"]}}
    status, body = post_json(running_server, "/api/foci", bad_payload)
    assert status == 400


def test_focus_create_coverage_frontier_recomputes_cells_server_side(running_server):
    leg_dir = config.leg_dir("demo", "round1")
    data = coverage_map.compute_data(str(leg_dir))
    interior_frontier = [
        c for c in data["cells"]
        if c["frontier"]
        and c["faith_lo"] is not None and c["faith_hi"] is not None
        and c["novelty_lo"] is not None and c["novelty_hi"] is not None
    ]
    assert interior_frontier, "test fixture must produce at least one interior frontier cell"
    target = interior_frontier[0]

    by_coord = {(c["fb"], c["nb"]): c for c in data["cells"]}
    adjacent_tags = []
    for nc in [(target["fb"] + 1, target["nb"]),
               (target["fb"] - 1, target["nb"]),
               (target["fb"], target["nb"] + 1),
               (target["fb"], target["nb"] - 1)]:
        cell = by_coord.get(nc)
        if cell and cell["count"] > 0:
            adjacent_tags.extend(item["tag"] for item in cell["items"])
    assert adjacent_tags

    payload = {
        "scope": {"expedition": "demo", "leg": "round1"},
        "label": "Empty bin",
        "source": {
            "view": "coverage",
            "kind": "coverage_frontier",
            "score_ranges": {
                "faithfulness": [target["faith_lo"], target["faith_hi"]],
                "novelty": [target["novelty_lo"], target["novelty_hi"]],
            },
            "adjacent_member_tags": adjacent_tags[:1],
            "real_anchor_tags": ["real.jpg"],
            "coverage_hint": {
                "binning_version": "sha256:abc",
                "row": target["fb"],
                "column": target["nb"],
            },
        },
        "question": "Push here",
        "observation": "Frontier cell.",
        "hypothesis_text": "Marks survive.",
        "test_contract": None,
    }

    status, created = post_json(running_server, "/api/foci", payload)
    assert status == 201
    assert created["source"]["kind"] == "coverage_frontier"


def test_focus_create_rejects_client_supplied_cell_that_does_not_match_recompute(running_server):
    # Use a score_range that is valid per the metric domain but doesn't match any cell's
    # exact (faith_lo, faith_hi, novelty_lo, novelty_hi) in the recomputed coverage_map. The
    # server must recompute Coverage server-side and reject the request with 400, not trust
    # the client to claim an empty frontier cell that does not exist.
    payload = {
        "scope": {"expedition": "demo", "leg": "round1"},
        "label": "Stale claim",
        "source": {
            "view": "coverage",
            "kind": "coverage_frontier",
            "score_ranges": {
                "faithfulness": [-0.95, -0.9],
                "novelty": [1.9, 1.95],
            },
            "adjacent_member_tags": ["t0"],
            "real_anchor_tags": ["real.jpg"],
        },
        "question": "",
        "observation": "",
        "hypothesis_text": "",
        "test_contract": None,
    }

    status, body = post_json(running_server, "/api/foci", payload)
    assert status == 400


def test_focus_create_rejects_unknown_real_anchor(running_server, focus_payload):
    bad_payload = {**focus_payload,
                   "source": {**focus_payload["source"], "real_anchor_tags": ["nope.jpg"]}}
    status, body = post_json(running_server, "/api/foci", bad_payload)
    assert status == 400


def test_focus_get_404_for_unknown_focus_id(running_server):
    status, body = get_json(
        running_server,
        "/api/foci/focus_0123456789abcdef0123456789abcdef?expedition=demo&leg=round1",
    )
    assert status == 404


def test_focus_patch_rejects_missing_changes(running_server, focus_payload):
    _, created = post_json(running_server, "/api/foci", focus_payload)
    status, body = patch_json(
        running_server,
        f"/api/foci/{created['focus_id']}?expedition=demo&leg=round1",
        {"expected_revision": 1},
    )
    assert status == 400


def test_focus_patch_rejects_missing_expected_revision(running_server, focus_payload):
    _, created = post_json(running_server, "/api/foci", focus_payload)
    status, body = patch_json(
        running_server,
        f"/api/foci/{created['focus_id']}?expedition=demo&leg=round1",
        {"changes": {"label": "x"}},
    )
    assert status == 400


def test_focus_does_not_change_active_leg_selection(running_server, focus_payload):
    """Per design spec: explicit-scope Focus calls must never fall through to the active
    selection. They also must not mutate it. Verify the global _active_selection is
    unchanged across create/get/list/patch/archive."""
    cs._set_active_selection("demo", "round1")
    before = dict(cs._active_selection)

    _, created = post_json(running_server, "/api/foci", focus_payload)
    get_json(running_server,
             f"/api/foci/{created['focus_id']}?expedition=demo&leg=round1")
    get_json(running_server, "/api/foci?expedition=demo&leg=round1")
    patch_json(running_server,
               f"/api/foci/{created['focus_id']}?expedition=demo&leg=round1",
               {"expected_revision": 1, "changes": {"label": "x"}})
    post_json(running_server,
              f"/api/foci/{created['focus_id']}/archive?expedition=demo&leg=round1",
              {"expected_revision": 2})

    assert cs._active_selection == before