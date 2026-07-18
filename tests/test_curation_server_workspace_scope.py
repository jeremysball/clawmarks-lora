import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

from PIL import Image
import pytest

from clawmarks import config
from clawmarks import curation_server as cs
from clawmarks.focus_store import FocusStore, Scope
from clawmarks.workspace_context import WorkspaceContext, generated_image_url


@pytest.fixture
def server_fixture(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    expeditions_dir = state_dir / "expeditions"
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "anchor.jpg").write_bytes(b"anchor")
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", expeditions_dir)
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", state_dir / "active_leg.json")
    monkeypatch.setattr(cs, "REAL_DIR", str(real_dir))
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    monkeypatch.setattr(cs, "_manifest_cache", {})
    cs._active_selection.update(expedition=None, leg=None)

    for leg, prefix in (("round1", "one"), ("round2", "two"), ("current", "current")):
        out_dir = config.leg_dir("demo", leg)
        out_dir.mkdir(parents=True)
        records = []
        for index in range(16):
            tag = f"{prefix}-{index}"
            image_path = out_dir / f"{tag}.png"
            Image.new("RGB", (300, 300), color=(index * 10, 20, 30)).save(image_path)
            records.append({
                "tag": tag,
                "prompt_name": "p",
                "prompt_type": "style",
                "prompt": "a test prompt",
                "category": "style",
                "centroid_sim": 0.1 + index * 0.05,
                "novelty": 0.1 + index * 0.05,
                "strength": 1.0,
                "cfg": 7.0,
                "seed": index,
                "steps": 20,
                "sampler": "ddim",
                "negative": "",
                "file": str(image_path),
            })
        if leg == "current":
            source_path = config.leg_dir("demo", "round1") / "one-0.png"
            records.append({"tag": "outside", "file": str(source_path)})
        (out_dir / "scored_manifest.json").write_text(json.dumps(records))

    focus_store = FocusStore(state_dir, real_dir)
    focus = focus_store.create(
        Scope("demo", "round1"),
        {
            "label": "Round one",
            "source": {
                "view": "map",
                "kind": "map_members",
                "member_tags": ["one-0"],
                "real_anchor_tags": ["anchor.jpg"],
            },
            "question": "Which mark survives?",
            "observation": "Round one.",
            "hypothesis_text": "The first leg is distinct.",
            "test_contract": None,
        },
        json.loads((config.leg_dir("demo", "round1") / "scored_manifest.json").read_text()),
    )

    config.ACTIVE_LEG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.ACTIVE_LEG_FILE.write_text(json.dumps({"expedition": "demo", "leg": "current"}))
    cs._active_selection.update(expedition="demo", leg="current")
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, focus
    server.shutdown()
    thread.join(timeout=2)


def get_response(server, path):
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_explicit_focus_page_reads_its_leg_without_switching_global_selection(server_fixture):
    server, focus = server_fixture
    before = config.ACTIVE_LEG_FILE.read_bytes()
    url = f"/coverage.html?expedition=demo&leg=round1&focus_id={focus['focus_id']}"

    status, body = get_response(server, url)

    assert status == 200
    assert b"one-0" in body
    assert b"two-0" not in body
    scan_status, scan_body = get_response(
        server, f"/scan.html?expedition=demo&leg=round1&focus_id={focus['focus_id']}"
    )
    assert scan_status == 200
    assert b"/thumbs/one-0.jpg?expedition=demo&leg=round1" in scan_body
    assert b'"thumb": "thumbs/one-0.jpg"' not in scan_body
    assert config.ACTIVE_LEG_FILE.read_bytes() == before
    assert cs._active_selection == {"expedition": "demo", "leg": "current"}


def test_scoped_pages_keep_two_legs_data_separate(server_fixture):
    server, _ = server_fixture

    first_status, first_body = get_response(server, "/coverage.html?expedition=demo&leg=round1")
    second_status, second_body = get_response(server, "/coverage.html?expedition=demo&leg=round2")

    assert first_status == second_status == 200
    assert b"one-0" in first_body and b"two-0" not in first_body
    assert b"two-0" in second_body and b"one-0" not in second_body


def test_explicit_generated_image_and_thumbnail_use_requested_leg(server_fixture):
    server, _ = server_fixture
    query = "?expedition=demo&leg=round1"

    full_status, full_body = get_response(server, f"/generated/one-0{query}")
    thumb_status, thumb_body = get_response(server, f"/thumbs/one-0.jpg{query}")

    assert full_status == 200
    assert full_body.startswith(b"\x89PNG")
    assert thumb_status == 200
    assert thumb_body.startswith(b"\xff\xd8")


def test_legacy_thumbnail_rejects_path_traversal_before_cache_write(server_fixture):
    server, _ = server_fixture

    status, _ = get_response(server, "/thumbs/../../outside.jpg")

    assert status == 404
    assert not (config.EXPEDITIONS_DIR / "demo" / "outside.jpg").exists()


def test_generated_image_rejects_path_traversal_tag_before_manifest_lookup(
    server_fixture, monkeypatch
):
    server, _ = server_fixture

    def unexpected_manifest_lookup(*args):
        pytest.fail("traversal-shaped generated tag reached manifest lookup")

    monkeypatch.setattr(cs, "manifest_entry_by_tag", unexpected_manifest_lookup)
    status, _ = get_response(
        server, "/generated/../../outside?expedition=demo&leg=round1"
    )

    assert status == 404


def test_generated_image_rejects_manifest_path_outside_requested_leg(server_fixture, tmp_path):
    server, _ = server_fixture
    outside = tmp_path / "outside.png"
    Image.new("RGB", (20, 20), color="red").save(outside)
    manifest_path = config.leg_dir("demo", "round1") / "scored_manifest.json"
    manifest_path.write_text(json.dumps([{"tag": "escape", "file": str(outside)}]))

    status, _ = get_response(server, "/generated/escape?expedition=demo&leg=round1")

    assert status == 404


@pytest.mark.parametrize(
    "path",
    [
        "/thumbs/one-0.jpg?focus_id=",
        "/thumbs/one-0.jpg?expedition=&leg=round1",
        "/thumbs/one-0.jpg?expedition=demo&leg=",
    ],
)
def test_scoped_thumbnail_rejects_blank_scope_values(server_fixture, path):
    server, _ = server_fixture

    status, _ = get_response(server, path)

    assert status == 400


def test_generated_image_url_preserves_leg_scope():
    context = WorkspaceContext(expedition="demo", leg="round1")

    assert generated_image_url("one-0", context) == "/generated/one-0?expedition=demo&leg=round1"
    assert generated_image_url("one-0", context, thumbnail=True) == "/thumbs/one-0.jpg?expedition=demo&leg=round1"
