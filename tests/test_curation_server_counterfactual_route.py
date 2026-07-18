import base64
import json
import threading
from http.server import HTTPServer
import urllib.error
import urllib.request

import pytest

from clawmarks import curation_server as cs

FAKE_PNG = base64.b64encode(b"fake-png-bytes").decode()


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_active_out_dir", lambda: tmp_path)
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    (tmp_path / "counterfactuals").mkdir()
    (tmp_path / "scored_manifest.json").write_text("[]")
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown()
    thread.join(timeout=2)


def _post_json(url, payload):
    req = urllib.request.Request(
        url, method="POST", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _stub_immediate_completion(monkeypatch, job_ids=None):
    """Every submitted job completes immediately with one fake image."""
    calls = {"submit": 0, "balance": 0}

    def fake_balance(api_key):
        calls["balance"] += 1
        return 10.0

    def fake_comfy_post(path, wf, api_key):
        calls["submit"] += 1
        return {"id": f"job-{calls['submit']}"}

    def fake_comfy_get(path, api_key):
        return {"status": "COMPLETED", "output": {"images": [{"data": FAKE_PNG}]}}

    monkeypatch.setattr(cs, "runpod_balance", fake_balance)
    monkeypatch.setattr(cs, "comfy_post", fake_comfy_post)
    monkeypatch.setattr(cs, "comfy_get", fake_comfy_get)
    return calls


def test_counterfactual_defaults_to_one_result(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    _stub_immediate_completion(monkeypatch)

    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/counterfactual",
        {"origin_tag": "a", "prompt": "a cat"},
    )

    assert status == 200
    assert data["ok"] is True
    assert len(data["results"]) == 1
    assert data["results"][0]["origin_tag"] == "a"


def test_counterfactual_n_generates_multiple_distinct_images(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    _stub_immediate_completion(monkeypatch)

    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/counterfactual",
        {"origin_tag": "a", "prompt": "a cat", "n": 3},
    )

    assert status == 200
    assert len(data["results"]) == 3
    tags = {r["tag"] for r in data["results"]}
    assert len(tags) == 3, "expected 3 distinct tags, got a collision"
    stored = json.loads((tmp_path / "user_counterfactuals.json").read_text())
    assert len(stored) == 3


def test_counterfactual_n_is_capped_at_six(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    _stub_immediate_completion(monkeypatch)

    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/counterfactual",
        {"origin_tag": "a", "prompt": "a cat", "n": 20},
    )

    assert status == 200
    assert len(data["results"]) == 6


def test_counterfactual_balance_checked_once_regardless_of_n(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    calls = _stub_immediate_completion(monkeypatch)

    _post_json(f"http://127.0.0.1:{port}/api/counterfactual",
               {"origin_tag": "a", "prompt": "a cat", "n": 4})

    assert calls["balance"] == 1
    assert calls["submit"] == 4


def test_counterfactual_pinned_seed_forces_a_single_job_even_if_n_is_higher(running_server, monkeypatch):
    """A pinned seed makes every job in a batch byte-identical (same prompt/strength/cfg too),
    so honoring n>1 would just pay RunPod for n copies of the same image."""
    server, tmp_path = running_server
    port = server.server_address[1]
    calls = _stub_immediate_completion(monkeypatch)

    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/counterfactual",
        {"origin_tag": "a", "prompt": "a cat", "n": 3, "seed": 42},
    )

    assert status == 200
    assert len(data["results"]) == 1
    assert data["results"][0]["seed"] == 42
    assert calls["submit"] == 1


def test_counterfactual_partial_batch_failure_returns_what_succeeded(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    calls = {"submit": 0}

    def fake_balance(api_key):
        return 10.0

    def fake_comfy_post(path, wf, api_key):
        calls["submit"] += 1
        if calls["submit"] == 2:
            raise RuntimeError("connection reset")
        return {"id": f"job-{calls['submit']}"}

    def fake_comfy_get(path, api_key):
        return {"status": "COMPLETED", "output": {"images": [{"data": FAKE_PNG}]}}

    monkeypatch.setattr(cs, "runpod_balance", fake_balance)
    monkeypatch.setattr(cs, "comfy_post", fake_comfy_post)
    monkeypatch.setattr(cs, "comfy_get", fake_comfy_get)

    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/counterfactual",
        {"origin_tag": "a", "prompt": "a cat", "n": 3},
    )

    assert status == 502
    assert len(data["results"]) == 1
    assert "error" in data
    assert calls["submit"] == 2, "batch should stop after the first failure, not keep spending"


def test_counterfactual_n_non_numeric_returns_400(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    calls = _stub_immediate_completion(monkeypatch)

    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/counterfactual",
        {"origin_tag": "a", "prompt": "a cat", "n": "lots"},
    )

    assert status == 400
    assert calls["balance"] == 0


def test_scoped_counterfactual_result_is_served_from_named_leg(running_server, monkeypatch, tmp_path):
    server, _ = running_server
    expeditions = tmp_path / "expeditions"
    leg_dir = expeditions / "demo" / "round1"
    leg_dir.mkdir(parents=True)
    (expeditions / "demo" / "legs").mkdir()
    (expeditions / "demo" / "expedition.json").write_text("{}")
    (expeditions / "demo" / "legs" / "round1.json").write_text("{}")
    monkeypatch.setattr(cs.config, "EXPEDITIONS_DIR", expeditions)
    _stub_immediate_completion(monkeypatch)

    port = server.server_address[1]
    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/counterfactual",
        {"origin_tag": "a", "prompt": "a cat", "expedition": "demo", "leg": "round1"},
    )

    assert status == 200
    image_url = data["results"][0]["file"]
    assert image_url.startswith("/counterfactuals/")
    assert "expedition=demo" in image_url and "leg=round1" in image_url
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{image_url}") as response:
        assert response.read() == b"fake-png-bytes"
    assert list((leg_dir / "counterfactuals").glob("*.png"))


def test_scoped_counterfactual_rejects_unsafe_scope(running_server, monkeypatch):
    calls = _stub_immediate_completion(monkeypatch)
    server, _ = running_server
    status, data = _post_json(
        f"http://127.0.0.1:{server.server_address[1]}/api/counterfactual",
        {"origin_tag": "a", "prompt": "a cat", "expedition": "../escape", "leg": "round1"},
    )

    assert status == 400
    assert "scope" in data["error"] or "separator" in data["error"]
    assert calls["balance"] == 0


def test_counterfactual_rejects_focus_from_a_different_leg_before_generation(
    running_server, monkeypatch, tmp_path
):
    server, _ = running_server
    expeditions = tmp_path / "expeditions"
    expedition_dir = expeditions / "demo"
    (expedition_dir / "legs").mkdir(parents=True)
    (expedition_dir / "expedition.json").write_text("{}")
    for leg in ("round1", "round2"):
        (expedition_dir / "legs" / f"{leg}.json").write_text("{}")
        (expedition_dir / leg).mkdir()
    monkeypatch.setattr(cs.config, "EXPEDITIONS_DIR", expeditions)
    state_dir = tmp_path / "state"
    monkeypatch.setattr(cs.config, "STATE_DIR", state_dir)

    source_path = expedition_dir / "round2" / "source.png"
    source_path.write_bytes(b"image")
    focus = cs.FocusStore(state_dir, tmp_path).create(
        cs.Scope("demo", "round2"),
        {
            "label": "Round two focus",
            "source": {
                "view": "map",
                "kind": "map_members",
                "member_tags": ["source"],
                "real_anchor_tags": [],
            },
            "question": "q",
            "observation": "o",
            "hypothesis_text": "h",
            "test_contract": None,
        },
        [{"tag": "source", "file": str(source_path)}],
    )
    calls = _stub_immediate_completion(monkeypatch)

    status, data = _post_json(
        f"http://127.0.0.1:{server.server_address[1]}/api/counterfactual",
        {
            "origin_tag": "a",
            "prompt": "a cat",
            "expedition": "demo",
            "leg": "round1",
            "focus_id": focus["focus_id"],
        },
    )

    assert status == 400
    assert "focus_id" in data["error"]
    assert calls["balance"] == 0
    assert calls["submit"] == 0
