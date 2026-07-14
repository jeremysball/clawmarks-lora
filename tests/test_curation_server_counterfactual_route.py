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
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "COUNTERFACTUALS_DIR", str(tmp_path / "counterfactuals"))
    monkeypatch.setattr(cs, "COUNTERFACTUALS_FILE", str(tmp_path / "user_counterfactuals.json"))
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
