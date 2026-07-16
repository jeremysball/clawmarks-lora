import json
import os
import threading
from http.server import HTTPServer
import urllib.error
import urllib.request

import pytest

from clawmarks import config
from clawmarks import curation_server as cs
from clawmarks.search import run_manager


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "LOCK_FILE", tmp_path / ".searchrun.lock")
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", tmp_path / "state" / "active_leg.json")
    (config.EXPEDITIONS_DIR / "demo" / "legs").mkdir(parents=True)
    (config.EXPEDITIONS_DIR / "demo" / "expedition.json").write_text("{}")
    (config.EXPEDITIONS_DIR / "demo" / "legs" / "leg1.json").write_text("{}")
    cs._set_active_selection("demo", "leg1")
    out_dir = config.leg_dir("demo", "leg1")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scored_manifest.json").write_text("[]")
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, out_dir
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


def _get_json(url):
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_status_reports_not_running_when_idle(running_server):
    server, _ = running_server
    port = server.server_address[1]

    status, data = _get_json(f"http://127.0.0.1:{port}/api/searchrun/status")

    assert status == 200
    assert data == {"running": False}


def test_launch_starts_a_run_and_status_reflects_it(running_server, monkeypatch):
    server, _ = running_server
    port = server.server_address[1]
    monkeypatch.setattr(run_manager, "runpod_balance", lambda key: 100.0)

    class FakeProc:
        pid = os.getpid()

    captured = {}

    def fake_popen(*a, **k):
        captured["args"] = a
        captured["kwargs"] = k
        return FakeProc()

    monkeypatch.setattr(cs.subprocess, "Popen", fake_popen)

    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/searchrun/launch",
        {"expedition": "demo", "leg": "leg1"},
    )

    assert status == 200
    assert data["ok"] is True
    assert data["pid"] == FakeProc.pid
    assert captured["kwargs"]["start_new_session"] is True

    status, data = _get_json(f"http://127.0.0.1:{port}/api/searchrun/status")
    assert status == 200
    assert data["running"] is True
    assert data["expedition"] == "demo"
    assert data["leg"] == "leg1"


def test_launch_refuses_when_already_running(running_server, monkeypatch):
    server, _ = running_server
    port = server.server_address[1]
    monkeypatch.setattr(run_manager, "runpod_balance", lambda key: 100.0)

    class FakeProc:
        pid = os.getpid()

    monkeypatch.setattr(cs.subprocess, "Popen", lambda *a, **k: FakeProc())

    status, _ = _post_json(f"http://127.0.0.1:{port}/api/searchrun/launch", {"expedition": "demo", "leg": "leg1"})
    assert status == 200

    status, data = _post_json(f"http://127.0.0.1:{port}/api/searchrun/launch", {"expedition": "demo", "leg": "leg1"})
    assert status == 409
    assert "already" in data["error"]


def test_launch_refuses_when_balance_below_floor(running_server, monkeypatch):
    server, _ = running_server
    port = server.server_address[1]
    monkeypatch.setattr(run_manager, "runpod_balance", lambda key: 0.0)

    status, data = _post_json(f"http://127.0.0.1:{port}/api/searchrun/launch", {"expedition": "demo", "leg": "leg1"})

    assert status == 402
    assert "floor" in data["error"]


def test_launch_rejects_unknown_leg(running_server):
    server, _ = running_server
    port = server.server_address[1]

    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/searchrun/launch",
        {"expedition": "demo", "leg": "does_not_exist"},
    )

    assert status == 400


def test_stop_is_noop_when_nothing_running(running_server):
    server, _ = running_server
    port = server.server_address[1]

    status, data = _post_json(f"http://127.0.0.1:{port}/api/searchrun/stop", {})

    assert status == 200
    assert data == {"running": False}


def test_report_reflects_state_and_manifest_on_disk(running_server):
    server, out_dir = running_server
    port = server.server_address[1]
    (out_dir / "allnight_state.json").write_text(json.dumps({
        "generation": 2, "stage": 0, "plateau_count": 1,
        "novelty_history": [0.3, 0.35], "gpt55_subjects": [],
        "start_balance": 5.0, "start_time": 1.0,
    }))
    (out_dir / "scored_manifest.json").write_text(json.dumps([
        {"tag": "gen1_explore_0", "category": "r2_explore"},
    ]))

    status, data = _get_json(f"http://127.0.0.1:{port}/api/searchrun/report?expedition=demo&leg=leg1")

    assert status == 200
    assert data["novelty_trajectory"] == [0.3, 0.35]
    assert data["plateau_count"] == 1
    assert data["total_images"] == 1


def test_stop_terminates_a_running_run(running_server, monkeypatch):
    server, _ = running_server
    port = server.server_address[1]
    monkeypatch.setattr(run_manager, "runpod_balance", lambda key: 100.0)

    class FakeProc:
        pid = os.getpid()

    monkeypatch.setattr(cs.subprocess, "Popen", lambda *a, **k: FakeProc())

    _post_json(f"http://127.0.0.1:{port}/api/searchrun/launch", {"expedition": "demo", "leg": "leg1"})

    monkeypatch.setattr(run_manager, "is_process_alive", lambda pid: False)
    monkeypatch.setattr(run_manager.os, "kill", lambda pid, sig: None)

    status, data = _post_json(f"http://127.0.0.1:{port}/api/searchrun/stop", {})

    assert status == 200
    assert data == {"running": False}


def test_stop_passes_confirmed_run_identity_to_manager(running_server, monkeypatch):
    server, _ = running_server
    port = server.server_address[1]
    captured = {}

    def fake_stop_run(**kwargs):
        captured.update(kwargs)
        return {"running": True}

    monkeypatch.setattr(run_manager, "stop_run", fake_stop_run)

    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/searchrun/stop",
        {"pid": 12345, "start_time_ticks": 999},
    )

    assert status == 200
    assert data == {"running": True}
    assert captured == {"pid": 12345, "start_time_ticks": 999}
