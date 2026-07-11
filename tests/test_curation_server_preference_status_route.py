import json
import threading
from http.server import HTTPServer
import urllib.request
import urllib.error

import pytest

from clawmarks import curation_server as cs
from clawmarks.search import preference_settings


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(cs.preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(cs.preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    (tmp_path / "scored_manifest.json").write_text(json.dumps([]))
    (tmp_path / "user_ratings.json").write_text(json.dumps({}))
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown()
    thread.join(timeout=2)


def test_preference_status_html_route_serves_page(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/preference_status.html") as resp:
        body = resp.read().decode()
    assert "Preference classifier status" in body


def test_api_preference_status_route_returns_json(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/preference_status") as resp:
        data = json.loads(resp.read().decode())
    assert data["has_model"] is False
    assert data["use_predicted_preference"] is False


def test_post_preference_toggle_rejects_enable_without_model(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/preference_toggle", method="POST",
        data=json.dumps({"enabled": True}).encode(), headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_post_preference_toggle_accepts_enable_with_model_and_persists(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    (tmp_path / "preference_model.joblib").write_text("fake model")

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/preference_toggle", method="POST",
        data=json.dumps({"enabled": True}).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    assert data["use_predicted_preference"] is True
    assert preference_settings.load()["use_predicted_preference"] is True


def test_archive_html_uses_persisted_setting_not_query_param(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    calls = []
    monkeypatch.setattr(cs.elite_archive, "compute_data", lambda sd, use_predicted_preference=False: calls.append(use_predicted_preference) or {"cells": []})

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/archive.html?use_predicted_preference=1") as resp:
        resp.read()
    assert calls == [False]

    preference_settings.save(True)
    (tmp_path / "preference_model.joblib").write_text("fake model")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/archive.html") as resp:
        resp.read()
    assert calls == [False, True]
