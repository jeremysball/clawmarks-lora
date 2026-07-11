import json
import threading
from http.server import HTTPServer
import urllib.request
import urllib.error

import numpy as np
import pytest

from clawmarks import curation_server as cs
from clawmarks.search import embed_cache
from clawmarks.search import preference_settings


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(cs.preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(cs.preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    monkeypatch.setattr(cs.preference_model, "MODEL_META_FILE", tmp_path / "preference_model_meta.json")
    monkeypatch.setattr(cs.preference_model, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(embed_cache, "EMBEDDINGS_FILE", tmp_path / "embeddings.npz")
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


def _post_json(url, payload=None):
    req = urllib.request.Request(
        url, method="POST",
        data=json.dumps(payload or {}).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _write_ratings(tmp_path, n_yes, n_no):
    ratings = {}
    for i in range(n_yes):
        ratings[f"y{i}"] = {"label": "yes", "rated_at": "t"}
    for i in range(n_no):
        ratings[f"n{i}"] = {"label": "no", "rated_at": "t"}
    (tmp_path / "user_ratings.json").write_text(json.dumps(ratings))


def test_post_preference_retrain_trains_and_returns_updated_status(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    _write_ratings(tmp_path, n_yes=30, n_no=30)
    rng = np.random.RandomState(0)
    embeddings = np.vstack([
        rng.normal(loc=5.0, scale=0.1, size=(30, 2)),
        rng.normal(loc=-5.0, scale=0.1, size=(30, 2)),
    ]).astype(np.float32)
    tags = [f"y{i}" for i in range(30)] + [f"n{i}" for i in range(30)]
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)

    data = _post_json(f"http://127.0.0.1:{port}/api/preference_retrain")

    assert data["has_model"] is True
    assert data["model_meta"]["n_labels"] == 60
    assert data["model_meta"]["p_value"] < 0.05


def test_post_preference_retrain_rejects_under_min_labels(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    _write_ratings(tmp_path, n_yes=10, n_no=10)
    embeddings = np.random.RandomState(0).normal(size=(20, 2)).astype(np.float32)
    tags = [f"y{i}" for i in range(10)] + [f"n{i}" for i in range(10)]
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)
    called = False

    def fake_train(argv):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(cs.preference_model, "main", fake_train)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/preference_retrain", method="POST",
        data=json.dumps({}).encode(), headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)

    body = json.loads(exc_info.value.read().decode())
    assert exc_info.value.code == 400
    assert "only 20 usable labels" in body["error"]
    assert called is False


def test_post_preference_retrain_rejects_class_imbalance(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    _write_ratings(tmp_path, n_yes=58, n_no=2)
    embeddings = np.random.RandomState(0).normal(size=(60, 2)).astype(np.float32)
    tags = [f"y{i}" for i in range(58)] + [f"n{i}" for i in range(2)]
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)
    called = False

    def fake_train(argv):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(cs.preference_model, "main", fake_train)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/preference_retrain", method="POST",
        data=json.dumps({}).encode(), headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)

    body = json.loads(exc_info.value.read().decode())
    assert exc_info.value.code == 400
    assert "5-fold" in body["error"]
    assert called is False


def test_post_preference_retrain_returns_500_on_training_crash(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    _write_ratings(tmp_path, n_yes=30, n_no=30)
    embeddings = np.random.RandomState(0).normal(size=(60, 2)).astype(np.float32)
    tags = [f"y{i}" for i in range(30)] + [f"n{i}" for i in range(30)]
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)

    def crashing_train(argv):
        raise RuntimeError("disk full")

    monkeypatch.setattr(cs.preference_model, "main", crashing_train)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/preference_retrain", method="POST",
        data=json.dumps({}).encode(), headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)

    assert exc_info.value.code == 500
    body = json.loads(exc_info.value.read().decode())
    assert "disk full" in body["error"]
