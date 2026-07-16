import json
import threading
import time
from http.server import HTTPServer
import urllib.request
import urllib.error

import numpy as np
import pytest

from clawmarks import curation_server as cs
from clawmarks.search import embed_cache, preference_settings


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_active_out_dir", lambda: tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text(json.dumps([]))
    (tmp_path / "user_comparisons.json").write_text(json.dumps([]))
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


def test_preference_rank_html_serves_no_model_state_before_first_manifest(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    (tmp_path / "scored_manifest.json").unlink()

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/preference_rank.html") as resp:
        body = resp.read().decode()

    assert "No trained model" in body


def test_prediction_watch_list_includes_model_paths_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_active_out_dir", lambda: tmp_path)

    watched_files = cs._prediction_watched_files()

    assert str(tmp_path / "preference_pairwise_model.joblib") in watched_files
    assert str(tmp_path / "preference_pairwise_model_meta.json") in watched_files


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
    (tmp_path / "preference_pairwise_model.joblib").write_text("fake model")

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/preference_toggle", method="POST",
        data=json.dumps({"enabled": True}).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    assert data["use_predicted_preference"] is True
    assert preference_settings.load(tmp_path)["use_predicted_preference"] is True


def test_preference_rank_flags_persist_without_becoming_training_labels(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]

    result = _post_json(
        f"http://127.0.0.1:{port}/api/preference_rank/flag",
        {"tag": "sample", "flag": "questionable"},
    )
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/preference_rank/flags") as resp:
        flags = json.loads(resp.read().decode())

    assert result == {"ok": True, "tag": "sample", "flag": "questionable"}
    assert flags["sample"]["flag"] == "questionable"
    assert json.loads((tmp_path / "user_comparisons.json").read_text()) == []


def test_archive_html_uses_persisted_setting_not_query_param(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    calls = []
    monkeypatch.setattr(cs.elite_archive, "compute_data", lambda sd, use_predicted_preference=False: calls.append(use_predicted_preference) or {"cells": []})

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/archive.html?use_predicted_preference=1") as resp:
        resp.read()
    assert calls == [False]

    preference_settings.save(True, tmp_path)
    (tmp_path / "preference_pairwise_model.joblib").write_text("fake model")
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


def _write_comparisons(tmp_path, n):
    comparisons = [{"winner": f"w{i}", "loser": f"l{i}", "compared_at": "t"} for i in range(n)]
    (tmp_path / "user_comparisons.json").write_text(json.dumps(comparisons))
    return comparisons


def test_post_preference_retrain_trains_and_returns_updated_status(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    comparisons = _write_comparisons(tmp_path, 50)
    tags = sorted({t for c in comparisons for t in (c["winner"], c["loser"])})
    rng = np.random.RandomState(0)
    embeddings = rng.normal(size=(len(tags), 2)).astype(np.float32)
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)

    data = _post_json(f"http://127.0.0.1:{port}/api/preference_retrain")

    assert data["has_model"] is True
    assert data["model_meta"]["n_comparisons"] == 50
    assert 0.0 <= data["model_meta"]["p_value"] <= 1.0


def test_post_preference_retrain_rejects_under_min_comparisons(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    comparisons = _write_comparisons(tmp_path, 10)
    tags = sorted({t for c in comparisons for t in (c["winner"], c["loser"])})
    embeddings = np.random.RandomState(0).normal(size=(len(tags), 2)).astype(np.float32)
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)
    called = False

    def fake_train_and_save(comparisons):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(cs.preference_pairwise_model, "train_and_save", fake_train_and_save)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/preference_retrain", method="POST",
        data=json.dumps({}).encode(), headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)

    body = json.loads(exc_info.value.read().decode())
    assert exc_info.value.code == 400
    assert "only 10 usable comparisons" in body["error"]
    assert called is False


def test_post_preference_retrain_reports_uncached_embeddings_distinctly(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    _write_comparisons(tmp_path, 50)
    # No embeddings.npz written at all: every comparison references an uncached tag.
    called = False

    def fake_train_and_save(comparisons):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(cs.preference_pairwise_model, "train_and_save", fake_train_and_save)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/preference_retrain", method="POST",
        data=json.dumps({}).encode(), headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)

    body = json.loads(exc_info.value.read().decode())
    assert exc_info.value.code == 400
    assert "cached embedding" in body["error"]
    assert called is False


@pytest.fixture
def threaded_running_server(tmp_path, monkeypatch):
    """Same setup as running_server, but backed by cs.ThreadingHTTPServer (what main() actually
    serves with) instead of the plain single-threaded HTTPServer: concurrency tests need real
    concurrent request handling, which the shared fixture's HTTPServer serializes away."""
    monkeypatch.setattr(cs, "_active_out_dir", lambda: tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text(json.dumps([]))
    (tmp_path / "user_comparisons.json").write_text(json.dumps([]))
    server = cs.ThreadingHTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown()
    thread.join(timeout=2)


def test_post_preference_retrain_does_not_hold_lock_during_training(threaded_running_server, monkeypatch):
    """A full model fit can take a while; holding _lock for its duration would block every other
    request (favorites, compare, cockpit) until it finishes. Assert a concurrent /api/compare
    request completes while a slow retrain is still in flight, proving the lock isn't held across
    the training call."""
    server, tmp_path = threaded_running_server
    port = server.server_address[1]
    comparisons = _write_comparisons(tmp_path, 50)
    tags = sorted({t for c in comparisons for t in (c["winner"], c["loser"])})
    embeddings = np.random.RandomState(0).normal(size=(len(tags), 2)).astype(np.float32)
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)

    training_started = threading.Event()
    release_training = threading.Event()
    retrain_errors = []

    def slow_train_and_save(comparisons, out_dir):
        training_started.set()
        # Long enough that a blocked compare request would clearly time out the assertion below;
        # release_training.set() (in the test's finally) cuts this short once compare has
        # already returned, so the happy path doesn't actually wait the full 30s.
        release_training.wait(timeout=30)
        return None

    monkeypatch.setattr(cs.preference_pairwise_model, "train_and_save", slow_train_and_save)

    def run_retrain():
        try:
            _post_json(f"http://127.0.0.1:{port}/api/preference_retrain")
        except urllib.error.HTTPError:
            pass
        except Exception as e:
            retrain_errors.append(e)

    retrain_thread = threading.Thread(target=run_retrain)
    retrain_thread.start()
    assert training_started.wait(timeout=5), "training never started"

    try:
        start = time.monotonic()
        compare_response = _post_json(
            f"http://127.0.0.1:{port}/api/compare",
            {"winner": "concurrent_w", "loser": "concurrent_l"},
        )
        elapsed = time.monotonic() - start
        assert compare_response["ok"] is True
        assert elapsed < 5, f"/api/compare took {elapsed:.1f}s, suggesting it waited on _lock held during training"
    finally:
        release_training.set()
        retrain_thread.join(timeout=5)
    assert retrain_errors == []


def test_post_preference_retrain_returns_500_on_training_crash(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    comparisons = _write_comparisons(tmp_path, 50)
    tags = sorted({t for c in comparisons for t in (c["winner"], c["loser"])})
    embeddings = np.random.RandomState(0).normal(size=(len(tags), 2)).astype(np.float32)
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)

    def crashing_train(comparisons, out_dir):
        raise RuntimeError("disk full")

    monkeypatch.setattr(cs.preference_pairwise_model, "train_and_save", crashing_train)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/preference_retrain", method="POST",
        data=json.dumps({}).encode(), headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)

    assert exc_info.value.code == 500
    body = json.loads(exc_info.value.read().decode())
    assert "disk full" in body["error"]
