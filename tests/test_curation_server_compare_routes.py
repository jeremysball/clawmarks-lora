import itertools
import json
import threading
import time
from http.server import HTTPServer
import urllib.error
import urllib.request

import numpy as np
import pytest

from clawmarks import curation_server as cs
from clawmarks.search import comparison_sampler, embed_cache, preference_pairwise_model


def _post_json(url, payload=None):
    req = urllib.request.Request(
        url, method="POST", data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    monkeypatch.setattr(cs, "COMPARISONS_FILE", str(tmp_path / "user_comparisons.json"))
    monkeypatch.setattr(cs.preference_pairwise_model, "MODEL_FILE", tmp_path / "preference_pairwise_model.joblib")
    monkeypatch.setattr(cs.preference_pairwise_model, "MODEL_META_FILE", tmp_path / "preference_pairwise_model_meta.json")
    manifest = [
        {"tag": f"t{i}", "prompt_name": "p", "prompt_type": "style", "centroid_sim": i / 20,
         "novelty": 1 - i / 20, "strength": 1.0, "cfg": 7.0, "file": f"{i}.png"}
        for i in range(20)
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown()
    thread.join(timeout=2)


@pytest.fixture
def threaded_running_server(tmp_path, monkeypatch):
    """Same setup as running_server, but backed by cs.ThreadingHTTPServer (what main() actually
    serves with) instead of the plain single-threaded HTTPServer: concurrency tests need real
    concurrent request handling, which the shared fixture's HTTPServer serializes away."""
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    monkeypatch.setattr(cs, "COMPARISONS_FILE", str(tmp_path / "user_comparisons.json"))
    monkeypatch.setattr(cs.preference_pairwise_model, "MODEL_FILE", tmp_path / "preference_pairwise_model.joblib")
    monkeypatch.setattr(cs.preference_pairwise_model, "MODEL_META_FILE", tmp_path / "preference_pairwise_model_meta.json")
    manifest = [
        {"tag": f"t{i}", "prompt_name": "p", "prompt_type": "style", "centroid_sim": i / 20,
         "novelty": 1 - i / 20, "strength": 1.0, "cfg": 7.0, "file": f"{i}.png"}
        for i in range(20)
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    server = cs.ThreadingHTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown()
    thread.join(timeout=2)


def test_compare_next_returns_two_distinct_images(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/compare/next") as resp:
        data = json.loads(resp.read().decode())
    assert data["img1"]["tag"] != data["img2"]["tag"]
    assert "faith" in data["img1"] and "novelty" in data["img1"]


def test_compare_next_returns_done_with_fewer_than_two_images(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    (tmp_path / "scored_manifest.json").write_text(json.dumps([
        {"tag": "only", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.5,
         "novelty": 0.5, "strength": 1.0, "cfg": 7.0, "file": "only.png"},
    ]))
    cs._manifest_cache["manifest"] = None
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/compare/next") as resp:
        data = json.loads(resp.read().decode())
    assert data == {"done": True}


def test_post_compare_appends_a_comparison_record(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/compare", method="POST",
        data=json.dumps({"winner": "t0", "loser": "t1"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    assert data["ok"] is True
    assert data["count"] == 1
    comparisons = json.loads((tmp_path / "user_comparisons.json").read_text())
    assert len(comparisons) == 1
    assert comparisons[0]["winner"] == "t0"
    assert comparisons[0]["loser"] == "t1"
    assert "compared_at" in comparisons[0]


def test_post_compare_retrains_and_caches_model_at_retrain_interval(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    embeddings_path = tmp_path / "embeddings.npz"
    tags = [f"t{i}" for i in range(20)]
    embeddings = np.random.RandomState(0).normal(size=(20, 2)).astype(np.float32)
    embed_cache.save_cache(embeddings_path, tags, embeddings)
    monkeypatch.setattr(embed_cache, "EMBEDDINGS_FILE", embeddings_path)
    monkeypatch.setitem(cs._pairwise_model_cache, "model", None)

    # Distinct pairs, not the same pair repeated: repeated judgments on one pair now consolidate
    # into a single piece of evidence (issue #13), so they'd never clear MIN_COMPARISONS usable
    # pairs and this test would never observe a retrain.
    pairs = list(itertools.combinations(tags, 2))[:comparison_sampler.MIN_COMPARISONS]
    for winner, loser in pairs:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/compare", method="POST",
            data=json.dumps({"winner": winner, "loser": loser}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())

    assert data["count"] == comparison_sampler.MIN_COMPARISONS
    assert cs._pairwise_model_cache["model"] is not None
    assert preference_pairwise_model.MODEL_FILE.exists()
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/compare/next") as resp:
        next_pair = json.loads(resp.read().decode())
    assert next_pair["img1"]["tag"] != next_pair["img2"]["tag"]


def test_post_compare_does_not_hold_lock_during_auto_retrain(threaded_running_server, monkeypatch):
    """The auto-retrain triggered by hitting a RETRAIN_EVERY interval (_maybe_retrain_pairwise_model,
    called from /api/compare) must not hold _lock across the model fit either -- same reasoning as
    the manual /api/preference_retrain endpoint's fix. Assert a concurrent /api/compare request
    completes quickly while a slow auto-retrain triggered by another request is still in flight."""
    server, tmp_path = threaded_running_server
    port = server.server_address[1]
    tags = [f"t{i}" for i in range(20)]
    embeddings = np.random.RandomState(0).normal(size=(20, 2)).astype(np.float32)
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)
    monkeypatch.setattr(embed_cache, "EMBEDDINGS_FILE", tmp_path / "embeddings.npz")
    monkeypatch.setitem(cs._pairwise_model_cache, "model", None)

    # Seed MIN_COMPARISONS - 1 distinct-pair comparisons directly on disk so the next /api/compare
    # POST lands exactly on the retrain interval.
    pairs = list(itertools.combinations(tags, 2))
    seed_pairs = pairs[:comparison_sampler.MIN_COMPARISONS - 1]
    (tmp_path / "user_comparisons.json").write_text(json.dumps([
        {"winner": winner, "loser": loser, "compared_at": "2020-01-01T00:00:00+00:00"}
        for winner, loser in seed_pairs
    ]))
    trigger_winner, trigger_loser = pairs[comparison_sampler.MIN_COMPARISONS - 1]
    concurrent_winner, concurrent_loser = pairs[comparison_sampler.MIN_COMPARISONS]

    training_started = threading.Event()
    release_training = threading.Event()
    retrain_errors = []

    def slow_train_and_save(comparisons):
        training_started.set()
        # Long enough that a blocked concurrent compare would clearly time out the assertion
        # below; release_training.set() (in the test's finally) cuts this short once the
        # concurrent compare has already returned, so the happy path doesn't wait the full 30s.
        release_training.wait(timeout=30)
        return None

    monkeypatch.setattr(cs.preference_pairwise_model, "train_and_save", slow_train_and_save)

    def trigger_retrain():
        try:
            _post_json(f"http://127.0.0.1:{port}/api/compare",
                       {"winner": trigger_winner, "loser": trigger_loser})
        except Exception as e:
            retrain_errors.append(e)

    retrain_thread = threading.Thread(target=trigger_retrain)
    retrain_thread.start()
    assert training_started.wait(timeout=5), "auto-retrain never started"

    try:
        start = time.monotonic()
        resp = _post_json(f"http://127.0.0.1:{port}/api/compare",
                           {"winner": concurrent_winner, "loser": concurrent_loser})
        elapsed = time.monotonic() - start
        assert resp["ok"] is True
        assert elapsed < 5, (
            f"/api/compare took {elapsed:.1f}s, suggesting it waited on _lock held during auto-retrain"
        )
    finally:
        release_training.set()
        retrain_thread.join(timeout=5)
    assert retrain_errors == []


def test_post_compare_rejects_missing_fields(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/compare", method="POST",
        data=json.dumps({"winner": "t0"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_post_compare_rejects_self_comparison(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/compare", method="POST",
        data=json.dumps({"winner": "t0", "loser": "t0"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400
    assert not (tmp_path / "user_comparisons.json").exists()


def test_compare_next_falls_back_to_random_when_model_has_no_embedded_tags(running_server, monkeypatch):
    # A trained model is cached, but the embedding cache covers only tags absent from the current
    # manifest (e.g. embeddings were rebuilt with new tags). The uncertainty path can score
    # nothing; the response must still offer a random pair rather than a false {"done": True}.
    server, tmp_path = running_server
    port = server.server_address[1]
    embeddings_path = tmp_path / "embeddings.npz"
    stale_tags = [f"gone{i}" for i in range(5)]
    embeddings = np.random.RandomState(0).normal(size=(5, 2)).astype(np.float32)
    embed_cache.save_cache(embeddings_path, stale_tags, embeddings)
    monkeypatch.setattr(embed_cache, "EMBEDDINGS_FILE", embeddings_path)
    monkeypatch.setitem(cs._pairwise_model_cache, "model", object())

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/compare/next") as resp:
        data = json.loads(resp.read().decode())
    assert "done" not in data
    assert data["img1"]["tag"] != data["img2"]["tag"]


def test_compare_html_route_serves_page(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/compare.html") as resp:
        body = resp.read().decode()
    assert "CLAWMARKS compare" in body


def test_rate_routes_no_longer_exist(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/rate.html")
    assert exc_info.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/rate/next")
    assert exc_info.value.code == 404
