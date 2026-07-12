import json
import threading
from http.server import HTTPServer
import urllib.error
import urllib.request

import numpy as np
import pytest

from clawmarks import curation_server as cs
from clawmarks.search import comparison_sampler, embed_cache, preference_pairwise_model


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

    for _ in range(comparison_sampler.MIN_COMPARISONS):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/compare", method="POST",
            data=json.dumps({"winner": "t0", "loser": "t1"}).encode(),
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
