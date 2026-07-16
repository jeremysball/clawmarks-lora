import json
import threading
import time
from http.server import HTTPServer
import urllib.request

import pytest
import torch

from clawmarks import config
from clawmarks import curation_server as cs
from clawmarks.build import cockpit


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", tmp_path / "state" / "active_leg.json")
    (config.EXPEDITIONS_DIR / "demo" / "legs").mkdir(parents=True)
    (config.EXPEDITIONS_DIR / "demo" / "expedition.json").write_text("{}")
    (config.EXPEDITIONS_DIR / "demo" / "legs" / "cockpit.json").write_text("{}")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None
    cs._set_active_selection("demo", "cockpit")
    cs._cockpit_scoring_state["model"] = None
    cs._cockpit_scoring_state["real_embs"] = None
    cs._cockpit_scoring_state["real_centroid"] = None
    yield


def test_score_cockpit_batch_pools_sibling_leg_images_as_exclusion(monkeypatch, tmp_path):
    from clawmarks.search import driver

    sibling_dir = config.leg_dir("demo", "round1")
    sibling_dir.mkdir(parents=True)
    (sibling_dir / "scored_manifest.json").write_text(json.dumps([{"tag": "r1_a", "file": "a.png"}]))

    captured = {}

    def fake_score_batch(model, real_embs, real_centroid, manifest_batch, prev_embs=None):
        captured["prev_embs"] = prev_embs
        return manifest_batch

    monkeypatch.setattr(driver, "score_batch", fake_score_batch)
    monkeypatch.setattr(
        cs, "_cockpit_scoring_context",
        lambda: (None, torch.zeros(1, 4), torch.zeros(4)),
    )
    monkeypatch.setattr(
        cs, "_sibling_leg_exclusion_embeddings",
        lambda expedition, leg, model: torch.ones(1, 4),
    )

    cs.score_cockpit_batch([], {"id": "t1", "mission": "freeform"}, "demo", "cockpit")

    assert captured["prev_embs"] is not None
    assert captured["prev_embs"].shape == (1, 4)


def test_sibling_leg_exclusion_embeddings_skips_missing_files(monkeypatch, tmp_path):
    from clawmarks.search import score_manifest

    sibling_dir = config.leg_dir("demo", "round1")
    sibling_dir.mkdir(parents=True)
    existing = sibling_dir / "existing.png"
    existing.write_bytes(b"png")
    missing = sibling_dir / "missing.png"
    (sibling_dir / "scored_manifest.json").write_text(json.dumps([
        {"tag": "existing", "file": str(existing)},
        {"tag": "missing", "file": str(missing)},
    ]))
    captured = {}

    def fake_embed_images(paths, model):
        captured["paths"] = paths
        captured["model"] = model
        return torch.ones(len(paths), 4)

    monkeypatch.setattr(score_manifest, "embed_images", fake_embed_images)
    model = object()

    embeddings = cs._sibling_leg_exclusion_embeddings("demo", "cockpit", model)

    assert captured == {"paths": [str(existing)], "model": model}
    assert embeddings.shape == (1, 4)


def test_cockpit_trial_keeps_launch_leg_when_active_leg_switches(monkeypatch):
    original_dir = config.leg_dir("demo", "cockpit")
    switched_dir = config.leg_dir("demo", "round1")
    original_dir.mkdir(parents=True)
    switched_dir.mkdir(parents=True)
    trial_id = "trial_race"
    trial = {"id": trial_id, "mission": "freeform", "status": "queued", "error": None}
    (original_dir / "cockpit_queue.json").write_text(json.dumps({trial_id: trial}))
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setattr(cs, "runpod_balance", lambda api_key: 10.0)
    monkeypatch.setattr(cs, "build_generation_jobs", lambda trial: [{
        "tag": "race_image", "prompt": "prompt", "seed": 1, "strength": 1.0,
        "cfg": 7.5, "steps": 28, "sampler": "ddim", "negative": "",
    }])
    monkeypatch.setattr(cs, "comfy_post", lambda path, workflow, api_key: {"id": "job-1"})
    generation_polled = threading.Event()
    finish_generation = threading.Event()

    def fake_comfy_get(path, api_key):
        generation_polled.set()
        assert finish_generation.wait(timeout=2)
        return {"status": "COMPLETED", "output": {"images": [{"data": "aW1hZ2U="}]}}

    monkeypatch.setattr(cs, "comfy_get", fake_comfy_get)
    monkeypatch.setattr(cs, "score_cockpit_batch", lambda results, trial, *args: results)

    def fake_generate_thumbnail(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"thumb")

    monkeypatch.setattr(cs, "generate_thumbnail", fake_generate_thumbnail)
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/cockpit/queue/{trial_id}/run",
            method="POST", data=b"{}", headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as response:
            assert json.load(response)["status"] == "running"
        assert generation_polled.wait(timeout=2)

        cs._set_active_selection("demo", "round1")
        finish_generation.set()

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            queue = json.loads((original_dir / "cockpit_queue.json").read_text())
            if queue[trial_id]["status"] == "completed":
                break
            time.sleep(0.01)
        assert queue[trial_id]["status"] == "completed"
    finally:
        finish_generation.set()
        server.shutdown()
        server_thread.join(timeout=2)

    assert (original_dir / "race_image.png").read_bytes() == b"image"
    assert (original_dir / "thumbs" / "race_image.jpg").read_bytes() == b"thumb"
    assert json.loads((original_dir / "scored_manifest.json").read_text())[0]["tag"] == "race_image"
    assert not (switched_dir / "race_image.png").exists()
    assert not (switched_dir / "scored_manifest.json").exists()
    assert not (switched_dir / "cockpit_queue.json").exists()


def test_cockpit_expedition_selector_switches_to_cockpit_leg():
    page = cockpit.render_html(
        expeditions=["demo", "other & more"],
        current_expedition="demo",
    )

    assert '<option value="demo" selected>demo</option>' in page
    assert '<option value="other &amp; more">other &amp; more</option>' in page
    assert "fetch('/api/active-leg'" in page
    assert "JSON.stringify({expedition, leg: 'cockpit'})" in page


def test_cockpit_has_a_dark_prefers_color_scheme_variant():
    page = cockpit.render_html()

    assert "@media (prefers-color-scheme: dark)" in page
    assert "--paper:#0b0b0d" in page
    assert ".topnav.cockpit-topnav" not in page


def test_cockpit_route_selects_cockpit_leg_and_passes_expeditions(monkeypatch):
    (config.EXPEDITIONS_DIR / "other" / "legs").mkdir(parents=True)
    (config.EXPEDITIONS_DIR / "other" / "expedition.json").write_text("{}")
    cs._set_active_selection("demo", "round1")
    captured = {}

    def fake_render_html(*, expeditions, current_expedition, active_expedition, active_leg, running):
        captured["expeditions"] = expeditions
        captured["current_expedition"] = current_expedition
        captured["active_expedition"] = active_expedition
        captured["active_leg"] = active_leg
        captured["running"] = running
        return "cockpit page"

    monkeypatch.setattr(cockpit, "render_html", fake_render_html)
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/cockpit.html"
        ) as response:
            assert response.read() == b"cockpit page"
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert captured == {
        "expeditions": ["demo", "other"],
        "current_expedition": "demo",
        "active_expedition": "demo",
        "active_leg": "cockpit",
        "running": None,
    }
    assert cs._active_selection == {"expedition": "demo", "leg": "cockpit"}
