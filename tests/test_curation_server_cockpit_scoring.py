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


def test_cockpit_keeps_generation_available_without_focus_provenance():
    page = cockpit.render_html(
        expeditions=["demo", "other & more"],
        current_expedition="demo",
    )

    assert "No Focus provenance" in page
    assert "JSON.stringify({expedition, leg: 'cockpit'})" not in page


def test_cockpit_has_no_dark_prefers_color_scheme_variant():
    page = cockpit.render_html()

    assert "prefers-color-scheme: dark" not in page
    # The .topnav.cockpit-topnav rule must not be inlined in the static HTML: the page only
    # adds that class via JS at runtime, so the in-page stylesheet must not assume it exists.
    assert ".topnav.cockpit-topnav" not in page


def test_cockpit_render_html_uses_sulfur_proof_shell():
    """Task 5 (cockpit) render contract: the cockpit sits on the Sulfur Proof foundation, has
    no prefers-color-scheme: dark branch (Sulfur Proof is the only theme), embeds the shared
    header's context-switcher script, and ships a semantic <header> from the shared topnav."""
    page = cockpit.render_html()
    assert "--paper:#C3C5BA" in page
    assert "prefers-color-scheme: dark" not in page
    assert "shared-ui.js" in page
    assert "<header" in page


def test_cockpit_render_html_embeds_sulfur_depth_classes():
    """Task 5 brief structural rule: 'Cockpit becomes one ruled recipe with a recessed settings
    area and one full-width mounted payload-review strip.' SULFUR_CSS+CONTROL_CSS supply the
    recessed-readout and mounted-evidence classes; both must be embedded in the page so the
    structural rule can be expressed via class names on the recipe and payload-review panels."""
    page = cockpit.render_html()
    assert "recessed-readout" in page
    assert "mounted-evidence" in page


def test_cockpit_render_html_uses_sulfur_font_stack():
    """Task 5 brief: bundled Barlow Condensed / IBM Plex fonts replace the system-ui stack.
    The page's @font-face declarations come from SULFUR_FONT_CSS, and the body/inline font
    references must use the SULFUR_CSS font tokens, not 'system-ui' or 'Segoe UI'."""
    page = cockpit.render_html()
    assert "Barlow Condensed" in page
    assert "IBM Plex" in page
    assert "system-ui" not in page
    assert "Segoe UI" not in page


def test_cockpit_layout_drops_two_column_workbench_for_full_width_strip():
    """Task 5 brief: 'one ruled recipe with a recessed settings area and one full-width
    mounted payload-review strip.' The 2-column .workbench grid (recipe | evidence) is replaced
    with a stacked layout so the payload-review panel can span the full width instead of
    sharing a narrower right-hand column with the recipe."""
    page = cockpit.render_html()
    assert "minmax(500px,1.25fr) minmax(330px,.75fr)" not in page


def test_cockpit_recipe_panel_uses_recessed_readout_and_evidence_uses_mounted_evidence():
    """Task 5 brief structural rule: the recipe settings area carries .recessed-readout (inner
    edge bevel, no outer shadow) and the payload-review area carries .mounted-evidence (5px
    hard-shadow depth class from CONTROL_CSS). Locking the class assignment down so a future
    layout refactor can't silently swap the depth treatment."""
    page = cockpit.render_html()
    assert 'class="recessed-readout recipe' in page
    assert 'class="mounted-evidence evidence' in page


def test_cockpit_route_selects_cockpit_leg_and_passes_expeditions(monkeypatch):
    (config.EXPEDITIONS_DIR / "demo" / "legs" / "round1.json").write_text("{}")
    config.leg_dir("demo", "round1").mkdir(parents=True)
    (config.EXPEDITIONS_DIR / "other" / "legs").mkdir(parents=True)
    (config.EXPEDITIONS_DIR / "other" / "expedition.json").write_text("{}")
    cs._set_active_selection("demo", "round1")
    captured = {}

    def fake_render_html(*, expeditions, current_expedition, active_expedition, active_leg, running, focus):
        captured["expeditions"] = expeditions
        captured["current_expedition"] = current_expedition
        captured["active_expedition"] = active_expedition
        captured["active_leg"] = active_leg
        captured["running"] = running
        captured["focus"] = focus
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
        "active_leg": "round1",
        "running": None,
        "focus": None,
    }
    assert cs._active_selection == {"expedition": "demo", "leg": "round1"}


def test_focus_scoped_cockpit_get_does_not_change_active_leg(monkeypatch):
    round1 = config.EXPEDITIONS_DIR / "demo" / "legs" / "round1.json"
    round1.write_text("{}")
    config.leg_dir("demo", "round1").mkdir(parents=True)
    cs._set_active_selection("demo", "round1")
    focus = {
        "focus_id": "focus_11111111111111111111111111111111",
        "label": "Ink anchor",
        "revision": 1,
    }

    class Store:
        def get(self, scope, focus_id):
            assert (scope.expedition, scope.leg) == ("demo", "round1")
            assert focus_id == focus["focus_id"]
            return focus

    monkeypatch.setattr(cs.Handler, "_focus_store", lambda _handler: Store())
    before = config.ACTIVE_LEG_FILE.read_bytes()
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_address[1]}/cockpit.html"
            f"?expedition=demo&leg=round1&focus_id={focus['focus_id']}"
        ) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert config.ACTIVE_LEG_FILE.read_bytes() == before
    assert cs._active_selection["leg"] != "cockpit"


def test_cockpit_marks_run_trial_as_billable_action():
    page = cockpit.render_html()
    assert 'class="primary-action billable-action"' in page
    assert "Spends money" in page


def test_cockpit_safe_actions_do_not_have_billable_markup():
    page = cockpit.render_html()
    assert 'id="sendDraft"' in page
    assert 'class="generate striate"' in page
    assert 'class="use-suggestion"' in page
    assert 'class="autopilot-refresh"' in page
