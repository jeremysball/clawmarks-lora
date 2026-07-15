import json

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

    cs.score_cockpit_batch([], {"id": "t1", "mission": "freeform"})

    assert captured["prev_embs"] is not None
    assert captured["prev_embs"].shape == (1, 4)


def test_cockpit_expedition_selector_switches_to_cockpit_leg():
    page = cockpit.render_html(
        expeditions=["demo", "other & more"],
        current_expedition="demo",
    )

    assert '<option value="demo" selected>demo</option>' in page
    assert '<option value="other &amp; more">other &amp; more</option>' in page
    assert "fetch('/api/active-leg'" in page
    assert "JSON.stringify({expedition, leg: 'cockpit'})" in page
