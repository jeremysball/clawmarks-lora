import json
import threading
import urllib.request
from http.server import HTTPServer

import pytest

from clawmarks import curation_server as cs
from clawmarks import config


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", tmp_path / "state" / "active_leg.json")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None
    yield


@pytest.fixture
def running_server_with_leg():
    """Stand up a real Handler instance on a free port with a leg selected that has no scored
    manifest. Used by test_status_page_shows_selected_leg_with_no_data to assert the root page
    distinguishes 'no leg selected' from 'leg selected, no scored data yet'."""
    cs._create_expedition({"name": "uncanny_frontier", "textures": [], "fallback_subjects": []})
    cs._set_active_selection("uncanny_frontier", "cockpit")
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=2)


def test_list_expeditions_empty_when_none_exist():
    assert cs._list_expeditions() == []


def test_create_expedition_writes_config_and_scaffolds_cockpit_leg():
    payload = {
        "name": "demo", "trigger_word": "trentbuckle style, ",
        "negative_prompt": "low quality, blurry, watermark",
        "textures": ["tex-a"], "fallback_subjects": ["subj-a"],
        "budget_usd_cap": 5.0, "budget_safety_margin": 0.5,
        "gen_batch_size": 20, "explore_fraction": 0.5, "max_generations": 100,
    }
    result = cs._create_expedition(payload)

    assert result == {"ok": True, "name": "demo"}
    expedition_file = config.EXPEDITIONS_DIR / "demo" / "expedition.json"
    assert json.loads(expedition_file.read_text())["trigger_word"] == "trentbuckle style, "
    cockpit_leg_file = config.EXPEDITIONS_DIR / "demo" / "legs" / "cockpit.json"
    assert cockpit_leg_file.exists()
    assert config.leg_dir("demo", "cockpit").exists()


def test_create_expedition_rejects_a_name_that_already_exists():
    payload = {"name": "demo", "textures": [], "fallback_subjects": []}
    cs._create_expedition(payload)

    with pytest.raises(ValueError, match="already exists"):
        cs._create_expedition(payload)


def test_list_expeditions_reports_every_leg():
    cs._create_expedition({"name": "demo", "textures": [], "fallback_subjects": []})
    (config.EXPEDITIONS_DIR / "demo" / "legs" / "round1.json").write_text("{}")

    expeditions = cs._list_expeditions()

    assert len(expeditions) == 1
    assert expeditions[0]["name"] == "demo"
    assert set(expeditions[0]["legs"]) == {"cockpit", "round1"}


def test_status_page_shows_selected_leg_with_no_data(running_server_with_leg):
    port = running_server_with_leg.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
        body = resp.read()

    assert b"no expedition/leg selected" not in body.lower()
    assert b"uncanny_frontier" in body
    assert b"cockpit" in body
    assert b"no scored" in body.lower() or b"no search data" in body.lower()


@pytest.fixture
def running_server_with_leg_and_data(tmp_path):
    """Stand up a real Handler with a leg selected that has at least one scored manifest
    image present on disk, so the root page renders the 'has data' branch
    (_status_page_data_body). Used by test_status_page_data_branch_surfaces_comparison_count."""
    cs._create_expedition({"name": "uncanny_frontier", "textures": [], "fallback_subjects": []})
    cs._set_active_selection("uncanny_frontier", "cockpit")
    leg_dir = config.leg_dir("uncanny_frontier", "cockpit")
    leg_dir.mkdir(parents=True, exist_ok=True)
    image_path = leg_dir / "gen1_a.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    manifest = [{"tag": "gen1_a", "file": str(image_path)}]
    (leg_dir / "scored_manifest.json").write_text(json.dumps(manifest))
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=2)


def test_status_page_data_branch_surfaces_comparison_count(running_server_with_leg_and_data):
    port = running_server_with_leg_and_data.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
        body = resp.read().decode()

    assert 'id="cmpStat"' in body
    assert "fetch('/api/preference_status')" in body
