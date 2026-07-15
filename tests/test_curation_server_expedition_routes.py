import json

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
