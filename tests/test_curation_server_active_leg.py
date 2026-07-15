import json

import pytest

from clawmarks import curation_server as cs
from clawmarks import config


@pytest.fixture(autouse=True)
def _reset_active_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", tmp_path / "active_leg.json")
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None
    yield


def test_active_out_dir_is_none_before_any_selection():
    assert cs._active_out_dir() is None


def test_set_active_selection_persists_and_resolves(tmp_path):
    (config.EXPEDITIONS_DIR / "demo").mkdir(parents=True)
    (config.EXPEDITIONS_DIR / "demo" / "expedition.json").write_text("{}")

    cs._set_active_selection("demo", "leg1")

    assert cs._active_out_dir() == config.leg_dir("demo", "leg1")
    assert json.loads(config.ACTIVE_LEG_FILE.read_text()) == {"expedition": "demo", "leg": "leg1"}


def test_set_active_selection_rejects_unknown_expedition():
    with pytest.raises(ValueError, match="unknown expedition"):
        cs._set_active_selection("does_not_exist", "leg1")


def test_load_active_selection_restores_from_disk(tmp_path):
    (config.EXPEDITIONS_DIR / "demo").mkdir(parents=True)
    (config.EXPEDITIONS_DIR / "demo" / "expedition.json").write_text("{}")
    config.ACTIVE_LEG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.ACTIVE_LEG_FILE.write_text(json.dumps({"expedition": "demo", "leg": "leg1"}))

    cs._load_active_selection()

    assert cs._active_out_dir() == config.leg_dir("demo", "leg1")
