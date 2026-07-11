# tests/test_preference_settings.py
import json

from clawmarks.search import preference_settings


def test_load_returns_false_default_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    assert preference_settings.load() == {"use_predicted_preference": False}


def test_save_then_load_round_trips_true(tmp_path, monkeypatch):
    path = tmp_path / "preference_settings.json"
    monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", path)
    preference_settings.save(True)
    assert preference_settings.load() == {"use_predicted_preference": True}


def test_save_writes_atomically_no_tmp_file_left_behind(tmp_path, monkeypatch):
    path = tmp_path / "preference_settings.json"
    monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", path)
    preference_settings.save(True)
    assert not (tmp_path / "preference_settings.json.tmp").exists()
    assert json.loads(path.read_text()) == {"use_predicted_preference": True}


def test_save_false_then_load_round_trips_false(tmp_path, monkeypatch):
    path = tmp_path / "preference_settings.json"
    monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", path)
    preference_settings.save(True)
    preference_settings.save(False)
    assert preference_settings.load() == {"use_predicted_preference": False}
