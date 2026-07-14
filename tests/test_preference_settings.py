# tests/test_preference_settings.py
import json

from clawmarks.search import preference_settings


def test_load_returns_false_default_when_file_missing(tmp_path):
    assert preference_settings.load(tmp_path) == {"use_predicted_preference": False}


def test_save_then_load_round_trips_true(tmp_path):
    path = tmp_path / "preference_settings.json"
    preference_settings.save(True, tmp_path)
    assert preference_settings.load(tmp_path) == {"use_predicted_preference": True}


def test_save_writes_atomically_no_tmp_file_left_behind(tmp_path):
    path = tmp_path / "preference_settings.json"
    preference_settings.save(True, tmp_path)
    assert not (tmp_path / "preference_settings.json.tmp").exists()
    assert json.loads(path.read_text()) == {"use_predicted_preference": True}


def test_save_false_then_load_round_trips_false(tmp_path):
    preference_settings.save(True, tmp_path)
    preference_settings.save(False, tmp_path)
    assert preference_settings.load(tmp_path) == {"use_predicted_preference": False}
