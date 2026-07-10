# tests/test_yes_rated_images.py
import json

from clawmarks.search import driver


def test_load_yes_rated_images_joins_ratings_against_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SWEEP_DIR", tmp_path)
    manifest = [
        {"tag": "a", "prompt_name": "p", "prompt": "trentbuckle style, a", "strength": 1.0,
         "cfg": 7.0, "centroid_sim": 0.5, "novelty": 0.5, "file": "a.png"},
        {"tag": "b", "prompt_name": "p", "prompt": "trentbuckle style, b", "strength": 1.0,
         "cfg": 7.0, "centroid_sim": 0.5, "novelty": 0.5, "file": "b.png"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "user_ratings.json").write_text(json.dumps({
        "a": {"label": "yes", "rated_at": "t0"},
        "b": {"label": "no", "rated_at": "t0"},
    }))
    result = driver._load_yes_rated_images()
    assert [m["tag"] for m in result] == ["a"]


def test_load_yes_rated_images_returns_empty_without_files(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SWEEP_DIR", tmp_path)
    assert driver._load_yes_rated_images() == []
