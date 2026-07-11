import json

from clawmarks.search import driver


def test_load_favorited_images_returns_favorite_records(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SWEEP_DIR", tmp_path)
    favorites = {
        "a": {"tag": "a", "prompt_name": "p", "prompt": "trentbuckle style, a", "strength": 1.0,
              "cfg": 7.0, "faith": 0.5, "novelty": 0.5, "favorited_at": "t0"},
    }
    (tmp_path / "user_favorites.json").write_text(json.dumps(favorites))
    result = driver._load_favorited_images()
    assert [m["tag"] for m in result] == ["a"]
    assert result[0]["prompt"] == "trentbuckle style, a"


def test_load_favorited_images_returns_empty_without_file(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SWEEP_DIR", tmp_path)
    assert driver._load_favorited_images() == []
