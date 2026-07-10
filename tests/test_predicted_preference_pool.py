from clawmarks.search import driver


class _FakeEmbedModel:
    def __call__(self, pixel_values):
        raise AssertionError("should not be called when no trained model exists")


def test_predicted_preference_pool_returns_empty_without_a_trained_model(tmp_path):
    manifest = [{"tag": "a", "file": str(tmp_path / "a.png")}]
    result = driver._predicted_preference_pool(manifest, tmp_path / "missing.joblib", _FakeEmbedModel())
    assert result == []


def test_predicted_preference_pool_returns_empty_for_empty_manifest(tmp_path):
    (tmp_path / "some_model.joblib").write_bytes(b"not a real model, never opened")
    result = driver._predicted_preference_pool([], tmp_path / "some_model.joblib", _FakeEmbedModel())
    assert result == []
