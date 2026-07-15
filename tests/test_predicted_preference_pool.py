import joblib
import numpy as np

from clawmarks.search import embed_cache
from clawmarks.search import driver


class _FakeEmbedModel:
    def __call__(self, pixel_values):
        raise AssertionError("embedding model should not be called in these tests")


class _FakePreferenceModel:
    def decision_function(self, embeddings):
        return embeddings[:, 0]


def test_predicted_preference_pool_returns_empty_without_a_trained_model(tmp_path):
    manifest = [{"tag": "a", "file": str(tmp_path / "a.png")}]
    result = driver._predicted_preference_pool(
        manifest, tmp_path / "missing.joblib", _FakeEmbedModel(), tmp_path
    )
    assert result == []


def test_predicted_preference_pool_returns_empty_for_empty_manifest(tmp_path):
    (tmp_path / "some_model.joblib").write_bytes(b"not a real model, never opened")
    result = driver._predicted_preference_pool(
        [], tmp_path / "some_model.joblib", _FakeEmbedModel(), tmp_path
    )
    assert result == []


def test_predicted_preference_pool_uses_the_given_output_directory(tmp_path):
    out_dir = tmp_path / "leg"
    out_dir.mkdir()
    manifest = [
        {"tag": "low", "file": str(out_dir / "low.png")},
        {"tag": "high", "file": str(out_dir / "high.png")},
    ]
    embed_cache.save_cache(
        embed_cache.embeddings_file(out_dir),
        ["low", "high"],
        np.array([[0.1, 1.0], [0.9, 0.0]], dtype=np.float32),
    )
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "preference_pairwise_model.joblib"
    joblib.dump(_FakePreferenceModel(), model_path)

    result = driver._predicted_preference_pool(
        manifest, model_path, _FakeEmbedModel(), out_dir
    )

    assert [item["tag"] for item in result] == ["high", "low"]
