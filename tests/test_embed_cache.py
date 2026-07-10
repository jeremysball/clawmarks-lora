# tests/test_embed_cache.py
import numpy as np
import torch
from PIL import Image

from clawmarks.search import embed_cache


class FakeOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class FakeModel:
    """Deterministic per-image 'embedding' derived from mean pixel value, so tests exercise
    embed_cache's own logic (batching, ordering, caching) without loading the real (slow,
    network-fetched) DINOv2 model."""
    def __call__(self, pixel_values):
        means = pixel_values.mean(dim=(1, 2, 3))
        feats = torch.stack([means, -means], dim=1)
        return FakeOutput(feats)


def _write_image(path, color):
    Image.new("RGB", (32, 32), color=color).save(path)


def test_embed_paths_returns_one_normalized_row_per_path(tmp_path):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    _write_image(p1, (10, 10, 10))
    _write_image(p2, (200, 200, 200))
    embs = embed_cache.embed_paths([str(p1), str(p2)], FakeModel())
    assert embs.shape == (2, 2)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_save_and_load_cache_round_trips(tmp_path):
    path = tmp_path / "embeddings.npz"
    tags = ["a", "b"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    embed_cache.save_cache(path, tags, embeddings)
    loaded_tags, loaded_embeddings = embed_cache.load_cache(path)
    assert loaded_tags == tags
    assert np.allclose(loaded_embeddings, embeddings)


def test_load_cache_missing_file_returns_empty(tmp_path):
    tags, embeddings = embed_cache.load_cache(tmp_path / "missing.npz")
    assert tags == []
    assert embeddings.shape == (0, 0)


def test_missing_tags_returns_manifest_tags_not_in_cache():
    assert embed_cache.missing_tags(["a", "b", "c"], ["a", "c"]) == ["b"]


def test_sync_adds_only_missing_tags_and_persists(tmp_path):
    manifest = [{"tag": "a"}, {"tag": "b"}]
    _write_image(tmp_path / "a.png", (10, 10, 10))
    _write_image(tmp_path / "b.png", (200, 200, 200))
    cache_path = tmp_path / "embeddings.npz"

    def image_path_for(tag):
        return str(tmp_path / f"{tag}.png")

    tags, embeddings = embed_cache.sync(manifest, cache_path, FakeModel(), image_path_for)
    assert tags == ["a", "b"]
    assert embeddings.shape == (2, 2)

    manifest.append({"tag": "c"})
    _write_image(tmp_path / "c.png", (100, 50, 25))
    tags2, embeddings2 = embed_cache.sync(manifest, cache_path, FakeModel(), image_path_for)
    assert tags2 == ["a", "b", "c"]
    assert np.allclose(embeddings2[:2], embeddings)


def test_sync_raises_on_missing_image_file(tmp_path):
    manifest = [{"tag": "missing"}]
    cache_path = tmp_path / "embeddings.npz"

    def image_path_for(tag):
        return str(tmp_path / "does_not_exist.png")

    try:
        embed_cache.sync(manifest, cache_path, FakeModel(), image_path_for)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
