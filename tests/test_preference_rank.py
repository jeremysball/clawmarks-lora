import json

import joblib
import numpy as np

from clawmarks.build.preference_rank import build_ranked_items, compute_data
from clawmarks.search import embed_cache, preference_pairwise_model


def _item(tag, tmp_path):
    return {"tag": tag, "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.5,
            "novelty": 0.5, "strength": 1.0, "cfg": 7.0, "file": str(tmp_path / f"{tag}.png")}


def test_build_ranked_items_sorts_descending_by_score(tmp_path):
    by_tag = {"a": _item("a", tmp_path), "b": _item("b", tmp_path)}
    items = build_ranked_items(by_tag, ["a", "b"], [0.2, 0.9], tmp_path)
    assert [it["tag"] for it in items] == ["b", "a"]
    assert items[0]["predicted_preference"] == 0.9


def test_build_ranked_items_respects_limit(tmp_path):
    by_tag = {f"t{i}": _item(f"t{i}", tmp_path) for i in range(10)}
    tags = list(by_tag.keys())
    scores = list(range(10))
    items = build_ranked_items(by_tag, tags, scores, tmp_path, limit=3)
    assert len(items) == 3


def test_build_ranked_items_skips_tags_missing_from_manifest(tmp_path):
    by_tag = {"a": _item("a", tmp_path)}
    items = build_ranked_items(by_tag, ["a", "ghost"], [0.5, 0.9], tmp_path)
    assert [it["tag"] for it in items] == ["a"]


class _FakeModel:
    def decision_function(self, embeddings):
        return np.arange(len(embeddings), dtype=float)


def test_compute_data_reads_the_out_dir_scoped_embeddings_cache(tmp_path):
    # Regression test: compute_data used to call embed_cache.EMBEDDINGS_FILE, a module
    # attribute removed by the expedition/leg migration, which raised AttributeError before it
    # ever got to reading the cache. Uncovered by build_ranked_items-only tests above.
    manifest = [_item("a", tmp_path), _item("b", tmp_path)]
    with open(tmp_path / "scored_manifest.json", "w") as f:
        json.dump(manifest, f)
    embed_cache.save_cache(embed_cache.embeddings_file(tmp_path), ["a", "b"], np.zeros((2, 4)))
    joblib.dump(_FakeModel(), preference_pairwise_model.model_file(tmp_path))

    data = compute_data(tmp_path)

    assert data["has_model"] is True
    assert {it["tag"] for it in data["items"]} == {"a", "b"}
