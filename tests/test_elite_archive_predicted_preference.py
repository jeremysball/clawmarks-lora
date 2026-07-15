import json

import joblib
import numpy as np

from clawmarks.build.elite_archive import build_item_summary, compute_data, elite_sort_key
from clawmarks.search import embed_cache, preference_pairwise_model


def _item(tag, tmp_path, novelty=0.5):
    return {"tag": tag, "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.5,
            "novelty": novelty, "strength": 1.0, "cfg": 7.0, "file": str(tmp_path / f"{tag}.png")}


def test_elite_sort_key_falls_back_to_novelty_when_no_predicted_scores(tmp_path):
    m = _item("a", tmp_path, novelty=0.7)
    assert elite_sort_key(m, {}) == -0.7


def test_elite_sort_key_prefers_predicted_preference_when_available(tmp_path):
    m = _item("a", tmp_path, novelty=0.1)
    predicted_scores = {"a": 0.9}
    assert elite_sort_key(m, predicted_scores) == -0.9


def test_elite_sort_key_treats_missing_score_as_neutral_when_scores_exist_for_others(tmp_path):
    m = _item("a", tmp_path, novelty=0.1)
    predicted_scores = {"other_tag": 0.9}
    assert elite_sort_key(m, predicted_scores) == 0.0


def test_build_item_summary_omits_predicted_preference_when_absent(tmp_path):
    m = _item("a", tmp_path)
    summary = build_item_summary(m, tmp_path, {})
    assert "predicted_preference" not in summary


def test_build_item_summary_includes_predicted_preference_when_present(tmp_path):
    m = _item("a", tmp_path)
    summary = build_item_summary(m, tmp_path, {"a": 0.8234567})
    assert summary["predicted_preference"] == 0.8235


def test_sorting_a_cell_with_predicted_scores_puts_highest_score_first(tmp_path):
    items = [_item("a", tmp_path, novelty=0.9), _item("b", tmp_path, novelty=0.1)]
    predicted_scores = {"a": 0.2, "b": 0.95}
    ranked = sorted(items, key=lambda m: elite_sort_key(m, predicted_scores))
    assert [m["tag"] for m in ranked] == ["b", "a"]


class _FakeModel:
    def decision_function(self, embeddings):
        return np.arange(len(embeddings), dtype=float)


def test_compute_data_reads_the_out_dir_scoped_embeddings_cache_when_predicted_preference_on(tmp_path):
    # Regression test: compute_data used to call embed_cache.EMBEDDINGS_FILE, a module
    # attribute removed by the expedition/leg migration, which raised AttributeError as soon as
    # use_predicted_preference was on and a trained model existed. Uncovered by the pure
    # elite_sort_key/build_item_summary tests above.
    manifest = [_item("a", tmp_path, novelty=0.2), _item("b", tmp_path, novelty=0.8)]
    with open(tmp_path / "scored_manifest.json", "w") as f:
        json.dump(manifest, f)
    embed_cache.save_cache(embed_cache.embeddings_file(tmp_path), ["a", "b"], np.zeros((2, 4)))
    joblib.dump(_FakeModel(), preference_pairwise_model.model_file(tmp_path))

    data = compute_data(tmp_path, use_predicted_preference=True)

    all_tags = {it["tag"] for cell in data["cells"] for it in cell["items"]}
    assert all_tags == {"a", "b"}
