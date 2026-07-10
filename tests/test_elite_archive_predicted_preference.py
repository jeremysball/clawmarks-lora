from clawmarks.build.elite_archive import build_item_summary, elite_sort_key


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
