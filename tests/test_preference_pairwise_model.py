import numpy as np

from clawmarks.search import embed_cache
from clawmarks.search import preference_pairwise_model as ppm


def test_build_training_set_mirrors_each_comparison_into_two_rows():
    tags = ["a", "b", "c"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    comparisons = [{"winner": "a", "loser": "b", "compared_at": "t0"}]
    X, y = ppm.build_training_set(tags, embeddings, comparisons)
    assert X.shape == (2, 2)
    assert list(y) == [1, 0]
    assert np.allclose(X[0], [1.0, -1.0])
    assert np.allclose(X[1], [-1.0, 1.0])


def test_build_training_set_skips_comparisons_with_unknown_tags():
    tags = ["a"]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    comparisons = [{"winner": "a", "loser": "missing", "compared_at": "t0"}]
    X, y = ppm.build_training_set(tags, embeddings, comparisons)
    assert X.shape == (0, 0)
    assert len(y) == 0


def test_build_training_set_handles_multiple_comparisons():
    tags = ["a", "b", "c", "d"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    comparisons = [
        {"winner": "a", "loser": "b", "compared_at": "t0"},
        {"winner": "c", "loser": "d", "compared_at": "t1"},
    ]
    X, y = ppm.build_training_set(tags, embeddings, comparisons)
    assert X.shape == (4, 2)
    assert list(y) == [1, 1, 0, 0]


def test_train_and_score_orders_a_clearly_preferred_cluster_above_another():
    rng = np.random.RandomState(0)
    winners = rng.normal(loc=5.0, scale=0.1, size=(20, 2))
    losers = rng.normal(loc=-5.0, scale=0.1, size=(20, 2))
    diffs = (winners - losers).astype(np.float32)
    X = np.concatenate([diffs, -diffs])
    y = np.concatenate([np.ones(20), np.zeros(20)])
    model = ppm.train(X, y)
    scores = ppm.score(model, np.array([[5.0, 0.0], [-5.0, 0.0]], dtype=np.float32))
    assert scores[0] > scores[1]


def test_cross_validate_returns_a_valid_accuracy_using_leave_one_out_below_min_comparisons():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(10, 2)).astype(np.float32)
    y = np.array([0, 1] * 5)
    acc = ppm.cross_validate(X, y)
    assert 0.0 <= acc <= 1.0


def test_train_and_save_returns_none_below_min_comparisons(tmp_path, monkeypatch):
    monkeypatch.setattr(ppm, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(ppm.embed_cache, "EMBEDDINGS_FILE", tmp_path / "embeddings.npz")
    comparisons = [{"winner": "a", "loser": "b", "compared_at": "t0"}] * 10
    assert ppm.train_and_save(comparisons) is None


def test_train_and_save_writes_model_and_meta_on_success(tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    tags = [f"t{i}" for i in range(120)]
    embeddings = rng.normal(size=(120, 2)).astype(np.float32)
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)

    comparisons = [
        {"winner": tags[i], "loser": tags[i + 1], "compared_at": "t"}
        for i in range(0, 100, 2)
    ]

    monkeypatch.setattr(ppm, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(ppm.embed_cache, "EMBEDDINGS_FILE", tmp_path / "embeddings.npz")
    monkeypatch.setattr(ppm, "MODEL_FILE", tmp_path / "preference_pairwise_model.joblib")
    monkeypatch.setattr(ppm, "MODEL_META_FILE", tmp_path / "preference_pairwise_model_meta.json")

    result = ppm.train_and_save(comparisons)
    assert result is not None
    assert 0.0 <= result["cv_accuracy"] <= 1.0
    assert result["n_comparisons"] == 50
    assert (tmp_path / "preference_pairwise_model.joblib").exists()

    import json
    meta = json.loads((tmp_path / "preference_pairwise_model_meta.json").read_text())
    assert meta["n_comparisons"] == 50
    assert "trained_at" in meta
    assert meta["baseline_accuracy"] == 0.5
    assert 0.0 <= meta["p_value"] <= 1.0
    assert meta["n_permutations"] == ppm.N_PERMUTATIONS

    tags_arr, embeddings_arr = embed_cache.load_cache(tmp_path / "embeddings.npz")
    expected_fingerprint = ppm.comparisons_fingerprint(tags_arr, embeddings_arr, comparisons)
    assert meta["comparisons_fingerprint"] == expected_fingerprint


def test_main_refuses_without_comparisons_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ppm, "SWEEP_DIR", tmp_path)
    rc = ppm.main([])
    assert rc == 1


def test_significance_reports_low_p_value_for_separable_data():
    rng = np.random.RandomState(0)
    winners = rng.normal(loc=5.0, scale=0.1, size=(12, 2))
    losers = rng.normal(loc=-5.0, scale=0.1, size=(12, 2))
    diffs = (winners - losers).astype(np.float32)
    X = np.concatenate([diffs, -diffs])
    y = np.concatenate([np.ones(12), np.zeros(12)]).astype(np.int64)

    stats = ppm.significance(X, y, n_permutations=20, random_state=0)

    assert stats["baseline_accuracy"] == 0.5
    assert stats["n_permutations"] == 20
    assert stats["p_value"] < 0.1


def test_significance_reports_high_p_value_for_shuffled_labels():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(24, 2)).astype(np.float32)
    y = np.array([0, 1] * 12, dtype=np.int64)
    rng.shuffle(y)

    stats = ppm.significance(X, y, n_permutations=20, random_state=0)

    assert stats["baseline_accuracy"] == 0.5
    assert stats["n_permutations"] == 20
    assert stats["p_value"] > 0.3


def test_comparisons_fingerprint_is_stable_regardless_of_comparison_order():
    tags = ["a", "b", "c", "d"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    forward = [
        {"winner": "a", "loser": "b", "compared_at": "t0"},
        {"winner": "c", "loser": "d", "compared_at": "t1"},
    ]
    reversed_order = list(reversed(forward))
    assert (ppm.comparisons_fingerprint(tags, embeddings, forward)
            == ppm.comparisons_fingerprint(tags, embeddings, reversed_order))


def test_comparisons_fingerprint_changes_when_a_new_comparison_is_added():
    tags = ["a", "b", "c"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    before = [{"winner": "a", "loser": "b", "compared_at": "t0"}]
    after = before + [{"winner": "a", "loser": "c", "compared_at": "t1"}]
    assert (ppm.comparisons_fingerprint(tags, embeddings, before)
            != ppm.comparisons_fingerprint(tags, embeddings, after))


def test_comparisons_fingerprint_ignores_comparisons_with_unknown_tags():
    tags = ["a", "b"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    without_unknown = [{"winner": "a", "loser": "b", "compared_at": "t0"}]
    with_unknown = without_unknown + [{"winner": "a", "loser": "missing", "compared_at": "t1"}]
    assert (ppm.comparisons_fingerprint(tags, embeddings, without_unknown)
            == ppm.comparisons_fingerprint(tags, embeddings, with_unknown))
