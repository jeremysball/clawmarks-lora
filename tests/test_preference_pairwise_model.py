import threading

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


def test_train_and_save_uses_a_unique_tmp_path_so_concurrent_calls_dont_clobber_each_other(tmp_path, monkeypatch):
    """curation_server.py now runs a retrain fit outside its request lock (both the manual
    /api/preference_retrain endpoint and the auto-retrain triggered from /api/compare), so two
    calls to train_and_save can genuinely overlap. Before this fix both wrote to the same fixed
    f"{MODEL_FILE}.tmp" path, risking one call's joblib.dump corrupting the other's in-flight
    write before either os.replace'd it into place. Force two calls to have their dump() calls
    in flight at the same time and assert they used distinct temp paths."""
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

    seen_paths = []
    seen_lock = threading.Lock()
    both_dumping = threading.Barrier(2, timeout=10)
    real_dump = ppm.joblib.dump

    def recording_dump(value, path, *a, **kw):
        with seen_lock:
            seen_paths.append(str(path))
        both_dumping.wait()  # forces both calls' writes to genuinely overlap
        return real_dump(value, path, *a, **kw)

    monkeypatch.setattr(ppm.joblib, "dump", recording_dump)

    results = []
    results_lock = threading.Lock()

    def run():
        result = ppm.train_and_save(comparisons)
        with results_lock:
            results.append(result)

    t1 = threading.Thread(target=run)
    t2 = threading.Thread(target=run)
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    assert len(seen_paths) == 2
    assert seen_paths[0] != seen_paths[1], "both calls wrote to the same tmp path"
    assert len(results) == 2
    assert all(r is not None for r in results)


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


def test_comparisons_fingerprint_changes_when_winner_and_loser_are_swapped():
    """A swapped winner/loser is a different training row (the mirrored sign flips), so it
    must not collide with the original pair's fingerprint."""
    tags = ["a", "b"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    original = [{"winner": "a", "loser": "b", "compared_at": "t0"}]
    swapped = [{"winner": "b", "loser": "a", "compared_at": "t0"}]
    assert (ppm.comparisons_fingerprint(tags, embeddings, original)
            != ppm.comparisons_fingerprint(tags, embeddings, swapped))


def test_comparisons_fingerprint_is_unchanged_when_a_judgment_is_repeated():
    """Regression test for issue #13: repeating the same judgment on a pair must not add another
    independent training row, since that inflates apparent accuracy/significance on what is
    really the same evidence submitted twice. The fingerprint (and the training set it tracks)
    must consolidate repeated judgments into a single verdict."""
    tags = ["a", "b"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    single = [{"winner": "a", "loser": "b", "compared_at": "t0"}]
    duplicated = single + [{"winner": "a", "loser": "b", "compared_at": "t1"}]
    assert (ppm.comparisons_fingerprint(tags, embeddings, single)
            == ppm.comparisons_fingerprint(tags, embeddings, duplicated))
    X, y = ppm.build_training_set(tags, embeddings, duplicated)
    assert X.shape == (2, 2)


def test_consolidate_pairs_uses_majority_vote_and_drops_ties():
    comparisons = (
        [{"winner": "a", "loser": "b", "compared_at": "t0"}] * 3
        + [{"winner": "b", "loser": "a", "compared_at": "t1"}]
        + [{"winner": "c", "loser": "d", "compared_at": "t2"}]
        + [{"winner": "d", "loser": "c", "compared_at": "t3"}]
    )
    consolidated = ppm._consolidate_pairs(comparisons)
    assert set(consolidated) == {("a", "b")}


def test_cross_validate_does_not_leak_mirrored_pairs_across_folds():
    """Regression test for a bug where StratifiedKFold split mirrored rows (a pair's diff and its
    negation) independently across folds, letting the model exploit the leaked mirror instead of
    learning real signal. On signal-free noise, that leak scored ~91% accuracy; grouping both
    mirrored rows of a pair into the same fold should score close to chance (~50%) instead."""
    rng = np.random.RandomState(0)
    n_pairs = 60
    diffs = rng.normal(size=(n_pairs, 768)).astype(np.float32)
    X = np.concatenate([diffs, -diffs]).astype(np.float32)
    y = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
    acc = ppm.cross_validate(X, y)
    assert acc < 0.65


def test_significance_does_not_report_false_significance_on_leaked_mirrored_noise():
    """Regression test for issue #12's second half: the permutation test built on top of the
    leaky CV also reported significance on pure noise. Grouping by pair must fix that too, not
    just the plain accuracy in test_cross_validate_does_not_leak_mirrored_pairs_across_folds."""
    rng = np.random.RandomState(0)
    n_pairs = 30
    diffs = rng.normal(size=(n_pairs, 32)).astype(np.float32)
    X = np.concatenate([diffs, -diffs]).astype(np.float32)
    y = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)]).astype(np.int64)
    stats = ppm.significance(X, y, n_permutations=30, random_state=0)
    assert stats["p_value"] > 0.2


def test_train_and_save_refuses_when_usable_comparisons_fall_below_raw_count(tmp_path, monkeypatch):
    """Regression test for a bug where train_and_save only checked the raw comparisons count
    against MIN_COMPARISONS, not the usable (embedding-cached) count. Here the raw count clears
    MIN_COMPARISONS but most comparisons reference tags missing from the embedding cache, so the
    usable count falls below the floor and training must still be refused."""
    monkeypatch.setattr(ppm, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(ppm, "MODEL_FILE", tmp_path / "preference_pairwise_model.joblib")
    monkeypatch.setattr(ppm, "MODEL_META_FILE", tmp_path / "preference_pairwise_model_meta.json")

    rng = np.random.RandomState(0)
    cached_tags = ["a", "b"]
    embeddings = rng.normal(size=(2, 2)).astype(np.float32)
    embed_cache.save_cache(tmp_path / "embeddings.npz", cached_tags, embeddings)
    monkeypatch.setattr(ppm.embed_cache, "EMBEDDINGS_FILE", tmp_path / "embeddings.npz")

    comparisons = [{"winner": "a", "loser": "b", "compared_at": "t0"}] * 5
    comparisons += [{"winner": f"missing{i}", "loser": f"missing{i}b", "compared_at": "t"} for i in range(55)]
    assert len(comparisons) >= ppm.MIN_COMPARISONS

    assert ppm.train_and_save(comparisons) is None
    assert not (tmp_path / "preference_pairwise_model.joblib").exists()


def test_train_and_save_refuses_when_repeated_judgments_consolidate_below_the_floor(tmp_path, monkeypatch):
    """Regression test for issue #13's headline exploit: 50 raw submissions of the SAME pair
    clear the early `len(comparisons) < MIN_COMPARISONS` check in train_and_save, but consolidate
    to a single usable pair, and must still be refused. Unlike
    test_train_and_save_refuses_when_usable_comparisons_fall_below_raw_count above (which covers
    the missing-embedding cause), every tag here has a cached embedding: the only reason usable
    falls below the floor is duplicate-judgment consolidation itself."""
    monkeypatch.setattr(ppm, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(ppm, "MODEL_FILE", tmp_path / "preference_pairwise_model.joblib")
    monkeypatch.setattr(ppm, "MODEL_META_FILE", tmp_path / "preference_pairwise_model_meta.json")

    tags = ["a", "b"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)
    monkeypatch.setattr(ppm.embed_cache, "EMBEDDINGS_FILE", tmp_path / "embeddings.npz")

    comparisons = [{"winner": "a", "loser": "b", "compared_at": "t0"}] * ppm.MIN_COMPARISONS
    assert len(comparisons) >= ppm.MIN_COMPARISONS

    assert ppm.train_and_save(comparisons) is None
    assert not (tmp_path / "preference_pairwise_model.joblib").exists()


def test_n_consolidated_pairs_counts_distinct_pairs_after_majority_vote():
    comparisons = (
        [{"winner": "a", "loser": "b", "compared_at": "t0"}] * 5
        + [{"winner": "c", "loser": "d", "compared_at": "t1"}]
        + [{"winner": "e", "loser": "f", "compared_at": "t2"}, {"winner": "f", "loser": "e", "compared_at": "t3"}]
    )
    # a/b: 1 distinct pair (5x same verdict), c/d: 1 distinct pair, e/f: tied 1-1, dropped.
    assert ppm.n_consolidated_pairs(comparisons) == 2
