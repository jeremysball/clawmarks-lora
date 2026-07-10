import numpy as np

from clawmarks.search import preference_model


def test_build_training_set_uses_only_tags_present_in_both_embeddings_and_ratings():
    tags = ["a", "b", "c"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    ratings = {
        "a": {"label": "yes", "rated_at": "t0"},
        "b": {"label": "no", "rated_at": "t1"},
        "missing_from_cache": {"label": "yes", "rated_at": "t2"},
    }
    X, y = preference_model.build_training_set(tags, embeddings, ratings)
    assert X.shape == (2, 2)
    assert list(y) == [1, 0]


def test_build_training_set_skips_unrecognized_labels():
    tags = ["a"]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    ratings = {"a": {"label": "maybe", "rated_at": "t0"}}
    X, y = preference_model.build_training_set(tags, embeddings, ratings)
    assert X.shape == (0, 0)
    assert len(y) == 0


def test_train_and_predict_proba_separates_obviously_different_clusters():
    rng = np.random.RandomState(0)
    yes_cluster = rng.normal(loc=5.0, scale=0.1, size=(20, 2))
    no_cluster = rng.normal(loc=-5.0, scale=0.1, size=(20, 2))
    X = np.vstack([yes_cluster, no_cluster]).astype(np.float32)
    y = np.array([1] * 20 + [0] * 20)
    model = preference_model.train(X, y)
    probs = preference_model.predict_proba(model, np.array([[5.0, 0.0], [-5.0, 0.0]], dtype=np.float32))
    assert probs[0] > 0.9
    assert probs[1] < 0.1


def test_cross_validate_returns_a_valid_accuracy_using_leave_one_out_below_min_labels():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(10, 2)).astype(np.float32)
    y = np.array([0, 1] * 5)
    acc = preference_model.cross_validate(X, y)
    assert 0.0 <= acc <= 1.0


def test_class_balance_error_flags_an_all_yes_label_set():
    y = np.ones(60, dtype=np.int64)
    error = preference_model.class_balance_error(y)
    assert "one class" in error


def test_class_balance_error_flags_an_all_no_label_set():
    y = np.zeros(60, dtype=np.int64)
    error = preference_model.class_balance_error(y)
    assert "one class" in error


def test_class_balance_error_flags_a_minority_class_below_the_fold_count():
    y = np.array([1] * 57 + [0] * 3, dtype=np.int64)
    error = preference_model.class_balance_error(y)
    assert "5-fold" in error


def test_class_balance_error_allows_a_well_balanced_label_set():
    y = np.array([1] * 30 + [0] * 30, dtype=np.int64)
    assert preference_model.class_balance_error(y) == ""


def test_class_balance_error_allows_an_imbalanced_but_above_fold_count_label_set_below_min_labels():
    y = np.array([1] * 8 + [0] * 2, dtype=np.int64)
    assert preference_model.class_balance_error(y) == ""
