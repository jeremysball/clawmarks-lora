"""
Trains a pairwise preference model on frozen DINOv2 embeddings (search/embed_cache.py) and the
user's head-to-head comparisons (user_comparisons.json), so images can be ranked by predicted
preference. Replaces search/preference_model.py's role: yes/no ratings are gone, comparisons are
head-to-head instead. See docs/superpowers/specs/2026-07-11-head-to-head-preference-design.md.

Fits a Bradley-Terry-style pairwise model with plain logistic regression: for each comparison,
the training row is embedding[winner] - embedding[loser] labeled 1, mirrored as its negation
labeled 0. The model learns a direction in embedding space such that "more in that direction"
predicts winning, which is why score() can rank any image in the pool, including one that was
never directly compared: it only depends on the image's own embedding, not a per-image win/loss
tally. Mirroring every row also guarantees exact class balance automatically, unlike the old
yes/no labels, so there's no balance-gate check needed here.

Refuses to train below MIN_COMPARISONS: with only a handful of comparisons, any model would be
overfitting noise, not learning taste. Run compare.html (via `clawmarks serve`) until this floor
is cleared.

Each train run also records a permutation-test p-value (significance()) alongside the plain
cross-validated accuracy, and a fingerprint of the exact comparisons used
(comparisons_fingerprint()) so callers like build/preference_status.py can tell whether new
comparisons have arrived since the last train without recomputing the whole training set.

Run with: python -m clawmarks.search.preference_pairwise_model
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score, permutation_test_score

from clawmarks.config import SWEEP_DIR
from clawmarks.search import embed_cache

MIN_COMPARISONS = 50
N_PERMUTATIONS = 200
MODEL_FILE = SWEEP_DIR / "preference_pairwise_model.joblib"
MODEL_META_FILE = SWEEP_DIR / "preference_pairwise_model_meta.json"


def _iter_usable_comparisons(tags, embeddings, comparisons):
    """Yields (winner, loser, winner_embedding, loser_embedding) for every comparison whose
    winner and loser tags are both present in the embedding cache. The single filtering pass
    both build_training_set and comparisons_fingerprint rely on, so they can't drift apart."""
    tag_to_row = {t: i for i, t in enumerate(tags)}
    for c in comparisons:
        winner, loser = c.get("winner"), c.get("loser")
        if winner not in tag_to_row or loser not in tag_to_row:
            continue
        yield winner, loser, embeddings[tag_to_row[winner]], embeddings[tag_to_row[loser]]


def build_training_set(tags, embeddings, comparisons):
    """`tags`/`embeddings` come from embed_cache.load_cache; `comparisons` is the loaded
    user_comparisons.json list. Returns (X, y): a mirrored pair of rows per usable comparison
    (embedding[winner] - embedding[loser] labeled 1, its negation labeled 0), skipping any
    comparison whose winner or loser tag isn't in the embedding cache."""
    diffs = [w - l for _, _, w, l in _iter_usable_comparisons(tags, embeddings, comparisons)]
    if not diffs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    diffs = np.stack(diffs)
    X = np.concatenate([diffs, -diffs])
    y = np.concatenate([np.ones(len(diffs)), np.zeros(len(diffs))])
    return X.astype(np.float32), y.astype(np.int64)


def comparisons_fingerprint(tags, embeddings, comparisons):
    """A hash of the exact (winner, loser) pairs a train run would use. Two fingerprints matching
    means retraining now would use identical data to last time. Unlike a bare comparison count,
    this also catches a comparison being added and another (already-counted) one becoming
    unusable, e.g. after an embedding cache rebuild drops a tag."""
    pairs = sorted((w, l) for w, l, _, _ in _iter_usable_comparisons(tags, embeddings, comparisons))
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def cross_validate(X, y):
    """Mean cross-validated accuracy at predicting which side of a mirrored pair is the winner.
    Leave-one-out below MIN_COMPARISONS rows, since every row matters at that scale; 5-fold
    StratifiedKFold at or above it."""
    cv = LeaveOneOut() if len(y) < MIN_COMPARISONS else StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv)
    return float(scores.mean())


def significance(X, y, n_permutations=N_PERMUTATIONS, random_state=0):
    """Permutation test: how often does a model trained on randomly shuffled labels score as
    well as the real one? A low p-value means the real accuracy is unlikely to be a fluke of
    this particular comparison set. baseline_accuracy is always 0.5 here because mirroring
    guarantees exact class balance, unlike preference_model.py's yes/no labels."""
    cv = LeaveOneOut() if len(y) < MIN_COMPARISONS else StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    _, _, p_value = permutation_test_score(
        LogisticRegression(max_iter=1000), X, y, cv=cv,
        n_permutations=n_permutations, random_state=random_state,
    )
    baseline_accuracy = max(np.bincount(y)) / len(y)
    return {
        "baseline_accuracy": float(baseline_accuracy),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }


def train(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def score(model, embeddings):
    """Returns a higher-is-more-preferred score for each row of `embeddings`. Uses
    decision_function rather than predict_proba: the model was trained on embedding
    *differences*, so there's no single well-defined "P(yes)" for a lone image, but
    decision_function is monotonic with the model's implied preference ranking, which is all
    every caller (elite_archive.py, driver.py, preference_rank.py) needs to sort by."""
    return model.decision_function(embeddings)


def train_and_save(comparisons):
    """Trains on `comparisons` (an already-loaded list) and persists MODEL_FILE/MODEL_META_FILE.
    Returns {"model", "cv_accuracy", "n_comparisons"}, or None if there aren't enough usable
    comparisons to train on (fewer than MIN_COMPARISONS, or none reference tags present in the
    embedding cache)."""
    if len(comparisons) < MIN_COMPARISONS:
        return None
    tags, embeddings = embed_cache.load_cache(embed_cache.EMBEDDINGS_FILE)
    X, y = build_training_set(tags, embeddings, comparisons)
    if X.shape[0] == 0:
        return None
    acc = cross_validate(X, y)
    stats = significance(X, y)
    model = train(X, y)
    model_tmp = f"{MODEL_FILE}.tmp"
    joblib.dump(model, model_tmp)
    os.replace(model_tmp, MODEL_FILE)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_comparisons": len(comparisons),
        "comparisons_fingerprint": comparisons_fingerprint(tags, embeddings, comparisons),
        "cv_accuracy": round(acc, 4),
        "baseline_accuracy": stats["baseline_accuracy"],
        "p_value": stats["p_value"],
        "n_permutations": stats["n_permutations"],
    }
    tmp = f"{MODEL_META_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, MODEL_META_FILE)
    return {"model": model, "cv_accuracy": acc, "n_comparisons": len(comparisons)}


def main(argv=None):
    comparisons_path = SWEEP_DIR / "user_comparisons.json"
    if not comparisons_path.exists():
        print(f"no comparisons file at {comparisons_path}; nothing to train on", flush=True)
        return 1
    with open(comparisons_path) as f:
        comparisons = json.load(f)

    if len(comparisons) < MIN_COMPARISONS:
        print(f"only {len(comparisons)} comparisons (need {MIN_COMPARISONS}); not training. "
              f"Compare more images via compare.html first.", flush=True)
        return 1

    result = train_and_save(comparisons)
    if result is None:
        print("no comparisons reference tags present in the embedding cache; nothing to train "
              "on. Run `python -m clawmarks.search.embed_cache` first.", flush=True)
        return 1

    print(f"{result['n_comparisons']} comparisons, cross-validated accuracy: "
          f"{result['cv_accuracy']:.3f}", flush=True)
    print(f"wrote {MODEL_FILE} and {MODEL_META_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
