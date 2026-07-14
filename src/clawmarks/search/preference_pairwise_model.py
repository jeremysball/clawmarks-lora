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

Repeated judgments on the same underlying pair (whether resubmitted or resampled) are
consolidated into a single majority-vote verdict before training (_consolidate_pairs), so
resampling the same pair never counts as independent evidence.

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
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_score, permutation_test_score

from clawmarks.atomic_io import atomic_json_write, atomic_write
from clawmarks.config import SWEEP_DIR
from clawmarks.search import embed_cache

MIN_COMPARISONS = 50
N_PERMUTATIONS = 200
MODEL_FILE = SWEEP_DIR / "preference_pairwise_model.joblib"
MODEL_META_FILE = SWEEP_DIR / "preference_pairwise_model_meta.json"


def _consolidate_pairs(comparisons):
    """Collapses every judgment on the same underlying (unordered) pair into a single verdict by
    majority vote, so a pair judged N times counts as one piece of evidence instead of N
    independent training rows. Without this, resubmitting (or resampling) the same pair inflates
    apparent model accuracy and permutation-test significance on what is really repeated, not
    independent, evidence. A tied pair (equal wins each way) is dropped as ambiguous."""
    tally = {}
    for c in comparisons:
        winner, loser = c.get("winner"), c.get("loser")
        if not winner or not loser or winner == loser:
            continue
        votes = tally.setdefault(frozenset((winner, loser)), {})
        votes[winner] = votes.get(winner, 0) + 1
    consolidated = []
    for key, votes in tally.items():
        a, b = sorted(key)
        wins_a, wins_b = votes.get(a, 0), votes.get(b, 0)
        if wins_a == wins_b:
            continue
        winner, loser = (a, b) if wins_a > wins_b else (b, a)
        consolidated.append((winner, loser))
    return consolidated


def n_consolidated_pairs(comparisons):
    """Count of distinct pairs left after majority-vote consolidation, before filtering by
    embedding-cache presence. Lets a caller like curation_server's retrain gate tell whether an
    under-floor usable count is caused by missing embeddings (n_consolidated_pairs clears the
    floor but n_usable from build_training_set doesn't) or by duplicate-judgment consolidation
    itself (n_consolidated_pairs is already below the floor, so refreshing the embedding cache
    wouldn't help)."""
    return len(_consolidate_pairs(comparisons))


def _iter_usable_comparisons(tags, embeddings, comparisons):
    """Yields (winner, loser, winner_embedding, loser_embedding) for every consolidated pair
    (see _consolidate_pairs) whose winner and loser tags are both present in the embedding cache.
    The single filtering pass both build_training_set and comparisons_fingerprint rely on, so
    they can't drift apart."""
    tag_to_row = {t: i for i, t in enumerate(tags)}
    for winner, loser in _consolidate_pairs(comparisons):
        if winner not in tag_to_row or loser not in tag_to_row:
            continue
        yield winner, loser, embeddings[tag_to_row[winner]], embeddings[tag_to_row[loser]]


def build_training_set(tags, embeddings, comparisons):
    """`tags`/`embeddings` come from embed_cache.load_cache; `comparisons` is the loaded
    user_comparisons.json list. Returns (X, y): a mirrored pair of rows per usable, consolidated
    pair (embedding[winner] - embedding[loser] labeled 1, its negation labeled 0), skipping any
    pair whose winner or loser tag isn't in the embedding cache. Repeated judgments on the same
    pair are consolidated first (see _consolidate_pairs), so they contribute one row, not one
    row per judgment."""
    diffs = [w - loser_emb for _, _, w, loser_emb in _iter_usable_comparisons(tags, embeddings, comparisons)]
    if not diffs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    diffs = np.stack(diffs)
    X = np.concatenate([diffs, -diffs])
    y = np.concatenate([np.ones(len(diffs)), np.zeros(len(diffs))])
    return X.astype(np.float32), y.astype(np.int64)


def comparisons_fingerprint(tags, embeddings, comparisons):
    """A hash of the exact consolidated (winner, loser) pairs a train run would use. Two
    fingerprints matching means retraining now would use identical data to last time. Unlike a
    bare comparison count, this also catches a comparison being added and another
    (already-counted) one becoming unusable, e.g. after an embedding cache rebuild drops a tag.
    Repeating an already-judged pair does not change the fingerprint, since it's consolidated
    into the same single verdict rather than counted as a new row."""
    pairs = sorted((w, loser) for w, loser, _, _ in _iter_usable_comparisons(tags, embeddings, comparisons))
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def _pair_groups(n_rows):
    """Maps each of the n_rows training rows to the underlying comparison it came from, so a CV
    split can be forced to keep both mirrored rows of a pair (embedding[winner]-embedding[loser]
    and its negation) together in one fold. build_training_set always lays rows out as
    [diffs..., -diffs...], so row i and row i + n_rows//2 share a pair index of i % (n_rows//2).
    Without this, a fold can see one mirrored row at train time and its exact negation at test
    time, letting the model "predict" by sign alone instead of learning real signal. Only valid
    for rows produced by build_training_set's own layout; every current caller (cross_validate,
    significance, both only ever called by train_and_save on build_training_set's output)
    satisfies that, but a future caller passing some other row order/subset would get silently
    wrong groups rather than an error."""
    assert n_rows % 2 == 0, f"expected an even, mirrored row count from build_training_set, got {n_rows}"
    n_pairs = n_rows // 2
    return np.concatenate([np.arange(n_pairs), np.arange(n_pairs)])


def cross_validate(X, y, groups=None):
    """Mean cross-validated accuracy at predicting which side of a mirrored pair is the winner.
    Grouped by underlying pair so both mirrored rows of a comparison always land in the same
    fold; otherwise the model can exploit the leaked mirror rather than learning real signal.
    Leave-one-group-out below MIN_COMPARISONS pairs, since every pair matters at that scale;
    5-fold GroupKFold at or above it. Note this threshold is now counted in underlying pairs
    (n_groups), not raw rows as it was before grouping: a comparison set with, say, 30 usable
    pairs (60 rows) now gets LeaveOneGroupOut, whereas the pre-fix code would have already
    switched such a set to 5-fold at 50 rows. This matches train_and_save's own MIN_COMPARISONS
    gate, which was already counted in pairs."""
    if groups is None:
        groups = _pair_groups(len(y))
    n_groups = len(np.unique(groups))
    cv = LeaveOneGroupOut() if n_groups < MIN_COMPARISONS else GroupKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv, groups=groups)
    return float(scores.mean())


def significance(X, y, n_permutations=N_PERMUTATIONS, random_state=0, groups=None):
    """Permutation test: how often does a model trained on randomly shuffled labels score as
    well as the real one? A low p-value means the real accuracy is unlikely to be a fluke of
    this particular comparison set. baseline_accuracy is always 0.5 here because mirroring
    guarantees exact class balance, unlike preference_model.py's yes/no labels. Grouped by
    underlying pair for the same reason as cross_validate: otherwise the permutation test also
    reports significance on pure noise."""
    if groups is None:
        groups = _pair_groups(len(y))
    n_groups = len(np.unique(groups))
    cv = LeaveOneGroupOut() if n_groups < MIN_COMPARISONS else GroupKFold(n_splits=5, shuffle=True, random_state=0)
    _, _, p_value = permutation_test_score(
        LogisticRegression(max_iter=1000), X, y, cv=cv, groups=groups,
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
    comparisons to train on: fewer than MIN_COMPARISONS reference tags present in the embedding
    cache, even if the raw comparisons list itself clears MIN_COMPARISONS. Checking the raw count
    first is only a cheap early exit; the usable count (X.shape[0] // 2, since build_training_set
    mirrors every row) is the floor that actually governs whether a model gets trained, and it's
    what curation_server.py's retrain-gate check mirrors."""
    if len(comparisons) < MIN_COMPARISONS:
        return None
    tags, embeddings = embed_cache.load_cache(embed_cache.EMBEDDINGS_FILE)
    X, y = build_training_set(tags, embeddings, comparisons)
    n_usable = X.shape[0] // 2
    if n_usable < MIN_COMPARISONS:
        return None
    acc = cross_validate(X, y)
    stats = significance(X, y)
    model = train(X, y)
    # curation_server.py runs this fit outside its request lock (both the manual retrain endpoint
    # and the auto-retrain triggered from /api/compare), so two calls can genuinely overlap.
    # atomic_write/atomic_json_write use a unique tempfile per call (tempfile.mkstemp), unlike a
    # fixed f"{MODEL_FILE}.tmp" path, which two concurrent calls would both write to and corrupt.
    atomic_write(MODEL_FILE, lambda f: joblib.dump(model, f))
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_comparisons": len(comparisons),
        "n_usable_comparisons": n_usable,
        "comparisons_fingerprint": comparisons_fingerprint(tags, embeddings, comparisons),
        "cv_accuracy": round(acc, 4),
        "baseline_accuracy": stats["baseline_accuracy"],
        "p_value": stats["p_value"],
        "n_permutations": stats["n_permutations"],
    }
    atomic_json_write(MODEL_META_FILE, meta)
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
