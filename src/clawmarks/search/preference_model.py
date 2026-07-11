"""
Trains a logistic-regression preference classifier on frozen DINOv2 embeddings
(search/embed_cache.py) and the user's yes/no ratings (user_ratings.json), so images can
eventually be ranked by predicted preference instead of raw novelty. See
docs/superpowers/specs/2026-07-09-preference-classifier-design.md, Component 3.

Refuses to train below MIN_LABELS: with only a handful of ratings, any model would be
overfitting noise, not learning taste. Run rate.html (via `clawmarks serve`) until this
floor is cleared.

Run with: python -m clawmarks.search.preference_model
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

MIN_LABELS = 50
N_PERMUTATIONS = 200
MODEL_FILE = SWEEP_DIR / "preference_model.joblib"
MODEL_META_FILE = SWEEP_DIR / "preference_model_meta.json"


def _iter_usable_ratings(tags, embeddings, ratings):
    """Yields (tag, label, embedding_row) for every rating that both has a recognized yes/no
    label and a tag present in the embedding cache. The single filtering pass both
    build_training_set and ratings_fingerprint rely on, so they can't drift apart."""
    tag_to_row = {t: i for i, t in enumerate(tags)}
    for tag, rating in ratings.items():
        if tag not in tag_to_row:
            continue
        label = rating.get("label")
        if label not in ("yes", "no"):
            continue
        yield tag, label, embeddings[tag_to_row[tag]]


def build_training_set(tags, embeddings, ratings):
    """`tags`/`embeddings` come from embed_cache.load_cache; `ratings` is the loaded
    user_ratings.json dict. Returns (X, y) using only tags present in both the embedding cache
    and the ratings file with a recognized label. Row order follows `tags`, not ratings-dict
    iteration order, so X stays aligned with `embeddings`."""
    X_rows, y = [], []
    for _, label, row in _iter_usable_ratings(tags, embeddings, ratings):
        X_rows.append(row)
        y.append(1 if label == "yes" else 0)
    if not X_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(X_rows), np.array(y, dtype=np.int64)


def ratings_fingerprint(tags, embeddings, ratings):
    """A hash of the exact (tag, label) pairs a train run would use. Two fingerprints matching
    means retraining now would use identical data to last time -- unlike a bare label count,
    this also catches a rating flipping from yes to no on a tag that was already counted."""
    pairs = sorted((tag, label) for tag, label, _ in _iter_usable_ratings(tags, embeddings, ratings))
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def class_balance_error(y, min_labels=MIN_LABELS):
    """Returns a human-readable refusal message if `y` doesn't have enough of both classes to
    train/cross-validate, or "" if training can proceed. A total label count clearing MIN_LABELS
    (checked separately, in main()) says nothing about per-class balance: a set that is all
    `yes` (the natural state right after the picks-to-ratings migration) would otherwise reach
    LogisticRegression.fit with a single class and crash, or reach StratifiedKFold(n_splits=5)
    with a minority class too small to split."""
    n_yes = int(y.sum())
    n_no = len(y) - n_yes
    if n_yes == 0 or n_no == 0:
        return (f"labels are all one class ({n_yes} yes / {n_no} no); need at least one of each "
                f"to train. Rate more images via rate.html.")
    if len(y) >= min_labels:
        n_splits = 5
        minority = min(n_yes, n_no)
        if minority < n_splits:
            return (f"minority class has only {minority} labels ({n_yes} yes / {n_no} no); need "
                    f"at least {n_splits} for {n_splits}-fold cross-validation. Rate more of the "
                    f"less-common label via rate.html.")
    return ""


def cross_validate(X, y):
    """Mean cross-validated accuracy. Leave-one-out below MIN_LABELS, since every label matters
    at that scale; 5-fold StratifiedKFold at or above it."""
    cv = LeaveOneOut() if len(y) < MIN_LABELS else StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv)
    return float(scores.mean())


def significance(X, y, n_permutations=N_PERMUTATIONS, random_state=0):
    cv = LeaveOneOut() if len(y) < MIN_LABELS else StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
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


def predict_proba(model, embeddings):
    """Returns P(yes) for each row of `embeddings`."""
    return model.predict_proba(embeddings)[:, 1]


def main(argv=None):
    tags, embeddings = embed_cache.load_cache(embed_cache.EMBEDDINGS_FILE)
    ratings_path = SWEEP_DIR / "user_ratings.json"
    if not ratings_path.exists():
        print(f"no ratings file at {ratings_path}; nothing to train on", flush=True)
        return 1
    with open(ratings_path) as f:
        ratings = json.load(f)

    X, y = build_training_set(tags, embeddings, ratings)
    if len(y) < MIN_LABELS:
        print(f"only {len(y)} usable labels (need {MIN_LABELS}); not training. "
              f"Rate more images via rate.html first.", flush=True)
        return 1

    balance_error = class_balance_error(y)
    if balance_error:
        print(balance_error, flush=True)
        return 1

    acc = cross_validate(X, y)
    stats = significance(X, y)
    print(f"{len(y)} labels ({int(y.sum())} yes / {len(y) - int(y.sum())} no), "
          f"cross-validated accuracy: {acc:.3f}", flush=True)

    model = train(X, y)
    model_tmp = f"{MODEL_FILE}.tmp"
    joblib.dump(model, model_tmp)
    os.replace(model_tmp, MODEL_FILE)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_labels": len(y),
        "n_yes": int(y.sum()),
        "n_no": len(y) - int(y.sum()),
        "ratings_fingerprint": ratings_fingerprint(tags, embeddings, ratings),
        "cv_accuracy": round(acc, 4),
        "baseline_accuracy": stats["baseline_accuracy"],
        "p_value": stats["p_value"],
        "n_permutations": stats["n_permutations"],
    }
    tmp = f"{MODEL_META_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, MODEL_META_FILE)
    print(f"wrote {MODEL_FILE} and {MODEL_META_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
