# Head-to-head preference comparisons: replace yes/no rating

## Background and motivation

The current preference system (`rate.html`, `search/preference_model.py`) shows one image at a
time and asks yes or no. That trains a logistic-regression classifier on the DINOv2 embedding of
each rated image, labeled 1 for yes and 0 for no. It works, but an absolute yes/no judgment is a
harder, noisier call than a relative one: deciding whether a single image clears some invisible
bar is less natural than picking the better of two images placed side by side. This design
replaces the yes/no system with head-to-head comparisons: show two images, the user picks the
one they prefer, and a pairwise model learns from the sequence of winners and losers.

The existing yes/no data (61 labels, `notes/uncanny_seedrun1/user_ratings.json` and
`preference_model.joblib`) is left on disk untouched but unused. Nothing reads it after this
change ships.

## Data model and storage

A new file, `user_comparisons.json`, sits alongside the sweep directory's other JSON stores
(`user_ratings.json`, `user_favorites.json`, etc.). It's a JSON list, not a dict keyed by tag,
because the same pair can be compared more than once over time and each comparison is its own
event, not a fact about one image:

```json
[
  {"winner": "art_batch_0042", "loser": "art_batch_0891", "compared_at": "2026-07-11T18:03:22Z"},
  ...
]
```

`user_ratings.json` and `preference_model.joblib` are not deleted, migrated, or read by any code
after this change. They remain as historical artifacts only.

## Pairwise model (`search/preference_pairwise_model.py`)

Replaces `search/preference_model.py`'s role. Same shape of code (embedding cache in,
`joblib`-persisted `sklearn` model out), different training-set construction and different
score semantics.

**Training set construction:** for each comparison record, compute
`embedding[winner] - embedding[loser]` as one training row labeled 1. Mirror every row as
`-(embedding[winner] - embedding[loser])` labeled 0. This is the standard trick for fitting a
Bradley-Terry-style pairwise preference model with plain binary logistic regression: the model
learns a direction in embedding space such that "more in that direction" predicts winning, and
because the training set is embedding differences rather than raw per-image labels, the learned
direction can score any image in the pool, including ones that were never directly compared.

```python
def build_training_set(tags, embeddings, comparisons):
    tag_to_row = {t: i for i, t in enumerate(tags)}
    diffs = []
    for c in comparisons:
        if c["winner"] not in tag_to_row or c["loser"] not in tag_to_row:
            continue
        diffs.append(embeddings[tag_to_row[c["winner"]]] - embeddings[tag_to_row[c["loser"]]])
    if not diffs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    diffs = np.stack(diffs)
    X = np.concatenate([diffs, -diffs])
    y = np.concatenate([np.ones(len(diffs)), np.zeros(len(diffs))])
    return X.astype(np.float32), y.astype(np.int64)
```

**Training floor:** `MIN_COMPARISONS = 50`, mirroring today's `MIN_LABELS`. Below the floor,
`main()` refuses to train and prints a message pointing at `compare.html`.

**Cross-validation:** same pattern as today: leave-one-out below `MIN_COMPARISONS`, 5-fold
`StratifiedKFold` at or above it, scored on held-out pair-outcome accuracy (does the model rank
the held-out winner above the held-out loser).

**Scoring:** `score(embeddings) = model.decision_function(embeddings)`, not
`predict_proba`. Higher means more preferred. This is a signed real number, not a probability,
but every caller (`elite_archive.py`, `driver.py`, `preference_rank.py`) only needs a value that
sorts correctly, and `decision_function` is monotonic with the model's implied preference
ranking. Callers pass their own embeddings through the same function signature
(`score(model, embeddings) -> np.ndarray`) so the swap from `preference_model.predict_proba` is
a one-line import change at each call site.

**Model files:** `MODEL_FILE = SWEEP_DIR / "preference_pairwise_model.joblib"`,
`MODEL_META_FILE = SWEEP_DIR / "preference_pairwise_model_meta.json"`. New filenames, so they
never collide with the legacy yes/no artifacts sitting in the same directory.

## Pair selection (`search/comparison_sampler.py`)

Replaces `search/rating_sampler.py`'s role.

**Below `MIN_COMPARISONS`:** stratified-random pair. Reuse the existing faithfulness×novelty
grid (`bin_manifest`, `bin_edges`, `bin_of` from `search/scoring.py`, unchanged). Pick two bins
independently at random (can be the same bin), then one random eligible image from each bin,
retrying the second pick if it lands on the same image as the first.

**At/above `MIN_COMPARISONS`:** model-uncertainty-guided. Retrain every `RETRAIN_EVERY = 10` new
comparisons (checked as `n_comparisons % RETRAIN_EVERY == 0` right after a comparison is
recorded). To pick the next pair: sample a candidate set of ~200 random images from the pool,
score all of them once with the current model, and pick the two whose scores are closest
together. The closest-scored pair is the model's best approximation of "least sure which one
wins," without enumerating all `O(n^2)` pairs across a pool of thousands of images.

**No exclusion of repeat pairs.** Comparing the same close pair again is useful signal, not
wasted effort: it sharpens the model exactly where it's currently weakest.

```python
MIN_COMPARISONS = 50
RETRAIN_EVERY = 10
CANDIDATE_POOL_SIZE = 200

def pick_next_pair(manifest, model=None, rng=random):
    if model is None:
        return _stratified_random_pair(manifest, rng)
    candidates = rng.sample(manifest, min(CANDIDATE_POOL_SIZE, len(manifest)))
    scores = score(model, embeddings_for(candidates))
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1])
    # closest-scored adjacent pair in sorted order is the most uncertain comparison
    best_gap, best_pair = None, None
    for i in range(len(ranked) - 1):
        gap = abs(ranked[i][1] - ranked[i + 1][1])
        if best_gap is None or gap < best_gap:
            best_gap, best_pair = gap, (ranked[i][0], ranked[i + 1][0])
    return best_pair
```

## UI (`build/compare_page.py`, served as `compare.html`)

Replaces `build/rate_page.py` / `rate.html` entirely. Same dark theme, nav bar, and mobile-base
CSS as the rest of the tool suite (`shared_ui.py`).

- Two images side by side; stacked vertically on narrow/mobile viewports.
- Tap or click an image to pick it as the winner. ←/→ arrow keys also pick left/right.
- A small magnifier icon in each image's corner opens a full-res zoom overlay for that image
  only, reusing the existing pan/zoom mechanics from the old `rate.html` (drag to look around
  while zoomed, tap/click to close). The image itself stays a single, unambiguous tap-to-pick
  target, with no double-tap or long-press disambiguation needed.
- A `#count` footer tracks comparisons made this session, matching the old page's convention.
- `shared_ui.py`'s nav bar list: rename the "rate.html" entry to "compare.html".

## Server API (`curation_server.py`)

Remove:
- `GET /api/rate/next`
- `POST /api/rate`
- the `preference_model` import and its use in the trained-model existence check

Add:
- `GET /api/compare/next` → `{img1: {...item fields}, img2: {...item fields}}`, or
  `{"done": true}` if fewer than two comparable images exist in the pool (not expected in
  practice given pool size, but the contract mirrors the old `{"done": true}` shape for
  consistency).
- `POST /api/compare` → body `{"winner": tag, "loser": tag}` → appends a record to
  `user_comparisons.json` with a `compared_at` timestamp, returns `{"ok": true}`.

`preference_settings.py`'s `use_predicted_preference` toggle is unchanged in shape; it now gates
on the new pairwise model file's existence instead of the old one.

## Downstream integration points

All four call sites swap their import from `search.preference_model` to
`search.preference_pairwise_model`. The `score(model, embeddings) -> np.ndarray` signature is
preserved, so each swap is a one-line import change plus a rename from `predict_proba` to
`score`, except where noted:

- **`build/elite_archive.py`**: predicted-score usage (Stage 5b) is a straight import swap. The
  per-cell manual-override tier changes from "yes-rated image in cell wins" to "favorited image
  in cell wins": reads `user_favorites.json` instead of filtering `user_ratings.json` for
  `label == "yes"`.
- **`search/driver.py`**: `_load_yes_rated_images()` is renamed `_load_favorited_images()` and
  reads `user_favorites.json` instead of yes-labeled ratings. This is Stage 5a's manual-pool
  fallback, and it needs the same favorited-image substitution as `elite_archive.py` for the
  same reason (the yes/no signal it depended on no longer exists). `_predicted_preference_pool`
  (Stage 5b) is a straight import/rename swap, plus its hardcoded path
  `SWEEP_DIR / "preference_model.joblib"` becomes
  `SWEEP_DIR / "preference_pairwise_model.joblib"`.
- **`build/preference_rank.py`**: straight import/rename swap; the "no trained model" error
  message updates to reference `compare.html` and the new module path instead of `rate.html`.
- **`build/preference_status.py`**: straight import swap; status page wording changes from
  "labels" to "comparisons" (e.g. "only 12 comparisons (need 50)").

## Testing

- Unit tests for `preference_pairwise_model.py`: training-set construction (mirrored rows,
  skipped comparisons referencing unknown tags), refusal below `MIN_COMPARISONS`,
  cross-validation switch at the floor, `score()` monotonicity, mirroring the structure of the
  existing `preference_model.py` test file.
- Unit tests for `comparison_sampler.py`: stratified pair picking (never returns the same image
  twice, only from eligible bins), most-uncertain pair picking against a stub model with known
  scores (verify it picks the closest-scored pair from a candidate set).
- Server API tests for `GET /api/compare/next` and `POST /api/compare` (append behavior, `ok`
  response, `done` when pool is exhausted).
- Existing `preference_model.py` tests remain in the suite untouched: legacy code, not deleted,
  not imported by anything new.
