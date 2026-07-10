# CLAWMARKS preference classifier: rate, learn, steer the search

## Motivation

The hyperparameter search (round 1 of the 5-round plan in `lab_notebook.md` Section 3) is
paused: the user wants to shift effort to the inference-time exploration side, which already
has a mature toolchain (`search/driver.py`'s adaptive MAP-Elites-style loop, `curation_server.py`
with picks/favorites/counterfactuals, 8 browsing tools). That toolchain optimizes for
faithfulness and novelty, both DINOv2-embedding distances with no aesthetic judgment. The lab
notebook already documents a real gap this causes: `build_elite_archive.py` falls back to
"highest novelty wins" whenever no human pick exists for a bin, which can select a worse-looking
image over a better one in the same bin, since novelty has no opinion about quality.

This project adds a preference classifier: a model that predicts how much the user will like an
image, trained on the user's own yes/no ratings. It closes the elite-selection gap above and,
longer term, lets the live search hunt for images the user will like, not just images that are
novel.

**This also replaces picking.** The binary yes/no rating becomes the single mechanism for
marking an image as a search-steering success, superseding the separate "pick as winner"
control. Favoriting (a pure bookmark with no effect on the search) is unaffected and stays
exactly as it is.

## Scope

In scope: a ratings-collection UI, an embedding cache, a preference model trained on frozen
DINOv2 embeddings, a pool re-ranking view to validate the model, removing the pick mechanism in
favor of ratings, and wiring first ratings, then the model, into `build_elite_archive.py`'s
fallback and `search/driver.py`'s exploit selection.

Out of scope: resuming the hyperparameter search (paused, not abandoned; picks back up as its
own thread later). Any change to the DINOv2 centroid/novelty scoring itself. A pairwise-
comparison or active-learning labeling scheme (out of scope for v1; plain random-eligible
sampling, stratified by existing bins, is enough to start). Favoriting: unchanged, out of scope.

## Current state (grounding numbers)

- `scored_manifest.json`: 3672 images, each with `centroid_sim` (faithfulness) and `novelty`
  scores already computed, but no raw embedding vectors persisted.
- `user_picks.json`: 40 picks (all positive; feeds the search's exploit pool today). Being
  retired in favor of `user_ratings.json`; see Component 2a.
- `user_favorites.json`: 1 favorite (pure bookmark, no search effect). Unchanged by this project.
- `user_counterfactuals.json`: 0 (feature built, unused so far).

40 labels, all positive, is not enough to train anything. The rating UI below exists to fix that
before any model training is attempted.

## Component 1: embedding cache

A one-time script (`src/clawmarks/search/embed_cache.py`) runs the DINOv2 model already used
by the scoring pipeline over every image referenced in `scored_manifest.json`, and writes the
resulting embeddings to `notes/uncanny_sweep/embeddings.npz` (or equivalent), keyed by image
tag. This runs locally (no RunPod cost). A second mode processes only tags missing from the
cache, so future search rounds can extend it incrementally rather than recomputing everything.

## Component 2: rating UI

A new page, `rate.html`, served by `src/clawmarks/curation_server.py`, shows one image at a time with a
binary yes/no control (mouse and keyboard, e.g. arrow keys or `y`/`n`). Two new endpoints:

- `GET /api/rate/next`: returns the next image to rate.
- `POST /api/rate`: records a label for a given tag.

Labels are stored in a new `notes/uncanny_sweep/user_ratings.json`,
`{tag: {label: "yes"|"no", rated_at}}`, parallel to the existing picks/favorites files.

**Sampling**: `GET /api/rate/next` excludes any tag already present in `user_favorites.json` or
`user_ratings.json`, so every rating adds a new label. From the remaining eligible pool, it
samples stratified across the existing faithfulness x novelty bins (the same grid
`build_elite_archive.py` already uses) rather than pure random, so an early session doesn't
over-sample whichever region happens to dominate the pool (e.g. late-generation exploit-heavy
images).

## Component 2a: retiring "pick as winner"

The lightbox's "pick as winner" star button, its badge, and the `/api/pick`/`/api/unpick`
endpoints are removed. A one-time migration script copies every entry in `user_picks.json` into
`user_ratings.json` as `{label: "yes", rated_at: <original picked_at>}` (skipping any tag that
already has a rating), so the 40 existing picks become the first 40 ratings rather than being
lost. `user_picks.json` itself is left on disk untouched after migration (not deleted), as a
historical record; nothing reads it anymore once this ships. `build_elite_archive.py` and
`search/driver.py` both move from reading `user_picks.json` to reading `user_ratings.json`,
treating any `label: "yes"` entry the way a pick was treated before (see Component 5).

Favoriting (`user_favorites.json`, the star/bookmark button, `/api/favorite`/`/api/unfavorite`)
is completely unaffected: it stays a separate mechanism with no search effect, per the project's
existing design ("I like this" and "build more like this" are different judgments).

## Component 3: training

A script, `src/clawmarks/search/preference_model.py`, loads the embedding cache and
`user_ratings.json`, and trains a logistic regression (scikit-learn) on the embeddings alone
(no generation metadata as input — see "Feature set" below). It reports validation accuracy via
k-fold (or leave-one-out if fewer than ~50 labels exist) and saves the fitted model to disk
(`notes/uncanny_sweep/preference_model.joblib`).

Training is a manual, explicit step (rerun the script), not automatic on every new rating, so
results stay easy to reason about. The script refuses to train below a floor of 50 labels and
prints a clear message instead of producing a model on too little data.

**Feature set: embedding only.** Generation metadata (strength, cfg, prompt_type, category) is
deliberately excluded. Metadata features risk the model keying off generation settings rather
than actual visual content, and they're meaningless for any image from outside this pipeline.
Embedding-only keeps the model a pure "does this look like something the user likes" predictor.

## Component 4: pool re-ranking view (validation stage)

A new view, either a sort mode added to the existing scan gallery or a standalone page, lists
every embedded image sorted by the model's predicted preference probability, highest first.
This is the human validation gate: before the model touches anything live, the user browses this
ranking and confirms it actually tracks their taste.

Built-in sanity check: the model should score the 40 migrated picks (label "yes" in
`user_ratings.json`, predating the model, though not the rating file itself) highly. If it
doesn't, that's a signal to investigate before proceeding to Component 5, not a green light to
plow ahead anyway.

## Component 5: steering the live search

This is a two-stage handoff, not a single permanent design:

**Stage 5a, immediate (no model dependency):** `build_elite_archive.py` and `search/driver.py`
move from reading `user_picks.json` to reading `user_ratings.json`, treating any `yes`-labeled
tag exactly as a pick was treated before: `build_elite_archive.py`'s per-bin fallback still
prefers a `yes`-rated image over the highest-novelty image when one exists in that cell, and
`search/driver.py`'s exploit pool draws from `yes`-rated images instead of `user_picks.json`.
This ships as part of Component 2a, works from the very first rating, and requires no trained
model.

**Stage 5b, gated on Component 4 passing:** once the preference model clears the Component 4
validation gate, exploit selection and the elite-archive fallback switch from the binary
yes-rated set to the model's continuous predicted-preference score, ranking all eligible images
by predicted preference rather than only distinguishing rated from unrated. This is opt-in
behind an explicit flag. Because it changes what a paid, live search actually generates, it gets
a dry run first: run the driver for one generation with the new fallback active, diff which
images it would have selected against the yes-rating-only logic from 5a, and eyeball the
difference before trusting it on a real budget-metered overnight run.

## Data flow

```
one-time: migrate user_picks.json -> user_ratings.json (label: "yes"); remove pick UI/endpoints
search driver generates images
  -> DINOv2 scoring (existing: centroid_sim, novelty)
  -> embedding cache (new, one-time + incremental)
  -> rating UI samples unreviewed images, stratified by bin
  -> user rates yes/no -> user_ratings.json
  -> Stage 5a (immediate): elite_archive fallback + driver.py exploit pool read yes-rated
     images from user_ratings.json, exactly where user_picks.json was read before
  -> preference_model.py trains on embeddings + ratings (manual, gated at 50+ labels)
  -> Component 4: re-rank pool, human validates against the migrated yes-ratings
  -> Stage 5b (only after validation passes): elite_archive fallback and driver.py exploit
     selection switch from yes-rated-set to continuous predicted-preference score, behind a
     flag, dry-run tested first
```

## Error handling

- Training below the 50-label floor: script exits with a clear message, no model produced.
- Rating UI sampler: must never re-serve a tag already in favorites/ratings; a unit test covers
  this directly.
- Re-rating an already-rated tag overwrites the existing entry rather than duplicating it.
- Embedding cache: incremental mode must not silently skip tags that exist in the manifest but
  are missing an image file on disk; it should report a hard error listing which tags failed.
- Migration script: skips (does not overwrite) any tag that already has a rating, so it's safe
  to rerun.

## Testing

- Unit tests: stratified sampler excludes reviewed tags; `POST /api/rate` overwrites, not
  duplicates; `preference_model.py` refuses to train under the label floor; migration script
  round-trips picks into ratings without overwriting existing ratings.
- Model-quality gate: Component 4's sanity check (the 40 migrated picks should score highly) is
  the primary go/no-go signal before Stage 5b is attempted.
- Stage 5b dry run: diff old (yes-rating-only) vs. new (predicted-preference) elite-selection
  output for one generation before it's allowed to affect a real search budget.
