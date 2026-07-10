# Preference Classifier Status and Toggle Design

**Status:** Draft for review

## Problem

The preference-classifier plan (`docs/superpowers/plans/2026-07-09-preference-classifier.md`,
merged in PR #6) built the whole pipeline: `rate.html` collects yes/no labels,
`search/preference_model.py` trains a logistic-regression classifier on those labels once
`MIN_LABELS` (50) is cleared, and two opt-in consumers can use the trained model's predictions
instead of raw novelty: `build/elite_archive.py`'s MAP-Elites fallback champion (behind a
`?use_predicted_preference=1` query param on `archive.html`, with no UI control) and
`search/driver.py`'s exploit pool (behind a `--use-predicted-preference` CLI flag on
`clawmarks run allnight`, set by hand over SSH).

There is no way to see, from the web UI, whether enough labels exist yet, whether a model has
actually been trained, or how good it is. Enabling predicted-preference in either consumer
today requires knowing the query param or CLI flag exists, and there's no single place that
turns it on for both at once. `lab_notebook.md`'s 2026-07-10 entry confirms Tasks 9-12 landed
but both Stage 5b flags stay off by default, with flipping them called out as "a human judgment
call" that has no supporting UI yet.

## Decisions (confirmed via brainstorming)

- **Toggle scope:** one shared, persisted setting that both `archive.html`'s rendering and the
  next `clawmarks run allnight` invocation read, not two independently-flipped controls.
- **UI placement:** a new dedicated page, `preference_status.html`, showing label counts, the
  `MIN_LABELS` gate, model metadata, and the toggle in one place. Other pages link to it rather
  than duplicating the controls.
- **Toggle gating:** disabled (not just discouraged) until `preference_model.joblib` actually
  exists on disk. The server enforces this too, not just the checkbox's `disabled` attribute.

## Architecture

### 1. Persisted setting

`config.py` gains `PREFERENCE_SETTINGS_FILE = SWEEP_DIR / "preference_settings.json"`. A new
module `search/preference_settings.py`:

```python
def load():
    """Returns {"use_predicted_preference": bool}. Missing file means the default, False."""

def save(enabled: bool):
    """Writes {"use_predicted_preference": enabled} to PREFERENCE_SETTINGS_FILE, atomically
    (write to a .tmp path, os.replace into place), matching the atomic-write pattern already
    used by embed_cache.save_cache and build/thumbnails.py."""
```

This file is the single source of truth. The existing `?use_predicted_preference=1` query
param on `archive.html` is removed; `archive.html` reads the persisted setting instead.

### 2. Model metadata sidecar

`search/preference_model.py`'s `main()` currently trains, prints a cross-validated accuracy to
the console, and discards it. It gains a sidecar write next to `MODEL_FILE`:

`MODEL_META_FILE = SWEEP_DIR / "preference_model_meta.json"`, written after a successful train:

```python
{
    "trained_at": "2026-07-10T19:53:00+00:00",  # UTC ISO 8601
    "n_labels": 62,
    "n_yes": 34,
    "n_no": 28,
    "cv_accuracy": 0.774,
}
```

This lets the status page show real model info without re-running cross-validation (which
scales with label count and DINOv2 embedding load) on every page view.

### 3. `build/preference_status.py` (new module, same shape as the other `build/*_view.py`
modules)

```python
def compute_data(sweep_dir):
    """Reads user_ratings.json for label counts, preference_model.class_balance_error for the
    gate message, MODEL_META_FILE if present, and preference_settings.load() for the current
    toggle value. Returns a dict covering all of that; never raises for the "not ready yet"
    states, since this page's whole purpose is showing users why it isn't ready."""

def render_html(data):
    """Renders the status page: label counts and gate status, model metadata table (or "no
    model trained yet" with the exact command to run), and a checkbox toggle. The checkbox is
    rendered `disabled` when data["has_model"] is False. Its onchange handler POSTs
    {"enabled": <bool>} to /api/preference_toggle and re-renders from the JSON response."""
```

### 4. `curation_server.py` routes

- `GET /preference_status.html` — served through the existing `_get_manifest_cached` LiveCache
  wrapper (matching every other tool page), watching `user_ratings.json`, `MODEL_FILE`,
  `MODEL_META_FILE`, and `PREFERENCE_SETTINGS_FILE`.
- `GET /api/preference_status` — same `compute_data` result as JSON, for the page's own
  post-toggle refresh (no full page reload needed).
- `POST /api/preference_toggle` — reads `{"enabled": bool}` from the request body. If
  `enabled` is `True` and `MODEL_FILE` doesn't exist, returns `400` with an error message
  (server-side gate, independent of the disabled checkbox). Otherwise calls
  `preference_settings.save(enabled)` and returns the fresh `compute_data` result.
- `archive.html`'s existing route drops its `use_predicted_preference` query-param parsing and
  instead calls `preference_settings.load()["use_predicted_preference"]` to pick the
  `"archive_actual"` vs `"archive_predicted"` cache target, same as today otherwise.

### 5. `cli.py` / `clawmarks run allnight`

`--use-predicted-preference`'s `argparse` default changes from `False` to `None` (a
three-state flag: not passed, passed explicitly true — `action="store_true"` can't express
"passed explicitly false", which is fine, nothing needs that). `cli.py`'s `main()` resolves the
effective value:

```python
if args.use_predicted_preference is None:
    from clawmarks.search import preference_settings
    effective = preference_settings.load()["use_predicted_preference"]
else:
    effective = args.use_predicted_preference
if effective:
    run_argv.append("--use-predicted-preference")
```

An explicit `--use-predicted-preference` on the command line still forces it on for that one
run regardless of the persisted setting (there's no way to explicitly force it *off* against a
`True` persisted setting via this flag, since `store_true` has no "explicit false" state; that's
an acceptable gap; a user who wants a one-off exception can flip the toggle first).
`search/driver.py`'s own `--use-predicted-preference` flag and its "no trained model found yet"
fallback message are unchanged; `cli.py` still just forwards the resolved boolean the same way
it forwards `--round` today.

## Out of scope

- Any change to how `preference_model.py`'s training itself works, or to `MIN_LABELS`.
- Auto-training a model when the label count crosses `MIN_LABELS` (training stays a manual,
  explicit step per the existing plan's Step 7; this feature only adds visibility and a toggle
  for using a model once one exists).
- Forcing predicted-preference *off* for a single `clawmarks run allnight` invocation when the
  persisted setting is on (see above).
- Any UI for retraining or deleting the model file.
