# Task 10 Report

## Result

Converted every `SWEEP_DIR`-derived constant and reference in
`src/clawmarks/curation_server.py` to resolve through `_active_out_dir()`.

The requested `.superpowers/sdd/task-10-brief.md` file was absent. I used the complete Task 10
requirements in `docs/superpowers/plans/2026-07-14-expedition-leg-generation-model.md`, lines
1711-1851. That section contains the exact helper implementations, replacement rules, search
command, verification command, and commit message supplied in the task description.

## Changes

- Replaced the five module-level file/directory constants with `_favorites_file()`,
  `_comparisons_file()`, `_counterfactuals_dir()`, `_counterfactuals_file()`, and
  `_cockpit_queue_file()`.
- Added `_seeds_file()` for the former imported `SEEDS_FILE` path.
- Routed manifest, cache, thumbnail, cockpit, counterfactual, seed, comparison, favorite,
  status-page, static-serving, startup-check, and search-run paths through `_active_out_dir()`.
- Preserved generated image paths as strings where they enter JSON-serialized records.
- Deferred counterfactual-directory creation until a counterfactual image is written. Calling
  `_counterfactuals_dir()` at import time would fail when no active leg is selected and would
  prevent the required clean import.
- Left Tasks 11-13 behavior unchanged beyond replacing stale path references.

## Completeness Check

Required command:

```text
rg -n "FAVORITES_FILE\b|COMPARISONS_FILE\b|COUNTERFACTUALS_DIR\b|COUNTERFACTUALS_FILE\b|COCKPIT_QUEUE_FILE\b|SEEDS_FILE\b" src/clawmarks/curation_server.py
```

Result: exit 1 with no output, confirming zero remaining occurrences.

An additional `rg -n "SWEEP_DIR|SWEEP2_DIR" src/clawmarks/curation_server.py` found only the
existing `CLAWMARKS_SWEEP_DIR` text in a startup error hint. No Python reference to either removed
constant remains.

## Verification

Red check before the edit:

```text
uv run python -c "import clawmarks.curation_server"
NameError: name 'SWEEP_DIR' is not defined
```

Green checks after the edit:

```text
uv run python -c "import clawmarks.curation_server"
exit 0

uv run python -m pytest -q tests/test_curation_server_active_leg.py
4 passed in 3.75s

uv run python -m py_compile src/clawmarks/curation_server.py
exit 0

git diff --check
exit 0
```

The affected pre-Task-15 suite ran as required:

```text
uv run python -m pytest -q tests/test_curation_server*.py
30 passed, 12 failed, 47 errors in 4.18s
```

All 59 failures/errors attempt `monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)`. Task 15 is
explicitly assigned to replace those fixtures with `_active_out_dir` monkeypatches.

The full suite also ran:

```text
uv run python -m pytest -q
275 passed, 21 failed, 47 errors in 41.54s
```

The 59 curation-server failures/errors have the same stale `SWEEP_DIR` fixture cause. The other
nine failures target constants removed by earlier migration tasks (`driver.SWEEP_DIR`,
`preference_rank.MODEL_FILE`, and `preference_settings.PREFERENCE_SETTINGS_FILE`) and are likewise
scheduled for later test migration.

Each `uv` command also printed the existing warning that `VIRTUAL_ENV` points at another checkout;
`uv` ignored it and used this worktree's `.venv`.

## Self-Review

- Confirmed every path helper resolves at call time, so changing the selected leg changes later
  reads and writes without restarting the process.
- Confirmed the module performs no active-leg path operation at import time.
- Confirmed cockpit manifest records retain string file paths rather than non-serializable `Path`
  values.
- Confirmed the diff changes only `src/clawmarks/curation_server.py` plus this required report.

## Concerns

- The requested task brief file was missing, so I used the matching Task 10 section in the plan.
- The full suite remains red until Task 15 updates stale fixtures. No observed failure indicates a
  remaining import error, syntax error, or stale path reference in `curation_server.py`.

Status: DONE_WITH_CONCERNS
