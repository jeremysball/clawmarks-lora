# Expedition / Leg Generation Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded round-1/round-2 pair in the `allnight` search driver with an
open-ended "expeditions of legs" model: any number of named generation runs (legs), grouped into
named projects (expeditions), each startable from the curation server's web UI with no code
change.

**Architecture:** `expeditions/<name>/expedition.json` (checked into git) holds shared defaults;
`expeditions/<name>/legs/<leg>.json` holds per-leg overrides, merged field-by-field at load time
into a `LegConfig`. Runtime output moves to `$XDG_STATE_HOME/clawmarks/expeditions/<name>/<leg>/`.
Novelty scoring excludes every *sibling* leg's already-generated images within the same
expedition, generalizing round 2's one-off "exclude round 1" special case. The curation server
gains a server-side "active expedition/leg" selection (a small persisted pointer, set via a new
`/api/active-leg` endpoint) so every existing tool page keeps working exactly as before, just
against whichever leg is currently selected, instead of one process-wide `SWEEP_DIR` constant.

**Tech Stack:** Python 3, stdlib `http.server`, `pytest`, `dataclasses`, existing DINOv2/torch
scoring pipeline (unchanged).

## Global Constraints

- Storage layout exactly as specified: config under `expeditions/<name>/{expedition.json,
  legs/<leg>.json}` (checked into git); runtime output under
  `$XDG_STATE_HOME/clawmarks/expeditions/<name>/<leg>/` (`config.STATE_DIR`-relative, per the
  project's XDG rule already in place).
- Launching a leg into an expedition name that does not exist is a clear, immediate error, never
  a silent auto-create.
- A leg's effective config is `expedition.json` merged field-by-field with `legs/<leg>.json`
  (leg wins on conflict); an empty leg file inherits everything.
- Novelty exclusion pools every *other* leg's `scored_manifest.json` within the same expedition,
  in addition to the always-global real-training-set centroid. A brand-new expedition's first leg
  therefore runs with an empty exclusion pool.
- Every expedition gets a standing `cockpit` leg, scaffolded automatically at creation time.
- No image data moves or gets migrated by script (round 1 and round 2's full-resolution images
  are permanently gone, per the lab notebook's 2026-07-09 and 2026-07-14 entries); the
  `uncanny_frontier` reference expedition is written by hand from the old rounds' parameters only.
- Every crash-sensitive write (state files, manifests) keeps using
  `clawmarks.atomic_io.atomic_json_write`/`atomic_write` (tmp-file + `os.replace`), per this
  project's data-integrity rule. Never introduce a new non-atomic JSON write to a file whose loss
  would be expensive.
- Run `python3 -m pytest -q <changed test files>` after each task; run the full suite
  (`python3 -m pytest -q`) once at the end per this project's testing convention.

---

## Why this plan is bigger than the design doc's file list

The design doc's "Code changes" section names four files: `driver.py`, `config.py`,
`run_manager.py`, `curation_server.py`. Tracing actual imports shows `config.SWEEP_DIR` is also a
hardcoded module-level constant in `embed_cache.py`, `preference_pairwise_model.py`,
`preference_settings.py`, `migrate_picks_to_ratings.py`, and `score_manifest.py` — none of which
route through `curation_server.py`. Removing `SWEEP_DIR` from `config.py` (which the design doc
explicitly calls for) breaks all of them at import time unless they're updated too. This plan
covers the full dependency graph, confirmed by user decision: make every `SWEEP_DIR`-derived
constant leg-relative and resolved at call time, the same pattern the design doc already uses for
`curation_server.py` itself. It also deletes `src/clawmarks/search/preference_model.py` and
`tests/test_preference_model.py` (a superseded scalar preference model with zero live callers,
same situation as `merge_round2.py`, which the design doc already deletes) rather than updating
dead code for no benefit.

---

### Task 1: `config.py` — expedition/leg paths, remove the two hardcoded round directories

**Files:**
- Modify: `src/clawmarks/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.EXPEDITIONS_DIR` (`Path`, `= ROOT / "expeditions"`), `config.ACTIVE_LEG_FILE`
  (`Path`, `= STATE_DIR / "active_leg.json"`), `config.leg_dir(expedition: str, leg: str) -> Path`
  (`= STATE_DIR / "expeditions" / expedition / leg`). Removes `config.SWEEP_DIR`,
  `config.SWEEP2_DIR`, `config.SEEDS_FILE`, `config.USER_PICKS_FILE`, `config.USER_RATINGS_FILE`,
  `config.PREFERENCE_SETTINGS_FILE`. `config.PROBE_DIR`/`config.PROBE_STRENGTH_DIR` are unchanged
  (unaffected by this migration).

- [x] **Step 1: Write the failing tests**

Replace the two removed-constant assertions and add tests for the new API. Edit
`tests/test_config.py`:

```python
# tests/test_config.py
import os
import subprocess
import sys

from clawmarks import config


def test_repo_root_finds_pyproject():
    root = config.repo_root()
    assert (root / "pyproject.toml").exists()


def test_repo_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAWMARKS_ROOT", str(tmp_path))
    assert config.repo_root() == tmp_path


def test_derived_paths_under_state_dir():
    assert config.PROBE_DIR == config.STATE_DIR / "probe_uncanny"
    assert config.PROBE_STRENGTH_DIR == config.STATE_DIR / "probe_strength"


def test_expeditions_dir_is_repo_relative():
    assert config.EXPEDITIONS_DIR == config.ROOT / "expeditions"


def test_active_leg_file_is_state_dir_relative():
    assert config.ACTIVE_LEG_FILE == config.STATE_DIR / "active_leg.json"


def test_leg_dir_resolves_under_state_dir_expeditions():
    assert config.leg_dir("uncanny_frontier", "round1") == (
        config.STATE_DIR / "expeditions" / "uncanny_frontier" / "round1"
    )


def test_state_dir_defaults_to_xdg_state_home(tmp_path):
    env = dict(os.environ, XDG_STATE_HOME=str(tmp_path), PYTHONPATH="src")
    env.pop("CLAWMARKS_STATE_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", "from clawmarks.config import STATE_DIR; print(STATE_DIR)"],
        env=env, capture_output=True, text=True, cwd=str(config.ROOT), check=True,
    )
    assert result.stdout.strip() == str(tmp_path / "clawmarks")


def test_state_dir_env_override(tmp_path):
    env = dict(os.environ, CLAWMARKS_STATE_DIR=str(tmp_path), PYTHONPATH="src")
    result = subprocess.run(
        [sys.executable, "-c",
         "from clawmarks.config import leg_dir; print(leg_dir('e', 'l'))"],
        env=env, capture_output=True, text=True, cwd=str(config.ROOT), check=True,
    )
    assert result.stdout.strip() == str(tmp_path / "expeditions" / "e" / "l")
```

- [x] **Step 2: Run to verify failure**

Run: `python3 -m pytest -q tests/test_config.py`
Expected: FAIL — `AttributeError: module 'clawmarks.config' has no attribute 'EXPEDITIONS_DIR'`
(and similar for the other new names).

- [x] **Step 3: Rewrite `config.py`**

```python
import os
from pathlib import Path


def repo_root() -> Path:
    env_override = os.environ.get("CLAWMARKS_ROOT")
    if env_override:
        return Path(env_override)
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(
        "Could not find pyproject.toml by walking up from clawmarks/config.py. "
        "If clawmarks is installed outside its source checkout, set CLAWMARKS_ROOT "
        "to the repo root explicitly."
    )


ROOT = repo_root()
NOTES_DIR = ROOT / "notes"
EXPEDITIONS_DIR = ROOT / "expeditions"
# Runtime state (generated images, manifests, checkpoints) lives outside the repo per the
# XDG Base Directory spec, not under notes/, so it survives a repo re-clone and doesn't
# tempt anyone into committing gitignored generation output. CLAWMARKS_STATE_DIR overrides
# the XDG default entirely; XDG_STATE_HOME overrides only the ~/.local/state root.
STATE_DIR = Path(os.environ["CLAWMARKS_STATE_DIR"]) if os.environ.get("CLAWMARKS_STATE_DIR") \
    else Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "clawmarks"
PROBE_DIR = STATE_DIR / "probe_uncanny"
PROBE_STRENGTH_DIR = STATE_DIR / "probe_strength"
# The curation server's currently-selected expedition/leg, persisted so a server restart
# doesn't forget it and fall back to the empty-state hub while real data still exists.
ACTIVE_LEG_FILE = STATE_DIR / "active_leg.json"


def leg_dir(expedition: str, leg: str) -> Path:
    """Runtime output directory for one leg: allnight_state.json, scored_manifest.json,
    thumbs/, real_thumbs/, seed_pool.json, and every curation artifact (favorites,
    comparisons, cockpit queue, ...) that today lives at a fixed SWEEP_DIR."""
    return STATE_DIR / "expeditions" / expedition / leg
```

- [x] **Step 4: Run to verify pass**

Run: `python3 -m pytest -q tests/test_config.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/clawmarks/config.py tests/test_config.py
git commit -m "feat(config): add expedition/leg paths, remove hardcoded round directories"
```

---

### Task 2: `driver.py` — `LegConfig` and the expedition/leg config merge

**Files:**
- Modify: `src/clawmarks/search/driver.py`
- Test: `tests/test_driver_state.py` (new fixtures added here; existing tests updated in Task 5)

**Interfaces:**
- Consumes: `config.EXPEDITIONS_DIR`, `config.leg_dir(expedition, leg)` (Task 1).
- Produces: `driver.LegConfig` (dataclass: `expedition: str`, `leg: str`, `dir: Path`,
  `wall_clock_cap_hours: float`, `budget_usd_cap: float`, `budget_safety_margin: float`,
  `gen_batch_size: int`, `explore_fraction: float`, `max_generations: int`, `textures: list`,
  `fallback_subjects: list`, `seed_from_start: bool`, `style_subject_count: int = 4`,
  `description: str = ""`, `widened_textures: list = field(default_factory=list)`,
  `widened_subjects: list = field(default_factory=list)`), `driver.load_leg_config(expedition:
  str, leg: str) -> LegConfig`, raising `RuntimeError` if `expedition.json` doesn't exist. Removes
  `driver.RoundConfig`, `driver.ROUND_CONFIGS`.

- [x] **Step 1: Write the failing tests**

Add to `tests/test_driver_state.py` (new file section; keep existing tests for now, they're
migrated in Task 5):

```python
def test_load_leg_config_merges_expedition_and_leg_fields(tmp_path, monkeypatch):
    from clawmarks.search import driver
    from clawmarks import config

    expeditions_dir = tmp_path / "expeditions"
    (expeditions_dir / "demo" / "legs").mkdir(parents=True)
    (expeditions_dir / "demo" / "expedition.json").write_text(json.dumps({
        "wall_clock_cap_hours": 7.5, "budget_usd_cap": 10.0, "budget_safety_margin": 1.5,
        "gen_batch_size": 60, "explore_fraction": 0.5, "max_generations": 400,
        "textures": ["tex-a"], "fallback_subjects": ["subj-a"], "seed_from_start": False,
    }))
    (expeditions_dir / "demo" / "legs" / "leg1.json").write_text(json.dumps({
        "explore_fraction": 0.85, "seed_from_start": True,
    }))
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", expeditions_dir)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")

    cfg = driver.load_leg_config("demo", "leg1")

    assert cfg.expedition == "demo"
    assert cfg.leg == "leg1"
    assert cfg.explore_fraction == 0.85  # leg override wins
    assert cfg.seed_from_start is True  # leg override wins
    assert cfg.wall_clock_cap_hours == 7.5  # inherited from expedition.json
    assert cfg.textures == ["tex-a"]  # inherited, no leg override present
    assert cfg.dir == config.leg_dir("demo", "leg1")


def test_load_leg_config_empty_leg_file_inherits_everything(tmp_path, monkeypatch):
    from clawmarks.search import driver
    from clawmarks import config

    expeditions_dir = tmp_path / "expeditions"
    (expeditions_dir / "demo" / "legs").mkdir(parents=True)
    (expeditions_dir / "demo" / "expedition.json").write_text(json.dumps({
        "wall_clock_cap_hours": 1.0, "budget_usd_cap": 1.0, "budget_safety_margin": 0.1,
        "gen_batch_size": 20, "explore_fraction": 0.85, "max_generations": 60,
        "textures": [], "fallback_subjects": [], "seed_from_start": True,
    }))
    (expeditions_dir / "demo" / "legs" / "cockpit.json").write_text("{}")
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", expeditions_dir)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")

    cfg = driver.load_leg_config("demo", "cockpit")

    assert cfg.explore_fraction == 0.85
    assert cfg.seed_from_start is True


def test_load_leg_config_missing_expedition_is_a_clear_error(tmp_path, monkeypatch):
    from clawmarks.search import driver
    from clawmarks import config

    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")

    with pytest.raises(RuntimeError, match="unknown expedition"):
        driver.load_leg_config("does_not_exist", "leg1")


def test_load_leg_config_missing_leg_file_is_ok(tmp_path, monkeypatch):
    """A leg file can be entirely absent (not just empty): inherit everything."""
    from clawmarks.search import driver
    from clawmarks import config

    expeditions_dir = tmp_path / "expeditions"
    (expeditions_dir / "demo" / "legs").mkdir(parents=True)
    (expeditions_dir / "demo" / "expedition.json").write_text(json.dumps({
        "wall_clock_cap_hours": 1.0, "budget_usd_cap": 1.0, "budget_safety_margin": 0.1,
        "gen_batch_size": 20, "explore_fraction": 0.5, "max_generations": 60,
        "textures": [], "fallback_subjects": [], "seed_from_start": False,
    }))
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", expeditions_dir)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")

    cfg = driver.load_leg_config("demo", "brand_new_leg")

    assert cfg.explore_fraction == 0.5
```

Add `import json` and `import pytest` at the top of `tests/test_driver_state.py` if not already
present (check the existing file header first).

- [x] **Step 2: Run to verify failure**

Run: `python3 -m pytest -q tests/test_driver_state.py -k load_leg_config`
Expected: FAIL — `AttributeError: module 'clawmarks.search.driver' has no attribute
'load_leg_config'`

- [x] **Step 3: Implement `LegConfig` and `load_leg_config`**

In `src/clawmarks/search/driver.py`, replace the `RoundConfig` dataclass and `ROUND_CONFIGS` dict
(lines 46–129) with:

```python
from clawmarks import config as clawmarks_config

_LEG_CONFIG_DEFAULTS = {
    "wall_clock_cap_hours": 7.5, "budget_usd_cap": 10.0, "budget_safety_margin": 1.5,
    "gen_batch_size": 60, "explore_fraction": 0.5, "max_generations": 400,
    "textures": [], "fallback_subjects": [], "seed_from_start": False,
    "style_subject_count": 4, "description": "",
    "widened_textures": [], "widened_subjects": [],
}


@dataclass
class LegConfig:
    expedition: str
    leg: str
    dir: object  # pathlib.Path; typed loosely to avoid a circular import-time annotation
    wall_clock_cap_hours: float
    budget_usd_cap: float
    budget_safety_margin: float
    gen_batch_size: int
    explore_fraction: float
    max_generations: int
    textures: list
    fallback_subjects: list
    seed_from_start: bool
    style_subject_count: int = 4
    description: str = ""
    widened_textures: list = field(default_factory=list)
    widened_subjects: list = field(default_factory=list)


def load_leg_config(expedition, leg):
    """Loads expeditions/<expedition>/expedition.json, merges legs/<leg>.json field-by-field
    on top (leg wins), and fills in any field neither file sets from _LEG_CONFIG_DEFAULTS.
    Missing expedition.json is a hard error: launching into a typo'd expedition name must
    never silently create an empty one."""
    expedition_file = clawmarks_config.EXPEDITIONS_DIR / expedition / "expedition.json"
    if not expedition_file.exists():
        raise RuntimeError(
            f"unknown expedition {expedition!r}: {expedition_file} does not exist. "
            f"Create it via the curation server's expedition-creation form first."
        )
    with open(expedition_file) as f:
        merged = json.load(f)

    leg_file = clawmarks_config.EXPEDITIONS_DIR / expedition / "legs" / f"{leg}.json"
    if leg_file.exists():
        with open(leg_file) as f:
            leg_overrides = json.load(f)
        merged.update(leg_overrides)

    fields = dict(_LEG_CONFIG_DEFAULTS)
    fields.update(merged)
    return LegConfig(
        expedition=expedition, leg=leg, dir=clawmarks_config.leg_dir(expedition, leg), **fields
    )
```

Remove the old module docstring's `ROUND_CONFIGS`/`--round` references (lines 1–20) and replace
with:

```python
"""
Merged all-night driver for the CLAWMARKS liminal-band / uncanny-frontier search
(lab_notebook.md Section 3b). Runs one named "leg" of generation within a named "expedition"
(see docs/superpowers/specs/2026-07-14-expedition-leg-generation-design.md): the expedition
holds shared prompt vocab and budget defaults (expedition.json); the leg holds whatever
overrides make this run different (legs/<leg>.json), merged via load_leg_config.

Novelty for a leg's new images is scored against the real training set plus every *other*
leg's already-generated images within the same expedition (see _load_sibling_leg_manifests),
so retreading a sibling leg's already-explored region no longer counts as novel.

Run with: uv run clawmarks run allnight --expedition <name> --leg <name>
"""
```

- [x] **Step 4: Run to verify pass**

Run: `python3 -m pytest -q tests/test_driver_state.py -k load_leg_config`
Expected: PASS (4 tests)

- [x] **Step 5: Commit**

```bash
git add src/clawmarks/search/driver.py tests/test_driver_state.py
git commit -m "feat(driver): add LegConfig and expedition/leg config merging"
```

---

### Task 3: `driver.py` — drop the round-numbered state/resume machinery and legacy shim

**Files:**
- Modify: `src/clawmarks/search/driver.py`
- Test: `tests/test_driver_state.py`

**Interfaces:**
- Consumes: `LegConfig` (Task 2).
- Produces: `driver._out_dir(cfg) -> Path` (now just `cfg.dir`), `driver._state_file(cfg) -> Path`
  (now always `cfg.dir / "allnight_state.json"`, no round suffix), `driver.load_state(cfg)`,
  `driver.save_state(cfg, state)`, `driver._validate_state(state, state_file)` (drops the
  `allow_legacy_round1_baseline` parameter entirely), `driver._validate_resume_agreement(state,
  manifest, state_file, manifest_path)` (same).

- [x] **Step 1: Update the failing tests first**

In `tests/test_driver_state.py`, every existing test that builds a cfg via
`driver.ROUND_CONFIGS[1]` or `driver.ROUND_CONFIGS[2]` needs a `LegConfig` instead. Replace the
whole file's fixture pattern: wherever a test currently does

```python
monkeypatch.setattr(driver, "SWEEP_DIR", tmp_path)
path = driver._state_file(driver.ROUND_CONFIGS[1])
```

replace with:

```python
cfg = driver.LegConfig(
    expedition="demo", leg="leg1", dir=tmp_path,
    wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
    gen_batch_size=60, explore_fraction=0.5, max_generations=400,
    textures=[], fallback_subjects=[], seed_from_start=False,
)
path = driver._state_file(cfg)
```

and wherever a test currently does

```python
monkeypatch.setattr(driver, "SWEEP2_DIR", tmp_path)
path = driver._state_file(driver.ROUND_CONFIGS[2])
```

replace with:

```python
cfg = driver.LegConfig(
    expedition="demo", leg="leg2", dir=tmp_path,
    wall_clock_cap_hours=1.0, budget_usd_cap=1.0, budget_safety_margin=0.1,
    gen_batch_size=20, explore_fraction=0.85, max_generations=60,
    textures=[], fallback_subjects=[], seed_from_start=True,
)
path = driver._state_file(cfg)
```

Every assertion on `state_file.name == "allnight_state.json"` (round 1's old test) and
`"allnight2_state.json"` (round 2's old test) collapses to one expectation: `_state_file(cfg).name
== "allnight_state.json"` regardless of `cfg.leg`, since the legacy round-numbered filename shim
is being deleted. Update the two tests that assert the exact filename accordingly, and delete any
test whose entire point was the `allow_legacy_round1_baseline` history-length-off-by-one
tolerance (search the file for `allow_legacy_round1_baseline` or a comment mentioning "one more
history value than completed generations" — that behavior no longer exists, so its test is
removed, not migrated).

Also remove `driver.ROUND_CONFIGS` from every remaining call site in the file the same way (each
`driver.ROUND_CONFIGS[1]`/`driver.ROUND_CONFIGS[2]` becomes an inline `LegConfig` built the same
way as above, reusing the `cfg` already constructed per test where the test builds only one).

- [x] **Step 2: Run to verify failure**

Run: `python3 -m pytest -q tests/test_driver_state.py`
Expected: FAIL — `AttributeError: module 'clawmarks.search.driver' has no attribute
'ROUND_CONFIGS'` (confirms the old tests were still referencing removed names before the
implementation catches up).

- [x] **Step 3: Simplify the state/resume functions**

In `src/clawmarks/search/driver.py`, replace `_out_dir`, `_state_file`, `load_state`,
`_validate_state`, `save_state`, `_validate_resume_agreement` with:

```python
def _out_dir(cfg):
    return cfg.dir


def _state_file(cfg):
    return cfg.dir / "allnight_state.json"


def load_state(cfg):
    state_file = _state_file(cfg)
    if not state_file.exists():
        return _new_state()
    try:
        with open(state_file) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"cannot resume: persisted state {state_file} is unreadable: {e}") from e
    _validate_state(state, state_file)
    return state


def _new_state():
    return {
        "generation": 0, "stage": 0, "plateau_count": 0,
        "novelty_history": [], "gpt55_subjects": [], "start_balance": None,
        "start_time": time.time(),
    }


def _validate_state(state, state_file):
    required = {
        "generation", "stage", "plateau_count", "novelty_history", "gpt55_subjects",
        "start_balance", "start_time",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise RuntimeError(
            f"cannot resume: persisted state {state_file} is malformed or missing required fields"
        )
    for name in ("generation", "stage", "plateau_count"):
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"cannot resume: persisted state {state_file} has invalid {name}")
    if not isinstance(state["novelty_history"], list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in state["novelty_history"]
    ):
        raise RuntimeError(f"cannot resume: persisted state {state_file} has invalid novelty_history")
    if len(state["novelty_history"]) != state["generation"]:
        raise RuntimeError(
            f"cannot resume: persisted state {state_file} has generation/history mismatch"
        )
    if not isinstance(state["gpt55_subjects"], list) or any(
        not isinstance(value, str) for value in state["gpt55_subjects"]
    ):
        raise RuntimeError(f"cannot resume: persisted state {state_file} has invalid gpt55_subjects")
    balance = state["start_balance"]
    if balance is not None and (
        isinstance(balance, bool) or not isinstance(balance, (int, float)) or not math.isfinite(balance)
    ):
        raise RuntimeError(f"cannot resume: persisted state {state_file} has invalid start_balance")
    start_time = state["start_time"]
    if isinstance(start_time, bool) or not isinstance(start_time, (int, float)) or not math.isfinite(start_time):
        raise RuntimeError(f"cannot resume: persisted state {state_file} has invalid start_time")
    if state["generation"] > 0 and balance is None:
        raise RuntimeError(
            f"cannot resume: persisted state {state_file} has generations but no start_balance"
        )


def save_state(cfg, state):
    state_file = _state_file(cfg)
    _validate_state(state, state_file)
    _atomic_json_write(state_file, state)
```

And `_validate_resume_agreement`:

```python
def _validate_resume_agreement(state, manifest, state_file, manifest_path):
    _validate_state(state, state_file)
    _validate_manifest(manifest, manifest_path)
    generations = [
        generation for entry in manifest
        if (generation := _current_driver_generation(entry["tag"])) is not None
    ]
    state_generation = state["generation"]
    if state_generation == 0 and generations:
        raise RuntimeError(
            f"cannot resume: persisted state {state_file} is behind manifest {manifest_path}"
        )
    if state_generation > 0 and (not generations or max(generations) != state_generation):
        if generations and max(generations) > state_generation:
            raise RuntimeError(
                f"cannot resume: persisted state {state_file} is behind manifest {manifest_path}"
            )
        raise RuntimeError(
            f"cannot resume: persisted state {state_file} is ahead of manifest {manifest_path}"
        )
```

Update every caller of `_validate_resume_agreement` and `_validate_state` to drop the
`allow_legacy_round1_baseline=...` keyword argument (there is one call site of each inside
`main()`, updated in Task 4).

- [x] **Step 4: Run to verify pass**

Run: `python3 -m pytest -q tests/test_driver_state.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/clawmarks/search/driver.py tests/test_driver_state.py
git commit -m "refactor(driver): drop the legacy round-1 state-history compatibility shim"
```

---

### Task 4: `driver.py` — sibling-leg exclusion pool and CLI

**Files:**
- Modify: `src/clawmarks/search/driver.py`
- Test: `tests/test_driver_state.py` or a new `tests/test_driver_sibling_exclusion.py`

**Interfaces:**
- Produces: `driver._load_sibling_leg_manifests(cfg) -> list[dict]` (replaces
  `_load_prev_round_state`), `driver.main(argv)` accepting `--expedition <name> --leg <name>`.

- [x] **Step 1: Write the failing test**

```python
# tests/test_driver_sibling_exclusion.py
import json

from clawmarks.search import driver


def _cfg(dir_path, expedition="demo", leg="leg1"):
    return driver.LegConfig(
        expedition=expedition, leg=leg, dir=dir_path,
        wall_clock_cap_hours=1.0, budget_usd_cap=1.0, budget_safety_margin=0.1,
        gen_batch_size=1, explore_fraction=0.5, max_generations=1,
        textures=[], fallback_subjects=[], seed_from_start=False,
    )


def test_new_expedition_has_no_sibling_leg_data_yet(tmp_path):
    expedition_root = tmp_path / "demo"
    leg_dir = expedition_root / "leg1"
    leg_dir.mkdir(parents=True)

    cfg = _cfg(leg_dir)
    assert driver._load_sibling_leg_manifests(cfg) == []


def test_sibling_leg_manifests_are_pooled_but_not_the_leg_s_own(tmp_path):
    expedition_root = tmp_path / "demo"
    leg1_dir = expedition_root / "leg1"
    leg2_dir = expedition_root / "leg2"
    leg1_dir.mkdir(parents=True)
    leg2_dir.mkdir(parents=True)
    (leg1_dir / "scored_manifest.json").write_text(json.dumps([{"tag": "leg1_a", "file": "a.png"}]))
    (leg2_dir / "scored_manifest.json").write_text(json.dumps([{"tag": "leg2_a", "file": "b.png"}]))

    cfg = _cfg(leg1_dir, leg="leg1")
    pooled = driver._load_sibling_leg_manifests(cfg)

    assert pooled == [{"tag": "leg2_a", "file": "b.png"}]


def test_sibling_leg_without_a_manifest_yet_is_skipped(tmp_path):
    expedition_root = tmp_path / "demo"
    leg1_dir = expedition_root / "leg1"
    leg2_dir = expedition_root / "leg2"  # no manifest written yet
    leg1_dir.mkdir(parents=True)
    leg2_dir.mkdir(parents=True)

    cfg = _cfg(leg1_dir, leg="leg1")
    assert driver._load_sibling_leg_manifests(cfg) == []


def test_third_leg_pools_both_earlier_siblings(tmp_path):
    expedition_root = tmp_path / "demo"
    for name in ("leg1", "leg2", "leg3"):
        (expedition_root / name).mkdir(parents=True)
    (expedition_root / "leg1" / "scored_manifest.json").write_text(json.dumps([{"tag": "l1"}]))
    (expedition_root / "leg2" / "scored_manifest.json").write_text(json.dumps([{"tag": "l2"}]))

    cfg = _cfg(expedition_root / "leg3", leg="leg3")
    pooled = driver._load_sibling_leg_manifests(cfg)

    assert {m["tag"] for m in pooled} == {"l1", "l2"}


def test_cli_requires_expedition_and_leg():
    parser = driver.main.__globals__["argparse"].ArgumentParser()
    # smoke-checks the real parser via main() itself rather than re-deriving its structure
    import pytest
    with pytest.raises(SystemExit):
        driver.main([])  # missing --expedition/--leg
```

- [x] **Step 2: Run to verify failure**

Run: `python3 -m pytest -q tests/test_driver_sibling_exclusion.py`
Expected: FAIL — `AttributeError: module 'clawmarks.search.driver' has no attribute
'_load_sibling_leg_manifests'`

- [x] **Step 3: Implement the sibling-leg pool and CLI**

Replace `_load_prev_round_state` in `src/clawmarks/search/driver.py` with:

```python
def _load_sibling_leg_manifests(cfg):
    """Every *other* leg directory within cfg's own expedition, concatenated, as the
    "already explored" exclusion pool -- generalizes round 2's one-off "exclude round 1"
    special case to any number of sibling legs. A brand-new expedition (or a leg with no
    sibling that has produced a manifest yet) returns []."""
    expedition_root = cfg.dir.parent
    manifests = []
    if not expedition_root.exists():
        return manifests
    for sibling_dir in sorted(expedition_root.iterdir()):
        if not sibling_dir.is_dir() or sibling_dir.name == cfg.leg:
            continue
        manifest_path = sibling_dir / "scored_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifests.extend(json.load(f))
    return manifests
```

Update `main()`'s argparse and every remaining `cfg.round`/`ROUND_CONFIGS` reference:

```python
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--expedition", required=True)
    parser.add_argument("--leg", required=True)
    parser.add_argument(
        "--use-predicted-preference", action="store_true", default=False,
        help="Stage 5b (opt-in, requires a trained preference_pairwise_model.joblib in this "
             "leg's directory and human validation via preference_rank.html first): rank the "
             "exploit pool by the trained model's predicted preference instead of favorited "
             "images. Defaults off; do not enable without having browsed preference_rank.html "
             "first.",
    )
    args = parser.parse_args(argv)
    cfg = load_leg_config(args.expedition, args.leg)

    import torch
    from clawmarks.search.score_manifest import (
        MODEL_ID, REAL_DIR, embed_images,
    )
    from transformers import AutoModel

    out_dir = cfg.dir
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(cfg)
    manifest = _load_resumable_manifest(out_dir)
    _validate_resume_agreement(state, manifest, _state_file(cfg), out_dir / "scored_manifest.json")
    if state["start_balance"] is None:
        state["start_balance"] = get_balance()
        save_state(cfg, state)
    start_time = state["start_time"]

    print("loading DINOv2 (once, kept warm across all generations)...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    real_paths = sorted(os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    real_embs = embed_images(real_paths, model=model)
    real_centroid = real_embs.mean(dim=0)
    real_centroid = real_centroid / real_centroid.norm()
    loo_sims = []
    for i in range(real_embs.shape[0]):
        others = torch.cat([real_embs[:i], real_embs[i + 1:]], dim=0)
        loo_c = others.mean(dim=0)
        loo_c = loo_c / loo_c.norm()
        loo_sims.append((real_embs[i] @ loo_c).item())
    real_ref = (sum(loo_sims) / len(loo_sims), min(loo_sims), max(loo_sims))
    print(f"real-image reference band: {real_ref}", flush=True)

    print("embedding every sibling leg's already-explored images as the exclusion set...", flush=True)
    prev_manifest = _load_sibling_leg_manifests(cfg)
    prev_paths = [m["file"] for m in prev_manifest if os.path.exists(m["file"])]
    prev_embs = embed_images(prev_paths, model=model) if prev_paths else None
    print(f"embedded {len(prev_paths)} sibling-leg images as the exclusion set" if prev_paths
          else "no sibling-leg manifests found yet; running without exclusion embeddings", flush=True)

    if cfg.seed_from_start:
        shared_pool_dict = seed_pool_load(out_dir / "seed_pool.json")
        shared_pool = list(shared_pool_dict.keys())
        print(f"loaded {len(shared_pool)} subjects from this leg's seed pool ({out_dir / 'seed_pool.json'})", flush=True)
        if not state["gpt55_subjects"]:
            print("seeding subject pool with GPT-5.5 from generation 1 (no plateau wait)...", flush=True)
            gpt_subjects = request_gpt55_subjects(cfg, cfg.fallback_subjects + shared_pool, n=30)
            state["gpt55_subjects"] = gpt_subjects
            if gpt_subjects:
                updated, _added = seed_pool_merge(
                    shared_pool_dict, gpt_subjects,
                    source=f"gpt5.5-{cfg.expedition}-{cfg.leg}",
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                seed_pool_save(out_dir / "seed_pool.json", updated)
            save_state(cfg, state)
        subjects = cfg.fallback_subjects + state["gpt55_subjects"] + shared_pool
        if not state["gpt55_subjects"]:
            print("GPT-5.5 handoff produced nothing usable; continuing with the fallback subject list only", flush=True)
    else:
        subjects = list(cfg.fallback_subjects)
    textures = list(cfg.textures)

    while True:
        elapsed_h = (time.time() - start_time) / 3600
        if elapsed_h > cfg.wall_clock_cap_hours:
            print(f"STOPPING: wall-clock cap reached ({elapsed_h:.2f}h > {cfg.wall_clock_cap_hours}h)", flush=True)
            break
        if state["generation"] >= cfg.max_generations:
            print(f"STOPPING: hit MAX_GENERATIONS sanity ceiling ({cfg.max_generations})", flush=True)
            break
        spent = _spent_or_none(state["start_balance"])
        if spent is None:
            break
        if abs(spent) > (cfg.budget_usd_cap - cfg.budget_safety_margin):
            print(f"STOPPING: projected spend ${abs(spent):.2f} crossed the "
                  f"${cfg.budget_usd_cap - cfg.budget_safety_margin:.2f} safety threshold "
                  f"(cap ${cfg.budget_usd_cap}, margin ${cfg.budget_safety_margin})", flush=True)
            break

        state["generation"] += 1
        gen = state["generation"]
        liminal_band_all = [m for m in manifest if real_ref[1] <= m["centroid_sim"] <= real_ref[2]]
        elites = sorted(liminal_band_all, key=lambda m: -m["novelty"])[:15]
        if not elites and not cfg.seed_from_start:
            elites = manifest[-30:] if manifest else []
        user_picks = _load_favorited_images(out_dir) if cfg.seed_from_start else []
        if args.use_predicted_preference:
            predicted_pool = _predicted_preference_pool(
                manifest, out_dir / "preference_pairwise_model.joblib", model,
            )
            if predicted_pool:
                user_picks = predicted_pool
            else:
                print("--use-predicted-preference set but no trained model found yet "
                       "(or nothing generated so far this leg); using favorited images "
                       "instead", flush=True)

        print(f"\n=== generation {gen} | elapsed {elapsed_h:.2f}h | spend ${abs(spent):.3f} | "
              f"stage {state['stage']} | plateau_count {state['plateau_count']} ===", flush=True)

        jobs = build_generation_jobs(gen, subjects, textures, elites, user_picks,
                                      cfg.gen_batch_size, cfg.explore_fraction,
                                      style_subject_count=cfg.style_subject_count)
        new_manifest = submit_and_collect(cfg, jobs, out_dir, f"gen{gen}")
        new_scored = score_batch(model, real_embs, real_centroid, new_manifest, prev_embs=prev_embs)
        manifest.extend(new_scored)
        _save_manifest(out_dir, manifest)

        best_novelty = build_gallery(cfg, manifest, real_ref) if manifest else 0.0
        state["novelty_history"].append(best_novelty)
        print(f"generation {gen}: {len(new_scored)} new images, cumulative {len(manifest)}, "
              f"liminal-band best novelty {best_novelty:.4f}", flush=True)

        hist = state["novelty_history"]
        if len(hist) > PLATEAU_WINDOW and max(hist[-PLATEAU_WINDOW:]) <= max(hist[:-PLATEAU_WINDOW]) + PLATEAU_EPSILON:
            state["plateau_count"] += 1
            if not cfg.seed_from_start:
                print(f"PLATEAU detected (count={state['plateau_count']}): best novelty hasn't "
                      f"improved by >{PLATEAU_EPSILON} over the last {PLATEAU_WINDOW} generations", flush=True)
                if state["stage"] == 0:
                    state["stage"] = 1
                    subjects = list(cfg.fallback_subjects) + list(cfg.widened_subjects)
                    textures = list(cfg.textures) + list(cfg.widened_textures)
                    print("SELF-IMPROVE stage 1: widened subject/texture vocabulary and "
                          "strength/CFG ranges for future generations", flush=True)
                elif state["stage"] == 1:
                    state["stage"] = 2
                    print("SELF-IMPROVE stage 2: local vocabulary widening didn't help either; "
                          "handing creative-subject generation off to GPT-5.5 via opencode so "
                          "fresh variety doesn't depend on this script's fixed lists", flush=True)
                    new_subjects = request_gpt55_subjects(cfg, subjects)
                    if new_subjects:
                        subjects = subjects + new_subjects
                        state["gpt55_subjects"] = state["gpt55_subjects"] + new_subjects
                    else:
                        print("gpt5.5 handoff produced nothing usable; continuing with the "
                              "widened deterministic vocabulary instead", flush=True)
            else:
                print(f"PLATEAU detected (count={state['plateau_count']})", flush=True)
                if state["plateau_count"] % 3 == 1:
                    more = request_gpt55_subjects(cfg, subjects, n=20)
                    if more:
                        subjects = subjects + more
                        state["gpt55_subjects"] = state["gpt55_subjects"] + more
                        shared_dict = seed_pool_load(out_dir / "seed_pool.json")
                        updated, _added = seed_pool_merge(
                            shared_dict, more, source=f"gpt5.5-{cfg.expedition}-{cfg.leg}",
                            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                        seed_pool_save(out_dir / "seed_pool.json", updated)
        save_state(cfg, state)

    print(f"\nLEG {cfg.expedition}/{cfg.leg} RUN ENDED at generation {state['generation']}, "
          f"{len(manifest)} total images, gallery at {out_dir / 'gallery.html'}", flush=True)
```

Update the remaining helpers that referenced `cfg.round`/`SWEEP_DIR`/`SEEDS_FILE` directly:

```python
def _load_favorited_images(out_dir):
    favorites_path = out_dir / "user_favorites.json"
    if not favorites_path.exists():
        return []
    with open(favorites_path) as f:
        favorites = json.load(f)
    return list(favorites.values())


def request_gpt55_subjects(cfg, existing_subjects, n=30):
    out_dir = cfg.dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gpt55_subjects.json"
    prompt = (
        f"Write {n} short, vivid, concrete visual scene or subject descriptions (5-15 words "
        f"each, no artist-style words, no medium words) suitable for testing where a "
        f"fine-tuned image-generation style survives on unfamiliar subject matter, versus "
        f"where it breaks down into visual noise. Favor liminal, uncanny, quietly unsettling "
        f"everyday scenes over gore or fantasy creatures. Prioritize genuinely different "
        f"categories of scene from each other (spaces, objects, weather, crowds, machines, "
        f"architecture), not variations on the same idea. Do not repeat or closely paraphrase "
        f"any of these already-used subjects: {json.dumps(existing_subjects)}. "
        f"Write ONLY a JSON array of {n} strings to the file {out_path}, nothing else in that "
        f"file. When done, print exactly: === DONE ==="
    )
    try:
        result = subprocess.run(
            ["opencode", "run", "--dir", str(clawmarks_config.ROOT), "--dangerously-skip-permissions",
             "-m", "openai/gpt-5.5", "--", prompt],
            capture_output=True, text=True, timeout=300,
        )
        print(f"[gpt5.5] exit={result.returncode} stdout_tail={result.stdout[-300:]!r}", flush=True)
    except Exception as e:
        print(f"[gpt5.5] FAILED to invoke opencode: {e}", flush=True)
        return []
    if out_path.exists():
        try:
            with open(out_path) as f:
                subjects = json.load(f)
            if isinstance(subjects, list) and subjects:
                print(f"[gpt5.5] got {len(subjects)} subjects", flush=True)
                return [str(s) for s in subjects]
        except Exception as e:
            print(f"[gpt5.5] couldn't parse {out_path}: {e}", flush=True)
    return []
```

Update the top-of-file import line (`from clawmarks.config import SEEDS_FILE, SWEEP2_DIR,
SWEEP_DIR` is removed; add `from clawmarks import config as clawmarks_config` instead), and update
`build_gallery`'s title/intro (it used `cfg.round`):

```python
    title = f"CLAWMARKS uncanny frontier atlas: {cfg.expedition}/{cfg.leg}"
    intro = cfg.description or f"Leg {cfg.leg} of expedition {cfg.expedition}."
```

(`build_gallery`'s body otherwise calls `_out_dir(cfg)`, which already resolves to `cfg.dir` from
Task 3 — no other change needed there.)

- [x] **Step 4: Run to verify pass**

Run: `python3 -m pytest -q tests/test_driver_sibling_exclusion.py tests/test_driver_state.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/clawmarks/search/driver.py tests/test_driver_sibling_exclusion.py tests/test_driver_state.py
git commit -m "feat(driver): pool every sibling leg's images as the novelty exclusion set"
```

---

### Task 5: `cli.py` — `--expedition`/`--leg`

**Files:**
- Modify: `src/clawmarks/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `driver.main(argv)` now expecting `--expedition`/`--leg` (Task 4).

- [x] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import subprocess

from clawmarks.cli import build_parser


def test_build_is_no_longer_a_valid_subcommand():
    result = subprocess.run(
        ["python", "-m", "clawmarks.cli", "build", "scan"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "invalid choice" in result.stdout


def test_run_allnight_expedition_and_leg_arguments_parse():
    parser = build_parser()
    args = parser.parse_args(["run", "allnight", "--expedition", "uncanny_frontier", "--leg", "round2"])
    assert args.command == "run"
    assert args.expedition == "uncanny_frontier"
    assert args.leg == "round2"


def test_run_allnight_requires_both_expedition_and_leg():
    import pytest
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "allnight", "--expedition", "uncanny_frontier"])


def test_serve_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
```

- [x] **Step 2: Run to verify failure**

Run: `python3 -m pytest -q tests/test_cli.py`
Expected: FAIL — `error: unrecognized arguments: --expedition uncanny_frontier --leg round2` (the
old parser still expects `--round`)

- [x] **Step 3: Update `cli.py`**

```python
    allnight_p = run_sub.add_parser("allnight")
    allnight_p.add_argument("--expedition", required=True)
    allnight_p.add_argument("--leg", required=True)
    allnight_p.add_argument(
        "--use-predicted-preference", action="store_true", default=False,
        help="Stage 5b: build the exploit pool from the trained preference model's top picks "
             "instead of yes-rated images. Defaults off; requires a trained preference model.",
    )
    allnight_p.set_defaults(command="run")
```

and in `main()`:

```python
    if args.command == "run":
        from clawmarks.search.driver import main as driver_main
        run_argv = ["--expedition", args.expedition, "--leg", args.leg]
        if args.use_predicted_preference:
            run_argv.append("--use-predicted-preference")
        return driver_main(run_argv)
```

- [x] **Step 4: Run to verify pass**

Run: `python3 -m pytest -q tests/test_cli.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/clawmarks/cli.py tests/test_cli.py
git commit -m "feat(cli): replace --round with --expedition/--leg"
```

---

### Task 6: `run_manager.py` — expedition/leg-keyed launch, lock, and report

**Files:**
- Modify: `src/clawmarks/search/run_manager.py`
- Test: `tests/test_run_manager.py`

**Interfaces:**
- Produces: `run_manager.launch_run(expedition: str, leg: str, out_dir, api_key, popen_fn=...,
  balance_fn=...) -> dict` (lock info now `{"expedition": ..., "leg": ..., "pid": ..., ...}`),
  `run_manager.build_report(out_dir, favorites=None, current_balance=None)` (state-file lookup
  simplified to `allnight_state.json` only).

- [x] **Step 1: Update the failing tests**

In `tests/test_run_manager.py`, every `{"pid": ..., "round": 1, ...}` lock-info dict becomes
`{"pid": ..., "expedition": "demo", "leg": "leg1", ...}`, and every `run_manager.launch_run(1,
...)` call becomes `run_manager.launch_run("demo", "leg1", ...)`. Apply this mechanically across
the whole file; two representative edits:

```python
def test_current_run_returns_info_for_live_pid(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    info = {"pid": os.getpid(), "expedition": "demo", "leg": "leg1", "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    assert run_manager.current_run() == info


def test_launch_run_backs_up_verifies_and_writes_lock(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)
    out_dir = tmp_path / "sweep"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text("[]")

    class FakeProc:
        pid = 12345

    info = run_manager.launch_run("demo", "leg1", out_dir, "fake-key",
                                   popen_fn=lambda *a, **k: FakeProc(),
                                   balance_fn=lambda key: 100.0)

    assert info["pid"] == 12345
    assert info["expedition"] == "demo"
    assert info["leg"] == "leg1"
    assert info["out_dir"] == str(out_dir)
    assert json.loads(lock_file.read_text()) == info
    backups = list(tmp_path.glob("sweep_backup_*"))
    assert len(backups) == 1
    assert (backups[0] / "manifest.json").read_text() == "[]"
```

Apply the same `1 -> "demo", "leg1"` (or `2 -> "demo", "leg2"` where the original test used round
2) substitution to every remaining `launch_run(...)` call and every hand-built lock-info dict in
the file (`test_current_run_clears_stale_lock_for_dead_pid`,
`test_launch_run_refuses_when_a_run_is_already_in_progress`,
`test_launch_run_refuses_when_balance_below_floor`,
`test_launch_run_skips_backup_when_out_dir_does_not_exist_yet`,
`test_status_reports_running_with_live_lock` — assert `result["expedition"] == "demo"` and
`result["leg"] == "leg2"` instead of `result["round"] == 2`, `test_stop_run_sends_sigterm_...`,
`test_stop_run_kills_the_whole_process_group_...`,
`test_stop_run_sigkills_after_grace_period_...`, `test_launch_run_is_race_free_...`,
`test_launch_run_reaps_the_child_...`, `test_current_run_treats_a_pid_reused_...`,
`test_stop_run_reaps_the_process_promptly_...`, `test_launch_run_records_pid_start_time_...`).

`test_build_report_reads_round2_state_file_name` renames to
`test_build_report_reads_allnight_state_file_name` and no longer needs a distinct name (there is
only one state filename now); keep its body (it already just writes `allnight_state.json` and
reads it back, which is now the *only* case — merge it into
`test_build_report_reads_novelty_trajectory_and_plateau_count_from_state` if you prefer, or leave
it as a second confirming test).

- [x] **Step 2: Run to verify failure**

Run: `python3 -m pytest -q tests/test_run_manager.py`
Expected: FAIL — `TypeError: launch_run() takes from 3 to 5 positional arguments but 6 were given`
(tests now call the new 2-positional-arg signature against the old implementation)

- [x] **Step 3: Update `run_manager.py`**

```python
def launch_run(expedition, leg, out_dir, api_key, popen_fn=subprocess.Popen, balance_fn=runpod_balance):
    current_run()

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise LaunchError("a search run is already in progress")

    proc = None
    try:
        balance = balance_fn(api_key)
        if balance < BALANCE_FLOOR_USD:
            raise LaunchError(f"balance ${balance:.2f} is below floor ${BALANCE_FLOOR_USD:.2f}")

        out_dir = Path(out_dir)
        if out_dir.exists():
            backup_dir = backup_out_dir(out_dir)
            verify_backup(out_dir, backup_dir)

        proc = popen_fn(
            [sys.executable, "-m", "clawmarks.search.driver",
             "--expedition", expedition, "--leg", leg],
            start_new_session=True,
        )
        info = {
            "pid": proc.pid, "expedition": expedition, "leg": leg,
            "started_at": time.time(), "out_dir": str(out_dir),
            "start_time_ticks": _process_start_time(proc.pid),
        }
        f = os.fdopen(fd, "w")
        fd = None
        with f:
            json.dump(info, f)
        return info
    except BaseException:
        if fd is not None:
            os.close(fd)
        _remove_lock()
        if proc is not None:
            proc.kill()
            proc.wait()
        raise
```

And `build_report`'s state-file lookup:

```python
def build_report(out_dir, favorites=None, current_balance=None):
    out_dir = Path(out_dir)
    favorites = favorites or {}

    state = {}
    state_file = out_dir / "allnight_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    manifest = []
    manifest_file = out_dir / "scored_manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)

    by_category = {}
    for m in manifest:
        cat = m.get("category", "unknown")
        counts = by_category.setdefault(cat, {"count": 0, "picked": 0})
        counts["count"] += 1
        if m.get("tag") in favorites:
            counts["picked"] += 1

    report = {
        "novelty_trajectory": state.get("novelty_history", []),
        "plateau_count": state.get("plateau_count", 0),
        "generation": state.get("generation", 0),
        "start_balance": state.get("start_balance"),
        "total_images": len(manifest),
        "explore_exploit_split": {cat: v["count"] for cat, v in by_category.items()},
        "pick_rate_by_category": {
            cat: (v["picked"] / v["count"] if v["count"] else 0.0)
            for cat, v in by_category.items()
        },
    }
    if current_balance is not None and state.get("start_balance") is not None:
        report["spend"] = state["start_balance"] - current_balance
    return report
```

(`status()`, `stop_run()`, and every other function in the file are unchanged — they only ever
read/pass through the lock-info dict, never construct it.)

- [x] **Step 4: Run to verify pass**

Run: `python3 -m pytest -q tests/test_run_manager.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/clawmarks/search/run_manager.py tests/test_run_manager.py
git commit -m "feat(run_manager): key launches, locks, and reports by expedition/leg"
```

---

### Task 7: Delete `merge_round2.py` and `preference_model.py` (superseded, zero live callers)

**Files:**
- Delete: `src/clawmarks/build/merge_round2.py`, `tests/test_merge_round2.py`,
  `src/clawmarks/search/preference_model.py`, `tests/test_preference_model.py`

**Interfaces:** none (pure deletion; nothing else imports either module — confirmed via `rg -ln
"clawmarks.search.preference_model|clawmarks.build.merge_round2" src/` returning only the modules
themselves and their own tests).

- [x] **Step 1: Confirm no other caller exists**

Run: `rg -n "merge_round2|search\.preference_model|search import preference_model" src/ tests/ --glob '!tests/test_merge_round2.py' --glob '!tests/test_preference_model.py' --glob '!src/clawmarks/build/merge_round2.py' --glob '!src/clawmarks/search/preference_model.py'`
Expected: no output (confirms both are safe to delete outright)

- [x] **Step 2: Delete the files**

```bash
git rm src/clawmarks/build/merge_round2.py tests/test_merge_round2.py
git rm src/clawmarks/search/preference_model.py tests/test_preference_model.py
```

- [x] **Step 3: Run the full suite to confirm nothing references them**

Run: `python3 -m pytest -q`
Expected: no collection errors from the removed modules (other failures are expected at this
point in the plan — later tasks fix them — but there must be zero `ModuleNotFoundError` or
`ImportError` mentioning `merge_round2` or `preference_model`)

- [x] **Step 4: Commit**

```bash
git commit -m "chore: delete merge_round2.py and preference_model.py, both superseded with no live callers"
```

---

### Task 8: Parameterize `embed_cache.py`, `preference_pairwise_model.py`, `preference_settings.py`, `score_manifest.py`, `migrate_picks_to_ratings.py` — drop their `SWEEP_DIR` imports

**Files:**
- Modify: `src/clawmarks/search/embed_cache.py`, `src/clawmarks/search/preference_pairwise_model.py`,
  `src/clawmarks/search/preference_settings.py`, `src/clawmarks/search/score_manifest.py`,
  `src/clawmarks/search/migrate_picks_to_ratings.py`
- Test: `tests/test_preference_settings.py`, `tests/test_preference_pairwise_model.py`,
  `tests/test_score_manifest.py` (any test currently monkeypatching a module's own `SWEEP_DIR`
  import or a `*_FILE` module constant)

**Interfaces:**
- Produces: `preference_settings.load(out_dir) -> dict`, `preference_settings.save(enabled,
  out_dir)` (both now take an explicit directory instead of reading a fixed module constant);
  `preference_pairwise_model.train_and_save(..., out_dir)` and its `MODEL_FILE`/`MODEL_META_FILE`
  become functions `model_file(out_dir)`/`model_meta_file(out_dir)`; `embed_cache.sync(...,
  embeddings_file, ...)` already takes its cache file path as a parameter (unchanged) — only its
  module-level `EMBEDDINGS_FILE` default and the two `SWEEP_DIR`-based path helpers
  (`image_path_for`-style helpers used only by its own tests) move to explicit parameters;
  `score_manifest.main(argv)` takes an explicit output directory instead of defaulting to
  `SWEEP_DIR`; `migrate_picks_to_ratings.main(argv)` takes `--expedition`/`--leg`.

- [x] **Step 1: Update `preference_settings.py` and its tests first (smallest, clearest case)**

```python
# src/clawmarks/search/preference_settings.py
"""
Single persisted setting shared by archive.html's rendering and `clawmarks run allnight`'s
exploit-pool source, so flipping predicted-preference on or off happens in one place instead
of two independent controls (a query param and a CLI flag). See
docs/superpowers/specs/2026-07-10-preference-toggle-design.md.

Takes an explicit out_dir (the active leg's directory) rather than a fixed module constant,
since there is no longer one process-wide sweep directory.
"""
import json
import os


def load(out_dir):
    """Returns {"use_predicted_preference": bool}. Missing file means the default, False."""
    path = out_dir / "preference_settings.json"
    if not os.path.exists(path):
        return {"use_predicted_preference": False}
    with open(path) as f:
        return json.load(f)


def save(enabled, out_dir):
    path = out_dir / "preference_settings.json"
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"use_predicted_preference": bool(enabled)}, f)
    os.replace(tmp, path)
```

In `tests/test_preference_settings.py`, every `monkeypatch.setattr(preference_settings,
"PREFERENCE_SETTINGS_FILE", some_path)` followed by `preference_settings.load()` /
`preference_settings.save(x)` becomes a direct call with the directory:

```python
def test_load_missing_file_returns_default(tmp_path):
    assert preference_settings.load(tmp_path) == {"use_predicted_preference": False}


def test_save_then_load_round_trips(tmp_path):
    preference_settings.save(True, tmp_path)
    assert preference_settings.load(tmp_path) == {"use_predicted_preference": True}
```

(Apply this same "drop the monkeypatch, pass `tmp_path` positionally instead" transform to every
test in the file; there is no remaining need to monkeypatch a module constant since the function
signature now takes the directory directly.)

- [x] **Step 2: Run to verify**

Run: `python3 -m pytest -q tests/test_preference_settings.py`
Expected: PASS

- [x] **Step 3: Update `preference_pairwise_model.py` and its tests**

Change the two module constants to functions, and thread `out_dir` through
`train_and_save`/`score`/wherever `MODEL_FILE`/`MODEL_META_FILE`/`comparisons_path` were read:

```python
# top of src/clawmarks/search/preference_pairwise_model.py — remove:
#   from clawmarks.config import SWEEP_DIR
#   MODEL_FILE = SWEEP_DIR / "preference_pairwise_model.joblib"
#   MODEL_META_FILE = SWEEP_DIR / "preference_pairwise_model_meta.json"
# replace with:

def model_file(out_dir):
    return out_dir / "preference_pairwise_model.joblib"


def model_meta_file(out_dir):
    return out_dir / "preference_pairwise_model_meta.json"
```

Update `train_and_save` (and any other function referencing `MODEL_FILE`/`MODEL_META_FILE`
directly) to accept `out_dir` as a parameter and call `model_file(out_dir)`/
`model_meta_file(out_dir)` instead of the removed constants; update the one `comparisons_path =
SWEEP_DIR / "user_comparisons.json"` line to `comparisons_path = out_dir /
"user_comparisons.json"` with `out_dir` threaded in from the same caller. Update
`tests/test_preference_pairwise_model.py`: every `monkeypatch.setattr(preference_pairwise_model,
"MODEL_FILE", ...)`/`"MODEL_META_FILE"` becomes a direct `tmp_path`-based call, e.g.:

```python
def test_train_and_save_writes_model_and_meta(tmp_path, ...):
    preference_pairwise_model.train_and_save(..., out_dir=tmp_path)
    assert preference_pairwise_model.model_file(tmp_path).exists()
    assert preference_pairwise_model.model_meta_file(tmp_path).exists()
```

- [x] **Step 4: Run to verify**

Run: `python3 -m pytest -q tests/test_preference_pairwise_model.py`
Expected: PASS

- [x] **Step 5: Update `embed_cache.py`**

```python
# remove: from clawmarks.config import SWEEP_DIR
#         EMBEDDINGS_FILE = SWEEP_DIR / "embeddings.npz"
# The three internal helpers that read SWEEP_DIR directly (loading scored_manifest.json,
# resolving a tag's full-res file, resolving its thumb path) take an explicit out_dir instead:

def embeddings_file(out_dir):
    return out_dir / "embeddings.npz"
```

Update the function bodies at the three `SWEEP_DIR`-referencing lines to take `out_dir` as a
parameter (threaded from whatever already calls them — `sync(...)`'s existing `image_path_for`
callback parameter already lets the *caller* supply file-path resolution, so check whether the
`with open(SWEEP_DIR / "scored_manifest.json")` line is inside a test-only helper or in `sync`
itself before deciding whether it needs a new parameter or can be deleted as dead code reached
only by the module's own tests; resolve this by reading the surrounding function with `Read`
before editing, since the exact call graph determines whether this is a signature change or a
deletion).

- [x] **Step 6: Run to verify**

Run: `python3 -m pytest -q tests/test_embed_cache.py`
Expected: PASS

- [x] **Step 7: Update `score_manifest.py`**

```python
# remove: from clawmarks.config import ROOT, SWEEP_DIR
# add:    from clawmarks.config import ROOT

def _default_manifest(out_dir):
    full = out_dir / "manifest.json"
    partial = out_dir / "manifest_partial.json"
    if full.exists():
        return str(full)
    if partial.exists():
        print(f"NOTE: {full} not found yet, building from partial results ({partial}). "
              f"Some planned jobs may still be missing; that's fine, this doesn't wait for "
              f"100% completion.", flush=True)
        return str(partial)
    raise FileNotFoundError("neither manifest.json nor manifest_partial.json exists yet")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        raise SystemExit("usage: python -m clawmarks.search.score_manifest <out_dir> [manifest_path]")
    out_dir = Path(argv[0])
    manifest_path = argv[1] if len(argv) > 1 else _default_manifest(out_dir)
    ...  # body unchanged except every "{SWEEP_DIR}/..." becomes "{out_dir}/..."
```

Add `from pathlib import Path` to the imports if not already present. Replace every remaining
`f"{SWEEP_DIR}/..."` in the function body (`quarantine_file`, the two final `atomic_json_write`
calls, and the closing print) with `f"{out_dir}/..."`.

- [x] **Step 8: Run to verify**

Run: `python3 -m pytest -q tests/test_score_manifest.py`
Expected: PASS

- [x] **Step 9: Update `migrate_picks_to_ratings.py`**

```python
"""One-time migration: user_picks.json entries become user_ratings.json entries with
label: "yes", so "pick as winner" can be retired without losing the existing picks. Safe to
rerun: any tag that already has a rating is left alone.

Run with: python -m clawmarks.search.migrate_picks_to_ratings --expedition <name> --leg <name>
"""
import argparse
import json
import os

from clawmarks import config


def merge_picks_into_ratings(picks, ratings):
    updated = dict(ratings)
    migrated = []
    for tag, pick in picks.items():
        if tag in updated:
            continue
        updated[tag] = {"label": "yes", "rated_at": pick.get("picked_at")}
        migrated.append(tag)
    return updated, migrated


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--expedition", required=True)
    parser.add_argument("--leg", required=True)
    args = parser.parse_args(argv)
    out_dir = config.leg_dir(args.expedition, args.leg)

    user_picks_file = out_dir / "user_picks.json"
    user_ratings_file = out_dir / "user_ratings.json"

    picks = {}
    if user_picks_file.exists():
        with open(user_picks_file) as f:
            picks = json.load(f)
    ratings = {}
    if user_ratings_file.exists():
        with open(user_ratings_file) as f:
            ratings = json.load(f)

    updated, migrated = merge_picks_into_ratings(picks, ratings)
    tmp = str(user_ratings_file) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(updated, f, indent=1)
    os.replace(tmp, user_ratings_file)

    print(f"migrated {len(migrated)} picks into {user_ratings_file} as yes-ratings "
          f"({len(picks) - len(migrated)} already had a rating and were left alone)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 10: Run the full search-package test slice**

Run: `python3 -m pytest -q tests/test_preference_settings.py tests/test_preference_pairwise_model.py tests/test_embed_cache.py tests/test_score_manifest.py`
Expected: PASS

- [x] **Step 11: Commit**

```bash
git add src/clawmarks/search/embed_cache.py src/clawmarks/search/preference_pairwise_model.py \
        src/clawmarks/search/preference_settings.py src/clawmarks/search/score_manifest.py \
        src/clawmarks/search/migrate_picks_to_ratings.py \
        tests/test_preference_settings.py tests/test_preference_pairwise_model.py \
        tests/test_embed_cache.py tests/test_score_manifest.py
git commit -m "refactor(search): make every SWEEP_DIR-derived path explicit and leg-relative"
```

---

### Task 9: `curation_server.py` — active-leg selection core

**Files:**
- Modify: `src/clawmarks/curation_server.py`
- Test: `tests/test_curation_server_active_leg.py` (new)

**Interfaces:**
- Consumes: `config.EXPEDITIONS_DIR`, `config.ACTIVE_LEG_FILE`, `config.leg_dir` (Task 1).
- Produces: `curation_server._active_out_dir() -> Path | None`,
  `curation_server._set_active_selection(expedition, leg)` (raises `ValueError` if the expedition
  doesn't exist), `GET /api/active-leg -> {"expedition": ..., "leg": ...} | {"expedition": None,
  "leg": None}`, `POST /api/active-leg` body `{"expedition": ..., "leg": ...}`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_curation_server_active_leg.py
import json

import pytest

from clawmarks import curation_server as cs
from clawmarks import config


@pytest.fixture(autouse=True)
def _reset_active_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", tmp_path / "active_leg.json")
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None
    yield


def test_active_out_dir_is_none_before_any_selection():
    assert cs._active_out_dir() is None


def test_set_active_selection_persists_and_resolves(tmp_path):
    (config.EXPEDITIONS_DIR / "demo").mkdir(parents=True)
    (config.EXPEDITIONS_DIR / "demo" / "expedition.json").write_text("{}")

    cs._set_active_selection("demo", "leg1")

    assert cs._active_out_dir() == config.leg_dir("demo", "leg1")
    assert json.loads(config.ACTIVE_LEG_FILE.read_text()) == {"expedition": "demo", "leg": "leg1"}


def test_set_active_selection_rejects_unknown_expedition():
    with pytest.raises(ValueError, match="unknown expedition"):
        cs._set_active_selection("does_not_exist", "leg1")


def test_load_active_selection_restores_from_disk(tmp_path):
    (config.EXPEDITIONS_DIR / "demo").mkdir(parents=True)
    (config.EXPEDITIONS_DIR / "demo" / "expedition.json").write_text("{}")
    config.ACTIVE_LEG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.ACTIVE_LEG_FILE.write_text(json.dumps({"expedition": "demo", "leg": "leg1"}))

    cs._load_active_selection()

    assert cs._active_out_dir() == config.leg_dir("demo", "leg1")
```

- [x] **Step 2: Run to verify failure**

Run: `python3 -m pytest -q tests/test_curation_server_active_leg.py`
Expected: FAIL — `AttributeError: module 'clawmarks.curation_server' has no attribute
'_active_selection'`

- [x] **Step 3: Implement the active-leg core**

Near the top of `src/clawmarks/curation_server.py`, replace the line
`from clawmarks.config import ROOT, SEEDS_FILE, SWEEP2_DIR, SWEEP_DIR` with
`from clawmarks import config` and `from clawmarks.config import ROOT`, and remove
`from clawmarks.search.driver import ROUND_CONFIGS` entirely. Then add, right after the
`_live_cache = LiveCache()` line:

```python
from clawmarks.atomic_io import atomic_json_write

_active_selection = {"expedition": None, "leg": None}


def _load_active_selection():
    if config.ACTIVE_LEG_FILE.exists():
        with open(config.ACTIVE_LEG_FILE) as f:
            data = json.load(f)
        _active_selection["expedition"] = data.get("expedition")
        _active_selection["leg"] = data.get("leg")


_load_active_selection()


def _active_out_dir():
    if _active_selection["expedition"] is None:
        return None
    return config.leg_dir(_active_selection["expedition"], _active_selection["leg"])


def _set_active_selection(expedition, leg):
    expedition_file = config.EXPEDITIONS_DIR / expedition / "expedition.json"
    if not expedition_file.exists():
        raise ValueError(f"unknown expedition {expedition!r}")
    _active_selection["expedition"] = expedition
    _active_selection["leg"] = leg
    atomic_json_write(config.ACTIVE_LEG_FILE, dict(_active_selection))
```

Add the two routes. In `_do_GET`, add near the top of the method:

```python
        if self.path == "/api/active-leg":
            self._json_response(200, dict(_active_selection))
            return
```

In `do_POST`, add:

```python
        if self.path == "/api/active-leg":
            expedition = payload.get("expedition")
            leg = payload.get("leg")
            if not expedition or not leg:
                self._json_response(400, {"error": "'expedition' and 'leg' are required"})
                return
            try:
                _set_active_selection(expedition, leg)
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            self._json_response(200, dict(_active_selection))
            return
```

- [x] **Step 4: Run to verify pass**

Run: `python3 -m pytest -q tests/test_curation_server_active_leg.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/clawmarks/curation_server.py tests/test_curation_server_active_leg.py
git commit -m "feat(server): add server-side active expedition/leg selection"
```

---

### Task 10: `curation_server.py` — convert every `SWEEP_DIR`-derived constant and reference to `_active_out_dir()`

**Files:**
- Modify: `src/clawmarks/curation_server.py`

**Interfaces:**
- Consumes: `_active_out_dir()` (Task 9).
- Produces: `_favorites_file()`, `_comparisons_file()`, `_counterfactuals_dir()`,
  `_counterfactuals_file()`, `_cockpit_queue_file()`, `_seeds_file()` — all functions now,
  replacing the module-level `FAVORITES_FILE`/`COMPARISONS_FILE`/`COUNTERFACTUALS_DIR`/
  `COUNTERFACTUALS_FILE`/`COCKPIT_QUEUE_FILE`/`SEEDS_FILE` constants. `_manifest_path()` keeps its
  name (already a function) but its body changes.

This task is mechanical: no new logic, only threading `_active_out_dir()` through every place that
used to read the fixed `SWEEP_DIR`. There is no dedicated new test; this task's correctness is
verified by the existing route tests once Task 15 updates their fixtures to select an active leg
instead of monkeypatching `SWEEP_DIR` directly. Do not skip running the affected tests after this
task, even though they're rewritten later — a broken import or `NameError` here should surface
immediately via a quick manual check.

- [x] **Step 1: Remove the six module-level `*_FILE`/`*_DIR` constants and add functions**

Delete these lines (originally right after `_preference_retrain_gate_error`):

```python
FAVORITES_FILE = f"{SWEEP_DIR}/user_favorites.json"
COMPARISONS_FILE = f"{SWEEP_DIR}/user_comparisons.json"
COUNTERFACTUALS_DIR = f"{SWEEP_DIR}/counterfactuals"
COUNTERFACTUALS_FILE = f"{SWEEP_DIR}/user_counterfactuals.json"
COCKPIT_QUEUE_FILE = f"{SWEEP_DIR}/cockpit_queue.json"
```

Replace with:

```python
def _favorites_file():
    return _active_out_dir() / "user_favorites.json"


def _comparisons_file():
    return _active_out_dir() / "user_comparisons.json"


def _counterfactuals_dir():
    return _active_out_dir() / "counterfactuals"


def _counterfactuals_file():
    return _active_out_dir() / "user_counterfactuals.json"


def _cockpit_queue_file():
    return _active_out_dir() / "cockpit_queue.json"


def _seeds_file():
    return _active_out_dir() / "candidate_seeds.json"
```

Remove `SEEDS_FILE` from the `from clawmarks.config import ...` line (already done in Task 9's
import-line rewrite).

- [x] **Step 2: Update every call site (add parentheses, one file, mechanical)**

Every bare reference to `FAVORITES_FILE` becomes `_favorites_file()`; `COMPARISONS_FILE` becomes
`_comparisons_file()`; `COUNTERFACTUALS_DIR` becomes `_counterfactuals_dir()`;
`COUNTERFACTUALS_FILE` becomes `_counterfactuals_file()`; `COCKPIT_QUEUE_FILE` becomes
`_cockpit_queue_file()`; `SEEDS_FILE` becomes `_seeds_file()`. Find every occurrence with:

```bash
rg -n "FAVORITES_FILE\b|COMPARISONS_FILE\b|COUNTERFACTUALS_DIR\b|COUNTERFACTUALS_FILE\b|COCKPIT_QUEUE_FILE\b|SEEDS_FILE\b" src/clawmarks/curation_server.py
```

and change each one from a bare name to a call (e.g. `load_store(FAVORITES_FILE)` becomes
`load_store(_favorites_file())`). This includes the `_preference_status_watched_files()`
function's `COMPARISONS_FILE` reference and every handler method (`_handle_cockpit_evidence`,
`_handle_cockpit_run`, `_run_cockpit_trial`, `_handle_seed_generate`, `_handle_cockpit_autopilot`,
`_reconcile_stuck_trials`, and the `GET`/`POST` branches for `/api/favorites`, `/api/favorite`,
`/api/unfavorite`, `/api/counterfactuals`, `/api/counterfactual`, `/api/seeds`,
`/api/seeds/generate`, `/api/cockpit/queue`).

- [x] **Step 3: Update the `_manifest_path`/`sweep_dir=str(SWEEP_DIR)` call sites**

```python
def _manifest_path():
    return str(_active_out_dir() / "scored_manifest.json")
```

Every `sweep_dir=str(SWEEP_DIR)` keyword argument (in `_get_scan_items`, `_get_solution_map_data`,
`_get_map_data`, `_get_redundancy_data`, `_get_manifest_cached`, `_get_preference_status_data`,
and the inline `_live_cache.get(..., sweep_dir=str(SWEEP_DIR))` calls for `/archive.html` and
`/preference_rank.html`) becomes `sweep_dir=str(_active_out_dir())`. `_solution_map_watched_files`
similarly replaces `f"{SWEEP_DIR}/solution_map_final_embs.pt"` with
`str(_active_out_dir() / "solution_map_final_embs.pt")`.

- [x] **Step 4: Update the remaining scattered references**

- `item_summary(a, SWEEP_DIR)` / `item_summary(b, SWEEP_DIR)` (in `next_compare_response`) and
  `item_summary(m, SWEEP_DIR)` (in `cockpit_evidence`) become `item_summary(a, _active_out_dir())`
  etc.
- `load_manifest()`'s `path = f"{SWEEP_DIR}/scored_manifest.json"` becomes `path =
  _active_out_dir() / "scored_manifest.json"` (and the `os.path.getmtime(path)` call below it
  still works unchanged against a `Path`).
- `_load_scored_manifest`/`_save_scored_manifest`'s `f"{SWEEP_DIR}/scored_manifest.json"` become
  `_active_out_dir() / "scored_manifest.json"`.
- `Handler.__init__`'s `directory=SWEEP_DIR` becomes:
  ```python
  def __init__(self, *args, **kwargs):
      active_dir = _active_out_dir()
      super().__init__(*args, directory=str(active_dir) if active_dir else str(config.STATE_DIR), **kwargs)
  ```
  (falls back to `STATE_DIR`, a directory guaranteed to exist, when no leg is selected yet —
  static file serving is meaningless before any expedition exists, and the empty-state hub, not
  static serving, handles that case; see Task 12).
- The two status-page `f"sweep dir: <code>{html.escape(str(SWEEP_DIR))}</code>"` lines become
  `f"sweep dir: <code>{html.escape(str(_active_out_dir() or 'none selected'))}</code>"`.
- `/thumbs/` and `/real_thumbs/` handlers' `f"{SWEEP_DIR}{self.path}"` and
  `f"{SWEEP_DIR}/real_thumbs/{name}"` become `str(_active_out_dir() / self.path.lstrip("/"))` and
  `str(_active_out_dir() / "real_thumbs" / name)` respectively.
- `_handle_seed_generate`'s `tmp_path = f"{SWEEP_DIR}/candidate_seeds_gen_{int(time.time())}.json"`
  becomes `tmp_path = str(_active_out_dir() / f"candidate_seeds_gen_{int(time.time())}.json")`.
- `_handle_cockpit_autopilot`'s equivalent `tmp_path` line follows the same pattern with
  `cockpit_autopilot_{int(time.time())}.json`.
- `_check_manifest_images`'s `manifest_path = f"{SWEEP_DIR}/scored_manifest.json"` becomes
  conditional on an active leg existing at all — see Task 12's startup handling, which replaces
  this function's call site in `main()`.
- `main()`'s final `print(f"serving {SWEEP_DIR} + ratings API on {host}:{port}", ...)` becomes
  `print(f"serving on {host}:{port} (active leg: {_active_out_dir() or 'none selected'})", ...)`.

- [x] **Step 5: Sanity-check with a quick import**

Run: `python3 -c "import clawmarks.curation_server"`
Expected: no `NameError`/`ImportError` (this only proves the module imports cleanly; full
behavioral verification happens once Task 15 updates the route tests)

- [x] **Step 6: Commit**

```bash
git add src/clawmarks/curation_server.py
git commit -m "refactor(server): resolve every SWEEP_DIR reference through the active leg selection"
```

---

### Task 11: `curation_server.py` — startup no longer requires an active leg

**Files:**
- Modify: `src/clawmarks/curation_server.py`
- Test: `tests/test_curation_server_startup.py` (new)

**Interfaces:**
- Produces: `curation_server._check_manifest_images()` now a no-op when no leg is active (instead
  of crashing on a missing `SWEEP_DIR`).

- [x] **Step 1: Write the failing test**

```python
# tests/test_curation_server_startup.py
from clawmarks import curation_server as cs
from clawmarks import config


def test_check_manifest_images_is_a_noop_with_no_active_leg(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None

    cs._check_manifest_images()  # should not raise, print a warning, or sys.exit
```

- [x] **Step 2: Run to verify failure**

Run: `python3 -m pytest -q tests/test_curation_server_startup.py`
Expected: FAIL — `TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'` (or
similar, since `_active_out_dir()` returns `None`)

- [x] **Step 3: Guard `_check_manifest_images`**

```python
def _check_manifest_images():
    active_dir = _active_out_dir()
    if active_dir is None:
        return  # nothing selected yet; the empty-state hub handles this case
    manifest_path = active_dir / "scored_manifest.json"
    if not manifest_path.exists():
        print(f"warning: no scored_manifest.json at {manifest_path}, skipping image check", flush=True)
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    n_total = len(manifest)
    if n_total == 0:
        return
    n_present = sum(1 for m in manifest if os.path.exists(m["file"]))
    if n_present == 0:
        example = manifest[0]["file"]
        print(
            f"FATAL: none of {n_total} images in {manifest_path} exist on disk "
            f"(e.g. {example!r} is missing). This usually means the manifest's paths are "
            "stale, most likely from the project directory being renamed or moved. Fix the "
            "manifest's 'file' paths, or select a different expedition/leg via "
            "POST /api/active-leg, before starting the server.",
            file=sys.stderr, flush=True,
        )
        sys.exit(1)
    if n_present < n_total:
        print(f"warning: only {n_present}/{n_total} manifest images found on disk", flush=True)
```

- [x] **Step 4: Run to verify pass**

Run: `python3 -m pytest -q tests/test_curation_server_startup.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/clawmarks/curation_server.py tests/test_curation_server_startup.py
git commit -m "fix(server): don't crash at startup when no expedition/leg is selected yet"
```

---

### Task 12: `curation_server.py` — empty-state hub becomes an expedition/leg picker + create-expedition form

**Files:**
- Modify: `src/clawmarks/curation_server.py`
- Test: `tests/test_curation_server_expedition_routes.py` (new)

**Interfaces:**
- Produces: `GET /api/expeditions -> {"expeditions": [{"name": ..., "legs": [...]}]}`,
  `POST /api/expeditions` body `{"name", "trigger_word", "negative_prompt", "textures",
  "fallback_subjects", "budget_usd_cap", "budget_safety_margin", "gen_batch_size",
  "explore_fraction", "max_generations"}` -> writes `expedition.json` and scaffolds the standing
  `cockpit` leg (`legs/cockpit.json` = `"{}"`, an empty leg dir), returns `{"ok": true, "name":
  ...}` or `400` if the name already exists.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_curation_server_expedition_routes.py
import json

import pytest

from clawmarks import curation_server as cs
from clawmarks import config


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", tmp_path / "state" / "active_leg.json")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None
    yield


def test_list_expeditions_empty_when_none_exist():
    assert cs._list_expeditions() == []


def test_create_expedition_writes_config_and_scaffolds_cockpit_leg():
    payload = {
        "name": "demo", "trigger_word": "trentbuckle style, ",
        "negative_prompt": "low quality, blurry, watermark",
        "textures": ["tex-a"], "fallback_subjects": ["subj-a"],
        "budget_usd_cap": 5.0, "budget_safety_margin": 0.5,
        "gen_batch_size": 20, "explore_fraction": 0.5, "max_generations": 100,
    }
    result = cs._create_expedition(payload)

    assert result == {"ok": True, "name": "demo"}
    expedition_file = config.EXPEDITIONS_DIR / "demo" / "expedition.json"
    assert json.loads(expedition_file.read_text())["trigger_word"] == "trentbuckle style, "
    cockpit_leg_file = config.EXPEDITIONS_DIR / "demo" / "legs" / "cockpit.json"
    assert cockpit_leg_file.exists()
    assert config.leg_dir("demo", "cockpit").exists()


def test_create_expedition_rejects_a_name_that_already_exists():
    payload = {"name": "demo", "textures": [], "fallback_subjects": []}
    cs._create_expedition(payload)

    with pytest.raises(ValueError, match="already exists"):
        cs._create_expedition(payload)


def test_list_expeditions_reports_every_leg():
    cs._create_expedition({"name": "demo", "textures": [], "fallback_subjects": []})
    (config.EXPEDITIONS_DIR / "demo" / "legs" / "round1.json").write_text("{}")

    expeditions = cs._list_expeditions()

    assert len(expeditions) == 1
    assert expeditions[0]["name"] == "demo"
    assert set(expeditions[0]["legs"]) == {"cockpit", "round1"}
```

- [x] **Step 2: Run to verify failure**

Run: `python3 -m pytest -q tests/test_curation_server_expedition_routes.py`
Expected: FAIL — `AttributeError: module 'clawmarks.curation_server' has no attribute
'_list_expeditions'`

- [x] **Step 3: Implement `_list_expeditions`/`_create_expedition` and their routes**

```python
def _list_expeditions():
    if not config.EXPEDITIONS_DIR.exists():
        return []
    result = []
    for expedition_dir in sorted(config.EXPEDITIONS_DIR.iterdir()):
        if not (expedition_dir / "expedition.json").exists():
            continue
        legs_dir = expedition_dir / "legs"
        legs = sorted(p.stem for p in legs_dir.glob("*.json")) if legs_dir.exists() else []
        result.append({"name": expedition_dir.name, "legs": legs})
    return result


def _create_expedition(payload):
    name = (payload.get("name") or "").strip()
    if not name:
        raise ValueError("'name' is required")
    expedition_dir = config.EXPEDITIONS_DIR / name
    if expedition_dir.exists():
        raise ValueError(f"expedition {name!r} already exists")

    expedition_fields = {
        "trigger_word": payload.get("trigger_word", ""),
        "negative_prompt": payload.get("negative_prompt", ""),
        "textures": payload.get("textures", []),
        "fallback_subjects": payload.get("fallback_subjects", []),
        "budget_usd_cap": payload.get("budget_usd_cap", 1.0),
        "budget_safety_margin": payload.get("budget_safety_margin", 0.1),
        "gen_batch_size": payload.get("gen_batch_size", 20),
        "explore_fraction": payload.get("explore_fraction", 0.5),
        "max_generations": payload.get("max_generations", 60),
    }
    (expedition_dir / "legs").mkdir(parents=True)
    atomic_json_write(expedition_dir / "expedition.json", expedition_fields)
    atomic_json_write(expedition_dir / "legs" / "cockpit.json", {})
    config.leg_dir(name, "cockpit").mkdir(parents=True, exist_ok=True)
    return {"ok": True, "name": name}
```

Add routes. In `_do_GET`:

```python
        if self.path == "/api/expeditions":
            self._json_response(200, {"expeditions": _list_expeditions()})
            return
```

In `do_POST`:

```python
        if self.path == "/api/expeditions":
            try:
                result = _create_expedition(payload)
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            self._json_response(200, result)
            return
```

- [x] **Step 4: Run to verify pass**

Run: `python3 -m pytest -q tests/test_curation_server_expedition_routes.py`
Expected: PASS

- [x] **Step 5: Replace the hardcoded "Launch Round 1"/"Launch Round 2" empty-state hub**

Rewrite `_status_page_empty_body` to render an expedition/leg picker plus a link to a
create-expedition form, and drive both through the two new endpoints and `/api/active-leg`:

```python
    def _status_page_empty_body(self, manifest_summary):
        expeditions = _list_expeditions()
        rows = "".join(
            f'<div class="exp-row"><strong>{html.escape(e["name"])}</strong> '
            + " ".join(
                f'<button class="leg-btn" data-expedition="{html.escape(e["name"])}" '
                f'data-leg="{html.escape(leg)}">{html.escape(leg)}</button>'
                for leg in e["legs"]
            )
            + "</div>"
            for e in expeditions
        )
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>clawmarks curation server</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee;
  --text-dim:#9a9aa4; --accent:#7c9eff; --down:#e0605e; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
h1 {{ font-size:18px; margin:0 0 4px; }}
p {{ color:var(--text-dim); font-size:13px; line-height:1.6; }}
p.sub {{ max-width:640px; }}
code {{ color:var(--text); }}
a {{ color:var(--accent); }}
.panel {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
  padding:16px; margin-top:16px; max-width:640px; }}
.exp-row {{ margin:8px 0; }}
button {{ font-size:13px; padding:6px 12px; border-radius:6px; border:1px solid var(--border);
  background:var(--accent); color:#0b0b0d; font-weight:600; cursor:pointer; }}
button:disabled {{ opacity:0.4; cursor:not-allowed; }}
#pickError {{ color:var(--down); font-size:12.5px; margin-top:8px; }}
</style></head><body>
<h1>clawmarks curation server</h1>
<p>{html.escape(manifest_summary)}</p>
<div class="panel">
<p class="sub">No expedition/leg selected. Pick an existing leg below, or create a new
expedition first if this is a genuinely new line of work.</p>
{rows or '<p class="sub">No expeditions exist yet.</p>'}
<div id="pickError"></div>
</div>
<script>
document.querySelectorAll('.leg-btn').forEach(btn => btn.addEventListener('click', () => {{
  document.getElementById('pickError').textContent = '';
  fetch('/api/active-leg', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{expedition: btn.dataset.expedition, leg: btn.dataset.leg}}),
  }}).then(async r => {{
    const d = await r.json();
    if (!r.ok) {{
      document.getElementById('pickError').textContent = d.error || 'selection failed';
    }} else {{
      location.reload();
    }}
  }});
}}));
</script>
</body></html>""".encode()
```

(The exact wording and visual polish of the create-expedition form is explicitly out of scope
per the design doc; this task wires the picker's read/select loop through the two real
endpoints. A follow-up UI pass can add the create-expedition form's HTML without touching any
server-side logic, since `POST /api/expeditions` already exists.)

- [x] **Step 6: Commit**

```bash
git add src/clawmarks/curation_server.py tests/test_curation_server_expedition_routes.py
git commit -m "feat(server): replace the round1/round2 launch hub with an expedition/leg picker"
```

---

### Task 13: `curation_server.py` — `/api/searchrun/launch`, `/report`, `/status` become expedition/leg-keyed

**Files:**
- Modify: `src/clawmarks/curation_server.py`
- Test: `tests/test_curation_server_searchrun_routes.py`

**Interfaces:**
- Consumes: `run_manager.launch_run(expedition, leg, ...)` (Task 6).

- [x] **Step 1: Update the failing tests**

```python
# tests/test_curation_server_searchrun_routes.py
import json
import os
import threading
from http.server import HTTPServer
import urllib.error
import urllib.request

import pytest

from clawmarks import config
from clawmarks import curation_server as cs
from clawmarks.search import run_manager


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "LOCK_FILE", tmp_path / ".searchrun.lock")
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key")
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", tmp_path / "state" / "active_leg.json")
    (config.EXPEDITIONS_DIR / "demo" / "legs").mkdir(parents=True)
    (config.EXPEDITIONS_DIR / "demo" / "expedition.json").write_text("{}")
    (config.EXPEDITIONS_DIR / "demo" / "legs" / "leg1.json").write_text("{}")
    cs._set_active_selection("demo", "leg1")
    out_dir = config.leg_dir("demo", "leg1")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scored_manifest.json").write_text("[]")
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, out_dir
    server.shutdown()
    thread.join(timeout=2)


def _post_json(url, payload):
    req = urllib.request.Request(
        url, method="POST", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _get_json(url):
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_status_reports_not_running_when_idle(running_server):
    server, _ = running_server
    port = server.server_address[1]

    status, data = _get_json(f"http://127.0.0.1:{port}/api/searchrun/status")

    assert status == 200
    assert data == {"running": False}


def test_launch_starts_a_run_and_status_reflects_it(running_server, monkeypatch):
    server, _ = running_server
    port = server.server_address[1]
    monkeypatch.setattr(run_manager, "runpod_balance", lambda key: 100.0)

    class FakeProc:
        pid = os.getpid()

    captured = {}

    def fake_popen(*a, **k):
        captured["args"] = a
        captured["kwargs"] = k
        return FakeProc()

    monkeypatch.setattr(cs.subprocess, "Popen", fake_popen)

    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/searchrun/launch",
        {"expedition": "demo", "leg": "leg1"},
    )

    assert status == 200
    assert data["ok"] is True
    assert data["pid"] == FakeProc.pid
    assert captured["kwargs"]["start_new_session"] is True

    status, data = _get_json(f"http://127.0.0.1:{port}/api/searchrun/status")
    assert status == 200
    assert data["running"] is True
    assert data["expedition"] == "demo"
    assert data["leg"] == "leg1"


def test_launch_refuses_when_already_running(running_server, monkeypatch):
    server, _ = running_server
    port = server.server_address[1]
    monkeypatch.setattr(run_manager, "runpod_balance", lambda key: 100.0)

    class FakeProc:
        pid = os.getpid()

    monkeypatch.setattr(cs.subprocess, "Popen", lambda *a, **k: FakeProc())

    status, _ = _post_json(f"http://127.0.0.1:{port}/api/searchrun/launch", {"expedition": "demo", "leg": "leg1"})
    assert status == 200

    status, data = _post_json(f"http://127.0.0.1:{port}/api/searchrun/launch", {"expedition": "demo", "leg": "leg1"})
    assert status == 409
    assert "already" in data["error"]


def test_launch_refuses_when_balance_below_floor(running_server, monkeypatch):
    server, _ = running_server
    port = server.server_address[1]
    monkeypatch.setattr(run_manager, "runpod_balance", lambda key: 0.0)

    status, data = _post_json(f"http://127.0.0.1:{port}/api/searchrun/launch", {"expedition": "demo", "leg": "leg1"})

    assert status == 402
    assert "floor" in data["error"]


def test_launch_rejects_unknown_leg(running_server):
    server, _ = running_server
    port = server.server_address[1]

    status, data = _post_json(
        f"http://127.0.0.1:{port}/api/searchrun/launch",
        {"expedition": "demo", "leg": "does_not_exist"},
    )

    assert status == 400


def test_stop_is_noop_when_nothing_running(running_server):
    server, _ = running_server
    port = server.server_address[1]

    status, data = _post_json(f"http://127.0.0.1:{port}/api/searchrun/stop", {})

    assert status == 200
    assert data == {"running": False}


def test_report_reflects_state_and_manifest_on_disk(running_server):
    server, out_dir = running_server
    port = server.server_address[1]
    (out_dir / "allnight_state.json").write_text(json.dumps({
        "generation": 2, "stage": 0, "plateau_count": 1,
        "novelty_history": [0.3, 0.35], "gpt55_subjects": [],
        "start_balance": 5.0, "start_time": 1.0,
    }))
    (out_dir / "scored_manifest.json").write_text(json.dumps([
        {"tag": "gen1_explore_0", "category": "r2_explore"},
    ]))

    status, data = _get_json(f"http://127.0.0.1:{port}/api/searchrun/report?expedition=demo&leg=leg1")

    assert status == 200
    assert data["novelty_trajectory"] == [0.3, 0.35]
    assert data["plateau_count"] == 1
    assert data["total_images"] == 1


def test_stop_terminates_a_running_run(running_server, monkeypatch):
    server, _ = running_server
    port = server.server_address[1]
    monkeypatch.setattr(run_manager, "runpod_balance", lambda key: 100.0)

    class FakeProc:
        pid = os.getpid()

    monkeypatch.setattr(cs.subprocess, "Popen", lambda *a, **k: FakeProc())

    _post_json(f"http://127.0.0.1:{port}/api/searchrun/launch", {"expedition": "demo", "leg": "leg1"})

    monkeypatch.setattr(run_manager, "is_process_alive", lambda pid: False)
    monkeypatch.setattr(run_manager.os, "kill", lambda pid, sig: None)

    status, data = _post_json(f"http://127.0.0.1:{port}/api/searchrun/stop", {})

    assert status == 200
    assert data == {"running": False}
```

- [x] **Step 2: Run to verify failure**

Run: `python3 -m pytest -q tests/test_curation_server_searchrun_routes.py`
Expected: FAIL — old handler still expects `{"round": ...}`

- [x] **Step 3: Update the launch/report handlers**

```python
        if self.path.startswith("/api/searchrun/report"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            expedition = (query.get("expedition") or [None])[0]
            leg = (query.get("leg") or [None])[0]
            if not expedition or not leg:
                self._json_response(400, {"error": "'expedition' and 'leg' query params are required"})
                return
            out_dir = config.leg_dir(expedition, leg)
            favorites = load_store(_favorites_file())
            self._json_response(200, run_manager.build_report(out_dir, favorites=favorites))
            return
```

```python
    def _handle_searchrun_launch(self, payload):
        expedition = payload.get("expedition")
        leg = payload.get("leg")
        if not expedition or not leg:
            self._json_response(400, {"error": "'expedition' and 'leg' are required"})
            return
        leg_file = config.EXPEDITIONS_DIR / expedition / "legs" / f"{leg}.json"
        expedition_file = config.EXPEDITIONS_DIR / expedition / "expedition.json"
        if not expedition_file.exists():
            self._json_response(400, {"error": f"unknown expedition {expedition!r}"})
            return
        if not leg_file.exists() and leg != "cockpit":
            self._json_response(400, {"error": f"unknown leg {leg!r} in expedition {expedition!r}"})
            return

        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            self._json_response(400, {"error": "RUNPOD_API_KEY not set in server environment"})
            return

        out_dir = config.leg_dir(expedition, leg)
        try:
            info = run_manager.launch_run(
                expedition, leg, out_dir, api_key,
                popen_fn=subprocess.Popen, balance_fn=run_manager.runpod_balance,
            )
        except run_manager.LaunchError as e:
            message = str(e)
            if "already in progress" in message:
                self._json_response(409, {"error": message})
            elif "below floor" in message:
                self._json_response(402, {"error": message})
            else:
                self._json_response(400, {"error": message})
            return
        self._json_response(200, {"ok": True, **info})
```

- [x] **Step 4: Run to verify pass**

Run: `python3 -m pytest -q tests/test_curation_server_searchrun_routes.py`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/clawmarks/curation_server.py tests/test_curation_server_searchrun_routes.py
git commit -m "feat(server): key /api/searchrun/launch and /report by expedition/leg"
```

---

### Task 14: Cockpit — expedition-targeted trials and sibling-leg exclusion embeddings

**Files:**
- Modify: `src/clawmarks/curation_server.py`, `src/clawmarks/build/cockpit.py`
- Test: `tests/test_curation_server_cockpit_scoring.py` (new)

**Interfaces:**
- Produces: `cockpit.render_html(expeditions: list[dict], current_expedition: str | None)` (was
  `render_html()`, no args). Cockpit's "expedition selector" reuses `/api/active-leg` with
  `leg="cockpit"` fixed, so cockpit trials always land in the selected expedition's standing
  `cockpit` leg without introducing a second, parallel selection mechanism.
  `curation_server._cockpit_scoring_context()` now also loads the current expedition's
  sibling-leg exclusion embeddings (mirroring `driver.py`'s `_load_sibling_leg_manifests`), instead
  of always scoring with `prev_embs=None`.

- [x] **Step 1: Write the failing test**

```python
# tests/test_curation_server_cockpit_scoring.py
import json

import pytest
import torch

from clawmarks import config
from clawmarks import curation_server as cs


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", tmp_path / "state" / "active_leg.json")
    (config.EXPEDITIONS_DIR / "demo" / "legs").mkdir(parents=True)
    (config.EXPEDITIONS_DIR / "demo" / "expedition.json").write_text("{}")
    (config.EXPEDITIONS_DIR / "demo" / "legs" / "cockpit.json").write_text("{}")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None
    cs._set_active_selection("demo", "cockpit")
    cs._cockpit_scoring_state["model"] = None
    cs._cockpit_scoring_state["real_embs"] = None
    cs._cockpit_scoring_state["real_centroid"] = None
    yield


def test_score_cockpit_batch_pools_sibling_leg_images_as_exclusion(monkeypatch, tmp_path):
    from clawmarks.search import driver

    sibling_dir = config.leg_dir("demo", "round1")
    sibling_dir.mkdir(parents=True)
    (sibling_dir / "scored_manifest.json").write_text(json.dumps([{"tag": "r1_a", "file": "a.png"}]))

    captured = {}

    def fake_score_batch(model, real_embs, real_centroid, manifest_batch, prev_embs=None):
        captured["prev_embs"] = prev_embs
        return manifest_batch

    monkeypatch.setattr(driver, "score_batch", fake_score_batch)
    monkeypatch.setattr(
        cs, "_cockpit_scoring_context",
        lambda: (None, torch.zeros(1, 4), torch.zeros(4)),
    )
    monkeypatch.setattr(
        cs, "_sibling_leg_exclusion_embeddings",
        lambda expedition, leg, model: torch.ones(1, 4),
    )

    cs.score_cockpit_batch([], {"id": "t1", "mission": "freeform"})

    assert captured["prev_embs"] is not None
    assert captured["prev_embs"].shape == (1, 4)
```

- [x] **Step 2: Run to verify failure**

Run: `python3 -m pytest -q tests/test_curation_server_cockpit_scoring.py`
Expected: FAIL — `AttributeError: module 'clawmarks.curation_server' has no attribute
'_sibling_leg_exclusion_embeddings'`

- [x] **Step 3: Implement sibling-leg exclusion embeddings for cockpit scoring**

```python
def _sibling_leg_exclusion_embeddings(expedition, leg, model):
    """Mirrors driver.py's _load_sibling_leg_manifests + embedding step, but scoped to the
    curation server's own long-lived DINOv2 instance (see _cockpit_scoring_context) instead
    of loading a fresh model per call."""
    from clawmarks.search.driver import _load_sibling_leg_manifests
    from clawmarks.search.score_manifest import embed_images

    class _Cfg:
        pass
    fake_cfg = _Cfg()
    fake_cfg.dir = config.leg_dir(expedition, leg)
    fake_cfg.leg = leg

    sibling_manifest = _load_sibling_leg_manifests(fake_cfg)
    paths = [m["file"] for m in sibling_manifest if os.path.exists(m["file"])]
    if not paths:
        return None
    return embed_images(paths, model=model)


def score_cockpit_batch(results, trial):
    from clawmarks.search.driver import score_batch

    model, real_embs, real_centroid = _cockpit_scoring_context()
    prev_embs = _sibling_leg_exclusion_embeddings(
        _active_selection["expedition"], _active_selection["leg"], model,
    )
    scored = score_batch(model, real_embs, real_centroid, results, prev_embs=prev_embs)
    for m in scored:
        m["prompt_type"] = "cockpit"
        m["category"] = "cockpit"
        m["round"] = 0
        m["trial_id"] = trial["id"]
        m["mission"] = trial["mission"]
    return scored
```

(Passing a lightweight duck-typed object instead of a real `LegConfig` keeps
`_load_sibling_leg_manifests` reusable without importing `LegConfig` here; it only reads
`cfg.dir` and `cfg.leg`, both present on `fake_cfg`.)

- [x] **Step 4: Run to verify pass**

Run: `python3 -m pytest -q tests/test_curation_server_cockpit_scoring.py`
Expected: PASS

- [x] **Step 5: Add the cockpit expedition selector (reuses `/api/active-leg`)**

Update `cockpit.render_html` to accept the expedition list and render a selector that posts to
`/api/active-leg` with `leg` fixed to `"cockpit"`:

```python
def render_html(expeditions=None, current_expedition=None):
    expeditions = expeditions or []
    options = "".join(
        f'<option value="{html.escape(e)}"{" selected" if e == current_expedition else ""}>{html.escape(e)}</option>'
        for e in expeditions
    )
    selector = f"""<div class="expedition-picker">
<label>Expedition: <select id="expeditionSelect">{options}</select></label>
<button id="expeditionSwitch">Switch</button>
</div>
<script>
document.getElementById('expeditionSwitch').addEventListener('click', () => {{
  const expedition = document.getElementById('expeditionSelect').value;
  fetch('/api/active-leg', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{expedition, leg: 'cockpit'}}),
  }}).then(() => location.reload());
}});
</script>"""
    # ... prepend `selector` to the existing rendered page body (read the current render_html
    # implementation with Read before editing, to splice this in at the right point rather than
    # guessing its exact template structure)
```

In `curation_server.py`'s `/cockpit.html` route, pass the expedition list through:

```python
        if self.path == "/cockpit.html":
            body = cockpit.render_html(
                expeditions=[e["name"] for e in _list_expeditions()],
                current_expedition=_active_selection["expedition"],
            ).encode()
```

- [x] **Step 6: Run the cockpit test slice**

Run: `python3 -m pytest -q tests/test_curation_server_cockpit_scoring.py tests/test_cockpit.py`
Expected: PASS (read `tests/test_cockpit.py` first if it exists and asserts on `render_html()`'s
old zero-arg signature; update any such assertion to pass the new optional arguments or rely on
their defaults)

- [x] **Step 7: Commit**

```bash
git add src/clawmarks/curation_server.py src/clawmarks/build/cockpit.py tests/test_curation_server_cockpit_scoring.py
git commit -m "feat(cockpit): pool sibling-leg exclusion embeddings and add an expedition selector"
```

---

### Task 15: Migrate every remaining test that monkeypatches `SWEEP_DIR`/`SWEEP2_DIR` directly

**Files:**
- Modify: `tests/test_curation_server_counterfactual_route.py`,
  `tests/test_curation_server_real_thumbs.py`, `tests/test_curation_server_lazy_thumbnails.py`,
  `tests/test_curation_server_compare_routes.py`, `tests/test_curation_server_static_assets.py`,
  `tests/test_curation_server_seeds_route.py`,
  `tests/test_curation_server_manifest_only_routes_cache.py`,
  `tests/test_curation_server_manifest_cache.py`, `tests/test_curation_server_solution_map_dep.py`,
  `tests/test_curation_server_map_redundancy_cache.py`, `tests/test_curation_server_scan_route.py`,
  `tests/test_favorited_images.py`, `tests/test_curation_server_preference_status_route.py`

**Interfaces:** none new; this task only fixes existing tests to match Tasks 9–14's mechanism.

- [x] **Step 1: Apply the mechanical transform to each file**

Every one of these files currently does, in some form:

```python
monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
```

Replace with:

```python
monkeypatch.setattr(cs, "_active_out_dir", lambda: tmp_path)
```

This is a safe, uniform substitution because every production call site that used to read
`SWEEP_DIR` now calls `_active_out_dir()` instead (Task 10) — pointing that one function at
`tmp_path` reproduces the exact same test isolation the old monkeypatch gave, with no other
changes needed in most of these files. Two files need one extra line beyond the substitution:

- `tests/test_curation_server_solution_map_dep.py`: identical transform (`SWEEP_DIR` → mock
  `_active_out_dir`); no other changes, since this file already isolates via `LiveCache()` and
  `solution_map.compute_data` monkeypatches that don't reference round/leg concepts at all.
- `tests/test_curation_server_seeds_route.py`: replace `monkeypatch.setattr(cs, "SEEDS_FILE",
  tmp_path / "candidate_seeds.json")` with `monkeypatch.setattr(cs, "_active_out_dir", lambda:
  tmp_path)` (the file constant is gone; `_seeds_file()` now derives from `_active_out_dir()`
  automatically, so pointing the latter at `tmp_path` is sufficient and the old
  `candidate_seeds.json` suffix no longer needs spelling out separately).
- `tests/test_curation_server_preference_status_route.py`: this file also monkeypatches
  `preference_settings.PREFERENCE_SETTINGS_FILE` directly (removed in Task 8) — additionally
  replace every `monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", ...)` /
  `monkeypatch.setattr(cs.preference_settings, "PREFERENCE_SETTINGS_FILE", ...)` by instead
  checking how the route under test calls `preference_settings.load()`/`.save()` — since those
  now require an explicit `out_dir` argument (Task 8), and the route itself resolves that
  argument via `_active_out_dir()` (once its call sites are updated as part of this task's
  mechanical `SWEEP_DIR` → `_active_out_dir` pass at line ~1024 and ~1199 in
  `curation_server.py`, which Task 10 already covered) — no monkeypatch of
  `PREFERENCE_SETTINGS_FILE` remains necessary at all; delete those monkeypatch lines rather than
  translate them.
- `tests/test_favorited_images.py`: same base transform; if it also imports
  `driver._load_favorited_images()` with the old zero-argument signature (removed in Task 4),
  update the call to pass the directory explicitly: `driver._load_favorited_images(tmp_path)`.

- [x] **Step 2: Run each migrated file individually**

Run: `python3 -m pytest -q tests/test_curation_server_counterfactual_route.py tests/test_curation_server_real_thumbs.py tests/test_curation_server_lazy_thumbnails.py tests/test_curation_server_compare_routes.py tests/test_curation_server_static_assets.py tests/test_curation_server_seeds_route.py tests/test_curation_server_manifest_only_routes_cache.py tests/test_curation_server_manifest_cache.py tests/test_curation_server_solution_map_dep.py tests/test_curation_server_map_redundancy_cache.py tests/test_curation_server_scan_route.py tests/test_favorited_images.py tests/test_curation_server_preference_status_route.py`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add tests/test_curation_server_counterfactual_route.py tests/test_curation_server_real_thumbs.py \
        tests/test_curation_server_lazy_thumbnails.py tests/test_curation_server_compare_routes.py \
        tests/test_curation_server_static_assets.py tests/test_curation_server_seeds_route.py \
        tests/test_curation_server_manifest_only_routes_cache.py tests/test_curation_server_manifest_cache.py \
        tests/test_curation_server_solution_map_dep.py tests/test_curation_server_map_redundancy_cache.py \
        tests/test_curation_server_scan_route.py tests/test_favorited_images.py \
        tests/test_curation_server_preference_status_route.py
git commit -m "test(server): migrate every remaining SWEEP_DIR monkeypatch to the active-leg mechanism"
```

---

### Task 16: `uncanny_frontier` reference expedition (round 1 / round 2 as worked example)

**Files:**
- Create: `expeditions/uncanny_frontier/expedition.json`, `expeditions/uncanny_frontier/legs/round1.json`,
  `expeditions/uncanny_frontier/legs/round2.json`, `expeditions/uncanny_frontier/legs/cockpit.json`

**Interfaces:** none (static config, no code reads these until someone launches a leg against
`uncanny_frontier` by name).

- [x] **Step 1: Write `expedition.json` from round 1's shared defaults**

```bash
mkdir -p expeditions/uncanny_frontier/legs
```

```json
{
  "trigger_word": "trentbuckle style, ",
  "negative_prompt": "low quality, blurry, watermark",
  "textures": [
    "marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
    "dark-rimmed eyes glowing pale blue, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting"
  ],
  "fallback_subjects": [
    "close-up cat portrait", "close-up wolf portrait", "close-up fox portrait",
    "close-up owl portrait", "close-up horse portrait",
    "close-up human face, pale skin, hand pressed beside cheek",
    "close-up cyborg face, half exposed circuitry and wiring, clawed metal hand pressed beside cheek",
    "close-up face mid-transformation, skin splitting to reveal clawed fingers pushing through the cheek",
    "figure standing alone in an empty fluorescent-lit hallway, clawed hand pressed against the wall",
    "dental x-ray radiograph of a jaw", "empty concrete stairwell viewed from below",
    "television weather map with swirling storm system", "crowd of human faces packed close together"
  ],
  "budget_usd_cap": 10.0,
  "budget_safety_margin": 1.5,
  "gen_batch_size": 60,
  "explore_fraction": 0.5,
  "max_generations": 400,
  "description": "Reference expedition: round 1 and round 2 of the original uncanny-frontier search (2026-07-09). Both rounds' full-resolution images are permanently gone (see the lab notebook's 2026-07-09 and 2026-07-14 entries); these files preserve the parameter record as a worked example only, not usable image data."
}
```

Write this to `expeditions/uncanny_frontier/expedition.json`.

- [x] **Step 2: Write `legs/round1.json` and `legs/round2.json`**

`expeditions/uncanny_frontier/legs/round1.json`:

```json
{
  "wall_clock_cap_hours": 7.5,
  "seed_from_start": false,
  "style_subject_count": 5,
  "widened_textures": [
    "loose watercolor wash bleeding at the edges, raw sketchbook page, mixed media",
    "heavy black ink crosshatching over torn found-paper collage edges, raw outsider-art painting"
  ],
  "widened_subjects": [
    "dollhouse interior seen through a broken window",
    "empty parking garage at night, one flickering light",
    "wall of surveillance camera monitors, mostly static",
    "vending machine humming alone in a dark hallway",
    "mannequin display missing its head",
    "storm drain grate half-submerged in still water",
    "airport terminal at night, all gates empty",
    "abandoned playground, swing set mid-motion with no one on it",
    "elevator interior, doors closing on an empty hallway",
    "waiting room with rows of identical empty chairs"
  ]
}
```

`expeditions/uncanny_frontier/legs/round2.json`:

```json
{
  "wall_clock_cap_hours": 1.0,
  "budget_usd_cap": 1.00,
  "budget_safety_margin": 0.10,
  "gen_batch_size": 20,
  "explore_fraction": 0.85,
  "max_generations": 60,
  "seed_from_start": true,
  "style_subject_count": 4,
  "textures": [
    "marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
    "dark-rimmed eyes glowing pale blue, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
    "loose watercolor wash bleeding at the edges, raw sketchbook page, mixed media",
    "heavy black ink crosshatching over torn found-paper collage edges, raw outsider-art painting"
  ],
  "fallback_subjects": [
    "close-up cat portrait", "close-up wolf portrait", "close-up fox portrait",
    "close-up human face, pale skin, hand pressed beside cheek",
    "dollhouse interior seen through a broken window",
    "empty parking garage at night, one flickering light",
    "wall of surveillance camera monitors, mostly static",
    "vending machine humming alone in a dark hallway",
    "abandoned playground, swing set mid-motion with no one on it",
    "waiting room with rows of identical empty chairs"
  ]
}
```

`expeditions/uncanny_frontier/legs/cockpit.json` (the standing leg every expedition gets):

```json
{}
```

- [x] **Step 3: Verify the merge loads cleanly**

Run: `python3 -c "
from clawmarks.search.driver import load_leg_config
r1 = load_leg_config('uncanny_frontier', 'round1')
r2 = load_leg_config('uncanny_frontier', 'round2')
assert r1.seed_from_start is False and r1.style_subject_count == 5
assert r2.seed_from_start is True and r2.explore_fraction == 0.85
print('OK')
"`
Expected: prints `OK` with no traceback

- [x] **Step 4: Commit**

```bash
git add expeditions/uncanny_frontier
git commit -m "docs(expeditions): add uncanny_frontier as a reference expedition (round1/round2 params only, no image data)"
```

---

### Task 17: Full test suite and manual server smoke check

**Files:** none (verification only)

- [x] **Step 1: Run the full suite**

Run: `python3 -m pytest -q`
Expected: PASS, zero failures, zero collection errors. If anything still references `SWEEP_DIR`,
`ROUND_CONFIGS`, `RoundConfig`, or `--round`, grep for it and fix the remaining call site before
considering this task done:

```bash
rg -n "SWEEP_DIR|SWEEP2_DIR|ROUND_CONFIGS|RoundConfig|--round\b" src/ tests/
```

Expected: no output.

- [x] **Step 2: Manual live check per this project's UI-change rule**

Start the server against a freshly created test expedition and confirm the empty-state hub,
expedition creation, leg selection, and cockpit page all render without error:

```bash
CLAWMARKS_STATE_DIR=$(mktemp -d) python3 -m clawmarks.cli serve 8421 &
sleep 1
curl -s http://127.0.0.1:8421/ | head -20
curl -s -X POST http://127.0.0.1:8421/api/expeditions \
  -H 'Content-Type: application/json' \
  -d '{"name": "smoke_test", "textures": [], "fallback_subjects": []}'
curl -s -X POST http://127.0.0.1:8421/api/active-leg \
  -H 'Content-Type: application/json' \
  -d '{"expedition": "smoke_test", "leg": "cockpit"}'
curl -s http://127.0.0.1:8421/cockpit.html | head -20
kill %1
```

Expected: the first `curl` shows the empty-state hub HTML with no expeditions listed; the
`POST /api/expeditions` call returns `{"ok": true, "name": "smoke_test"}`; the `POST
/api/active-leg` call returns `{"expedition": "smoke_test", "leg": "cockpit"}`; the final `curl`
returns real HTML (not a 500) for `/cockpit.html`. Per this project's convention for live-server
changes, do this manual check even though the automated suite passed — it verifies the page
actually renders, not just unit-level correctness.

- [x] **Step 3: Commit (if Steps 1–2 required any follow-up fixes)**

```bash
git add -A
git commit -m "fix: address remaining SWEEP_DIR/ROUND_CONFIGS stragglers found during full-suite verification"
```

(Skip this commit if Steps 1–2 passed clean with no further changes.)
