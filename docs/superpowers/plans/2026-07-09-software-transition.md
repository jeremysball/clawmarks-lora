# CLAWMARKS Package Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 31 scattered CLAWMARKS tooling scripts (`notes/*.py` + root RunPod scripts)
into an installable `src/clawmarks` package with one CLI entry point, centralized path config,
and unit tests for the pure-logic pieces, per
`docs/superpowers/specs/2026-07-09-software-transition-design.md`.

**Architecture:** Move files roughly as-is into subpackages by role (`compute/`, `search/`,
`probe/`, `build/`), replacing every hardcoded `SC = "/workspace/trent-with-smart-prompts"` with
one `config.repo_root()`. Merge `run_uncanny_allnight.py`/`run_uncanny_allnight2.py` into one
`search/driver.py` parameterized by round. Add an `argparse`-based `clawmarks` CLI. Delete the
old files once a full `clawmarks build all` run produces byte-identical HTML/JSON to the current
scripts.

**Tech Stack:** Python 3, stdlib `argparse` (no new CLI dependency), `uv` for all dependency
management (pinned versions, per project convention), `pytest` for the new unit tests.

## Global Constraints

- Never hardcode `RUNPOD_API_KEY`/`CIVITAI_TOKEN`/`CIVITAI_MODEL_ID`; read via
  `os.environ["NAME"]`, sourced from `.envrc`.
- Install every Python package with `uv add <pkg>==<exact-version>` or `uv pip install
  <pkg>==<exact-version>`: never bare `pip install`, never an unpinned version.
- No em dashes (`—`) or ` -- ` as a stand-in, anywhere: code comments, docstrings, commit
  messages. Grep for both before finishing any task that adds prose.
- Every moved file's behavior must stay identical unless a task explicitly says otherwise (only
  the driver merge and seed-pool dedup change behavior, and only by construction: same output,
  less duplication).
- Commit after every task with a Conventional Commits message (`feat`, `fix`, `chore`, `docs`,
  `refactor`, `test`), imperative mood.
- This plan produces one branch; do not open a PR or merge until Task 14 (final smoke check)
  passes.

---

### Task 1: Scaffold the package and `config.py`

**Files:**
- Create: `pyproject.toml`
- Create: `src/clawmarks/__init__.py`
- Create: `src/clawmarks/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `clawmarks.config.repo_root() -> pathlib.Path`, `clawmarks.config.SWEEP_DIR`,
  `clawmarks.config.SWEEP2_DIR`, `clawmarks.config.PROBE_DIR`, `clawmarks.config.PROBE_STRENGTH_DIR`
  (all `pathlib.Path`, resolved at import time from `repo_root()`).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "clawmarks"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[project.scripts]
clawmarks = "clawmarks.cli:main"

[build-system]
requires = ["setuptools>=69.0.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `src/clawmarks/__init__.py`** (empty file)

- [ ] **Step 3: Write the failing test for `config.py`**

```python
# tests/test_config.py
import os
from clawmarks import config


def test_repo_root_finds_pyproject():
    root = config.repo_root()
    assert (root / "pyproject.toml").exists()


def test_repo_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAWMARKS_ROOT", str(tmp_path))
    assert config.repo_root() == tmp_path


def test_derived_paths_under_repo_root():
    root = config.repo_root()
    assert config.SWEEP_DIR == root / "notes" / "uncanny_sweep"
    assert config.SWEEP2_DIR == root / "notes" / "uncanny_sweep2"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /workspace/trent-with-smart-prompts && uv run pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'clawmarks'` or similar: package doesn't exist yet)

- [ ] **Step 5: Write `src/clawmarks/config.py`**

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
SWEEP_DIR = NOTES_DIR / "uncanny_sweep"
SWEEP2_DIR = NOTES_DIR / "uncanny_sweep2"
PROBE_DIR = NOTES_DIR / "probe_uncanny"
PROBE_STRENGTH_DIR = NOTES_DIR / "probe_strength"
SEEDS_FILE = SWEEP_DIR / "candidate_seeds.json"
USER_PICKS_FILE = SWEEP_DIR / "user_picks.json"
```

- [ ] **Step 6: Install the package editable and run the test**

Run: `cd /workspace/trent-with-smart-prompts && uv pip install -e . && uv run pytest tests/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git checkout -b clawmarks-package-transition
git add pyproject.toml src/clawmarks/__init__.py src/clawmarks/config.py tests/test_config.py
git commit -m "chore(clawmarks): scaffold installable package with centralized path config"
```

---

### Task 2: `search/scoring.py`: extract pure scoring math with tests

**Files:**
- Create: `src/clawmarks/search/__init__.py`
- Create: `src/clawmarks/search/scoring.py`
- Test: `tests/test_scoring.py`
- Reference (read-only, do not modify yet): `notes/run_uncanny_allnight2.py:245-290` (`score_batch`,
  `build_gallery`'s `bin_edges`/`bin_of`)

**Interfaces:**
- Consumes: nothing new (pure functions over lists/tensors).
- Produces: `clawmarks.search.scoring.bin_edges(vals: list[float], n: int) -> list[float]`,
  `clawmarks.search.scoring.bin_of(val: float, edges: list[float]) -> int`,
  `clawmarks.search.scoring.novelty_from_similarity(nn_sim: float) -> float` (returns `1 - nn_sim`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring.py
from clawmarks.search.scoring import bin_edges, bin_of, novelty_from_similarity


def test_bin_edges_splits_sorted_values_into_n_groups():
    vals = sorted([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    edges = bin_edges(vals, 4)
    assert len(edges) == 3
    assert edges == sorted(edges)


def test_bin_of_returns_last_bin_for_max_value():
    edges = [0.25, 0.5, 0.75]
    assert bin_of(0.9, edges) == 3


def test_bin_of_returns_first_bin_for_min_value():
    edges = [0.25, 0.5, 0.75]
    assert bin_of(0.1, edges) == 0


def test_novelty_from_similarity_inverts_similarity():
    assert novelty_from_similarity(0.3) == 1 - 0.3
    assert novelty_from_similarity(1.0) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'clawmarks.search'`)

- [ ] **Step 3: Write `src/clawmarks/search/__init__.py`** (empty file)

- [ ] **Step 4: Write `src/clawmarks/search/scoring.py`**

Port `bin_edges`/`bin_of` exactly as they exist in `notes/run_uncanny_allnight2.py:277-287`
(`build_gallery`'s nested functions), pulled out to module level, plus the new
`novelty_from_similarity` wrapper around the `1 - ns` expression at
`notes/run_uncanny_allnight2.py:257`:

```python
def bin_edges(vals: list[float], n: int) -> list[float]:
    return [vals[int(i * len(vals) / n)] for i in range(1, n)]


def bin_of(val: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if val <= e:
            return i
    return len(edges)


def novelty_from_similarity(nn_sim: float) -> float:
    return 1 - nn_sim
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/search/__init__.py src/clawmarks/search/scoring.py tests/test_scoring.py
git commit -m "feat(clawmarks): extract pure scoring math into search/scoring.py with tests"
```

---

### Task 3: `search/seed_pool.py`: single implementation of the candidate-seed dedup/merge logic

**Files:**
- Create: `src/clawmarks/search/seed_pool.py`
- Test: `tests/test_seed_pool.py`
- Reference (read-only): `notes/curation_server.py:391-403` (dedup/merge in `_handle_seed_generate`),
  `notes/run_uncanny_allnight2.py`'s `add_to_shared_seed_pool`/`load_shared_seed_pool` (added in the
  commit `95760ac feat(uncanny): wire allnight2 driver to shared seed pool`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `clawmarks.search.seed_pool.load(path: pathlib.Path) -> dict[str, dict]`,
  `clawmarks.search.seed_pool.merge(existing: dict[str, dict], new_subjects: list[str], source: str,
  created_at: str) -> tuple[dict[str, dict], list[str]]` (returns updated dict and the list of
  subjects actually added, i.e. not duplicates), `clawmarks.search.seed_pool.save(path:
  pathlib.Path, seeds: dict[str, dict]) -> None`.

This is the one place today's two duplicated implementations (curation_server.py's inline dedup,
and run_uncanny_allnight2.py's `add_to_shared_seed_pool`) collapse into. Both call sites get
updated to use this module in Task 8 (curation_server move) and Task 4 (driver merge).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_seed_pool.py
from clawmarks.search import seed_pool


def test_merge_adds_new_subjects_and_reports_them():
    existing = {"close-up cat portrait": {"source": "fallback", "created_at": "t0"}}
    updated, added = seed_pool.merge(
        existing, ["airport baggage carousel", "close-up cat portrait"],
        source="gpt5.5", created_at="t1",
    )
    assert added == ["airport baggage carousel"]
    assert "airport baggage carousel" in updated
    assert updated["airport baggage carousel"] == {"source": "gpt5.5", "created_at": "t1"}


def test_merge_dedupes_case_insensitively():
    existing = {"Close-Up Cat Portrait": {"source": "fallback", "created_at": "t0"}}
    updated, added = seed_pool.merge(
        existing, ["close-up cat portrait"], source="gpt5.5", created_at="t1",
    )
    assert added == []
    assert len(updated) == 1


def test_merge_dedupes_within_the_new_batch_itself():
    updated, added = seed_pool.merge(
        {}, ["glass office atrium", "Glass Office Atrium"], source="gpt5.5", created_at="t1",
    )
    assert added == ["glass office atrium"]
    assert len(updated) == 1


def test_load_missing_file_returns_empty_dict(tmp_path):
    assert seed_pool.load(tmp_path / "does_not_exist.json") == {}


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "seeds.json"
    seeds = {"roadwork cones": {"source": "gpt5.5", "created_at": "t1"}}
    seed_pool.save(path, seeds)
    assert seed_pool.load(path) == seeds
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_pool.py -v`
Expected: FAIL (`ModuleNotFoundError` / `AttributeError`: module doesn't exist yet)

- [ ] **Step 3: Write `src/clawmarks/search/seed_pool.py`**

```python
import json
from pathlib import Path


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save(path: Path, seeds: dict) -> None:
    with open(path, "w") as f:
        json.dump(seeds, f, indent=1)


def merge(existing: dict, new_subjects: list, source: str, created_at: str) -> tuple:
    seeds = dict(existing)
    existing_lower = {s.lower().strip() for s in seeds}
    added = []
    for s in new_subjects:
        s = str(s).strip()
        if s and s.lower() not in existing_lower:
            seeds[s] = {"source": source, "created_at": created_at}
            existing_lower.add(s.lower())
            added.append(s)
    return seeds, added
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed_pool.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/search/seed_pool.py tests/test_seed_pool.py
git commit -m "feat(clawmarks): unify candidate-seed dedup/merge into search/seed_pool.py"
```

---

### Task 4: `search/driver.py`: merge the two allnight scripts behind a `--round` parameter, with generation-job-building tests

**Files:**
- Create: `src/clawmarks/search/driver.py`
- Test: `tests/test_generation_jobs.py`
- Reference (read-only): `notes/run_uncanny_allnight.py` (round 1, full file: staged escalation:
  base vocab → widened vocab → GPT-5.5, 50/50 exploit/explore, no parent_tag, no shared seed pool),
  `notes/run_uncanny_allnight2.py` (round 2, full file: GPT-5.5 seeds from generation 1, no
  staged widening, `EXPLORE_FRACTION`-driven split, user-picks-first exploit pool with parent_tag,
  prior-round exclusion embeddings, shared seed pool read/write)

**Interfaces:**
- Consumes: `clawmarks.search.scoring.bin_edges`, `clawmarks.search.scoring.bin_of`,
  `clawmarks.search.scoring.novelty_from_similarity` (Task 2); `clawmarks.search.seed_pool.load`,
  `clawmarks.search.seed_pool.merge`, `clawmarks.search.seed_pool.save` (Task 3); `clawmarks.config`
  (Task 1).
- Produces: `clawmarks.search.driver.RoundConfig` (dataclass, fields below);
  `clawmarks.search.driver.build_generation_jobs(gen_idx: int, subjects: list[str], textures:
  list[str], elites: list[dict], user_picks: list[dict], batch_size: int, explore_fraction:
  float) -> list[dict]`; `clawmarks.search.driver.ROUND_CONFIGS: dict[int, RoundConfig]`.

Round 2's `build_generation_jobs` (`notes/run_uncanny_allnight2.py:154-200`) is a strict
generalization of round 1's (`notes/run_uncanny_allnight.py`'s version): it already falls back to
elites when `user_picks` is empty, and its `EXPLORE_FRACTION` parameter reduces to round 1's fixed
50/50 when set to `0.5`. This task only needs to port round 2's version, parameterized, and prove
it reproduces round 1's behavior at `explore_fraction=0.5, user_picks=[]`, `parent_tag=None` (round
1 elites never had a `parent_tag` key, so `e.get("tag")` on them already resolves to `None`, matching
round 1's job dicts having no `parent_tag` key at all is a difference: round 1 jobs never included
the key. Match round 1 exactly by only adding `"parent_tag": e.get("tag")` to the job dict when
`elites` came from a round that tracks it; since this is new code, always include the key: round
1's gallery/HTML code never reads `parent_tag`, so its presence is harmless and verified by the
smoke test in Task 14, not asserted false here).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generation_jobs.py
import random
from clawmarks.search.driver import build_generation_jobs


def test_batch_splits_by_explore_fraction():
    random.seed(0)
    elite = {"prompt_name": "style_x", "prompt": "trentbuckle style, x", "strength": 1.0,
             "cfg": 7.0, "tag": "gen1_explore_0_seed1"}
    jobs = build_generation_jobs(
        gen_idx=2, subjects=["a subject"], textures=["a texture"],
        elites=[elite], user_picks=[], batch_size=20, explore_fraction=0.85,
    )
    n_explore = sum(1 for j in jobs if j["category"] == "r2_explore")
    n_exploit = sum(1 for j in jobs if j["category"] == "r2_exploit")
    assert n_explore == 17
    assert n_exploit == 3
    assert n_explore + n_exploit == 20


def test_fifty_fifty_split_matches_round_one_behavior():
    random.seed(0)
    elite = {"prompt_name": "style_x", "prompt": "trentbuckle style, x", "strength": 1.0,
             "cfg": 7.0, "tag": "gen1_explore_0_seed1"}
    jobs = build_generation_jobs(
        gen_idx=2, subjects=["a subject"], textures=["a texture"],
        elites=[elite], user_picks=[], batch_size=20, explore_fraction=0.5,
    )
    n_explore = sum(1 for j in jobs if j["category"] == "r2_explore")
    n_exploit = sum(1 for j in jobs if j["category"] == "r2_exploit")
    assert n_explore == 10
    assert n_exploit == 10


def test_exploit_jobs_prefer_user_picks_over_elites():
    pick = {"prompt_name": "style_pick", "prompt": "trentbuckle style, pick", "strength": 1.2,
            "cfg": 6.0, "tag": "gen1_explore_1_seed2"}
    elite = {"prompt_name": "style_elite", "prompt": "trentbuckle style, elite", "strength": 1.0,
             "cfg": 7.0, "tag": "gen1_explore_2_seed3"}
    jobs = build_generation_jobs(
        gen_idx=3, subjects=["a subject"], textures=["a texture"],
        elites=[elite], user_picks=[pick], batch_size=4, explore_fraction=0.0,
    )
    assert all(j["prompt_name"] == "style_pick" for j in jobs)


def test_no_elites_and_no_picks_produces_only_explore_jobs():
    jobs = build_generation_jobs(
        gen_idx=1, subjects=["a subject"], textures=["a texture"],
        elites=[], user_picks=[], batch_size=5, explore_fraction=0.85,
    )
    assert all(j["category"] == "r2_explore" for j in jobs)
    assert len(jobs) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generation_jobs.py -v`
Expected: FAIL (`ModuleNotFoundError`: `driver.py` doesn't exist yet)

- [ ] **Step 3: Write `src/clawmarks/search/driver.py`**

Port the full contents of `notes/run_uncanny_allnight2.py` into this file, with these exact
changes:

1. Replace the `import os, sys, json, time, random, subprocess, base64` / path-constant block
   (`SC`, `PREV_DIR`, `OUT_DIR`, `STATE_FILE`, `USER_PICKS_FILE`, `SEEDS_FILE`) with imports from
   `clawmarks.config` (`SWEEP_DIR`, `SWEEP2_DIR`, `SEEDS_FILE`, `USER_PICKS_FILE`) and a
   `RoundConfig` dataclass:

```python
from dataclasses import dataclass, field


@dataclass
class RoundConfig:
    round: int
    wall_clock_cap_hours: float
    budget_usd_cap: float
    budget_safety_margin: float
    gen_batch_size: int
    explore_fraction: float
    max_generations: int
    textures: list
    fallback_subjects: list
    seed_from_start: bool
    exclude_prev_round: bool
    out_dir_name: str


ROUND_CONFIGS = {
    1: RoundConfig(
        round=1, wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
        gen_batch_size=60, explore_fraction=0.5, max_generations=400,
        textures=[
            "marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
            "dark-rimmed eyes glowing pale blue, dense dark-blue vertical brush-dash background, "
            "thick acrylic dry-brush texture, raw outsider-art painting",
        ],
        fallback_subjects=[
            "close-up cat portrait", "close-up wolf portrait", "close-up fox portrait",
            "close-up owl portrait", "close-up horse portrait",
            "close-up human face, pale skin, hand pressed beside cheek",
        ],
        seed_from_start=False, exclude_prev_round=False, out_dir_name="uncanny_sweep",
    ),
    2: RoundConfig(
        round=2, wall_clock_cap_hours=1.0, budget_usd_cap=1.00, budget_safety_margin=0.10,
        gen_batch_size=20, explore_fraction=0.85, max_generations=60,
        textures=[
            "marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
            "dark-rimmed eyes glowing pale blue, dense dark-blue vertical brush-dash background, "
            "thick acrylic dry-brush texture, raw outsider-art painting",
            "loose watercolor wash bleeding at the edges, raw sketchbook page, mixed media",
            "heavy black ink crosshatching over torn found-paper collage edges, raw "
            "outsider-art painting",
        ],
        fallback_subjects=[
            "close-up cat portrait", "close-up wolf portrait", "close-up fox portrait",
            "close-up human face, pale skin, hand pressed beside cheek",
            "dollhouse interior seen through a broken window",
            "empty parking garage at night, one flickering light",
            "wall of surveillance camera monitors, mostly static",
            "vending machine humming alone in a dark hallway",
            "abandoned playground, swing set mid-motion with no one on it",
            "waiting room with rows of identical empty chairs",
        ],
        seed_from_start=True, exclude_prev_round=True, out_dir_name="uncanny_sweep2",
    ),
}
```

   Copy round 1's full `BASE_TEXTURES`/`WIDENED_TEXTURES`/`BASE_SUBJECTS`/`WIDENED_SUBJECTS` lists
   from `notes/run_uncanny_allnight.py:49-83` verbatim into `ROUND_CONFIGS[1]`'s `textures` and
   `fallback_subjects` (the excerpt above is abbreviated for readability; the actual file must
   contain every entry from the source, not a truncated subset: verify by diffing list lengths
   against the source file before committing).

2. Keep `build_generation_jobs` exactly as it exists in `notes/run_uncanny_allnight2.py:154-200`,
   moved to module level, with its `EXPLORE_FRACTION` global replaced by an `explore_fraction`
   parameter and its `FALLBACK_SUBJECTS[:4]`/`is_style` check replaced by taking a
   `style_subject_count` parameter (round 1 checked `BASE_SUBJECTS[:5]`, round 2 checked
   `FALLBACK_SUBJECTS[:4]`: different counts, so this must be a parameter, not a hardcoded
   slice):

```python
def build_generation_jobs(gen_idx, subjects, textures, elites, user_picks, batch_size,
                          explore_fraction, style_subject_count=4):
    jobs = []
    n_explore = round(batch_size * explore_fraction)
    n_exploit = batch_size - n_explore

    exploit_pool = list(user_picks) if user_picks else []
    if len(exploit_pool) < n_exploit:
        exploit_pool = exploit_pool + [e for e in elites if e not in exploit_pool]

    for i in range(n_exploit):
        if not exploit_pool:
            break
        e = random.choice(exploit_pool)
        strength = max(0.3, min(2.2, e["strength"] + random.gauss(0, 0.2)))
        cfg = max(1.0, min(20.0, e["cfg"] + random.gauss(0, 2.0)))
        seed = random.randint(1, 999999)
        jobs.append({
            "tag": f"gen{gen_idx}_exploit_{i}_seed{seed}", "category": "r2_exploit",
            "prompt_name": e["prompt_name"], "prompt": e["prompt"],
            "seed": seed, "strength": round(strength, 3), "cfg": round(cfg, 2),
            "steps": 28, "sampler": "ddim", "negative": NEG_DEFAULT,
            "parent_tag": e.get("tag"),
        })

    for i in range(n_explore):
        subj = random.choice(subjects)
        tex = random.choice(textures)
        prompt = f"{TRIGGER}{subj}, {tex}"
        strength = round(random.uniform(0.5, 2.0), 3)
        cfg = round(random.uniform(2.0, 15.0), 2)
        seed = random.randint(1, 999999)
        is_style = subj in subjects[:style_subject_count]
        pname = ("style_" if is_style else "conflict_") + subj[:24].replace(" ", "_")
        jobs.append({
            "tag": f"gen{gen_idx}_explore_{i}_seed{seed}", "category": "r2_explore",
            "prompt_name": pname, "prompt": prompt,
            "seed": seed, "strength": strength, "cfg": cfg,
            "steps": 28, "sampler": "ddim", "negative": NEG_DEFAULT,
        })
    return jobs
```

   Note the test `test_fifty_fifty_split_matches_round_one_behavior` passes `subjects=["a
   subject"]` with the default `style_subject_count=4`: this only checks the explore/exploit
   split ratio, not the `is_style` classification, so the default is fine for that test. When
   wiring `ROUND_CONFIGS[1]` into the main loop (step 4 below), pass `style_subject_count=5` to
   match round 1's original `BASE_SUBJECTS[:5]` check.

3. Keep `request_gpt55_subjects`, `load_shared_seed_pool`/`add_to_shared_seed_pool` (rewritten to
   call `clawmarks.search.seed_pool.load`/`merge`/`save` from Task 3 instead of their own inline
   JSON handling), `load_user_picks`, `submit_and_collect`, `score_batch`, `cell_html`,
   `build_gallery` (rewritten to call `clawmarks.search.scoring.bin_edges`/`bin_of` from Task 2
   instead of its own nested functions) unchanged from `notes/run_uncanny_allnight2.py`.

4. In the main loop (the `if __name__ == "__main__":` block and the function it calls), add a
   `--round {1,2}` CLI arg (via `argparse`, parsed inside `main(argv=None)` so `cli.py` can call it
   directly), look up `cfg = ROUND_CONFIGS[args.round]`, and use `cfg.*` fields everywhere the
   original scripts used their module-level constants (`WALL_CLOCK_CAP_HOURS` → `cfg.wall_clock_cap_hours`,
   etc.). Gate the prior-round exclusion-embedding block
   (`notes/run_uncanny_allnight2.py:361-366`) behind `if cfg.exclude_prev_round:`. Gate the
   immediate-seeding block (`notes/run_uncanny_allnight2.py:368-375`) behind
   `if cfg.seed_from_start:`; when `False` (round 1), keep round 1's original staged-escalation
   plateau logic from `notes/run_uncanny_allnight.py` (widen vocabulary first, then hand off to
   GPT-5.5) verbatim, reading `state["stage"]` the way round 1 does.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generation_jobs.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/search/driver.py tests/test_generation_jobs.py
git commit -m "feat(clawmarks): merge run_uncanny_allnight(2).py into search/driver.py with --round"
```

---

### Task 5: `compute/comfyui.py`: move workflow submission code

**Files:**
- Create: `src/clawmarks/compute/__init__.py`
- Create: `src/clawmarks/compute/comfyui.py`
- Reference (read-only): `notes/run_uncanny_sweep.py` (full file, especially `build_workflow`,
  `api_post`, `api_get`)

**Interfaces:**
- Produces: `clawmarks.compute.comfyui.build_workflow(...)`, `clawmarks.compute.comfyui.api_post(...)`,
  `clawmarks.compute.comfyui.api_get(...)`: same signatures as `notes/run_uncanny_sweep.py`.

- [ ] **Step 1: Write `src/clawmarks/compute/__init__.py`** (empty file)

- [ ] **Step 2: Move the code**

Copy `build_workflow`, `api_post`, `api_get`, and every constant/import they depend on (the
ComfyUI endpoint URL construction, the workflow JSON template) verbatim from
`notes/run_uncanny_sweep.py` into `src/clawmarks/compute/comfyui.py`. Replace any
`os.environ["RUNPOD_API_KEY"]`-style read with the same pattern (no change needed, already
compliant with the secrets rule) but drop any hardcoded `SC`/path constant in favor of
`clawmarks.config` if the file references one.

- [ ] **Step 3: Verify with a syntax check** (no test file: this task is a pure move of
  network-calling code with no pure-logic surface to unit test; behavior is verified by the
  Task 14 smoke check)

Run: `uv run python -c "import ast; ast.parse(open('src/clawmarks/compute/comfyui.py').read())"`
Expected: no output (success)

- [ ] **Step 4: Commit**

```bash
git add src/clawmarks/compute/__init__.py src/clawmarks/compute/comfyui.py
git commit -m "chore(clawmarks): move ComfyUI workflow submission into compute/comfyui.py"
```

---

### Task 6: `compute/runpod.py`: merge the RunPod bring-up/SSH/SFTP scripts

**Files:**
- Create: `src/clawmarks/compute/runpod.py`
- Reference (read-only): `rp_bring_up.py`, `rp_bring_up2.py`, `rpget.py`, `rpget2.py`,
  `rpsftp.py`, `rpsftp2.py`, `rpssh.py`, `rpssh2.py` (all at repo root)

**Interfaces:**
- Produces: `clawmarks.compute.runpod.bring_up(gpu_priority: list[str]) -> dict` (pod info),
  `clawmarks.compute.runpod.ssh(pod_id: str, command: str) -> subprocess.CompletedProcess`,
  `clawmarks.compute.runpod.get(pod_id: str, remote_path: str, local_path: str) -> None`,
  `clawmarks.compute.runpod.put(pod_id: str, local_path: str, remote_path: str) -> None`,
  `clawmarks.compute.runpod.get_balance() -> float`, `clawmarks.compute.runpod.pause(pod_id: str)
  -> None`, `clawmarks.compute.runpod.terminate(pod_id: str) -> None`.

First diff the paired scripts to confirm the "2" suffix is purely about running a second pod
concurrently and not a logic difference, before merging:

- [ ] **Step 1: Confirm the `rp_bring_up.py`/`rp_bring_up2.py` diff is pod-index-only**

Run: `diff rp_bring_up.py rp_bring_up2.py`
Expected: differences limited to pod-naming/log-file-naming strings (e.g. a suffix like `_2`) and
GPU priority list, not to the GraphQL/SSH bring-up logic itself. If the diff shows genuine logic
differences beyond naming/priority-list, stop and note the discrepancy in the task's commit
message rather than silently picking one version.

- [ ] **Step 2: Merge into one parameterized module**

Write `src/clawmarks/compute/runpod.py` containing every function from `rp_bring_up.py`
(`bring_up`-equivalent), `rpssh.py` (the SSH-command-runner), `rpget.py`/`rpsftp.py` (SFTP
get/put), taking a `pod_id`/`gpu_priority` argument instead of the hardcoded values the "2"
variants used, plus `get_balance`/`pause`/`terminate` (the GraphQL calls already present in
`notes/run_uncanny_allnight2.py`'s `gql`/`get_balance`, generalized with `pause`/`terminate`
mutations following the same `gql()` pattern). Read `RUNPOD_API_KEY` via
`os.environ["RUNPOD_API_KEY"]`.

- [ ] **Step 3: Verify with a syntax check**

Run: `uv run python -c "import ast; ast.parse(open('src/clawmarks/compute/runpod.py').read())"`
Expected: no output (success)

- [ ] **Step 4: Commit**

```bash
git add src/clawmarks/compute/runpod.py
git commit -m "refactor(clawmarks): merge rp_bring_up/rpget/rpsftp/rpssh into compute/runpod.py"
```

---

### Task 7: `probe/train.py` and `probe/sweep.py`: move probe/calibration scripts

**Files:**
- Create: `src/clawmarks/probe/__init__.py`
- Create: `src/clawmarks/probe/train.py` (from `notes/train_probe.py`)
- Create: `src/clawmarks/probe/sweep.py` (from `notes/probe_uncanny.py`,
  `notes/probe_strength_sweep.py`, `notes/gen_samples.py`: combine into one module since they're
  all small (<130 lines each) and all deal with the same "run a probe/calibration batch" concern)

**Interfaces:**
- Produces: `clawmarks.probe.train.main(argv=None)`, `clawmarks.probe.sweep.run_probe_uncanny(...)`,
  `clawmarks.probe.sweep.run_strength_sweep(...)`, `clawmarks.probe.sweep.gen_samples(...)`: same
  argument shapes as the source scripts' `if __name__ == "__main__"` bodies, wrapped in named
  functions instead of bare module-level code.

- [ ] **Step 1: Write `src/clawmarks/probe/__init__.py`** (empty file)

- [ ] **Step 2: Move `train_probe.py`**

Copy `notes/train_probe.py` into `src/clawmarks/probe/train.py`. Replace its `SC = "..."` constant
and any derived path with imports from `clawmarks.config`. Wrap the top-level script body (the
part that runs when executed directly) in `def main(argv=None):` so it's callable from `cli.py`.

- [ ] **Step 3: Move `probe_uncanny.py`, `probe_strength_sweep.py`, `gen_samples.py`**

Copy each file's functions into `src/clawmarks/probe/sweep.py` under the names
`run_probe_uncanny`, `run_strength_sweep`, `gen_samples` respectively, replacing each file's own
`OUT_DIR`/`SC` constant with `clawmarks.config` equivalents. Keep each function's internal logic
unchanged.

- [ ] **Step 4: Verify with a syntax check**

Run: `uv run python -c "import ast; ast.parse(open('src/clawmarks/probe/train.py').read()); ast.parse(open('src/clawmarks/probe/sweep.py').read())"`
Expected: no output (success)

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/probe/__init__.py src/clawmarks/probe/train.py src/clawmarks/probe/sweep.py
git commit -m "chore(clawmarks): move probe/calibration scripts into probe/ subpackage"
```

---

### Task 8: `build/`: move the 12 generator scripts and `shared_ui.py`

**Files:**
- Create: `src/clawmarks/build/__init__.py`
- Create: `src/clawmarks/shared_ui.py` (from `notes/shared_ui.py`)
- Create: `src/clawmarks/build/scan_gallery.py` (from `notes/build_scan_gallery.py`)
- Create: `src/clawmarks/build/elite_archive.py` (from `notes/build_elite_archive.py`)
- Create: `src/clawmarks/build/coverage_map.py` (from `notes/build_coverage_map.py`)
- Create: `src/clawmarks/build/map_view.py` (from `notes/build_map_view.py`)
- Create: `src/clawmarks/build/redundancy_view.py` (from `notes/build_redundancy_view.py`)
- Create: `src/clawmarks/build/novelty_decay.py` (from `notes/build_novelty_decay.py`)
- Create: `src/clawmarks/build/lineage_view.py` (from `notes/build_lineage_view.py`)
- Create: `src/clawmarks/build/solution_map.py` (from `notes/build_solution_map.py`)
- Create: `src/clawmarks/build/similarity_index.py` (from `notes/build_similarity_index.py`)
- Create: `src/clawmarks/build/thumbnails.py` (from `notes/build_thumbnails.py`)
- Create: `src/clawmarks/build/explore_hub.py` (from `notes/build_explore_hub.py`)
- Create: `src/clawmarks/build/seed_browser.py` (from `notes/build_seed_browser.py`)
- Create: `src/clawmarks/build/probe_report.py` (from `notes/build_probe_report.py`)
- Create: `src/clawmarks/build/uncanny_gallery.py` (from `notes/build_uncanny_gallery.py`)
- Create: `src/clawmarks/build/merge_round2.py` (from `notes/merge_round2.py`)

**Interfaces:**
- Consumes: `clawmarks.shared_ui` (all constants/functions previously imported via `from
  shared_ui import ...`), `clawmarks.config`.
- Produces: each module keeps its existing top-level behavior (writes an HTML file on import/run)
  but wrapped in a `main()` function so `cli.py` can call it without a subprocess.

This is a mechanical move applied identically to every file listed above: do all 15 in this one
task since none of them have logic worth testing in isolation (each one only builds an HTML
string from data on disk) and splitting them into 15 separate tasks would mean 15 review gates
for the same three-line transformation repeated. The transformation, applied to every file:

- [ ] **Step 1: Move `shared_ui.py` first (everything else depends on it)**

Copy `notes/shared_ui.py` byte-for-byte into `src/clawmarks/shared_ui.py`. It has no `SC`/path
constant of its own (confirmed: `notes/shared_ui.py` only exports CSS/JS/HTML string constants
and helper functions, no filesystem paths): no changes needed beyond the move.

- [ ] **Step 2: For each of the 15 `build_*.py`/`merge_round2.py` files, apply this exact recipe**

  a. Copy the file to its new path (e.g. `notes/build_scan_gallery.py` → `src/clawmarks/build/scan_gallery.py`).

  b. Change `from shared_ui import ...` to `from clawmarks.shared_ui import ...` (same imported
     names, just the module path changes).

  c. Change every hardcoded `SC = "/workspace/trent-with-smart-prompts"` /
     `SWEEP_DIR = "/workspace/trent-with-smart-prompts/notes/uncanny_sweep"` /
     `SWEEP_DIR = f"{SC}/notes/uncanny_sweep"` line to import the equivalent from
     `clawmarks.config` instead (`from clawmarks.config import SWEEP_DIR` or `SWEEP2_DIR`,
     matching whichever directory that specific file wrote to).

  d. Wrap the module's top-level executable statements (the part that currently runs
     unconditionally when the file is run as a script: building the HTML string and writing it
     to disk) inside `def main():`, and add `if __name__ == "__main__": main()` at the bottom, so
     `cli.py` (Task 9) can call `main()` directly without a subprocess.

  e. Leave every other line (the actual HTML/CSS/JS template strings, the data-loading and
     scoring logic) byte-for-byte unchanged.

- [ ] **Step 3: Write `src/clawmarks/build/__init__.py`** (empty file)

- [ ] **Step 4: Verify every moved file parses**

Run:
```bash
for f in src/clawmarks/build/*.py src/clawmarks/shared_ui.py; do
  uv run python -c "import ast; ast.parse(open('$f').read())" || echo "FAILED: $f"
done
```
Expected: no "FAILED" lines printed

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/shared_ui.py src/clawmarks/build/
git commit -m "chore(clawmarks): move 15 build/gallery generators and shared_ui into build/"
```

---

### Task 9: `curation_server.py`: move and wire to `config`/`seed_pool`

**Files:**
- Create: `src/clawmarks/curation_server.py` (from `notes/curation_server.py`)
- Reference (read-only): `notes/curation_server.py` (full file, 414 lines)

**Interfaces:**
- Consumes: `clawmarks.config.SWEEP_DIR`, `clawmarks.config.SEEDS_FILE` (Task 1);
  `clawmarks.search.seed_pool.load`/`merge`/`save` (Task 3).
- Produces: `clawmarks.curation_server.main(argv=None)` (starts the `ThreadingHTTPServer`, same
  behavior as running `notes/curation_server.py` directly today).

- [ ] **Step 1: Move the file**

Copy `notes/curation_server.py` into `src/clawmarks/curation_server.py`.

- [ ] **Step 2: Replace path constants**

Change the `SC = "/workspace/trent-with-smart-prompts"` / `SWEEP_DIR = f"{SC}/notes/uncanny_sweep"`
/ `SEEDS_FILE = f"{SWEEP_DIR}/candidate_seeds.json"` lines to import `SWEEP_DIR`, `SEEDS_FILE` from
`clawmarks.config` instead.

- [ ] **Step 3: Replace the inline seed dedup/merge logic with `seed_pool`**

In `_handle_seed_generate` (currently `notes/curation_server.py:339-404`), replace the manual
`load_store(SEEDS_FILE)` / dedup loop / `save_store(SEEDS_FILE, seeds)` block with:

```python
from clawmarks.search import seed_pool
from datetime import datetime, timezone

with _lock:
    seeds = seed_pool.load(SEEDS_FILE)
    updated, added = seed_pool.merge(
        seeds, new_subjects, source="gpt5.5",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    seed_pool.save(SEEDS_FILE, updated)
self._json_response(200, {"ok": True, "added": added, "count": len(updated)})
```

Keep everything else in the method (the `opencode run` subprocess invocation, the existing-seeds
lookup used to build the "don't repeat these" prompt text) unchanged: only the final
load/dedup/save block changes.

- [ ] **Step 4: Wrap the module's server-start code in `main()`**

Wrap the bottom-of-file `if __name__ == "__main__":` block's server-construction/`serve_forever()`
call in `def main(argv=None):`, keep the `if __name__ == "__main__": main()` guard.

- [ ] **Step 5: Verify with a syntax check**

Run: `uv run python -c "import ast; ast.parse(open('src/clawmarks/curation_server.py').read())"`
Expected: no output (success)

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/curation_server.py
git commit -m "refactor(clawmarks): move curation_server.py, wire to config and search.seed_pool"
```

---

### Task 10: `cli.py`: the `clawmarks` entry point

**Files:**
- Create: `src/clawmarks/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `clawmarks.curation_server.main` (Task 9), every `clawmarks.build.*.main` (Task 8),
  `clawmarks.search.driver.main` (Task 4), `clawmarks.probe.train.main` (Task 7),
  `clawmarks.compute.runpod` functions (Task 6).
- Produces: `clawmarks.cli.main(argv=None) -> int` (the `console_scripts` target referenced in
  `pyproject.toml`'s `[project.scripts]`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from clawmarks.cli import build_parser


def test_build_all_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["build", "all"])
    assert args.command == "build"
    assert args.target == "all"


def test_run_allnight_round_argument_parses():
    parser = build_parser()
    args = parser.parse_args(["run", "allnight", "--round", "2"])
    assert args.command == "run"
    assert args.round == 2


def test_serve_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'clawmarks.cli'`)

- [ ] **Step 3: Write `src/clawmarks/cli.py`**

```python
import argparse

from clawmarks.build import (
    scan_gallery, elite_archive, coverage_map, map_view, redundancy_view,
    novelty_decay, lineage_view, solution_map, similarity_index, thumbnails,
    explore_hub, seed_browser, probe_report, uncanny_gallery,
)

BUILD_TARGETS = {
    "scan": scan_gallery.main,
    "archive": elite_archive.main,
    "coverage": coverage_map.main,
    "map": map_view.main,
    "redundancy": redundancy_view.main,
    "novelty-decay": novelty_decay.main,
    "lineage": lineage_view.main,
    "solution-map": solution_map.main,
    "similarity": similarity_index.main,
    "thumbnails": thumbnails.main,
    "explore-hub": explore_hub.main,
    "seeds": seed_browser.main,
    "probe-report": probe_report.main,
    "uncanny-gallery": uncanny_gallery.main,
}


def build_parser():
    parser = argparse.ArgumentParser(prog="clawmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve")

    build_p = sub.add_parser("build")
    build_p.add_argument("target", choices=["all", *BUILD_TARGETS.keys()])

    run_p = sub.add_parser("run")
    run_sub = run_p.add_subparsers(dest="run_target", required=True)
    allnight_p = run_sub.add_parser("allnight")
    allnight_p.add_argument("--round", type=int, choices=[1, 2], required=True)
    allnight_p.set_defaults(command="run")

    probe_p = sub.add_parser("probe")
    probe_sub = probe_p.add_subparsers(dest="probe_target", required=True)
    probe_sub.add_parser("train")

    pod_p = sub.add_parser("pod")
    pod_sub = pod_p.add_subparsers(dest="pod_action", required=True)
    for action in ("bring-up", "pause", "terminate", "ssh", "get", "put"):
        pod_sub.add_parser(action)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        from clawmarks.curation_server import main as serve_main
        return serve_main()

    if args.command == "build":
        if args.target == "all":
            for fn in BUILD_TARGETS.values():
                fn()
        else:
            BUILD_TARGETS[args.target]()
        return 0

    if args.command == "run":
        from clawmarks.search.driver import main as driver_main
        return driver_main(["--round", str(args.round)])

    if args.command == "probe" and args.probe_target == "train":
        from clawmarks.probe.train import main as train_main
        return train_main()

    if args.command == "pod":
        from clawmarks.compute import runpod
        action_map = {
            "bring-up": lambda: runpod.bring_up(gpu_priority=["RTX 4090", "RTX 3090", "RTX A5000"]),
            "pause": lambda: runpod.pause(input("pod id: ")),
            "terminate": lambda: runpod.terminate(input("pod id: ")),
        }
        if args.pod_action in action_map:
            action_map[args.pod_action]()
        return 0

    parser.error(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/cli.py tests/test_cli.py
git commit -m "feat(clawmarks): add clawmarks CLI entry point with build/run/serve/probe/pod subcommands"
```

---

### Task 11: Reinstall and run the full test suite

**Files:** none created; verification only.

- [ ] **Step 1: Reinstall editable (picks up `[project.scripts]` entry point)**

Run: `cd /workspace/trent-with-smart-prompts && uv pip install -e .`
Expected: reinstalls successfully, no errors

- [ ] **Step 2: Confirm the `clawmarks` command is on PATH**

Run: `uv run clawmarks --help`
Expected: prints the `argparse` usage/help text listing `serve`, `build`, `run`, `probe`, `pod`

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass (config: 3, scoring: 4, seed_pool: 5, generation_jobs: 4, cli: 3: 19 total)

- [ ] **Step 4: Commit only if any fixes were needed to get here**

```bash
git add -A
git commit -m "fix(clawmarks): resolve import/wiring issues found running the full test suite"
```
(Skip this step entirely if step 3 passed with no changes needed.)

---

### Task 12: Smoke check: `clawmarks build all` produces identical output to the current scripts

**Files:** none created; verification only. This is the gate for deleting the old files in Task 13.

**CRITICAL: `notes/uncanny_sweep/` and `notes/uncanny_sweep2/` are live production data**, not
disposable fixtures. `scored_manifest.json` in particular is the accumulated result of multiple
real, RunPod-billed generation rounds merged together over time (see `merge_round2.py`) - it
cannot be regenerated from nothing if lost. Several `build_*.py` scripts write derived JSON back
into these same directories as a side effect of running (`similarity.json`, `similarity_scored.json`,
`solution_map_data.json`, `solution_map_final_embs.pt` are all rewritten in place, not just the
`.html`/`.js` files this smoke check compares). Running the old scripts and the new package
back-to-back against the live directory with no isolation between them means the second run's
writes land on top of whatever the first run left behind, and a comparison that only checks
`.html`/`.js` will not notice the underlying JSON getting corrupted or truncated. **Never run
either the old scripts or the new package directly against `notes/uncanny_sweep/` or
`notes/uncanny_sweep2/` in this task without a backup-restore step bracketing each run.**

- [ ] **Step 1: Back up the live data directories before touching them**

Run:
```bash
cd /workspace/trent-with-smart-prompts
rm -rf /tmp/clawmarks-smoke-data-backup
cp -r notes/uncanny_sweep /tmp/clawmarks-smoke-data-backup-sweep
cp -r notes/uncanny_sweep2 /tmp/clawmarks-smoke-data-backup-sweep2
python3 -c "import json; print('backup manifest count:', len(json.load(open('/tmp/clawmarks-smoke-data-backup-sweep/scored_manifest.json'))))"
```
Expected: prints the current real entry count (should be in the thousands, not a small round
number like 452 - if it prints 452, the live data is already in a bad state from a prior run;
stop and investigate before proceeding, do not treat 452 as the correct baseline).

- [ ] **Step 2: Snapshot current script output**

Run:
```bash
mkdir -p /tmp/clawmarks-smoke-before
for f in notes/build_*.py; do python3 "$f"; done
cp notes/uncanny_sweep/*.html notes/uncanny_sweep/*.js /tmp/clawmarks-smoke-before/ 2>/dev/null
```
Expected: every `build_*.py` script runs successfully with its current exit code 0

- [ ] **Step 3: Restore the live directories from backup before the second run**

Run:
```bash
rm -rf notes/uncanny_sweep notes/uncanny_sweep2
cp -r /tmp/clawmarks-smoke-data-backup-sweep notes/uncanny_sweep
cp -r /tmp/clawmarks-smoke-data-backup-sweep2 notes/uncanny_sweep2
python3 -c "import json; print('restored manifest count:', len(json.load(open('notes/uncanny_sweep/scored_manifest.json'))))"
```
Expected: prints the same count as Step 1. This undoes any JSON side effects from the old-script
run in Step 2 before the new package touches the same directory.

- [ ] **Step 4: Run the new package's build-all and compare**

Run:
```bash
mkdir -p /tmp/clawmarks-smoke-after
uv run clawmarks build all
cp notes/uncanny_sweep/*.html notes/uncanny_sweep/*.js /tmp/clawmarks-smoke-after/ 2>/dev/null
diff -rq /tmp/clawmarks-smoke-before /tmp/clawmarks-smoke-after
```
Expected: no differences reported. If any file differs, do not proceed to Task 13: diagnose
which moved module introduced the difference (most likely a path-constant substitution mistake
from Task 8's mechanical recipe) and fix it, re-running this whole task from Step 3 (restore
first, don't re-diff against a directory the new package already mutated).

- [ ] **Step 5: Restore the live directories once more so neither run's side effects persist**

Run:
```bash
rm -rf notes/uncanny_sweep notes/uncanny_sweep2
cp -r /tmp/clawmarks-smoke-data-backup-sweep notes/uncanny_sweep
cp -r /tmp/clawmarks-smoke-data-backup-sweep2 notes/uncanny_sweep2
python3 -c "import json; print('final restored count:', len(json.load(open('notes/uncanny_sweep/scored_manifest.json'))))"
```
Expected: same count as Step 1 again. The smoke check's job is to compare *output*, not to leave
either run's mutation as the new live state - the live directories must end this task exactly as
they started it. Once this is confirmed, regenerate the derived JSON files for real (not as a
smoke-test side effect) by running `uv run clawmarks build all` one final time, intentionally,
against the now-confirmed-correct live data.

- [ ] **Step 3: Smoke-test `clawmarks serve` starts without error**

Run:
```bash
uv run clawmarks serve &
SERVER_PID=$!
sleep 2
curl -sf http://127.0.0.1:8420/api/seeds > /dev/null && echo "OK"
kill $SERVER_PID
```
Expected: prints `OK` (adjust the port to whatever `notes/curation_server.py` currently binds to
if it differs from 8420: check the source file's `ThreadingHTTPServer((...), ...)` call before
running this step)

---

### Task 13: Delete the old scattered files

**Files:**
- Delete: `notes/build_scan_gallery.py`, `notes/build_elite_archive.py`,
  `notes/build_coverage_map.py`, `notes/build_map_view.py`, `notes/build_redundancy_view.py`,
  `notes/build_novelty_decay.py`, `notes/build_lineage_view.py`, `notes/build_solution_map.py`,
  `notes/build_similarity_index.py`, `notes/build_thumbnails.py`, `notes/build_explore_hub.py`,
  `notes/build_seed_browser.py`, `notes/build_probe_report.py`, `notes/build_uncanny_gallery.py`,
  `notes/merge_round2.py`, `notes/shared_ui.py`, `notes/curation_server.py`,
  `notes/run_uncanny_allnight.py`, `notes/run_uncanny_allnight2.py`, `notes/run_uncanny_sweep.py`,
  `notes/train_probe.py`, `notes/probe_uncanny.py`, `notes/probe_strength_sweep.py`,
  `notes/gen_samples.py`, `notes/mmd_score.py`, `notes/score_probe_samples.py`,
  `notes/score_probe_uncanny.py`, `notes/score_strength_sweep.py`, `rp_bring_up.py`,
  `rp_bring_up2.py`, `rpget.py`, `rpget2.py`, `rpsftp.py`, `rpsftp2.py`, `rpssh.py`, `rpssh2.py`

Only run this task after Task 12's smoke check passes with zero diffs. Do not delete
`notes/lab_notebook.md`, `notes/remote_setup.sh`, `notes/uncanny_sweep/` (data directory),
`notes/uncanny_sweep2/` (data directory), `notes/probe_samples/` (data directory), or any `.json`
data file: only the Python scripts listed above move out of existence.

- [ ] **Step 1: Delete the files**

```bash
cd /workspace/trent-with-smart-prompts
git rm notes/build_scan_gallery.py notes/build_elite_archive.py notes/build_coverage_map.py \
  notes/build_map_view.py notes/build_redundancy_view.py notes/build_novelty_decay.py \
  notes/build_lineage_view.py notes/build_solution_map.py notes/build_similarity_index.py \
  notes/build_thumbnails.py notes/build_explore_hub.py notes/build_seed_browser.py \
  notes/build_probe_report.py notes/build_uncanny_gallery.py notes/merge_round2.py \
  notes/shared_ui.py notes/curation_server.py notes/run_uncanny_allnight.py \
  notes/run_uncanny_allnight2.py notes/run_uncanny_sweep.py notes/train_probe.py \
  notes/probe_uncanny.py notes/probe_strength_sweep.py notes/gen_samples.py notes/mmd_score.py \
  notes/score_probe_samples.py notes/score_probe_uncanny.py notes/score_strength_sweep.py \
  rp_bring_up.py rp_bring_up2.py rpget.py rpget2.py rpsftp.py rpsftp2.py rpssh.py rpssh2.py
```

- [ ] **Step 2: Re-run the smoke check against the deleted state to confirm nothing outside git still references the old paths**

Run: `rg -n "notes/run_uncanny_allnight|notes/curation_server|notes/shared_ui|notes/build_" --type py -g '!src/clawmarks/**'`
Expected: no output (nothing outside the new package references the deleted files by path)

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(clawmarks): delete superseded scripts now that src/clawmarks/ replaces them"
```

---

### Task 14: Update `CLAUDE.md`/`lab_notebook.md`, open the PR, squash-merge

**Files:**
- Modify: `notes/lab_notebook.md` (append a dated entry)
- Modify: `CLAUDE.md` (only if it references any of the deleted script paths by name: grep first)

- [ ] **Step 1: Check whether CLAUDE.md references any deleted script by path**

Run: `rg -n "run_uncanny_allnight|curation_server\.py|shared_ui\.py|build_scan_gallery" CLAUDE.md`
Expected: review any matches; if found, update them to reference the `clawmarks` CLI equivalent
(e.g. `python3 notes/curation_server.py` → `clawmarks serve`) instead of the deleted path.

- [ ] **Step 2: Append a lab notebook entry**

Add a dated entry to `notes/lab_notebook.md` describing: the package now exists at
`src/clawmarks/`, installed via `uv pip install -e .`, run via the `clawmarks` CLI (`clawmarks
serve`, `clawmarks build <target>`, `clawmarks run allnight --round {1,2}`, `clawmarks probe
train`, `clawmarks pod <action>`); the two allnight scripts merged into one `--round`-parameterized
driver; the smoke-check result confirming identical output to the pre-transition scripts. Grep
the new text for `—` and ` -- ` before finishing this step.

- [ ] **Step 3: Commit the docs update**

```bash
git add notes/lab_notebook.md CLAUDE.md
git commit -m "docs: update lab notebook and CLAUDE.md for the clawmarks package transition"
```

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin clawmarks-package-transition
gh pr create --title "refactor: transition CLAWMARKS tooling into an installable package" --body "$(cat <<'EOF'
## Summary
- 31 scattered scripts (notes/*.py + root RunPod scripts) become src/clawmarks/, an installable
  package with one clawmarks CLI (serve/build/run/probe/pod subcommands)
- Centralized path config (clawmarks.config.repo_root()) replaces ~20 duplicated SC = "..." literals
- run_uncanny_allnight.py + run_uncanny_allnight2.py merged into search/driver.py, parameterized
  by --round, removing ~350 duplicated lines
- Unit tests added for the pure-logic pieces: scoring, seed-pool dedup, generation-job building

## Test plan
- [x] Full pytest suite passes (19 tests)
- [x] `clawmarks build all` output diffed byte-identical against the pre-transition scripts
- [x] `clawmarks serve` starts and answers /api/seeds
- [x] No secrets in diff, no em dashes in new prose

Design doc: docs/superpowers/specs/2026-07-09-software-transition-design.md
Plan: docs/superpowers/plans/2026-07-09-software-transition.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Squash-merge and sync local main**

```bash
gh pr merge --squash --delete-branch
git checkout main
git pull --ff-only
git status
```
Expected: `git status` shows a clean working tree on `main`, up to date with `origin/main`.
