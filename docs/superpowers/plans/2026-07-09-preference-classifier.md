# Preference Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace "pick as winner" with a yes/no rating UI, and build a preference classifier
trained on those ratings (on top of frozen DINOv2 embeddings) that eventually ranks images by
predicted taste instead of raw novelty.

**Architecture:** A stratified sampler serves unreviewed images from `scored_manifest.json` to a
new `rate.html` page; ratings land in `user_ratings.json`, which immediately takes over the role
`user_picks.json` played in the search's exploit pool and the elite archive's fallback (Stage
5a — no model required). A separate embedding cache and training script produce a logistic
regression on frozen DINOv2 embeddings; a ranked-pool view lets the user validate it before an
opt-in flag switches exploit selection from yes-rated images to the model's continuous
predicted-preference score (Stage 5b).

**Tech Stack:** Python 3.10+, scikit-learn (new dependency), the existing torch/transformers
DINOv2 pipeline, stdlib `http.server`, vanilla JS.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-09-preference-classifier-design.md` exactly; it is the
  source of truth for behavior this plan doesn't repeat verbatim.
- Pin every new dependency version exactly (project convention — see `pyproject.toml`'s existing
  `==` pins). Install with `uv add <package>==<version>`, never bare `pip install`.
- All file paths in code come from `clawmarks.config` (`ROOT`, `SWEEP_DIR`, etc.), never a
  hardcoded `/workspace/trent-with-smart-prompts` string.
- Every new pure-logic module gets unit tests under `tests/`, following this repo's existing
  pattern of testing pure functions directly rather than booting a real HTTP server (see
  `tests/test_seed_pool.py`, `tests/test_scoring.py`, `tests/test_generation_jobs.py`).
- Run `pytest` after every task's implementation step, not just at the end.
- Favoriting (`user_favorites.json`, the star/bookmark button, `/api/favorite`,
  `/api/unfavorite`) is never touched by this plan.
- Stage 5b (the trained model steering the live search) ships behind an opt-in flag that
  defaults off. Do not flip it on as part of this plan — that's a manual step for the project
  owner after eyeballing `preference_rank.html` (Component 4's validation gate).

---

### Task 1: Shared manifest helpers + `USER_RATINGS_FILE` config

**Files:**
- Create: `src/clawmarks/search/manifest_index.py`
- Create: `tests/test_manifest_index.py`
- Modify: `src/clawmarks/config.py`

**Interfaces:**
- Produces: `manifest_index.index_by_tag(manifest: list[dict]) -> dict[str, dict]`,
  `manifest_index.item_summary(m: dict, sweep_dir) -> dict` (keys: `tag`, `prompt_name`,
  `prompt_type`, `faith`, `novelty`, `strength`, `cfg`, `thumb`, `file`), both used by later
  tasks (`build/elite_archive.py`, `curation_server.py`, `build/preference_rank.py`).
  `config.USER_RATINGS_FILE: Path`, used by every later task that reads/writes ratings.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_manifest_index.py
from clawmarks.search import manifest_index


def test_index_by_tag_builds_lookup():
    manifest = [{"tag": "a", "x": 1}, {"tag": "b", "x": 2}]
    idx = manifest_index.index_by_tag(manifest)
    assert idx == {"a": {"tag": "a", "x": 1}, "b": {"tag": "b", "x": 2}}


def test_item_summary_falls_back_to_basename_when_no_thumb(tmp_path):
    m = {"tag": "t1", "prompt_name": "style_x", "prompt_type": "style",
         "centroid_sim": 0.5, "novelty": 0.25, "strength": 1.0, "cfg": 7.0,
         "file": str(tmp_path / "images" / "t1.png")}
    summary = manifest_index.item_summary(m, tmp_path)
    assert summary["thumb"] == "t1.png"
    assert summary["file"] == "t1.png"
    assert summary["faith"] == 0.5
    assert summary["novelty"] == 0.25


def test_item_summary_uses_thumb_when_present(tmp_path):
    thumbs_dir = tmp_path / "thumbs"
    thumbs_dir.mkdir()
    (thumbs_dir / "t1.jpg").write_bytes(b"x")
    m = {"tag": "t1", "prompt_name": "style_x", "prompt_type": "style",
         "centroid_sim": 0.5, "novelty": 0.25, "strength": 1.0, "cfg": 7.0,
         "file": str(tmp_path / "t1.png")}
    summary = manifest_index.item_summary(m, tmp_path)
    assert summary["thumb"] == "thumbs/t1.jpg"
```

```python
# addition to tests/test_config.py
def test_user_ratings_file_path():
    from clawmarks import config
    assert config.USER_RATINGS_FILE == config.SWEEP_DIR / "user_ratings.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manifest_index.py tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clawmarks.search.manifest_index'` and
`AttributeError: module 'clawmarks.config' has no attribute 'USER_RATINGS_FILE'`

- [ ] **Step 3: Write the implementation**

```python
# src/clawmarks/search/manifest_index.py
"""Shared helpers for looking up scored_manifest.json entries by tag, and for building the
small per-image summary dict several tool pages need. Extracted out of build/elite_archive.py
so curation_server.py's ratings endpoints and build/preference_rank.py can reuse the exact same
summary shape instead of re-deriving it."""
import os


def index_by_tag(manifest):
    return {m["tag"]: m for m in manifest}


def item_summary(m, sweep_dir):
    thumb_path = os.path.join(str(sweep_dir), "thumbs", f"{m['tag']}.jpg")
    return {
        "tag": m["tag"], "prompt_name": m["prompt_name"], "prompt_type": m["prompt_type"],
        "faith": round(m["centroid_sim"], 4), "novelty": round(m["novelty"], 4),
        "strength": m["strength"], "cfg": m["cfg"],
        "thumb": (f"thumbs/{m['tag']}.jpg" if os.path.exists(thumb_path)
                  else os.path.basename(m["file"])),
        "file": os.path.basename(m["file"]),
    }
```

Modify `src/clawmarks/config.py` — add after the existing `USER_PICKS_FILE` line:

```python
USER_PICKS_FILE = SWEEP_DIR / "user_picks.json"
USER_RATINGS_FILE = SWEEP_DIR / "user_ratings.json"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_manifest_index.py tests/test_config.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/search/manifest_index.py src/clawmarks/config.py tests/test_manifest_index.py tests/test_config.py
git commit -m "feat(clawmarks): add shared manifest_index helpers and USER_RATINGS_FILE config"
```

---

### Task 2: Stratified rating sampler

**Files:**
- Create: `src/clawmarks/search/rating_sampler.py`
- Create: `tests/test_rating_sampler.py`

**Interfaces:**
- Consumes: `clawmarks.search.scoring.bin_edges(vals, n)`, `bin_of(val, edges)` (already exist,
  see `tests/test_scoring.py`).
- Produces: `rating_sampler.bin_manifest(manifest) -> dict[tuple[int,int], list[dict]]`,
  `rating_sampler.eligible_grid(manifest, reviewed_tags: set[str]) -> dict[tuple[int,int], list[dict]]`,
  `rating_sampler.pick_next(manifest, reviewed_tags, rng=random) -> dict | None`. `pick_next` is
  consumed directly by Task 4's `GET /api/rate/next` handler.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rating_sampler.py
import random

from clawmarks.search import rating_sampler


def _manifest(n):
    return [{"tag": f"t{i}", "centroid_sim": i / n, "novelty": 1 - i / n} for i in range(n)]


def test_bin_manifest_splits_into_n_bins_by_bin_count():
    manifest = _manifest(16)
    grid = rating_sampler.bin_manifest(manifest)
    assert len(grid) <= rating_sampler.N_BINS * rating_sampler.N_BINS
    assert sum(len(v) for v in grid.values()) == 16


def test_eligible_grid_excludes_reviewed_tags():
    manifest = _manifest(8)
    reviewed = {"t0", "t1", "t2"}
    grid = rating_sampler.eligible_grid(manifest, reviewed)
    all_tags = {m["tag"] for items in grid.values() for m in items}
    assert reviewed.isdisjoint(all_tags)
    assert len(all_tags) == 5


def test_pick_next_returns_none_when_everything_reviewed():
    manifest = _manifest(4)
    reviewed = {m["tag"] for m in manifest}
    assert rating_sampler.pick_next(manifest, reviewed) is None


def test_pick_next_only_returns_eligible_items():
    manifest = _manifest(20)
    reviewed = {f"t{i}" for i in range(15)}
    rng = random.Random(0)
    for _ in range(30):
        item = rating_sampler.pick_next(manifest, reviewed, rng=rng)
        assert item is not None
        assert item["tag"] not in reviewed


def test_pick_next_can_return_from_a_sparsely_populated_bin():
    # one bin has a single eligible item, another has many; over enough draws the sparse
    # bin's item should still turn up, proving bins are chosen uniformly, not by size.
    manifest = _manifest(100)
    reviewed = {m["tag"] for m in manifest[1:50]}  # leaves t0 alone in its low bin
    rng = random.Random(1)
    seen = set()
    for _ in range(200):
        item = rating_sampler.pick_next(manifest, reviewed, rng=rng)
        seen.add(item["tag"])
    assert "t0" in seen
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rating_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clawmarks.search.rating_sampler'`

- [ ] **Step 3: Write the implementation**

```python
# src/clawmarks/search/rating_sampler.py
"""
Stratified sampler for the ratings UI (rate.html / GET /api/rate/next): picks an unreviewed
image to show next, spread across the faithfulness x novelty grid build/elite_archive.py
already uses, so an early rating session doesn't over-sample whichever region happens to
dominate the pool. See docs/superpowers/specs/2026-07-09-preference-classifier-design.md,
Component 2.
"""
import random

from clawmarks.search.scoring import bin_edges, bin_of

N_BINS = 4  # matches build/elite_archive.py's grid


def bin_manifest(manifest):
    faith_vals = sorted(m["centroid_sim"] for m in manifest)
    novelty_vals = sorted(m["novelty"] for m in manifest)
    faith_edges = bin_edges(faith_vals, N_BINS)
    novelty_edges = bin_edges(novelty_vals, N_BINS)
    grid = {}
    for m in manifest:
        fb = bin_of(m["centroid_sim"], faith_edges)
        nb = bin_of(m["novelty"], novelty_edges)
        grid.setdefault((fb, nb), []).append(m)
    return grid


def eligible_grid(manifest, reviewed_tags):
    grid = bin_manifest(manifest)
    return {key: [m for m in items if m["tag"] not in reviewed_tags]
            for key, items in grid.items()}


def pick_next(manifest, reviewed_tags, rng=random):
    """Returns a random manifest item from a randomly chosen non-empty bin, or None if every
    image is already reviewed. Choosing the bin uniformly at random (not weighted by how many
    eligible images remain in it) is what makes this stratified rather than plain random: a bin
    with 5 eligible images is exactly as likely to be sampled from as a bin with 500."""
    grid = eligible_grid(manifest, reviewed_tags)
    nonempty = [items for items in grid.values() if items]
    if not nonempty:
        return None
    bin_items = rng.choice(nonempty)
    return rng.choice(bin_items)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rating_sampler.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/search/rating_sampler.py tests/test_rating_sampler.py
git commit -m "feat(clawmarks): add stratified rating sampler"
```

---

### Task 3: Migration script (picks -> ratings)

**Files:**
- Create: `src/clawmarks/search/migrate_picks_to_ratings.py`
- Create: `tests/test_migrate_picks_to_ratings.py`

**Interfaces:**
- Consumes: `clawmarks.config.USER_PICKS_FILE`, `clawmarks.config.USER_RATINGS_FILE`.
- Produces: `migrate_picks_to_ratings.merge_picks_into_ratings(picks: dict, ratings: dict) -> tuple[dict, list[str]]`
  (pure, unit-tested); `migrate_picks_to_ratings.main(argv=None) -> int` (I/O wrapper, run once
  by hand, not wired into `cli.py` — see step 3's note).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_migrate_picks_to_ratings.py
from clawmarks.search.migrate_picks_to_ratings import merge_picks_into_ratings


def test_migrates_picks_not_already_rated():
    picks = {"a": {"picked_at": "t0"}, "b": {"picked_at": "t1"}}
    ratings = {}
    updated, migrated = merge_picks_into_ratings(picks, ratings)
    assert migrated == ["a", "b"]
    assert updated["a"] == {"label": "yes", "rated_at": "t0"}
    assert updated["b"] == {"label": "yes", "rated_at": "t1"}


def test_does_not_overwrite_an_existing_rating():
    picks = {"a": {"picked_at": "t0"}}
    ratings = {"a": {"label": "no", "rated_at": "t9"}}
    updated, migrated = merge_picks_into_ratings(picks, ratings)
    assert migrated == []
    assert updated["a"] == {"label": "no", "rated_at": "t9"}


def test_leaves_existing_ratings_not_derived_from_picks_untouched():
    picks = {}
    ratings = {"c": {"label": "yes", "rated_at": "t5"}}
    updated, migrated = merge_picks_into_ratings(picks, ratings)
    assert migrated == []
    assert updated == {"c": {"label": "yes", "rated_at": "t5"}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_migrate_picks_to_ratings.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/clawmarks/search/migrate_picks_to_ratings.py
"""One-time migration: user_picks.json entries become user_ratings.json entries with
label: "yes", so "pick as winner" can be retired without losing the existing picks. Safe to
rerun: any tag that already has a rating is left alone. Not wired into `clawmarks` as a
permanent CLI subcommand since it's a one-shot migration, not a recurring operation. See
docs/superpowers/specs/2026-07-09-preference-classifier-design.md, Component 2a.

Run with: python -m clawmarks.search.migrate_picks_to_ratings
"""
import json
import os

from clawmarks.config import USER_PICKS_FILE, USER_RATINGS_FILE


def merge_picks_into_ratings(picks, ratings):
    """Returns (updated_ratings, migrated_tags). Does not overwrite a tag that already has a
    rating, whatever its label, since a rating recorded through rate.html reflects a more
    deliberate, later judgment than an old pick."""
    updated = dict(ratings)
    migrated = []
    for tag, pick in picks.items():
        if tag in updated:
            continue
        updated[tag] = {"label": "yes", "rated_at": pick.get("picked_at")}
        migrated.append(tag)
    return updated, migrated


def main(argv=None):
    picks = {}
    if USER_PICKS_FILE.exists():
        with open(USER_PICKS_FILE) as f:
            picks = json.load(f)
    ratings = {}
    if USER_RATINGS_FILE.exists():
        with open(USER_RATINGS_FILE) as f:
            ratings = json.load(f)

    updated, migrated = merge_picks_into_ratings(picks, ratings)
    tmp = str(USER_RATINGS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(updated, f, indent=1)
    os.replace(tmp, USER_RATINGS_FILE)

    print(f"migrated {len(migrated)} picks into {USER_RATINGS_FILE} as yes-ratings "
          f"({len(picks) - len(migrated)} already had a rating and were left alone)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_migrate_picks_to_ratings.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/search/migrate_picks_to_ratings.py tests/test_migrate_picks_to_ratings.py
git commit -m "feat(clawmarks): add one-time picks-to-ratings migration script"
```

---

### Task 4: Retire pick endpoints, add ratings endpoints to `curation_server.py`

**Files:**
- Modify: `src/clawmarks/curation_server.py`
- Create: `tests/test_curation_server.py`

**Interfaces:**
- Consumes: `rating_sampler.pick_next` (Task 2), `manifest_index.item_summary`,
  `manifest_index.index_by_tag` (Task 1).
- Produces: HTTP `GET /api/ratings`, `GET /api/rate/next`, `POST /api/rate` (consumed by Task 5's
  `rate.html` and Task 6's `elite_archive.py`). Removes `GET /api/picks`, `POST /api/pick`,
  `POST /api/unpick`.

This task tests the pure logic directly (matching this repo's existing pattern — see
`tests/test_seed_pool.py`) rather than booting a real `ThreadingHTTPServer`, since the handler
class only does thin I/O around functions worth testing on their own.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_curation_server.py
from clawmarks import curation_server as cs


def test_next_rating_response_returns_item_summary_shape():
    manifest = [
        {"tag": "a", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.5,
         "novelty": 0.3, "strength": 1.0, "cfg": 7.0, "file": "a.png"},
    ]
    result = cs.next_rating_response(manifest, reviewed_tags=set())
    assert result["tag"] == "a"
    assert result["faith"] == 0.5
    assert "done" not in result


def test_next_rating_response_reports_done_when_all_reviewed():
    manifest = [{"tag": "a", "centroid_sim": 0.5, "novelty": 0.3, "prompt_name": "p",
                 "prompt_type": "style", "strength": 1.0, "cfg": 7.0, "file": "a.png"}]
    result = cs.next_rating_response(manifest, reviewed_tags={"a"})
    assert result == {"done": True}


def test_record_rating_upserts_with_timestamp():
    ratings = {}
    updated = cs.record_rating(ratings, "a", "yes", now="2026-07-10T00:00:00Z")
    assert updated["a"] == {"label": "yes", "rated_at": "2026-07-10T00:00:00Z"}


def test_record_rating_overwrites_not_duplicates():
    ratings = {"a": {"label": "no", "rated_at": "t0"}}
    updated = cs.record_rating(ratings, "a", "yes", now="t1")
    assert updated == {"a": {"label": "yes", "rated_at": "t1"}}
    assert len(updated) == 1


def test_record_rating_rejects_invalid_label():
    try:
        cs.record_rating({}, "a", "maybe", now="t0")
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_curation_server.py -v`
Expected: FAIL with `AttributeError: module 'clawmarks.curation_server' has no attribute 'next_rating_response'`

- [ ] **Step 3: Modify `curation_server.py`**

Add imports (after the existing `from clawmarks.search.seed_pool import merge as seed_pool_merge` line):

```python
from clawmarks.search import rating_sampler
from clawmarks.search.manifest_index import item_summary
```

Replace the `PICKS_FILE` constant and add `RATINGS_FILE`:

```python
FAVORITES_FILE = f"{SWEEP_DIR}/user_favorites.json"
RATINGS_FILE = f"{SWEEP_DIR}/user_ratings.json"
```

(Delete the old `PICKS_FILE = f"{SWEEP_DIR}/user_picks.json"` line entirely — nothing reads
`user_picks.json` after this task, per the spec's Component 2a.)

Delete `load_picks()` and `save_picks(picks)` (no longer used anywhere).

Add pure helper functions (near the top-level functions, after `save_store`):

```python
def next_rating_response(manifest, reviewed_tags, rng=None):
    """Returns an item_summary dict for the next image to rate, or {"done": True} if every
    image in `manifest` is already in `reviewed_tags`."""
    item = rating_sampler.pick_next(manifest, reviewed_tags, rng=rng) if rng is not None \
        else rating_sampler.pick_next(manifest, reviewed_tags)
    if item is None:
        return {"done": True}
    return item_summary(item, SWEEP_DIR)


def record_rating(ratings, tag, label, now):
    if label not in ("yes", "no"):
        raise ValueError(f"label must be 'yes' or 'no', got {label!r}")
    updated = dict(ratings)
    updated[tag] = {"label": label, "rated_at": now}
    return updated


_manifest_cache = {"manifest": None}


def load_manifest():
    if _manifest_cache["manifest"] is None:
        with open(f"{SWEEP_DIR}/scored_manifest.json") as f:
            _manifest_cache["manifest"] = json.load(f)
    return _manifest_cache["manifest"]
```

In `Handler.do_GET`, delete the `/api/picks` branch:

```python
        if self.path == "/api/picks":
            with _lock:
                self._json_response(200, load_picks())
            return
```

and replace it with:

```python
        if self.path == "/api/ratings":
            with _lock:
                self._json_response(200, load_store(RATINGS_FILE))
            return
        if self.path == "/api/rate/next":
            with _lock:
                ratings = load_store(RATINGS_FILE)
                favorites = load_store(FAVORITES_FILE)
                reviewed = set(ratings) | set(favorites)
                response = next_rating_response(load_manifest(), reviewed)
            self._json_response(200, response)
            return
```

In `Handler.do_POST`, delete the `/api/pick` and `/api/unpick` branches entirely:

```python
        if self.path == "/api/pick":
            tag = payload.get("tag")
            if not tag:
                self._json_response(400, {"error": "missing 'tag'"})
                return
            with _lock:
                picks = load_picks()
                payload["picked_at"] = datetime.now(timezone.utc).isoformat()
                picks[tag] = payload
                save_picks(picks)
            self._json_response(200, {"ok": True, "count": len(picks)})
            return

        if self.path == "/api/unpick":
            tag = payload.get("tag")
            with _lock:
                picks = load_picks()
                picks.pop(tag, None)
                save_picks(picks)
            self._json_response(200, {"ok": True, "count": len(picks)})
            return
```

and add in their place:

```python
        if self.path == "/api/rate":
            tag = payload.get("tag")
            label = payload.get("label")
            if not tag or label not in ("yes", "no"):
                self._json_response(400, {"error": "missing 'tag' or invalid 'label' (must be 'yes' or 'no')"})
                return
            with _lock:
                ratings = load_store(RATINGS_FILE)
                ratings = record_rating(ratings, tag, label, datetime.now(timezone.utc).isoformat())
                save_store(RATINGS_FILE, ratings)
            self._json_response(200, {"ok": True, "count": len(ratings)})
            return
```

Update the module docstring's `API:` block: remove the `GET /api/picks` / `POST /api/pick` /
`POST /api/unpick` lines and add:

```
  GET  /api/ratings           -> {tag: {label, rated_at}}
  GET  /api/rate/next         -> item_summary dict for the next unreviewed image, or {"done": true}
  POST /api/rate               body: {"tag": "...", "label": "yes"|"no"} -> upserts, returns ok
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_curation_server.py -v`
Expected: PASS (5 tests)

Run the full suite to confirm nothing else broke: `pytest -v`
Expected: all tests PASS (this task removes `load_picks`/`save_picks`; confirm no other module
imports them — `grep -rn "load_picks\|save_picks" src/` should return nothing).

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/curation_server.py tests/test_curation_server.py
git commit -m "feat(clawmarks): replace pick endpoints with ratings endpoints in curation_server"
```

---

### Task 5: Remove "pick as winner" from the lightbox; add `rate.html`

**Files:**
- Modify: `src/clawmarks/shared_ui.py`
- Create: `src/clawmarks/build/rate_page.py`
- Modify: `src/clawmarks/cli.py`
- Create: `tests/test_rate_page.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `build.rate_page.main(argv=None)` writes `rate.html`, wired into
  `clawmarks build rate`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rate_page.py
from clawmarks.build import rate_page


def test_main_writes_rate_html(tmp_path, monkeypatch):
    monkeypatch.setattr(rate_page, "SWEEP_DIR", tmp_path)
    rate_page.main([])
    out = tmp_path / "rate.html"
    assert out.exists()
    content = out.read_text()
    assert "/api/rate/next" in content
    assert "/api/rate" in content
```

```python
# addition to tests/test_cli.py
def test_build_rate_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["build", "rate"])
    assert args.command == "build"
    assert args.target == "rate"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rate_page.py tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clawmarks.build.rate_page'` and an
`argparse` error for the unrecognized `rate` choice.

- [ ] **Step 3: Modify `shared_ui.py`**

Add `"rate.html"` to `NAV_OPTIONS`, right after `"explore.html"`:

```python
NAV_OPTIONS = [
    ("explore.html", "all tools (hub)"),
    ("rate.html", "rate images (yes/no)"),
    ("scan.html", "scan gallery"),
```

Remove the pick button and its tooltip from `_LIGHTBOX_JS`'s `el.innerHTML` template. Change:

```javascript
    <button class="lb-back" style="display:none;">&#8592; back</button>
    <button class="lb-pick">&#9733; pick as winner</button>
    <span class="infobtn" data-id="lb-tip-pick" data-tip="Picking marks this image as a human-approved success. The next search generation uses picked images as starting points for new variations, ahead of the algorithm's own ranking: it's how your judgment steers where the search goes next.">?</span>
    <button class="lb-favorite">&#9825; favorite</button>
```

to:

```javascript
    <button class="lb-back" style="display:none;">&#8592; back</button>
    <button class="lb-favorite">&#9825; favorite</button>
```

Remove the now-unused CSS rule `#lb-overlay button.picked { ... }` from the style block.

Remove the `let picks = {};` declaration (search the file for `let favorites = {};` — the pick
variable sits right before it).

Remove the wiring line `el.querySelector('.lb-pick').onclick = togglePick;` from the DOM-setup
block (keep `el.querySelector('.lb-favorite').onclick = toggleFavorite;`).

Remove the pick keyboard shortcut from the `keydown` listener:

```javascript
      if (e.key === ' ') { e.preventDefault(); togglePick(); }
```

Remove `loadPicks()`:

```javascript
  function loadPicks(){
    return fetch('/api/picks').then(r => r.json()).then(p => { picks = p; }).catch(() => {});
  }
```

Change the `Promise.all` call in `open()` from:

```javascript
    Promise.all([loadData(), loadPicks(), loadFavorites(), loadCounterfactuals()]).then(() => {
```

to:

```javascript
    Promise.all([loadData(), loadFavorites(), loadCounterfactuals()]).then(() => {
```

Remove the pick-button rendering block from `render()`:

```javascript
    const isPicked = !!picks[d.tag];
    const pickBtn = el.querySelector('.lb-pick');
    pickBtn.textContent = isPicked ? '★ picked (click to unpick)' : '☆ pick as winner';
    pickBtn.classList.toggle('picked', isPicked);
```

Remove the `togglePick()` function entirely:

```javascript
  function togglePick(){
    const d = order[idx];
    const isPicked = !!picks[d.tag];
    const endpoint = isPicked ? '/api/unpick' : '/api/pick';
    const body = isPicked ? {tag: d.tag} : Object.assign({}, d);
    fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})
      .then(r => r.json())
      .then(() => {
        if (isPicked) delete picks[d.tag]; else picks[d.tag] = body;
        render();
        document.dispatchEvent(new CustomEvent('lightbox:pick', {detail: {tag: d.tag, picked: !isPicked}}));
      });
  }
```

(Keep `toggleFavorite()` exactly as-is.)

- [ ] **Step 4: Create `build/rate_page.py`**

```python
# src/clawmarks/build/rate_page.py
"""
Generates rate.html: a full-screen, keyboard-driven yes/no rating page. Unlike every other
build/*.py generator, this page bakes in no per-image data at build time — it fetches
GET /api/rate/next itself and POSTs to /api/rate, both served by curation_server.py, so the page
never goes stale between rebuilds. Rebuilding only matters if this file itself changes.

Run with: python3 -m clawmarks.build.rate_page (or `clawmarks build rate`)
"""
from clawmarks.config import SWEEP_DIR
from clawmarks.shared_ui import (
    nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, write_scrollnav_asset, write_infotip_asset,
    INFOTIP_CSS, info_btn,
)


def main(argv=None):
    write_scrollnav_asset(SWEEP_DIR)
    write_infotip_asset(SWEEP_DIR)

    rate_tip = info_btn(
        "Rating trains the preference classifier: yes/no on as many images as you can stand "
        "to look at. Yes-rated images immediately take over the search's exploit pool (the same "
        "role picking used to play); once enough ratings exist, a model trained on them takes "
        "over ranking automatically."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS rate</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee;
  --text-dim:#9a9aa4; --yes:#5ec98a; --no:#e0605e; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px;
  display:flex; flex-direction:column; align-items:center; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
h1 {{ font-size:18px; margin:0 0 4px; align-self:flex-start; }}
p.sub {{ color:var(--text-dim); max-width:640px; font-size:13px; line-height:1.6; align-self:flex-start; }}
#stage {{ margin-top:20px; width:100%; max-width:640px; display:flex; flex-direction:column; align-items:center; }}
#img {{ max-width:100%; max-height:60vh; border-radius:10px; box-shadow:0 20px 60px rgba(0,0,0,0.6); }}
#meta {{ color:var(--text-dim); font-size:12.5px; margin-top:10px; text-align:center; }}
#buttons {{ display:flex; gap:16px; margin-top:18px; }}
#buttons button {{ font-size:16px; padding:14px 34px; border-radius:10px; cursor:pointer; border:1px solid var(--border); background:var(--panel); color:var(--text); }}
#buttons .no {{ border-color:var(--no); color:var(--no); }}
#buttons .yes {{ border-color:var(--yes); color:var(--yes); }}
#count {{ color:var(--text-dim); font-size:12px; margin-top:14px; }}
#done {{ color:var(--text-dim); font-size:14px; margin-top:40px; text-align:center; }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('rate.html')}
<h1>Rate{rate_tip}</h1>
<p class="sub">Yes or no, as fast as you can go. Keyboard: &larr; or n = no, &rarr; or y = yes.</p>

<div id="stage">
  <img id="img" style="display:none;">
  <div id="meta"></div>
  <div id="buttons" style="display:none;">
    <button class="no" onclick="rate('no')">&larr; no</button>
    <button class="yes" onclick="rate('yes')">yes &rarr;</button>
  </div>
  <div id="done" style="display:none;">Nothing left to rate right now &mdash; every image in the pool has been rated or favorited.</div>
</div>
<div id="count"></div>

<script>
let current = null;
let ratedThisSession = 0;

function loadNext() {{
  document.getElementById('buttons').style.display = 'none';
  fetch('/api/rate/next').then(r => r.json()).then(d => {{
    if (d.done) {{
      current = null;
      document.getElementById('img').style.display = 'none';
      document.getElementById('done').style.display = 'block';
      return;
    }}
    current = d;
    const img = document.getElementById('img');
    img.src = d.thumb;
    img.style.display = 'block';
    document.getElementById('meta').textContent =
      `${{d.prompt_name}} | faith=${{d.faith}} novelty=${{d.novelty}}`;
    document.getElementById('buttons').style.display = 'flex';
  }});
}}

function rate(label) {{
  if (!current) return;
  const tag = current.tag;
  fetch('/api/rate', {{method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{tag, label}})}})
    .then(r => r.json())
    .then(() => {{
      ratedThisSession++;
      document.getElementById('count').textContent = `${{ratedThisSession}} rated this session`;
      loadNext();
    }});
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft' || e.key === 'n' || e.key === 'N') rate('no');
  if (e.key === 'ArrowRight' || e.key === 'y' || e.key === 'Y') rate('yes');
}});

loadNext();
</script>
<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    with open(f"{SWEEP_DIR}/rate.html", "w") as f:
        f.write(html)

    print(f"wrote {SWEEP_DIR}/rate.html", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Wire `rate` into `cli.py`**

In `_build_targets()`, add `rate_page` to the import and the returned dict:

```python
    from clawmarks.build import (
        scan_gallery, elite_archive, coverage_map, map_view, redundancy_view,
        novelty_decay, lineage_view, solution_map, similarity_index, thumbnails,
        explore_hub, seed_browser, probe_report, uncanny_gallery, rate_page,
    )
    return {
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
        "rate": rate_page.main,
    }
```

Add `"rate"` to the `build_p.add_argument("target", choices=[...])` list.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_rate_page.py tests/test_cli.py -v`
Expected: PASS (2 new tests, plus existing `test_cli.py` tests still pass)

Run the full suite: `pytest -v`
Expected: all PASS. `grep -rn "lb-pick\|togglePick\|loadPicks" src/` should return nothing.

- [ ] **Step 7: Commit**

```bash
git add src/clawmarks/shared_ui.py src/clawmarks/build/rate_page.py src/clawmarks/cli.py tests/test_rate_page.py tests/test_cli.py
git commit -m "feat(clawmarks): remove pick-as-winner from the lightbox, add rate.html"
```

---

### Task 6: `build/elite_archive.py` reads ratings, not picks

**Files:**
- Modify: `src/clawmarks/build/elite_archive.py`
- Create: `tests/test_elite_archive.py`

**Interfaces:**
- Consumes: `manifest_index.item_summary` (Task 1), replacing the file's own inline
  `item_summary` definition.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_elite_archive.py
import json
import re

from clawmarks.build import elite_archive


def test_main_uses_yes_rated_images_not_user_picks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(elite_archive, "SWEEP_DIR", tmp_path)
    # Force every image into a single cell, regardless of its faith/novelty values, so the test
    # doesn't depend on how a 2-item manifest happens to quantile-split across N_BINS x N_BINS
    # cells (bin_edges(vals, 1) always returns [], so bin_of always returns 0).
    monkeypatch.setattr(elite_archive, "N_BINS", 1)
    manifest = [
        {"tag": "a", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.9,
         "novelty": 0.1, "strength": 1.0, "cfg": 7.0, "file": "a.png"},
        {"tag": "b", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.9,
         "novelty": 0.9, "strength": 1.0, "cfg": 7.0, "file": "b.png"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    # "a" has lower novelty than "b" but is yes-rated: it should win the cell despite that,
    # exactly the behavior user_picks.json used to provide.
    (tmp_path / "user_ratings.json").write_text(json.dumps({"a": {"label": "yes", "rated_at": "t0"}}))
    # a stale user_picks.json should be ignored entirely
    (tmp_path / "user_picks.json").write_text(json.dumps({"b": {"picked_at": "t0"}}))

    elite_archive.main([])

    captured = capsys.readouterr()
    assert "1 occupied cells, 1 human-picked elites" in captured.out

    html = (tmp_path / "archive.html").read_text()
    match = re.search(r"const CELLS = (\[.+?\]);\nlet picks", html)
    assert match is not None, "could not find 'const CELLS = [...]; let picks' in archive.html"
    cells = json.loads(match.group(1))
    assert len(cells) == 1
    tags_in_cell = {item["tag"] for item in cells[0]["items"]}
    assert tags_in_cell == {"a", "b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_elite_archive.py -v`
Expected: FAIL — the current code reads `user_picks.json`, so `picks = {"b": ...}` (not `{"a":
...}`), and the printed line reads `1 occupied cells, 1 human-picked elites` for the wrong tag
(`b`, not `a`); the test's own `tags_in_cell` assertion still passes (both tags are in the one
forced cell either way) but a manual check confirms `n_human` is counted against `b`'s pick, not
`a`'s rating, before the fix. The test as written will start passing only once Step 3's change
makes `picks` come from `user_ratings.json`'s yes-labels instead of `user_picks.json` — until
then, it fails because `elite_archive.py` still has the inline `item_summary` function and the
`picks_path`/`user_picks.json` loading block Step 3 removes, which the test doesn't yet
reference but which govern `n_human`'s count in a way not yet driven by ratings.

- [ ] **Step 3: Modify `elite_archive.py`**

Change the import block:

```python
from clawmarks.config import SWEEP_DIR
from clawmarks.search.manifest_index import item_summary
from clawmarks.shared_ui import (
    nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, write_lightbox_asset, write_scrollnav_asset,
    write_infotip_asset, INFOTIP_CSS, info_btn,
)
```

Update the module docstring's second paragraph to say:

```
Elite selection per cell: a yes-rated image (notes/uncanny_sweep/user_ratings.json) wins if one
exists in that cell, since a person's judgment substitutes for the coherence/quality scorer this
project doesn't have (lab_notebook.md Section 3b). Otherwise falls back to highest novelty in
the cell, matching the ranking the search itself uses to build its automated "elites" list.
```

Replace the picks-loading block:

```python
    picks = {}
    picks_path = f"{SWEEP_DIR}/user_picks.json"
    if os.path.exists(picks_path):
        with open(picks_path) as f:
            picks = json.load(f)
```

with:

```python
    ratings = {}
    ratings_path = f"{SWEEP_DIR}/user_ratings.json"
    if os.path.exists(ratings_path):
        with open(ratings_path) as f:
            ratings = json.load(f)
    picks = {tag: r for tag, r in ratings.items() if r.get("label") == "yes"}
```

(Everything downstream — `n_human`, `picked_here`, `.human` CSS class — keeps working unchanged
since `picks` still ends up as "the set of tags that won their cell.")

Delete the inline `item_summary` function:

```python
    def item_summary(m):
        return {
            "tag": m["tag"], "prompt_name": m["prompt_name"], "prompt_type": m["prompt_type"],
            "faith": round(m["centroid_sim"], 4), "novelty": round(m["novelty"], 4),
            "strength": m["strength"], "cfg": m["cfg"],
            "thumb": (f"thumbs/{m['tag']}.jpg" if os.path.exists(f"{SWEEP_DIR}/thumbs/{m['tag']}.jpg")
                      else os.path.basename(m["file"])),
            "file": os.path.basename(m["file"]),
        }
```

and update its call site:

```python
                "items": [item_summary(m) for m in sorted(items, key=lambda m: -m["novelty"])],
```

to:

```python
                "items": [item_summary(m, SWEEP_DIR) for m in sorted(items, key=lambda m: -m["novelty"])],
```

In the JS template, rename the source label so it matches reality. Change:

```javascript
function eliteFor(c) {{
  const pickedHere = c.items.filter(it => picks[it.tag]);
  if (pickedHere.length) return {{ item: pickedHere[0], source: 'human pick' }};
  return {{ item: c.items[0], source: 'highest novelty' }};  // items pre-sorted by -novelty
}}
```

to:

```javascript
function eliteFor(c) {{
  const pickedHere = c.items.filter(it => picks[it.tag]);
  if (pickedHere.length) return {{ item: pickedHere[0], source: 'yes-rated' }};
  return {{ item: c.items[0], source: 'highest novelty' }};  // items pre-sorted by -novelty
}}
```

Update the one place that compares against the old string:

```javascript
  const human = source === 'human pick';
```

to:

```javascript
  const human = source === 'yes-rated';
```

Replace the picks-fetch-plus-live-update block at the bottom of the script (the pick button that
fired `lightbox:pick` no longer exists after Task 5, so this listener never fires):

```javascript
document.addEventListener('lightbox:pick', e => {{
  if (e.detail.picked) picks[e.detail.tag] = true; else delete picks[e.detail.tag];
  render();
  if (document.getElementById('modal').classList.contains('open')) {{
    document.querySelectorAll('#modalGrid .item').forEach(el => {{
      el.classList.toggle('human', !!picks[el.title]);
    }});
  }}
}});

fetch('/api/picks').then(r => r.json()).then(p => {{ picks = p; render(); }}).catch(() => {{ render(); }});
```

with:

```javascript
fetch('/api/ratings').then(r => r.json()).then(ratings => {{
  picks = {{}};
  Object.entries(ratings).forEach(([tag, r]) => {{ if (r.label === 'yes') picks[tag] = true; }});
  render();
}}).catch(() => {{ render(); }});
```

Update the page's descriptive copy: change `"Gold-bordered cells are human-picked winners;"` to
`"Gold-bordered cells are yes-rated winners;"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_elite_archive.py -v`
Expected: PASS

Run the full suite: `pytest -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/build/elite_archive.py tests/test_elite_archive.py
git commit -m "feat(clawmarks): elite archive reads yes-ratings instead of picks"
```

---

### Task 7: `search/driver.py` exploit pool reads yes-ratings (Stage 5a)

**Files:**
- Modify: `src/clawmarks/search/driver.py`
- Modify: `tests/test_generation_jobs.py` (no changes expected — confirms `build_generation_jobs`
  itself is untouched; this task only changes what feeds it)
- Create: `tests/test_yes_rated_images.py`

**Interfaces:**
- Consumes: `manifest_index.index_by_tag` (Task 1).
- Produces: `driver._load_yes_rated_images() -> list[dict]`, replacing `_load_user_picks()`
  everywhere it was called.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_yes_rated_images.py
import json

from clawmarks.search import driver


def test_load_yes_rated_images_joins_ratings_against_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SWEEP_DIR", tmp_path)
    manifest = [
        {"tag": "a", "prompt_name": "p", "prompt": "trentbuckle style, a", "strength": 1.0,
         "cfg": 7.0, "centroid_sim": 0.5, "novelty": 0.5, "file": "a.png"},
        {"tag": "b", "prompt_name": "p", "prompt": "trentbuckle style, b", "strength": 1.0,
         "cfg": 7.0, "centroid_sim": 0.5, "novelty": 0.5, "file": "b.png"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "user_ratings.json").write_text(json.dumps({
        "a": {"label": "yes", "rated_at": "t0"},
        "b": {"label": "no", "rated_at": "t0"},
    }))
    result = driver._load_yes_rated_images()
    assert [m["tag"] for m in result] == ["a"]


def test_load_yes_rated_images_returns_empty_without_files(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SWEEP_DIR", tmp_path)
    assert driver._load_yes_rated_images() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_yes_rated_images.py -v`
Expected: FAIL with `AttributeError: module 'clawmarks.search.driver' has no attribute '_load_yes_rated_images'`

- [ ] **Step 3: Modify `driver.py`**

Add to the imports near the top of the file:

```python
from clawmarks.search.manifest_index import index_by_tag
```

Replace `_load_user_picks()`:

```python
def _load_user_picks():
    """Human-in-the-loop MAP-Elites: this project has no automated coherence/quality scorer,
    so per lab_notebook.md Section 3b there's no way for an image to automatically 'win' a
    bin. A person reviewing notes/uncanny_sweep/scan.html (served by
    notes/curation_server.py, which is what actually persists picks) can mark specific images
    as winners instead. When present, those picks anchor the exploit step's mutations in
    place of the raw novelty ranking, which is only ever a proxy for 'interesting,' not a
    verdict on it."""
    if SWEEP_DIR.joinpath("user_picks.json").exists():
        with open(SWEEP_DIR / "user_picks.json") as f:
            picks = json.load(f)
        return list(picks.values())
    return []
```

with:

```python
def _load_yes_rated_images():
    """Ratings supersede picks: a human's yes/no judgment on an image, not raw novelty, decides
    what the exploit step mutates near. user_ratings.json stores only {label, rated_at} per tag
    (the image metadata already lives in scored_manifest.json), so yes-rated tags are joined
    against that manifest to recover prompt/strength/cfg for mutation."""
    ratings_path = SWEEP_DIR / "user_ratings.json"
    manifest_path = SWEEP_DIR / "scored_manifest.json"
    if not ratings_path.exists() or not manifest_path.exists():
        return []
    with open(ratings_path) as f:
        ratings = json.load(f)
    yes_tags = {tag for tag, r in ratings.items() if r.get("label") == "yes"}
    if not yes_tags:
        return []
    with open(manifest_path) as f:
        manifest = json.load(f)
    by_tag = index_by_tag(manifest)
    return [by_tag[t] for t in yes_tags if t in by_tag]
```

Update the call site in `main()`:

```python
        user_picks = _load_user_picks() if cfg.seed_from_start else []
```

to:

```python
        user_picks = _load_yes_rated_images() if cfg.seed_from_start else []
```

(`build_generation_jobs`'s parameter stays named `user_picks` — it's an internal name meaning
"the exploit seed pool," and `tests/test_generation_jobs.py` already exercises it directly by
keyword; renaming it is out of scope for this task and would touch tests that don't need to
change.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_yes_rated_images.py tests/test_generation_jobs.py -v`
Expected: PASS (2 new tests, 4 existing tests untouched and still passing)

Run the full suite: `pytest -v`
Expected: all PASS. `grep -rn "_load_user_picks\|user_picks.json" src/` should return nothing
under `src/clawmarks/search/driver.py` (a hit in `migrate_picks_to_ratings.py` and `config.py` is
expected and correct — those still read the historical file on purpose).

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/search/driver.py tests/test_yes_rated_images.py
git commit -m "feat(clawmarks): search driver exploit pool reads yes-ratings instead of picks"
```

---

### Task 8: DINOv2 embedding cache

**Files:**
- Create: `src/clawmarks/search/embed_cache.py`
- Create: `tests/test_embed_cache.py`

**Interfaces:**
- Produces: `embed_cache.EMBEDDINGS_FILE: Path`, `embed_cache.embed_paths(paths, model, batch_size=16) -> np.ndarray`,
  `embed_cache.load_cache(path) -> tuple[list[str], np.ndarray]`,
  `embed_cache.save_cache(path, tags, embeddings)`,
  `embed_cache.missing_tags(manifest_tags, cached_tags) -> list[str]`,
  `embed_cache.sync(manifest, cache_path, model, image_path_for) -> tuple[list[str], np.ndarray]`.
  Tasks 9, 10, and 11 all consume `load_cache`/`sync`/`EMBEDDINGS_FILE`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embed_cache.py
import numpy as np
import torch
from PIL import Image

from clawmarks.search import embed_cache


class FakeOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class FakeModel:
    """Deterministic per-image 'embedding' derived from mean pixel value, so tests exercise
    embed_cache's own logic (batching, ordering, caching) without loading the real (slow,
    network-fetched) DINOv2 model."""
    def __call__(self, pixel_values):
        means = pixel_values.mean(dim=(1, 2, 3))
        feats = torch.stack([means, -means], dim=1)
        return FakeOutput(feats)


def _write_image(path, color):
    Image.new("RGB", (32, 32), color=color).save(path)


def test_embed_paths_returns_one_normalized_row_per_path(tmp_path):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    _write_image(p1, (10, 10, 10))
    _write_image(p2, (200, 200, 200))
    embs = embed_cache.embed_paths([str(p1), str(p2)], FakeModel())
    assert embs.shape == (2, 2)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_save_and_load_cache_round_trips(tmp_path):
    path = tmp_path / "embeddings.npz"
    tags = ["a", "b"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    embed_cache.save_cache(path, tags, embeddings)
    loaded_tags, loaded_embeddings = embed_cache.load_cache(path)
    assert loaded_tags == tags
    assert np.allclose(loaded_embeddings, embeddings)


def test_load_cache_missing_file_returns_empty(tmp_path):
    tags, embeddings = embed_cache.load_cache(tmp_path / "missing.npz")
    assert tags == []
    assert embeddings.shape == (0, 0)


def test_missing_tags_returns_manifest_tags_not_in_cache():
    assert embed_cache.missing_tags(["a", "b", "c"], ["a", "c"]) == ["b"]


def test_sync_adds_only_missing_tags_and_persists(tmp_path):
    manifest = [{"tag": "a"}, {"tag": "b"}]
    _write_image(tmp_path / "a.png", (10, 10, 10))
    _write_image(tmp_path / "b.png", (200, 200, 200))
    cache_path = tmp_path / "embeddings.npz"

    def image_path_for(tag):
        return str(tmp_path / f"{tag}.png")

    tags, embeddings = embed_cache.sync(manifest, cache_path, FakeModel(), image_path_for)
    assert tags == ["a", "b"]
    assert embeddings.shape == (2, 2)

    manifest.append({"tag": "c"})
    _write_image(tmp_path / "c.png", (100, 50, 25))
    tags2, embeddings2 = embed_cache.sync(manifest, cache_path, FakeModel(), image_path_for)
    assert tags2 == ["a", "b", "c"]
    assert np.allclose(embeddings2[:2], embeddings)


def test_sync_raises_on_missing_image_file(tmp_path):
    manifest = [{"tag": "missing"}]
    cache_path = tmp_path / "embeddings.npz"

    def image_path_for(tag):
        return str(tmp_path / "does_not_exist.png")

    try:
        embed_cache.sync(manifest, cache_path, FakeModel(), image_path_for)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embed_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clawmarks.search.embed_cache'`

- [ ] **Step 3: Write the implementation**

```python
# src/clawmarks/search/embed_cache.py
"""
DINOv2 embedding cache: computes and persists an embedding per image in scored_manifest.json so
the preference model (search/preference_model.py) can train on frozen features without
re-running the (slow) DINOv2 model every time. Runs locally, no RunPod cost. See
docs/superpowers/specs/2026-07-09-preference-classifier-design.md, Component 1.

Run with: python -m clawmarks.search.embed_cache
"""
import json
import os

import numpy as np
import torch
from PIL import Image

from clawmarks.config import SWEEP_DIR

MODEL_ID = "facebook/dinov2-base"
EMBEDDINGS_FILE = SWEEP_DIR / "embeddings.npz"

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess(img):
    img = img.convert("RGB")
    w, h = img.size
    shortest = 256
    if w < h:
        new_w, new_h = shortest, round(h * shortest / w)
    else:
        new_h, new_w = shortest, round(w * shortest / h)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = (t - IMAGE_MEAN) / IMAGE_STD
    return t


def embed_paths(paths, model, batch_size=16):
    """Returns an (N, D) float32 array of L2-normalized embeddings, one row per path, in the
    same order as `paths`."""
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            tensors = [preprocess(Image.open(p)) for p in batch_paths]
            pixel_values = torch.stack(tensors, dim=0)
            out = model(pixel_values=pixel_values)
            feats = out.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.detach().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def load_cache(path):
    """Returns (tags, embeddings). Empty list/array if the file doesn't exist yet."""
    if not os.path.exists(path):
        return [], np.zeros((0, 0), dtype=np.float32)
    data = np.load(path)
    return list(data["tags"]), data["embeddings"]


def save_cache(path, tags, embeddings):
    tmp = str(path) + ".tmp"
    np.savez(tmp, tags=np.array(tags), embeddings=np.asarray(embeddings, dtype=np.float32))
    os.replace(tmp, path)


def missing_tags(manifest_tags, cached_tags):
    cached = set(cached_tags)
    return [t for t in manifest_tags if t not in cached]


def sync(manifest, cache_path, model, image_path_for):
    """Loads the existing cache, embeds any manifest tag missing from it, appends, and saves.
    `image_path_for(tag)` resolves a manifest tag to its image file path. Raises
    FileNotFoundError (listing the offending tag) if a manifest tag's image file doesn't exist,
    rather than silently skipping it. Returns (tags, embeddings) for the full, updated cache."""
    tags, embeddings = load_cache(cache_path)
    manifest_tags = [m["tag"] for m in manifest]
    to_add = missing_tags(manifest_tags, tags)
    if not to_add:
        return tags, embeddings

    missing_paths = []
    for t in to_add:
        p = image_path_for(t)
        if not os.path.exists(p):
            raise FileNotFoundError(f"tag {t!r} is in the manifest but its image file is missing: {p}")
        missing_paths.append(p)

    new_embeddings = embed_paths(missing_paths, model)
    all_tags = list(tags) + to_add
    all_embeddings = new_embeddings if embeddings.size == 0 else np.concatenate([embeddings, new_embeddings], axis=0)
    save_cache(cache_path, all_tags, all_embeddings)
    return all_tags, all_embeddings


def main(argv=None):
    from transformers import AutoModel

    with open(SWEEP_DIR / "scored_manifest.json") as f:
        manifest = json.load(f)
    by_tag = {m["tag"]: m for m in manifest}

    def image_path_for(tag):
        return str(SWEEP_DIR / by_tag[tag]["file"])

    print("loading DINOv2 model...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    tags, _ = sync(manifest, EMBEDDINGS_FILE, model, image_path_for)
    print(f"embedding cache now covers {len(tags)} images at {EMBEDDINGS_FILE}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embed_cache.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/search/embed_cache.py tests/test_embed_cache.py
git commit -m "feat(clawmarks): add DINOv2 embedding cache"
```

---

### Task 9: Preference model training

**Files:**
- Modify: `pyproject.toml` (add scikit-learn dependency)
- Create: `src/clawmarks/search/preference_model.py`
- Create: `tests/test_preference_model.py`

**Interfaces:**
- Consumes: `embed_cache.load_cache`, `embed_cache.EMBEDDINGS_FILE` (Task 8).
- Produces: `preference_model.MIN_LABELS`, `preference_model.MODEL_FILE: Path`,
  `preference_model.build_training_set(tags, embeddings, ratings) -> tuple[np.ndarray, np.ndarray]`,
  `preference_model.cross_validate(X, y) -> float`, `preference_model.train(X, y) -> model`,
  `preference_model.predict_proba(model, embeddings) -> np.ndarray`. Consumed by Task 10
  (`build/preference_rank.py`) and Task 11 (Stage 5b in `driver.py`).

- [ ] **Step 1: Add the dependency**

```bash
uv add scikit-learn==1.6.1
```

Verify `pyproject.toml`'s `dependencies` list now includes `"scikit-learn==1.6.1"`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_preference_model.py
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_preference_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clawmarks.search.preference_model'`

- [ ] **Step 4: Write the implementation**

```python
# src/clawmarks/search/preference_model.py
"""
Trains a logistic-regression preference classifier on frozen DINOv2 embeddings
(search/embed_cache.py) and the user's yes/no ratings (user_ratings.json), so images can
eventually be ranked by predicted preference instead of raw novelty. See
docs/superpowers/specs/2026-07-09-preference-classifier-design.md, Component 3.

Refuses to train below MIN_LABELS: with only a handful of ratings, any model would be
overfitting noise, not learning taste. Run rate.html (via `clawmarks build rate` +
`clawmarks serve`) until this floor is cleared.

Run with: python -m clawmarks.search.preference_model
"""
import json
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score

from clawmarks.config import SWEEP_DIR
from clawmarks.search import embed_cache

MIN_LABELS = 50
MODEL_FILE = SWEEP_DIR / "preference_model.joblib"


def build_training_set(tags, embeddings, ratings):
    """`tags`/`embeddings` come from embed_cache.load_cache; `ratings` is the loaded
    user_ratings.json dict. Returns (X, y) using only tags present in both the embedding cache
    and the ratings file with a recognized label. Row order follows `tags`, not ratings-dict
    iteration order, so X stays aligned with `embeddings`."""
    tag_to_row = {t: i for i, t in enumerate(tags)}
    X_rows, y = [], []
    for tag, rating in ratings.items():
        if tag not in tag_to_row:
            continue
        label = rating.get("label")
        if label not in ("yes", "no"):
            continue
        X_rows.append(embeddings[tag_to_row[tag]])
        y.append(1 if label == "yes" else 0)
    if not X_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(X_rows), np.array(y, dtype=np.int64)


def cross_validate(X, y):
    """Mean cross-validated accuracy. Leave-one-out below MIN_LABELS, since every label matters
    at that scale; 5-fold StratifiedKFold at or above it."""
    cv = LeaveOneOut() if len(y) < MIN_LABELS else StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv)
    return float(scores.mean())


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

    acc = cross_validate(X, y)
    print(f"{len(y)} labels ({int(y.sum())} yes / {len(y) - int(y.sum())} no), "
          f"cross-validated accuracy: {acc:.3f}", flush=True)

    model = train(X, y)
    joblib.dump(model, MODEL_FILE)
    print(f"wrote {MODEL_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_preference_model.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/clawmarks/search/preference_model.py tests/test_preference_model.py
git commit -m "feat(clawmarks): add preference model training on frozen DINOv2 embeddings"
```

---

### Task 10: Predicted-preference ranking view (Component 4, validation gate)

**Files:**
- Create: `src/clawmarks/build/preference_rank.py`
- Modify: `src/clawmarks/shared_ui.py` (add nav entry)
- Modify: `src/clawmarks/cli.py` (add build target)
- Create: `tests/test_preference_rank.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `embed_cache.load_cache`, `embed_cache.EMBEDDINGS_FILE` (Task 8);
  `preference_model.MODEL_FILE`, `preference_model.predict_proba` (Task 9);
  `manifest_index.index_by_tag`, `manifest_index.item_summary` (Task 1).
- Produces: `preference_rank.build_ranked_items(by_tag, tags, scores, sweep_dir, limit=500) -> list[dict]`
  (pure, unit-tested), `preference_rank.main(argv=None)` writes `preference_rank.html`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preference_rank.py
from clawmarks.build.preference_rank import build_ranked_items


def _item(tag, tmp_path):
    return {"tag": tag, "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.5,
            "novelty": 0.5, "strength": 1.0, "cfg": 7.0, "file": str(tmp_path / f"{tag}.png")}


def test_build_ranked_items_sorts_descending_by_score(tmp_path):
    by_tag = {"a": _item("a", tmp_path), "b": _item("b", tmp_path)}
    items = build_ranked_items(by_tag, ["a", "b"], [0.2, 0.9], tmp_path)
    assert [it["tag"] for it in items] == ["b", "a"]
    assert items[0]["predicted_preference"] == 0.9


def test_build_ranked_items_respects_limit(tmp_path):
    by_tag = {f"t{i}": _item(f"t{i}", tmp_path) for i in range(10)}
    tags = list(by_tag.keys())
    scores = list(range(10))
    items = build_ranked_items(by_tag, tags, scores, tmp_path, limit=3)
    assert len(items) == 3


def test_build_ranked_items_skips_tags_missing_from_manifest(tmp_path):
    by_tag = {"a": _item("a", tmp_path)}
    items = build_ranked_items(by_tag, ["a", "ghost"], [0.5, 0.9], tmp_path)
    assert [it["tag"] for it in items] == ["a"]
```

```python
# addition to tests/test_cli.py
def test_build_preference_rank_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["build", "preference-rank"])
    assert args.command == "build"
    assert args.target == "preference-rank"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preference_rank.py tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError` and an `argparse` choice error.

- [ ] **Step 3: Write the implementation**

```python
# src/clawmarks/build/preference_rank.py
"""
Component 4 of the preference-classifier design: ranks every embedded image by the trained
model's predicted P(yes), highest first, so the model's judgment can be eyeballed against the
user's own taste before Stage 5b lets it steer anything live. Requires
search/preference_model.py to have already produced notes/uncanny_sweep/preference_model.joblib
(needs 50+ ratings — see search/preference_model.py's MIN_LABELS).

Run with: python3 -m clawmarks.build.preference_rank (or `clawmarks build preference-rank`)
"""
import json
import os

import joblib

from clawmarks.config import SWEEP_DIR
from clawmarks.search import embed_cache
from clawmarks.search.manifest_index import index_by_tag, item_summary
from clawmarks.search.preference_model import MODEL_FILE, predict_proba
from clawmarks.shared_ui import (
    INFOTIP_CSS, MOBILE_BASE_CSS, TOPNAV_CSS, info_btn, nav_bar_html, write_infotip_asset,
    write_lightbox_asset, write_scrollnav_asset,
)


def build_ranked_items(by_tag, tags, scores, sweep_dir, limit=500):
    ranked = sorted(
        ((t, s) for t, s in zip(tags, scores) if t in by_tag),
        key=lambda pair: -pair[1],
    )[:limit]
    items = []
    for tag, score in ranked:
        summary = item_summary(by_tag[tag], sweep_dir)
        summary["predicted_preference"] = round(float(score), 4)
        items.append(summary)
    return items


def main(argv=None):
    if not os.path.exists(MODEL_FILE):
        print(f"no trained model at {MODEL_FILE}; run `python -m "
              f"clawmarks.search.preference_model` first (needs 50+ ratings)", flush=True)
        return 1

    write_lightbox_asset(SWEEP_DIR)
    write_scrollnav_asset(SWEEP_DIR)
    write_infotip_asset(SWEEP_DIR)

    with open(f"{SWEEP_DIR}/scored_manifest.json") as f:
        manifest = json.load(f)
    by_tag = index_by_tag(manifest)

    tags, embeddings = embed_cache.load_cache(embed_cache.EMBEDDINGS_FILE)
    model = joblib.load(MODEL_FILE)
    scores = predict_proba(model, embeddings)
    items = build_ranked_items(by_tag, tags, scores, SWEEP_DIR)

    rank_tip = info_btn(
        "Sorted by the trained preference model's predicted probability that you'd rate this "
        "image 'yes,' highest first. This view exists to sanity-check the model before it's "
        "allowed to steer the live search: does the top of this list actually look like things "
        "you like?"
    )
    data_json = json.dumps(items)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS predicted preference</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee; --text-dim:#9a9aa4; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
h1 {{ font-size:18px; margin:0 0 4px; }}
p.sub {{ color:var(--text-dim); max-width:760px; font-size:13px; line-height:1.6; }}
#grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap:10px; margin-top:20px; }}
.cell {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
.cell img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; cursor:pointer; }}
.cell .meta {{ padding:6px 8px; font-size:11px; color:var(--text-dim); }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('preference_rank.html')}
<h1>Predicted preference{rank_tip}</h1>
<p class="sub">Top {len(items)} images by predicted P(yes), highest first.</p>
<div id="grid"></div>
<script>
const ITEMS = {data_json};
document.getElementById('grid').innerHTML = ITEMS.map(it => `
  <div class="cell">
    <img src="${{it.thumb}}" loading="lazy" data-tag="${{it.tag}}" onclick="Lightbox.open('${{it.tag}}')">
    <div class="meta">p=${{it.predicted_preference}} | f=${{it.faith}} n=${{it.novelty}}</div>
  </div>`).join('');
</script>
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    with open(f"{SWEEP_DIR}/preference_rank.html", "w") as f:
        f.write(html)
    print(f"wrote {SWEEP_DIR}/preference_rank.html ({len(items)} ranked images)", flush=True)
    return 0


if __name__ == "__main__":
    main()
```

Add to `shared_ui.py`'s `NAV_OPTIONS` (after `"archive.html"`):

```python
    ("archive.html", "elite archive"),
    ("preference_rank.html", "predicted preference"),
```

Wire into `cli.py`: add `preference_rank` to `_build_targets()`'s import and dict
(`"preference-rank": preference_rank.main`), and add `"preference-rank"` to the `build`
subcommand's `choices` list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preference_rank.py tests/test_cli.py -v`
Expected: PASS

Run the full suite: `pytest -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/build/preference_rank.py src/clawmarks/shared_ui.py src/clawmarks/cli.py tests/test_preference_rank.py tests/test_cli.py
git commit -m "feat(clawmarks): add predicted-preference ranking view (Component 4)"
```

---

### Task 11: Stage 5b — opt-in predicted-preference exploit pool

**Files:**
- Modify: `src/clawmarks/search/driver.py`
- Create: `tests/test_predicted_preference_pool.py`

**Interfaces:**
- Consumes: `embed_cache.sync`, `embed_cache.EMBEDDINGS_FILE` (Task 8); `preference_model.predict_proba`
  (Task 9).
- Produces: `driver._predicted_preference_pool(manifest, model_path, embed_model, top_n=15) -> list[dict]`,
  a new `--use-predicted-preference` CLI flag on `clawmarks run allnight`.

**This flag defaults to off and must stay off after this task ships.** Turning it on is a manual
decision for the project owner, made only after browsing `preference_rank.html` (Task 10) and
confirming the model's judgment actually matches their own taste, per the spec's Component 4
validation gate. Do not enable it, run it, or recommend enabling it as part of finishing this
plan.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predicted_preference_pool.py
from clawmarks.search import driver


class _FakeEmbedModel:
    def __call__(self, pixel_values):
        raise AssertionError("should not be called when no trained model exists")


def test_predicted_preference_pool_returns_empty_without_a_trained_model(tmp_path):
    manifest = [{"tag": "a", "file": str(tmp_path / "a.png")}]
    result = driver._predicted_preference_pool(manifest, tmp_path / "missing.joblib", _FakeEmbedModel())
    assert result == []


def test_predicted_preference_pool_returns_empty_for_empty_manifest(tmp_path):
    (tmp_path / "some_model.joblib").write_bytes(b"not a real model, never opened")
    result = driver._predicted_preference_pool([], tmp_path / "some_model.joblib", _FakeEmbedModel())
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_predicted_preference_pool.py -v`
Expected: FAIL with `AttributeError: module 'clawmarks.search.driver' has no attribute '_predicted_preference_pool'`

- [ ] **Step 3: Modify `driver.py`**

Add the function near `_load_yes_rated_images` (Task 7):

```python
def _predicted_preference_pool(manifest, model_path, embed_model, top_n=15):
    """Stage 5b (opt-in via --use-predicted-preference): ranks this round's own generated
    images by the trained preference model's P(yes) instead of yes-rating membership. Extends
    the shared embedding cache with any new images first, so an image is never re-embedded
    across generations. Returns [] (callers fall back to Stage 5a's yes-rated pool) if no model
    has been trained yet or the manifest is empty."""
    if not manifest or not os.path.exists(model_path):
        return []

    import joblib

    from clawmarks.search import embed_cache
    from clawmarks.search.preference_model import predict_proba

    by_tag = {m["tag"]: m for m in manifest}

    def image_path_for(tag):
        return by_tag[tag]["file"]

    tags, embeddings = embed_cache.sync(manifest, embed_cache.EMBEDDINGS_FILE, embed_model, image_path_for)
    model = joblib.load(model_path)
    scores = predict_proba(model, embeddings)
    ranked = sorted(
        ((by_tag[t], s) for t, s in zip(tags, scores) if t in by_tag),
        key=lambda pair: -pair[1],
    )
    return [m for m, _ in ranked[:top_n]]
```

Add the flag to `main()`'s argument parser:

```python
    parser.add_argument("--round", type=int, choices=list(ROUND_CONFIGS.keys()), required=True)
    parser.add_argument(
        "--use-predicted-preference", action="store_true", default=False,
        help="Stage 5b (opt-in, requires notes/uncanny_sweep/preference_model.joblib and "
             "human validation via preference_rank.html first): rank the exploit pool by the "
             "trained model's predicted preference instead of yes-rated images. Defaults off; "
             "do not enable without having browsed preference_rank.html first.",
    )
```

Update the generation-loop call site:

```python
        user_picks = _load_yes_rated_images() if cfg.seed_from_start else []
```

to:

```python
        user_picks = _load_yes_rated_images() if cfg.seed_from_start else []
        if args.use_predicted_preference:
            predicted_pool = _predicted_preference_pool(
                manifest, SWEEP_DIR / "preference_model.joblib", model,
            )
            if predicted_pool:
                user_picks = predicted_pool
            else:
                print("--use-predicted-preference set but no trained model found yet "
                      "(or nothing generated so far this round); using yes-rated images "
                      "instead", flush=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_predicted_preference_pool.py -v`
Expected: PASS (2 tests)

Run the full suite: `pytest -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/search/driver.py tests/test_predicted_preference_pool.py
git commit -m "feat(clawmarks): add opt-in Stage 5b predicted-preference exploit pool (flag off by default)"
```

---

### Task 12: Stage 5b — opt-in predicted-preference fallback in the elite archive

**Files:**
- Modify: `src/clawmarks/build/elite_archive.py`
- Modify: `src/clawmarks/cli.py`
- Create: `tests/test_elite_archive_predicted_preference.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `embed_cache.load_cache`, `embed_cache.EMBEDDINGS_FILE` (Task 8);
  `preference_model.MODEL_FILE`, `preference_model.predict_proba` (Task 9).
- Produces: `elite_archive.elite_sort_key(m, predicted_scores) -> float`,
  `elite_archive.build_item_summary(m, sweep_dir, predicted_scores) -> dict` (both pure,
  unit-tested), a `--use-predicted-preference` flag on `clawmarks build archive`.

Today the archive's per-cell fallback (when no cell member is yes-rated) is "highest novelty
wins" — a proxy, not a judgment of quality. This task makes the same swap Task 11 made for the
search driver: when a trained model exists and the flag is passed, the fallback becomes "highest
predicted-preference wins" instead. Diversity is preserved exactly as before (the faith x novelty
bins are untouched — MAP-Elites needs those to keep the search from collapsing onto a single
mode); only the *fitness* function used to pick the cell's champion changes. **This flag defaults
to off and must stay off after this task ships**, for the same reason as Task 11: the project
owner validates the model via `preference_rank.html` first.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_elite_archive_predicted_preference.py
from clawmarks.build.elite_archive import build_item_summary, elite_sort_key


def _item(tag, tmp_path, novelty=0.5):
    return {"tag": tag, "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.5,
            "novelty": novelty, "strength": 1.0, "cfg": 7.0, "file": str(tmp_path / f"{tag}.png")}


def test_elite_sort_key_falls_back_to_novelty_when_no_predicted_scores(tmp_path):
    m = _item("a", tmp_path, novelty=0.7)
    assert elite_sort_key(m, {}) == -0.7


def test_elite_sort_key_prefers_predicted_preference_when_available(tmp_path):
    m = _item("a", tmp_path, novelty=0.1)
    predicted_scores = {"a": 0.9}
    assert elite_sort_key(m, predicted_scores) == -0.9


def test_elite_sort_key_treats_missing_score_as_neutral_when_scores_exist_for_others(tmp_path):
    m = _item("a", tmp_path, novelty=0.1)
    predicted_scores = {"other_tag": 0.9}
    assert elite_sort_key(m, predicted_scores) == 0.0


def test_build_item_summary_omits_predicted_preference_when_absent(tmp_path):
    m = _item("a", tmp_path)
    summary = build_item_summary(m, tmp_path, {})
    assert "predicted_preference" not in summary


def test_build_item_summary_includes_predicted_preference_when_present(tmp_path):
    m = _item("a", tmp_path)
    summary = build_item_summary(m, tmp_path, {"a": 0.8234567})
    assert summary["predicted_preference"] == 0.8235


def test_sorting_a_cell_with_predicted_scores_puts_highest_score_first(tmp_path):
    items = [_item("a", tmp_path, novelty=0.9), _item("b", tmp_path, novelty=0.1)]
    predicted_scores = {"a": 0.2, "b": 0.95}
    ranked = sorted(items, key=lambda m: elite_sort_key(m, predicted_scores))
    assert [m["tag"] for m in ranked] == ["b", "a"]
```

```python
# addition to tests/test_cli.py
def test_build_archive_with_predicted_preference_flag_parses():
    parser = build_parser()
    args = parser.parse_args(["build", "archive", "--use-predicted-preference"])
    assert args.command == "build"
    assert args.target == "archive"
    assert args.use_predicted_preference is True


def test_build_archive_without_flag_defaults_false():
    parser = build_parser()
    args = parser.parse_args(["build", "archive"])
    assert args.use_predicted_preference is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_elite_archive_predicted_preference.py tests/test_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'elite_sort_key'` and an `argparse` error for
the unrecognized `--use-predicted-preference` flag.

- [ ] **Step 3: Modify `elite_archive.py`**

Add to the imports:

```python
import argparse

from clawmarks.search.preference_model import MODEL_FILE as PREFERENCE_MODEL_FILE
```

Add these two module-level functions (near `item_summary`'s import, before `main`):

```python
def elite_sort_key(m, predicted_scores):
    """Sort key for ranking a cell's candidates, most-preferred first (caller sorts ascending
    on this value). Falls back to novelty when no predicted-preference scores exist at all
    (Stage 5a behavior); once scores exist, a tag missing its own score (e.g. an image added to
    the manifest after the embedding cache was last synced) is treated as neutral (0.0) rather
    than assumed bad, so a sync gap doesn't quietly bury an otherwise-good image."""
    if predicted_scores:
        return -predicted_scores.get(m["tag"], 0.0)
    return -m["novelty"]


def build_item_summary(m, sweep_dir, predicted_scores):
    summary = item_summary(m, sweep_dir)
    if m["tag"] in predicted_scores:
        summary["predicted_preference"] = round(float(predicted_scores[m["tag"]]), 4)
    return summary
```

Change `main`'s signature and add argument parsing at the top of the function body:

```python
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-predicted-preference", action="store_true", default=False,
        help="Stage 5b (opt-in, requires notes/uncanny_sweep/preference_model.joblib and human "
             "validation via preference_rank.html first): rank each cell's fallback candidate "
             "(the one shown when no yes-rated image exists in that cell) by predicted "
             "preference instead of raw novelty. Defaults off.",
    )
    args = parser.parse_args(argv if argv is not None else [])

    write_lightbox_asset(SWEEP_DIR)
```

(This replaces the existing first line of `main`, `write_lightbox_asset(SWEEP_DIR)`, which now
follows the new parsing block instead of being the first statement.)

After the existing picks/ratings-loading block, add predicted-score loading:

```python
    predicted_scores = {}
    if args.use_predicted_preference and os.path.exists(PREFERENCE_MODEL_FILE):
        import joblib

        from clawmarks.search import embed_cache
        from clawmarks.search.preference_model import predict_proba

        tags, embeddings = embed_cache.load_cache(embed_cache.EMBEDDINGS_FILE)
        model = joblib.load(PREFERENCE_MODEL_FILE)
        scores = predict_proba(model, embeddings)
        predicted_scores = dict(zip(tags, scores))
```

Delete the now-redundant inline `item_summary` import usage points: update the cell-building
loop's call site from:

```python
                "items": [item_summary(m, SWEEP_DIR) for m in sorted(items, key=lambda m: -m["novelty"])],
```

to:

```python
                "items": [build_item_summary(m, SWEEP_DIR, predicted_scores)
                          for m in sorted(items, key=lambda m: elite_sort_key(m, predicted_scores))],
```

In the CSS block, add a third accent color and its cell/badge rules, alongside the existing
`--pick`/`--style`/`--conflict` variables:

```css
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee;
  --text-dim:#9a9aa4; --pick:#f5c542; --style:#5ec98a; --conflict:#e0a25e; --predicted:#7c9eff; }}
```

```css
.cell.human {{ box-shadow:0 0 0 2px var(--pick); }}
.cell.predicted {{ box-shadow:0 0 0 2px var(--predicted); }}
.badge {{ display:inline-block; padding:1px 6px; border-radius:4px; font-size:10px; margin-left:4px; }}
.badge.human {{ background:rgba(245,197,66,0.18); color:var(--pick); }}
.badge.predicted {{ background:rgba(124,158,255,0.18); color:var(--predicted); }}
.badge.auto {{ background:rgba(154,154,164,0.15); color:var(--text-dim); }}
```

In the JS, update `eliteFor` to recognize the new predicted-preference source (detectable via
the field `build_item_summary` only attaches when a score exists):

```javascript
function eliteFor(c) {{
  const pickedHere = c.items.filter(it => picks[it.tag]);
  if (pickedHere.length) return {{ item: pickedHere[0], source: 'yes-rated' }};
  if (c.items[0].predicted_preference !== undefined) return {{ item: c.items[0], source: 'predicted preference' }};
  return {{ item: c.items[0], source: 'highest novelty' }};  // items pre-sorted by elite_sort_key
}}
```

Update `render()` to branch on the third source value:

```javascript
function render() {{
  const grid = document.getElementById('grid');
  grid.innerHTML = CELLS.map((c, i) => {{
    const {{ item: elite, source }} = eliteFor(c);
    const human = source === 'yes-rated';
    const predicted = source === 'predicted preference';
    const badgeClass = human ? 'human' : (predicted ? 'predicted' : 'auto');
    const cellClass = human ? 'human' : (predicted ? 'predicted' : '');
    return `
    <div class="cell ${{cellClass}}">
      <img src="${{elite.thumb}}" loading="lazy" data-tag="${{elite.tag}}" onclick="Lightbox.open('${{elite.tag}}')">
      <div class="meta">
        <b>${{elite.prompt_name}}</b> <span class="badge ${{badgeClass}}">${{source}}</span><br>
        faith=${{elite.faith}} novelty=${{elite.novelty}}<br>
        n=${{c.n}} in cell | s=${{elite.strength}} cfg=${{elite.cfg}}
      </div>
      <button class="viewall" onclick="openModal(${{i}})">view all ${{c.n}} in this cell</button>
    </div>`;
  }}).join('');
}}
```

Update the page's descriptive copy (the `<p class="sub">` block) to mention the new fallback:

```
Gold-bordered cells are yes-rated winners; blue-bordered cells (only when this page is built
with --use-predicted-preference) are the trained model's top pick for that cell; others fall
back to the highest-novelty image the automated search found.
```

- [ ] **Step 4: Wire the flag through `cli.py`**

Add the flag to the `build` subparser, alongside its existing `target` argument:

```python
    build_p = sub.add_parser("build")
    build_p.add_argument(
        "target",
        choices=["all", "scan", "archive", "coverage", "map", "redundancy", "novelty-decay",
                 "lineage", "solution-map", "similarity", "thumbnails", "explore-hub", "seeds",
                 "probe-report", "uncanny-gallery", "rate", "preference-rank"],
    )
    build_p.add_argument(
        "--use-predicted-preference", action="store_true", default=False,
        help="Stage 5b, archive target only: rank each cell's fallback by predicted preference "
             "instead of novelty. Defaults off; requires a trained preference model.",
    )
```

Update `main()`'s `build` branch to pass the flag through only to the target actually invoked:

```python
    if args.command == "build":
        targets = _build_targets()
        extra_argv = ["--use-predicted-preference"] if args.use_predicted_preference else []
        if args.target == "all":
            for fn in targets.values():
                fn([])
        else:
            targets[args.target](extra_argv)
        return 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_elite_archive_predicted_preference.py tests/test_cli.py -v`
Expected: PASS (6 + 2 new tests)

Run the full suite: `pytest -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/build/elite_archive.py src/clawmarks/cli.py tests/test_elite_archive_predicted_preference.py tests/test_cli.py
git commit -m "feat(clawmarks): add opt-in Stage 5b predicted-preference fallback to the elite archive"
```

---

### Task 13: Run the migration, verify end to end

**Files:** none created; this task exercises Tasks 1-11 together against the real project data.

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: every test across all 12 tasks passes.

- [ ] **Step 2: Run the one-time picks-to-ratings migration**

```bash
python -m clawmarks.search.migrate_picks_to_ratings
```

Expected output: `migrated 40 picks into <path>/user_ratings.json as yes-ratings (0 already had a
rating and were left alone)` (the exact count may differ slightly if ratings were already
collected by the time this runs; that's fine, the script is idempotent).

- [ ] **Step 3: Confirm `user_ratings.json` now exists and looks right**

```bash
python3 -c "
import json
r = json.load(open('notes/uncanny_sweep/user_ratings.json'))
print(len(r), 'ratings;', sum(1 for v in r.values() if v['label']=='yes'), 'yes')
"
```

Expected: at least 40 ratings, all labeled `yes` (nothing has run `rate.html` yet at this point).

- [ ] **Step 4: Build the embedding cache**

```bash
python -m clawmarks.search.embed_cache
```

Expected: takes a while (3672 images through DINOv2 locally); ends with
`embedding cache now covers 3672 images at .../embeddings.npz`.

- [ ] **Step 5: Rebuild the archive and confirm it no longer reads `user_picks.json`**

```bash
python -m clawmarks.cli build archive
grep -c "user_picks" src/clawmarks/build/elite_archive.py
```

Expected: `archive.html` regenerates without error; the grep returns `0`.

- [ ] **Step 6: Build the rate page and smoke-test the server**

```bash
python -m clawmarks.cli build rate
python -m clawmarks.cli serve 8420 &
sleep 1
curl -s http://localhost:8420/api/rate/next | head -c 300
curl -s -X POST http://localhost:8420/api/rate -H 'Content-Type: application/json' \
  -d '{"tag": "some_real_tag_from_the_previous_curl_response", "label": "yes"}'
curl -s http://localhost:8420/api/ratings | head -c 300
kill %1
```

Expected: the first `curl` returns an item summary (or `{"done": true}` if somehow everything's
already reviewed); the `POST` returns `{"ok": true, "count": ...}`; the ratings fetch includes
the tag just rated. Confirm `curl -s http://localhost:8420/api/pick` (the old endpoint) now
returns `{"error": "unknown endpoint"}`.

- [ ] **Step 7: Do NOT run `preference_model.py`, `preference_rank.py`, or enable
`--use-predicted-preference` yet**

`preference_model.py` will correctly refuse to train (only ~40 yes-only labels exist right
after migration, no `no` labels at all, and below the 50-label floor). Training meaningfully
requires the project owner to actually use `rate.html` for a session or more first — that's
expected, not a bug to fix in this plan.

- [ ] **Step 8: Final commit (only if any of the above manual steps touched tracked files)**

```bash
git status
```

If `user_ratings.json`, `embeddings.npz`, or regenerated HTML files under `notes/uncanny_sweep/`
are untracked/gitignored data artifacts (check `.gitignore`), leave them as local state, not a
commit. If `pyproject.toml`/`uv.lock` have any stray uncommitted changes from Task 9, commit
those now. Otherwise this task produces no additional commit.
