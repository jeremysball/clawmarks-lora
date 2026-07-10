# PR #7 GLM Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every actionable finding from the GLM automated code review of PR #7 (the
live-tool-pages work): a correctness race in `LiveCache`, two thumbnail/manifest races, a
missed cache-invalidation dependency, six routes that bypass caching entirely, an inconsistent
dependency-wiring pattern, an O(n) lookup, and four pieces of dead code / stale documentation.

**Architecture:** No new subsystems. Every fix is a small, targeted change inside the existing
`LiveCache` / `curation_server.py` / `build/*.py` files already merged in PR #7. Tasks are
ordered correctness-first (races and cache bugs), then consistency (unwiring the ad hoc
dependency injection, wrapping the remaining routes), then cheap wins (the O(n) scan), then
pure cleanup (dead code, stale docstrings) last, since cleanup has zero behavioral risk and
should never block on anything else.

**Tech Stack:** Python 3.10+, stdlib `http.server`/`threading`, `pytest`, Pillow, no new
dependencies.

## Global Constraints

- Work happens in the existing worktree at `/workspace/trent-live-tool-pages` (branch tracking
  PR #7), not a fresh worktree: this plan patches the already-open PR, it does not start a new
  branch.
- Run tests with `PYTHONPATH=src uv run pytest tests/<file> -v` from
  `/workspace/trent-live-tool-pages`. Never bare `pytest` without `PYTHONPATH=src`, since the
  package lives under `src/`.
- No em dashes in any commit message, docstring, or comment touched by this plan (project-wide
  writing-style rule). Grep for the em dash character and for ` -- ` before committing text
  changes.
- Conventional Commits format for every commit: `<type>(<scope>): <description>`.
- Follow existing test conventions exactly: plain `def test_...():` functions (no test classes),
  `monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)` to redirect the server's data directory,
  `tmp_path` fixtures for on-disk state, real `HTTPServer` + background thread +
  `urllib.request.urlopen` for route-level tests (see `tests/test_curation_server_gallery_route.py`
  for the pattern).

---

### Task 1: Fix `LiveCache`'s dependency-read race (C1)

**Files:**
- Modify: `src/clawmarks/live_cache.py:27-49` (the `get` method)
- Test: `tests/test_live_cache.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `LiveCache.get(...)` keeps its exact existing signature and return value; only its
  internal read of a dependency's cached entry changes from two separate `self._entries[dep_name]`
  lookups to one.

**Context:** Today, `get()` does this when a target has `depends_on`:

```python
deps[dep_name] = self._entries[dep_name]["data"]
```

...and then, several lines later, separately:

```python
dep_mtimes = {name: self._entries[name]["mtimes"] for name in depends_on}
```

These are two independent reads of `self._entries[dep_name]`. Between them, another thread
holding `dep_name`'s own lock can finish recomputing `dep_name` and replace
`self._entries[dep_name]` with a brand-new dict (fresh `data` and fresh `mtimes`). If that
replacement happens in the gap between the two reads above, the target ends up caching an entry
built from the *old* `data` but tagged with the *new* `mtimes`. On every future request, the
target's `dep_mtimes` check will match (since it was captured after the dep's update), so the
target will look "fresh" forever, permanently serving stale data computed from a dependency
snapshot that no longer exists anywhere.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_live_cache.py`:

```python
def test_get_reads_dependency_data_and_mtimes_from_a_single_snapshot():
    """Regression test for a race where two separate self._entries[dep_name] reads (one for
    data, one for mtimes) could straddle a concurrent update to the dependency, pairing old
    data with new mtimes and caching a stale result that never self-invalidates. The fix reads
    each dependency's entry once and takes both fields from that single snapshot."""

    class _CountingDict(dict):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.access_count = 0

        def __getitem__(self, key):
            self.access_count += 1
            return super().__getitem__(key)

    cache = LiveCache()
    cache.get("dep", lambda sweep_dir: "dep-data-v1", watched_files=[])

    cache._entries = _CountingDict(cache._entries)
    result = cache.get(
        "target", lambda sweep_dir, deps: deps["dep"], watched_files=[], depends_on=["dep"],
    )

    assert result == "dep-data-v1"
    assert cache._entries.access_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_live_cache.py::test_get_reads_dependency_data_and_mtimes_from_a_single_snapshot -v`
Expected: FAIL, `assert 2 == 1` (the current code performs two separate lookups: one for
`["data"]`, one for `["mtimes"]`).

- [ ] **Step 3: Write minimal implementation**

Replace `src/clawmarks/live_cache.py:27-49`'s `get` method body with:

```python
    def get(self, target_name, compute_fn, watched_files, depends_on=(), sweep_dir=None):
        with self._lock_for(target_name):
            deps = None
            dep_mtimes = {}
            if depends_on:
                deps = {}
                for dep_name in depends_on:
                    if dep_name not in self._entries:
                        raise KeyError(
                            f"target {target_name!r} depends on {dep_name!r}, "
                            f"but {dep_name!r} has never been computed yet. "
                            f"Call cache.get({dep_name!r}, ...) before {target_name!r}."
                        )
                    dep_entry = self._entries[dep_name]
                    deps[dep_name] = dep_entry["data"]
                    dep_mtimes[dep_name] = dep_entry["mtimes"]

            mtimes = self._current_mtimes(watched_files)
            entry = self._entries.get(target_name)
            if entry is not None and entry["mtimes"] == mtimes and entry["dep_mtimes"] == dep_mtimes:
                return entry["data"]

            data = compute_fn(sweep_dir, deps) if depends_on else compute_fn(sweep_dir)
            self._entries[target_name] = {"data": data, "mtimes": mtimes, "dep_mtimes": dep_mtimes}
            return data
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_live_cache.py -v`
Expected: all tests in the file PASS, including the new one.

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-live-tool-pages
git add src/clawmarks/live_cache.py tests/test_live_cache.py
git commit -m "$(cat <<'EOF'
fix(live-cache): read a dependency's data and mtimes from one snapshot

Two separate self._entries[dep_name] lookups could straddle a concurrent
recompute of the dependency, pairing old data with new mtimes and caching a
stale result that never self-invalidates.
EOF
)"
```

---

### Task 2: Fix `solution-map`'s missing embeddings-file watch (C7)

**Files:**
- Modify: `src/clawmarks/curation_server.py:94-98` (`_get_solution_map_data`)
- Test: `tests/test_curation_server_solution_map_dep.py`

**Interfaces:**
- Consumes: `LiveCache.get(target_name, compute_fn, watched_files, depends_on=(), sweep_dir=None)`
  from Task 1 (signature unchanged).
- Produces: `_get_solution_map_data()` keeps its exact signature and return value.

**Context:** `solution_map.compute_data` (`src/clawmarks/build/solution_map.py:94-99`) skips
DINOv2 entirely and loads `{sweep_dir}/solution_map_final_embs.pt` when it exists and matches
the current manifest's paths. `_get_solution_map_data`'s `watched_files` only lists
`scored_manifest.json`, so a manual swap of `solution_map_final_embs.pt` (e.g. after running
`merge_round2.py`, which overwrites this file) never invalidates the cached `"solution-map"`
entry: the live server keeps serving whatever it computed before the swap.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_curation_server_solution_map_dep.py`:

```python
def test_get_solution_map_data_watches_the_final_embeddings_file_too(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")
    embs_file = tmp_path / "solution_map_final_embs.pt"
    embs_file.write_text("v1")

    calls = []
    monkeypatch.setattr(
        cs.solution_map, "compute_data",
        lambda sweep_dir: calls.append(1) or {"n": len(calls)},
    )

    first = cs._get_solution_map_data()
    assert first == {"n": 1}

    # Second call with nothing changed should hit the cache.
    second = cs._get_solution_map_data()
    assert second == {"n": 1}

    # Swap the embeddings file (simulating merge_round2.py overwriting it) without touching
    # scored_manifest.json. Without watching this file, the cache would never notice.
    import os, time
    new_mtime = os.path.getmtime(embs_file) + 5
    embs_file.write_text("v2")
    os.utime(embs_file, (new_mtime, new_mtime))

    third = cs._get_solution_map_data()
    assert third == {"n": 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_curation_server_solution_map_dep.py::test_get_solution_map_data_watches_the_final_embeddings_file_too -v`
Expected: FAIL, `assert {"n": 1} == {"n": 2}` (the cache never noticed the embeddings-file swap).

- [ ] **Step 3: Write minimal implementation**

Replace `src/clawmarks/curation_server.py:94-98` with:

```python
def _solution_map_watched_files():
    files = [_manifest_path()]
    embs_file = f"{SWEEP_DIR}/solution_map_final_embs.pt"
    if os.path.exists(embs_file):
        files.append(embs_file)
    return files


def _get_solution_map_data():
    return _live_cache.get(
        "solution-map", solution_map.compute_data,
        watched_files=_solution_map_watched_files(), sweep_dir=str(SWEEP_DIR),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_curation_server_solution_map_dep.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-live-tool-pages
git add src/clawmarks/curation_server.py tests/test_curation_server_solution_map_dep.py
git commit -m "$(cat <<'EOF'
fix(curation-server): watch solution_map_final_embs.pt for cache invalidation

compute_data() skips DINOv2 and loads this file directly when it exists, but
the live cache only watched scored_manifest.json, so a manual swap of the
embeddings file (e.g. after merge_round2.py) never invalidated the cached
solution-map entry.
EOF
)"
```

---

### Task 3: Wire `map`/`redundancy` through `LiveCache`'s `depends_on`, not manual injection (C6)

**Files:**
- Modify: `src/clawmarks/curation_server.py:285-305` (the `/map.html` and `/redundancy.html`
  routes)
- Test: `tests/test_curation_server_map_redundancy_cache.py` (new file)

**Interfaces:**
- Consumes: `_get_solution_map_data()` from Task 2 (signature unchanged); `LiveCache.get(...,
  depends_on=(), ...)` from Task 1.
- Produces: two new module-level helpers, `_get_map_data()` and `_get_redundancy_data()`, each
  taking no arguments and returning the same dict shape `map_view.compute_data` /
  `redundancy_view.compute_data` already return.

**Context:** `scan.html`'s route already uses the supported pattern for "target B depends on
target A": it calls `_live_cache.get("similarity", ...)` first, then calls
`_live_cache.get("scan", ..., depends_on=["similarity"])`. `/map.html` and `/redundancy.html`
instead call `map_view.compute_data(str(SWEEP_DIR), {"solution-map": _get_solution_map_data()})`
directly, bypassing `LiveCache` entirely for these two targets: they recompute on every single
request (folded into finding C5's fix scope, but worth calling out here since it's this task
that removes it) and never go through the `depends_on` mtime-tracking machinery that already
exists and is exercised by `scan.html`. This is the "two different code paths for the same
dependency pattern" GLM flagged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_curation_server_map_redundancy_cache.py`:

```python
from clawmarks import curation_server as cs


def test_get_map_data_is_cached_and_depends_on_solution_map(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")

    solution_map_calls = []
    map_calls = []
    monkeypatch.setattr(
        cs.solution_map, "compute_data",
        lambda sweep_dir: solution_map_calls.append(1) or {"points": []},
    )
    monkeypatch.setattr(
        cs.map_view, "compute_data",
        lambda sweep_dir, deps: map_calls.append(1) or {"from_solution_map": deps["solution-map"]},
    )

    first = cs._get_map_data()
    second = cs._get_map_data()

    assert first == {"from_solution_map": {"points": []}}
    assert second is first
    assert len(map_calls) == 1  # not recomputed on the second call
    assert len(solution_map_calls) == 1  # solution-map itself also only computed once


def test_get_redundancy_data_is_cached_and_depends_on_solution_map(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")

    monkeypatch.setattr(cs.solution_map, "compute_data", lambda sweep_dir: {"points": []})
    redundancy_calls = []
    monkeypatch.setattr(
        cs.redundancy_view, "compute_data",
        lambda sweep_dir, deps: redundancy_calls.append(1) or {"from_solution_map": deps["solution-map"]},
    )

    first = cs._get_redundancy_data()
    second = cs._get_redundancy_data()

    assert first == {"from_solution_map": {"points": []}}
    assert second is first
    assert len(redundancy_calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_curation_server_map_redundancy_cache.py -v`
Expected: FAIL with `AttributeError: module 'clawmarks.curation_server' has no attribute
'_get_map_data'` (the helper doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Add these two helpers right after `_get_solution_map_data` in `src/clawmarks/curation_server.py`
(after the `_solution_map_watched_files`/`_get_solution_map_data` pair from Task 2):

```python
def _get_map_data():
    _get_solution_map_data()
    return _live_cache.get(
        "map", map_view.compute_data,
        watched_files=[], depends_on=["solution-map"], sweep_dir=str(SWEEP_DIR),
    )


def _get_redundancy_data():
    _get_solution_map_data()
    return _live_cache.get(
        "redundancy", redundancy_view.compute_data,
        watched_files=[], depends_on=["solution-map"], sweep_dir=str(SWEEP_DIR),
    )
```

Then replace the `/map.html` route body (`src/clawmarks/curation_server.py:285-294`):

```python
        if self.path == "/map.html":
            html = map_view.render_html(_get_map_data())
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

And the `/redundancy.html` route body (`src/clawmarks/curation_server.py:296-305`):

```python
        if self.path == "/redundancy.html":
            html = redundancy_view.render_html(_get_redundancy_data())
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_curation_server_map_redundancy_cache.py tests/test_curation_server_solution_map_dep.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-live-tool-pages
git add src/clawmarks/curation_server.py tests/test_curation_server_map_redundancy_cache.py
git commit -m "$(cat <<'EOF'
fix(curation-server): wire map/redundancy through LiveCache's depends_on

These two routes built solution-map's dependency dict by hand and called
compute_data() directly instead of using the depends_on mechanism scan.html
already exercises, so they recomputed on every request and used a second,
inconsistent code path for the same "depends on another target" pattern.
EOF
)"
```

---

### Task 4: Wrap the six remaining bypass routes in `LiveCache` (C5)

**Files:**
- Modify: `src/clawmarks/curation_server.py` (routes for `/coverage.html`, `/novelty_decay.html`,
  `/lineage.html`, `/archive.html`, `/preference_rank.html`, `/gallery.html`)
- Test: `tests/test_curation_server_manifest_only_routes_cache.py` (new file)

**Interfaces:**
- Consumes: `LiveCache.get(...)` from Task 1.
- Produces: a new helper `_get_manifest_cached(target_name, compute_fn)` other future routes can
  reuse for any target whose only input is `scored_manifest.json`.

**Context:** `coverage_map`, `novelty_decay`, `lineage_view`, `elite_archive`,
`preference_rank`, and `uncanny_gallery` all call `compute_data()` (or, for `elite_archive`,
`compute_data(sweep_dir, use_predicted_preference=...)`) directly inline in their route
handlers, with zero caching: every single page load recomputes from scratch, even though each of
these targets only reads `scored_manifest.json`. `elite_archive` additionally takes a query-param
flag that changes its output, so it needs two distinct cache entries (one per flag value), not
one shared entry.

- [ ] **Step 1: Write the failing test**

Create `tests/test_curation_server_manifest_only_routes_cache.py`:

```python
from clawmarks import curation_server as cs


def test_get_manifest_cached_reuses_cache_across_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")

    calls = []

    def compute(sweep_dir):
        calls.append(1)
        return {"n": len(calls)}

    first = cs._get_manifest_cached("coverage", compute)
    second = cs._get_manifest_cached("coverage", compute)

    assert first == {"n": 1}
    assert second is first
    assert len(calls) == 1


def test_get_manifest_cached_keeps_targets_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")

    a = cs._get_manifest_cached("novelty_decay", lambda sweep_dir: {"which": "novelty_decay"})
    b = cs._get_manifest_cached("lineage", lambda sweep_dir: {"which": "lineage"})

    assert a == {"which": "novelty_decay"}
    assert b == {"which": "lineage"}


def test_archive_route_caches_actual_and_predicted_preference_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")

    calls = []

    def fake_compute_data(sweep_dir, use_predicted_preference=False):
        calls.append(use_predicted_preference)
        return {"use_predicted_preference": use_predicted_preference}

    monkeypatch.setattr(cs.elite_archive, "compute_data", fake_compute_data)

    actual = cs._get_manifest_cached(
        "archive_actual", lambda sd: cs.elite_archive.compute_data(sd, use_predicted_preference=False),
    )
    predicted = cs._get_manifest_cached(
        "archive_predicted", lambda sd: cs.elite_archive.compute_data(sd, use_predicted_preference=True),
    )
    actual_again = cs._get_manifest_cached(
        "archive_actual", lambda sd: cs.elite_archive.compute_data(sd, use_predicted_preference=False),
    )

    assert actual == {"use_predicted_preference": False}
    assert predicted == {"use_predicted_preference": True}
    assert actual_again is actual
    assert calls == [False, True]  # only two real computes, not three
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_curation_server_manifest_only_routes_cache.py -v`
Expected: FAIL with `AttributeError: module 'clawmarks.curation_server' has no attribute
'_get_manifest_cached'`.

- [ ] **Step 3: Write minimal implementation**

Add this helper to `src/clawmarks/curation_server.py`, next to `_get_solution_map_data` /
`_get_map_data` / `_get_redundancy_data`:

```python
def _get_manifest_cached(target_name, compute_fn):
    return _live_cache.get(
        target_name, compute_fn,
        watched_files=[_manifest_path()], sweep_dir=str(SWEEP_DIR),
    )
```

Then replace each of the six route bodies. `/coverage.html`
(`src/clawmarks/curation_server.py:307-315`):

```python
        if self.path == "/coverage.html":
            html = coverage_map.render_html(_get_manifest_cached("coverage", coverage_map.compute_data))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

`/novelty_decay.html` (`src/clawmarks/curation_server.py:317-325`):

```python
        if self.path == "/novelty_decay.html":
            html = novelty_decay.render_html(_get_manifest_cached("novelty_decay", novelty_decay.compute_data))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

`/lineage.html` (`src/clawmarks/curation_server.py:327-335`):

```python
        if self.path == "/lineage.html":
            html = lineage_view.render_html(_get_manifest_cached("lineage", lineage_view.compute_data))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

`/archive.html` (`src/clawmarks/curation_server.py:337-349`):

```python
        if self.path.startswith("/archive.html"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            use_predicted = query.get("use_predicted_preference", ["0"])[0] == "1"
            target_name = "archive_predicted" if use_predicted else "archive_actual"
            data = _get_manifest_cached(
                target_name,
                lambda sd: elite_archive.compute_data(sd, use_predicted_preference=use_predicted),
            )
            html = elite_archive.render_html(data)
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

`/preference_rank.html` (`src/clawmarks/curation_server.py:351-359`):

```python
        if self.path == "/preference_rank.html":
            html = preference_rank.render_html(_get_manifest_cached("preference_rank", preference_rank.compute_data))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

`/gallery.html` (`src/clawmarks/curation_server.py:361-369`):

```python
        if self.path == "/gallery.html":
            html = uncanny_gallery.render_html(_get_manifest_cached("gallery", uncanny_gallery.compute_data))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_curation_server_manifest_only_routes_cache.py tests/test_curation_server_gallery_route.py -v`
Expected: all tests PASS. (`test_curation_server_gallery_route.py`'s existing
`test_gallery_html_served_live` still exercises the real route end-to-end and must keep passing
unmodified, confirming the caching change is behavior-preserving.)

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-live-tool-pages
git add src/clawmarks/curation_server.py tests/test_curation_server_manifest_only_routes_cache.py
git commit -m "$(cat <<'EOF'
fix(curation-server): cache coverage/novelty_decay/lineage/archive/preference_rank/gallery

These six routes called their module's compute_data() directly on every
request with no caching at all, unlike scan/solution-map which already went
through LiveCache. archive.html additionally caches its "actual" and
"predicted" preference variants under separate target names, since the
use_predicted_preference query flag changes its output.
EOF
)"
```

---

### Task 5: Fix thumbnail generation's non-atomic write (C2)

**Files:**
- Modify: `src/clawmarks/build/thumbnails.py`
- Test: `tests/test_thumbnails.py` (new file)

**Interfaces:**
- Consumes: nothing new.
- Produces: `generate_thumbnail(src_path, dst_path)` keeps its exact signature; internally it now
  writes to a per-call temp file and `os.replace`s it into place, matching the
  `curation_server.save_store` pattern already used for ratings/favorites/counterfactuals/seeds.

**Context:** `generate_thumbnail` calls `img.save(dst_path, ...)` directly. If two concurrent
requests race to generate the same missing thumbnail (both pass `curation_server.py`'s
`not os.path.exists(thumb_path)` check before either finishes), or if the process is
interrupted mid-write, a reader can see a partially-written JPEG at `dst_path`. Per this
module's own docstring, "once made, a thumbnail never goes stale" (there's no mtime check on
thumbnails), so a corrupt thumbnail written this way is corrupt forever.

- [ ] **Step 1: Write the failing test**

Create `tests/test_thumbnails.py`:

```python
import pytest
from PIL import Image

from clawmarks.build.thumbnails import generate_thumbnail


def test_generate_thumbnail_produces_a_valid_small_jpeg(tmp_path):
    src = tmp_path / "src.png"
    Image.new("RGB", (500, 500), color="red").save(src)
    dst = tmp_path / "thumb.jpg"

    generate_thumbnail(str(src), str(dst))

    img = Image.open(dst)
    assert img.format == "JPEG"
    assert max(img.size) <= 220


def test_generate_thumbnail_never_leaves_a_corrupt_file_at_dst_on_write_failure(tmp_path, monkeypatch):
    """Regression test: the old implementation wrote directly to dst_path via img.save(dst_path,
    ...), so a write failure partway through (disk full, process killed) could leave a
    truncated/corrupt JPEG at dst_path. Since thumbnails are never re-validated once dst_path
    exists, that corruption would be permanent. Writing to a temp file first and only
    os.replace-ing it into place on success means dst_path is never touched unless the write
    fully succeeded."""
    src = tmp_path / "src.png"
    Image.new("RGB", (256, 256), color="blue").save(src)
    dst = tmp_path / "thumb.jpg"

    def failing_save(self, fp, *a, **k):
        f = open(fp, "wb") if isinstance(fp, str) else fp
        f.write(b"partial-jpeg-bytes-then-crash")
        if isinstance(fp, str):
            f.close()
        raise IOError("simulated disk-full mid-write")

    monkeypatch.setattr(Image.Image, "save", failing_save)

    with pytest.raises(IOError):
        generate_thumbnail(str(src), str(dst))

    assert not dst.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_thumbnails.py -v`
Expected: `test_generate_thumbnail_produces_a_valid_small_jpeg` PASSES already (no behavior
change needed for the happy path); `test_generate_thumbnail_never_leaves_a_corrupt_file_at_dst_on_write_failure`
FAILS with `assert not True` (the current code writes the partial bytes straight to `dst_path`).

- [ ] **Step 3: Write minimal implementation**

Replace `src/clawmarks/build/thumbnails.py:13-16` (the whole `generate_thumbnail` function) and
add the needed imports:

```python
import os
import threading

from PIL import Image

THUMB_SIZE = 220
QUALITY = 78


def generate_thumbnail(src_path, dst_path):
    img = Image.open(src_path).convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.BICUBIC)
    tmp_path = f"{dst_path}.tmp-{os.getpid()}-{threading.get_ident()}"
    img.save(tmp_path, format="JPEG", quality=QUALITY)
    os.replace(tmp_path, dst_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_thumbnails.py tests/test_curation_server_lazy_thumbnails.py -v`
Expected: all tests PASS. (`test_curation_server_lazy_thumbnails.py`'s existing
`test_thumb_generated_on_first_request` exercises the route end-to-end and must keep passing.)

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-live-tool-pages
git add src/clawmarks/build/thumbnails.py tests/test_thumbnails.py
git commit -m "$(cat <<'EOF'
fix(thumbnails): write via temp file + rename instead of straight to dst_path

Thumbnails are never re-validated once dst_path exists, so a write failure or
a race between two concurrent generate_thumbnail calls for the same tag could
leave a permanently corrupt JPEG on disk. Matches the temp+rename pattern
curation_server.save_store already uses for its JSON stores.
EOF
)"
```

---

### Task 6: Fix `load_manifest`'s reload race (C3)

**Files:**
- Modify: `src/clawmarks/curation_server.py:196-206`
- Test: `tests/test_curation_server_manifest_cache.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `load_manifest()` keeps its exact signature and return value.

**Context:** `load_manifest`'s staleness check ("is `_manifest_cache["manifest"]` `None`, or does
the cached mtime differ from the file's current mtime") and its subsequent read-and-reassignment
of `_manifest_cache` are not synchronized. Two threads that both observe staleness can both
open, parse, and reassign `_manifest_cache` concurrently. If a thread that read an *older* file
state finishes reassigning after a thread that read a *newer* state, the cache is left holding
stale `manifest` data paired with a stale `mtime` (self-correcting on the next call, since that
mtime will itself look stale again, but real reads and parses are duplicated under concurrent
load in the meantime, and a request in that window can see incorrect data). This route is also
now hit more often than before this PR, since the newly added lazy-thumbnail route
(`curation_server.py:398-409`) calls `load_manifest()` on every cold thumbnail request.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_curation_server_manifest_cache.py`:

```python
import threading


def test_load_manifest_parses_only_once_under_concurrent_access(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_manifest_cache", {"manifest": None, "mtime": None})

    path = tmp_path / "scored_manifest.json"
    path.write_text(json.dumps([{"tag": "a"}]))

    parse_calls = []
    real_json_load = json.load

    def counting_load(f):
        parse_calls.append(1)
        time.sleep(0.02)
        return real_json_load(f)

    monkeypatch.setattr(cs.json, "load", counting_load)

    threads = [threading.Thread(target=cs.load_manifest) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)

    assert len(parse_calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_curation_server_manifest_cache.py::test_load_manifest_parses_only_once_under_concurrent_access -v`
Expected: FAIL, `assert 5 == 1` (all five threads race past the unsynchronized staleness check
and each parses the file).

- [ ] **Step 3: Write minimal implementation**

Replace `src/clawmarks/curation_server.py:196-206`:

```python
_manifest_cache = {"manifest": None, "mtime": None}
_manifest_cache_lock = threading.Lock()


def load_manifest():
    path = f"{SWEEP_DIR}/scored_manifest.json"
    with _manifest_cache_lock:
        mtime = os.path.getmtime(path)
        if _manifest_cache["manifest"] is None or _manifest_cache["mtime"] != mtime:
            with open(path) as f:
                _manifest_cache["manifest"] = json.load(f)
            _manifest_cache["mtime"] = mtime
        return _manifest_cache["manifest"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_curation_server_manifest_cache.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-live-tool-pages
git add src/clawmarks/curation_server.py tests/test_curation_server_manifest_cache.py
git commit -m "$(cat <<'EOF'
fix(curation-server): serialize load_manifest's staleness check and reload

Two threads could both observe a stale cache and reload concurrently; if the
one that read an older file state finished reassigning last, the cache held
stale data (self-correcting next call, but duplicating real reads and briefly
serving wrong data under concurrent load in the meantime).
EOF
)"
```

---

### Task 7: Replace the O(n) manifest scan for thumbnail lookups with an index (C8)

**Files:**
- Modify: `src/clawmarks/curation_server.py:196-206` (extend the manifest cache from Task 6),
  `src/clawmarks/curation_server.py:398-409` (the thumbnail route)
- Test: `tests/test_curation_server_manifest_cache.py`

**Interfaces:**
- Consumes: `load_manifest()` from Task 6.
- Produces: new function `manifest_entry_by_tag(tag)` returning the manifest dict for `tag`, or
  `None` if absent.

**Context:** The cold-thumbnail path does `next((m for m in manifest if m["tag"] == tag), None)`,
an O(n) linear scan over the whole manifest for every single cache-miss thumbnail request. With
thousands of images, and every image needing exactly one such lookup the first time its
thumbnail is requested, this adds up. A dict keyed by tag, built once alongside the manifest and
invalidated together, turns each lookup into O(1).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_curation_server_manifest_cache.py`:

```python
def test_manifest_entry_by_tag_finds_existing_and_missing_tags(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_manifest_cache", {"manifest": None, "mtime": None, "by_tag": None})

    manifest = [{"tag": f"t{i}", "file": f"f{i}.png"} for i in range(50)]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))

    assert cs.manifest_entry_by_tag("t25") == {"tag": "t25", "file": "f25.png"}
    assert cs.manifest_entry_by_tag("missing") is None


def test_manifest_entry_by_tag_index_rebuilds_on_manifest_change(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_manifest_cache", {"manifest": None, "mtime": None, "by_tag": None})

    path = tmp_path / "scored_manifest.json"
    path.write_text(json.dumps([{"tag": "a", "file": "a.png"}]))
    assert cs.manifest_entry_by_tag("b") is None

    new_mtime = os.path.getmtime(path) + 5
    path.write_text(json.dumps([{"tag": "a", "file": "a.png"}, {"tag": "b", "file": "b.png"}]))
    os.utime(path, (new_mtime, new_mtime))

    assert cs.manifest_entry_by_tag("b") == {"tag": "b", "file": "b.png"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_curation_server_manifest_cache.py::test_manifest_entry_by_tag_finds_existing_and_missing_tags -v`
Expected: FAIL with `AttributeError: module 'clawmarks.curation_server' has no attribute
'manifest_entry_by_tag'`.

- [ ] **Step 3: Write minimal implementation**

Replace the `_manifest_cache`/`load_manifest` block from Task 6
(`src/clawmarks/curation_server.py:196-206` in the pre-Task-6 file) with:

```python
_manifest_cache = {"manifest": None, "mtime": None, "by_tag": None}
_manifest_cache_lock = threading.Lock()


def load_manifest():
    path = f"{SWEEP_DIR}/scored_manifest.json"
    with _manifest_cache_lock:
        mtime = os.path.getmtime(path)
        if _manifest_cache["manifest"] is None or _manifest_cache["mtime"] != mtime:
            with open(path) as f:
                manifest = json.load(f)
            _manifest_cache["manifest"] = manifest
            _manifest_cache["by_tag"] = {m["tag"]: m for m in manifest}
            _manifest_cache["mtime"] = mtime
        return _manifest_cache["manifest"]


def manifest_entry_by_tag(tag):
    load_manifest()
    return _manifest_cache["by_tag"].get(tag)
```

Then update the thumbnail route (`src/clawmarks/curation_server.py:398-409`):

```python
        if self.path.startswith("/thumbs/") and self.path.endswith(".jpg"):
            thumb_path = f"{SWEEP_DIR}{self.path}"
            if not os.path.exists(thumb_path):
                tag = os.path.basename(self.path)[: -len(".jpg")]
                match = manifest_entry_by_tag(tag)
                if match is None:
                    self.send_error(404, "no manifest entry for this tag")
                    return
                os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                generate_thumbnail(match["file"], thumb_path)
            # fall through to super().do_GET() below, which now finds the file on disk
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_curation_server_manifest_cache.py tests/test_curation_server_lazy_thumbnails.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-live-tool-pages
git add src/clawmarks/curation_server.py tests/test_curation_server_manifest_cache.py
git commit -m "$(cat <<'EOF'
perf(curation-server): index the manifest by tag instead of scanning it

The cold-thumbnail route did a linear O(n) scan over the whole manifest for
every cache-miss request. A tag-keyed dict, built once alongside the
manifest and invalidated together, makes each lookup O(1).
EOF
)"
```

---

### Task 8: Remove the dead `write_*_asset` stubs (C9)

**Files:**
- Modify: `src/clawmarks/shared_ui.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (pure deletion; no other file references these names, confirmed via
  `rg -n "write_infotip_asset|write_lightbox_asset|write_scrollnav_asset"` returning only the
  three definitions themselves).

**Context:** `write_infotip_asset`, `write_lightbox_asset`, and `write_scrollnav_asset` are all
no-op (`pass`) functions left over from when these assets were written to disk by the old
`clawmarks build` step; PR #7 made `curation_server.py` serve them directly instead, and nothing
calls these three functions anymore. The module's own top-of-file docstring still references
`write_lightbox_asset` as something to import and use.

- [ ] **Step 1: Delete the three dead functions**

Remove `src/clawmarks/shared_ui.py:131-132`:

```python
def write_infotip_asset(sweep_dir):
    pass  # served directly by curation_server.py's /infotip.js route; no on-disk copy anymore
```

Remove `src/clawmarks/shared_ui.py:581-582`:

```python
def write_lightbox_asset(sweep_dir):
    pass  # served directly by curation_server.py's /lightbox.js route; no on-disk copy anymore
```

Remove `src/clawmarks/shared_ui.py:585-587`:

```python
def write_scrollnav_asset(sweep_dir):
    pass  # served directly by curation_server.py's /scrollnav.js route
```

- [ ] **Step 2: Fix the stale module docstring**

Replace `src/clawmarks/shared_ui.py:1-13`:

```python
"""
Shared UI pieces used by every notes/build_*.py tool-page generator, so the lightbox, the
top navigation bar, and its scroll-to-hide behavior are defined once instead of duplicated
across 8 scripts. Import and use:

    from shared_ui import write_lightbox_asset, nav_bar_html, TOPNAV_CSS, SCROLLNAV_JS

`write_lightbox_asset(sweep_dir)` copies the static lightbox.js module into the sweep
directory (idempotent, safe to call from every builder). Every generated page includes it
with `<script src="lightbox.js"></script>` and opens images via `Lightbox.open(tag)` instead
of `window.open('scan.html?open=...')`: no new tab, no page load, works from any page
because the module fetches notes/uncanny_sweep/scan_data.json itself.
"""
```

with:

```python
"""
Shared UI pieces used by every build/*.py tool-page generator, so the lightbox, the top
navigation bar, and its scroll-to-hide behavior are defined once instead of duplicated across
every page. Import and use:

    from clawmarks.shared_ui import nav_bar_html, TOPNAV_CSS, SCROLLNAV_JS, _LIGHTBOX_JS

`curation_server.py` serves `_LIGHTBOX_JS`, `SCROLLNAV_JS`, and `INFOTIP_JS` directly from
`/lightbox.js`, `/scrollnav.js`, and `/infotip.js` routes; every generated page includes them
with `<script src="lightbox.js"></script>` and opens images via `Lightbox.open(tag)` instead of
`window.open('scan.html?open=...')`: no new tab, no page load, works from any page because the
module fetches scan_data.json itself.
"""
```

- [ ] **Step 3: Run the full test suite to confirm nothing referenced the deleted functions**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/ -v`
Expected: all tests PASS (no test imports or calls any of the three deleted functions).

- [ ] **Step 4: Commit**

```bash
cd /workspace/trent-live-tool-pages
git add src/clawmarks/shared_ui.py
git commit -m "$(cat <<'EOF'
chore(shared-ui): remove dead write_*_asset stubs and their stale docstring

These became no-ops when curation_server.py started serving lightbox.js,
scrollnav.js, and infotip.js directly instead of writing them to disk. Zero
callers remained; the module docstring still described the old behavior.
EOF
)"
```

---

### Task 9: Remove the dead `MODEL_ID` import (C10)

**Files:**
- Modify: `src/clawmarks/build/uncanny_gallery.py:15`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (pure deletion).

**Context:** `from clawmarks.search.score_manifest import MODEL_ID` is unused. The module's own
`thumb_data_uri` function loads images with `PIL.Image` directly and never touches `MODEL_ID`
(DINOv2 model identifier); this file doesn't run DINOv2 at all.

- [ ] **Step 1: Remove the import**

Delete `src/clawmarks/build/uncanny_gallery.py:15`:

```python
from clawmarks.search.score_manifest import MODEL_ID
```

- [ ] **Step 2: Run the module's tests to confirm nothing needed it**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/test_uncanny_gallery.py tests/test_curation_server_gallery_route.py -v`
Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /workspace/trent-live-tool-pages
git add src/clawmarks/build/uncanny_gallery.py
git commit -m "chore(uncanny-gallery): remove unused MODEL_ID import"
```

---

### Task 10: Fix stale docstrings describing on-disk build outputs that no longer exist (C11)

**Files:**
- Modify: `src/clawmarks/build/map_view.py:1-14`
- Modify: `src/clawmarks/build/redundancy_view.py:1-17`
- Modify: `src/clawmarks/build/solution_map.py:1-24`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (docstring-only changes; no code behavior changes).

**Context:** All three docstrings predate PR #7 and still describe a `clawmarks build` era that
no longer exists: they tell the reader to run the module standalone after some `.json` file has
been built by an earlier step, and (for `solution_map.py`) describe `solution_map_data.json` /
`similarity_scored.json` as this module's own on-disk outputs. Since PR #7, `compute_data()` in
each of these modules returns its result in memory for `curation_server.py` to cache and render
live; it writes nothing to disk except `solution_map.py`'s own internal
`solution_map_final_embs.pt` cache file (an implementation detail, not a "build output" other
tools consume). `solution_map.py:22-23` already documents the *current*, correct behavior; the
fix aligns the rest of each docstring with that.

- [ ] **Step 1: Fix `map_view.py`'s docstring**

Replace `src/clawmarks/build/map_view.py:1-14`:

```python
"""
Ideas 1, 2, and 6 from Fable's exploration-tooling brainstorm (2026-07-09), built into a single
page: an interactive UMAP scatter of the full embedding space (real training images as gold
stars, every generated image as a dot), a generation slider/play control that ghosts earlier
generations to show whether the search is finding new territory or re-treading old ground, and
a "nearest real image" bar chart (mode-collapse check: if the population only ever anchors to a
handful of the 31 real training images, faithfulness is being measured against a sliver of the
style, not the whole thing).

Depends on build_solution_map.py's output (solution_map_data.json), which does the actual
DINOv2 re-embedding and UMAP fit; this script only lays out the already-computed points.

Run after solution_map_data.json exists: python3 -m clawmarks.build.map_view
"""
```

with:

```python
"""
Ideas 1, 2, and 6 from Fable's exploration-tooling brainstorm (2026-07-09), built into a single
page: an interactive UMAP scatter of the full embedding space (real training images as gold
stars, every generated image as a dot), a generation slider/play control that ghosts earlier
generations to show whether the search is finding new territory or re-treading old ground, and
a "nearest real image" bar chart (mode-collapse check: if the population only ever anchors to a
handful of the 31 real training images, faithfulness is being measured against a sliver of the
style, not the whole thing).

Depends on solution_map.py's compute_data(), which does the actual DINOv2 re-embedding and UMAP
fit; this module only lays out the already-computed points. compute_data(sweep_dir, deps) takes
solution_map's result via `deps["solution-map"]`, served live by curation_server.py through
LiveCache's depends_on=["solution-map"] mechanism, not a standalone build step.
"""
```

- [ ] **Step 2: Fix `redundancy_view.py`'s docstring**

Replace `src/clawmarks/build/redundancy_view.py:1-17`:

```python
"""
Idea 4 from Fable's exploration-tooling brainstorm (2026-07-09): a redundancy/duplicate-cluster
view. With 3000+ generated images, a meaningful fraction are likely near-copies of each other
(same subject/settings, different seed noise), which inflates how much of the map actually
looks "covered." This clusters images by DINOv2 cosine similarity at an adjustable threshold
(connected components over the precomputed top-16 nearest-neighbor edges) so you can see the
population's true effective size and which "different" bins are actually duplicates.

Clustering happens client-side in JS (union-find over ~3400 nodes / ~54k edges is instant), so
one threshold slider can be dragged live instead of needing a rebuild per threshold.

Depends on build_solution_map.py's similarity_scored.json (top-16 neighbors WITH cosine
scores; the original build_similarity_index.py only stores neighbor identity, not the score,
which isn't enough to threshold on).

Run after similarity_scored.json exists: python3 -m clawmarks.build.redundancy_view
"""
```

with:

```python
"""
Idea 4 from Fable's exploration-tooling brainstorm (2026-07-09): a redundancy/duplicate-cluster
view. With 3000+ generated images, a meaningful fraction are likely near-copies of each other
(same subject/settings, different seed noise), which inflates how much of the map actually
looks "covered." This clusters images by DINOv2 cosine similarity at an adjustable threshold
(connected components over the precomputed top-16 nearest-neighbor edges) so you can see the
population's true effective size and which "different" bins are actually duplicates.

Clustering happens client-side in JS (union-find over ~3400 nodes / ~54k edges is instant), so
one threshold slider can be dragged live instead of needing a rebuild per threshold.

Depends on solution_map.py's compute_data(), which includes top-16 neighbors WITH cosine scores
(the separate similarity_index.py only stores neighbor identity, not the score, which isn't
enough to threshold on). compute_data(sweep_dir, deps) takes solution_map's result via
`deps["solution-map"]`, served live by curation_server.py through LiveCache's
depends_on=["solution-map"] mechanism, not a standalone build step.
"""
```

- [ ] **Step 3: Fix `solution_map.py`'s docstring**

Replace `src/clawmarks/build/solution_map.py:1-24`:

```python
"""
Re-embeds every image in scored_manifest.json (plus the real training images) with DINOv2,
then builds the data three of the new exploration tools need:

  1/2. solution_map_data.json  - a 2D UMAP projection of the full embedding space (real images
       + every generated image), with generation number attached, for map.html's scatter plot
       and generation slider. Answers "what does the search space actually look like" and
       "is round N finding new territory or re-treading round N-1's," which the faithfulness x
       novelty plane (two derived scalars) can't show on its own.
  4.   similarity_scored.json  - same top-K nearest-neighbor lists as build_similarity_index.py,
       but with the actual cosine similarity values attached (not just neighbor identity), so
       redundancy.html can cluster near-duplicates at an adjustable threshold.
  6.   nearest_real_idx per image - which of the ~31 real training images each generation is
       closest to, folded into solution_map_data.json so map.html and real_anchor.html can both
       use it (mode-collapse check: if generations only ever anchor to a handful of the real
       images, the search is faithful to a sliver of the style, not the whole thing).

This duplicates build_similarity_index.py's embedding pass rather than importing its output,
because that script discards the raw embeddings once it's done with them (only top-16 neighbor
*tags* survive to disk) and UMAP/nearest-real-image both need the actual vectors.

compute_data() is a data-only live-cache target with no route of its own; map.html and
redundancy.html both depend on it (DEPENDS_ON = ["solution-map"]).
"""
```

with:

```python
"""
Re-embeds every image in scored_manifest.json (plus the real training images) with DINOv2,
then returns the data three of the exploration tools need:

  1/2. a 2D UMAP projection of the full embedding space (real images + every generated image),
       with generation number attached, for map.html's scatter plot and generation slider.
       Answers "what does the search space actually look like" and "is round N finding new
       territory or re-treading round N-1's," which the faithfulness x novelty plane (two
       derived scalars) can't show on its own.
  4.   top-K nearest-neighbor lists WITH the actual cosine similarity values attached (not just
       neighbor identity, unlike similarity_index.py), so redundancy.html can cluster
       near-duplicates at an adjustable threshold.
  6.   nearest_real_idx per image - which of the ~31 real training images each generation is
       closest to (mode-collapse check: if generations only ever anchor to a handful of the
       real images, the search is faithful to a sliver of the style, not the whole thing).

This duplicates similarity_index.py's embedding pass rather than importing its output, because
that module discards the raw embeddings once it's done with them (only top-16 neighbor *tags*
survive) and UMAP/nearest-real-image both need the actual vectors. The finished embeddings are
cached on disk at solution_map_final_embs.pt purely as an internal speed-up (skips DINOv2
entirely on a cache hit); this is an implementation detail, not an output other tools consume.

compute_data() is a data-only live-cache target with no route of its own; map.html and
redundancy.html both depend on it (target name "solution-map"), via curation_server.py calling
LiveCache.get(..., depends_on=["solution-map"]).
"""
```

- [ ] **Step 4: Run the full test suite to confirm docstring-only changes broke nothing**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-live-tool-pages
git add src/clawmarks/build/map_view.py src/clawmarks/build/redundancy_view.py src/clawmarks/build/solution_map.py
git commit -m "$(cat <<'EOF'
docs(build): fix stale docstrings describing a pre-live-serving build step

map_view.py, redundancy_view.py, and solution_map.py's docstrings still told
the reader to run each module standalone after some .json file existed on
disk. Since PR #7, compute_data() returns its result in memory for
curation_server.py to cache and serve live via LiveCache's depends_on
mechanism; these files are no longer build-step outputs any other tool reads.
EOF
)"
```

---

### Task 11: Remove `merge_round2.py`'s orphaned file writes (C12)

**Files:**
- Modify: `src/clawmarks/build/merge_round2.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `main()` keeps its exact signature and its `scored_manifest.json` /
  `solution_map_final_embs.pt` writes (still genuinely read: the manifest is the live server's
  actual data source, and the embeddings file is `solution_map.compute_data`'s cache-skip input,
  now correctly watched per Task 2). Only the three writes nothing reads anymore are removed.

**Context:** `merge_round2.py:113-120` writes `similarity_scored.json` and `similarity.json`,
and `:166-167` writes `solution_map_data.json`. Before PR #7, `build_scan_gallery.py` and
`build_map_view.py` read these files directly off disk. Since PR #7, `curation_server.py`
computes this data in memory via `similarity_index.compute_data` / `solution_map.compute_data`
and never reads any of these three files; they're pure orphaned writes now, wasted work (a full
UMAP refit and similarity matrix computation over 3672+ images, discarded to disk for nothing)
every time round 2+ gets merged in.

- [ ] **Step 1: Remove the two orphaned similarity-file writes**

Replace `src/clawmarks/build/merge_round2.py:100-121`:

```python
    # --- Recompute similarity (with scores) over the merged set ---
    print("recomputing pairwise similarity over the merged set...", flush=True)
    tags = [m["tag"] for m in merged_manifest]
    assert [m["file"] for m in merged_manifest] == merged_paths, "manifest/embedding order mismatch after merge"

    sim = merged_gen_embs @ merged_gen_embs.T
    sim.fill_diagonal_(-1.0)
    TOP_K = 16
    topk_vals, topk_idx = sim.topk(TOP_K, dim=1)
    neighbors_scored = {
        tags[i]: [[tags[j], round(topk_vals[i][k].item(), 4)] for k, j in enumerate(topk_idx[i].tolist())]
        for i in range(len(tags))
    }
    with open(f"{SWEEP_DIR}/similarity_scored.json", "w") as f:
        json.dump(neighbors_scored, f)

    # Old-format similarity.json (neighbor tags only, no scores) that build_scan_gallery.py's "show
    # similar" feature reads.
    neighbors_plain = {t: [n[0] for n in v] for t, v in neighbors_scored.items()}
    with open(f"{SWEEP_DIR}/similarity.json", "w") as f:
        json.dump(neighbors_plain, f)
    print(f"wrote similarity_scored.json and similarity.json ({len(tags)} images, top-{TOP_K} each)", flush=True)
```

with:

```python
    # Similarity and solution-map data are no longer written to disk here: since PR #7,
    # curation_server.py computes both live (similarity_index.compute_data /
    # solution_map.compute_data) from scored_manifest.json and solution_map_final_embs.pt,
    # neither of which any code still reads these two files for.
    tags = [m["tag"] for m in merged_manifest]
    assert [m["file"] for m in merged_manifest] == merged_paths, "manifest/embedding order mismatch after merge"
```

- [ ] **Step 2: Remove the orphaned `solution_map_data.json` write**

Replace `src/clawmarks/build/merge_round2.py:123-168` (everything from the UMAP refit comment
through the `print("wrote solution_map_data.json ...")` line, keeping the final `"DONE merging
round 2"` print):

```python
    # --- Refit UMAP on the merged real+generated embedding space ---
    print("refitting UMAP on the merged embedding space...", flush=True)
    import umap
    import numpy as np

    all_embs = torch.cat([real_embs, merged_gen_embs], dim=0).numpy()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(all_embs)
    real_coords = coords[:len(real_paths)]
    gen_coords = coords[len(real_paths):]

    nn_real_matrix = merged_gen_embs @ real_embs.T
    nearest_real_idx = nn_real_matrix.argmax(dim=1).tolist()
    nearest_real_sim = nn_real_matrix.max(dim=1).values.tolist()
    real_names = [os.path.basename(p) for p in real_paths]

    def generation_of(tag):
        m = re.match(r"(?:r2_)?gen(\d+)_", tag)
        return int(m.group(1)) if m else 0

    by_tag = {m["tag"]: m for m in merged_manifest}
    points = []
    for i, tag in enumerate(tags):
        m = by_tag[tag]
        points.append({
            "tag": tag,
            "x": round(float(gen_coords[i][0]), 4),
            "y": round(float(gen_coords[i][1]), 4),
            "gen": generation_of(tag),
            "round": m["round"],
            "category": m["category"],
            "prompt_type": m["prompt_type"],
            "prompt_name": m["prompt_name"],
            "faith": round(m["centroid_sim"], 4),
            "novelty": round(m["novelty"], 4),
            "nearest_real": real_names[nearest_real_idx[i]],
            "nearest_real_sim": round(nearest_real_sim[i], 4),
            "thumb": f"thumbs/{tag}.jpg" if os.path.exists(f"{SWEEP_DIR}/thumbs/{tag}.jpg") else os.path.basename(m["file"]),
        })
    real_points = [
        {"name": real_names[i], "x": round(float(real_coords[i][0]), 4), "y": round(float(real_coords[i][1]), 4)}
        for i in range(len(real_paths))
    ]
    with open(f"{SWEEP_DIR}/solution_map_data.json", "w") as f:
        json.dump({"points": points, "real_points": real_points}, f)
    print(f"wrote solution_map_data.json ({len(points)} points)", flush=True)

    print("DONE merging round 2", flush=True)
```

with:

```python
    # UMAP is no longer refit and written here either: solution_map.compute_data() refits it
    # live off solution_map_final_embs.pt (updated above) the next time map.html or
    # redundancy.html is requested, and curation_server.py's LiveCache keeps that result cached
    # across requests until the manifest or embeddings file changes again.

    print("DONE merging round 2", flush=True)
```

Also remove the now-unused `import re` at `src/clawmarks/build/merge_round2.py:21` (only
`generation_of`, just deleted, used it) if nothing else in the file uses the `re` module. Verify
with:

```bash
rg -n "\bre\." src/clawmarks/build/merge_round2.py
```

If that returns nothing, remove `re` from the `import json, os, sys, shutil, re` line at the
top of the file.

- [ ] **Step 3: Run the module's tests to confirm nothing depended on the removed writes**

Run: `cd /workspace/trent-live-tool-pages && PYTHONPATH=src uv run pytest tests/ -k merge_round2 -v`

If no test file targets `merge_round2.py` directly, instead run the full suite to confirm no
other test reads `similarity_scored.json`, `similarity.json`, or `solution_map_data.json`:

```bash
rg -n "similarity_scored\.json|solution_map_data\.json" tests/
```

Expected: the full suite (`PYTHONPATH=src uv run pytest tests/ -v`) PASSES, and the `rg` search
above returns no hits (confirming no test relies on these files existing on disk).

- [ ] **Step 4: Commit**

```bash
cd /workspace/trent-live-tool-pages
git add src/clawmarks/build/merge_round2.py
git commit -m "$(cat <<'EOF'
chore(merge-round2): stop writing similarity/solution-map files nothing reads

Since PR #7, curation_server.py computes this data live from
scored_manifest.json and solution_map_final_embs.pt (both still written and
still read). similarity_scored.json, similarity.json, and
solution_map_data.json were pure orphaned writes: a full UMAP refit and
similarity matrix over the whole merged set, discarded to disk for nothing.
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** every actionable GLM finding (C1, C2, C3, C5, C6, C7, C8, C9, C10, C11,
  C12) has exactly one task above. C4 (map/redundancy correctly depend on fresh solution-map
  data) and C13 (`probe_report.py`'s standalone `__main__` entry point) were confirmed sound by
  direct source inspection during the review; no task needed for either.
- **Ordering:** Task 3 (C6) must land before or alongside Task 4 (C5) touches `/map.html` /
  `/redundancy.html`'s neighbors in the same file, but the two tasks touch disjoint route blocks
  (`/map.html`+`/redundancy.html` vs. the other six), so either order is safe; they're listed in
  GLM's severity order (C6 before C5) since C6 is the more specific/severe inconsistency.
  Task 7 (C8) builds directly on Task 6 (C3)'s `_manifest_cache` shape, so Task 7 must run after
  Task 6, which this plan's ordering already respects.
- **No placeholders:** every step shows the exact code to write, the exact command to run, and
  the exact expected output or failure message.
