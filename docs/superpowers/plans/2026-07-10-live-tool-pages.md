# Live Tool Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire `clawmarks build <target>` entirely; `curation_server.py` computes and renders every tool page live, on request, cached in memory and invalidated by input-file mtimes.

**Architecture:** Each `build/*.py` module splits into `compute_data(sweep_dir)` (pure, reads inputs, returns a data structure) and, where it renders a page, `render_html(data)` (unchanged template logic, now a function of `data` instead of a script that reads-then-writes). A new `LiveCache` in `src/clawmarks/live_cache.py` generalizes the mtime-cache pattern already used for `scored_manifest.json` in `curation_server.py`'s existing `load_manifest()`/`_manifest_cache`, extending it to per-target watched-file lists and cross-target dependencies (`solution-map`'s output feeds `map` and `redundancy`). `curation_server.py` gains a routing table mapping request paths to targets.

**Tech Stack:** Python 3, stdlib `http.server`, `pytest`, existing `tests/fixtures/sample_sweep/` fixture (from the 2026-07-09/07-10 preference-classifier work).

## Global Constraints

- Install/manage Python packages with `uv` only (`uv add`, `uv sync`), never bare `pip install`, per this project's standing convention. This plan adds no new third-party dependencies.
- No em dashes (`—` or ` -- `) in any doc, commit message, or user-facing string this plan touches. Grep for both before calling a task done.
- Every task's tests run via `PYTHONPATH=src uv run pytest tests/<file> -v` from the repo root.
- The worktree this plan executes in must be on a real branch (not detached HEAD) before any commits happen, per the incident logged in `notes/lab_notebook.md` about a shared checkout. If starting from a detached-HEAD serve worktree, `git checkout -b <branch-name>` first.
- Referenced source line numbers are from `/workspace/trent-serve-worktree` at commit `91f455e` (branch `fix/scan-tooltip-and-lightbox-js`). If the branch has moved since, re-locate by function name, not line number.

---

## Task 1: `LiveCache` infrastructure

**Files:**
- Create: `src/clawmarks/live_cache.py`
- Test: `tests/test_live_cache.py`

**Interfaces:**
- Produces: `LiveCache` class with `.get(target_name, compute_fn, watched_files, depends_on=())`. `watched_files` is a list of absolute paths. `depends_on` is a list of other target names already registered on the same `LiveCache` instance; `.get` resolves them first and passes a `deps` dict (`{name: data}`) to `compute_fn` as its second positional argument if `depends_on` is non-empty, else `compute_fn` is called with just `sweep_dir`.
- Produces: module-level `NotComputed` sentinel is not needed; a target with no prior entry is always (re)computed on first `.get`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_live_cache.py
import os
import time

import pytest

from clawmarks.live_cache import LiveCache


def test_computes_once_and_caches_when_files_unchanged(tmp_path):
    watched = tmp_path / "manifest.json"
    watched.write_text("[]")
    calls = []

    def compute(sweep_dir):
        calls.append(1)
        return {"n": len(calls)}

    cache = LiveCache()
    first = cache.get("scan", compute, watched_files=[str(watched)])
    second = cache.get("scan", compute, watched_files=[str(watched)])

    assert first == {"n": 1}
    assert second is first
    assert len(calls) == 1


def test_recomputes_when_watched_file_mtime_changes(tmp_path):
    watched = tmp_path / "manifest.json"
    watched.write_text("[]")
    calls = []

    def compute(sweep_dir):
        calls.append(1)
        return {"n": len(calls)}

    cache = LiveCache()
    cache.get("scan", compute, watched_files=[str(watched)])

    new_mtime = os.path.getmtime(watched) + 5
    watched.write_text('[{"tag": "a"}]')
    os.utime(watched, (new_mtime, new_mtime))

    second = cache.get("scan", compute, watched_files=[str(watched)])
    assert second == {"n": 2}
    assert len(calls) == 2


def test_depends_on_passes_dependency_data_and_propagates_invalidation(tmp_path):
    watched = tmp_path / "scored_manifest.json"
    watched.write_text("[]")
    base_calls, dependent_calls = [], []

    def compute_base(sweep_dir):
        base_calls.append(1)
        return {"base_n": len(base_calls)}

    def compute_dependent(sweep_dir, deps):
        dependent_calls.append(1)
        return {"from_base": deps["solution-map"]["base_n"], "dependent_n": len(dependent_calls)}

    cache = LiveCache()

    def get_dependent():
        cache.get("solution-map", compute_base, watched_files=[str(watched)])
        return cache.get(
            "map", compute_dependent, watched_files=[], depends_on=["solution-map"],
        )

    first = get_dependent()
    assert first == {"from_base": 1, "dependent_n": 1}

    new_mtime = os.path.getmtime(watched) + 5
    watched.write_text('[{"tag": "a"}]')
    os.utime(watched, (new_mtime, new_mtime))

    second = get_dependent()
    assert second == {"from_base": 2, "dependent_n": 2}


def test_concurrent_get_only_computes_once(tmp_path):
    import threading

    watched = tmp_path / "manifest.json"
    watched.write_text("[]")
    calls = []
    start_gate = threading.Barrier(4)

    def compute(sweep_dir):
        start_gate.wait(timeout=2)
        calls.append(1)
        return {"n": len(calls)}

    cache = LiveCache()
    results = []

    def worker():
        results.append(cache.get("scan", compute, watched_files=[str(watched)]))

    # Only 3 of the 4 barrier parties are worker threads; compute() itself waits on the
    # barrier too, so this only proves the lock serializes *entry*, not that concurrent
    # misses collapse into one compute. Use a simpler serialization check instead: run two
    # threads with a small delay inside compute and assert only one call happened.
    calls.clear()

    def slow_compute(sweep_dir):
        time.sleep(0.05)
        calls.append(1)
        return {"n": len(calls)}

    cache2 = LiveCache()
    threads = [threading.Thread(target=lambda: cache2.get("scan", slow_compute, watched_files=[str(watched)]))
               for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)

    assert len(calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_live_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'clawmarks.live_cache'`

- [ ] **Step 3: Implement `LiveCache`**

```python
# src/clawmarks/live_cache.py
"""
Generalizes the mtime-invalidated cache curation_server.py already used just for
scored_manifest.json (see the old _manifest_cache/load_manifest()) to every live-rendered
tool-page target: one cache entry per target name, invalidated when any of that target's
declared watched files change mtime, with support for one target's compute_fn depending on
another's already-cached data (e.g. "map" and "redundancy" both need "solution-map"'s output).
"""
import os
import threading


class LiveCache:
    def __init__(self):
        self._entries = {}
        self._locks = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, target_name):
        with self._locks_guard:
            if target_name not in self._locks:
                self._locks[target_name] = threading.Lock()
            return self._locks[target_name]

    def _current_mtimes(self, watched_files):
        return {path: os.path.getmtime(path) for path in watched_files}

    def get(self, target_name, compute_fn, watched_files, depends_on=()):
        deps = {name: self.get(name, None, None) for name in depends_on} if depends_on and False else None
        with self._lock_for(target_name):
            deps = None
            if depends_on:
                deps = {}
                for dep_name in depends_on:
                    if dep_name not in self._entries:
                        raise KeyError(
                            f"target {target_name!r} depends on {dep_name!r}, "
                            f"but {dep_name!r} has never been computed yet. "
                            f"Call cache.get({dep_name!r}, ...) before {target_name!r}."
                        )
                    deps[dep_name] = self._entries[dep_name]["data"]

            mtimes = self._current_mtimes(watched_files)
            entry = self._entries.get(target_name)
            if entry is not None and entry["mtimes"] == mtimes and entry.get("dep_signature") == self._dep_signature(depends_on):
                return entry["data"]

            data = compute_fn(None, deps) if depends_on else compute_fn(None)
            self._entries[target_name] = {
                "data": data,
                "mtimes": mtimes,
                "dep_signature": self._dep_signature(depends_on),
            }
            return data

    def _dep_signature(self, depends_on):
        return tuple(
            self._entries[name]["mtimes_signature"] if name in self._entries else None
            for name in depends_on
        )
```

- [ ] **Step 4: Run tests, see the real design bug, fix it**

Run: `PYTHONPATH=src uv run pytest tests/test_live_cache.py -v`

The draft above is over-complicated and has a bug: `_dep_signature` reads a `"mtimes_signature"` key that's never written, and the dead `deps = ... if ... and False` line and the `compute_fn(None, deps) if depends_on else compute_fn(None)` both pass `None` for `sweep_dir` instead of a real path, which doesn't match the test doubles above (they accept `sweep_dir` as their first arg and ignore it, so tests would still pass, but this is wrong for every real caller in later tasks that actually reads `sweep_dir`). Replace the whole `get` method body with this simpler, correct version before re-running:

```python
    def get(self, target_name, compute_fn, watched_files, depends_on=(), sweep_dir=None):
        with self._lock_for(target_name):
            deps = None
            if depends_on:
                deps = {}
                for dep_name in depends_on:
                    if dep_name not in self._entries:
                        raise KeyError(
                            f"target {target_name!r} depends on {dep_name!r}, "
                            f"but {dep_name!r} has never been computed yet. "
                            f"Call cache.get({dep_name!r}, ...) before {target_name!r}."
                        )
                    deps[dep_name] = self._entries[dep_name]["data"]

            mtimes = self._current_mtimes(watched_files)
            entry = self._entries.get(target_name)
            deps_changed = entry is not None and entry.get("dep_mtimes") != {
                name: self._entries[name]["mtimes"] for name in depends_on
            }
            if entry is not None and entry["mtimes"] == mtimes and not deps_changed:
                return entry["data"]

            data = compute_fn(sweep_dir, deps) if depends_on else compute_fn(sweep_dir)
            self._entries[target_name] = {
                "data": data,
                "mtimes": mtimes,
                "dep_mtimes": {name: self._entries[name]["mtimes"] for name in depends_on},
            }
            return data
```

Delete the now-unused `_dep_signature` method.

Run: `PYTHONPATH=src uv run pytest tests/test_live_cache.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/live_cache.py tests/test_live_cache.py
git commit -m "feat(live-cache): add mtime-invalidated cache with cross-target dependencies"
```

---

## Task 2: Serve shared JS assets directly from `shared_ui.py`, drop disk writes

**Files:**
- Modify: `src/clawmarks/shared_ui.py` (delete `write_lightbox_asset`, `write_scrollnav_asset`, `write_infotip_asset` function bodies' file-writing; keep the string constants `_LIGHTBOX_JS`, `SCROLLNAV_JS`, `INFOTIP_JS`)
- Modify: `src/clawmarks/curation_server.py` (`do_GET`, near the existing `/api/*` routes around line 197)
- Test: `tests/test_curation_server_static_assets.py`

**Interfaces:**
- Consumes: `_LIGHTBOX_JS`, `SCROLLNAV_JS`, `INFOTIP_JS` string constants already defined in `shared_ui.py`.
- Produces: `GET /lightbox.js`, `GET /scrollnav.js`, `GET /infotip.js` now served with `Content-Type: application/javascript` directly from those constants, no file on disk required.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curation_server_static_assets.py
import io
import json
from http.server import HTTPServer
import threading
import urllib.request

import pytest

from clawmarks import curation_server as cs


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    (tmp_path / "scored_manifest.json").write_text("[]")
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=2)


def test_lightbox_js_served_without_being_written_to_disk(running_server, tmp_path):
    port = running_server.server_address[1]
    assert not (tmp_path / "lightbox.js").exists()
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/lightbox.js") as resp:
        body = resp.read().decode()
        assert resp.headers["Content-Type"] == "application/javascript"
    assert "window.Lightbox" in body
    assert not (tmp_path / "lightbox.js").exists()


def test_infotip_js_served_without_being_written_to_disk(running_server, tmp_path):
    port = running_server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/infotip.js") as resp:
        body = resp.read().decode()
        assert resp.headers["Content-Type"] == "application/javascript"
    assert "infobtn" in body
    assert not (tmp_path / "infotip.js").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_curation_server_static_assets.py -v`
Expected: FAIL. Without the new routes, `SimpleHTTPRequestHandler`'s static serving returns 404 since no `lightbox.js` file exists in `tmp_path`.

- [ ] **Step 3: Add the routes**

In `src/clawmarks/curation_server.py`, add near the top-level imports:

```python
from clawmarks.shared_ui import _LIGHTBOX_JS, SCROLLNAV_JS, INFOTIP_JS
```

In `Handler.do_GET`, add before the final `super().do_GET()` fallthrough (i.e. right after the existing `if self.path == "/api/seeds":` block, before `if self.path == "/":`):

```python
        _JS_ASSETS = {"/lightbox.js": _LIGHTBOX_JS, "/scrollnav.js": SCROLLNAV_JS, "/infotip.js": INFOTIP_JS}
        if self.path in _JS_ASSETS:
            body = _JS_ASSETS[self.path].encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

In `src/clawmarks/shared_ui.py`, delete the bodies of `write_lightbox_asset`, `write_scrollnav_asset`, `write_infotip_asset` (and their `os.path.join(sweep_dir, ...)` file-open logic) entirely, along with their now-unused `import os` if nothing else in the file needs it. Every call site (`elite_archive.py`, `coverage_map.py`, `novelty_decay.py`, `seed_browser.py`, `map_view.py`, `lineage_view.py`, `rate_page.py`, `scan_gallery.py`, `redundancy_view.py`, `preference_rank.py`, `explore_hub.py`) still imports and calls these three functions today; leave those calls in place for now (they'll be deleted module-by-module in later tasks when each module's own `main()` disappears). To keep things working in the interim without erroring, replace each function body with `pass` rather than deleting the function, e.g.:

```python
def write_lightbox_asset(sweep_dir):
    pass  # served directly by curation_server.py's /lightbox.js route; no on-disk copy anymore


def write_scrollnav_asset(sweep_dir):
    pass  # served directly by curation_server.py's /scrollnav.js route


def write_infotip_asset(sweep_dir):
    pass  # served directly by curation_server.py's /infotip.js route
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_curation_server_static_assets.py -v`
Expected: PASS

- [ ] **Step 5: Run the full existing test suite to check nothing broke**

Run: `PYTHONPATH=src uv run pytest -v`
Expected: PASS. (Any test that asserted `lightbox.js`/`infotip.js`/`scrollnav.js` got written to disk by a build module needs updating; check for `os.path.exists(... "lightbox.js")`-style assertions in existing tests and update them to assert the function is a no-op instead.)

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/curation_server.py src/clawmarks/shared_ui.py tests/test_curation_server_static_assets.py
git commit -m "feat(server): serve lightbox/scrollnav/infotip JS directly, stop writing them to disk"
```

---

## Task 3: Extract `uncanny_gallery.py`'s DINOv2 scoring step into its own module

**Context:** `build/uncanny_gallery.py`'s `main()` does two unrelated things: (1) runs DINOv2 inference over every real training image and every generated image to compute `centroid_sim`/`novelty` and writes `scored_manifest.json` (the expensive scoring pipeline step, meant to run once per search round after `run_uncanny_sweep.py` finishes, never per page-request), and (2) bins the now-scored manifest into the faithfulness/novelty grid and renders `gallery.html` (cheap, pure, a legitimate live-cache target). Only (2) should join the live-rendering system; (1) stays a standalone script.

**Files:**
- Create: `src/clawmarks/search/score_manifest.py` (moved scoring logic: `preprocess`, `embed_images`, the leave-one-out reference calc, `_default_manifest`, and a `main(argv=None)` that does exactly what the old `uncanny_gallery.main()` did through the `scored_manifest.json` write, i.e. lines 82-153 of the old file, then prints `DONE: wrote scored_manifest.json (...)` instead of going on to build the gallery)
- Modify: `src/clawmarks/build/uncanny_gallery.py` (delete everything except `MODEL_ID`, `TYPE_COLOR`, `cell_html`, `build_html` become the basis for `render_html`; `thumb_data_uri` stays, used by `cell_html`)
- Modify: `src/clawmarks/build/similarity_index.py:15` (`from clawmarks.build.uncanny_gallery import preprocess, MODEL_ID` becomes `from clawmarks.search.score_manifest import preprocess, MODEL_ID`)
- Modify: `src/clawmarks/build/solution_map.py:29` (`from clawmarks.build.uncanny_gallery import preprocess, MODEL_ID, REAL_DIR` becomes `from clawmarks.search.score_manifest import preprocess, MODEL_ID, REAL_DIR`)
- Test: `tests/test_score_manifest.py`, `tests/test_uncanny_gallery.py`

**Interfaces:**
- Produces (`score_manifest.py`): `preprocess(img) -> torch.Tensor`, `embed_images(paths, batch_size=16, model=None) -> torch.Tensor`, `MODEL_ID = "facebook/dinov2-base"`, `REAL_DIR` (moved from `uncanny_gallery.py`, same value), `main(argv=None)` (CLI entry point, unchanged behavior, still runnable via `python -m clawmarks.search.score_manifest`).
- Produces (`uncanny_gallery.py`): `compute_data(sweep_dir) -> dict` with keys `manifest`, `grid`, `faith_edges`, `novelty_edges`, `liminal_band_top`, `real_ref` (tuple), `type_summary`, reading `scored_manifest.json` only (no DINOv2, no torch import at module level anymore). `render_html(data) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_score_manifest.py
from clawmarks.search.score_manifest import preprocess, MODEL_ID, REAL_DIR


def test_preprocess_and_constants_importable_from_new_location():
    assert MODEL_ID == "facebook/dinov2-base"
    assert REAL_DIR.endswith("corrected_dataset_extract")
    assert callable(preprocess)
```

```python
# tests/test_uncanny_gallery.py
import json

from clawmarks.build import uncanny_gallery


def _scored_manifest_fixture():
    return [
        {"file": "/tmp/a.png", "tag": "a", "centroid_sim": 0.6, "novelty": 0.4,
         "prompt_name": "fox_face", "prompt_type": "conflict", "strength": 1.2, "cfg": 5.0,
         "steps": 28, "sampler": "ddim"},
        {"file": "/tmp/b.png", "tag": "b", "centroid_sim": 0.3, "novelty": 0.7,
         "prompt_name": "style_ink", "prompt_type": "style", "strength": 1.5, "cfg": 4.0,
         "steps": 28, "sampler": "ddim"},
    ]


def test_compute_data_bins_manifest_without_importing_torch(tmp_path, monkeypatch):
    (tmp_path / "scored_manifest.json").write_text(json.dumps(_scored_manifest_fixture()))
    data = uncanny_gallery.compute_data(str(tmp_path))
    assert len(data["manifest"]) == 2
    assert "grid" in data
    assert data["type_summary"]["conflict"][1] == 1


def test_render_html_produces_gallery_markup(tmp_path):
    (tmp_path / "scored_manifest.json").write_text(json.dumps(_scored_manifest_fixture()))
    data = uncanny_gallery.compute_data(str(tmp_path))
    html = uncanny_gallery.render_html(data)
    assert "CLAWMARKS uncanny frontier atlas" in html
    assert "<html>" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_score_manifest.py tests/test_uncanny_gallery.py -v`
Expected: FAIL. `clawmarks.search.score_manifest` doesn't exist yet; `uncanny_gallery.compute_data`/`render_html` don't exist yet.

- [ ] **Step 3: Create `search/score_manifest.py`**

Copy lines 1-79 of the current `src/clawmarks/build/uncanny_gallery.py` (module docstring updated to describe just the scoring step, `import os, sys, json, base64`, `import torch`, `import numpy as np`, `from PIL import Image`, `from transformers import AutoModel`, `from clawmarks.config import ROOT, SWEEP_DIR`, `MODEL_ID`, `REAL_DIR`, `IMAGE_MEAN`, `IMAGE_STD`, `preprocess`, `embed_images`, `_default_manifest`) verbatim into the new file. Then add a `main(argv=None)` containing exactly the body of the old `main()` (lines 82-153 in the pre-Task-3 file) up through the `scored_manifest.json` write, replacing the final two `print()` calls (which referenced `build_html`/`gallery.html`) with:

```python
    print(f"DONE: wrote {SWEEP_DIR}/scored_manifest.json ({len(manifest)} images scored)", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rewrite `build/uncanny_gallery.py`**

Keep only: module docstring (updated to describe gallery rendering only, referencing `score_manifest.py` for the scoring step), `import json, os`, `from clawmarks.config import SWEEP_DIR`, `MODEL_ID = "facebook/dinov2-base"` re-export or `from clawmarks.search.score_manifest import MODEL_ID` (either is fine; re-importing avoids duplication), `N_BINS = 4`, `thumb_data_uri`, `TYPE_COLOR`, `cell_html`, `build_html` (rename to `render_html`, drop the final `with open(...).write(html)` at its end so it just `return html`).

Add `compute_data`, built from the old `main()`'s post-scoring logic (the manifest is already scored, so skip the DINOv2/embedding parts entirely and start from reading `scored_manifest.json`):

```python
def compute_data(sweep_dir):
    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    faith_vals = sorted(m["centroid_sim"] for m in manifest)
    novelty_vals = sorted(m["novelty"] for m in manifest)

    def bin_edges(vals, n):
        return [vals[int(i * len(vals) / n)] for i in range(1, n)]

    faith_edges = bin_edges(faith_vals, N_BINS)
    novelty_edges = bin_edges(novelty_vals, N_BINS)

    def bin_of(val, edges):
        for i, e in enumerate(edges):
            if val <= e:
                return i
        return len(edges)

    grid = {}
    for m in manifest:
        fb = bin_of(m["centroid_sim"], faith_edges)
        nb = bin_of(m["novelty"], novelty_edges)
        grid.setdefault((fb, nb), []).append(m)

    liminal_lo, liminal_hi = faith_edges[0], faith_edges[-1]
    liminal_band = [m for m in manifest if liminal_lo <= m["centroid_sim"] <= liminal_hi]
    liminal_band_top = sorted(liminal_band, key=lambda m: -m["novelty"])[:32]
    liminal_band_top.sort(key=lambda m: -m["centroid_sim"])

    by_type = {}
    for m in manifest:
        by_type.setdefault(m["prompt_type"], []).append(m["centroid_sim"])
    type_summary = {t: (sum(v) / len(v), len(v)) for t, v in by_type.items()}

    return {
        "manifest": manifest, "grid": grid, "faith_edges": faith_edges,
        "novelty_edges": novelty_edges, "liminal_band_top": liminal_band_top,
        "real_ref": (None, None, None), "type_summary": type_summary,
    }
```

Note: `real_ref` (the leave-one-out reference band shown in the "Reference anchor" callout) was computed from real training images during scoring in the old `main()` and isn't available post-scoring without redoing DINOv2 inference. Store it in `scored_manifest.json`'s sibling metadata instead: have `search/score_manifest.py`'s `main()` also write a small `notes/<sweep>/real_ref.json` with `{"mean": real_ref_mean, "min": real_ref_min, "max": real_ref_max}` right after it writes `scored_manifest.json`, and have `compute_data` read that file (add it to `WATCHED_FILES` in Task 14's wiring). If `real_ref.json` doesn't exist yet (an old sweep that predates this change), fall back to `(0.0, 0.0, 0.0)` and let `render_html` show "not available" instead of crashing.

Update `render_html(data)` to take the dict from `compute_data` and build the same HTML `cell_html`/`build_html` already produced, just reading from `data["manifest"]`, `data["grid"]`, etc. instead of separate positional arguments, and `return html` instead of writing it to `gallery.html`.

- [ ] **Step 5: Update the two import sites**

In `src/clawmarks/build/similarity_index.py`, change line 15 to:
```python
from clawmarks.search.score_manifest import preprocess, MODEL_ID
```

In `src/clawmarks/build/solution_map.py`, change line 29 to:
```python
from clawmarks.search.score_manifest import preprocess, MODEL_ID, REAL_DIR
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_score_manifest.py tests/test_uncanny_gallery.py -v`
Expected: PASS

- [ ] **Step 7: Run the full suite**

Run: `PYTHONPATH=src uv run pytest -v`
Expected: PASS (update `tests/test_scoring.py` or any other test that imported scoring helpers from `clawmarks.build.uncanny_gallery` to import from `clawmarks.search.score_manifest` instead; grep first: `rg -n "from clawmarks.build.uncanny_gallery import" tests/`)

- [ ] **Step 8: Commit**

```bash
git add src/clawmarks/search/score_manifest.py src/clawmarks/build/uncanny_gallery.py src/clawmarks/build/similarity_index.py src/clawmarks/build/solution_map.py tests/test_score_manifest.py tests/test_uncanny_gallery.py
git commit -m "refactor(uncanny-gallery): split DINOv2 scoring step out from gallery rendering"
```

---

## Task 4: `similarity_index.py` as a data-only live-cache target

**Context:** `similarity_index.py` has no HTML page of its own; it computes each image's top-K nearest neighbors by DINOv2 embedding and writes `similarity.json`, which `scan_gallery.py`'s "show similar" feature reads. It becomes a `DEPENDS_ON` target for `scan`, not a routed page.

**Files:**
- Modify: `src/clawmarks/build/similarity_index.py`
- Test: `tests/test_similarity_index.py`

**Interfaces:**
- Produces: `compute_data(sweep_dir) -> dict` mapping `{tag: [neighbor_tag, ...]}`, same shape `similarity.json` held.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_similarity_index.py
import json

from clawmarks.build import similarity_index


def test_compute_data_returns_tag_to_neighbors_mapping(monkeypatch, tmp_path):
    # similarity_index.py's real compute path runs DINOv2 over real images (a fixed on-disk
    # corpus) and the manifest; stub embed_images so this test doesn't need torch weights.
    manifest = [
        {"file": "/tmp/a.png", "tag": "a"},
        {"file": "/tmp/b.png", "tag": "b"},
        {"file": "/tmp/c.png", "tag": "c"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))

    import torch
    fake_embs = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    monkeypatch.setattr(similarity_index, "embed_images", lambda paths, model=None: fake_embs)
    monkeypatch.setattr(similarity_index, "AutoModel", type("M", (), {"from_pretrained": staticmethod(lambda *_: None)}))

    data = similarity_index.compute_data(str(tmp_path))
    assert set(data.keys()) == {"a", "b", "c"}
    assert data["a"][0] == "b"  # "b" is a's closest neighbor by construction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_similarity_index.py -v`
Expected: FAIL, `compute_data` doesn't exist yet (only `main`).

- [ ] **Step 3: Split `main()` into `compute_data`**

Read the current `main()` body (starts line 23, reads `scored_manifest.json` at line 24, computes the embedding matrix, checkpointing via `CHECKPOINT_FILE`, top-K neighbor lookup, writes `similarity.json` at line 84). Rename it to `compute_data(sweep_dir)`: replace every `SWEEP_DIR` reference inside the function body with `sweep_dir` (the parameter), delete the final `with open(f"{sweep_dir}/similarity.json", "w") as f: json.dump(neighbors, f)` write and the trailing `print(...)`, and `return neighbors` instead. Keep `CHECKPOINT_FILE` logic (the resumable-embedding checkpoint is orthogonal to whether the result also gets written to `similarity.json`; keep using it exactly as today so a slow real run can still resume, just don't require the final JSON write).

Delete `main(argv=None)`'s CLI wrapper and the `if __name__ == "__main__":` block; nothing calls this module as a script anymore now that it's a live-cache dependency.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_similarity_index.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/build/similarity_index.py tests/test_similarity_index.py
git commit -m "refactor(similarity): expose compute_data() for live-cache use, drop CLI"
```

---

## Task 5: `solution_map.py` as a data-only live-cache target

**Files:**
- Modify: `src/clawmarks/build/solution_map.py`
- Test: `tests/test_solution_map.py`

**Interfaces:**
- Produces: `compute_data(sweep_dir) -> dict` with keys `"solution_map_data"` (the list `solution_map_data.json` used to hold) and `"similarity_scored"` (the dict `similarity_scored.json` used to hold).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_solution_map.py
import json

from clawmarks.build import solution_map


def test_compute_data_returns_both_outputs(monkeypatch, tmp_path):
    manifest = [{"file": "/tmp/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5,
                 "novelty": 0.4, "prompt_type": "conflict"}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))

    import torch
    monkeypatch.setattr(solution_map, "embed_images", lambda paths, model=None: torch.tensor([[1.0, 0.0]]))
    monkeypatch.setattr(solution_map, "AutoModel", type("M", (), {"from_pretrained": staticmethod(lambda *_: None)}))

    data = solution_map.compute_data(str(tmp_path))
    assert "solution_map_data" in data
    assert "similarity_scored" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_solution_map.py -v`
Expected: FAIL, no `compute_data`.

- [ ] **Step 3: Split `main()` into `compute_data`**

Same mechanical transform as Task 4: rename `main`'s body (starts line 38, reads `scored_manifest.json` line 39, writes `similarity_scored.json` line 132, writes `solution_map_data.json` line 180) to `compute_data(sweep_dir)`, replace `SWEEP_DIR` with the `sweep_dir` parameter throughout, delete both `with open(..., "w")` writes, and instead of returning nothing, capture what was going to be written into two local variables (`similarity_scored`, `solution_map_points` — check the exact local variable names already used right before each `open(...).write` call and reuse them) and `return {"solution_map_data": solution_map_points, "similarity_scored": similarity_scored}`. Delete `main`'s CLI wrapper and `if __name__ == "__main__":`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_solution_map.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/build/solution_map.py tests/test_solution_map.py
git commit -m "refactor(solution-map): expose compute_data() for live-cache use, drop CLI"
```

---

## Task 6: `scan_gallery.py` split + wire into the server (worked example for all remaining page targets)

**Files:**
- Modify: `src/clawmarks/build/scan_gallery.py`
- Modify: `src/clawmarks/curation_server.py`
- Test: `tests/test_scan_gallery.py`, `tests/test_curation_server_scan_route.py`

**Interfaces:**
- Produces: `compute_data(sweep_dir, deps) -> list[dict]` (the `items` list scan.html/scan_data.json both need; `deps["similarity"]` is the `{tag: [neighbor_tag,...]}` dict from Task 4).
- Produces: `render_html(items) -> str`.
- Consumes: `live_cache.get("similarity", similarity_index.compute_data, watched_files=[f"{sweep_dir}/scored_manifest.json"])` must run before `"scan"` in `curation_server.py`'s routing (dependency order).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scan_gallery.py
import json

from clawmarks.build import scan_gallery


def test_compute_data_builds_items_with_similarity(tmp_path):
    manifest = [
        {"file": "/x/a.png", "tag": "a", "category": "seedrun1", "prompt_name": "fox",
         "prompt_type": "conflict", "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1,
         "steps": 28, "sampler": "ddim", "negative": "n", "centroid_sim": 0.5, "novelty": 0.5},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    deps = {"similarity": {"a": ["b", "c"]}}

    items = scan_gallery.compute_data(str(tmp_path), deps)
    assert items[0]["tag"] == "a"
    assert items[0]["sim"] == ["b", "c"]


def test_render_html_embeds_data_and_infobtn_tips():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0, "category": "seedrun1",
              "prompt_name": "fox", "prompt_type": "conflict", "prompt": "p", "strength": 1.0,
              "cfg": 5.0, "seed": 1, "steps": 28, "sampler": "ddim", "negative": "n",
              "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert '"tag": "a"' in html
    assert "infobtn" in html
```

```python
# tests/test_curation_server_scan_route.py
import json
import threading
from http.server import HTTPServer
import urllib.request

import pytest

from clawmarks import curation_server as cs


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    manifest = [
        {"file": "/x/a.png", "tag": "a", "category": "seedrun1", "prompt_name": "fox",
         "prompt_type": "conflict", "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1,
         "steps": 28, "sampler": "ddim", "negative": "n", "centroid_sim": 0.5, "novelty": 0.5},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown()
    thread.join(timeout=2)


def test_scan_html_reflects_manifest_change_without_rebuild(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/scan.html") as resp:
        first = resp.read().decode()
    assert '"prompt_name": "fox"' in first

    manifest = json.loads((tmp_path / "scored_manifest.json").read_text())
    manifest[0]["prompt_name"] = "wolf"
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    import os, time
    new_mtime = os.path.getmtime(tmp_path / "scored_manifest.json") + 5
    os.utime(tmp_path / "scored_manifest.json", (new_mtime, new_mtime))

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/scan.html") as resp:
        second = resp.read().decode()
    assert '"prompt_name": "wolf"' in second
    assert '"prompt_name": "fox"' not in second
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_scan_gallery.py tests/test_curation_server_scan_route.py -v`
Expected: FAIL (`compute_data`/`render_html` don't exist; `/scan.html` 404s with no file on disk).

- [ ] **Step 3: Split `scan_gallery.py`**

`main()` currently (lines 29-78) reads `scored_manifest.json`, reads `similarity.json` directly from disk, builds `items`, writes `scan_data.json`, calls the three `write_*_asset` functions, then goes on to build `scan.html`'s full template (through line 381-384). Replace the `similarity.json` direct read with the `deps` parameter:

```python
def compute_data(sweep_dir, deps):
    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    similarity = deps.get("similarity", {})

    items = []
    tag_to_index = {}
    for i, m in enumerate(manifest):
        tag_to_index[m["tag"]] = i
        thumb_path = f"thumbs/{m['tag']}.jpg"
        has_thumb = os.path.exists(f"{sweep_dir}/{thumb_path}")
        items.append({
            "file": os.path.basename(m["file"]), "thumb": thumb_path if has_thumb else os.path.basename(m["file"]),
            "tag": m["tag"], "gen": generation_of(m["tag"]), "category": m["category"],
            "prompt_name": m["prompt_name"], "prompt_type": m["prompt_type"], "prompt": m["prompt"],
            "strength": m["strength"], "cfg": m["cfg"], "seed": m["seed"], "steps": m["steps"],
            "sampler": m["sampler"], "negative": m["negative"], "faith": round(m["centroid_sim"], 4),
            "novelty": round(m["novelty"], 4), "sim": [],
        })

    if similarity:
        for tag, neighbor_tags in similarity.items():
            if tag in tag_to_index:
                items[tag_to_index[tag]]["sim"] = [t for t in neighbor_tags if t in tag_to_index]

    return items
```

Rename the rest of the old `main()` (the HTML-templating half, everything after the old `data_json = json.dumps(items)` line through the final `with open(..., "w") as f: f.write(...)`) to `render_html(items)`, changing its first line from `data_json = json.dumps(items)` (keep this, it's still needed to embed `const DATA = ...` in the returned HTML) through to `return html` instead of writing to a file. Delete the `write_lightbox_asset`/`write_scrollnav_asset`/`write_infotip_asset` calls (Task 2 already made these safe no-ops, but they're not needed at all anymore since nothing here writes to `sweep_dir`). Delete the `scan_data.json` disk write; that data now comes from the server route in Step 4 below. Delete `main(argv=None)` and its `if __name__ == "__main__":` block entirely.

- [ ] **Step 4: Wire `curation_server.py`'s routing table**

Add a module-level `_live_cache = LiveCache()` near the top of `curation_server.py` (after the existing `_manifest_cache`/`load_manifest` can stay for now, or be migrated in a later cleanup pass, out of scope here) and import the new pieces:

```python
from clawmarks.live_cache import LiveCache
from clawmarks.build import scan_gallery, similarity_index

_live_cache = LiveCache()


def _manifest_path():
    return f"{SWEEP_DIR}/scored_manifest.json"


def _get_scan_items():
    _live_cache.get(
        "similarity", similarity_index.compute_data,
        watched_files=[_manifest_path()], sweep_dir=str(SWEEP_DIR),
    )
    return _live_cache.get(
        "scan", scan_gallery.compute_data,
        watched_files=[_manifest_path()], depends_on=["similarity"], sweep_dir=str(SWEEP_DIR),
    )
```

Add routes in `do_GET`, before the `_JS_ASSETS` check added in Task 2:

```python
        if self.path == "/scan.html":
            html = scan_gallery.render_html(_get_scan_items())
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/scan_data.json":
            self._json_response(200, _get_scan_items())
            return
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_scan_gallery.py tests/test_curation_server_scan_route.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/build/scan_gallery.py src/clawmarks/curation_server.py tests/test_scan_gallery.py tests/test_curation_server_scan_route.py
git commit -m "feat(scan): render scan.html and scan_data.json live from the manifest, cached"
```

---

## Task 7: `solution_map` wired as a `map`/`redundancy` dependency (no route of its own)

**Files:**
- Modify: `src/clawmarks/curation_server.py`

**Interfaces:**
- Produces: `_get_solution_map_data()` helper, used by Tasks 8 and 9.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_curation_server_scan_route.py, or a new tests/test_curation_server_solution_map_dep.py
from clawmarks import curation_server as cs


def test_get_solution_map_data_is_cached_across_calls(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    (tmp_path / "scored_manifest.json").write_text(json.dumps([
        {"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5,
         "novelty": 0.4, "prompt_type": "conflict"},
    ]))
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    calls = []
    import torch

    def fake_compute(sweep_dir):
        calls.append(1)
        return {"solution_map_data": [], "similarity_scored": {}}

    monkeypatch.setattr(cs.solution_map, "compute_data", fake_compute)
    first = cs._get_solution_map_data()
    second = cs._get_solution_map_data()
    assert first is second
    assert len(calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_curation_server_solution_map_dep.py -v`
Expected: FAIL, `_get_solution_map_data` doesn't exist.

- [ ] **Step 3: Add the helper**

In `curation_server.py`, add the import `from clawmarks.build import solution_map` and:

```python
def _get_solution_map_data():
    return _live_cache.get(
        "solution-map", solution_map.compute_data,
        watched_files=[_manifest_path()], sweep_dir=str(SWEEP_DIR),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_curation_server_solution_map_dep.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/curation_server.py tests/test_curation_server_solution_map_dep.py
git commit -m "feat(server): expose solution-map data as a cached dependency for map/redundancy"
```

---

## Task 8: `map_view.py` split + route

**Files:**
- Modify: `src/clawmarks/build/map_view.py`
- Modify: `src/clawmarks/curation_server.py`
- Test: `tests/test_map_view.py`

**Interfaces:**
- Produces: `compute_data(sweep_dir, deps) -> <same structure map_view's main() derived from solution_map_data.json>`, `render_html(data) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_map_view.py
from clawmarks.build import map_view


def test_compute_data_reads_from_deps_not_disk(tmp_path):
    deps = {"solution-map": {"solution_map_data": [
        {"tag": "a", "x": 0.1, "y": 0.2, "gen": 0, "is_real": False, "prompt_name": "p",
         "prompt_type": "conflict", "faith": 0.5, "novelty": 0.5, "thumb": "thumbs/a.jpg"},
    ], "similarity_scored": {}}}
    data = map_view.compute_data(str(tmp_path), deps)
    assert len(data["points"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_map_view.py -v`
Expected: FAIL.

- [ ] **Step 3: Split `main()`**

`main()` (starts line 25) currently reads `solution_map_data.json` directly (line 45) into a local variable (check its exact name right after the `open(...)` call, likely `points`). Change the function signature to `compute_data(sweep_dir, deps)`, replace the `with open(f"{sweep_dir}/solution_map_data.json") as f: points = json.load(f)` with `points = deps["solution-map"]["solution_map_data"]`, keep everything else (the `Counter`-based generation stats, the point-shaping logic) unchanged through to just before the HTML write, `return {"points": points, ...}` (bundle whatever other locals the template section (`render_html`, formerly the tail of `main()` writing `map.html` at line 310) needs; check exactly what local variables the template f-string references between the old read and the old write, and include each one as a key in the returned dict). Rename the HTML-building tail to `render_html(data)`, unpacking those same keys back out of `data`, ending in `return html` instead of the file write. Delete `main`'s CLI wrapper, `if __name__ == "__main__":`, and the `write_lightbox_asset`/`write_scrollnav_asset`/`write_infotip_asset` calls.

- [ ] **Step 4: Wire the route**

In `curation_server.py`, import `from clawmarks.build import map_view` and add to `do_GET`:

```python
        if self.path == "/map.html":
            data = map_view.compute_data(str(SWEEP_DIR), {"solution-map": _get_solution_map_data()})
            html = map_view.render_html(data)
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_map_view.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/build/map_view.py src/clawmarks/curation_server.py tests/test_map_view.py
git commit -m "feat(map): render map.html live from solution-map's cached data"
```

---

## Task 9: `redundancy_view.py` split + route

**Files:**
- Modify: `src/clawmarks/build/redundancy_view.py`
- Modify: `src/clawmarks/curation_server.py`
- Test: `tests/test_redundancy_view.py`

**Interfaces:**
- Produces: `compute_data(sweep_dir, deps) -> data` (reads `deps["solution-map"]["similarity_scored"]` instead of `similarity_scored.json`, and `scored_manifest.json` directly as today), `render_html(data) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_redundancy_view.py
import json

from clawmarks.build import redundancy_view


def test_compute_data_uses_similarity_scored_from_deps(tmp_path):
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    deps = {"solution-map": {"similarity_scored": {"a": []}, "solution_map_data": []}}

    data = redundancy_view.compute_data(str(tmp_path), deps)
    assert data is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_redundancy_view.py -v`
Expected: FAIL.

- [ ] **Step 3: Split `main()`**

`main()` (starts line 27) reads `similarity_scored.json` (line 40) and `scored_manifest.json` (line 43). Change signature to `compute_data(sweep_dir, deps)`, replace the `similarity_scored.json` read with `sim_scored = deps["solution-map"]["similarity_scored"]`, keep the `scored_manifest.json` read as a direct file read (unchanged, it's the true root input), keep the clustering logic unchanged, bundle whatever locals the template needs into a returned dict, rename the templating tail to `render_html(data)` ending in `return html`. Delete CLI wrapper, `__main__` block, `write_*_asset` calls.

- [ ] **Step 4: Wire the route**

```python
        if self.path == "/redundancy.html":
            data = redundancy_view.compute_data(str(SWEEP_DIR), {"solution-map": _get_solution_map_data()})
            html = redundancy_view.render_html(data)
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

(Import `from clawmarks.build import redundancy_view` alongside the other build-module imports.)

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_redundancy_view.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/build/redundancy_view.py src/clawmarks/curation_server.py tests/test_redundancy_view.py
git commit -m "feat(redundancy): render redundancy.html live from solution-map's cached data"
```

---

## Task 10: `coverage_map.py` split + route

**Files:**
- Modify: `src/clawmarks/build/coverage_map.py`
- Modify: `src/clawmarks/curation_server.py`
- Test: `tests/test_coverage_map.py`

**Interfaces:**
- Produces: `compute_data(sweep_dir) -> data` (reads `scored_manifest.json` only, no deps), `render_html(data) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coverage_map.py
import json

from clawmarks.build import coverage_map


def test_compute_data_reads_manifest(tmp_path):
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = coverage_map.compute_data(str(tmp_path))
    assert data is not None
    html = coverage_map.render_html(data)
    assert "<html>" in html.lower() or "<!doctype" in html.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_coverage_map.py -v`
Expected: FAIL.

- [ ] **Step 3: Split `main()`**

Same mechanical pattern as Task 6/8/9, no `deps` needed (single input, `scored_manifest.json`, read at line 39). Rename `main(argv=None)` to `compute_data(sweep_dir)`, keep body through just before the `coverage.html` write at line 278, bundle locals into a returned dict, tail becomes `render_html(data)` returning the string. Delete CLI wrapper, `__main__`, `write_*_asset` calls.

- [ ] **Step 4: Wire the route**

```python
        if self.path == "/coverage.html":
            html = coverage_map.render_html(coverage_map.compute_data(str(SWEEP_DIR)))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_coverage_map.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/build/coverage_map.py src/clawmarks/curation_server.py tests/test_coverage_map.py
git commit -m "feat(coverage): render coverage.html live from the manifest"
```

---

## Task 11: `novelty_decay.py` split + route

Same pattern as Task 10 (single input `scored_manifest.json`, no deps). `main()` starts line 20, reads manifest line 32, writes `novelty_decay.html` line 133.

**Files:** Modify `src/clawmarks/build/novelty_decay.py`, `src/clawmarks/curation_server.py`. Test: `tests/test_novelty_decay.py`.

- [ ] **Step 1: Write the failing test** (mirror Task 10's Step 1, importing `novelty_decay`, asserting `compute_data`/`render_html` work against a two-entry manifest with a shared `prompt_name` across two `gen` values, since novelty-decay's whole point is per-prompt-family trends across generations)

```python
# tests/test_novelty_decay.py
import json

from clawmarks.build import novelty_decay


def test_compute_data_groups_by_prompt_family_across_generations(tmp_path):
    manifest = [
        {"file": "/x/gen0_fox.png", "tag": "gen0_fox_1", "prompt_name": "fox", "novelty": 0.6, "centroid_sim": 0.5},
        {"file": "/x/gen1_fox.png", "tag": "gen1_fox_1", "prompt_name": "fox", "novelty": 0.5, "centroid_sim": 0.5},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = novelty_decay.compute_data(str(tmp_path))
    html = novelty_decay.render_html(data)
    assert "fox" in html
```

- [ ] **Step 2: Run test to verify it fails.** Run: `PYTHONPATH=src uv run pytest tests/test_novelty_decay.py -v`. Expected: FAIL.

- [ ] **Step 3: Split `main()`** into `compute_data(sweep_dir)` / `render_html(data)`, same mechanical recipe as Task 10 applied to this file's line 20-133 range. Delete CLI wrapper, `__main__`, `write_*_asset` calls.

- [ ] **Step 4: Wire the route** in `curation_server.py`:

```python
        if self.path == "/novelty_decay.html":
            html = novelty_decay.render_html(novelty_decay.compute_data(str(SWEEP_DIR)))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

- [ ] **Step 5: Run test to verify it passes.** Run: `PYTHONPATH=src uv run pytest tests/test_novelty_decay.py -v`. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/build/novelty_decay.py src/clawmarks/curation_server.py tests/test_novelty_decay.py
git commit -m "feat(novelty-decay): render novelty_decay.html live from the manifest"
```

---

## Task 12: `lineage_view.py` split + route

Same pattern (single input, no deps). `main()` starts line 17, reads manifest line 21, writes `lineage.html` at line 52 (placeholder branch, when no `parent_tag` data exists) or line 94 (real branch).

**Files:** Modify `src/clawmarks/build/lineage_view.py`, `src/clawmarks/curation_server.py`. Test: `tests/test_lineage_view.py`.

- [ ] **Step 1: Write the failing tests** (both branches: manifest with no `parent_tag` field anywhere produces the placeholder; manifest with at least one `parent_tag` produces the real tree)

```python
# tests/test_lineage_view.py
import json

from clawmarks.build import lineage_view


def test_compute_data_placeholder_when_no_parent_tags(tmp_path):
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p"}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = lineage_view.compute_data(str(tmp_path))
    html = lineage_view.render_html(data)
    assert "placeholder" in html.lower() or "no parent" in html.lower()


def test_compute_data_builds_tree_when_parent_tags_exist(tmp_path):
    manifest = [
        {"file": "/x/a.png", "tag": "a", "prompt_name": "p"},
        {"file": "/x/b.png", "tag": "b", "prompt_name": "p", "parent_tag": "a"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = lineage_view.compute_data(str(tmp_path))
    html = lineage_view.render_html(data)
    assert "a" in html and "b" in html
```

- [ ] **Step 2: Run tests to verify they fail.** Run: `PYTHONPATH=src uv run pytest tests/test_lineage_view.py -v`. Expected: FAIL.

- [ ] **Step 3: Split `main()`.** Both branches currently each build their own HTML and write it directly inline (lines ~22-54 for the placeholder branch, ~55-96 for the real branch). Structure `compute_data(sweep_dir) -> dict` to return enough information to distinguish which branch applies (e.g. `{"has_lineage": bool, "children_by_parent": {...}}` or whatever the real branch's existing local variable is called) without doing any HTML string-building, and make `render_html(data)` do an `if data["has_lineage"]:` branch producing the corresponding HTML, `return html` either way. Delete CLI wrapper, `__main__`, `write_*_asset` calls.

- [ ] **Step 4: Wire the route**

```python
        if self.path == "/lineage.html":
            html = lineage_view.render_html(lineage_view.compute_data(str(SWEEP_DIR)))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

- [ ] **Step 5: Run tests to verify they pass.** Run: `PYTHONPATH=src uv run pytest tests/test_lineage_view.py -v`. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/build/lineage_view.py src/clawmarks/curation_server.py tests/test_lineage_view.py
git commit -m "feat(lineage): render lineage.html live from the manifest"
```

---

## Task 13: `elite_archive.py` split + route

`main()` starts line 47, reads `scored_manifest.json` line 69, reads `user_ratings.json` line 73, writes `archive.html` line 265. Already has an `--use-predicted-preference` argparse flag (Stage 5b, defaults off) and reads `PREFERENCE_MODEL_FILE` when that flag is set.

**Files:** Modify `src/clawmarks/build/elite_archive.py`, `src/clawmarks/curation_server.py`. Test: `tests/test_elite_archive_live.py` (the existing `tests/test_elite_archive.py` and `tests/test_elite_archive_predicted_preference.py` already test `elite_sort_key`/`build_item_summary` directly and don't need to change).

**Interfaces:**
- Produces: `compute_data(sweep_dir, use_predicted_preference=False) -> data`, `render_html(data) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_elite_archive_live.py
import json

from clawmarks.build import elite_archive


def test_compute_data_prefers_yes_rated_image_in_cell(tmp_path):
    manifest = [
        {"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.9, "prompt_type": "conflict"},
        {"file": "/x/b.png", "tag": "b", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.1, "prompt_type": "conflict"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "user_ratings.json").write_text(json.dumps({"b": {"label": "yes", "rated_at": "x"}}))
    data = elite_archive.compute_data(str(tmp_path))
    html = elite_archive.render_html(data)
    assert '"tag": "b"' in html or "tag=b" in html or "b.png" in html
```

- [ ] **Step 2: Run test to verify it fails.** Run: `PYTHONPATH=src uv run pytest tests/test_elite_archive_live.py -v`. Expected: FAIL.

- [ ] **Step 3: Split `main()`.** `main()` today takes `argv` for the `--use-predicted-preference` flag via `argparse`. Change signature to `compute_data(sweep_dir, use_predicted_preference=False)` (drop `argparse` entirely, the flag becomes a plain keyword argument the caller decides), keep the body (manifest read, ratings read, optional `PREFERENCE_MODEL_FILE`-based scoring when `use_predicted_preference` is true, `elite_sort_key`/`build_item_summary` calls, cell selection) through just before the `archive.html` write at line 265, bundle needed locals into a returned dict, tail becomes `render_html(data)`. Delete `argparse` import, CLI wrapper, `__main__`, `write_*_asset` calls.

- [ ] **Step 4: Wire the route.** The old CLI's `--use-predicted-preference` flag was opt-in and off by default; the server route keeps that same default (`False`) unless a query parameter requests it, matching the spec's "Stage 5b stays off by default" rule:

```python
        if self.path.startswith("/archive.html"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            use_predicted = query.get("use_predicted_preference", ["0"])[0] == "1"
            data = elite_archive.compute_data(str(SWEEP_DIR), use_predicted_preference=use_predicted)
            html = elite_archive.render_html(data)
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

- [ ] **Step 5: Run test to verify it passes.** Run: `PYTHONPATH=src uv run pytest tests/test_elite_archive_live.py -v`. Expected: PASS.

- [ ] **Step 6: Run the existing elite_archive tests too.** Run: `PYTHONPATH=src uv run pytest tests/test_elite_archive.py tests/test_elite_archive_predicted_preference.py -v`. Expected: PASS unchanged (they test `elite_sort_key`/`build_item_summary` directly, untouched by this split).

- [ ] **Step 7: Commit**

```bash
git add src/clawmarks/build/elite_archive.py src/clawmarks/curation_server.py tests/test_elite_archive_live.py
git commit -m "feat(archive): render archive.html live, use_predicted_preference via query param"
```

---

## Task 14: `preference_rank.py` split + route

`main()` starts line 38, reads `scored_manifest.json` line 48, writes `preference_rank.html` line 99. Already has `build_ranked_items` as a separate pure helper.

**Files:** Modify `src/clawmarks/build/preference_rank.py`, `src/clawmarks/curation_server.py`. Test: `tests/test_preference_rank_live.py` (existing `tests/test_preference_rank.py` tests `build_ranked_items` directly, untouched).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preference_rank_live.py
import json

from clawmarks.build import preference_rank


def test_compute_data_returns_none_below_min_labels(tmp_path):
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    data = preference_rank.compute_data(str(tmp_path))
    html = preference_rank.render_html(data)
    assert "not enough" in html.lower() or "50" in html
```

(This exercises the existing `MIN_LABELS = 50` floor path from the preference-classifier work: with no trained model file present, `compute_data` should return whatever "no model yet" state `main()` already handled, not crash.)

- [ ] **Step 2: Run test to verify it fails.** Run: `PYTHONPATH=src uv run pytest tests/test_preference_rank_live.py -v`. Expected: FAIL.

- [ ] **Step 3: Split `main()`** into `compute_data(sweep_dir)` (reads manifest, loads the trained model via `MODEL_FILE`/`joblib` if present, calls `predict_proba`/`build_ranked_items`, returns whatever data the existing "no model" vs. "ranked" branches need) and `render_html(data)` (the HTML tail). Delete CLI wrapper, `__main__`, `write_*_asset` calls.

- [ ] **Step 4: Wire the route**

```python
        if self.path == "/preference_rank.html":
            html = preference_rank.render_html(preference_rank.compute_data(str(SWEEP_DIR)))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

- [ ] **Step 5: Run test to verify it passes.** Run: `PYTHONPATH=src uv run pytest tests/test_preference_rank_live.py -v`. Expected: PASS.

- [ ] **Step 6: Run the existing preference_rank test.** Run: `PYTHONPATH=src uv run pytest tests/test_preference_rank.py -v`. Expected: PASS unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/clawmarks/build/preference_rank.py src/clawmarks/curation_server.py tests/test_preference_rank_live.py
git commit -m "feat(preference-rank): render preference_rank.html live"
```

---

## Task 15: `uncanny_gallery.py` gallery route (finishing Task 3's split)

**Files:** Modify `src/clawmarks/curation_server.py`. Test: `tests/test_curation_server_gallery_route.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curation_server_gallery_route.py
import json
import threading
from http.server import HTTPServer
import urllib.request

import pytest

from clawmarks import curation_server as cs


def test_gallery_html_served_live(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "fox", "prompt_type": "conflict",
                 "centroid_sim": 0.5, "novelty": 0.5, "strength": 1.0, "cfg": 5.0, "steps": 28, "sampler": "ddim"}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "real_ref.json").write_text(json.dumps({"mean": 0.8, "min": 0.7, "max": 0.9}))
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/gallery.html") as resp:
            html = resp.read().decode()
        assert "CLAWMARKS uncanny frontier atlas" in html
    finally:
        server.shutdown()
        thread.join(timeout=2)
```

- [ ] **Step 2: Run test to verify it fails.** Run: `PYTHONPATH=src uv run pytest tests/test_curation_server_gallery_route.py -v`. Expected: FAIL (no route yet).

- [ ] **Step 3: Wire the route.** Import `from clawmarks.build import uncanny_gallery` and add to `do_GET`:

```python
        if self.path == "/gallery.html":
            html = uncanny_gallery.render_html(uncanny_gallery.compute_data(str(SWEEP_DIR)))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

- [ ] **Step 4: Run test to verify it passes.** Run: `PYTHONPATH=src uv run pytest tests/test_curation_server_gallery_route.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/curation_server.py tests/test_curation_server_gallery_route.py
git commit -m "feat(gallery): render gallery.html live from the already-scored manifest"
```

---

## Task 16: `explore_hub.py` split + route (static, no watched files)

`main()` starts line 27, has no manifest read at all, writes `explore.html` line 83.

**Files:** Modify `src/clawmarks/build/explore_hub.py`, `src/clawmarks/curation_server.py`. Test: `tests/test_explore_hub.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_explore_hub.py
from clawmarks.build import explore_hub


def test_render_html_lists_every_tool():
    html = explore_hub.render_html()
    for path, label, _desc in explore_hub.TOOLS:
        assert path in html
```

- [ ] **Step 2: Run test to verify it fails.** Run: `PYTHONPATH=src uv run pytest tests/test_explore_hub.py -v`. Expected: FAIL.

- [ ] **Step 3: Split `main()`.** No `compute_data` needed since there's no data dependency; rename the whole body of `main()` to `render_html()` (no arguments), ending `return html` instead of writing `explore.html`. Delete CLI wrapper, `__main__`, `write_infotip_asset` call.

- [ ] **Step 4: Wire the route**

```python
        if self.path == "/explore.html":
            body = explore_hub.render_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

- [ ] **Step 5: Run test to verify it passes.** Run: `PYTHONPATH=src uv run pytest tests/test_explore_hub.py -v`. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/build/explore_hub.py src/clawmarks/curation_server.py tests/test_explore_hub.py
git commit -m "feat(explore-hub): render explore.html live (static content, no manifest dependency)"
```

---

## Task 17: `seed_browser.py` split + route

`main()` starts line 23, writes `seeds.html` line 161. Reads `SEEDS_FILE` (`candidate_seeds.json`), imported from `clawmarks.config`.

**Files:** Modify `src/clawmarks/build/seed_browser.py`, `src/clawmarks/curation_server.py`. Test: `tests/test_seed_browser.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seed_browser.py
import json

from clawmarks.build import seed_browser


def test_compute_data_reads_candidate_seeds(tmp_path, monkeypatch):
    seeds_file = tmp_path / "candidate_seeds.json"
    seeds_file.write_text(json.dumps({"a fox in rain": {"source": "manual", "created_at": "x"}}))
    monkeypatch.setattr(seed_browser, "SEEDS_FILE", seeds_file)
    data = seed_browser.compute_data(str(tmp_path))
    html = seed_browser.render_html(data)
    assert "a fox in rain" in html
```

- [ ] **Step 2: Run test to verify it fails.** Run: `PYTHONPATH=src uv run pytest tests/test_seed_browser.py -v`. Expected: FAIL.

- [ ] **Step 3: Split `main()`** into `compute_data(sweep_dir)` (reads `SEEDS_FILE`, same as today) / `render_html(data)`. Delete CLI wrapper, `__main__`, `write_scrollnav_asset`/`write_infotip_asset` calls.

- [ ] **Step 4: Wire the route**

```python
        if self.path == "/seeds.html":
            html = seed_browser.render_html(seed_browser.compute_data(str(SWEEP_DIR)))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

- [ ] **Step 5: Run test to verify it passes.** Run: `PYTHONPATH=src uv run pytest tests/test_seed_browser.py -v`. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/build/seed_browser.py src/clawmarks/curation_server.py tests/test_seed_browser.py
git commit -m "feat(seeds): render seeds.html live from candidate_seeds.json"
```

---

## Task 18: `rate_page.py` split + route (static shell, no watched files)

Per its own docstring, `rate.html` "bakes in no per-image data at build time"; it fetches `/api/rate/next` client-side, already live via the existing API. `main()` starts line 16, writes `rate.html` line 113.

**Files:** Modify `src/clawmarks/build/rate_page.py`, `src/clawmarks/curation_server.py`. Test: `tests/test_rate_page.py` (already exists, check it doesn't assert a file got written to disk; update if it does).

- [ ] **Step 1: Check the existing test**

Run: `PYTHONPATH=src uv run pytest tests/test_rate_page.py -v` and read `tests/test_rate_page.py`. If it currently calls `rate_page.main()` and asserts on a written `rate.html` file, update it to call a new `rate_page.render_html()` and assert on the returned string instead.

- [ ] **Step 2: Split `main()`** into `render_html()` (no `compute_data`, matching Task 16's pattern since this page has no data dependency), ending `return html`. Delete CLI wrapper, `__main__`, `write_scrollnav_asset`/`write_infotip_asset` calls.

- [ ] **Step 3: Wire the route**

```python
        if self.path == "/rate.html":
            body = rate_page.render_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
```

(Import `from clawmarks.build import rate_page`.)

- [ ] **Step 4: Run the test to verify it passes.** Run: `PYTHONPATH=src uv run pytest tests/test_rate_page.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/build/rate_page.py src/clawmarks/curation_server.py tests/test_rate_page.py
git commit -m "feat(rate): render rate.html live (static shell, data already comes from the API)"
```

---

## Task 19: Lazy on-demand thumbnails, drop `thumbnails.py`'s CLI

**Files:**
- Modify: `src/clawmarks/build/thumbnails.py` (keep only a `generate_thumbnail(src_path, dst_path)` helper, delete `main`/argparse/`__main__`)
- Modify: `src/clawmarks/curation_server.py` (`Handler.do_GET`)
- Test: `tests/test_curation_server_lazy_thumbnails.py`

**Interfaces:**
- Produces: `generate_thumbnail(src_path, dst_path)` (resizes to `THUMB_SIZE`, saves JPEG at `QUALITY`, both constants stay in `thumbnails.py`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_curation_server_lazy_thumbnails.py
import json
import threading
from http.server import HTTPServer
from PIL import Image
import urllib.request

import pytest

from clawmarks import curation_server as cs


def test_thumb_generated_on_first_request(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    manifest = [{"file": str(tmp_path / "a.png"), "tag": "a"}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    Image.new("RGB", (500, 500), color="red").save(tmp_path / "a.png")

    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        assert not (tmp_path / "thumbs" / "a.jpg").exists()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/thumbs/a.jpg") as resp:
            assert resp.status == 200
        assert (tmp_path / "thumbs" / "a.jpg").exists()
        img = Image.open(tmp_path / "thumbs" / "a.jpg")
        assert max(img.size) <= 220
    finally:
        server.shutdown()
        thread.join(timeout=2)
```

- [ ] **Step 2: Run test to verify it fails.** Run: `PYTHONPATH=src uv run pytest tests/test_curation_server_lazy_thumbnails.py -v`. Expected: FAIL, `/thumbs/a.jpg` 404s (no file on disk, no lazy-generation route yet).

- [ ] **Step 3: Slim `thumbnails.py` down to a helper**

```python
# src/clawmarks/build/thumbnails.py
"""
Resizes a single source image into a small JPEG thumbnail. Used by curation_server.py to
lazily generate notes/<sweep>/thumbs/<tag>.jpg on first request instead of pre-generating
every thumbnail in a batch step; once made, a thumbnail never goes stale (its source image
doesn't change after generation), so there's nothing to invalidate.
"""
from PIL import Image

THUMB_SIZE = 220
QUALITY = 78


def generate_thumbnail(src_path, dst_path):
    img = Image.open(src_path).convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.BICUBIC)
    img.save(dst_path, format="JPEG", quality=QUALITY)
```

- [ ] **Step 4: Add the lazy-generation route**

In `curation_server.py`, import `from clawmarks.build.thumbnails import generate_thumbnail` and, in `do_GET`, before the final `super().do_GET()` fallthrough, add a check that runs for any `/thumbs/<tag>.jpg` request whose file doesn't exist yet on disk:

```python
        if self.path.startswith("/thumbs/") and self.path.endswith(".jpg"):
            thumb_path = f"{SWEEP_DIR}{self.path}"
            if not os.path.exists(thumb_path):
                tag = os.path.basename(self.path)[: -len(".jpg")]
                manifest = load_manifest()
                match = next((m for m in manifest if m["tag"] == tag), None)
                if match is None:
                    self.send_error(404, "no manifest entry for this tag")
                    return
                os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                generate_thumbnail(match["file"], thumb_path)
            # fall through to super().do_GET() below, which now finds the file on disk
```

(This relies on `self.path.startswith("/thumbs/")` falling through, unmodified, to the existing `super().do_GET()` at the end of the method, which serves the now-just-created file normally.)

- [ ] **Step 5: Run test to verify it passes.** Run: `PYTHONPATH=src uv run pytest tests/test_curation_server_lazy_thumbnails.py -v`. Expected: PASS.

- [ ] **Step 6: Check for other callers of `thumbnails.main`**

Run: `rg -n "thumbnails.main|clawmarks.build.thumbnails" src/ tests/` and update or remove any that called the old CLI/`main()`.

- [ ] **Step 7: Commit**

```bash
git add src/clawmarks/build/thumbnails.py src/clawmarks/curation_server.py tests/test_curation_server_lazy_thumbnails.py
git commit -m "feat(thumbnails): generate lazily on first request, drop the batch CLI"
```

---

## Task 20: Remove the `clawmarks build` CLI subcommand

**Files:**
- Modify: `src/clawmarks/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Read the current test coverage**

Run: `rg -n "build" tests/test_cli.py`. Note every test that exercises the `build` subcommand, `_BUILD_MODULES`, or `_ALL_TARGETS`.

- [ ] **Step 2: Update `tests/test_cli.py`**

Delete or rewrite each test found in Step 1 that asserted `build` is a valid subcommand or exercised `_build_target_main`. Add:

```python
def test_build_is_no_longer_a_valid_subcommand():
    import subprocess
    result = subprocess.run(
        ["python", "-m", "clawmarks.cli", "build", "scan"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "invalid choice" in result.stdout
```

- [ ] **Step 3: Run the new test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_cli.py::test_build_is_no_longer_a_valid_subcommand -v`
Expected: FAIL (the subcommand still exists today).

- [ ] **Step 4: Remove the subcommand from `cli.py`**

Delete `_BUILD_MODULES`, `_ALL_TARGETS`, `_build_target_main`, `build_parser()`'s `build_p = sub.add_parser("build")` block and its `--use-predicted-preference` argument, and the `if args.command == "build":` branch in `main()`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_cli.py -v`
Expected: PASS (all of `test_cli.py`, not just the new test).

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=src uv run pytest -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/clawmarks/cli.py tests/test_cli.py
git commit -m "feat(cli)!: remove the build subcommand, clawmarks serve now renders everything live

BREAKING CHANGE: 'clawmarks build <target>' and 'clawmarks build all' no longer exist.
Every tool page is now computed and rendered live by 'clawmarks serve' on each request."
```

---

## Task 21: Update the `run-clawmarks` skill and clean up dead static artifacts

**Files:**
- Modify: `.claude/skills/run-clawmarks/SKILL.md` (or wherever its content lives; confirm exact path with `fd run-clawmarks .claude/skills`)
- No test (documentation + data cleanup, not code)

- [ ] **Step 1: Rewrite the skill's dispatch section**

Find and remove the `clawmarks build <target>` bullet from the "Dispatch" section. Replace it with a short note that every tool page is served live by `clawmarks serve`, computed on request and cached until its underlying data changes, so there is no separate build step anymore. Keep the `clawmarks serve`, `clawmarks run allnight`, `clawmarks probe train`, and `clawmarks pod` bullets as they are (all untouched by this plan). Update the "Known bug" section's reference to `cli.py`'s `serve` dispatch if it still reads correctly; it does not need edits otherwise.

Grep the finished file for `—` and ` -- ` before finishing, per this project's writing-style rule.

- [ ] **Step 2: Delete dead static artifacts from the live sweep directories**

```bash
rm -f notes/uncanny_seedrun1/scan.html notes/uncanny_seedrun1/archive.html \
      notes/uncanny_seedrun1/scan_data.json notes/uncanny_seedrun1/lightbox.js \
      notes/uncanny_seedrun1/infotip.js notes/uncanny_seedrun1/scrollnav.js \
      notes/uncanny_seedrun1/coverage.html notes/uncanny_seedrun1/map.html \
      notes/uncanny_seedrun1/redundancy.html notes/uncanny_seedrun1/novelty_decay.html \
      notes/uncanny_seedrun1/lineage.html notes/uncanny_seedrun1/explore.html \
      notes/uncanny_seedrun1/seeds.html notes/uncanny_seedrun1/rate.html \
      notes/uncanny_seedrun1/preference_rank.html notes/uncanny_seedrun1/similarity.json \
      notes/uncanny_seedrun1/solution_map_data.json notes/uncanny_seedrun1/similarity_scored.json
```

Do not delete `scored_manifest.json`, `thumbs/`, `user_ratings.json`, `user_favorites.json`, `user_counterfactuals.json`, `candidate_seeds.json`, `counterfactuals/`, `job_map.json`, `manifest.json` in that directory. These are real inputs or user data, not build output.

Restart `clawmarks serve` against `notes/uncanny_seedrun1` and confirm `curl http://127.0.0.1:<port>/scan.html` (and each other route) returns `200` with the expected content, proving nothing depended on the deleted files being present on disk.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/run-clawmarks/SKILL.md
git commit -m "docs(run-clawmarks): remove build <target> dispatch, describe live serving"
```

(The `rm -f` in Step 2 touches `notes/`, which is gitignored build output in this sweep directory; nothing to `git add` for that step.)

---

## Self-Review Notes

- **Spec coverage:** `LiveCache` (Task 1), shared JS assets served directly (Task 2), `uncanny_gallery` scoring/rendering split (Task 3), all data-only dependency targets `similarity`/`solution-map` (Tasks 4-5, 7), every routed page (Tasks 6, 8-18), lazy thumbnails (Task 19), CLI removal (Task 20), skill doc + dead-artifact cleanup (Task 21). Out-of-scope items from the spec (ratings/favorites/counterfactuals/seeds API, actual scoring logic, multi-process serving) are untouched by every task above, matching the spec's "Out of scope" section.
- **`probe-report` target:** intentionally has no task here. Per `cli.py`'s existing comment, it's a fixed historical report that doesn't read `SWEEP_DIR` at all, so it was never part of "build all" and isn't part of "serve live" either; it stays a standalone script, unaffected by removing the `build` subcommand's *other* targets. Flag this explicitly to the user before Task 20 lands, since deleting `_BUILD_MODULES` entirely also removes the ability to invoke `probe-report` via `clawmarks build probe-report`; if that capability is still wanted, it needs its own tiny `if __name__ == "__main__":` entry point preserved directly in `probe_report.py` (a one-line addition, not part of this plan's scope, called out here so it isn't silently lost).
- **Type/name consistency check:** every `render_html` in this plan takes exactly the `data` (or `items`) value its sibling `compute_data` returns; every route handler in `curation_server.py` calls the pair by the same two names used within that module. `_get_scan_items`/`_get_solution_map_data` are the only cross-task helper names, defined once (Tasks 6, 7) and referenced by name in every later task that needs them (Tasks 8, 9).
