# Live tool pages: retire `clawmarks build`, serve everything live

## Problem

Every CLAWMARKS tool page (scan, archive, map, coverage, redundancy, novelty-decay, lineage,
similarity, explore-hub, seeds, uncanny-gallery, preference-rank, rate) is a static HTML file
written once by `clawmarks build <target>`, with its data baked in as a `const DATA = [...]`
blob. Ratings, favorites, counterfactuals, and seeds are already live (API-backed in
`curation_server.py`), but the pages themselves, and anything that comes from
`scored_manifest.json` (categories, faithfulness/novelty scores, prompt metadata), go stale the
moment the manifest changes upstream. The only fix today is remembering to rerun `clawmarks
build <target>` (or `build all`) by hand. This surfaced directly: scan.html's category labels
were wrong, and the fix was "the file is stale," not a labeling bug.

The command/rebuild step doesn't match how this tool is actually used: it's operated as a
running server (`clawmarks serve`) that someone actively interacts with (rating images,
generating counterfactuals) while a separate offline process (search rounds, manifest scoring)
changes the underlying data. A page should reflect the current manifest the moment it's
requested, not the manifest as of the last manual build.

## Decision

Retire `clawmarks build <target>` and `build all` entirely. `curation_server.py` computes every
page's data live, on request, cached in memory and invalidated by input-file mtimes. There is no
more static HTML artifact sitting in the sweep directory as the source of truth; the served
response is generated directly from `scored_manifest.json` and friends every time they change.

## Architecture

### Module split

Each `src/clawmarks/build/*.py` module splits into two pure pieces, replacing its current
`main()`:

- `compute_data(sweep_dir) -> data`: reads whatever inputs the page needs
  (`scored_manifest.json`, `similarity.json`, `user_ratings.json`, etc.) and returns a plain
  Python data structure (dict/list of dicts). No file writes. No side effects.
- `render_html(data) -> str`: the existing HTML/CSS/JS template logic, now taking `data` as an
  argument instead of reading it from a just-written JSON file or embedding it via file I/O.

`main()` and any direct `scan_data.json`-style disk writes are deleted from each module.

### `LiveCache`

A new small class in `curation_server.py` (or a new `clawmarks/live_cache.py`), one instance per
running server, holds:

```
{
  target_name: {
    "data": <last computed data>,
    "watched_mtimes": {path: mtime, ...},   # snapshot at last compute
  }
}
```

Each build module declares two module-level constants:

- `WATCHED_FILES = ["scored_manifest.json", ...]`: paths (relative to `sweep_dir`) whose mtime
  invalidates this target's cache entry.
- `DEPENDS_ON = ["solution-map"]` (optional, default `[]`): other target names this target's
  `compute_data()` needs the (possibly-cached) output of. `solution-map` writes data that `map`
  and `redundancy` both read; encoding this as an explicit dependency replaces the old
  hand-maintained run-order dict in `cli.py` (`_BUILD_MODULES`'s "dict order is also build all's
  run order" comment goes away along with the dict).

`LiveCache.get(target_name)`:
1. Resolve `DEPENDS_ON` recursively first (each dependency goes through the same cache logic).
2. Compare each of `WATCHED_FILES`'s current mtime against the cached snapshot. If any differ
   (or there's no cache entry yet), call `compute_data(sweep_dir)`, passing resolved dependency
   data in if the module's signature needs it, store the result and new mtimes.
3. Return the (possibly just-recomputed) data.

A `threading.Lock` per target name (a small dict of locks, created lazily) wraps step 2-3, so two
concurrent requests hitting a stale, expensive target (e.g. `uncanny-gallery`, which imports
torch/transformers) don't both trigger a recompute; the second request blocks briefly and then
reads the first request's fresh cache entry instead of recomputing independently.

### Server routing

`curation_server.py`'s `do_GET` gains a routing table: `{"scan.html": "scan", "archive.html":
"archive", ...}` mapping request paths to target names. On a match: `data = live_cache.get(name)`,
`html = render_html(data)`, respond with `200` and `Content-Type: text/html`. Everything else
(images, thumbs, JS assets not covered below) falls through to the existing static serving, same
as today.

### Special cases

- **`thumbnails.py`**: doesn't fit the data/render split, it resizes source images to
  `thumbs/<tag>.jpg` on disk. This stays a lazy per-file generator, not a cached data compute: the
  server checks whether `thumbs/<tag>.jpg` exists when an image is requested and generates it
  on first miss. Resized thumbnails don't go stale once made (a source image doesn't change after
  generation), so there's no cache-invalidation question here, just "generate once, on demand."
- **`lightbox.js` / `infotip.js` / `scrollnav.js`**: pure behavior, not data: these already
  live as Python string constants in `shared_ui.py` (`_LIGHTBOX_JS`, `INFOTIP_JS`,
  `SCROLLNAV_JS`). Instead of `write_*_asset(sweep_dir)` writing them to disk at build time, the
  server serves them directly from those constants via a small dedicated route
  (`/lightbox.js`, `/infotip.js`, `/scrollnav.js`). This removes the on-disk copy step entirely,
  which is also where the `\2039` template-literal bug and the write-asset staleness both lived.

### CLI changes

`cli.py` loses the `build` subcommand, `_BUILD_MODULES`, `_ALL_TARGETS`, and
`_build_target_main`. `clawmarks --help` after this change shows only `serve`, `run`, `probe`,
`pod`. The `run-clawmarks` skill's dispatch section (which currently documents `clawmarks build
<target>` as its first bullet) gets rewritten to remove that path and describe the live-serving
model instead.

### Migration of existing sweep directories

Static files left over from the old build step (`scan.html`, `archive.html`, `scan_data.json`,
`lightbox.js`, `infotip.js`, `scrollnav.js`, etc., in any sweep directory including
`notes/uncanny_seedrun1/` and `notes/uncanny_sweep/`) become dead output once the server no
longer reads or writes them. They're safe to delete; nothing depends on their presence anymore
(the server generates the response directly, it doesn't read these files back).

## Error handling

If a target's `compute_data()` or `render_html()` raises (missing input file, malformed JSON,
whatever), `do_GET` catches it, logs the traceback to the server's own log (already happens for
existing exceptions per the existing log lines), and returns a `500` with the exception's text
as the plain-text body. This is a single-operator internal tool: a raw error is strictly more
useful for debugging than a generic error page, and there's no external user to shield from
implementation detail.

## Testing

- **Unit tests per module**: `compute_data()` is now a pure function (fixture manifest in,
  data structure out), directly testable without any file-serving or HTML-rendering machinery.
  Reuses the `tests/fixtures/sample_sweep/` fixture built for the preference-classifier work
  (Task 13, 2026-07-10 lab log entry): trimmed 100-entry manifest, real thumbnails, no full-res
  images needed since `compute_data()` never touches image bytes.
- **Integration test**: start `curation_server.py` against the fixture sweep dir, `GET` a page,
  mutate the fixture's `scored_manifest.json`, `GET` the same page again, assert the response
  changed with no rebuild step run in between.
- **Cache-actually-caches test**: instrument (or count calls to) `compute_data()` for a target
  whose `WATCHED_FILES` did *not* change between two requests, assert it was called exactly once
  across both requests: this is the test that would have caught "cache silently does nothing,
  we recompute every request" regressions, which defeats the whole point of caching the
  torch-backed targets.
- **Dependency resolution test**: for `map`/`redundancy` (which `DEPENDS_ON = ["solution-map"]`),
  assert that touching `solution-map`'s watched input invalidates both `solution-map`'s own cache
  entry and, transitively, `map`/`redundancy`'s.

## Out of scope

- Changing the ratings/favorites/counterfactuals/seeds API surface: already live, untouched by
  this work.
- Any change to the actual data/scoring logic inside each `compute_data()`: this is a plumbing
  change (build-time to request-time), not a rewrite of what any page computes or displays.
- Multi-process or multi-worker serving. `ThreadingHTTPServer` with in-process, in-memory caching
  assumes a single server process per sweep directory, same as today.
