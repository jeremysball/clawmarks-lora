# Live Status and Incremental Recompute Design

**Status:** Draft for review

## Problem

The live-tool-pages work (docs/superpowers/plans/2026-07-10-live-tool-pages.md, merging as
PR #7) replaced `clawmarks build` with on-request rendering, cached in `LiveCache` and
invalidated by file mtime. Two gaps surfaced once real traffic hit it:

1. **No debounce, no partial recompute.** Any mtime change on a target's watched files
   triggers a full from-scratch recompute on the next request, even for the embedding-heavy
   targets (`scan`/`similarity_index`, `solution-map`, `gallery`/`uncanny_gallery`) where DINOv2
   embedding 100+ images takes tens of seconds to a minute. A burst of ratings within a few
   seconds can trigger that recompute repeatedly.
2. **No status while computing.** The first-ever request for an uncached target blocks the
   whole HTTP response until `compute_data` returns. The browser tab shows nothing until it's
   done: no indication anything is happening, no indication of how long it'll take.

Separately: the project has a RunPod serverless ComfyUI endpoint (`uix4vdb2cec7sb`,
`src/clawmarks/compute/comfyui.py`) that the search driver calls directly and that has been
found wedged before (lab_notebook.md, 2026-07-09 entry). There's currently no visibility into
its health from the web UI at all.

## Decisions (confirmed via brainstorming)

- **Progress detail:** granular, not just a spinner. Live text like
  `"embedding 64/100 images (2.2 img/s, ETA 0.3 min)"`, matching what already prints to the
  server's console today.
- **Transport:** polling a JSON `/api/status/<target>` endpoint, not SSE. Simplest to build
  on the stdlib `ThreadingHTTPServer`, and matches the client-side `fetch` pattern `rate.html`
  and `seed_browser.html` already use.
- **Staleness model:** stale-while-revalidate. A page that has been computed at least once
  always serves instantly, even if its data just went stale. The recompute happens in the
  background and a later request picks up the fresh result. Only a target's very first-ever
  computation blocks the browser (via the loading shell described below), since there's no
  "stale but usable" data to fall back to yet.
- **Incremental recompute scope:** only the three DINOv2-embedding-heavy targets
  (`scan`/`similarity_index`, `solution-map`, `gallery`/`uncanny_gallery`). The cheap
  manifest-scanning targets (`coverage`, `novelty_decay`, `lineage`, `elite_archive`,
  `preference_rank`) keep recomputing wholesale on every change: already sub-second, not
  worth the complexity.
- **RunPod status placement:** a small persistent strip in the shared nav bar
  (`shared_ui.py`'s `nav_bar_html`, already rendered on every tool page), showing ComfyUI
  endpoint worker/queue counts, polled periodically.

## Architecture

### 1. `LiveCache` becomes non-blocking after first compute

Today, `LiveCache.get()` blocks the calling (request-handling) thread on `compute_fn` every
time the cache is stale. This changes to:

- **Never computed yet:** spawn a background thread running `compute_fn`, return a sentinel
  meaning "not ready" to the caller. The route handler uses this to serve a **loading shell**
  page instead of the real one.
- **Computed and fresh:** return the cached data immediately, as today.
- **Computed but stale:** return the *stale* cached data immediately. If no background
  recompute is already running for this target, spawn one. If one is already running, don't
  spawn a second. This is the debounce: at most one recompute in flight per target, and a
  burst of file changes within that window collapses into whatever the recompute sees when it
  actually reads the files.
- **Recompute finishes:** swap in the fresh entry. If the data changed *again* while
  recomputing, the next request that notices staleness starts another recompute. There's no
  fixed time window, just "one at a time, always eventually catches up."
- **Recompute raises:** keep serving the last-known-good cached entry (if any); record the
  error in the target's status so `/api/status/<target>` surfaces it instead of polling forever
  on a target that will never finish.

`LiveCache` also gains a per-target status record: `{"state": "computing" | "done" | "error",
"stage": str, "detail": str}`, exposed via a new `status(target_name)` method. Route handlers
and the new `/api/status/<target>` endpoint both read this.

### 2. Progress reporting from inside `compute_data`

The three embedding-heavy modules (`similarity_index.py`, `solution_map.py`,
`uncanny_gallery.py`) gain an optional `report` callback parameter to their embedding loops,
called at the same points they already `print()` progress
(`"embedded {n_done}/{n} ({rate:.1f} img/s, ETA {eta_min:.1f} min)"`, `"loading DINOv2..."`,
`"fitting UMAP..."`). `LiveCache` detects (via `inspect.signature`) whether a target's
`compute_fn` accepts `report` and, if so, passes a bound function that updates that target's
status record thread-safely. Targets that don't accept `report` (the cheap ones) are called
exactly as before, with no signature change needed for them at all.

### 3. Loading shell + `/api/status/<target>`

When a route handler gets the "not ready yet" sentinel from `LiveCache.get()`, it returns a
small static HTML page instead of the real tool page: a `<div>` with placeholder text and a
`<script>` that polls `GET /api/status/<target>` roughly once a second, updating that div's
text from the JSON body's `detail` field. When the endpoint reports `"state": "done"`, the
script does `location.reload()`, which now hits the warm cache and renders instantly. On
`"state": "error"`, the script stops polling and shows the error text instead.

`/api/status/<target>` returns `{"state": ..., "stage": ..., "detail": ...}`, a thin read of
`LiveCache.status(target_name)` with no computation of its own.

### 4. Incremental recompute for the three embedding-heavy targets

`similarity_index.py` and `solution_map.py` already have checkpoint-resume logic
(`embed_with_progress`'s `checkpoint_file`): if a saved embedding tensor's path list is a
prefix of the current manifest's path list, embedding resumes from where it left off instead
of starting over. Today this only helps mid-run interruptions (the process died partway
through one embedding pass); it doesn't yet help the common live-serving case of "one image
got added or rated since the last full compute." This design wires the same mechanism into the
live-cache path: on recompute, `compute_data` first checks whether the current manifest is the
previous manifest plus new entries appended at the end (using the already-cached previous
result's known image list, threaded through as a new `previous_result` argument), and if so,
embeds only the new entries and reuses the rest, before rebuilding whatever aggregate
computation depends on the full embedding set (similarity join, UMAP projection). `gallery`/
`uncanny_gallery.py`'s DINOv2 scoring step gets the equivalent treatment.

If the manifest changed in a way that isn't a clean append (an entry removed, reordered, or a
scored value changed for an existing entry), this falls back to a full recompute, same as
today. Appends-only is the realistic common case (new images from ratings/generation), so this
covers the actual pain point without needing full diffing.

### 5. RunPod/ComfyUI status bar

New `src/clawmarks/compute/comfyui.py` function `health()` wrapping the existing
`api_get("/health")` call (RunPod serverless's health endpoint, already authenticated via
`API_KEY`), returning `{"workers": {...}, "jobs": {...}}`.

`curation_server.py` gets a small in-memory cache (plain TTL, not `LiveCache`, since this polls
an external API rather than invalidating on a local file) refreshed at most once every 15 seconds
regardless of how many page loads happen in between, so navigating between tool pages doesn't
hammer RunPod's API.

`shared_ui.py`'s `nav_bar_html` gains a small status strip populated by client-side JS polling
a new `GET /api/comfy-status` route every 15 seconds, rendering something like
`"● ComfyUI: 2 workers ready, 0 queued"` (or a red dot plus "unreachable" if the health call
itself fails). That failure case is exactly the "wedged endpoint" mode from the 2026-07-09
incident, so the status bar needs to surface it clearly, not just silently show stale numbers.

## Out of scope

- SSE/WebSocket transport (polling was the explicit choice).
- Incremental recompute for the cheap manifest-scanning targets.
- A fixed-time-window debounce; the single-flight-background-recompute model replaces it.
- Historical/logged status (this is live "right now" state only, not a dashboard with history).
- Changing how the search driver itself submits jobs to ComfyUI: this only adds a read-only
  health check, not retry/backoff logic for the submission path.
