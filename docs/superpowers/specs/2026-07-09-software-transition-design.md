# CLAWMARKS tooling: transition from scripts to package

## Motivation

The uncanny-frontier search/curation tooling, the probe/calibration tooling, and the RunPod
bring-up scripts exist as 31 standalone files (23 under `notes/`, 8 at repo root) with no shared
packaging. Three concrete pains, roughly equal weight:

- **Path/config duplication**: 20+ files each define their own `SC = "/workspace/trent-with-smart-prompts"`
  and derive `SWEEP_DIR`/`OUT_DIR`/`PREV_DIR` from it. Moving the repo or running from a different
  checkout means editing every file.
- **No single entry point**: nothing to run except `python3 notes/<filename>.py`, in an order you
  have to remember (run a sweep, then `build_scan_gallery`, then `curation_server`, etc.).
- **No tests / fragile changes**: `shared_ui.py` is imported by 9 `build_*.py` scripts; a change to
  it currently can only be verified by clicking through the generated pages by hand.

## Scope

In scope: everything under `notes/*.py` (23 files: search driver, curation server, all
`build_*.py` generators, `shared_ui.py`, probe/calibration scripts, `mmd_score.py`), plus the
root-level RunPod scripts (`rp_bring_up.py`, `rp_bring_up2.py`, `rpget.py`, `rpget2.py`,
`rpsftp.py`, `rpsftp2.py`, `rpssh.py`, `rpssh2.py`).

Out of scope: root-level one-offs from earlier phases that aren't part of this tooling
(`clip_score.py`, `dino_score.py`, `epoch_sheet.py`, `gen_batch.py`, `perturb_captions.py`,
`train_compare_sheet*.py`). They stay where they are.

Compute-backend abstraction (RunPod today, a different cloud later) is explicitly **not**
built now: RunPod code gets tidied into one module with shared config, not hidden behind a
provider interface. Building that interface before a second provider exists is speculative work
with no way to validate the boundary is right; revisit when there's an actual second backend to
support.

## Package layout

```
src/clawmarks/
  config.py              # repo root resolution (env override, else Path(__file__) walk-up),
                          # all path constants (SWEEP_DIR, OUT_DIR, etc.) in one place
  compute/
    runpod.py             # GraphQL client, pod bring-up/pause/terminate, SSH/SFTP helpers -
                           # merges rp_bring_up.py/rp_bring_up2.py (the "2" suffix was only ever
                           # about running two pods concurrently, not different logic) and
                           # rpget/rpsftp/rpssh, parameterized by pod id instead of duplicated
    comfyui.py             # workflow submission: build_workflow/api_post/api_get from
                           # run_uncanny_sweep.py
  search/
    driver.py              # run_uncanny_allnight.py + run_uncanny_allnight2.py merged into one
                           # driver taking round-specific params (explore fraction, prior-round
                           # exclusion embeddings, budget cap) instead of two 90%-identical files
    scoring.py              # centroid/novelty math shared by the driver and score_probe_*.py
    seed_pool.py            # candidate_seeds.json load/merge/dedup - currently duplicated
                           # between curation_server.py and run_uncanny_allnight2.py, becomes
                           # one implementation both import
  probe/
    train.py                # train_probe.py
    sweep.py                 # probe_uncanny.py, probe_strength_sweep.py, gen_samples.py
  build/                     # one module per generator, 12 files, internals unchanged:
                           # scan_gallery.py, elite_archive.py, coverage_map.py, map_view.py,
                           # redundancy_view.py, novelty_decay.py, lineage_view.py,
                           # solution_map.py, similarity_index.py, thumbnails.py,
                           # explore_hub.py, seed_browser.py, probe_report.py
  shared_ui.py               # moved, internals unchanged
  curation_server.py          # moved; imports config.py instead of its own SC/SWEEP_DIR
  cli.py                      # argparse subparsers, the `clawmarks` entry point
pyproject.toml                # console-script entry point; deps pinned; `uv pip install -e .`
tests/
  test_scoring.py
  test_seed_pool.py
  test_generation_jobs.py
```

Files not named above (`merge_round2.py`, `build_probe_report.py`, `score_*.py`) move into the
subpackage matching their role (`search/`, `probe/`, `build/`) with internals unchanged, same as
every other pure "move."

## Data flow: path/config resolution

`config.py` exposes one function, `repo_root()`: reads `CLAWMARKS_ROOT` from the environment if
set, else walks up from `config.py`'s own file location to find the directory containing
`pyproject.toml`. Every other path constant (`SWEEP_DIR`, `OUT_DIR2`, `PROBE_DIR`, etc.) is
derived from `repo_root()` at import time, replacing the ~20 duplicated `SC = "..."` literals.
This is the only behavior change with any risk: every file that currently hardcodes the path
switches to importing `config.SWEEP_DIR` etc. Verified by running one full build cycle
(`clawmarks build all` against the existing `notes/uncanny_sweep` data) and diffing output
against the current script-generated files before considering the move done.

## CLI shape

`argparse` with subparsers (stdlib, no new dependency, sufficient at this scale):

```
clawmarks serve                      # curation_server.py, unchanged behavior
clawmarks build <name>                # one of the 12 generators, or:
clawmarks build all                   # run every generator in dependency order
clawmarks run allnight --round 1|2    # search/driver.py, round selects explore fraction etc.
clawmarks probe train ...             # train_probe.py passthrough
clawmarks pod bring-up / pause / terminate / ssh / get / put   # compute/runpod.py passthrough
```

Every subcommand's actual logic stays in its module (`search/driver.py`, `build/scan_gallery.py`,
etc.) as a plain function; `cli.py` only parses args and calls it. No behavior lives in `cli.py`
itself, so testing the underlying functions doesn't require going through argparse.

## Error handling

No new error-handling design: each moved module keeps its current error handling exactly
(timeouts, budget caps, retry loops already in the search driver and curation server carry over
unchanged). The only new failure mode this introduces is `repo_root()` not finding
`pyproject.toml` (e.g. run from outside the checkout) - it raises a clear `RuntimeError` naming
the env var override rather than failing on a confusing downstream `FileNotFoundError`.

## Testing

Unit tests for pure logic only, no GPU/network/filesystem mocking:

- `test_scoring.py`: `bin_of`, `bin_edges`, centroid similarity, novelty computation against
  known fixture embeddings.
- `test_seed_pool.py`: dedup (case-insensitive), merge, source tagging, both call sites
  (browser-triggered and driver-triggered) going through the same function.
- `test_generation_jobs.py`: explore/exploit split math, prompt construction, parent-tag
  propagation.

Everything else (RunPod calls, ComfyUI submission, HTML generation, the actual curation server)
is verified by the manual smoke check described in "Data flow" above, consistent with how this
project has verified UI/generation features all session: live-test, inspect actual output, don't
trust a self-report.

## Migration approach

Single PR, since every file move is mechanical and the only substantive change (path config,
driver merge, seed-pool dedup) is small and reviewable as one unit. Old `notes/*.py` and
root-level RunPod scripts are deleted in the same PR once the package versions are confirmed to
produce identical output - no parallel-old-and-new period, since keeping both around risks
someone editing the wrong copy.
