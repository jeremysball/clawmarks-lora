# Close All Open Threads Implementation Plan

> Execute this plan task by task with attended OpenCode development. Review every task before
> starting the next one. Human visual judgments and paid RunPod launches remain explicit gates.

## Goal

Bring `origin/main` up to the safety and correctness state already developed on unmerged branches,
resolve every actionable open GitHub issue and unchecked software task in `TODO.txt`, reconcile stale
designs with the current code, and leave only decisions that require Jeremy's visual judgment or
explicit approval for paid compute.

## Starting State

- Base: `origin/main` at `70dd9f6`.
- Phase branch: `feat/close-open-threads`.
- PR #27 merged the generation cockpit.
- PRs #22 and #26 remain open and conflict with current `main`.
- Open issues #12 through #21 and #23 through #25 need reconciliation. Issue #25's eight findings
  landed in PR #27 but the issue remains open.
- `notes/lab_notebook.md` has an uncommitted July 13 seed-generation entry in the main worktree.
- Two complete `notes/uncanny_seedrun1.backup_candidate_seeds_*` mirrors remain untracked. Preserve
  them. Never restore them over the live sweep because they predate later votes and seed files.
- No on-demand RunPod pods exist. Endpoint `uix4vdb2cec7sb` has two non-running workers, zero billed
  seconds in the last three hours, and a user approval that expires July 14.

## Global Constraints

- Data integrity outranks all other goals. Never run a write-capable command against
  `notes/uncanny_sweep`, `notes/uncanny_sweep2`, `notes/uncanny_seedrun1`, or real training data
  without a complete mirror backup and a verified file-count/content comparison first.
- Never delete a cache before recomputing it. Compute to a temporary sibling, validate the complete
  result, then replace atomically with `os.replace`.
- Never restore an old mirror over a directory that has received new votes, seeds, images, or model
  files since the mirror was taken.
- Use `uv` for Python dependency work. Pin every dependency version.
- Keep secrets in `.envrc` or runtime environment variables. Never write tokens into source files.
- Use Conventional Commits. Each task commits only its listed files.
- Run targeted tests while iterating. Run `uv run pytest -q`, `uv run ruff check .`, and
  `uv run mypy src` before final review.
- UI tasks require a live server restart and Playwright checks at desktop and mobile widths.
- Statistical claims must satisfy the `statistics-rigor` checklist. Record the independent unit,
  attainable p-value floor, multiplicity family, effect estimate, and uncertainty. Never report an
  empirical permutation p-value of zero.
- Do not launch paid generation or training from this plan until Jeremy approves the named resource,
  estimated cap, and duration at the corresponding gate.
- Preserve current plain-Python, server-rendered architecture. Add no JS framework, database, or
  background service.
- Do not modify or remove unrelated user changes in the original worktree.

## Task 1: Recover Pairwise-Model Correctness Fixes

**Issues:** #12, #13 and the usable-count half of the July 12 deferred follow-up.

**Files:**
- `src/clawmarks/search/preference_pairwise_model.py`
- `src/clawmarks/search/comparison_sampler.py`
- `src/clawmarks/curation_server.py`
- `src/clawmarks/build/preference_status.py`
- `tests/test_preference_pairwise_model.py`
- `tests/test_comparison_sampler.py`
- `tests/test_preference_status.py`

**Implementation:**
1. Replay the behavior from commits `ee6e95d`, `4d44642`, `69f9dd4`, and `bd0a4e9` onto current
   `main`; resolve conflicts against the cockpit code instead of accepting either side wholesale.
2. Treat one unordered image pair as the cross-validation group so mirrored rows never cross folds.
3. Consolidate repeated judgments on the same pair before training. Preserve net direction and count,
   but never count mirrored or repeated rows as independent validation units.
4. Exclude already-judged unordered pairs from active resampling.
5. Make status and retrain gates report both raw comparison count and usable unique-pair count. Gate
   training on usable pairs.
6. Add assertions in tests that every pair group appears in exactly one fold and duplicated rows do
   not inflate effective sample size or accuracy.

**Verify:**
`uv run pytest -q tests/test_preference_pairwise_model.py tests/test_comparison_sampler.py tests/test_preference_status.py tests/test_curation_server_compare_routes.py`

**Commit:** `fix(preference): restore pairwise validation integrity`

## Task 2: Recover Fresh-Install and Round-Two Fixes

**Issues:** #14 and #19.

**Files:**
- `pyproject.toml`
- `uv.lock`
- `clip_score.py`
- `src/clawmarks/build/merge_round2.py`
- covering tests

**Implementation:**
1. Replay the behavior from `468a74c` and `cc9f668`.
2. Add the exact pinned Paramiko dependency chosen by the existing fix and regenerate `uv.lock` with
   `uv`, not pip.
3. Use `.pooler_output` for the installed Transformers API.
4. Replace the invalid `torch_maximum` call with the tested PyTorch API.
5. Preserve current package and Docker constraints while resolving lockfile conflicts.

**Verify:**
`uv sync --extra dev && uv run pytest -q tests/test_merge_round2.py tests/test_config.py`

**Commit:** `fix(deps): restore fresh install and round-two scoring`

## Task 3: Recover Search Resume and Spending Guards

**Issues:** #15 and #16.

**Files:**
- `src/clawmarks/search/driver.py`
- `src/clawmarks/compute/comfyui.py`
- `tests/test_driver_state.py`
- related compute tests

**Implementation:**
1. Replay `60915ff`, `3048764`, `0807557`, and `6582ad0` onto the current driver.
2. Resume from persisted state and manifest without truncating either file.
3. Validate state/manifest agreement before submitting another job. Fail closed on malformed or
   mismatched persisted data.
4. Fail closed when balance lookup fails. Never substitute zero spend after a failed lookup.
5. Cancel every timed-out or abandoned serverless job and surface cancellation failures in the final
   summary.
6. Write state and manifest through temporary siblings plus `os.replace`.

**Verify:**
`uv run pytest -q tests/test_driver_state.py tests/test_comfyui.py`

**Commit:** `fix(search): restore resumable fail-closed execution`

## Task 4: Recover Stored-XSS Fixes

**Issues:** #17 and PR #26.

**Files:**
- current live build modules under `src/clawmarks/build/`
- `src/clawmarks/shared_ui.py`
- `tests/test_*` for every changed renderer

**Implementation:**
1. Replay the live behavior from `ebe6a93` and `911c1dd`, adapting it to the current source tree.
2. Do not resurrect deleted `uncanny_gallery.py` or its route.
3. Escape `</script` in JSON embedded inside script elements.
4. Use `textContent` or a shared `escapeHtml` helper for every manifest-, prompt-, tag-, sampler-, and
   model-controlled string inserted into HTML.
5. Add payload tests containing `</script><script>`, quotes, ampersands, and an image `onerror` value
   for every renderer family.

**Verify:**
`uv run pytest -q tests/test_shared_ui.py tests/test_scan_gallery.py tests/test_compare_page.py tests/test_cockpit.py`

**Commit:** `fix(security): escape model-controlled UI content`

## Task 5: Make Server Binding Configurable

**Issue:** #23 and PR #22.

**Files:**
- `src/clawmarks/curation_server.py`
- `.env.example`
- relevant server tests

**Implementation:**
1. Port `5eff2b4` and `c276634` onto current `main`.
2. Bind to `CLAWMARKS_HOST` when set. Default to the current safe tailnet behavior, but preserve an
   explicit `0.0.0.0` option for the Docker/Tailscale sidecar deployment.
3. Document host and port behavior in module help and `.env.example`.
4. Test explicit host, default host, and Docker override without opening a real socket.

**Verify:**
`uv run pytest -q tests/test_curation_server.py tests/test_config.py`

**Commit:** `fix(server): make bind host explicit`

## Task 6: Make Recovery Tools Transactional

**Issue:** #18.

**Files:**
- `src/clawmarks/search/score_manifest.py`
- `src/clawmarks/build/merge_round2.py`
- new focused tests

**Implementation:**
1. Split each script into pure planning/computation functions and a final commit step so tests can
   exercise failure points without touching generation data.
2. `score_manifest` must retain missing-image manifest entries and report them. It may score present
   entries, but it must never silently delete metadata because a file is temporarily unavailable.
3. Build the complete scored manifest and real-reference document in memory, write temporary siblings,
   validate row counts and tags, then atomically replace outputs.
4. `merge_round2` must load and validate both manifests and the existing embedding cache before writing.
5. Compute new embeddings and construct the merged cache first. Validate exact path order and tensor
   row count. Only then atomically replace both cache and manifest.
6. If either pre-commit step fails, leave both original files byte-for-byte unchanged. Tests must inject
   a failure between computation and commit and prove this invariant.
7. Real-data smoke checks are read-only unless a complete mirror backup has just been made and verified.

**Verify:**
`uv run pytest -q tests/test_score_manifest.py tests/test_merge_round2.py`

**Commit:** `fix(recovery): commit manifests and embeddings atomically`

## Task 7: Correct the Statistical Record and Power Analysis

**Issues:** #20 and #21.

**Files:**
- `notes/mmd_score.py`
- `notes/probe_power.py` (new)
- `notes/lab_notebook.md`
- focused statistical tests

**Implementation:**
1. Change the MMD Monte Carlo p-value to `(b + 1) / (B + 1)`, where `b` counts shuffled statistics
   at least as extreme as observed. Add a test proving the minimum is `1 / (B + 1)`.
2. Create a deterministic power-analysis module that enumerates all `2^n` sign flips for small `n`
   instead of sampling duplicate sign patterns. Compute and print the attainable one-sided p-value
   floor before simulating power.
3. Use paired deltas as the independent unit. Preserve the eight canonical training seeds as the
   planned units; do not treat prompt rows or mirrored deltas as independent.
4. Run at least 10,000 null simulations and verify the rejection rate is near the nominated alpha when
   rejection is attainable. Run positive controls at the prespecified 0.05 and 0.08 effects.
5. Retract the impossible significance/power entries at `n=3` and `n=4`. Replace the old table with
   reproducible results from the corrected program. If `n=8` no longer reaches 80% power, revise the
   round-one design rather than preserving the prior decision.
6. Replace the statement that cumulative best novelty proves reinforcement was "real, not noise."
   Describe the observed trajectory as exploratory and selection-biased. Add a per-generation cohort
   statistic or an untouched replay comparison before making any reinforcement claim.
7. Add a dated notebook entry with the correction, impact on prior conclusions, code/data provenance,
   and open limitations.

**Verify:**
`uv run pytest -q tests/test_mmd_score.py tests/test_probe_power.py && uv run python notes/probe_power.py`

**Commit:** `fix(stats): correct permutation tests and exploratory claims`

## Task 8: Move Preference Retraining Off the Request Lock

**Thread:** July 12 deferred head-to-head follow-up.

**Files:**
- `src/clawmarks/curation_server.py`
- `src/clawmarks/build/preference_status.py`
- `src/clawmarks/build/compare_page.py`
- preference route tests

**Implementation:**
1. Add one process-local retrain coordinator with `idle`, `running`, `succeeded`, and `failed` state.
2. Save the comparison under `_lock`, decide whether a retrain is due, release `_lock`, then start one
   daemon worker. Never hold `_lock` during model fitting.
3. Reject duplicate manual retrain starts with HTTP 409 while one worker runs.
4. Keep the previous model active until the new model and metadata have atomically replaced it.
5. Expose retrain state and last error through `/api/preference_status`; let both compare and status
   pages poll while training runs.
6. Base progress and gate copy on usable unique pairs from Task 1, while still displaying raw votes.

**Verify:**
`uv run pytest -q tests/test_curation_server_compare_routes.py tests/test_curation_server_preference_status_route.py tests/test_preference_status.py tests/test_compare_page.py`

**Commit:** `fix(preference): retrain outside the request lock`

## Task 9: Finish Detail View and Compare Follow-Ups

**Threads:** detail-view spec tasks 11 through 14.

**Files:**
- `src/clawmarks/shared_ui.py`
- `src/clawmarks/build/compare_page.py`
- `src/clawmarks/build/map_view.py`
- `src/clawmarks/curation_server.py`
- `src/clawmarks/build/thumbnails.py`
- related tests

**Implementation:**
1. Replace compare's separate zoom overlay with its corner affordance calling
   `Lightbox.open(current.imgN.tag)`. Stop propagation so inspection never records a vote.
2. Keep generated counterfactuals as side tools. Do not inject them into the active pair.
3. Accept integer `n` on `/api/counterfactual`, default 1, clamp 1 through 6, perform one fail-closed
   balance check, then submit sequential jobs with distinct random seeds unless a seed is pinned.
4. Return `{ok: true, results: [...]}` and update every in-repo client and test in the same task.
5. Render an `n` stepper and clickable result grid in the Lightbox.
6. Add `mountProgressive(img, thumb, full)` with a request token so a late full-image load can never
   replace a newer selection. Reconcile this with the existing visibility-gated prefetch cache rather
   than issuing duplicate full-resolution requests.
7. Add `/real_thumbs/<name>` with basename sanitization and on-demand cache under
   `SWEEP_DIR/real_thumbs`. Never write into `corrected_dataset_extract`.
8. Use progressive loading for the Lightbox main image and map's nearest real image.
9. Keep the existing detailed `accuracy_tip`; mark TODO task 13 complete after a live touch check.
10. When pairs exhaust, show links to `cockpit.html` and `seeds.html` with concrete copy that suggests
    generating unfamiliar candidates before comparing again.

**Verify:**
`uv run pytest -q tests/test_compare_page.py tests/test_shared_ui.py tests/test_curation_server.py tests/test_curation_server_lazy_thumbnails.py`

**Live check:** compare inspection never votes; N generation uses mocked network in local verification;
progressive images do not race during rapid navigation; desktop and 390px mobile layouts work.

**Commit:** `feat(curation): finish shared detail and generation flows`

## Task 10: Add Safe Search-Run Launching and Reports

**Thread:** overnight-search launch design plus per-run report.

**Files:**
- new `src/clawmarks/search/run_manager.py`
- new `src/clawmarks/build/run_status.py`
- `src/clawmarks/curation_server.py`
- `src/clawmarks/shared_ui.py`
- `src/clawmarks/build/explore_hub.py`
- `src/clawmarks/search/driver.py`
- focused tests

**Implementation:**
1. Put backup, verification, lock ownership, process spawning, status parsing, and stopping in
   `run_manager.py`; keep HTTP handlers thin.
2. Before every launch, create a complete sibling mirror of the selected output directory and verify
   file count plus `filecmp.dircmp` content equality. Fail closed before process spawn.
3. Check RunPod balance once before spawn and fail closed on lookup errors or floor breach.
4. Use an atomically-created lock containing PID, process start time, round, output directory, and
   launch parameters. Verify PID start time to prevent PID-reuse mistakes.
5. Spawn the driver in its own process group and return immediately. Redirect logs to a named file
   outside irreplaceable image directories.
6. Status combines live process identity with persisted driver state. A stale lock reports `crashed`
   and remains available for diagnosis until explicitly acknowledged.
7. Stop sends SIGTERM to the process group, waits a bounded grace period, then SIGKILLs only that
   verified process group. Report both stages.
8. Add `runs.html` with launch, poll, stop, and completed-run report views. Reports include generation
   count, per-generation cohort novelty, cumulative-best novelty labeled as descriptive only, spend,
   plateau events, pick rate, and explore/exploit split.
9. Accept budget and duration only within server-side bounds. Never let the browser choose an arbitrary
   command or output directory.
10. Tests use temporary directories and fake processes. Never point automated tests at a real sweep.

**Verify:**
`uv run pytest -q tests/test_run_manager.py tests/test_run_status.py tests/test_curation_server_searchrun_routes.py`

**Live check:** use a fake driver command and temporary output directory. Do not submit RunPod jobs.

**Commit:** `feat(search): add guarded web launch and run reports`

## Task 11: Reconcile Backups, Notebook, TODO, PRs, and Issues

**Files:**
- `.gitignore`
- `TODO.txt` in the original worktree only (gitignored, never commit)
- `notes/lab_notebook.md`

**Implementation:**
1. Delete `docs/superpowers/specs/2026-07-11-toml-config-design.md` and
   `docs/superpowers/specs/2026-07-11-ui-redesign-design.md`. Jeremy rejected both designs on July 13;
   do not replace or implement them.
2. Copy the uncommitted July 13 notebook entry from the original worktree into this branch after
   confirming no newer notebook entry supersedes it.
3. Record every recovered issue fix, statistical correction, feature completion, and verification in
   dated notebook entries as each task lands. Do not wait until the end.
4. Ignore `notes/*.backup_candidate_seeds_*/` so complete mirrors can remain on disk without polluting
   git status. Do not delete or restore either mirror.
5. Remove the rejected TOML and UI-redesign entries from `TODO.txt`. Update the remaining list from
   actual merged state. Mark branch protection and CI image build complete; both succeeded on July 13.
6. After the phase PR merges, close superseded PRs #22 and #26 with a comment naming the replacement
   commit. Close each fixed issue with exact tests and commit references. Keep any partially resolved
   issue open with a reduced checklist.
7. Verify clean status in all auxiliary worktrees. Remove a worktree only after its commit is confirmed
   present in `main`; otherwise preserve it.

**Verify:**
`git status --short`, `gh pr list --state open`, and `gh issue list --state open` match the remaining
human/paid gates below.

**Commit:** `docs: reconcile project state and completed work`

## Gate A: Human Preference Validation

These tasks cannot be inferred from accuracy numbers or Playwright:

1. Serve `preference_rank.html` against the current comparison model and give Jeremy a verified link.
2. Ask whether the ranking matches his taste and whether Stage 5b should remain off, collect more
   comparisons, or turn on.
3. Serve round 2's 280 images through the archive browser and ask Jeremy to review them before any
   round 3 decision.
4. Record both decisions in `TODO.txt` and the notebook. Only then change the Stage 5b setting.

## Gate B: Paid Search and Training

Before each resource mutation, run the RunPod account guard and present resource, cap, and duration.

1. Ask Jeremy to approve or revise the eight-direction round-one slate. Resolve `cycles1`: run it at
   780 steps or remove it from probe screening because a 260-step run is not one full cycle.
2. Ask approval for the canonical-seed control batch: eight 260-step RTX 4090 probes.
3. Run, score, and notebook-log controls before directions. Use the same eight seeds for every direction.
4. Run only the approved directions. Apply exact paired sign-flip tests from Task 7, practical effect
   threshold, and prespecified multiplicity handling.
5. Ask separately before the full 780-step commit run and before each later round. Pause idle pods after
   downloads and verify local files before any pause.
6. Ask separately before an explore-heavy round 3 serverless run, with a hard dollar cap and the new
   launcher from Task 11.

## Gate C: Near-Term Reveal

1. Ask Jeremy whether a usable reference photo of the artist exists and where it is stored.
2. If supplied, write a separate generation brief choosing img2img/IP-Adapter or text-only likeness.
3. Present RunPod cost cap before generation.
4. Back up and verify the target output directory, generate, score only as a floor check, curate by eye,
   and record the result in the notebook.

## Final Verification and Delivery

1. Run `uv sync --extra dev`.
2. Run `uv run ruff check .`.
3. Run `uv run mypy src`.
4. Run `uv run pytest -q`.
5. Build the Docker image through CI; local Docker remains unavailable in this sandbox.
6. Run live desktop/mobile Playwright checks for every changed UI flow and inspect browser console.
7. Dispatch an independent whole-branch correctness, data-safety, security, and statistical review.
8. Fix every blocking or important finding and repeat targeted plus full verification.
9. Open a PR, wait for `check` and `verify`, merge only when both pass, then confirm the merge commit is
   on `origin/main` before removing the phase worktree.
