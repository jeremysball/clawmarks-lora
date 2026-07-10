# Preference Classifier: Delegation Bootstrap Brief

**Plan:** `docs/superpowers/plans/2026-07-09-preference-classifier.md` (13 tasks)
**Spec:** `docs/superpowers/specs/2026-07-09-preference-classifier-design.md`

## Verdict

**~90% ready.** Tasks 1-12 pass the delegation rubric after one in-plan fix already applied
(Task 6's hedge and test-design bug, see Pre-resolve list). Task 13 does not belong to opencode
at all; it runs against real production data and needs live judgment throughout. Split the run
into two opencode phases plus one supervisor-only task, as below.

## Load order for the supervisor

**Load the `delegating-to-opencode` skill before doing anything else.** This brief only gates
and splits the plan; it does not replace that skill's run mechanics.

## Three-tier triage

### Opencode-delegatable (unattended) — Tasks 1-12

Every step in Tasks 1-12 is an exact find/replace or exact new-file creation with a runnable
`pytest` command and a literal expected result. Task 9's new `scikit-learn==1.6.1` pin was
verified this session via `uv pip install --dry-run "scikit-learn==1.6.1"` (resolved cleanly,
one package installed) — not a disqualifying unverified-version issue. No other task in 1-12
contains a reproduce-first step, iterative visual tuning, an external human prerequisite, or a
prose "replicate the pattern" instruction. Task 6 needed a fix before it qualified; see below.

### Supervisor-inline (mid-tier model, NOT opencode) — Task 13, in full

Task 13 ("Run the migration, verify end to end") does not belong to opencode, for four reasons:

- **Step 4** (build the embedding cache over all 3672 real images) is a long-running live compute
  job with no fixed completion time — opencode has no way to bound or verify it unattended.
- **Step 6** (smoke-test `rate.html` against a live server) requires reading a curl response and
  manually substituting a real tag from it into the next command — a live round-trip, not a
  fixed edit.
- **Step 8** (final commit) is explicitly a judgment call in the plan's own text ("if X is
  untracked/gitignored... leave it; otherwise commit") — the rubric's soft-target disqualifier.
- The task operates on **real production data**: the actual `user_picks.json` (40 real ratings)
  and the actual 3672-image manifest, not test fixtures. A live migration and a live embedding
  build over real data warrant a human-supervised hand, not an unattended one.

Run Task 13 yourself, in the supervisor session, after both opencode phases pass their own test
suites and after your own QA pass (below) on Tasks 1-12's diff.

### Human-only (parked)

None. Task 13's live steps are supervisor-inline, not human-only — the supervisor session (a
Claude session with shell access) can run curl and background a server itself. No step in this
plan requires credentials, third-party consoles, or DNS/infra the supervisor lacks access to.

## Run-splitting guidance

Two opencode phases, split at the dependency boundary:

- **Phase 1 — Tasks 1-8** (8 tasks, at the ~7-8 task/run ceiling). Rating infrastructure and
  embedding cache: manifest index, rating sampler, picks-to-ratings migration, curation server
  endpoints, rate.html, elite_archive Stage 5a wiring, driver.py Stage 5a wiring, embed_cache.py.
  No dependency bump in this phase — plain Go build/test throughout.
- **Phase 2 — Tasks 9-12** (4 tasks, comfortably under the ceiling). Model training and Stage 5b:
  Task 9 carries the `scikit-learn==1.6.1` pyproject.toml bump (run `uv add` for it as Task 9
  itself specifies, then Go); Tasks 10-12 build on it (preference_rank.html, driver.py Stage 5b,
  elite_archive.py Stage 5b). Launch Phase 2 only after Phase 1's PR has merged, since Tasks
  9-12 import modules Phase 1 creates.
- **Task 13 is not part of either opencode phase.** Run it yourself after Phase 2 merges.

## Scope note

Every `pytest` run in Tasks 1-12 is opencode's to run and report. Nothing in Tasks 1-12 requires
visual or live verification — they're pure-function unit tests and HTML-string assertions. The
one place a live/visual step could sneak in is Task 5's `rate.html` (a keyboard-driven UI): the
plan's own Task 5 test only checks the generated HTML string, not the live page, so opencode can
complete it unattended. Actually clicking through `rate.html` in a browser is Task 13's job (via
the supervisor), not opencode's.

## Integration path

**PR-gated. Default, no deviation.** Current branch is `clawmarks-package-transition` (confirmed
via `git branch --show-current`); it is ahead of `main` with no commits unique to `main`
(confirmed via `git log --oneline main..HEAD` / `HEAD..main`); remote `origin` points to
`github.com/jeremysball/clawmarks-lora.git`. Because `clawmarks-package-transition` — not
`main` — is this project's actual active integration line right now, each opencode phase works
on its own branch cut from `clawmarks-package-transition` (e.g.
`preference-classifier-phase-1`, `preference-classifier-phase-2`), and the supervisor opens a
PR back into `clawmarks-package-transition` after its own QA pass on that phase. Do not push
either phase's commits directly to `clawmarks-package-transition` or to `main`.

## Pre-resolve list

- **Done this session:** Task 6's Step 1 test rewritten to monkeypatch `N_BINS=1` (forcing both
  fixture images into one cell regardless of the file's quantile-based bin math) and to assert
  on an exact `capsys`-captured print string plus a regex-extracted, `json.loads`'d `CELLS`
  array, replacing the prior vague substring checks. Step 2's hedge ("if this doesn't fail
  clearly... the important thing is a failing test before the fix, not the exact assertion
  shape") is gone, replaced with the literal, specific reason the test fails pre-fix. No further
  pre-resolve items identified in Tasks 1-12.

## Final QA — explicit reminder

The supervisor's own QA pass on each phase's diff, before opening its PR, is mandatory and is
its own separate step — not opencode's self-reported "tests pass," and not this brief. Read the
diff, rerun the suite yourself, and only then open the PR into `clawmarks-package-transition`.

---

**This brief ends here.** Starting the supervised opencode cycle (loading
`delegating-to-opencode`, launching Phase 1) is the next action — it is yours to take, not
mine, per `preparing-plans-for-delegation`'s stop line.
