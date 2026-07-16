# Expedition / leg generation model

## Motivation

The `allnight` search driver hardcodes exactly two generation runs, round 1 and round 2, as a
fixed `ROUND_CONFIGS` dict keyed by an integer. Round 1 explored the "uncanny frontier" image
space from scratch; round 2 generalized round 1's job-building logic and excluded round 1's
images from its novelty scoring. Both rounds' full-resolution images are gone for good (see the
lab notebook's 2026-07-09 and 2026-07-14 entries), and the project has no further use for that
specific two-round exploration. But the underlying capability, launching a generation run and
pooling its output for future novelty scoring, is still exactly what the project needs going
forward. This design replaces the fixed round 1/round 2 pair with an open-ended model: any number
of named generation runs, grouped into named projects, each startable from the web UI without a
code change.

## Vocabulary

- **Expedition**: a themed container for one line of work, e.g. `uncanny_frontier`. Holds shared
  prompt vocab and budget defaults, plus a pool of every image any of its legs has generated.
  Starting a genuinely new direction means creating a new expedition, so its pool starts empty.
- **Leg**: one generation run within an expedition. Either an automated overnight `allnight`
  search or a manual cockpit session. Every leg's output feeds its expedition's shared pool.

## Storage layout

Config, checked into git:

```
expeditions/
  <expedition>/
    expedition.json       # shared defaults
    legs/
      <leg>.json          # per-leg overrides and specifics
```

Runtime output, outside the repo under `$XDG_STATE_HOME/clawmarks/` (`config.py`'s `STATE_DIR`):

```
$XDG_STATE_HOME/clawmarks/expeditions/
  <expedition>/
    <leg>/
      allnight_state.json      # allnight legs only
      scored_manifest.json
      thumbs/
      real_thumbs/
      seed_pool.json
```

Every expedition gets a standing `cockpit` leg, scaffolded automatically at creation time. Manual
one-off trials from the cockpit UI land there rather than needing their own name.

## Creation flow

Creating an expedition is explicit and happens through the curation server web UI: a form for the
shared defaults (trigger word, negative prompt, textures, fallback subjects, default budget caps
and margins, default batch size, explore fraction, and generation cap) writes `expedition.json`
and scaffolds the `cockpit` leg directory. There is no bare CLI command for this step.

Launching a leg (`clawmarks run allnight --expedition <name> --leg <name>`) into an expedition
name that does not exist is a clear, immediate error, never a silent auto-create. A typo in the
expedition name must not quietly start a new, empty expedition.

## Config merge rule

A leg's effective configuration is `expedition.json`'s fields, overridden field-by-field by
anything present in that leg's own `legs/<leg>.json`. A leg file can be empty (inherit everything)
or override only the fields that make it different, such as `explore_fraction`, `seed_from_start`,
or `gen_batch_size`.

## Image pooling and novelty exclusion

When scoring a leg's generations for novelty, the driver (and the cockpit's scoring path) loads
every *other* leg's `scored_manifest.json` within the *same* expedition as the "already explored"
exclusion set, in addition to the real training-set centroid, which stays global across
expeditions since it represents the actual training data, not generated output. A brand-new
expedition therefore starts with an empty exclusion pool: nothing outside the real training set
counts as already explored until some leg in that expedition has actually run.

This replaces round 2's single hardcoded "exclude round 1" relationship with "exclude every
sibling leg," which also correctly generalizes to a third, fourth, or later leg in the same
expedition.

## Code changes

### `src/clawmarks/search/driver.py`

- Replace the `RoundConfig` dataclass and `ROUND_CONFIGS` dict with a `LegConfig` built by
  merging `expedition.json` and `legs/<leg>.json` at load time.
- Delete the `allow_legacy_round1_baseline` compatibility shim and every code path that checks
  for it (`_validate_state`, `_validate_resume_agreement`, and their call sites). It existed only
  to tolerate the very first round-1 run's state-file quirk; no data in that old format survives,
  and every new leg starts from one consistent state format.
- Replace `_load_prev_round_state`'s single-predecessor lookup with a function that lists every
  sibling leg directory under the current expedition, loads each one's `scored_manifest.json`
  if present, and concatenates their images into one exclusion set.
- Replace the `--round {1,2}` CLI argument with `--expedition <name> --leg <name>`, both required
  strings.

### `src/clawmarks/config.py`

- Remove `SWEEP_DIR` and `SWEEP2_DIR`.
- Add `EXPEDITIONS_DIR` (repo-relative, e.g. `<repo_root>/expeditions`) for config files.
- Add `leg_dir(expedition, leg)`, returning
  `STATE_DIR / "expeditions" / expedition / leg` for runtime output.

### `src/clawmarks/search/run_manager.py`

- Change `launch_run(round_num, out_dir, api_key, ...)` to
  `launch_run(expedition, leg, out_dir, api_key, ...)`.
- Change the lock file's stored fields from `{"round": round_num, ...}` to
  `{"expedition": expedition, "leg": leg, ...}`.
- Change the subprocess invocation from `["...", "driver", "--round", str(round_num)]` to
  `["...", "driver", "--expedition", expedition, "--leg", leg]`.
- Simplify `build_report`'s state-file lookup from trying `allnight_state.json` then
  `allnight2_state.json` to reading `allnight_state.json` only, since every new leg uses one
  consistent filename.

### `src/clawmarks/curation_server.py`

- Replace every `ROUND_CONFIGS`/`SWEEP_DIR`/`SWEEP2_DIR` reference with expedition/leg-aware
  directory resolution via `config.leg_dir`.
- Replace the empty-state hub's hardcoded "Launch Round 1" / "Launch Round 2" buttons with an
  expedition picker (list existing expeditions, each showing its legs) plus a "create expedition"
  form per the creation flow above.
- Add an expedition selector to the cockpit page. Cockpit trials write into the selected
  expedition's `cockpit` leg; the cockpit's scoring context (real centroid, exclusion embeddings)
  is built from that expedition's pooled legs rather than the current fixed `SWEEP_DIR`.
- Change `/api/searchrun/launch` and the launch-status/report endpoints to accept
  `{"expedition": ..., "leg": ...}` instead of `{"round": ...}`.

### Deletions

- `src/clawmarks/build/merge_round2.py` and `tests/test_merge_round2.py`. This script's only job
  was migrating the original round1-to-round2 handoff (merging round 1's manifest and embeddings
  into round 2's exclusion set). That handoff no longer exists as a special case; the new
  "exclude every sibling leg" logic in `driver.py` supersedes it.

### Migration of round 1 / round 2 as a reference expedition

No image data moves, since none of round 1 or round 2's full-resolution images survive. Create
`expeditions/uncanny_frontier/expedition.json` from round 1's shared defaults (trigger word,
textures, fallback subjects, budget caps and margins), and
`expeditions/uncanny_frontier/legs/round1.json` and `legs/round2.json` capturing what
distinguished each leg (`explore_fraction`, `seed_from_start`, `gen_batch_size`, `max_generations`,
round 2's widened vocab). This preserves the parameter record as a worked example for the next
expedition, without implying there is any usable image data behind it.

### Tests to update

`tests/test_cli.py`, `tests/test_run_manager.py`, `tests/test_config.py`, and
`tests/test_curation_server_solution_map_dep.py` currently assert on round-numbered behavior
(`--round`, `ROUND_CONFIGS`, `SWEEP_DIR`/`SWEEP2_DIR`) and need updating to the expedition/leg
vocabulary and the new merge/exclusion logic.

## Out of scope

- Any UI mockup or exact wording for the expedition-creation form; that is an implementation
  detail for the plan and its execution, not this design.
- Automated migration tooling. Since no image data survives to migrate, `uncanny_frontier`'s
  config files are written once by hand as part of implementation, not by a script.
