# Generation cockpit: an interactive trial workbench

Design only, not an implementation plan.

## Where this design came from

Five independent LLM brainstorms fed the synthesis below: Claude Fable 5, GPT-5.6-sol (short
pass), GLM-5.2, and GPT-5.6-terra each answered the same open brainstorm prompt cold, with no
cross-contamination. GPT-5.6-sol then read all four and produced a ~6,200-word synthesis at max
reasoning effort, explicitly asked to find prior art beyond image-generation tools: adaptive
experimentation (Ax/BoTorch), active learning, design of experiments (JMP), quality-diversity
search (pyribs/MAP-Elites), experiment tracking (W&B/DVC), OSF-style preregistration, and
electronic lab notebooks (Benchling). That synthesis's central idea, that the page's real unit of
work is a **trial** (mission, hypothesis, recipe, forecast, outcome) rather than a bare prompt
form, is what this spec builds from.

Everything below has been re-grounded against this codebase. Two of the synthesis's working
assumptions turned out to be wrong once checked against the actual server code, which reshapes
the design in ways worth stating up front.

## What already exists (verified in code, not assumed from memory)

**`/api/counterfactual` is already a from-scratch text-to-image generator, not img2img.**
`build_workflow()` (`curation_server.py:222`) always starts from `EmptyLatentImage` at
`denoise: 1.0`; there is no `VAEEncode` node reading an existing image anywhere in the workflow.
`origin_tag` is pure bookkeeping (used to build the output tag and stored in the record for
lineage), not an image the generation is conditioned on. "Strength" is the LoRA's
`strength_model`/`strength_clip` on `LoraLoader`, not an img2img denoise amount. This means the
cockpit's "Freeform" and "Develop a candidate" missions need **no new generation backend at
all**: they call the existing endpoint with a prompt and no `origin_tag`-derived meaning beyond
optional lineage tagging. All four brainstorms and the synthesis assumed this was img2img-style
variation; it isn't.

**There is no pre-generation prompt-scoring model anywhere in this codebase.** The preference
model (`search/preference_pairwise_model.py`) is Bradley-Terry-style logistic regression trained
on *differences between DINOv2 image embeddings*. It scores images, not text. The only text
encoder in the project is the `CLIPTextEncode` node inside the ComfyUI generation workflow itself
(conditioning for the diffusion model, not a scoring model). `coverage_map.py`'s faithfulness/
novelty grid and `elite_archive.py`'s per-cell champions are likewise computed from
`centroid_sim`/`novelty`, both derived from an *already-generated* image's DINOv2 embedding
compared against the real-photo centroid and prior generations
(`search/driver.py:score_batch`). **Nothing in this codebase can take a draft prompt string and
predict where it will land or how it will score before the image exists.** All five brainstorms
(Fable, sol, GLM, terra, and sol's own synthesis) describe a live "predicted preference of this
recipe" readout that updates as the user types. That is not implementable today without building
a new prompt-to-embedding proxy model, which is a real ML project in its own right (e.g. a
sentence-transformer or the workflow's own CLIP text tower, cosine-mapped into the DINOv2
preference space, then validated against held-out generations) and explicitly out of scope for
this UI-layer spec. See "Explicitly deferred" below.

**What *can* be computed cheaply, because it only touches existing images:**
- `search/embed_cache.py:embed_paths()` embeds a single new image on demand (no batch minimum);
  `search/driver.py:score_batch()` turns one or more fresh embeddings into real `centroid_sim`
  (faithfulness) and `novelty` figures using the already-cached real-photo centroid and prior
  generations. Once a trial's images actually exist, they can be scored for real within the same
  request that saved them, no separate offline `python -m clawmarks.search.embed_cache` batch
  step required.
- `preference_pairwise_model.score()` needs only an embedding row and the trained model, so
  scoring a freshly generated image against the user's trained taste is equally cheap once it
  exists.
- What "which coverage cell does my *target* sit in, and what's already there" needs is just the
  existing `coverage_map.compute_data()` grid, unchanged. That's a legitimate, already-built
  ambient-knowledge source for the left "Mission" column.
- So the honest v1 forecast is **evidence about the neighborhood, not a live score of the draft**:
  nearest already-generated images with a similar subject string, the target coverage cell's
  current occupancy, and (for "Develop a candidate") whether this exact seed subject has been
  tried before. That's less than what all five brainstorms pictured, but it's real and buildable
  today, and it's still meaningfully more than a blank prompt box.

**`coverage.html`'s grid is faithfulness x novelty, not UMAP embedding space.** That's a distinct
page and a distinct axis system from `map.html` (the actual UMAP "solution map"). GLM and terra's
brainstorms both said "void map" loosely enough to blur this; this spec uses "coverage grid" to
mean specifically `coverage_map.py`'s 8x8 faith/novelty bins (same grid `elite_archive.py` uses
for its per-cell champions), and reserves "solution map" for the separate UMAP page. The cockpit's
"Fill a coverage gap" mission targets the coverage grid, not the UMAP map.

**An automated overnight search loop already exists and does something adjacent to this.**
`search/driver.py` (`clawmarks run allnight`) already picks subjects/textures, batches
generations, and scores them into the same coverage grid and elite archive, unattended. The
cockpit is the interactive, human-driven sibling of that loop, not a replacement for it: both
read and write the same coverage grid, elite archive, and seed pool, but the cockpit is for a
person steering one deliberate trial at a time, not for kicking off an unattended batch. This
spec doesn't touch `driver.py`.

**Lineage tracking is currently split across two unrelated mechanisms.** `search/driver.py`'s
batch generations set `parent_tag` on manifest entries, which `lineage_view.py` reads to build
its tree. The interactive counterfactual flow (`curation_server.py`) uses a separate `origin_tag`
field on `user_counterfactuals.json` records, and lineage_view.py never looks at it. "Continue a
lineage" as a cockpit mission means bridging both: an origin image can be either a manifest entry
(from a driver.py batch) or a counterfactual record. This spec treats that bridge as a small,
explicit lookup (try the manifest by tag, fall back to `user_counterfactuals.json`), not a data
model unification; unifying the two lineage mechanisms is a separate concern this spec doesn't
take on.

**The n-variation extension to `/api/counterfactual` doesn't exist yet.** The prior detail-view
spec (`2026-07-12-detail-view-and-generation-design.md`) proposed it; as of this writing
`_handle_counterfactual` still submits exactly one job per request. The cockpit's batch-of-n
primary action depends on this landing first (or being folded into the same work), since driving
n sequential single-image requests from cockpit.html client JS would duplicate that server-side
batching logic instead of reusing it.

## The core object: a trial

Adopting the synthesis's central idea, scoped to what's actually computable per the corrections
above. A trial is:

```json
{
  "trial_id": "t_1720900000",
  "mission": "fill_gap | develop_candidate | continue_lineage | freeform",
  "hypothesis": "optional one-line free text: what the user is testing",
  "target": {"fb": 3, "nb": 6, "void_label": "sparse: weathered owl portraits on pale paper"},
  "recipe": {
    "prompt": "...", "negative": "...", "seed_strategy": "random|fixed",
    "seeds": [123, 456], "strength": 1.0, "cfg": 7.5, "steps": 28, "sampler": "ddim", "n": 4
  },
  "parent_tag": "gen15_explore_22_seed609298",
  "candidate_subject": "weathered owl portrait, pale paper, ink wash",
  "status": "draft | queued | running | completed | failed",
  "created_at": "...",
  "started_at": "...",
  "completed_at": "...",
  "outputs": [
    {"tag": "cf_...", "file": "counterfactuals/cf_....png", "centroid_sim": 0.71,
     "novelty": 0.42, "fb": 3, "nb": 6, "preference_score": 1.8, "decision": "keep|reject|null"}
  ],
  "conclusion": "optional free text, appended after review"
}
```

Persisted at `SWEEP_DIR/trials.json`, a flat `{trial_id: {...}}` dict following the exact pattern
`user_favorites.json`/`user_counterfactuals.json` already use (`load_store`/`save_store` in
`curation_server.py`), so no new persistence mechanism is needed.

**What this keeps from the synthesis:** the mission/hypothesis/recipe/outcome shape, the idea
that a launched trial is immutable (edits after `queued` create a new trial, not a mutation), and
that outputs link back to their trial for review.

**What this drops from the synthesis, and why:** no revision graph, no Pareto-frontier candidate
slate, no response-profiler sensitivity preview, no separate acquisition-strategy math. Those all
assume either a working forecast model or a volume of trial history this project doesn't have yet
(the preference model itself refuses to train below 50 comparisons; a useful sensitivity
readout would need far more trial volume than that). Building UI for statistics the project can't
yet produce would be decoration, not function. These are candidates for a later revision once
real trial-history volume exists; see "Explicitly deferred."

## Layout

Three columns plus a bottom drawer, closely following the synthesis's structure but with each
column's contents corrected to what's actually derivable.

### Left: Mission

Four mutually exclusive entry points, each populated from data that already exists:

- **Fill a coverage gap**: pulls `coverage_map.compute_data()`'s frontier cells (already
  computed: empty cells adjacent to a well-populated one). Show 2-3, each with its faith/novelty
  range in plain language (reuse the existing `axes_tip` framing from `coverage_map.py`) and its
  nearest occupied cell's top image as a thumbnail. Selecting one sets the trial's `target` and
  suggests a `candidate_subject` if any unused seed pool entry's history places it nearby (a
  simple heuristic for v1: prefer seed pool entries never used in a completed trial yet, no
  embedding-similarity ranking, since there's nothing to embed a subject string against without
  the missing prompt-to-embedding bridge).
- **Develop a candidate**: a one-card-at-a-time deck from `seed_pool.load()`
  (`SWEEP_DIR/candidate_seeds.json`), same data `seeds.html` already shows. Needs one small
  addition to `seed_pool.py`: a way to tell whether a subject has already been used in a
  completed trial (cheapest approach: scan `trials.json` for a `candidate_subject` match at
  render time, no new persisted field). Use / Skip / Show another.
- **Continue a lineage**: recent elites (`elite_archive.py`'s per-cell champions) and recent
  counterfactuals, selecting one sets `parent_tag`. The prompt editor starts pre-filled with the
  parent's prompt so the diff is visible from the start, not inferred after the fact.
- **Freeform**: no suggestion, no forced target. Still gets the same recipe editor and evidence
  columns everyone else gets; the point of this mission is that the system stays out of the way.

### Center: Recipe

- Trial brief at top: mission, target (if any), an optional single-line "what are you testing?"
  free-text field. Never required, but pre-filled with a plain-language suggestion when the
  mission implies one (e.g. "Test whether this candidate subject fills the {region} gap").
- Prompt textarea, large, primary. Reference thumbnails (parent / nearest elite / coverage
  neighbor, whichever apply) sit directly above it with a one-word role label each, reusing the
  Lightbox's existing similarity-strip visual pattern rather than inventing a new one.
- Visible controls: seed strategy (random vs. fixed vs. explicit list), batch size `n` (default
  4, capped 6 to match the existing counterfactual-endpoint plan), LoRA strength.
- Collapsed "Advanced" drawer: CFG, steps, sampler, negative prompt. Collapsed state always shows
  the active values as one line (`DPM++ 2M · 28 steps · CFG 7.5`), so nothing is hidden, only
  deferred. Reuses `info_btn` for a one-time explanation of what CFG/steps/sampler mean, matching
  every other page's approach to unfamiliar vocabulary.
- Below the recipe, a compact **evidence band** (see right column) and a cost/time estimate
  computed from the last N actual `_handle_counterfactual` runs' wall-clock time, not a fixed
  guess: `SWEEP_DIR/trials.json`'s own history is the source once it has enough completed trials;
  until then, fall back to `GENERATION_TIMEOUT_S`'s documented cold-start figure (~215s) as a
  conservative estimate.
- Primary action: **Generate {n} images** (not "4 probes" by default: n is visible and edited
  right above the button, so a separate "probes" vocabulary would just be a second name for the
  same number). Disabled with an inline reason if `RUNPOD_API_KEY` is unset or the balance floor
  check (`BALANCE_FLOOR_USD`) would reject it, checked the same way `_handle_counterfactual`
  already checks it, surfaced before the click instead of after.

### Right: Evidence

Explicitly not a forecast, since nothing here predicts the unmade image. It answers "what do we
already know nearby":

- **Nearest existing images**, by real text-embedding distance, not substring match. The
  ComfyUI workflow already loads a CLIP text encoder (`CLIPTextEncode`) for conditioning; the
  same encoder can embed every past prompt once (cached, same pattern as `embed_cache.py`) and
  embed the current draft on each debounce tick, both cheap CPU/GPU-idle operations with no
  RunPod job involved. Show the 3-5 closest past prompts by cosine distance, each with its actual
  outcome (score, novelty, kept/rejected) if it has one.
  - **Novelty flag, not a score.** Alongside the neighbor list, show the raw cosine distance to
    the single nearest neighbor as a plain badge: "closest prior prompt: 0.91 similarity" down to
    "no similar prompt tried before" below some threshold (tune empirically once real prompt
    history exists; start conservative). This is the honest version of "quantify uncertainty":
    it doesn't predict a score for the draft, it only tells you whether you're in territory the
    system has evidence about or territory it has none for. That distinction is real,
    computable today, and doesn't require the DINOv2-preference-space mapping the deferred
    proxy model needs. Text-to-text distance is a much smaller, already-available task than
    text-to-image-embedding-then-score.
- **Target coverage context** (only when a target cell is set): that cell's current occupant
  count, its current champion thumbnail if any, and its faith/novelty range, straight from
  `coverage_map.compute_data()`.
- **Candidate history** (only for "Develop a candidate"): has this subject been tried before, and
  if so, what did the resulting images score.
- **Cost and time estimate**, as above.

No dominant single score, no gauge, no percentile band, because there is no live per-draft score
to display honestly. This is the biggest visible departure from all five brainstorms: GLM
explicitly wanted a dominant preference gauge, terra wanted a safer/surprising slider tied to a
live forecast, sol's synthesis wanted a four-part live forecast (preference/novelty/coverage/
confidence). All of that is real, good design for a *future* version once a prompt-to-embedding
bridge exists; shipping fake or misleadingly-precise numbers now would be worse than shipping
none.

### Bottom drawer: Queue and Results

Two tabs over `trials.json`:

- **Queue**: draft and queued trials, each showing mission, hypothesis, recipe summary, and
  n/cost/time. A queued trial's snapshot is frozen at commit time (matches the synthesis's
  preregistration framing); editing after queuing creates a new trial rather than mutating the
  queued one, so the eventual outcome can always be compared against what was actually predicted
  at commit time, not a retroactively edited version of it.
- **Results**: completed and failed trials, grouped by trial rather than as a flat image stream.
  Each output image shows its *real, computed* `centroid_sim`/`novelty`/coverage cell and
  preference score (computed on save via `embed_paths()` + `score_batch()` +
  `preference_pairwise_model.score()`, all reused as-is), plus Keep/Reject/Compare/Archive-
  candidate/Locate-on-map actions routing to the existing pages. An optional one-line "what did
  this teach you?" free-text field on the trial itself, never required.

Failed trials stay visible with their failure reason (`_handle_counterfactual`'s existing error
paths: balance floor, submit failure, timeout, job FAILED/CANCELLED) rather than disappearing, so
a stalled or refused generation doesn't quietly vanish from the record.

## New server surface

- `GET /cockpit.html`: new page module `build/cockpit.py`, following the existing
  `render_html(data)` / `nav_bar_html()` pattern every other tool page uses.
- `GET /api/trials`: returns `trials.json` for the queue/results drawer.
- `POST /api/trial`: creates a trial record (mission, hypothesis, target, recipe) at `status:
  "queued"`, no generation yet. Separates "I've committed to this plan" from "the GPU is now
  running," matching the synthesis's queue-then-run split.
- `POST /api/trial/<id>/run`: runs the trial's `n` generations (reusing
  `_handle_counterfactual`'s submit-and-poll loop, generalized for n>1 per the pending n-variation
  work), scores each output on save, and updates the trial record to `completed` or `failed`.
- `POST /api/trial/<id>/decide`: records a per-output keep/reject decision and/or the trial-level
  conclusion text.

All four reuse `load_store`/`save_store`, the existing `_lock`, and the existing balance-floor and
timeout handling verbatim; nothing here needs new generation infrastructure, only new
orchestration around the existing one.

## What to explicitly avoid

Carried forward from the brainstorms where they agreed, since the underlying reasons hold
regardless of the forecast-model correction above:

- No auto-generation on keystroke or on page load; every paid run is an explicit click with a
  visible n/cost/time.
- No score-threshold gate that blocks a low-evidence trial; evidence is advisory, never a veto.
- No silent prompt rewriting; suggestions insert visible, editable text.
- No sliders for CFG/steps/strength; numeric fields with steppers.
- No full interactive coverage grid or UMAP map embedded in the page; both stay one click away on
  their existing pages, with only a small static crop shown here.
- No flat, ungrouped image stream in the results drawer; every output stays attached to its
  trial.
- Editing a queued or completed trial in place; corrections become a new trial.

## Explicitly deferred (not this spec)

- **A prompt-to-preference-score proxy model**, which is the actual prerequisite for the live
  numeric forecast panel every brainstorm pictured ("this draft will probably score 0.8"). Would
  need its own design: how a text embedding maps into the existing DINOv2 preference space, how
  it's validated against real generations before being trusted enough to display. A real ML
  project, not a UI addition. The nearest-neighbor novelty flag above is deliberately *not* this:
  it never predicts a score, only whether evidence exists, which is a much smaller and
  already-computable claim. Confidence-interval or ensemble-based versions of the full proxy model (see
  the "will look confidently precise while quietly being wrong" discussion this spec grew out of)
  stay deferred until there's enough trial history to calibrate them honestly.
- **Named acquisition strategies / Pareto-frontier candidate slates** (borrowed from Ax/BoTorch
  in the synthesis): meaningful once there's a working forecast to rank candidates by; without
  one there's nothing to rank.
- **Response-profiler-style sensitivity previews** (borrowed from JMP): needs substantially more
  trial history than this project will have for a long while; premature before real volume exists.
- **Unifying `parent_tag` and `origin_tag` into one lineage mechanism**: real cleanup, but
  orthogonal to shipping the cockpit; the bridge lookup described above is enough for v1.
- **A structured "recipe chip" prompt grammar** (subject/setting/composition/material/palette/
  mood as separate fields): sol's synthesis itself argued against this becoming a hidden syntax
  that drifts from manual edits; if built at all, it should insert plain editable text, never
  become the source of truth over the prompt string.

## Open questions for whoever plans the implementation

- Should `/api/trial/<id>/run` block until all `n` generations complete (simplest, matches
  `_handle_counterfactual`'s current synchronous style) or return immediately and let the client
  poll, given `n` sequential jobs at up to ~215s cold-start each could mean several minutes per
  trial? The existing counterfactual flow is already a synchronous wait-for-it interaction with
  its own timeout, so blocking is consistent, but a 6-image trial could hit
  `GENERATION_TIMEOUT_S` per job well past what a browser request should sit open for.
- Should the seed-pool "has this been used" check (for "Develop a candidate") get a persisted
  field on the seed pool entry itself, or stay a scan over `trials.json` at render time? The scan
  is simpler and can't drift out of sync, but gets slower as `trials.json` grows; probably fine
  for the trial volumes this project will realistically produce, worth confirming before building.
- Does "Continue a lineage" need to search *all* of `scored_manifest.json` plus
  `user_counterfactuals.json` for candidate parents, or only recent/favorited ones? Searching
  everything could be slow once the manifest is large; the existing elite-archive and favorites
  data are natural, already-fast starting shortlists.
