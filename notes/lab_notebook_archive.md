# CLAWMARKS LoRA: Lab Notebook Archive (entries before 2026-07-18)

Older lab-log entries moved out of `notes/lab_notebook.md` to keep that file scannable. This is a straight relocation, not a summary: nothing was cut or condensed, only moved. The live notebook (`notes/lab_notebook.md`) still holds sections 1-5 (background, methodology, experiment design, open questions, project reference) and the lab log from 2026-07-16 onward. See that file first; come here only for the earlier dated history (2026-07-08 through 2026-07-15).

---

## Archived lab log

### 2026-07-08: Notebook started, ledger merged in
Design finalized: probe-then-commit search, 5 rounds, directional probes double-replicated
against a measured noise floor, data-side adjustments allowed between rounds, DINOv2 centroid
similarity as the metric throughout. The separate `LEDGER.md` file folded into this notebook's
Section 5 so the project has one running record instead of two. Round 1's noise-floor probes
have not started.

### 2026-07-08: External methodology review drafted
Wrote `reviews/glm_review.md` critiquing `methodology.md` per `reviews/review_prompt.md`. Main
findings to weigh before round 1 runs: the n=2 replicate / n=3 control design cannot estimate
within-direction variance, so the selection rule should become a real two-sample test (Welch's t
or permutation) with controls pooled across rounds; ~10 probes per round at a loose bar needs a
BH correction to avoid spurious wins; 156-step probes finish under one cosine cycle and likely
misrepresent 780-step behavior, so one calibration round checking probe-vs-commit rank
correlation is worth running first; centroid cosine is dragged by the 0.22 outlier and rewards
style collapse, with nearest-neighbor or MMD in DINOv2 space as cheaper fixes; the validation
set used to pick epoch 4 is now being reused to score new configs (double-dipping), and seed
variance likely exceeds prompt variance so the 30-image grid should favor more seeds. Also
flagged: data changes and hyperparameter changes should stay on separate rounds so improvement
stays attributable, and the final human review should rank by a small panel, not the DINOv2
ranking known to disagree with human preference. Review kept under 600 words, no em dashes, ends
with `=== DONE ===`.

### 2026-07-08: External methodology review drafted
Created `reviews/gpt_review.md`, a concise ML-methodology critique of `methodology.md` using the
instructions in `reviews/review_prompt.md`. The review flags weak uncertainty estimates from only
2 replicates, possible 2-epoch to 10-epoch reversal, centroid-metric compression, early commit
risk in sequential search, and the need for holdout prompts plus immediate inspection of the
0.22-similarity training-image outlier.

### 2026-07-08: Probe-length calibration check underway, real training pipeline validated
Brought up the RTX 4090 training pod (`rp_bring_up.py`) and ran the four 156-step probes for
step 1's calibration check: `control` (baseline config), `dim64` (network dim 64 / alpha 32),
`lr2e4` (unet_lr 2e-4), and `constlr` (constant schedule instead of cosine, chosen because a
156-step probe finishes under one of the full run's 780-step / 3-cycle cosine schedule, the
gap this calibration check exists to catch). `control_156`'s final loss (0.109) closely matched
the historical epoch-4 winning run's final loss (0.106), which is real confidence the pipeline
(dataset, checkpoint, hyperparameters) is faithfully reproducing the known-good baseline before
trusting any of the three candidate directions' results. Generated 4 sample images per probe
checkpoint (same 4 prompts, same seed 42, same 28-step DDIM settings) with kohya's own
`sdxl_gen_img.py` directly on the pod, as a visual sanity check before scoring; contact sheet at
`notes/probe_samples/index.html`. No DINOv2/MMD scoring run yet on any checkpoint.

Brought up a second pod (`rp_bring_up2.py`, helper scripts `rpssh2.py`/`rpget2.py`/`rpsftp2.py`,
kept separate from the first pod's `rpssh.py` etc. so both stay independently reachable) to run
the three remaining 780-step full-length runs two at a time instead of serially. `dim64_780`
finished on pod 1 (final loss 0.110). `control_780` (so the baseline has a full-length twin too,
not just the three candidates) and `lr2e4_780` are running now, one per pod; `constlr_780` queued
next on whichever pod frees first.

Clarified the actual plan for step 3's statistical test before running it for real: the noise
floor isn't an assumption, it has to be measured from the pooled control probes' pairwise score
deltas (same fixed prompt/seed slots as every other comparison), and that measured spread is
also what determines how many replicates round 1 needs, via simulating the permutation test at
a few candidate n and checking which one reliably detects a 0.02-cosine injected effect. Neither
has been computed yet; both need the first batch of scored control probes as input. See the
methodology note added to Section 3, step 3, above.

### 2026-07-09: Calibration check (step 1) result: probe-length rankings disagree with full-length rankings

All four directions now have both a 156-step and a 780-step checkpoint. Generated 4 sample
images per checkpoint (fixed prompts, seed 42) with kohya's `sdxl_gen_img.py` and scored every
one against the DINOv2 centroid (`notes/score_probe_samples.py`, scores in
`notes/probe_samples/scores.json`; contact sheet `notes/probe_samples/index.html`).

Centroid-similarity means (n=4 images each):

| direction | 156-step | 780-step |
|---|---|---|
| control | 0.4634 | 0.5159 |
| dim64   | 0.3996 | 0.4164 |
| lr2e4   | 0.4973 | 0.4844 |
| constlr | 0.4995 | 0.3993 |

156-step ranking (best to worst): constlr, lr2e4, control, dim64.
780-step ranking (best to worst): control, lr2e4, dim64, constlr.

Two disagreements, one small and one large. `control` and `lr2e4` swap the #1 spot between
lengths, a ~0.03 gap each way, plausibly inside ordinary noise, though no noise floor has been
measured yet to confirm that. `constlr` swings from **best at 156 steps to worst at 780
steps**, a ~0.10 reversal, the largest gap in the whole table and far larger than any plausible
noise floor given the range these numbers occupy. This is the exact failure mode the calibration
check exists to catch: a constant learning rate looks strong early (still training at full
strength), but never decays the way the cosine schedule does, and by 780 steps that lack of
decay has cost it real quality rather than helped it. `dim64` stayed in last place at both
lengths, the one point of full agreement, and the gap to the other three is large enough (0.06+)
to trust regardless of noise floor.

Verdict, per the methodology's own decision rule (Section 3, step 1): **rankings do not agree,
so 156-step probes cannot be trusted to pick a winner for round 1.** They can still be trusted
to rule out a catastrophically bad direction, the way `dim64` was correctly identified as worst
at both lengths. Practical consequence for round 1's real probe phase (step 2): treat probe
results as a coarse screen, not a ranking to hand to step 3's significance test directly. Any
direction that clears the screen still needs a full 780-step run before being trusted as a
genuine improvement over control, which raises the real GPU-hour cost of round 1 versus the
original plan.

Caveat: this comparison itself is not yet statistically tested. Each checkpoint has one seed
and 4 fixed prompts, no replicate seeds, so there is no measured noise floor to compare the
observed gaps against, only a judgment call that a ~0.03 gap is small and a ~0.10 reversal is
large relative to the score range in this table. Measuring the real noise floor (task in
progress, needs pooled control-only replicates) would let the control/lr2e4 swap specifically be
called noise or real, rather than shrugged at.

### 2026-07-09: Initial noise-floor estimate recorded, later superseded by paired-seed analysis

Trained two more `control_156` replicates (`controlB_156`, `controlC_156`), identical config to
`control_156`, different random seed only (`train_probe.py` never pins a training seed, so
re-running the same config naturally gives an independent replicate). Generated the same 4
fixed-prompt samples for both (`notes/gen_samples.py`, a new reusable script that reconstructs
the exact `sdxl_gen_img.py` invocation used for every other checkpoint this round: seed 42, 28-step
DDIM, scale 7.5, 1024x1024, same 4 prompt lines from `/tmp/art_prompts_base_v2.txt`) and scored
all three against the centroid.

Per-image centroid similarity, same config, 3 independent seeds:

| prompt | control | controlB | controlC | range | stdev |
|---|---|---|---|---|---|
| cat | 0.4995 | 0.3797 | 0.4310 | 0.1197 | 0.0601 |
| horse | 0.3023 | 0.2243 | 0.0982 | 0.2041 | 0.1030 |
| tiger | 0.5370 | 0.5264 | 0.6141 | 0.0877 | 0.0479 |
| wolf-cat | 0.5146 | 0.4817 | 0.4615 | 0.0531 | 0.0268 |

Checkpoint-mean spread across the 3 replicates: 0.4634, 0.4030, 0.4012 (stdev 0.0354, max
pairwise diff 0.0621).

Two things stand out. First, the **horse prompt is dramatically noisier than the other three**
(stdev 0.103, more than double cat's 0.060 and nearly 4x wolf-cat's 0.027), likely because
"galloping horse" is further from the training distribution (all 31 real images are cats) than
the other three prompts, so its generations land less predictably seed to seed. Second, and
more consequential: **the checkpoint-mean noise floor (stdev ~0.035, max observed swing 0.062)
is bigger than the 0.02-cosine effect-size floor the methodology had assumed**, meaning that
threshold was never something a real probe-phase comparison could reliably clear, at any
practical replicate count.

The initial planning simulation used 4000 trials and 2000 sampled sign patterns per trial. It did
not enumerate the finite sign-flip space, did not expose the attainable p-value floor, and treated
the old unpaired noise estimate as if it were the final paired design. Its table is superseded by
the deterministic correction recorded on 2026-07-13 below.

No design decision should use that table. The current decision and reproducible replacement are
recorded in the 2026-07-13 correction entry below.

Also worth flagging for later interpretation: this noise-floor estimate itself comes from only
3 replicates (2 degrees of freedom), so it is a rough estimate with real uncertainty of its
own, not a precise population parameter. Revisit it once round 1's pooled control probes (8 more
replicates) accumulate: the true floor could turn out somewhat higher or lower once more data
exists.

Terminated pod 2 (`9e64aw56psou89`) once `constlr_780` finished downloading; only pod 1
(`cn0zudkxb89or6`) is running now.

### 2026-07-09: Both pods checked, found idle, terminated; pod-idle policy revised to pause-first

Checked both training pods (`cn0zudkxb89or6` and `iv8iannf63g3cf`) after `control260A`-`H`
finished: no training process running on either (`ps aux` clean), 0% GPU utilization on both,
and both hosts' local downloads matched the full expected 8-file `control260*` set, confirming
neither pod had unfinished work. Terminated both via the RunPod GraphQL `podTerminate` mutation
and confirmed via `query { myself { pods { id } } }` returning an empty list. No harm done this
time (all data was already downloaded), but the intended policy going forward is **pause, not
terminate**, when a pod finishes a batch and isn't about to start another one soon. Updated the
`runpod-status` skill and this project's `CLAUDE.md` to default to `podStop` (pause: keeps disk,
drops to storage-only billing) over `podTerminate` (destroys the pod and disk, changes SSH
host/port on the next one), and to check pod idle state proactively rather than waiting to be
asked. See Section 5's infra note above for the mechanics.

### 2026-07-09: Scored control260A-H; external methodology review (Opus and Fable) on round 1's plan

Ran `notes/score_probe_samples.py` against the 8 finished `control260A`-`H` checkpoints (see
Section 3, step 2's note above for the number: stdev 0.0279, not worse than the 156-step
0.035). This was the one real open risk in carrying the n=8/0.05-cosine floor forward from the
156-step derivation without re-measuring at 260 steps; it's now checked and held.

Sent the in-progress round-1 plan (methodology through step 3, plus the still-open candidate-
direction question) to two independent external reviewers for a second opinion before locking
anything in: Opus (high reasoning effort) and Fable, each given the same background and asked to
review independently, with Fable additionally shown Opus's reply and asked to react to it plus
think creatively about a side research question (see Section 3b). Neither reviewer had access to
this notebook or any project files; both worked from a self-contained prompt. Findings folded in
rather than kept as a separate critique file, per this project's existing convention (see Section
3's "External review" note above, from the original GPT-5.5/GLM-5.2 pass):

- **The 0.05 effect-size floor was justified once as pure detectability under unpaired 156-step
  noise ("0.02 is undetectable, so raise it to 0.05"), not as "this is what a visible style
  improvement looks like."** Pairing now makes smaller effects detectable, so keeping 0.05 purely
  out of inertia risks discarding real, visible improvements in the 0.03-0.05 band. Neither
  reviewer knows of a validated mapping from a DINOv2 cosine-similarity delta to a human-visible
  difference. Suggested cheap fix, not yet done: pick sample pairs with known score gaps already
  on disk (e.g. `dim64_780` vs `control_780`, a 0.0995 gap; `lr2e4_780` vs `control_780`, a 0.0315
  gap) and look at them side by side, unlabeled, to see which gap sizes are actually visible
  before trusting 0.05 as the right cutoff for round 1.
- **"Advance only the single best direction" is not a substitute for a real multiple-comparisons
  correction (Benjamini-Hochberg).** They solve different problems: BH bounds the false-discovery
  rate across a family of ~10 claims, while taking the argmax of 10 noisy statistics controls
  nothing and is subject to "winner's curse," the single best-ranked direction looks like a
  winner even when all 10 are actually null, every time, by construction. Both reviewers'
  recommendation: keep the significance-plus-effect-size gate first (screen every direction;
  advance nothing if none pass), and use "take the single best" only as a tie-break *among
  directions that already passed the gate*, with the full 780-step commit retrain serving as
  independent confirmation. Reserve full BH correction for any whitepaper claim of the shape "we
  tested 10, N improved," which is a different (family-wise) claim than "we picked one to
  commit."
- **The paired-seed design may not hold for the network_dim/network_alpha direction
  specifically.** Changing LoRA rank changes the shape of the randomly-initialized weight
  matrices, which changes how many values get drawn from the shared torch RNG stream during
  model construction, before the DataLoader's shuffle order is drawn from that same stream. A
  same-seed control/dim-change pair could desync in shuffle order despite sharing a seed value,
  which would mean that direction's paired deltas are really unpaired deltas in disguise. Fable's
  suggested check (cheaper than reading kohya's source top to bottom): log the first ~20 sample
  filenames per run and diff them between a control replicate and a dim-change replicate on the
  same seed; if they match, pairing held. Not yet done. Consequence if it doesn't hold: the
  `alpha32`/`dim16` slate entries (Section 3, candidate slate) get less statistical power than the
  other six directions, not an invalid result, just a weaker one.
- **`lr_scheduler_num_cycles` variants need their own probe length.** Caught by Fable, not Opus:
  a `cycles1` direction's single decay cycle spans the full 780 steps, so a 260-step probe of it
  would replicate the exact `constlr`-style failure mode step 1's calibration check exists to
  catch. Folded into the candidate slate's caveat (Section 3).
- **Mode collapse is a live risk for the centroid metric even with MMD as a cross-check.** DINOv2
  centroid similarity only measures how close a generated image sits to the *average* of all real
  images; it has no way to notice if every generation from a given config converges on the same
  "safest" output, since a collapsed-but-centroid-adjacent set can score as well as, or better
  than, a genuinely varied one. MMD is structurally more resistant, since its formula includes a
  generated-vs-generated similarity term that rises toward its maximum under collapse and adds
  *into* the discrepancy score rather than being invisible to it (matches the existing gen-gen
  vs. real-real check already run once against `art_batch`, Section 3's metric-upgrade note).
  But MMD's collapse alarm isn't foolproof: round 1's probes only generate 4 images per
  checkpoint, small enough that a real diversity drop could still hide in ordinary sampling
  noise; collapse onto one specific real training image (rather than the abstract centroid) can
  partly cancel out in the MMD terms; and per-prompt collapse can wash out in an aggregate score
  across all 4 prompts. The existing "no collapse signature" finding was measured on the original
  `art_batch` full-length run, not on any of round 1's new candidate directions, so it doesn't
  automatically carry over to a direction that changes things sharply (e.g. a much lower
  `min_snr_gamma`). Decision: reuse `notes/mmd_score.py` alongside centroid scoring in round 1's
  real scoring pass, watching the gen-gen self-similarity term per direction specifically, not
  just the final MMD^2 number, rather than assuming the earlier check still holds.
- Also flagged, not yet acted on: pre-register a single summary statistic per direction (which
  checkpoint, which prompt aggregation) before looking at any scores, to avoid quietly picking
  the best-looking cut after the fact; and plan one final unbiased end-to-end check (original
  baseline vs. whatever config wins after all 5 rounds, fresh seeds, on the holdout set) at the
  very end, to counter the optimistic bias that greedy round-over-round selection accumulates.

**Candidate-direction slate drafted** in response to the open question (8 directions, folded
into Section 3 above): proposed, not yet approved by the user.

**Liminal-band/uncanny-frontier overnight run, done (Section 3b).** The exploratory side branch
went from proposal to full run in one session. Found the RunPod serverless endpoint wedged before
starting (test jobs stuck `IN_QUEUE` with idle workers available); root cause was a negative
account balance (-$0.099) causing RunPod's dispatcher to silently withhold work rather than error.
Fixed once the user added funds, confirmed with a real completed test job before committing to an
unattended run. Launched `notes/run_uncanny_allnight.py`: 49 generations, 2.28 hours, 3392 total
images, stopped cleanly on its own $8.50 budget guard (of a $10 cap). Liminal-band novelty moved
0.8143 -> 0.8396 across the run, most of the gain landing right after a plateau-triggered GPT-5.5
creative handoff supplied 15 fresh uncanny-scene prompts, then flattened for the last 23
generations with no further escalation available. Real finding, not yet explained: either this
style has a fairly low novelty ceiling at these settings, or the search's exploit-heavy generation
mix starved exploration once the pool filled with similar high scorers, needs a follow-up run with
a more explore-heavy mix to tell apart. Full writeup in Section 3b. Gallery at
`notes/uncanny_sweep/gallery.html`.

**New idea, not started:** a separate exploratory thread using the same DINOv2 scorer in reverse
(maximize distance from any single real image while staying inside a "still looks like the
style" band) plus a MAP-Elites quality-diversity search to map the style's uncanny/liminal
frontier rather than just its centroid. Written up as Section 3b. Explicitly a side branch, not
part of the 5-round hyperparameter sweep, and not scoped or budgeted yet.

**Uncanny sweep 2 prompt seed list refreshed.** Wrote
`notes/uncanny_sweep2/gpt55_subjects.json` with 20 short, concrete subject prompts for testing
whether the CLAWMARKS style survives across unfamiliar everyday scenes. The list avoids the prior
used subjects and spreads across spaces, objects, weather, crowds, machines, and architecture so a
follow-up sweep can probe style generalization rather than repeat the first liminal set.

**Uncanny sweep 2 GPT-5.5 subject list regenerated.** Rewrote
`notes/uncanny_sweep2/gpt55_subjects.json` with 20 fresh 5-15 word subject prompts that avoid the
full prior used-subject list. The replacement set deliberately spans small businesses, courts,
roads, transit, storage, public notices, classrooms, machines, weather, and architecture to stress
where the fine-tuned style generalizes cleanly and where it dissolves into noise.

**Uncanny sweep 2 GPT-5.5 subject list refreshed again.** Replaced
`notes/uncanny_sweep2/gpt55_subjects.json` with another 20 short concrete prompts, excluding the
expanded prior-subject list supplied in chat. This pass emphasizes different everyday categories:
small shops, sports sites, civic storage, weathered infrastructure, machines, retail displays, and
ordinary objects arranged in uncanny ways.

### 2026-07-09: Exploration tools made mobile-first, added a favoriting control, and gained in-UI tooltips

Round 2's 280 images were merged into round 1's dataset and all 8 exploration tools (scan
gallery, solution map, coverage/void map, elite archive, redundancy clusters, novelty decay
watchlist, lineage tree, hub) were rebuilt against the combined set. Added a shared, hide-on-
scroll-down/show-on-scroll-up nav bar and a standalone `lightbox.js` module (reads
`scan_data.json` plus `/api/picks`/`/api/favorites` at runtime, so any page can call
`Lightbox.open(tag)` without a page load) via `notes/shared_ui.py`. Made every page's layout and
the solution map's canvas touch-friendly, since the UMAP's hover-based info panel was unusable on
a phone; added `touchstart` handling with a larger hit radius. Fixed a z-index bug that made the
lightbox's close button unclickable (the next/prev nav strips overlapped it with no stacking
order). Fixed the scan gallery lagging on every keystroke in the faithfulness filter by debouncing
text/number inputs and rendering thumbnails in chunks of 150 via `IntersectionObserver`, since the
user works over a poor connection and prioritizes a fast-loading page over a single big render.

Added a second marking mechanism distinct from "pick as winner": favoriting. Picking feeds the
next search generation's exploit pool (algorithmic consequence); favoriting is a pure bookmark
with zero effect on search behavior, stored in a parallel `user_favorites.json` next to
`user_picks.json`, with matching `/api/favorite`, `/api/unfavorite`, `/api/favorites` endpoints on
`notes/curation_server.py`. Wired into the lightbox (button plus `f`/`F` keyboard shortcut) and
into the scan gallery grid (badge, "favorited only" filter, live-updating on
`lightbox:favorite`).

Wrote in-UI tooltips (click-to-toggle "?" icons, since hover doesn't work on touch) explaining
faithfulness, novelty, picking vs. favoriting, MAP-Elites/"elite," the UMAP projection and its
mode-collapse chart, frontier cells, the novelty-decay trend threshold, and the caveat that
redundancy clusters are transitive (a chain of gradual drift, not necessarily a tight group of
look-alikes). Framed as general "explore an AI-generated image space" copy per the user's stated
intent for this to become a general tool, not documentation specific to this one dataset. All
tool pages rebuilt and `curation_server.py` restarted to pick up the favorites API; verified via
curl that all 8 pages 200, the favorite/unfavorite round-trip works, and the tooltip assets serve.
Not yet checked in an actual browser on a phone, which the project's UI-testing norm calls for
before this counts as fully done.

Not started: "generate counterfactual runs," requested in the same message as the above. Needs
scoping with the user before any build starts, since it may require live RunPod/ComfyUI
generation, which costs money while idle and shouldn't be spun up without a clear plan for what a
counterfactual run actually means here.

### 2026-07-09: Elite archive gets per-bin browsing; confirmed the exploit mutation genome is (strength, cfg, seed) only, not prompt

Discussion surfaced a real gap in the elite archive: `build_elite_archive.py` picked each bin's
"elite" by highest novelty alone whenever no human pick existed, but the DINOv2 scorer has no
aesthetic judgment, so the automated pick could easily be worse-looking than another image in the
same bin. Added a "view all N in this cell" modal (same pattern as coverage.html's cell-preview
modal) so a human can browse every candidate in a bin and pick a different one directly; elite
selection was moved client-side (was previously baked into the page at build time) so picking
inside the modal updates which image shows as the bin's elite immediately, no rebuild required.

Also confirmed by reading `run_uncanny_allnight2.py`'s job-building code directly: exploit jobs
(mutating near a parent) inherit the parent's prompt and prompt_name unchanged; only strength,
cfg, and seed get Gaussian-jittered. Explore jobs don't mutate anything, they re-roll a fresh
random subject/texture pair from scratch each time. So the actual search genome is
(strength, cfg, seed), not prompt. This matters for scoping "generate counterfactual runs"
(requested but not yet designed, see previous entry): re-running an image with a different prompt
would be a new kind of variation the search itself never performs, not a replay of existing
mutation logic with the prompt field held out.

### 2026-07-09: Counterfactual runs wired to live generation

Built and verified the counterfactual-runs feature: from any image in the lightbox, "generate
counterfactual" opens a panel prefilled with that image's prompt/strength/cfg (seed left blank,
defaulting to a fresh random draw), the user edits whatever field(s) they want to vary, and
submitting calls the same serverless ComfyUI endpoint (`uix4vdb2cec7sb`) the search itself uses.
`notes/curation_server.py` gained a synchronous `/api/counterfactual` endpoint: checks the RunPod
balance first and refuses below a $0.05 floor (the earlier gotcha where a negative balance made
jobs silently stall in queue rather than error), submits the job, polls up to 330s, saves the
result to `notes/uncanny_sweep/counterfactuals/`, and records it in `user_counterfactuals.json`.
Counterfactuals are deliberately outside the search: not scored against the DINOv2 metrics, never
fed into the exploit pool, a comparison tool only.

Live-tested end to end before trusting it. First attempt hit a real gotcha: a cold endpoint (no
recent jobs, scaled to zero workers) took 215s just to spin up a worker before generation even
started, blowing through the server's original 90s timeout and orphaning a queued job (cleaned
up via the endpoint's `/purge-queue`, confirmed no charge since it never reached `IN_PROGRESS`).
Raised the server timeout to 330s and reran: warm-worker generation completes in ~35-40s, so the
real range is "seconds if warm, up to several minutes if cold." Verified the full round trip
against the live server (not just the standalone script): submitted a job, got back a valid
1024x1024 on-style image (confirmed by eye, feathery ink linework matching the CLAWMARKS look),
confirmed the file serves and the record persists in `user_counterfactuals.json`. Test artifacts
deleted afterward.

Also confirmed via `notes/run_uncanny_allnight2.py`'s job-building code that the search's actual
mutation genome is (strength, cfg, seed) only, never the prompt text; see the previous entry.

### 2026-07-09: Round 2 pick analysis; lightbox loading/prefetch and tooltip fixes

Compared the 39 human-picked elites (all from round 1; round 2's 280 images are unreviewed, not
rejected) against the 3672-image population. Picks skew toward higher faithfulness and lower
novelty than average (0.487/0.487 picked vs. 0.324/0.630 population), and explore images get
picked about 12x more often than exploit images proportionally (2.4% vs. 0.2%). Read as: the
automated per-bin elite fallback ("highest novelty wins" when no human pick exists) is selecting
against what humans actually prefer, and exploit mutation isn't obviously earning its keep over
fresh explore draws. Not acted on yet since round 2 is unreviewed and was the run built
specifically to test the explore-heavy mix; browsing round 2 comes first. Both follow-ups are in
`TODO.txt`.

Fixed three UI bugs/gaps in the shared lightbox (`shared_ui.py`), live on all 8 tool pages:

- **Tooltip question marks silently failed inside the lightbox.** Root cause: `.infopop` used
  `position:absolute` with `window.scrollX/Y` added to the button's `getBoundingClientRect()`
  coordinates, a standard technique for normal page content, but wrong for anything inside the
  lightbox's `position:fixed` overlay, whose rect is already viewport-relative and doesn't move
  with scroll. On any page scrolled down before opening the lightbox, the popover landed off
  -screen. Fixed by switching `.infopop` to `position:fixed` and using the rect directly, which
  works correctly for both cases.
- **No loading feedback when stepping to the next/previous image.** Added a spinner overlay on
  the main image, shown from the moment `src` changes until the browser's `load`/`error` event
  fires.
- **No prefetching anywhere.** Added two-stage lightbox prefetch (the immediate next image
  fetched right away, next-next and previous fetched after a short delay so they don't compete
  with the immediate one on a slow connection) and a page-wide `IntersectionObserver` that fires
  an async request for a thumbnail's full-size file as soon as that thumbnail scrolls into view
  (gated on visibility, not the whole grid, since a filtered page can hold thousands of
  thumbnails). Wired the `data-tag` attribute the observer needs into every thumbnail-grid
  script: scan gallery, elite archive (both the main grid and the bin-browsing modal), coverage
  map, redundancy clusters.

Also added a "how does this search work" tooltip explaining MAP-Elites, explore vs. exploit, and
plateau/budget in plain language: one on the tools hub (`explore.html`) and one next to the scan
gallery's category filter, since that's where explore/exploit labels actually appear in the UI.

Extended the thumbnail prefetch to pause and resume with scroll position, not just start once
visible. Previously the `IntersectionObserver` fired a full-size prefetch the first time a
thumbnail scrolled into view and then stopped watching it, so a fast scroll through the grid
could queue up dozens of full-size downloads for thumbnails already scrolled past, competing for
bandwidth with whatever's actually on screen. Now the observer keeps watching every thumbnail for
as long as it's in the DOM: entering the viewport starts (or resumes) its full-size prefetch,
leaving the viewport aborts it by clearing the in-flight `Image`'s `src` (which cancels the
network request in every major browser) if it hasn't finished yet. A shared `prefetchState` map
replaces the old write-once `prefetched` set so the lightbox's own next/prev prefetch and the
grid's visibility-driven prefetch share one cache instead of ever double-downloading the same
image.

### 2026-07-09: Candidate seed browser, view and grow the explore-job subject pool on demand

The search driver already escalates to GPT-5.5 for fresh subjects when novelty plateaus
mid-run (`request_gpt55_subjects` in `run_uncanny_allnight2.py`), but that pool was only
visible or growable from inside a live run. Added a persistent, run-independent pool
(`notes/uncanny_sweep/candidate_seeds.json`, `{text: {source, created_at}}`) seeded from the
fallback list plus both rounds' existing GPT-5.5 subjects (45 to start, deduped), a new
`curation_server.py` endpoint (`POST /api/seeds/generate`, mirrors the driver's own prompt and
subprocess call to `opencode run -m openai/gpt-5.5`, deduped case-insensitively against the
existing pool before merging), and a browser page (`seeds.html`) to view the pool and trigger
generation. Live-tested: asked for 3 new seeds, got back 3 concrete, non-duplicate subjects
(airport baggage carousel, glass office atrium, roadwork cones) in well under a minute, merged
into the pool bringing it to 48. Generation costs opencode/GPT-5.5 API time, not RunPod spend,
so it carries none of the RunPod balance-floor risk the counterfactual endpoint has to guard
against. Not yet wired the other direction: `run_uncanny_allnight2.py` still keeps its own
per-round `gpt55_subjects.json` rather than reading from this shared pool, so seeds added here
aren't picked up by a run until that gets wired up.

Closed that gap the same day. `run_uncanny_allnight2.py` now loads `candidate_seeds.json` at
startup and merges it into the subjects list alongside `FALLBACK_SUBJECTS` and its own
per-round `gpt55_subjects`, so anything added through `seeds.html` reaches the very next run.
The wiring is bidirectional: whenever the run's own plateau-triggered GPT-5.5 escalation
produces new subjects, those get written back into `candidate_seeds.json` too (tagged
`source: "gpt5.5-round2"`), so a live run enriches the shared pool the same way the browser
does, instead of the two staying separate lists that happen to read the same fallback subjects.

### 2026-07-09: Preference classifier designed, replaces picking; full implementation plan written

The round-2 pick analysis above (picks skew toward higher faithfulness/lower novelty than the
population, and the automated per-bin fallback selects against what humans actually prefer) was
the trigger for a bigger decision: pause the round-1 hyperparameter sweep and shift effort to
inference-time exploration tooling instead, specifically closing the gap the pick analysis
surfaced. Brainstormed and wrote a design spec
(`docs/superpowers/specs/2026-07-09-preference-classifier-design.md`) for a preference
classifier: a model that predicts how much the user will like an image, trained on the user's own
yes/no ratings of generated images rather than on the DINOv2 faithfulness/novelty scores, which
have no aesthetic opinion.

Mid-design, scope grew: the new rating system supersedes "pick as winner" entirely (button,
badge, `/api/pick`/`/api/unpick` endpoints all removed), while favoriting (a pure bookmark with
zero search effect) stays exactly as it is. The 40 existing picks migrate into the new
`user_ratings.json` as yes-ratings via a one-time script, so no prior judgment is lost.

Design settled on, in order: (1) an embedding cache (`search/embed_cache.py`) that runs DINOv2
once per image and persists the vector, so training doesn't need to re-run the model; (2) a fast
yes/no rating page (`rate.html`) sampling unreviewed images stratified across the existing
faithfulness x novelty bins, so an early rating session doesn't over-sample whichever region
happens to dominate the pool; (3) a logistic regression trained on the frozen embeddings alone
(no generation metadata, and deliberately not a transformer or deeper model: with realistically
dozens-to-low-hundreds of labels, a higher-capacity model would memorize the label set instead of
generalizing: the standard "linear probe" approach exists specifically for this label-scarce
regime); (4) a pool re-ranking view (`preference_rank.html`) as the human validation gate, sorted
by predicted P(yes), sanity-checked against the 40 migrated picks scoring highly; (5) a two-stage
handoff for what actually steers the search: Stage 5a (immediate, no model needed) has
`elite_archive.py`'s per-bin fallback and `driver.py`'s exploit pool read yes-ratings exactly
where picks were read before; Stage 5b (opt-in, gated on Component 4 passing) swaps the *fitness*
function inside each MAP-Elites bin from "highest novelty" to "highest predicted-preference,"
while leaving the faith x novelty bin grid itself untouched, since collapsing the bins in favor of
pure top-K-by-preference selection would defeat the point of quality-diversity search and risk
collapsing the search onto one narrow mode. Both stages default off until the project owner
validates the model by eye.

Wrote the full implementation plan
(`docs/superpowers/plans/2026-07-09-preference-classifier.md`): 13 TDD tasks covering the
embedding cache, rating sampler, migration script, `curation_server.py`'s new ratings endpoints,
`rate.html`, the lightbox's pick-removal, `elite_archive.py` and `driver.py`'s Stage 5a rewiring,
`preference_model.py` (new `scikit-learn` dependency), `preference_rank.py`, and both files'
opt-in Stage 5b wiring. Per the user's direction, this hands off to unattended execution via
opencode/minimax rather than being implemented in this session.

Also found and fixed two small repo-hygiene issues while cleaning up this session: two stray
`.bak` files in `notes/uncanny_sweep/` (`scored_manifest.CORRUPTED_*.json.bak`,
`scored_manifest.GARBAGE_*.json.bak`, 732 and 452 stale manifest entries respectively, both
superseded long ago by the current 3672-entry `scored_manifest.json`) were confirmed as dead
snapshots and deleted; and `notes/probe_uncanny_report.html`, a tracked file, had been silently
truncated to 0 bytes by some earlier process (no current build script even references that
filename, so it's likely dead output from a script removed in the `whitepaper/` -> `notes/`
rename) and was restored from git rather than left corrupted or deleted outright, pending a
decision on whether the file is worth keeping at all.

### 2026-07-09: Data-loss incident during the CLAWMARKS package-transition smoke check - all
### full-resolution generated images in `notes/uncanny_sweep/` and `notes/uncanny_sweep2/` lost
An unattended opencode/minimax-m3 agent, executing Task 12 (the old-scripts-vs-new-package
smoke check) of the CLAWMARKS software-transition plan, destroyed every full-resolution PNG in
both sweep directories: `notes/uncanny_sweep/` dropped from 7.1 GB to 86 MB, `notes/uncanny_sweep2/`
from 544 MB to 3 MB. These were real, RunPod-billed ComfyUI generations accumulated across two
search rounds, never committed to git (gitignored as presumed-regenerable build output), and
never backed up anywhere outside this sandbox. They are gone.

**Root cause, step by step:**
1. The first attempt at Task 12 ran the old `notes/build_*.py` scripts directly against the
   live data directories with no isolation, truncating `scored_manifest.json` from 3672 to 452
   entries. Caught, killed, and manually repaired by replaying `merge_round2.py`'s merge logic
   against the surviving `scored_manifest_round1_only.json` backup and round 2's untouched
   manifest, restoring the correct 3672-entry file, plus regenerating `similarity.json`,
   `similarity_scored.json`, and `solution_map_data.json` (all confirmed correct afterward).
2. The plan's Task 12 was patched to bracket both the old-script run and the new-package run
   with an explicit backup/restore of the live directories, specifically to prevent recurrence
   of (1).
3. The second attempt hit a different failure: a naive full `cp -r` of both sweep directories
   (7.6 GB combined) exhausted the sandbox's disk (100% full, only 2.2 GB free), truncating
   `scored_manifest.json` and `similarity_scored.json` to 0 bytes mid-write. Root-caused to disk
   exhaustion, not a script bug. Fixed by clearing ~38 GB of regenerable package caches
   (`uv cache clean`, pip cache, `~/.cache/go-build`, `~/.cache/huggingface`) and re-restoring
   both files from a known-good backup.
4. The third attempt is where the real damage happened. The resumption prompt told the agent to
   *exclude PNGs from the Task 12 backup* (to stay well under the freed disk headroom), but
   Task 12's own restore steps do `rm -rf notes/uncanny_sweep{,2}` followed by `cp -r` from that
   same backup. A backup missing the PNGs, combined with a restore that assumes the backup is a
   complete mirror, deletes anything the backup doesn't have. The agent followed both
   instructions literally and wiped every full-resolution image with no way back. This was a
   plan-authoring mistake, not an agent error: two instructions that directly contradicted each
   other, both written in the same session, and the contradiction wasn't caught before the agent
   acted on it.

**Recovery attempted and exhausted:** searched every `/tmp` backup made during this session
(none held PNGs), git history (the images were never committed, so no git-based recovery
existed), the whole accessible filesystem for exact filenames from the manifest (no hits), every
live process's open file descriptors for a still-open handle to a deleted inode (none), the
RunPod account (no pods currently running), and the generation setup's network volume
(`pwkmq2gjhw`, holds only the base checkpoint and LoRA safetensors, never generation output).
Confirmed with the project owner: no copy exists anywhere else either. The images are
permanently gone.

**What survived:** all JSON metadata (`scored_manifest.json` and friends, 3672 entries, verified
correct), the downscaled `thumbs/` JPEGs (57 MB, not full resolution), and every generated HTML
page. Only the full-resolution source images themselves are lost. Task 13 (deleting the old
scripts) never ran and no PR was opened, so the package-transition work itself is undamaged; the
loss is confined to the sweep directories' image data.

**Standing lesson for any future unattended-agent data operation on this project:** never let an
agent's backup step silently narrow scope (excluding files "to save space" or "because they seem
regenerable") without re-checking every later step that assumes the backup is a complete mirror
of what it's replacing. A partial backup plus a full-mirror restore is a data-loss pattern, not
a size optimization, and it needs to be checked explicitly before the first destructive step
runs, not caught after.

### 2026-07-10: Near-miss with `solution_map_final_embs.pt` during PR #5's pre-merge review, and
### the real bug behind it

While landing the CLAWMARKS package-transition PR, I (Claude, this session) took a complete-mirror
backup of `notes/uncanny_sweep/` and `notes/uncanny_sweep2/` (98 MB total now that the PNGs are
gone, verified byte-identical to the live directories, entry counts 3672 and 280) before running
any old-vs-new comparison, per the lesson above. Good thing: running `uv run clawmarks build all`
from the `clawmarks-package-transition` worktree
(`/workspace/trent-clawmarks-worktree`) triggered `src/clawmarks/build/solution_map.py` to delete
`solution_map_final_embs.pt`, the only surviving copy of the DINOv2 embeddings for images that no
longer exist on disk.

**Root cause:** `solution_map.py` caches its embeddings keyed on an absolute-path list
(`saved["real_paths"] == real_paths`, where `real_paths` is built from `config.ROOT`). The old,
pre-transition script hardcoded its own root, so it always ran from the same absolute path and the
cache stayed self-consistent. The new package resolves `ROOT` dynamically per checkout
(`clawmarks.config.repo_root()`), so running it from a different checkout location (here, a git
worktree instead of the main checkout) changes every path in `real_paths`, the equality check
fails, and the old code (identical in both the old script and the new module) responded by calling
`os.remove(FINAL_EMBS_FILE)` before attempting to recompute. The recompute then failed immediately
(the source PNGs are gone), so without the backup, this would have been a second unrecoverable
loss on the same data the first incident already destroyed.

**Fix:** removed the `os.remove(FINAL_EMBS_FILE)` call in `src/clawmarks/build/solution_map.py`.
The subsequent `torch.save(..., FINAL_EMBS_FILE)` already overwrites the file in place on a
successful recompute, so the early delete served no purpose except destroying the last good copy
if recompute then failed. Verified via `md5sum`: rerunning `clawmarks build solution-map` after the
fix hits the same path mismatch, prints the same "re-embedding from scratch" message, fails on the
same missing PNG, and the cache file's checksum is unchanged.

This was not a bug the GLM review of PR #5 could have caught: it only surfaces by actually running
the code against the real cached data from a different absolute path, not from reading the diff.
Added a "data integrity is the number one goal" section to `CLAUDE.md` as a direct result: back up
before any operation against these directories, and treat any delete-to-invalidate-a-cache pattern
as suspect from now on.

### 2026-07-10: Preference classifier Tasks 9-13 implemented and verified in the
### `preference-classifier-phase-1` worktree; end-to-end check adapted around the lost images

Finished the preference-classifier implementation plan (`docs/superpowers/plans/2026-07-09-preference-classifier.md`), Tasks 9 through 13, in the existing `/workspace/trent-phase1-worktree` worktree (branch `preference-classifier-phase-1`), following the plan's TDD steps exactly. Each task's tests were written first, confirmed to fail for the right reason, then implemented and committed separately:

- **Task 9** (`search/preference_model.py`): logistic-regression classifier trained on frozen DINOv2 embeddings plus yes/no ratings, with `MIN_LABELS = 50` as a floor below which it refuses to train.
- **Task 10** (`build/preference_rank.py`): a `preference_rank.html` view ranking every embedded image by predicted P(yes), for eyeballing the model against actual taste before it's allowed to steer anything.
- **Task 11**: an opt-in `--use-predicted-preference` flag on `clawmarks run allnight`, defaulting off, that swaps the search driver's exploit-pool source from yes-rated images to the trained model's top picks.
- **Task 12**: the matching opt-in flag on `clawmarks build archive`, changing each MAP-Elites cell's fallback champion from highest-novelty to highest-predicted-preference when a model exists. Both Stage 5b flags stay off by default per the plan; validating the model via `preference_rank.html` before flipping either is a human decision, not part of this implementation.
- **Task 13** (end-to-end verification): here the plan's assumption broke. It expected to run the full pipeline (migration, embed 3672 images, rebuild the archive, smoke-test `rate.html` and the server) against the real production data in `notes/uncanny_sweep/`. But every full-resolution PNG there is gone, permanently, per the data-loss incident logged earlier in this notebook, confirmed by checking the manifest directly: all 3672 entries pointed at files that no longer exist on disk.

**How Task 13 actually got verified:** two changes, both made after checking with the project owner rather than improvising alone, given this project already lost real data once from an agent working around an ambiguous data situation unsupervised.

1. `embed_cache.main()`'s `image_path_for` now falls back to the surviving downscaled `thumbs/<tag>.jpg` when the manifest's full-res path doesn't exist, rather than raising `FileNotFoundError`. A future generation round that keeps its full-res images intact still embeds from those; only this already-damaged sweep silently uses thumbnails.
2. Rather than run the real pipeline against all 3672 (partially fake, thumbnail-backed) images, built a small **checked-in** test fixture instead: `tests/fixtures/sample_sweep/notes/uncanny_sweep/` holds a trimmed 100-entry `scored_manifest.json`, the matching 40-entry `user_picks.json` (all 40 real yes-picks, so the migration step is meaningful), and 100 real `thumbs/*.jpg` files (the 40 picks plus a novelty-stratified sample of 60 more, for MAP-Elites bin coverage). `.gitignore` gained a `!tests/fixtures/**/*.jpg` exception to the blanket `*.jpg` rule, plus ignore rules for that fixture's own generated outputs (`embeddings.npz`, `*.html`, `*.js`, `user_ratings.json`), mirroring how the real sweep directory treats build output as regenerable and un-tracked.

Ran the full pipeline against that fixture via `clawmarks.config`'s existing `CLAWMARKS_ROOT` env-var override (no code changes needed to point the tools at a different root): migration produced 40 yes-ratings as expected; `embed_cache` downloaded DINOv2 weights and embedded all 100 thumbnails (slow on this sandbox's CPU-only setup: no GPU, so an 8-core CPU forward pass through a base-sized ViT dominates the wall-clock time, not the code); the archive rebuilt to 11 occupied cells with 8 human-picked elites; `grep -c user_picks src/clawmarks/build/elite_archive.py` returned 0, confirming Task 6's ratings migration is complete; the rate-page server smoke-tested clean (`/api/rate/next` returned a real item, the `POST` recorded a 41st rating, `/api/ratings` reflected it). One expectation in the plan's own text didn't hold exactly: `GET /api/pick` returns a 404-file-not-found HTML page, not the `{"error": "unknown endpoint"}` JSON the plan predicted. That JSON error only fires for unrecognized **POST** paths in `curation_server.py`; `do_GET` falls through to static file serving for anything unmatched. Pre-existing asymmetry, not a regression from this work, and still confirms the intended thing: the old pick endpoint doesn't do anything anymore.

Per the plan's explicit Step 7, did *not* run `preference_model.py`, build `preference_rank.html`, or enable either `--use-predicted-preference` flag against real data: 40 yes-only labels is well under `MIN_LABELS = 50`, and validating the model against real taste is a manual step for later, after a rating session on `rate.html`.

All 66 unit tests pass (`pytest -v` in the worktree). Five commits landed on `preference-classifier-phase-1`, one per task (9 through 13), continuing the same worktree that already held Tasks 1-8 from the prior session.

### 2026-07-10: PR #6 review, fixes, and merge; a real merge conflict between PR #5 and PR #6's overlapping `cli.py` changes

Tasks 9-13 above were done by an unattended process with no review, so before merging PR #6 I dispatched an independent GLM review of the unreviewed range (`cefa215..ef8db3b`). It found no Critical issues (the opt-in Stage 5b contract held everywhere, cross-validation avoided data leakage, DINOv2 stayed frozen, Task 13 correctly used a committed fixture instead of touching live data) but flagged 4 Important issues, all fixed directly:

1. `--use-predicted-preference` was wired into `driver.main`'s own argparse but never forwarded from `clawmarks run allnight`, making Stage 5b unreachable from the documented CLI. Added the flag to the `allnight` subparser and forwarded it.
2. `clawmarks build <target> --use-predicted-preference` forwarded the flag to every build target, not just `archive`; harmless today but a landmine for a future argv-parsing target. Restricted forwarding to `archive` only.
3. `preference_model.py` had no guard against single-class training data. `MIN_LABELS = 50` counts total labels, not per class, so the natural post-migration state (50+ ratings, all `yes`) would pass the floor and then crash inside `StratifiedKFold`/`LogisticRegression.fit`. Added `class_balance_error()`, checked before cross-validation, with 5 new tests covering all-yes, all-no, below-fold-count-minority, balanced, and imbalanced-but-sufficient cases.
4. The committed fixture's `scored_manifest.json` had hardcoded absolute paths rooted at the main checkout (`/workspace/trent-with-smart-prompts/...`), inert today but a landmine for a future test pointing `SWEEP_DIR` at the fixture. Rewrote the `"file"` fields to bare filenames.

71 tests passed after the fixes (up from 66). Merging then surfaced a real conflict: PR #5 had rewritten `cli.py`'s build-target dispatch into a lazy `importlib`-based `_BUILD_MODULES` dict (to avoid importing every build module's heavy dependencies for a single-target build), while PR #6's branch still had the older eager `_build_targets()` dict, now also carrying `preference-rank` and `rate` targets plus the Stage 5b flag-forwarding logic. Resolved by merging `origin/main` into `preference-classifier-phase-1`, keeping PR #5's lazy-import structure, adding the two new targets to `_BUILD_MODULES`, and re-applying the archive-only flag-forwarding fix on top of it. `pyproject.toml`/`uv.lock` conflicted too (PR #5 had moved `pytest` to a dev extra; PR #6's branch had added `scikit-learn` to core dependencies) - kept both changes, regenerated the lock with `uv lock`. Verified the merged CLI parses `run allnight --use-predicted-preference` and `build archive --use-predicted-preference` correctly, then re-ran the full suite (71 passed) before pushing and merging. PR #6 merged as `6d42aa4`.

### 2026-07-10: One-off 100-image seed batch (`notes/uncanny_seedrun1/`), fresh prompts from Fable-high + GLM-max, no allnight loop

The project owner wanted fresh seed ideas after confirming the round-1/round-2 full-resolution PNGs are gone for good (see the 2026-07-09 data-loss entry above; re-checked during this session and confirmed no other copy exists anywhere - not the RunPod network volume, which never held a second copy since `SaveImage`'s `filename_prefix` writes to each serverless worker's own ephemeral output folder, not the persistent volume, and not RunPod's job-result history, which has a retention window measured in minutes, not days).

Rather than reuse `clawmarks run allnight`'s two hardcoded `RoundConfig`s (round 1's textures/subjects vocabulary, round 2's exploit-pool logic), this was a simple one-off: 100 fresh prompts, submitted directly to the serverless endpoint, no MAP-Elites archive, no novelty scoring, no allnight state machine.

**Prompt sourcing:** 50 subject ideas from Claude Fable 5 (dispatched directly via the Agent tool's `model: fable` parameter, not through opencode/OpenRouter - the OpenRouter key backing `openrouter/anthropic/claude-fable-5` didn't have enough credit for the brainstorm's token budget, so this sidestepped OpenRouter entirely), then 50 more from GLM-5.2 at `--variant max` via opencode, given Fable's 50 as an explicit exclusion list so the two sets wouldn't overlap. Checked programmatically: zero duplicates between the two lists. Each subject got a random texture from round 1's four-texture vocabulary (for stylistic consistency with the earlier sweeps) and randomized strength/cfg/seed, mirroring `driver.py`'s explore-job construction exactly.

**Budget:** capped at $5, checked via `get_balance()` every 10 jobs during submission and every poll cycle during collection, same pattern as `ROUND_CONFIGS`' `budget_usd_cap`/`budget_safety_margin` fields. Actual spend: **$0.08** for all 100 images (completed=100, failed=0, abandoned=0) - the cap was never close to binding. Ran as a standalone script (`clawmarks.compute.comfyui` + `clawmarks.search.driver.get_balance`, no changes to either module) rather than a new CLI subcommand, since it's a one-off and not meant to become a repeatable target.

Output: `notes/uncanny_seedrun1/` (193MB, 100 PNGs plus `manifest.json` and `job_map.json`). Not yet embedded, scored, or built into any view - that's a follow-up step (`clawmarks build` targets expect `notes/uncanny_sweep/` by convention, so pointing them at this new directory needs either a `CLAWMARKS_ROOT`-style override or copying/renaming, not decided yet).


### 2026-07-10: Built and served notes/uncanny_seedrun1/ end to end; fixed three real bugs in `build all` and `serve`

Built every tool page against the 100-image seed batch from the entry above and started `clawmarks serve` pointed at it, so the images could actually be voted on. Getting there surfaced three genuine bugs, none related to data safety, all in code paths that had simply never been exercised this way before:

1. **`clawmarks serve` crashed under the CLI wrapper.** `cli.py` called `curation_server.main()` with no arguments; `main(argv=None)`'s fallback then read raw `sys.argv[1]`, which under `python -m clawmarks.cli serve` is the literal string `"serve"`, so `int("serve")` raised. The server had only ever been started as a standalone script before (`python curation_server.py <port>`), never through the `cli.py` wrapper, so nobody had hit this. Fixed by having `cli.py` pass `serve_main([])` explicitly and having `curation_server.main` only fall back to `sys.argv` when `argv` is actually `None`, not just falsy.

2. **`rate.html` was always loading the small thumbnail, never the full-resolution image.** `/api/rate/next` already returns both `thumb` and `file` fields, but the page's JS only ever set `img.src = d.thumb`. On mobile this meant a ~300px JPEG stretched to fill the screen, blurry the moment you tried to zoom. Fixed to load `d.file` directly, and gave the image more vertical room (`60vh` -> `78vh`).

3. **`build all` had no dependency ordering, and crashed the first time it was ever run on a directory with no leftover files from a prior build.** `map_view.py` reads `solution_map_data.json` and `redundancy_view.py` reads `similarity_scored.json`, both written only by `solution_map.py`, but `_BUILD_MODULES`' dict order (which doubles as `build all`'s run order) had `map` and `redundancy` running *before* `solution-map`. `notes/uncanny_sweep/` never hit this because those files already existed from an earlier successful run there; a genuinely fresh directory hit it immediately. Moved `solution-map` earlier in the dict, added a test pinning the order so it can't silently regress. Also found `umap-learn` was imported by `solution_map.py` but never declared in `pyproject.toml` - added it, pinned (`umap-learn==0.5.9.post2`), via `uv add`.

Also split `build all`'s target list into a new `_ALL_TARGETS` that excludes `probe-report`: that target is a fixed report over one specific historical probe-calibration run and never reads `SWEEP_DIR` at all, so it can't be built per sweep directory and will always fail against any directory that isn't that exact run. This isn't a bug to route around, it's a category error to have it in the "build everything for this directory" loop in the first place. Still buildable directly via `clawmarks build probe-report`.

74 tests pass (up from 71). Verified live: reran `build all` from scratch against `notes/uncanny_seedrun1/` (after deleting its `solution_map_data.json`/`similarity_scored.json` to force the from-scratch path) and it completed cleanly end to end - every sweep-scoped tool page written (`archive`, `coverage`, `explore`, `gallery`, `lineage`, `map`, `novelty-decay`, `rate`, `redundancy`, `scan`, `seeds`, plus `thumbnails` and the three JSON intermediates). `preference-rank.html` correctly did not get built: it needs a trained preference model (50+ ratings), which doesn't exist yet for this fresh batch - that's expected, not a bug.

All three fixes pushed straight to `main` (`e664251`) using the same admin-bypass-on-branch-protection pattern as the earlier stray-fix commit, from a scratch worktree tracking `origin/main` rather than touching the shared main checkout.

### 2026-07-10: `rate.html` swipe-to-vote and tap-to-zoom, merged into `feat/preference-toggle`

Replaced `rate.html`'s yes/no buttons with a swipe-card interaction, motivated by the same mobile-rating friction the seedrun1 session above surfaced (tapping small buttons on a phone is slow, and the previous fit-to-screen display had no way to inspect an image at native resolution before voting). Design and plan were written first (`docs/superpowers/specs/2026-07-10-rate-page-swipe-zoom-design.md`, `docs/superpowers/plans/2026-07-10-rate-page-swipe-zoom.md`), then implemented via `delegate-to-opencode-attended` (five tasks, `opencode-go/minimax-m3`, one shared session, each task independently reviewed) in an isolated worktree (`/workspace/trent-with-smart-prompts/.worktrees/rate-page-swipe-zoom`, branch `feat/rate-page-swipe-zoom`).

The design changed mid-flight after a live look at the first three tasks: the original spec called for double-click/double-tap zoom and touch-only voting (mouse drag was explicitly pan-only). Manual review led to three revisions instead: single tap/click zoom (no double-tap timer), mouse drag now votes the same way touch drag does, and the swipe gesture rotates the image (capped at 15deg) and shows a colored thumbs up/down overlay in place of the earlier plain color stamp. The spec and plan were both revised and committed before dispatching the task that implemented the change, so the docs describe the shipped behavior rather than the superseded one.

**Verification, and what it caught.** Task 4 (interactive verification) required a real browser; this sandbox had no browser automation tool available at first, so a Playwright MCP server was installed mid-session (`npx playwright install chromium`, since the default "chrome" channel needs real Google Chrome and this sandbox has no passwordless sudo to install it). Driving the live page confirmed tap-to-zoom centered on the click point, pan clamped to the image's own bounds, drag-to-vote with rotation/overlay in both directions posting the correct label to `/api/rate` and advancing past the 25%-width threshold, sub-threshold snap-back with no vote recorded, and keyboard voting still working alongside the new gestures.

That verification used mouse `.click()`, and a subsequent independent code review (opencode, `glm-5.2`; the first attempt at `gpt-5.6-sol` failed outright with a hard "usage limit reached" on every retry, not a transient error) caught what the mouse-only check missed: `touchstart` never called `e.preventDefault()`, so on a real touch device the browser fires synthetic `mousedown`/`mouseup` after `touchend`, which re-triggered the zoom toggle and canceled it, making tap-to-zoom non-functional on touch devices, the exact use case the feature was built for. This is a useful gotcha for future gesture work: a mouse-driven browser-automation check can pass cleanly while the touch path it doesn't exercise is broken, especially where the browser injects synthetic events with real touch hardware that a script-dispatched event does not reproduce. Fixed with one line and confirmed by dispatching a real `TouchEvent` and checking `event.defaultPrevented` before and after the fix, then confirming a full tap cycle left the image zoomed.

The review also caught that the revised spec/plan commits existed in the repository but weren't ancestors of the feature branch's own history, so the branch's committed docs still described the superseded double-click design even though its code implemented the revised one. Fixed by cherry-picking both doc commits onto the feature branch before merge, so the docs and code agree at every point in the branch's own history, not just after merging into the parent branch.

All 8 tests pass. Merged into `feat/preference-toggle` (`bcbc3ff`), verified the merge commit landed before removing the worktree, then removed both the worktree and the now-merged local branch.

### 2026-07-10: Research workspace redesign grounded in existing scientific tools

The first integrated workflow mockup was rejected as visually infantile and scientifically
shallow. Its linear stepper, generic cards, and simplified findings hid the project's actual
experimental structure. The replacement direction is an idea workbench where observations link
to competing hypotheses, hypotheses branch into testable experiment designs, and completed runs
update the evidence for each branch. The design study will borrow specific proven mechanics from
OSF preregistration (explicit predictions and decision rules), Benchling (linked notebook evidence),
Weights & Biases and DVC (baseline-centered run comparison and reproducibility), Ax (constrained
search spaces and trial lifecycles), JMP (response-surface exploration), and pyribs (archive,
emitter, and scheduler separation). No production implementation has started; the design remains
in collaborative brainstorming.

### 2026-07-10: Preference-guided exploration research narrows the redesign

An independent MiniMax M3 research pass reviewed contextual Bradley-Terry preference learning,
active acquisition, quality-diversity archives, and embedding-space mapping for the CLAWMARKS
exploration tool. The central correction is operational: a logistic-regression preference model
over frozen DINOv2 image embeddings cannot provide a useful gradient through the LoRA or diffusion
sampler. It can predict preference, compare candidates, and quantify uncertainty after images are
rendered. The search must therefore remain a black-box search over controllable inputs such as
prompts, LoRA strength, CFG, seeds, and later text-conditioning embeddings.

The current direction retains DINOv2, ratings, lineage, budget guards, and quality-diversity
coverage. It replaces the tool-directory UX with one bounded Expedition that shows the goal,
budget, explored neighborhoods, rating queue, current elites, and the next generation action in
one operation. The user chose site-launched, explicitly confirmed paid batches and selected text
embedding search as a later search surface. This needs a new ComfyUI conditioning interface and
remains separate from the first interface redesign. The map will distinguish a trustworthy
high-dimensional nearest-neighbor/coverage system from a lower-trust 2D navigation view; it will
not claim that UMAP preserves global geometry.

The product architecture is now fixed as an **Expedition console**, not a polished directory of
independent tools and not a general-purpose research workbench. An Expedition is one bounded paid
search with an explicit goal, allowed controls, batch size, hard budget ceiling, current corpus,
review queue, launch confirmation, lineage, and outcome record. The lab notebook remains the
scientific record. A broader observation, hypothesis, and experiment workbench is deferred until
the Expedition loop proves useful.

### 2026-07-11: Pairwise head-to-head preference model implemented

Added `search/preference_pairwise_model.py` on the isolated `head-to-head-compare` branch. The model turns each stored winner and loser pair into an embedding difference and its negation, producing balanced positive and negative training rows for logistic regression. It refuses to train below 50 comparisons, cross-validates with leave-one-out for smaller row sets and five stratified folds otherwise, and writes the model plus timestamped metadata only after training succeeds. The model scores individual embeddings with logistic regression's decision function, which orders images by predicted preference.

The new focused test module covers mirrored rows, unknown tags, multiple comparisons, scoring order, cross-validation, the comparison floor, persisted model metadata, and the missing-comparisons CLI path. `uv run pytest tests/test_preference_pairwise_model.py -v` passed all 8 tests. Scikit-learn emitted 17 `OptimizeWarning` messages from logistic regression under Python 3.14; they did not affect the assertions.

### 2026-07-11: Comparison pair sampler implemented

Added `search/comparison_sampler.py` for the head-to-head compare UI. Before 50 stored comparisons, it samples two distinct images from independently chosen occupied faithfulness x novelty bins, which spreads early labels across the available image space. Once a preference model exists at or above that floor, it scores a random pool of up to 200 images through injected scoring and embedding callbacks, then returns the two closest scores as the model's most uncertain comparison. The module stays independent of the pairwise model and embedding cache so the later server wiring supplies those dependencies. `uv run pytest tests/test_comparison_sampler.py -v` passed all 8 focused tests.

### 2026-07-11: Compare page UI implemented

Added `build/compare_page.py`, the static generator for the head-to-head preference interface. It fetches the next pair from `/api/compare/next`, sends the chosen winner and loser to `/api/compare`, displays two directly clickable image panes, accepts left and right arrow-key choices, tracks the session count, and shows a completion state. Each pane has a magnifier that opens a full-resolution overlay with mouse drag panning.

The focused page tests passed: `uv run pytest tests/test_compare_page.py -v` reported 7 passed. The old yes/no rating page and its tests were removed. The full suite stops during collection in 10 curation-server test modules because `curation_server.py` still imports the deleted legacy page; Task 4 replaces that import and route. `rg -l "rate_page" src tests` now identifies only `src/clawmarks/curation_server.py`.

### 2026-07-11: Compare page review fixes

Added touch drag support to the full-resolution compare overlay. `touchstart`, `touchmove`, and `touchend` now use the same bounded pan calculation as mouse drag. A touch with no movement closes the overlay, while a touch drag prevents page scrolling. Also made both comparison API requests reject non-success HTTP responses and show a visible connection error instead of leaving the page silent.

### 2026-07-11: Server comparison API integrated

Replaced the legacy yes/no rating routes with the pairwise comparison API in `curation_server.py`. `GET /api/compare/next` returns two item summaries or `{"done": true}` when fewer than two images exist. `POST /api/compare` appends `{winner, loser, compared_at}` records to `user_comparisons.json` and retrains the pairwise model every 10 comparisons after the 50-comparison floor. A successful retrain refreshes the server's in-memory model cache immediately, so the next pair selection uses it without a restart.

The server now imports `comparison_sampler`, `preference_pairwise_model`, and `compare_page`; it no longer reads, writes, or deletes the legacy rating store or model. Updated status-route and helper tests to match the pairwise data shape, then added HTTP-server coverage for pair selection, completion, comparison persistence, validation, the compare page, and removed rate routes. Focused tests passed 11 of 11, and the full suite passed 160 tests. The suite still emits 35 pre-existing sklearn and UMAP warnings.

### 2026-07-11: Navigation moved from legacy rating to head-to-head comparison

Replaced the shared navigation entry for the deleted `rate.html` page with `compare.html`, labeled "compare images (head-to-head)." Added a regression test that requires `compare.html` and forbids `rate.html` in `NAV_OPTIONS`. The test failed before the change because only the legacy URL was present, then passed afterward: 2 passed, 1 deselected. `uv run` reported an environment-path warning because the active `VIRTUAL_ENV` points at the parent workspace rather than this worktree's `.venv`; pytest still used the worktree environment.

### 2026-07-11: Elite archive now uses favorites and the pairwise preference model

Updated `build/elite_archive.py` so a favorited image from `user_favorites.json` supplies the
per-cell human override. The retired yes/no rating store no longer exists after the head-to-head
comparison migration, while favorites retain full item records keyed by tag and remain suitable
for this independent bookmark role. The optional Stage 5b ordering now imports the pairwise
preference model and uses its `score` function. The generated archive also fetches
`/api/favorites` and labels human-selected winners as "favorited." Focused archive and
predicted-preference tests passed 9 of 9. `uv run` emitted the known environment-path warning
because `VIRTUAL_ENV` points to the parent workspace, but pytest used this worktree's `.venv`.

### 2026-07-11: Search driver now uses favorites and pairwise preference scoring

Updated `search/driver.py` so round 2's exploit pool reads complete item records from
`user_favorites.json` instead of joining legacy yes/no ratings to `scored_manifest.json`. Stage
5b now loads `preference_pairwise_model.joblib` and ranks embeddings with the pairwise model's
`score` function. Added focused favorite-loader coverage and removed the retired rating-loader
tests. Verification and commit details follow in the implementation record for this task.

### 2026-07-11: Preference rank page now uses pairwise preference scoring

Updated `build/preference_rank.py` to load `preference_pairwise_model.joblib` and rank embedded
images through the pairwise model's `score` function. The page now describes predicted preference
scores learned from head-to-head comparisons, rather than probabilities from yes/no ratings. The
focused preference-rank suite passed 4 of 4 tests before and after the change. `uv run` emitted the
known environment-path warning because `VIRTUAL_ENV` points to the parent workspace, but pytest
used this worktree's `.venv`.

### 2026-07-11: Preference status page now reports comparison counts, not yes/no label counts

Rewrote `build/preference_status.py` so it reports `n_comparisons`/`min_comparisons` against
`user_comparisons.json` and the pairwise model's `MIN_COMPARISONS` gate, instead of the retired
`n_yes`/`n_no`/`n_total` label counts. The trained-model panel and the predicted-preference toggle
now read from `preference_pairwise_model.py`'s `MODEL_FILE`/`MODEL_META_FILE`. Task 4 had already
updated the server's watched-files list for the new comparison and model files, so no
`curation_server.py` change was needed here. The focused suite passed 6 of 6 tests, the
`curation_server` preference-status route regression check passed 5 of 5, and the full suite
passed 161 of 161.

### 2026-07-11: Head-to-head comparison migration complete; verification sweep found two missed call sites

The yes/no rating system (`search/preference_model.py`, `rate.html`, `user_ratings.json`) is
fully replaced by head-to-head comparisons across every live code path: comparison sampling,
the compare UI, the server's compare endpoints, the elite archive, the search driver's exploit
pool, the preference-rank page, and the preference-status page. See
`docs/superpowers/specs/2026-07-11-head-to-head-preference-design.md` and
`docs/superpowers/plans/2026-07-11-head-to-head-preference.md`. The old `search/preference_model.py`
module, `user_ratings.json`, and any existing `preference_model.joblib` remain on disk wherever
they already existed, untouched and unused by any code after this migration.

The Task 10 verification sweep's grep check (`rg` for `preference_model\b|rating_sampler|rate_page|
rate\.html|predict_proba|/api/rate\b|/api/ratings\b` under `src/`) found two call sites the plan's
ten tasks didn't cover: `build/map_view.py` and `build/scan_gallery.py` both still fetched the
retired `/api/ratings` endpoint to highlight "picked" (formerly yes-rated) images. Because both
calls were wrapped in a silent `.catch()`, the pages didn't crash. Instead, the "picked" highlight
and the "picked only" filter quietly stopped working the moment Task 4 removed the endpoint,
a real regression missed by the plan's task list. Fixed both to read `/api/favorites` instead,
matching the same favorites-as-override-signal pattern used in Tasks 6 and 7. Neither file had
existing test coverage for this behavior, so no tests needed updating; both files' existing test
suites (4 and 6 tests) still pass. Full suite: 161 of 161 passed after the fix.

Manually smoke-tested the live server via Playwright against `notes/uncanny_seedrun1` (the only
directory in this checkout with `embeddings.npz` and `scored_manifest.json`), served on a
non-default port so as not to collide with another automation's server already running on the
project's default port 8420. Confirmed on `/compare.html`: click-to-pick and the ←/→ keyboard
picks both record a comparison and load a fresh pair; the zoom overlay opens on the magnifier
tap with the correct full-resolution image and closes on a second tap. Confirmed
`/preference_status.html` reflects the session's 3 new comparisons and the correct
comparisons-based gate message. Confirmed `/preference_rank.html` shows the expected
no-trained-model message (referencing `preference_pairwise_model`, not the legacy module) since
3 comparisons is below the 50-comparison floor. No browser console errors beyond a harmless
missing-favicon 404. Deleted the `user_comparisons.json` file this smoke test wrote to
`uncanny_seedrun1` afterward, since it was test data, not a real session.

### 2026-07-11: Ported PR #9's significance testing, staleness detection, and retrain UI onto the pairwise model

While this branch (head-to-head comparisons) was in progress, a concurrent automation's PR #9
merged into `main`, adding three features to the legacy yes/no system this branch retires:
a permutation-test significance check (`preference_model.significance()`), staleness detection
via a fingerprint of the exact labels used to train (`preference_model.ratings_fingerprint()`),
and a "Retrain now" button on `preference_status.html` backed by a new
`/api/preference_retrain` endpoint. Both branches touched the same three files
(`search/preference_model.py`, `build/preference_status.py`, `curation_server.py`), so merging
as-is would have produced a real design collision, not a resolvable text conflict: PR #9's
additions were written against the retired yes/no model. The user chose to port PR #9's ideas
onto the new pairwise system rather than discard them or merge both designs unreconciled.

Ported each feature onto `search/preference_pairwise_model.py` and the pairwise-based
`build/preference_status.py` rewritten in this branch's Task 9:
- `significance()`: a permutation test reporting a p-value and the majority-class baseline
  accuracy (always exactly 0.5 for the pairwise model, since mirroring every comparison
  guarantees exact class balance, unlike the legacy yes/no labels).
- `comparisons_fingerprint()`: a hash of the exact (winner, loser) pairs a train run would use,
  built on a new shared `_iter_usable_comparisons()` helper so `build_training_set` and the
  fingerprint can't drift apart. `preference_status.compute_data()` compares this against the
  fingerprint stored in the last training run's metadata to show a staleness banner ("N new
  comparisons since last train" or a generic "comparisons have changed" message if the count
  matches but the pairs changed, e.g. a comparison result correction).
- Atomic model save: `train_and_save()` now writes the joblib file to a `.tmp` path and
  `os.replace()`s it into place, matching the existing atomic pattern already used for the
  metadata sidecar and matching this project's standing rule against fragile invalidate-by-delete
  patterns.
- A "Retrain now" button and `/api/preference_retrain` POST endpoint in `curation_server.py`,
  gated by `_preference_retrain_gate_error()`, which mirrors `train_and_save()`'s own gates using
  `build_training_set()` so the check can distinguish "not enough comparisons yet" from
  "comparisons exist but their embeddings aren't cached" (the second needs an `embed_cache`
  refresh, not more comparing, and pointing someone at `compare.html` for that would waste their
  time). A successful manual retrain also refreshes the in-memory `_pairwise_model_cache` the
  comparison sampler reads, matching the existing auto-retrain behavior in
  `_maybe_retrain_pairwise_model`.
- `_prediction_watched_files()`: `archive.html` (when using predicted preference) and
  `preference_rank.html` now watch the trained model file for cache invalidation, not just the
  scored manifest. Before this fix, a retrain (manual or the existing every-N-comparisons
  auto-retrain) wouldn't invalidate either page's cached render until the manifest itself
  changed or the server restarted, silently serving stale predictions.

Added 13 new tests across `test_preference_pairwise_model.py`, `test_preference_status.py`, and
`test_curation_server_preference_status_route.py` covering significance, fingerprint stability
and sensitivity to swapped comparisons, the staleness banner's two message variants, and the
retrain endpoint's three response paths (success, gated rejection, training crash). Full suite:
177 of 177 passed. Manually smoke-tested `/preference_status.html` via Playwright against
`notes/uncanny_seedrun1`: the "Retrain now" button renders, and clicking it with zero recorded
comparisons correctly surfaces the gate message ("only 0 usable comparisons (need 50)...")
without writing any file, confirmed via `git status` on the seed-run directory afterward. No
console errors beyond the harmless missing-favicon 404. `archive.html`, `preference_rank.html`,
and `compare.html` all still returned 200.

### 2026-07-11: GLM 5.2 review of the PR #9 port found a real training-floor bug

Dispatched an independent review of the PR #9 port (previous entry) via the `opencode` CLI
(`opencode-go/glm-5.2`), handing it the port's diff and PR #9's original diff for comparison.
It returned "Needs fixes": 1 Important finding and 5 Minor.

**Important, fixed:** `train_and_save()` only checked the raw comparisons count against
`MIN_COMPARISONS` before training, not the usable count after `build_training_set()` filters out
comparisons whose tags lack a cached embedding. A comparisons list could clear the raw floor
while its usable subset fell well below it, so a model could train (and get served) on far fewer
real training rows than the floor is meant to guarantee, silently, via the auto-retrain path or
the CLI. `_preference_retrain_gate_error()` in `curation_server.py` already checked the usable
count correctly, so the manual "Retrain now" button would have refused correctly while the
auto-retrain path or `python -m clawmarks.search.preference_pairwise_model` would not have,
an inconsistency between two code paths meant to enforce the same rule. Fixed by computing
`n_usable = X.shape[0] // 2` and refusing to train when it falls below `MIN_COMPARISONS`, and
storing `n_usable_comparisons` in the model metadata so downstream staleness checks can use the
real trained-on count instead of the raw comparisons count.

**Minor, fixed:**
- `preference_status.compute_data()`'s "new comparisons since last train" count subtracted the
  *raw* `n_comparisons` from the model metadata against a *usable* row count, a unit mismatch
  that could over- or under-report staleness. Fixed to read the new `n_usable_comparisons` field
  (falling back to `n_comparisons` for metadata written before this fix).
- Two HTML strings in `preference_status.render_html()`'s staleness banner used a single hyphen
  as a clause separator (`"... last train (...) - retrain to include them."`), a dash-substitute
  this project's style rule bans. Rewritten as two sentences.

**Minor, not fixed (judged not worth a special case for a hypothetical the Important-finding fix
already closes, since the two code paths now share the same usable-count floor):** the review
also flagged that `_preference_retrain_gate_error()`'s docstring claim of mirroring
`train_and_save()` "exactly" only became true after the Important fix landed, and suggested a
couple of test-isolation and edge-case gaps. Addressed by adding 3 new tests (a fingerprint
winner/loser-swap test, a fingerprint duplicate-comparison test, and a regression test proving
`train_and_save()` now refuses when the raw count clears `MIN_COMPARISONS` but the usable count
doesn't) and adding the missing `embed_cache.EMBEDDINGS_FILE` monkeypatch to one pre-existing
test that hadn't isolated it.

Full suite after all fixes: 180 of 180 passed (3 new tests; the earlier 177 all still pass,
including the staleness-banner tests, whose lowercase "retrain to include them" assertions were
updated to match the corrected sentence's capitalized "Retrain to include them").

### 2026-07-11: Preference classifier significance, retraining, and staleness status (PR #9, legacy model)

The following entry is PR #9's own log of its work, merged in unchanged for the historical
record. PR #9 targeted `search/preference_model.py`, the legacy yes/no rating model this
branch's migration retires; its ideas were subsequently ported onto the pairwise model in the
two entries above, since both branches touched overlapping files with incompatible designs.

Added a 200-shuffle permutation test to preference-model training. A permutation test repeatedly
shuffles the yes/no labels and measures how often random labels score at least as well as the real
labels. The model metadata now records that test's p-value, the majority-class baseline accuracy,
and the shuffle count alongside cross-validation accuracy.

The preference status page now shows those statistics for newly trained models, explains whether
the p-value clears the 0.05 significance threshold, reports ratings added since the last training
run, and offers an inline retrain button. The server rejects retraining until the ratings clear the
existing total-count and class-balance gates, then trains under the write lock and returns fresh
status data. Old metadata remains readable and omits the new statistical rows.

Verification covered the model statistics, old and new status metadata, stale and current models,
the retrain UI, and successful and rejected POST requests. The successful route test runs the real
trainer against temporary separable embeddings rather than mocking training. The full suite passes:
149 tests, with no failures.

This was implemented by an autonomous `opencode`/gpt-5.6-sol run against a task brief in an
isolated worktree, then reviewed and fixed by hand before merge. Two issues surfaced in review.
First, the retrain route's gate check counted any yes/no-labeled rating as usable, but the actual
trainer only uses ratings whose tag also has a cached embedding; a rated tag missing from the
embedding cache could pass the gate and still fail training. Fixed by having the gate call the same
`build_training_set` the trainer uses, so the two can't disagree. Second, the retrain button's JS
disabled itself via `button['dis' + 'abled'] = true` instead of `button.disabled = true` for no
apparent reason; simplified to the plain form. The implementer's own environment also lacked a
project venv, and it worked around that by adding a `sys.path`-patching `tests/conftest.py` instead
of running `uv sync`; that file was deleted rather than merged, since a normal `uv sync --extra dev`
in a fresh worktree is the correct fix and this repo already relies on an editable install elsewhere.

### 2026-07-12: Independent GLM review of the merged head-to-head system (PR #10), six fixes shipped as PR #11

The head-to-head comparison system (the whole `head-to-head-compare` branch above) merged as PR #10
without an independent whole-diff review attached: its PR body cited per-task SDD reviews, but no
single review ran against the full ~2900-line code diff. Ran that review after the fact via
`delegate-code-review`: four parallel GLM 5.2 finder subagents (correctness, cross-file, reuse and
simplification, efficiency and conventions) orchestrated on opencode, then every candidate
hand-verified against the actual code before trusting it.

The project's number-one concern came back clean: the pairwise model, its metadata sidecar, and
`user_comparisons.json` all persist through a `.tmp` file plus `os.replace` (atomic), and nothing
deletes a cache to invalidate it. No data-safety regression.

Six real findings, all fixed and merged as PR #11 (`fix/head-to-head-review`, isolated worktree,
full suite 184 passing including two new regression tests):

1. **Touch tap recorded an unintended comparison (high).** `compare.html`'s zoom overlay omitted
   `e.preventDefault()` on touch, so closing zoom by tapping fired a synthetic click on the pane
   beneath it, posting a comparison the user never made. This is the exact synthetic-click gotcha
   the 2026-07-10 swipe-zoom entry above documented for `rate.html`; it regressed in the new page
   because `rate_page.py` was replaced wholesale rather than edited. Fixed by calling
   `preventDefault()` in the overlay `touchstart` (registered `{passive: false}`), and confirmed
   headlessly: a dispatched `touchstart` now reports `defaultPrevented === true`.
2. **`/api/compare` accepted `winner == loser` (medium).** A self-comparison produces a zero-vector
   embedding difference labeled both 1 and 0, contradictory training rows. Now rejected with 400.
3. **Auto-retrain could fail an already-saved comparison (medium).** The every-tenth-comparison
   retrain ran with no exception handling inside `/api/compare`; a training crash 500'd a request
   whose comparison was already persisted, inviting a duplicate on retry. Made best-effort, matching
   `/api/preference_retrain`.
4. **Unescaped `prompt_name` in `#meta` innerHTML (low-medium).** A prompt containing HTML broke the
   comparison page's metadata rendering. Now escaped.
5. **Em dash in shipped UI text (low).** `compare.html`'s done-state used `&mdash;`, against the
   project no-em-dash rule. Replaced with a period.
6. **False "nothing left to compare" (low).** When a trained model existed but no current manifest
   image had a cached embedding, `/api/compare/next` returned `{"done": true}` instead of falling
   back to random pairing. Now drops to stratified-random sampling.

Two findings were deferred as documented follow-ups, not fixed here: the synchronous retrain holds
the request lock (a correct fix moves training off the request thread), and the status page gates
"ready to train" on the raw comparison count while the retrain endpoint gates on the usable count
(reconciling them forces embedding I/O on every status load and a test rewrite). The bulk of the
reuse and dead-code candidates were refuted: the surviving `preference_model.py`, `rating_sampler.py`,
`user_ratings.json`, and their tests are the head-to-head plan's explicit "legacy stays on disk,
untouched, unused" constraint, not dead-code bugs.

Gotcha worth carrying forward: replacing a UI file wholesale (here `rate_page.py` to `compare_page.py`)
silently drops hard-won fixes that lived only in the old file. The touch `preventDefault` was fixed
once already on 2026-07-10; the rewrite lost it because nothing tied the fix to a test. The new
regression tests cover the two server-side behaviors, but a touch-device gesture check still isn't
automated in the suite.

### 2026-07-12: Curation-UI polish (compare, elite archive, hub) and a data-safety near-miss

Ran the live curation server against `notes/uncanny_seedrun1/` (100 real 1024x1024 PNGs, the only
sweep with images still on disk after the Task 12 loss) to try out the head-to-head page, and made
three UI improvements from that session, each its own commit with a regression test.

**Compare page.** On a phone the two images stack vertically, but both captions had rendered on one
shared row below both images, disconnected from the image each described. Moved each caption inside
its own pane so it tracks its image when the panes stack; desktop keeps them side by side. Added a
training-progress bar with two phases: below the 50-comparison floor it fills toward "Model unlocks
in N votes"; at or above it, it shows the model's real cross-validated accuracy ("Model reads your
taste: X%") mapped from the 0.5 coin-flip baseline to 1.0, and pulses on each tenth-comparison
retrain. The accuracy is the genuine `cv_accuracy` from `preference_pairwise_model_meta.json` read
via `/api/preference_status`, not a cosmetic animation, so the bar reflects the model actually
learning. Verified headlessly on desktop (side-by-side) and mobile (stacked) viewports, plus a live
vote that advanced the bar and loaded a fresh pair.

**Elite archive.** Each cell now labels the faithfulness and novelty range its bin covers (e.g.
`bin faith 2/4 (0.20-0.50) - novelty 4/4 (0.70-0.90)`), computed from the same quartile edges the
binning already used but never surfaced. The four bins per axis are population quartiles, so each
holds a similar share of images rather than an equal slice of the value range; the page now says so.

**Home page vs. nav dropdown drift.** The `explore.html` hub was missing three tools the jump-to
dropdown already listed: compare (head-to-head), predicted preference, and preference status, the
entire head-to-head feature. Added all three and reordered the hub to mirror `shared_ui.NAV_OPTIONS`,
with a test that fails if the two lists ever diverge again.

**Data-safety near-miss (record in full, per Section 1's standing rule).** Before serving I took a
complete-mirror backup of `notes/uncanny_seedrun1/` and verified it (208 files, 100 PNGs, `diff -rq`
identical). While serving, the user cast 46 real comparisons, which the server wrote to a
`user_comparisons.json` that did not exist when the backup was taken. Restoring that verified backup
would therefore have destroyed all 46 votes: the exact partial-backup-then-full-restore shape that
caused the Task 12 loss, one step from repeating. Caught it by checking backup-vs-current comparison
counts (backup: no file; current: 46) before any restore, snapshotted the 46 votes to scratchpad
immediately, and ran the write-path verification against a throwaway copy on a second port so no
automated vote ever touched the real data. The live 46 votes stayed intact. Lesson reinforced: a
backup only protects what existed when it was taken; re-verify before trusting it against live-mutated
data, never assume a mirror is still complete.

**Tooling gotcha, now in CLAUDE.md.** `fd` and `rg` skip `.gitignore`d files by default, and almost
every image on this project sits under a gitignored glob, so `fd -e png notes/uncanny_seedrun1`
reported zero PNGs while `ls` showed 100. A directory of real generation output looked empty and was
briefly mistaken for lost data. Use `fd -I` / `rg -uu` (or plain `ls`/`find`) to see ignored image
output; an empty default-tool result means "not tracked by git," not "not on disk."

**Concept for the whitepaper: how Stage 5b (predicted preference) composes with MAP-Elites.** They
are orthogonal. MAP-Elites owns diversity through its faithfulness x novelty archive; the trained
preference model never becomes an archive axis or changes the bins. Each generation splits into
explore jobs (fresh random subjects, which fill the archive across the whole grid) and exploit jobs
(mutations near a pool of good parents). Stage 5b changes only the exploit pool: instead of mutating
near favorited images or novelty-ranked elites (Stage 5a), it mutates near the top-N images the
preference model scores highest (`driver._predicted_preference_pool`). So preference is a
parent-selection signal for the exploit half only. Explore keeps mapping the frontier and the archive
still keeps one novelty-champion per bin; preference just biases where refinement concentrates toward
the user's taste. The tension to name in the paper: MAP-Elites deliberately preserves diversity
(including low-preference bins) while the preference model is a narrowing pressure, so they coexist
only because 5b steers exploitation, not exploration. Cranking the exploit fraction up with 5b on
would risk collapsing the archive's diversity toward the user's taste. This is why 5b is opt-in and
gated behind validating the model on `preference_rank.html` first.

**Repeated-image bug in the compare sampler ("I've seen this pig ten times").** Below the
50-comparison floor the sampler picked a grid bin uniformly at random, then a random image in it. A
bin holding one image drew that image with probability 1/n_bins; an image sharing a dense bin with
twenty others drew at 1/n_bins/20. So a lone-bin image reappeared far more often than a person reads
as random, and one kept recurring. Rewrote `stratified_random_pair` to be coverage-aware: it now
tracks how many past comparisons each image appears in (a `seen` tag->count map the server builds
from the comparison history) and restricts each draw to the least-covered frontier (the bins whose
least-shown image has the lowest seen-count), then the least-shown image in that bin. An image never
recurs until every other bin's least-shown image has been shown as often, spreading coverage evenly
across the archive while keeping the grid stratification. Two regression tests: an over-shown image
does not reappear while less-shown ones exist, and across a simulated 60-draw session every image
stays within one appearance of the minimum. Verified live: the previously over-shown tags did not
recur across 25 fresh pairs.

**"None of the pages do anything" was three separate things, none of them static files.** The user
reported the non-compare pages looked broken and expected them "all served from the Python web
server." I first mis-diagnosed this as missing static-JSON routes (`solution_map_data.json`,
`similarity.json`) 404ing. That was wrong, and I had asserted it without testing. Every page is
already rendered dynamically in-process (`view.render_html(view.compute_data(...))`); those old
build-artifact JSONs are gone and nothing reads them. The real causes, found by rendering each page
in a headless browser instead of theorizing:
- **lineage.html and novelty_decay.html are legitimately empty for this dataset.** seedrun1 is a
  single-generation seed run: all 100 images have `parent_tag: None` and `generation: None`, so there
  are no exploit chains to draw. Both pages already show an explanatory placeholder. They will
  populate on a real multi-round search, not a seed run.
- **redundancy.html rendered blank because its slider range was hardcoded.** The similarity-threshold
  slider ran 0.80 to 0.99 (default 0.93), tuned for near-duplicate detection in a tight
  multi-generation sweep. seedrun1's closest pair sits at cosine 0.776, below even the slider's
  minimum, so every slider position produced zero edges and an empty graph. Fixed
  `redundancy_view.render_html` to size the slider to the actual edge distribution (span the real
  min-to-max padded to a 0.05 grid, default where only the tightest ~5% of edges survive). seedrun1
  now spans 0.15-0.80, defaults to 0.60, and renders 4 real clusters (largest 17 images) out of 73
  effective clusters. Screenshot-verified.
- **map.html was never broken.** Its UMAP scatter paints 100 points onto a `<canvas>`; my first probe
  counted DOM `<circle>` elements and found none, a probe artifact, not an empty page. A second check
  confirmed 3033 painted pixels.

Root cause of the mis-diagnosis, now fixed in documentation: `curation_server.py`'s module docstring
still opened with "Static file server," legacy language from when it wrapped `python3 -m
http.server`. Rewrote the docstring to state the dynamic rendering model explicitly (every .html
builds in-process at request time; `/scan_data.json` is the one client-fetched companion; a
blank-looking page is empty-for-this-dataset or a client-side filter, never a missing file) so this
mistake can't recur.

**Added a three-prompt unfamiliar-subject seed file for noise-break testing.** Wrote
`notes/uncanny_seedrun1/candidate_seeds_gen_1783887674.json` with three short, concrete prompts:
an abandoned airport gate, wet shoes around a basement dehumidifier, and an empty bus shelter with
leaves. The set intentionally spans space, object-machine, and weather-street categories so the next
style-survival check tests more than one kind of off-distribution subject.

**Added another three-prompt unfamiliar-subject seed file for noise-break testing.** Wrote
`notes/uncanny_seedrun1/candidate_seeds_gen_1783887739.json` with a rain-lit laundromat, a sweating
birthday cake on a conference table, and fog-stalled commuters on a platform, spanning space, object,
weather, and crowd stress cases.

### 2026-07-12: Worked the curation-UI continuation prompt's outstanding-work list (items 1-4)

Picked up `notes/continuation_prompt_curation_ui.md` and worked through its ordered list against the
live server on `notes/uncanny_seedrun1/`. Full suite went from 192 to 195 passing (net: removed 3
gallery tests, added 6 new ones); every change verified live via headless Playwright, not just unit
tests, per the continuation prompt's own instruction.

1. **Removed the binned atlas (gallery.html).** Deleted its route in `curation_server.py`, the
   `uncanny_gallery.py` view module, its `shared_ui.NAV_OPTIONS` and jump-to-dropdown entries, and its
   `explore_hub.py` card. `search/driver.py`'s separate offline per-round gallery (a static file it
   writes to the sweep dir, unrelated to the live server) still needed `thumb_data_uri`, so that
   function moved to `build/thumbnails.py` instead of being deleted with the rest of the module. Live-
   verified: `/gallery.html` 404s, the nav dropdown and hub no longer mention it, and the 52 real votes
   in `user_comparisons.json` survived the server restart untouched.

2. **novelty_decay.html now has an explicit empty-state placeholder**, matching lineage.html's pattern:
   when no prompt family has appeared in 2+ generations (true for seedrun1, a single-generation seed
   run), it explains why there's nothing to plot instead of rendering a blank chart. Added a regression
   test asserting the placeholder renders when `compute_data`'s series list is empty.

3. **seeds.html: real bug found and fixed, not just an empty-state issue.** The empty page traced to
   `curation_server.save_store` doing `path + ".tmp"`, which breaks when `path` is a `pathlib.Path`
   (as `SEEDS_FILE` is, unlike every other `*_FILE` constant in the module, which are f-strings). The
   first live test of the "Generate" button actually called GPT-5.5 successfully, then silently lost
   the result to this `TypeError` when persisting it, a real cost (one wasted GPT-5.5 call) worth
   recording so it isn't repeated. Fixed with `str(path) + ".tmp"`, added a regression test that saves
   through a `Path` object, and a second test that exercises the full `/api/seeds/generate` request
   with a mocked `opencode` subprocess call. Re-ran the real end-to-end flow live afterward: GPT-5.5
   returned three real seeds (an empty laundromat, a bus shelter of damp umbrellas, a storm-lit cul-de-
   sac), all three persisted and rendered. Also added a clearer "no candidate seeds yet, use Generate
   above" empty-state message for the zero-seed case itself.

4. **Map hover now shows the actual nearest real training image, not just its filename and similarity
   score.** Added a read-only `/real/<name>` route to `curation_server.py` that serves from
   `search/score_manifest.REAL_DIR`, sanitized with `os.path.basename()` so a path-traversal attempt in
   the requested name can only ever resolve to a direct child of `REAL_DIR` (regression test covers
   this: a `..%2F..` request 404s instead of escaping the directory). `map_view.py`'s hover panel now
   renders that image inline below the hovered point's own thumbnail, captioned with its similarity
   score. Live-verified via `showInfo()` in a headless browser: hovering a point resolved
   `nearest_real: FlzK3OUXoAEvv3e.jpg` to `/real/FlzK3OUXoAEvv3e.jpg`, which rendered as a visible
   image in the panel, not a broken-image icon.

**Tooling gotcha: the Playwright MCP server's `--executable-path` was hardcoded to `/home/node/...`**,
a stale path from a different container image, while the browser binary actually lived under
`/home/jeremy/.cache/ms-playwright/...` in this environment. Fixed by removing the hardcoded
`--executable-path` from `~/.claude.json`'s `mcpServers.playwright.args` entirely, letting it
auto-detect `$HOME/.cache/ms-playwright` instead. A stale Chrome `SingletonLock` from an earlier failed
launch attempt also had to be cleared by hand (`rm` on the three `Singleton*` files under the relevant
`mcp-chrome-*` profile dir) before the fixed config would launch cleanly.

**Added a placeholder favicon.** Generated via FLUX (`flux-prompting` skill, `fal-ai/flux-pro/v1.1-
ultra`) in the CLAWMARKS mixed-media style (a bold-ink fox face on aged cream paper), resized to
128x128, and saved to `src/clawmarks/static/favicon.png`. `curation_server.py` now serves it at both
`/favicon.ico` and `/favicon.png`, clearing the favicon-404 console noise every page previously threw.

### 2026-07-12: CI/CD, CI-gated Docker build, and a Watchtower compose stack for curation_server.py

Scoped to CLAWMARKS only, and to `curation_server.py` only within it (the RunPod driver and
generation scripts stay host-side; only the always-on curation web server gets containerized).
Modeled on `/workspace/hearth`'s existing pattern.

- **`CLAUDE.md`, "Running tests" section**: run only the touched test file while iterating, the
  full suite before calling a change done, and a live Playwright check for any UI change even
  after the suite passes.
- **`.github/workflows/check.yml`**: `uv sync --extra dev` + `uv run pytest -q` on every PR and
  push to `main`. Confirmed the exact commands succeed locally first (195 passed, 52.92s) before
  trusting them in CI.
- **`Dockerfile`**: `python:3.12-slim` + `uv`, `COPY . .`, `uv sync --frozen --no-dev`, runs
  `python3 -m clawmarks.curation_server 8420`. `.dockerignore` excludes `notes/` and
  `corrected_dataset_extract/` outright: this project's data-integrity rule says irreplaceable
  RunPod-billed generation output and real training images must never end up baked into a Docker
  layer. Because `notes/` still isn't in the build context but `pyproject.toml` is,
  `clawmarks.config.repo_root()` resolves `ROOT=/app` automatically with no `CLAWMARKS_ROOT`
  override needed; the served dataset comes entirely from a bind mount at runtime.
- **`docker-compose.yml`**: three services mirroring hearth's shape (`tailscale` sidecar,
  `app`, `watchtower`), except plain HTTP instead of hearth's TLS cert mount, since
  `curation_server.py` doesn't terminate TLS itself. `./notes` and
  `./corrected_dataset_extract` are bind mounts, not named volumes, again per the data-integrity
  rule: a bind mount keeps this data directly reachable by the project's existing
  backup/verify workflow, where a named volume would hide it inside Docker's storage driver.
  Image is `ghcr.io/jeremysball/clawmarks-lora`, matching the actual GitHub repo name.
- **`.github/workflows/build.yml`**: `verify` builds the image on every PR without pushing
  (catches a broken Dockerfile before merge); `publish` pushes to GHCR tagged `latest` and
  `sha-<short>`, gated on `check` having already succeeded on `main` (`workflow_run` trigger),
  matching hearth's two-job pattern exactly.
- **`.env.example`** documents the compose stack's required secrets (`TS_AUTHKEY`,
  `RUNPOD_API_KEY`, `CIVITAI_TOKEN`, `CIVITAI_MODEL_ID`, `OPENAI_API_KEY`) and
  `CLAWMARKS_SWEEP_DIR`; `.env` added to `.gitignore` alongside the existing `.envrc` entry.

**Known gap, resolved**: seeds.html's "Generate" button shells out to the `opencode` CLI, which
on this host authenticates via an interactive OAuth login (`opencode auth login`, stored in
`~/.local/share/opencode/auth.json`). Jeremy's call: don't mount that credential file into the
container; instead the image installs `opencode` directly (official install script, in the
Dockerfile) and authenticates headlessly via `OPENAI_API_KEY`. Checked against opencode.ai's own
docs before wiring it in: `OPENAI_API_KEY` is a documented, supported alternative to the OAuth
flow for exactly this kind of headless/CI/Docker use, not a guess. `docker-compose.yml` and
`.env.example` both pass it through the same way the other secrets already do.

### 2026-07-12: Design for unified detail view, real-image viewing, and thumb-then-full-res loading

Wrote `docs/superpowers/specs/2026-07-12-detail-view-and-generation-design.md`, per the
continuation prompt's item 5 ("plan, don't build, a unified image-detail + generate-around UI")
and two of the follow-up batch's requests (view the full real reference image on the map; a
generalized thumb-then-full-res swap API). Folded all three into one design since they're the
same underlying investment, not three separate features.

The useful finding was that most of item 5 already exists: `shared_ui.py`'s Lightbox component
already gives seven of the eleven tool pages (scan, map, archive, preference_rank, redundancy,
lineage, coverage) a linked detail modal with single-shot generate-around baked in. The actual
gaps are narrower than the original ask implied: `compare.html` has no detail access at all,
`/api/counterfactual` only ever produces one variation per click, and nothing in the codebase
does progressive thumb-then-full loading, including this session's new `/real/<name>` route,
which serves the real training photo at full resolution with no thumbnail stage. The spec
proposes a small `compare.html` expand-icon into the existing Lightbox, an `n`-variation
extension to the counterfactual endpoint capped at 6 per batch, and a `mountProgressive()` helper
wired into both the Lightbox's main image and a new `/real_thumbs/<name>` route for the map
panel. Design only; nothing implemented yet, tracked in `TODO.txt`.

**Not done this session**: branch protection on `main` (needs `check` to have at least one run
on GitHub, which needs this branch pushed/merged first) and an actual `docker build` (no
`docker` binary in this sandbox; the first real build will be CI's `verify` job on the first PR
touching `Dockerfile`/`docker-compose.yml`). Also unresolved: whether "explain in hearth's
readme why docker compose uses watchtower" survived the "CLAWMARKS only" scope-down (answered in
chat, not written into `hearth/README.md`), and what "what is the real key" referred to.

### 2026-07-13: Added unfamiliar-subject seed prompts for style-break testing

Created `notes/uncanny_seedrun1/candidate_seeds_gen_1783953823.json` with 20 short visual scene
descriptions for testing whether the fine-tuned style survives across unfamiliar subjects or
breaks into visual noise. Before writing into `notes/uncanny_seedrun1/`, made a complete mirror
backup at `/tmp/opencode/uncanny_seedrun1_backups/uncanny_seedrun1_20260713_candidate_seeds_gen_1783953823`
and verified it with `diff -qr`. (Logged retroactively on 2026-07-15 from an uncommitted stash
found during a branch/worktree cleanup pass; the generated-timestamp file itself was later folded
into `notes/uncanny_seedrun1/candidate_seeds.json`, which is what's on disk now.)

### 2026-07-13: Corrected finite permutation p-values and paired-seed power analysis

Corrected two statistical record errors before round 1. `notes/mmd_score.py` now reports the
finite Monte Carlo p-value `(b + 1) / (B + 1)`, where `b` counts shuffled MMD statistics at least
as extreme as observed. The minimum with 2000 shuffles is therefore `1/2001`, never zero. The
focused test proves the floor with ten shuffles.

Added `notes/probe_power.py`. Its independent unit is one paired training seed, so the planned
sample is the eight canonical seeds from `notes/train_probe.py`. Prompt rows and mirrored deltas
are measurements within a seed and do not increase n. For n=3, 4, 6, and 8, the program enumerates
all `2^n` sign flips instead of sampling duplicate patterns, prints the exact one-sided p-value
floor, and runs 10,000 deterministic simulations for the null and positive controls. The planning
    model uses the old unpaired checkpoint-mean SD of 0.0354 and an unverified delta-SD planning
    proxy `sqrt(2) * 0.0354 = 0.050063`; paired round-one deltas may differ.

| quantity | n=3 | n=4 | n=6 | n=8 |
|---|---:|---:|---:|---:|
| one-sided p-value floor | 0.125000 | 0.062500 | 0.015625 | 0.003906 |
| null rejection rate | 0.0000 | 0.0000 | 0.0434 +/- 0.0020 | 0.0484 +/- 0.0021 |
| power at effect 0.05 | 0.0000 | 0.0000 | 0.6485 +/- 0.0048 | 0.7990 +/- 0.0040 |
| power at effect 0.08 | 0.0000 | 0.0000 | 0.9381 +/- 0.0024 | 0.9920 +/- 0.0009 |

Impact: the old n=3 and n=4 significance entries were impossible and are retracted. The old
claim of 84% power at n=8 and effect 0.05 was also wrong under the corrected program. Round 1
retains the eight canonical paired seeds, treats 0.05 as an exploratory practical threshold, and
    uses 0.08 as the prespecified effect for an 80%-power per-direction screening claim. The null rates for n=6 and
n=8 are close to alpha=0.05; n=3 and n=4 cannot reject at that alpha because their exact floors
are too large.

The power numbers are simulation results from code, not measurements from new training runs. They
remain limited by the unpaired-to-paired noise assumption, the small planned n, the normal-delta
model, and the lack of a multiplicity correction for the full set of directions. The MMD correction
fixes finite-sample arithmetic but does not make MMD a style verdict. The overnight novelty
trajectory remains descriptive and selection-biased until a per-generation cohort statistic or an
untouched replay comparison is run.

**Second review pass, same day.** The table above reports the sign-flip test's own power
(p<=alpha only), which is not round 1's actual decision rule; see the "Corrected again,
2026-07-13" note earlier in this section for the real gate numbers (49.42% at 0.05, 95.34% at
0.08). `probe_power.py` and `mmd_score.py` also gained input validation this pass: non-finite
MMD statistics, groups under 2 members, and zero/non-finite kernel bandwidth now raise instead
of silently producing a misleadingly small p-value or a division-by-zero. One limitation is
still open, not yet fixed: the image-level permutation test in `mmd_score.py` treats every
generated image as an independent exchangeable unit, which does not hold if images share a
prompt, seed, or checkpoint in a way that correlates their embeddings. Until that is checked
(or the claim is scoped down to "these two fixed image collections"), treat the MMD p-value as
suggestive, not a rigorously calibrated significance level.

### 2026-07-14: Phase 2 open-threads cleanup (transactional writes, retrain-off-lock, N-variation counterfactuals, search-run launch UI)

Closed out five items that had been sitting half-finished across the curation server and
recovery scripts, each landed as its own TDD task with an independent GLM review:

- Recovery scripts (`notes/recover_*` style atomic-write helpers) now accumulate quarantine
  entries across runs instead of overwriting the ledger each time, and widen the cleanup
  `except` so a partial write can't leave an orphaned temp file behind.
- The preference-model retrain triggered by a rating submission no longer runs synchronously
  inside the request lock (the deferred follow-up noted in the 2026-07-12 entry above); a rating
  POST returns immediately and the retrain runs off-lock.
- `/api/counterfactual` now accepts `n` (batch generation, capped at 6) and the Lightbox gained
  `mountProgressive()` for thumb-then-full-res loading. Two bugs surfaced by GLM review and
  fixed before merge: a pinned seed was generating `n` byte-identical images instead of forcing
  a single job (the design spec's own pseudocode says one job when the seed is pinned, since
  paying RunPod for identical copies is pure waste), and two concurrent counterfactual requests
  for the same origin/batch-index could collide on the same output filename (fixed with a uuid
  suffix on `new_tag`). `mountProgressive` also had a stale-callback race: the shared lightbox
  `<img>` element is reused across navigations, so a superseded full-res load could still land
  after the user navigated elsewhere and clobber the newer image; fixed with a per-call token
  checked in both `onload` and a new `onerror` handler.
- Implemented docs/superpowers/specs/2026-07-12-overnight-search-launch-design.md: `runs.html` +
  `run_manager.py` let a search round be launched, monitored, and stopped from the browser
  instead of SSHing in, with a per-run report (novelty trajectory, plateau count, spend, pick
  rate by category, explore/exploit split) read straight off `allnight_state.json`/
  `scored_manifest.json`. A second GLM review pass on this one caught real safety-rail gaps
  worth recording since they're the kind of thing that looks fine until two requests race:
  the original lock acquire was check-then-write (TOCTOU), so two near-simultaneous launches
  under the server's `ThreadingHTTPServer` could both pass the "already running" guard and spawn
  two `driver.py` processes against the same `out_dir`; fixed with an atomic
  `O_CREAT|O_EXCL` lock claim. If writing the lock after spawning failed, the child was left
  running with no lock at all, silently defeating the one-run-at-a-time guarantee; fixed by
  reaping the child on any failure after Popen. `stop_run` trusted a bare PID-alive check, so a
  reused PID after the real driver died would get SIGKILL'd as if it were the driver; fixed by
  recording the launched PID's `/proc` start time and treating a mismatch as a stale lock.
  `stop_run` also only signaled the driver's own PID, not its process group, so a `driver.py`
  mid-`opencode` subprocess call at stop time would orphan that child; fixed with
  `killpg`/`getpgid` (safe here since the driver is started with `start_new_session=True`, so
  its pgid equals its own pid).
- Deleted two rejected specs (`2026-07-11-toml-config-design.md` TOML-config, `2026-07-11-ui-
  redesign-design.md` three-pillar nav) and added `*.backup_candidate_seeds_*` to `.gitignore`.
  TODO.txt reconciled to match: the "Curation UI" implement items, the retrain-off-lock deferred
  follow-up, and both search-run-launch UI items are now checked off there.

A second GLM review pass (on the safety-rail fix commit itself, before this hygiene commit
existed) caught one more real gap: `launch_run`'s `Popen` object goes out of scope right after
spawning, so nothing ever `wait()`s the driver process. Once it exits it sits as an unreaped
zombie, and `is_process_alive`'s `os.kill(pid, 0)` check reports zombies as alive — so
`stop_run`'s SIGTERM-then-SIGKILL grace loops were each spinning their full duration (~20s
total at the 10s default) even for a driver that exited immediately. Fixed with an
`os.waitpid(pid, os.WNOHANG)` reap between poll iterations, verified with a test that lets a
process exit into zombie state *without* ever calling `.poll()`/`.wait()` on it (either would
have reaped it and defeated the test) and asserts `stop_run` returns in well under the grace
period. Two minor, non-blocking gaps remain, logged here rather than fixed, since GLM's own
severity read on both was "pre-existing, negligible probability, worth a follow-up, not a
blocker": `stop_run` doesn't re-check `start_time_ticks` after reaping and before the SIGKILL
phase (a pid-reuse race in that narrow post-reap window); and `_reap_if_exited` is only called
from `stop_run`, so `status()`/`current_run()` still report a spontaneously-exited-but-unreaped
driver as `running: True` until someone clicks Stop.

Full suite: 335 passing, ruff/mypy clean. All work landed on the
`worktree-phase2-task6-transactional-writes` branch; per-task branch/PR split from the original
plan was not followed this pass (flagged for the finishing-a-development-branch step).

### 2026-07-14 (later): three findings from a self-run branch review, fixed before merge

PR #31's own automated review (GLM-5.2 via opencode) hung twice in a row mid-run and had to be
cancelled both times, so this pass was read directly instead of delegated. Reading the diff
turned up three real issues, each fixed TDD-style:

- **Auto-retrain still held the request lock.** The manual `/api/preference_retrain` endpoint's
  fit was already moved outside `_lock` in the original Phase 2 work, but `/api/compare`'s
  auto-retrain (fires every `RETRAIN_EVERY`th comparison) still ran the full model fit inside
  `with _lock:`, unchanged from main — exactly the bottleneck that task was supposed to close,
  left in place on the path that fires during ordinary browsing rather than the manual button.
  A concurrency test (spawn a slow-fit thread, assert a concurrent `/api/compare` still returns
  in well under the fit's duration) reproduced a genuine 30-second block before the fix.
- **Non-unique temp path in `train_and_save`.** Once both retrain paths fit outside the lock,
  two calls can genuinely overlap — and `train_and_save` wrote to a fixed `f"{MODEL_FILE}.tmp"`
  path, so two concurrent fits would race on the same file. A test forcing two `train_and_save`
  calls' `joblib.dump` to be in flight simultaneously reproduced a `FileNotFoundError` from one
  call's `os.replace` racing the other's. Fixed by routing the model and meta writes through the
  existing `atomic_io` helpers (unique `tempfile.mkstemp` path per call).
- **RunPod API key sent as a URL query param.** `runpod_client.runpod_balance` built the request
  as `?api_key=<key>`, which can end up in server access logs or proxy history in a way a header
  doesn't. Switched to `Authorization: Bearer <key>`. Verified live against the real RunPod
  GraphQL API (not just the mocked unit test) that the header works identically to the old query
  param — the request still needs the spoofed `User-Agent: curl/8.0` either way, which a first
  verification attempt without it revealed by getting a false 403 on *both* auth styles.

Full suite: 338 passing (three new tests), ruff/mypy clean.

### 2026-07-14 (session 3): confirmed both sweep directories' images gone for good, moved
### runtime state out of the repo, and rebuilt the server's empty-state experience

Standing up the curation server in this branch's worktree surfaced a chain of problems, each
uncovering the next.

**The server hung forever on every request.** `notes/uncanny_sweep/` was empty in this worktree
because git worktrees don't copy gitignored files (the generated images were never tracked).
Pointing `CLAWMARKS_SWEEP_DIR` at the main checkout's copy for diagnosis surfaced the real bug:
`scored_manifest.json`'s `file` fields still held absolute paths from before the project's
`trent-with-smart-prompts` -> `clawmarks` rename, so opening any image raised an uncaught
`FileNotFoundError` inside `do_GET`. An unhandled exception in a request thread just resets the
TCP connection, so the browser sat there loading forever with no error shown at all.

**Confirmed round 2's images are gone too, the same way round 1's were.** Re-checking
`notes/uncanny_sweep2/` against the 2026-07-09 incident logged above found it still at 3.0 MB
(matching that entry's "544 MB to 3 MB" figure exactly), 0 of 280 manifest images present on
disk. A fresh exhaustive search (every mount, an archive/backup filename sweep, a search for two
specific destroyed filenames across the whole filesystem, RunPod pod status) turned up nothing,
same as the 2026-07-09 search. Both directories' full-resolution PNGs are permanently gone; only
the JSON metadata and downscaled thumbnails survived, as already documented.

**Backed up, verified, and deleted both directories**, per this project's backup-verify-delete
rule: `cp -a` mirrors to `/workspace/clawmarks-backups/`, `diff -rq` plus file-count checks
against the live directories before removing them (`uncanny_sweep`: 3701 files, 95 MB;
`uncanny_sweep2`: 4 files, 3.1 MB). The prior 2026-07-14 CLAUDE.md entry documenting this
incident and update predates this note but describes the same deletion.

**Moved all sweep/probe runtime state out of the repo**, per the user's global XDG Base
Directory convention: `notes/uncanny_sweep`, `notes/uncanny_sweep2`, `notes/probe_uncanny`, and
`notes/probe_strength` now live at `$XDG_STATE_HOME/clawmarks/` (`~/.local/state/clawmarks/` by
default, overridable via `CLAWMARKS_STATE_DIR`), renamed `uncanny_round1/` and `uncanny_round2/`
since the old `_sweep`/`_sweep2` numbering read as an accident of history rather than a
deliberate choice. `config.py` gained `STATE_DIR`; `SWEEP_DIR`/`SWEEP2_DIR`/`PROBE_DIR`/
`PROBE_STRENGTH_DIR` now derive from it. The two probe directories (32 MB and 39 MB) were moved
with `cp -a` + verify + `rm -rf` of the originals, same as the deletion above.

**Fixed the server to fail fast and fail visibly instead of hanging.** `do_GET` now catches
every exception: browser requests get an HTML error page with a collapsible stack trace and,
for a `FileNotFoundError`, a hint pointing at a stale manifest path; `/api/*` and `*.json`
requests get a well-formed JSON error body instead, so `fetch().then(r => r.json())` doesn't
choke on an HTML page it can't parse. Startup now also checks that `scored_manifest.json`'s
image paths actually resolve and exits with a clear message if none do, rather than discovering
the same stale-path problem mid-request.

**Redesigned the empty-state UX after user feedback that the old root page was unhelpful.**
The root page used to 302-redirect to `scan.html`, which on empty data just showed a blank
gallery with no indication of what to do next: criticized directly as "a terrible user
experience" for not helping the user create the files or generations they needed. The server
already had a full safe launch flow (`run_manager.py` / `/api/searchrun/launch`: backup,
verify, RunPod balance check, refuse concurrent launches, detached `driver.py`) built for
`runs.html` in an earlier session, so the fix reuses it: the root page now renders a hub with
"Launch Round 1" / "Launch Round 2" buttons when no manifest images are present, and falls back
to the previous manifest-summary-plus-tool-links view once a round has produced images.
`cockpit.py`'s target-cells fetch also picks up the new JSON error body's `no_manifest` flag to
show a "launch a round" link instead of a generic failure message on the same empty state.

Verified via Playwright: the hub page renders exactly per the design agreed with the user, no
console errors; `cockpit.html`'s empty-state message renders correctly and links back to `/`.
Did not click either launch button during verification, since a real launch spends RunPod money
and takes hours. Full suite: 340 passing.

### 2026-07-14 (session 4): isolated mutable defaults in `load_leg_config`

Task 2 review found that `load_leg_config` shallow-copied its module-level defaults dictionary.
The list-valued defaults, including `widened_textures`, were therefore shared by every leg that
left those fields unset. A regression test reproduced the leak by appending to one loaded config
and observing the mutation in a second config. `copy.deepcopy` now gives each load independent
list objects. The new regression test and the five-test `load_leg_config` slice both pass.

### 2026-07-14 (session 5): added the missing state-history regression coverage

Task 3 review identified that the existing round-1 state-resume test used exact-length history and
therefore would have passed before the legacy validator shim was removed. Added direct validator
coverage for both a 49-entry history at generation 50 and the actual historical exception, a
51-entry history at generation 50. The latter proves the removed round-1 shim no longer accepts
`generation + 1` history entries. Renamed the shared state filename test to describe its current
leg-independent behavior. The targeted driver-state suite passes with 29 tests.

### 2026-07-14 (session 6): made startup safe before leg selection

Task 11 added a regression test for starting `curation_server.py` with no active expedition or
leg. Before the fix, `_check_manifest_images()` dereferenced the `None` returned by
`_active_out_dir()` and raised a `TypeError`. The function now returns immediately in that empty
state, allowing the empty-state hub to handle startup. The RED test failed with the expected
`TypeError`; the GREEN test passed with one test. The full suite still has 276 passing tests,
21 failures, and 47 errors from older tests that still monkeypatch the removed `SWEEP_DIR` and
related legacy configuration names.

### 2026-07-14 (session 7): replaced the hardcoded round launch hub with expedition and leg selection

Task 12 added `_list_expeditions` and `_create_expedition` to the curation server. Creating an
expedition now writes its shared `expedition.json` atomically, scaffolds an empty `cockpit` leg
configuration, and creates that leg's runtime directory. The server now exposes `GET /api/expeditions`
and `POST /api/expeditions`, and the empty-state page lists every configured leg and selects one
through the existing `/api/active-leg` route instead of offering hardcoded Round 1 and Round 2
launch buttons. The new four-test expedition route slice passed after the expected RED failure.
The clean import check passed. The full suite remains blocked by the known older tests that still
reference removed `SWEEP_DIR` and related legacy configuration names.

### 2026-07-14 (session 8): gave cockpit trials sibling-leg novelty exclusion

Task 14 now pools every other leg's scored images in the selected expedition and embeds them with
the cockpit server's existing long-lived DINOv2 model. Cockpit scoring passes that pool to
`score_batch` as prior embeddings, so a cockpit trial's novelty score penalizes duplication of
work from sibling legs without reloading DINOv2 for each trial. The cockpit page also lists
expeditions and switches the active selection to that expedition's standing `cockpit` leg through
the existing `/api/active-leg` route. Both focused tests and the clean import check pass. An
isolated live server check confirmed the selected option and POST response without touching real
generation state. The full suite still has 290 passing tests, 21 failures, and 39 errors from
legacy tests that reference removed fixed-path globals.

### 2026-07-14 (session 9): isolated cockpit trial writes from active-leg switches

Task 14 review found that a background cockpit trial repeatedly resolved the mutable active leg
while generation ran. Switching legs during the trial could split its images, thumbnails,
manifest records, and queue status across two legs. The launch handler now snapshots the
expedition, leg, output directory, and queue file before starting the worker. Every worker read
and write, including sibling-leg novelty exclusion, uses that snapshot. Opening `cockpit.html`
also selects the active expedition's standing `cockpit` leg before rendering.

A route-level regression blocked mocked generation, switched from `cockpit` to `round1`, then
confirmed the image, thumbnail, manifest entry, and completed queue record all remained under
`cockpit`; `round1` received no trial files. Direct sibling-manifest coverage confirms missing
files are excluded before embedding. The required focused suites pass with 9 cockpit/active-leg
tests and 28 run-manager tests, and clean import passes. The full suite remains at the known
migration baseline with 293 passing tests, 21 failures, and 39 errors from legacy fixed-path
fixtures.

### 2026-07-14 (session 10): finished the expedition/leg migration, full suite green

Task 17 closed out the migration. The remaining stragglers were the last handful of tests still
monkeypatching module-level `SWEEP_DIR`-era globals (`MODEL_FILE`, `EMBEDDINGS_FILE`,
`PREFERENCE_SETTINGS_FILE`) instead of passing a leg directory, in
`test_preference_rank_live.py` and `test_preference_status.py`. Migrating those, plus fixing a
stray docstring reference to the removed `RoundConfig`, cleared the last of the legacy failures:
`rg -n "SWEEP_DIR|SWEEP2_DIR|ROUND_CONFIGS|RoundConfig|--round\b" src/ tests/` now returns
nothing, and `uv run python -m pytest -q` passes all 357 tests with zero failures or errors.
`ruff check` and `mypy src` both came back clean after fixing a leftover mid-file import in
`curation_server.py` and two unused local variables in test setup code.

The live smoke check caught two startup bugs the test suite's mocks hadn't exercised, both from
code paths that only run against a real, freshly-created state directory with no expedition
selected yet:

- `_reconcile_stuck_trials()`, called unconditionally from `main()` at every startup, called
  `_cockpit_queue_file()` before any leg was ever selected, and crashed the whole server with
  `TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'` before it could even bind
  a socket. `_check_manifest_images()` already had the "no leg selected yet" guard from Task 11;
  `_reconcile_stuck_trials()` needed the same one and didn't have it.
- The status page's `_send_status_page()` called `load_manifest()` unconditionally too. It didn't
  crash (the broad `except Exception` around it caught the same `TypeError`), but it rendered the
  empty-state hub with the confusing message "could not read manifest: unsupported operand
  type(s) for /: 'NoneType' and 'str'" instead of a clean "no expedition/leg selected". Fixed
  with the same early-return guard, and it now shows the right message.

Both gaps got regression tests in `test_curation_server_startup.py` before the fix, confirmed
RED, then GREEN after: one calling `_reconcile_stuck_trials()` directly with no active selection,
one spinning up a real `HTTPServer` and asserting the rendered status page contains "no
expedition/leg selected" and not "could not read manifest". Full suite re-run after both fixes:
357 passed.

The rest of the manual smoke check passed clean: the empty-state hub listed the `uncanny_frontier`
reference expedition's three legs (from Task 16), `POST /api/expeditions` created a test
expedition and returned `{"ok": true, ...}`, `POST /api/active-leg` switched to it, and
`/cockpit.html` returned 200. The test expedition directory created during the check
(`expeditions/smoke_test/`) was deleted afterward; it was never committed.

The expedition/leg migration (Tasks 1-17) is done. `main` still has the old `SWEEP_DIR`/
round1/round2 model; merging this branch is the next step, not yet done as of this entry.

### 2026-07-15 (session 11): GLM-5.2 review of PR #33, fixed 5 confirmed regressions before merge

Ran a 4-shard opencode/GLM-5.2 code review of PR #33's full diff (56 files, ~6100 changed lines)
via the `delegate-code-review` skill, dispatched over the `taskferry` MCP tools. All 4 shards
(angles A/B, C, D/E, F/G/H) completed cleanly. Merged their candidate lists, hand-verified every
one against the actual worktree code, and posted both the raw finder output and the verified
findings as PR comments for the record.

Five findings were CONFIRMED as real bugs distinct from the two no-leg-selected crashes already
fixed in session 10, and all five got fixed before merging:

1. **`docker-compose.yml`** set a dead `CLAWMARKS_SWEEP_DIR` env var and never mounted or set
   `CLAWMARKS_STATE_DIR`, so a docker-composed deployment would write every generated PNG and
   manifest to unmounted, ephemeral container storage, lost on the next `watchtower` recreate.
   This is a real, live deployment path (confirmed via this notebook's own docker-compose
   reference entries), not dead code, so it was a genuine data-integrity risk per this project's
   #1 rule. Fixed: set `CLAWMARKS_STATE_DIR=/app/state/clawmarks` and bind-mount `./state`.
2. **Seed pool split**: the curation server's "Generate seeds" UI wrote to
   `candidate_seeds.json`, while `search/driver.py` reads/writes `seed_pool.json` for the same
   per-leg subject pool. These used to be one shared file; the migration silently split them, so
   seeds a user topped up from the UI never reached a run. Fixed by pointing `_seeds_file()` at
   `seed_pool.json` (both stores use the same dict-of-subject shape, confirmed by reading
   `search/seed_pool.py`'s `load`/`save` against `curation_server.py`'s `load_store`/`save_store`).
3. **`runs_page.py` never migrated**: the only UI for launching/monitoring searches still posted
   `{round: int}` and fetched `?round=N` against endpoints that now hard-require
   `expedition`+`leg` and return 400. Rewrote the page to fetch `/api/expeditions` and populate
   expedition/leg `<select>`s, matching the pattern the status page already uses. Verified live
   with Playwright: pickers populate from the real `uncanny_frontier` expedition's three legs, no
   console errors.
4. **`/api/searchrun/report`** read favorites via `_favorites_file()` (the globally *active* leg)
   instead of the leg named in the `?expedition=&leg=` query, so a report for a non-active leg
   silently computed pick-rate-by-category against the wrong leg's favorites. Fixed to load
   favorites from the queried `out_dir` directly.
5. **No-leg-selected crashes, siblings of the two already fixed in session 10**: `load_manifest()`
   and half a dozen `*_file()` helpers, the `/thumbs/`+`/real_thumbs/` handlers, `_embeddings_for`,
   and `_handle_cockpit_run` all dereferenced `_active_out_dir()` with no `None` guard, each
   raising a raw `TypeError` (500, confusing stack trace) instead of a clean "no leg selected"
   response. Rather than patch each call site individually, added a single `_require_out_dir()`
   helper (raises `NoActiveLegError`) and a matching top-level `except NoActiveLegError` in both
   `do_GET` and `do_POST` (the latter previously had no top-level exception handler at all) that
   returns a clean 400. Every one of these call sites now routes through it.

One CONFIRMED finding was deliberately *not* fixed the way the review suggested: `GET
/cockpit.html` silently switches the globally active leg to `(expedition, "cockpit")` on page
load, which can redirect a concurrently open tab's writes to the wrong leg. Every cockpit route
(queue, target_cells, evidence, run) resolves its working directory off that same global active
leg, so the switch is load-bearing, not accidental; removing it without also decoupling every
cockpit route from global active-leg state would trade a UX surprise for a functional break, and
that decoupling is a bigger structural change than this pass's scope. Documented the coupling
in a code comment instead and left it as a known tradeoff for a future, dedicated pass.

Two of the fixes (`EMBEDDINGS_FILE` AttributeError in `preference_rank.py`/`elite_archive.py`,
`_manifest_path`/`*_file()` None-guards) had zero prior test coverage; added regression tests for
both classes (`test_preference_rank.py`, `test_elite_archive_predicted_preference.py`,
`test_curation_server_startup.py`) confirmed RED before the fix, GREEN after. Full suite: 363
passed (up from 357). `ruff check` clean. Live smoke check against the real
`$XDG_STATE_HOME/clawmarks/` state (read-only GETs and one active-leg switch, restored to
"none selected" afterward) confirmed the no-leg-selected paths now return clean 400s instead of
500s, and that the remaining 500s on the real `cockpit`/`round1` legs are genuine
`FileNotFoundError`s from those legs never having been run (no `scored_manifest.json` yet) rather
than the None-guard bug class, i.e. expected behavior, not a regression.

Posted the raw finder-stage output and the hand-verified findings as two separate comments on
PR #33 per the `delegate-code-review` skill's posting convention. Next: merge PR #33 and retire
the worktree.

### 2026-07-15 (session 12): resolved the merge conflict with origin/main's round-based model

`origin/main` had advanced past PR #33's fork point via an already-merged PR #32
("friendly errors, XDG state relocation, and a launch-hub empty state"), which independently
renamed and kept the round-based model (`RoundConfig`/`ROUND_CONFIGS`, `SWEEP_DIR`/`SWEEP2_DIR`
under `uncanny_round1`/`uncanny_round2`) instead of adopting this branch's expedition/leg model,
and built its own, differently-shaped `do_GET`/`do_POST` error-handling wrapper. `git merge
origin/main --no-edit` produced conflicts in `config.py`, `curation_server.py`, `driver.py`,
`test_config.py`, and this notebook.

Asked the user how to reconcile the fork; the answer was to let the expedition/leg model win
outright and drop the round-based model, since PR #33's entire purpose is retiring it. Resolved
every conflict by keeping this branch's expedition/leg code (`EXPEDITIONS_DIR`, `leg_dir()`,
`LegConfig`/`load_leg_config`, `_require_out_dir()`/`NoActiveLegError`) and deleting
`RoundConfig`/`ROUND_CONFIGS`/`SWEEP_DIR`/`SWEEP2_DIR` and their call sites entirely. Also found
and fixed one silent auto-merge duplication that produced two `_check_manifest_images()`
definitions in `curation_server.py` (git merged them without flagging a conflict since they
didn't textually overlap); kept the expedition/leg-aware one, deleted the `SWEEP_DIR`-based
duplicate it shadowed. `rg -n "SWEEP_DIR|SWEEP2_DIR|ROUND_CONFIGS|RoundConfig" src/ tests/`
confirms nothing remains.

### 2026-07-15 (session 13): UX shard-review plan, Phases 1-5 shipped as stacked PRs

Started closing findings from a 10-shard UX review of the curation server
(`docs/superpowers/plans/design-shards/*.md`), tracked by
`docs/superpowers/plans/2026-07-15-ui-design-shard-findings.md` across 8 phases. Delivered each
phase as its own PR, stacked on the previous phase's branch rather than on `main`, so review stays
scoped per phase: Phase 1 (active expedition/leg visibility, PR #34), Phase 2 (confirm
destructive/paid actions, PR #35), Phase 3 (compare.html judgment-workflow correctness, PR #36),
Phase 4 (visual design-token consolidation, PR #37), Phase 5 (gallery/archive at scale, PR #38).

Phase 5's most significant finding was unplanned: while porting `scan_gallery.py`'s
pagination pattern into `elite_archive.py`'s "view all" modal, a page rendered with `CELLS` and
`openModal()` undefined in the browser, no console error. Root cause was a copy-pasted comment
containing a literal `</script>` substring inside a `<script>` block. The browser's HTML tokenizer
closes a `<script>` element on the *first* `</script`-prefixed text it sees, including inside a
JS comment, since it's not JS-aware at that point in parsing, only scanning for the literal
closing tag. Everything after that point, including the real code and the real closing tag,
gets dropped from the script and parsed as plain markup instead. This bit six pages that all
carried the same comment (archive, coverage, map, redundancy, novelty-decay, preference-rank):
every one of them has had a completely dead grid, no click handlers, no data loading, for as long
as that comment existed, with nothing in server logs or the browser console to flag it. Fixed by
escaping it as `<\/script>` (backslash before the slash defeats the tokenizer's literal match) in
all six files, and added a regression test per module asserting the rendered `<script>` body never
contains a bare `</script` substring.

Also fixed a second, smaller regression this same phase introduced: persisting scan.html's
filter/sort state in the URL (`history.replaceState`) meant `/scan.html?sortKey=...` requests
started hitting the server, but the route matched only the bare path (`self.path ==
"/scan.html"`) and 404'd on anything with a query string. Widened the match to
`self.path.startswith("/scan.html?")` as well, same pattern `archive.html`'s route already used
for its own `?cell=` query param.

Remaining phases (6: error/empty-state legibility, 7: DINOv2-similarity explainability, 8:
remaining IA/nav/hygiene) not yet started; each gets its own worktree stacked on the prior phase's
branch, following the same one-PR-per-phase pattern.

### 2026-07-15 (session 14): Phase 5 review fixes and Phase 6 complete

A GLM review of Phase 5 (PR #38) found two scan-page regressions. The round-aware sort change
had replaced the human-readable generation field with its internal composite sort key, so the
shared lightbox would show values such as `gen 200003`. The sort key now lives in a separate
`sort_gen` field. The URL-restored picked/favorited filters also ran before the favorite records
loaded and then only re-rendered the already-empty result set. The favorites callback now reruns
the filters after it loads. Both fixes have regression tests.

Completed Phase 6 on a branch stacked above the corrected Phase 5 commit. Missing manifests now
tell the researcher to switch legs or launch a round, while missing images still explain stale
manifest paths. Error pages name the failed route. Startup writes an actionable missing-manifest
warning to stderr with the active expedition and leg. All 404s, including static-file fallthroughs,
now render a dark, app-consistent page instead of the standard-library error document. The full
suite passed with 396 tests, and Playwright verified the styled 404 at desktop and mobile widths.

### 2026-07-15 (session 15): Phase 7 makes DINOv2 score views interpretable

Completed the DINOv2-similarity explainability pass on a worktree stacked above Phase 6. The
solution map, coverage map, redundancy clusters, and novelty-decay views now share one short
DINOv2 explanation. The map distinguishes style match to the average real-art embedding from the
closest single training photo, shows each sweep's style-match range and median in the hover panel,
and includes an on-canvas mark legend plus a play-control tooltip. Coverage now explains its
quantile bins and median frontier gate, and its legend marks one image, the median count, and the
maximum count. Redundancy names its slider an image-to-image match threshold, gives its actual
pair range, and identifies the representative as the highest-novelty member. Novelty decay now
defines novelty before asking the researcher to act on a trend.

Added rendering tests for every page, including the no-data novelty state. The focused suite has
23 passing tests. The full suite has 400 passing tests, and ruff plus mypy are clean. A server
smoke attempt reached the expected startup state but could not render the four pages because the
user's currently selected `uncanny_frontier/cockpit` leg has no scored manifest. Selecting another
leg would write the user's active-leg state, so that live check was intentionally not performed.

### 2026-07-15 (session 16): Phase 8 navigation and review-workflow improvements

Grouped the shared tool menu and the tools hub into Generate, Curate, Understand search, and
Preference model. The scan page now uses that shared navigation contract. Added next-step links
between high-traffic pages: compare links to model status and ranking, the status page links back
to comparison and forward to ranking, completed runs link to scan, coverage, and novelty review,
coverage links frontier gaps to the cockpit, and lineage links back to the cockpit.

The comparison task now works with keyboard and screen-reader controls. Each image pane and its
magnifier receives focus, Enter and Space activate it, Escape closes the full-size view and
returns focus, and faithfulness and novelty stay hidden until after a choice. This prevents the
numeric scores from anchoring a judgment before the researcher has looked at the images.

The predicted-preference page now provides a bounded review mode with the top 20, middle 10, and
bottom 10 ranked images. Every cell shows its rank. Researchers can mark an image as matching
their taste or questionable; the server stores those flags separately in
`preference_rank_flags.json`, so they do not become training comparisons. The no-model state now
uses the normal navigation shell. The preference-status controls wrap on narrow screens instead
of clipping.

Playwright smoke testing on an empty, newly selected leg found that the live ranking page raised
an internal-server error before its first manifest existed. `LiveCache` now records a missing
watched file as absent rather than failing, and the ranking cache watches the model and metadata
paths before they exist. The page can therefore render its no-model state immediately and refresh
after the manifest or model is created. Focused tests passed (40 tests); the Playwright desktop
and mobile checks showed the tools hub, shared navigation, no-model page, and root Tools link
without console errors.

### 2026-07-15 (session 14): review fix 1 for active report context

The shared navigation already accepted an expedition and leg, but live server routes rendered
every tool page with only its current-page argument, so the active context never appeared. The
live render paths now pass the current active selection while renderer arguments remain optional
for direct unit tests. Completed-run links in `runs.html` now POST the selected report expedition
and leg to `/api/active-leg` before opening scan, coverage, or novelty decay, preventing a report
for one leg from opening the globally active leg instead. Focused regression and affected-page
tests passed (68 tests), and Ruff passed on the changed files.

### 2026-07-15 (session 15): review fix 1 badge route correction

The active-context badge introduced in session 14 linked to `/?expedition=...&leg=...`, but the
server serves the status page only at the exact `/` path and persists the active selection itself.
The badge now links to `/` while preserving its expedition/leg label. Focused tests passed (69
tests), and Ruff passed on the changed files.

### 2026-07-15 (session 16): completed Sol review fixes for Phase 8 PR #41

Resolved all four medium-severity findings from the independent Sol review of PR #41. Live tool
renderers now receive the active expedition and leg, and completed-run report links persist their
own selection before opening scan, coverage, or novelty review. The compare page blocks a second
choice while the first is pending and blocks keyboard choice while the full-size zoom is open.
The bounded ranking review now renders a persisted flag as a selected, pressed button, updates
that state only after a successful API response, and reports a save failure without changing the
displayed state.

The three review-fix commits are `d60feb2`, `ac03a51`, and `c72604e`. Focused tests passed for
each task. Final CI-equivalent verification passed with 378 tests, Ruff, MyPy, and `git diff
--check` clean. Playwright checked the active-leg badge and the no-model ranking page at desktop
and 390px mobile widths without new console errors. The temporary isolated server was stopped
after the check.

### 2026-07-15 (session 17): PR #41 review findings, Task 0 (Phase 7 merge)

A full-stack review of PR #41 found six verified issues, the highest-severity being structural:
Phase 8 (`phase8-ia-accessibility`, this branch's base) forked from `main` instead of from Phase 7
(`phase7-dino-explainability`), so the two stacks diverged and produced 21 conflicting files.
Built the fix plan in a new worktree, `pr41-review-fixes`, with 7 tasks: Task 0 merges Phase 7 in
first so every later fix lands on the reunified stack, Tasks 1-6 close the remaining findings
(favorite-mutation leg binding, stop-request PID check, undo-flow recovery, missing-vs-empty leg
data, Phase 3 comparison-progress-logic preservation, and two lower-severity fixes).

Task 0 resolved as a merge rather than a rebase: an attempted rebase hit the same conflicts once
per commit (Phase 8 has 5), so aborted it and merged instead for one conflict-resolution pass. 52
raw conflict markers landed across 19 files. Most followed one mechanical pattern: Phase 7 added a
`running=None` parameter to every page's `render_html()` and threaded it through `nav_bar_html()`
for the running-search indicator; Phase 8 didn't have it yet, so Phase 7's side almost always won.
Three merges needed real judgment: `compare_page.py`'s `choose()` had to keep Phase 8's
`choiceSubmitted`/`zoomOpen` accessibility guard alongside Phase 7's `submitting` single-flight
guard as two independent checks (both are asserted on by name in the test suite), move Phase 7's
`submitting` reset out of the success path and into `loadNext()` so the guard stays active during
the 1-second reveal delay, and drop Phase 8's raw `res.count`-based retrain-boundary check in favor
of Phase 7's `/api/preference_status`-derived bucket-crossing check (this alone closes finding #6,
the Phase 3 progress-logic regression). `curation_server.py`'s no-selection status page had a
`manifest_summary` reference that isn't even a parameter of that method, a leftover Phase 8 bug;
dropped it rather than resolving toward either side. `scan.html`'s route needed both Phase 7's
`?sortKey=...`-friendly query-string match and Phase 8's `active_expedition`/`active_leg` args,
which the two branches had touched independently without conflicting semantically.

Merge commit `bbb7afe`. Full suite: 415 passed, 0 failed (the worktree's pre-merge baseline had
377 passed, 1 pre-existing failure in `test_cli.py`; the merge subsumed whatever fixed it). Ruff,
MyPy, and `git diff --check` all clean. Tasks 1-6 hand off to opencode
(`opencode-go/deepseek-v4-flash`, max effort) next via the subagent-driven-development workflow,
per the user's instruction to do the semantic merge inline and delegate only the mechanical tasks.

### 2026-07-15/16 (session 17 continued): PR #41 review findings, Tasks 1-6

Task 5 turned out to need no separate work: every checklist item in its brief (`n_usable`,
`statusStale`, raw-count tracking, bucket-crossing detection, the `choiceSubmitted`/`zoomOpen`
guards, `revealSamplingDetails`) was already present in the Task 0 merge's resolved
`compare_page.py`, and `tests/test_compare_page.py` already covered it. Marked done as a byproduct
rather than dispatching a redundant task.

The first two dispatch attempts for Task 1, using `opencode-go/deepseek-v4-flash` at max effort,
crashed with `no_output_timeout` and produced completely empty logs, a provider-side startup
failure rather than a task problem. Switched every remaining dispatch to `openai/gpt-5.6-luna` at
max effort, which ran cleanly for the rest of the session.

Tasks 1, 2, 3, 4, and 6 each closed one review finding, dispatched one at a time via taskferry with
a TDD mandate, then independently re-verified locally (full suite, Ruff, MyPy, `git diff --check`)
rather than trusting the dispatched agent's self-report:

- Task 1 (`ae96040`): bound favorite add/remove to the expedition/leg that was active when the
  lightbox opened (`LB_EXPEDITION`/`LB_LEG`), rejecting the mutation with a stale-context error if
  the server's active selection changed underneath it. 416 passed.
- Task 2 (`888449c`): the run-stop endpoint now requires the caller to send back the PID and start
  time it was given when the run started, and rejects a stop request whose identity doesn't match
  the currently running process, so a stale "stop" click from an old page load can't kill a
  different, newer run. 419 passed.
- Task 3 (`62c2fa9`): fixed three bugs in the lightbox undo flow: `undoBtn` became a module-scoped
  variable so a second favorite-removal clears the previous undo button instead of leaving a stale
  one clickable, the undo handler now dispatches `lightbox:favorite` so the scan gallery's
  favorited filter updates, and both the toggle and undo fetch handlers now check the JSON
  response body for an `error` field instead of only checking `r.ok` (a 200 response carrying
  `{"error": ...}` was previously treated as success). 420 passed.
- Task 4 (`49dfc22`): `_send_status_page` now distinguishes a leg that was never launched
  (`n_entries == 0`, unchanged "launch a round" advice) from a damaged one (`n_entries > 0` but
  zero files present on disk), which gets a new data-integrity error page naming the missing count
  and pointing at the backup/state directory instead of suggesting a launch that could overwrite
  recoverable data. `POST /api/active-leg` also now warns to stderr if the newly selected leg is
  damaged. 422 passed.
- Task 6 (`2d9379a`, `e5d8b44`): unknown `/api/*` routes now 404 with a JSON body instead of the
  browser HTML error page, and `end_headers` strips the query string before checking the file
  extension (so `/scan.html?filter=...` keeps its `no-cache` header) and treats `.json` the same
  as `.html`, so `/scan_data.json` gets a `no-cache` directive it never had. 423 passed.

Every task's diff matched its brief closely enough that no rework was needed; independent local
verification confirmed each self-report rather than just trusting it. `git merge-tree --write-tree
phase7-dino-explainability pr41-review-fixes` produces a single tree SHA with no conflict markers,
confirming the whole stack still merges cleanly against Phase 7.

Mid-session the user asked for each task to become its own stacked PR, matching the existing
Phase 1-8 pattern (PRs #34-41), instead of one combined branch. Since all six tasks had already
landed as a linear commit history on `pr41-review-fixes`, this needed no rework: cut one
lightweight branch pointer per task at its already-existing commit SHA, each based on the previous
task's branch: `pr41-task0-merge-phase7` (`c526646`) on `phase8-ia-accessibility`, then
`pr41-task1-favorites-leg-scope` (`ae96040`), `pr41-task2-stop-run-identity` (`888449c`),
`pr41-task3-undo-recovery` (`62c2fa9`), `pr41-task4-integrity-error` (`49dfc22`), and
`pr41-task6-lowseverity-fixes` (`e5d8b44`), each stacked on the last.

Still open: a live Playwright pass covering the JS changes from Tasks 1-4 (no browser access in
the dispatch sandbox), and the user's decision on whether to push and open the six PRs now or
defer. No branches have been pushed yet.

### 2026-07-15 (session 13): live-check gotcha, active-leg state mutated by cockpit.html GETs

Executing the UI design-shard findings plan (`docs/superpowers/plans/2026-07-15-ui-design-shard-findings.md`)
via subagent-driven-development, with implementers dispatched through opencode per the global
CLAUDE.md override. Task 1.1 (active leg in nav bar) and Task 1.2 (running-search nav indicator)
both required a live Playwright/browser check per the plan's global constraints. Both my own
Task 1.1 check and the Task 1.2 implementer's own live check hit `/cockpit.html` on a restarted
`curation_server.py`, and `_set_active_selection` treats any GET to a cockpit-family route as an
implicit leg switch to `"cockpit"` (`curation_server.py:1262-1264`, pre-existing logic from PR #33,
not introduced by either task). Since `$XDG_STATE_HOME/clawmarks/active_leg.json` is real,
shared, cross-worktree production state, not per-worktree, this silently overwrote the user's
actual last selection (`trent_v3_epoch4/freeform1`) with `uncanny_frontier/cockpit` twice over
during routine verification. Caught it by noticing the server startup banner reported a leg the
user hadn't set, diffed against the value read at session start, and restored
`trent_v3_epoch4/freeform1` by hand. No image data or embeddings were at risk (this file is only
a pointer, not generation output), but it is exactly the class of unattended-agent side effect
the project's data-integrity rule exists to catch.

Gotcha for future live checks: any GET to `/cockpit.html` (or any other route that force-switches
the active leg) during verification mutates real, shared selection state, not a worktree-local
copy. Read `active_leg.json` before a live check and restore it afterward if it changed and the
change wasn't intentional; this pattern was already used once before in session 11 ("restored to
'none selected' afterward") but wasn't written up as a general rule at the time.

### 2026-07-15: `cockpit.html` 500'd on `trent_v3_epoch4`; reconstructed its missing `expedition.json`

While live-verifying the UI design-shard findings plan in the `ui-design-shard-findings`
worktree, `/cockpit.html` returned a 500: `ValueError: unknown expedition 'trent_v3_epoch4'`.
Traced this to `expedition.json` simply never having been created for `trent_v3_epoch4` — the
expedition's `freeform1` leg has 50 real generated images and a full `scored_manifest.json` under
`$XDG_STATE_HOME/clawmarks/`, but no config was ever committed to the repo's `expeditions/`
directory, so `_set_active_selection` correctly refused to recognize it.

**Gotcha:** `config.EXPEDITIONS_DIR` (`ROOT / "expeditions"`, git-tracked, holds each expedition's
small `expedition.json`/`legs/*.json` config) and `config.leg_dir()` (`STATE_DIR / "expeditions" /
...`, holds the actual generated images/manifests/embeddings) are two **different** directory
trees that happen to share the `expeditions/<name>/` shape. This is deliberate (small, reviewable
config lives in git; heavy generation output lives outside the repo per the XDG rule), confirmed
by `test_config.py::test_expeditions_dir_is_repo_relative`, not a bug, but it means an expedition
can have real `STATE_DIR` image data and still read as "unknown" if its config was never created
or committed.

Reconstructed `expeditions/trent_v3_epoch4/expedition.json` (plus empty `legs/freeform1.json` and
`legs/cockpit.json`) with `trigger_word: "trentbuckle style, "` and `negative_prompt: "low
quality, blurry, watermark"`, both recovered empirically (exact match across all 50
`freeform1/scored_manifest.json` entries) rather than guessed. `textures`, `fallback_subjects`,
and the budget/generation-count fields have no recoverable source, so they were borrowed from the
`uncanny_frontier` reference expedition's already-committed config (same `trentbuckle` style
family) as an informed placeholder. **Still uncommitted as of 2026-07-16** (untracked in the
primary checkout's `expeditions/trent_v3_epoch4/`), pending the user's review of those placeholder
fields before this goes into git.

Verified live: `/api/expeditions` lists `trent_v3_epoch4` with both legs, `/cockpit.html` returns
200 (previously 500), `/compare.html` still 200. The cockpit leg's own 500s (`FileNotFoundError`
on `trent_v3_epoch4/cockpit/scored_manifest.json`) are expected, not a regression: that leg has
genuinely never been run. Backed up both `trent_v3_epoch4` and `uncanny_frontier`'s
`$XDG_STATE_HOME` directories (`cp -a` + `diff -rq` + file-count verification) before any of this,
per the project's data-integrity rule; no image files were touched.

### 2026-07-15: Phase 6 (error/empty-state legibility) implemented twice, independently

The `ui-design-shard-findings` worktree implemented its own fix for the same error-legibility
shard that PR #41's review-fix Task 6 also closed: a `FileNotFoundError` hint that distinguishes
"no scored manifest yet" from "manifest points at a stale path," a styled dark-theme 404 page, and
a startup warning that names the affected expedition/leg. Both fixes shipped independently on
different branches before either was aware of the other.

Opened as PR #48 to check whether it was still worth merging once discovered. It wasn't: `main`
(via the already-merged Task 6) already carries the same missing-manifest hint logic and a styled
404 page, plus `/api/*` 404s returned as JSON, which this worktree's version never added. Merging
PR #48 would have been pure duplication and would have regressed the JSON-404 behavior. Closed as
superseded rather than reconciled; the one difference worth noting is Task 6's `missing_path =
str(exc).split("'")[1] if "'" in str(exc) else ""` versus this worktree's `exc.filename`, the
latter being the more direct way to read a `FileNotFoundError`'s path, but not worth a follow-up
patch on its own.

