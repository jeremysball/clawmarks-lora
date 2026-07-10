# CLAWMARKS LoRA: Lab Notebook

Running notes for a whitepaper on training a style-transfer SDXL LoRA and finding the
hyperparameter configuration that best reproduces the source art's style. Written for a
non-academic reader: every technical term gets defined in plain language the first time it
appears, since this notebook doubles as the first draft of the paper's methods section, and as
the single project ledger (infra, datasets, checkpoints, gotchas).

Author's assistant: Claude. This notebook is the shared record between us. Update it after
every meaningful step, not just at the end.

---

## 1. Background and motivation

The subject is a small, personal art style called CLAWMARKS: sketchbook-style animal portraits
(mostly cats, plus wolves, foxes, horses, owls) in marker, ink, colored pencil, and mixed media.
31 real images make up the full training set. The goal is an SDXL LoRA (a small add-on model
that teaches a big pretrained image model, here an SDXL checkpoint called Illustrious, a new
style without retraining the whole thing) that reproduces this style under the trigger word
`trentbuckle`.

**A data bug forced a full restart before anything else could work.** The first training attempt
used a dataset folder that looked correct by its name but wasn't. A caption/image consistency
check (does the text description attached to each training image actually describe that image)
found that 9 of 31 captions described the wrong image. This is a data-quality bug, not a model
bug, and it left no trace in the training loss curve. Loss looked normal throughout. Only
generating sample images at different training checkpoints and inspecting them by eye revealed
the problem. The lesson carries through everything below: an automated score is a filter, not a
verdict. Confirm the top candidates with human eyes before trusting a number.

**Epoch 4 became the current-best checkpoint this way:** after retraining on the corrected
dataset, a validation grid (the same 10 prompts × 3 random seeds × several candidate epochs) was
generated and reviewed against a rubric: style consistency, no broken or garbled compositions,
faithfulness to the training subjects. Epoch 4 won. It is the baseline this whitepaper's search
starts from.

---

## 2. An objective style-similarity metric

Epoch 4 generated 250 new images through a RunPod serverless endpoint. That raised a question:
which of those 250 images actually looks closest to the real training art, not just "which looks
best at a glance"? Two candidate tools:

- **CLIP** (2021) learns to match images with text descriptions. Its embeddings (a list of
  numbers representing an image's content in a form that supports mathematical comparison) skew
  toward semantic content: what object is in the picture. That's what its training objective
  rewards.
- **DINOv2** (Meta, self-supervised, trained on images alone with no text) captures visual style:
  texture, brushwork, composition, directly, since no text signal pulls it toward "what is this a
  picture of."

The real question is style match, not subject match, so DINOv2 is the better tool. It became the
primary metric.

**Method:** embed all 31 real training images with DINOv2, average them into one vector (the
centroid: the geometric center of the real art's style in embedding space), then score any
generated image by its cosine similarity (a number from -1 to 1 measuring how aligned two vectors
are; 1 means identical direction) to that centroid. Higher means closer to the real style.

**A surprising early finding:** scoring the existing 250 generated images, and a "curated 25"
picked earlier by eye, showed CLIP and DINOv2 disagree sharply. They shared only 6 of their top
30 picks. Several hand-picked "best" images, photorealistic graphite cat portraits, watercolor
horse paintings, scored near the bottom of all 250 by DINOv2 (ranks 200-246 of 250). The real
CLAWMARKS set is dominated by flat graphic marker and ink work, not painterly realism, so those
appealing realistic renders are stylistic outliers. DINOv2 confirmed this numerically instead of
leaving it as a hunch. The curated-25 selection was redone using the DINOv2 ranking.

**A second finding, still open (see Section 4):** scoring the 31 real training images against
their own centroid gives a mean self-similarity of 0.61, but a minimum of 0.22. At least one real
training image is a serious style outlier from the rest of the set. If the LoRA has to reproduce
a genuine outlier alongside 30 consistent images, that could dilute how sharply it learns the
dominant style. Whether this is a captioning error (like the original bug) or a legitimate
stylistic one-off worth down-weighting is round 1's first job.

---

## 3. Experiment design: the hyperparameter search

**Goal:** find the LoRA training configuration that produces generations closest to the real art
by DINOv2 similarity, past what the single epoch-4 checkpoint already achieves, in a way that
actually generalizes rather than just re-matching the prompts already used to pick epoch 4.

**Why not a full grid?** A full grid across every hyperparameter worth testing means dozens of
full retrains, each about 35 minutes of GPU time. That's affordable in dollars (rented cloud GPUs
run well under $1/hour), but most of that grid would test directions that don't help. A
sequential search spends the compute where the evidence points instead.

**External review.** Before running this for real, the design below went to two outside models
acting as ML-expert reviewers (GPT-5.5 and GLM-5.2, prompted independently, see
`notes/reviews/`). Both converged on the same core problems with the original plan, and
their fixes are folded into the method below rather than kept as a separate critique. Where a
step exists because of that review, it says so.

**Method: probe-then-commit sequential search, revised.**

0. **Resolve the data-side outlier first, before round 1's hyperparameter probes.** One real
   training image scores 0.22 against the centroid of the other 30 (mean 0.61). Both external
   reviewers independently flagged this as the first thing to resolve, since it distorts the
   centroid every later score gets compared against. Determine whether it's a caption error or a
   genuine stylistic one-off, and decide whether to down-weight or fix it, before any
   hyperparameter probe runs. This is its own step, not folded into a later round, so its effect
   is never tangled up with a hyperparameter change (see step 6).

1. **Calibration check, once, before trusting probes at all.** Take 2-3 candidate directions and
   run each at both probe length (~156 steps) and full length (780 steps), then check whether
   the two lengths rank the directions the same way. The learning-rate schedule runs 3 cosine
   cycles over the full 780 steps, so a 156-step probe finishes under a single cycle, a different
   part of the training dynamics, not just a shorter version of the same run. If probe-length and
   full-length rankings agree, probes are trustworthy for screening. If they don't, probes can
   only be trusted to catch catastrophically bad directions, not to pick a winner.

2. **Probe phase.** Probe length is **260 steps, one full cosine cycle** (780 steps / 3
   cycles), not the calibration check's 156 steps, revised 2026-07-09. 156 steps cuts a probe
   off 60% of the way through the first cycle, before the learning rate has decayed back down
   near zero, so the probe never sees the part of training the cosine schedule is named for.
   260 steps lets the LR complete one full ramp-down, a much closer match to what happens by
   the end of any single cycle in the real 780-step run, at roughly 12-16 minutes per probe
   instead of 6-10. Each direction gets **8 replicates** (different seeds, not the earlier
   3-4 guess, see the derivation below), since fewer can't reliably separate a real effect
   from noise at the effect size actually worth acting on. Control probes (current-best
   config, seeds only) get pooled across rounds rather than re-measured each time, since the
   pooled estimate only improves as more accumulate.

   **Note:** the noise floor and n=8 derivation below were measured at the old 156-step probe
   length (`control_156`/`controlB_156`.../`controlH_156`, 8 replicates). That data stays as a
   permanent record of the calibration finding, but it does not carry over to 260-step probes:
   noise floor is a property of probe length, since a longer probe gives the model more
   steps to converge and likely changes how much seed-to-seed variance remains. Round 1's real
   noise floor needs a fresh batch of 260-step control replicates before probing starts for
   real.

   **Note added 2026-07-09: paired training seeds, not just pooled ones.** `train_probe.py`
   never pinned a training seed (kohya's `set_seed` is skipped whenever `--seed` is omitted), so
   every probe run so far, including the calibration checkpoints and all 8 `control_*_156`
   noise-floor replicates, got an uncontrolled, different LoRA weight init and batch-shuffle
   order every time (confirmed by reading kohya's source: `set_seed` covers `random`/`numpy`/
   `torch.manual_seed`/`cuda.manual_seed_all`, and the DataLoader's `shuffle=True` draws from
   that same global torch RNG with no separate generator, so a single seed value pins both init
   and shuffle order together). Generation was already held fixed (`gen_samples.py` always uses
   `--seed 42`), so all of the measured noise floor is training-seed variance, nothing else.

   From round 1 onward, `train_probe.py` accepts `--seed`, and every direction (including
   control) reuses the same fixed list of 8 seeds (`CANONICAL_SEEDS` in that script) so replicate
   *i* of any direction shares its training seed with replicate *i* of control. This turns the
   comparison into a genuinely paired design: the delta at each seed index cancels out the
   shared init/shuffle-order variance, rather than just averaging over it, which should tighten
   the paired-delta variance and could lower the n needed for a given effect size once measured.
   Pinning a *single* global seed for every run instead was considered and rejected: it would
   collapse replication to n=1 and risk mistaking one seed's lucky interaction with a
   hyperparameter for a real effect. The `control260A`-`H` batch (launched before this decision)
   used unpinned seeds, so it stays useful as a raw/unpaired 260-step noise-floor reference, but
   round 1's real probe phase uses the paired-seed design above.

   **Done, 2026-07-09: scored `control260A`-`H`.** Checkpoint-mean stdev across the 8 unpaired
   260-step replicates is **0.0279**, not larger than the 0.035 measured at 156 steps. This
   confirms the one open risk in carrying the n=8/0.05-cosine derivation forward unchanged
   (that noise might grow with probe length) didn't happen; if anything the 260-step floor
   looks slightly tighter. Caveat raised by an external reviewer (Fable, see the review-summary
   log entry below): at n=8 a standard deviation estimate carries roughly 25% relative
   uncertainty on its own, so 0.0279 and 0.035 are not statistically distinguishable from each
   other. Read this as "not obviously worse," not "noise got smaller."

3. **Selection rule: a real statistical test, not "beats the floor."** Score every probe on the
   same fixed prompt/seed slots so each direction's replicates and the pooled controls can be
   compared as paired deltas. Run a permutation test (or bootstrap a confidence interval over
   the paired deltas) rather than just checking "average beats the noise floor." Require both a
   real effect (most of the bootstrap mass positive) and a practically meaningful size (roughly
   >0.02 DINOv2 cosine, not just statistically nonzero). With ~10 probes tested per round, apply
   a multiple-comparisons correction (Benjamini-Hochberg) or the equivalent discipline of
   ranking every direction and only ever advancing the single best one, not the first to clear a
   bar. Taking the first direction that clears a loose bar, with this many tested per round,
   guarantees some spurious wins by chance alone.

   **Note on step 3 in practice: the noise floor has to be measured before it can be used.**
   "Beats the noise floor" only means something once the floor itself is a real number, not an
   assumption. The floor comes from the pooled control-only probes: score each one, take
   pairwise deltas between them (same fixed prompt/seed slots as everything else), and the
   spread of those deltas, which should average to zero since they're all the same config, is
   the noise floor's empirical standard deviation. Every direction's delta against control has
   to clear this floor, both statistically (permutation test or bootstrap CI, above) and in
   practical size (>0.02 cosine).

   This same number also decides how many replicates (n) round 1 actually needs, which the
   "3-4 replicates" starting assumption above was a guess at, not a derived figure. Once the
   noise floor's spread is measured from real control probes, simulate: generate synthetic
   paired deltas with that measured spread plus an injected effect, run the same permutation
   test at a few candidate n values (3, 4, 6, 8), and see which n detects the injected effect
   at least 80% of the time. That n, not the guess, is what round 1 should use.

   **Done, 2026-07-09** (see the lab log entry below for the full numbers): the noise floor
   measured from 3 control_156 replicates turned out bigger than the original 0.02-cosine
   effect floor could ever clear: at n=8 replicates, a true 0.02 effect is only detected 24%
   of the time, no matter how many more replicates get added within a practical budget. The
   effect-size floor that step 3 actually enforces is revised to **0.05 cosine, not 0.02**,
   and round 1's replicate count is set to **n=8** (84% power at 0.05, 98%+ at 0.08, the
   scale of gaps actually seen in the calibration table, e.g. dim64's ~0.06 gap from the other
   three directions, constlr's ~0.10 probe-to-full swing).

   **Note added 2026-07-09: paired seeds make a separately-measured noise floor optional, not
   required.** A sign-flip permutation test builds its null distribution from the very deltas
   under test. Given a direction's 8 paired deltas (`direction_score[seed_i] -
   control_score[seed_i]`, same seed on both sides via `CANONICAL_SEEDS`), the test asks how
   often random sign-flips of those same 8 numbers would produce a mean this large. It never
   needs an externally-measured floor constant to run. That constant was only ever a stand-in for
   two side calculations: the practical effect-size floor (still worth a working number, 0.05
   cosine, but it can come from round 1's own paired deltas rather than a dedicated batch) and
   the replicate count n (n=8 was derived from unpaired noise; pairing can only shrink variance,
   never grow it, so n=8 is if anything safer than that derivation implied). Practical
   consequence: the `control260A`-`H` unpaired batch is a useful reference, not a gate. Round 1's
   real probe phase does not need to wait on a fresh unpaired-floor measurement first.

4. **Commit phase.** The single best-ranked direction from step 3 gets one full 780-step
   retrain.

5. **Score the full run at multiple checkpoints, not one.** Score epochs 2, 4, 6, 8, and final,
   not just the endpoint, and look at the trajectory shape. A direction that only wins because it
   learns fastest early (and might overfit later) should not automatically win over one that
   catches up by epoch 10. Score against **two validation sets**: the original 10-prompt set (for
   continuity with epoch 4's history) and a **holdout set** of prompts never used to pick epoch 4,
   weighted toward the subjects the original validation grid found weak (owl, tiger; later,
   human face / cyborg / liminal once those are established prompts). The holdout set exists
   because the original set already favors whatever epoch 4 happens to be good at, so it's
   structurally biased toward rewarding "epoch-4 lookalikes" over configs that generalize better.
   This is a holdout of *generation prompts*, not of real training images or synthetic training
   images, a different axis from any dataset-augmentation work.
   Alongside the centroid score, also compute nearest-neighbor similarity (max similarity to any
   single real image, not the average) as a companion number: our own strength-sweep probe
   (Section 5, gotcha log) showed switching from centroid to nearest-neighbor doesn't flip which
   generations look good or bad, so it's a cheap secondary check rather than a replacement.
   A win becomes the new current-best config for the next round.

6. **Data-side check, kept to its own round.** Reconsider the training data itself between
   rounds, e.g. a caption fix or a repeat-count change, but never in the same round as a
   hyperparameter change, so any improvement stays attributable to one cause or the other, not
   both at once.

7. Repeat for 5 rounds. Keep a running ledger of directions that were probed but not committed,
   in case a later round's data changes make a previously-rejected direction worth revisiting
   (guards against the search settling into a local optimum simply by never looking back).

**Hyperparameters in play**, starting from the epoch-4 config (network dim 32 / alpha 16, unet
learning rate 1e-4, text-encoder learning rate 5e-5, min_snr_gamma 5, cosine learning-rate
schedule, 10 epochs):
- Network dim/alpha: how much capacity the LoRA has to learn new detail
- Learning rates: how large a step the model takes per batch
- min_snr_gamma: a loss-weighting trick that can stabilize training on noisy or varied data
- Learning-rate schedule shape: cosine vs. constant (the calibration check in step 1 matters
  especially here, since a 156-step probe can't complete even one full cosine cycle)
- Epoch/repeat count

**Metric upgrade, considered but not adopted for round 1.** Both reviewers, independently,
suggested a distributional metric (Frechet-style distance, the same idea behind FID, or MMD)
in place of centroid cosine similarity, since centroid similarity rewards a checkpoint that
produces safe, repetitive, mean-hugging images as much as one that reproduces the style's real
range. That's a real critique, but a Frechet-style distance needs a reliable covariance estimate
in DINOv2's 768-dimensional embedding space, and 31 real images can't support that (badly
underdetermined). Revisit this once more real or validated-synthetic images exist, or after
reducing embedding dimensionality (e.g. PCA to ~15-20 components); not before round 1.

**After all 5 rounds:** rank every full-run checkpoint, and their own epoch sub-checkpoints, by
DINOv2 score. Then, per Section 1's lesson (and per both external reviews), have a **human
panel review the top few**, not DINOv2 alone. The metric has already been shown to disagree with
human preference (Section 2), so the final call belongs to human eyes.

**Budget, revised 2026-07-09 after the noise-floor derivation:** with n=8 replicates (up from
the earlier 3-4 guess) across roughly 10 directions per round, that's ~80 probes per round at
6-10 minutes each, 8-13 hours of probing alone, plus one 34-minute commit run scored at 5
checkpoints. Call it **9-14 hours per round**, not 2.2-2.5. Five rounds plus the one-time
calibration check: **45-70 hours of GPU time**, a large jump from the original 11-13 hour
estimate, and worth running two pods in parallel (as calibration already did) rather than
serially. This is the direct, unavoidable cost of the effect-size floor moving from 0.02 to
0.05 cosine: a smaller detectable effect needs proportionally more replicates to see reliably,
and 0.02 was never achievable at any practical n given the measured noise (see step 3's note
above and the 2026-07-09 log entry).

**Candidate direction slate for round 1, proposed 2026-07-09, not yet approved.** Baseline
config: `network_dim=32, network_alpha=16, unet_lr=1e-4, text_encoder_lr=5e-5, min_snr_gamma=5,
clip_skip=2, cosine schedule x3 cycles`. Eight directions proposed, triaged by "genuinely
uncertain, worth an 8-replicate probe" rather than re-testing settled defaults:

| # | Direction | Change | Why it's uncertain |
|---|---|---|---|
| 1 | `alpha32` | `network_alpha=32` (alpha=dim) | Only `dim` has been probed (dim64, worst at both lengths); `alpha` alone, which controls how much the LoRA's learned change is scaled down before being added to the base model, is untested |
| 2 | `dim16` | `network_dim=16, network_alpha=8` | dim64 was worse than dim32; checks whether smaller keeps helping or reverses |
| 3 | `snr1` | `min_snr_gamma=1` | Stronger loss-reweighting; brackets whether gamma=5 is actually near-optimal |
| 4 | `snr20` | `min_snr_gamma=20` | Much weaker reweighting (near-off); brackets from the other side |
| 5 | `clipskip1` | `clip_skip=1` | clip_skip=2 is an SD1.5-era convention; unclear it means the same thing for SDXL's two text encoders |
| 6 | `telr_match` | `text_encoder_lr=1e-4` (matches unet_lr) | Currently pinned at half of unet_lr, untested untied |
| 7 | `telr_freeze` | `text_encoder_lr=0` | Tests whether the text encoder needs training at all for this style |
| 8 | `cycles1` | `lr_scheduler_num_cycles=1` | Single decay to zero instead of 3 restarts over 780 steps |

**Caveat on `cycles1`, flagged by an external reviewer (Fable) and not yet resolved:** the whole
justification for a 260-step probe length is "one full cosine cycle" (780 steps / 3). A direction
that changes `lr_scheduler_num_cycles` to 1 has a 780-step cycle, so a 260-step probe of *that
specific direction* would cut it off mid-decay, exactly the failure mode step 1's calibration
check was built to catch for `constlr`. Before probing `cycles1` at 260 steps, either give it its
own full-780-step probe length, or drop it from probe-based screening and test it only at commit
phase.

**Not included:** `lr_scheduler_num_cycles` values above 3 (adds cost with no clear rationale for
a 780-step run), and no further `min_snr_gamma`/`clip_skip` variants beyond the two brackets each
above, since both hyperparameters already sit at common community defaults.

---

## 3b. Exploratory side branch: mapping the style's frontier (idea only, not started)

Everything above optimizes one thing: make generations score as close as possible to the real
style's centroid, i.e. pure imitation. A different, complementary question came up in review
(external reviewer Fable, 2026-07-09): within the space of prompts and generation settings for
the same fine-tuned style, where are the outputs that are recognizably *in the style* but
surprising or unsettling rather than safe reproductions, and can that region be searched for on
purpose instead of stumbled into by accident?

**The idea, in plain terms:** the same DINOv2 centroid-similarity scorer already built for round
1 can be read two ways at once. Score every candidate image on two axes: how close it is to the
real-art centroid (stay in a band that still reads as the style, not off-style noise), and how
far it is from the *single nearest* real training image (maximize this, as a novelty measure). A
"liminal band," e.g. centroid similarity around 0.55-0.70 if faithful outputs score ~0.80 and
off-style garbage scores ~0.40 (illustrative numbers only, not measured yet), is the region worth
mapping: still CLAWMARKS, but far from any one specific training image.

**Search method suggested: MAP-Elites**, a technique from the *quality-diversity* branch of
evolutionary computation (a subfield of computer science under optimization/AI that searches by
maintaining and varying a population of candidates, rather than following a single gradient).
Ordinary optimization asks "what is the single best output." Quality-diversity search asks "what
is the best output *for every distinct kind of output*," and returns a whole map rather than one
winner. Concretely: divide the two-axis space above into a grid of bins, generate broadly (varied
prompts and settings), and keep only the single best-scoring image per bin. The result is a
contact sheet that doubles as an atlas of the style's frontier, faithful reproductions in one
corner, increasingly strange-but-recognizable outputs fanning out toward the others, rather than
a single "winner" that tells you nothing about the shape of the space around it.

**Concrete knobs floated to drive generation for this search** (not yet tried), each tied to a
mechanical reason it should push toward the band rather than the centroid:
- **LoRA strength overdrive** (weight 1.4-1.8x instead of the normal 1.0, or negative) to
  extrapolate past the learned style rather than reproduce it exactly; per-block weighting
  (applying the LoRA only to some of the model's internal blocks) to separate "texture" from
  "composition" effects.
- **CFG extremes**: very low classifier-free-guidance scale (1-2.5) lets the generation drift
  toward the base model's own distribution, now flavored by the LoRA, rather than tightly
  following the prompt; very high (18+) over-sharpens and can break composition.
- **Conflicted conditioning**: prompt for subjects the LoRA never saw during training (the real
  set is animal portraits only), or interpolate between a style-typical prompt and an unrelated
  one and render the midpoints.
- **Truncated generation trajectories**: very few sampling steps, low-denoise image-to-image, or
  generating from the half-trained 156/260-step probe checkpoints already sitting on disk from
  round 1's calibration and noise-floor work, which learned the paper/mark-making texture before
  they learned full compositions.

**Standing hygiene rule if this ever gets picked up:** exploration outputs must never be folded
into the real-image reference set the centroid is built from, and this stays its own separate
thread from the 5-round hyperparameter sweep, not mixed into any round's probing.

**Status, 2026-07-09: run and concluded.** The idea above went from proposal to a full overnight
search in one session. What actually happened, and what it found, follows. This section keeps the
original proposal above as-written for the record; results live here rather than rewriting the
proposal after the fact.

**What ran.** `notes/run_uncanny_sweep.py` first generated a fixed 452-image grid (5 style prompts
x 8 conflicted prompts, 4 LoRA strengths x 4 CFG values, 2 seeds, plus a negative-trigger arm and a
truncated-trajectory arm) against the existing serverless endpoint. This became "generation 0" of
an adaptive follow-on, `notes/run_uncanny_allnight.py`: a checkpointed loop that scores every image
with DINOv2 (centroid similarity = faithfulness, 1 minus nearest-real-image similarity = novelty),
rebuilds a descriptor-grid gallery after every batch, and generates the next batch as a mix of
"exploit" (small strength/CFG mutations near the current best liminal-band images) and "explore"
(fresh subject/texture prompt recombinations). This is descriptor binning with every image kept per
bin, not true MAP-Elites (no automated coherence scorer exists to pick one elite per bin, so a human
still has to curate the gallery by eye), consistent with the honest framing already used for the
original idea.

The loop also implements a two-stage self-improvement ladder for when novelty plateaus (no
improvement greater than 0.01 over 3 generations): stage 1 widens the built-in subject/texture
vocabulary and strength/CFG ranges; stage 2, if that also plateaus, hands subject-idea generation to
GPT-5.5 via a one-shot non-interactive `opencode run` call, so fresh creative variety doesn't depend
on the script's own fixed lists. Budget guard: stop once cumulative spend crosses $8.50 (a $1.50
safety margin under the $10 cap), checked against the real RunPod account balance before every
generation, plus a 7.5-hour wall-clock cap as a backstop.

**Infrastructure hiccup before the run started:** the RunPod serverless endpoint was found wedged
before launch, test jobs sitting in `IN_QUEUE` forever despite idle workers available. Root cause:
the account balance had gone negative (-$0.099) from the earlier work that session, and RunPod's
dispatcher silently refuses to hand work to a negative-balance account rather than erroring, which
made a billing problem look like a hung endpoint. Fixed by the user adding funds; confirmed healthy
with a real test job before committing to an unattended overnight run.

**Results.** The loop ran 49 generations over 2.28 hours, 3392 total images, and stopped exactly on
its own budget guard at $8.58 spend (comfortably under the $10 cap, none of it wasted on the earlier
wedged-endpoint jobs since those never completed and weren't billed). Liminal-band novelty (the
score this search exists to push up) moved 0.8143 (generation 0 baseline) -> 0.8264 -> 0.8273 ->
0.8352 -> 0.8356 -> 0.8396, then held flat at 0.8396 from generation 26 through generation 49, the
last 23 generations straight, despite both rungs of the self-improvement ladder firing on schedule:
a vocabulary/range widening after the first plateau (generation 6), then a GPT-5.5 handoff after the
second (generation 7), which worked cleanly, returning 15 genuinely fresh uncanny-scene ideas (e.g.
"public restroom mirror showing one extra sink reflection," "subway platform clock stopped, train
lights approaching slowly"). Those ideas did move the needle, novelty ticked up through three real
steps (0.827 -> 0.836 -> 0.840) right after they entered the pool, but the gain topped out there and
no further escalation exists past stage 2, so the remaining ~1900 images generated after generation
26 explored the same plateaued region without finding anything past it.

**Honest read for the whitepaper:** the novelty gain from creative reinforcement was real, not
noise, but small and short-lived. Two candidate explanations, neither confirmed: (1) this specific
style, at these settings, may simply have a fairly low novelty ceiling under the current LoRA, the
liminal band it can support without breaking faithfulness is narrower than the original proposal's
illustrative numbers assumed; or (2) the exploit/explore split (half of every batch mutates near
existing high scorers) may have been too exploit-heavy once the pool filled with similar winners,
starving exploration of the room it needed to find a genuinely different region rather than a
refinement of the same one. Distinguishing these needs a follow-up run with a more explore-heavy
mix and no comparison to this run's already-explored region, not yet done, worth doing before
citing a novelty ceiling as a real finding rather than an artifact of this run's search bias.

**Deliverables:** `notes/uncanny_sweep/gallery.html` (3392 images across the faithfulness x novelty
descriptor grid, served locally over the tailnet during the run), plus `notes/uncanny_sweep/
scored_manifest.json` (full per-image metadata and scores) and `notes/uncanny_sweep/
allnight_state.json` (full generation-by-generation novelty history and the GPT-5.5 subject list).
Final curation of the liminal-band highlights is still a human task, per this section's standing
rule that these scores filter, they don't verdict.

---

## 4. Open questions for round 1

- Which real training image scores 0.22 against the centroid, and is that a caption bug or a
  genuine stylistic outlier? (Now step 0 of Section 3, resolved before probing starts.)
- Does the probe-length calibration check (step 1) confirm 156-step probes rank directions the
  same way as full 780-step runs? If not, probes can only screen out bad directions, not pick
  winners, and the whole budget estimate above needs revisiting.
- Does a hyperparameter direction that wins before a data change still win after one? If the data
  changes substantially, the pooled noise floor likely needs a fresh control-probe batch too.

---

## 4a. Whitepaper framing notes (deliberately unresolved)

Decided so far:

- The paper is a **technically rigorous accounting of what's been learned**, not a tutorial and
  not a claim of novel research. It won't oversell a 31-image, 5-round sweep as generalizable
  science.
- Center of gravity: a blend of **the search methodology** (probe-then-commit, the noise floor,
  double-replicated probes) and **the metric-disagreement finding** (CLIP vs. DINOv2, "pretty"
  images scoring near the bottom of the real style match). Neither is a mere setup for the
  other.
- Real mistakes belong in the paper as content, not as sanitized background: the dataset-caption
  bug, the duplicate-job incident, the torchvision workaround. These are part of what was
  actually learned.

Still open, on purpose, until more of the sweep exists to write about:

- Exact section structure and where the paper's scope ends (does it cover the serverless
  deployment/curation pipeline as a main section, or stay narrowly on the metric + search?).
- Whether the outlier training image (Section 4) turns into its own discussed finding or a
  footnote.
- Final framing of the limitations section: how bluntly to state the small-sample caveat, and
  whether to include what a larger-scale version of this study would need.

Revisit this section once round 1 (or a few rounds) of the sweep produces real numbers. Deciding
the paper's final shape now, before the data exists, would be guessing.

---

## 5. Project reference

Everything below is quick-reference state carried over from the project ledger: what's valid,
what's stale, and how to reach the infrastructure. Consult this before reusing any file in this
directory.

### Datasets: which one is correct

| File/dir | Images | Status |
|---|---|---|
| `art/`, `art.zip`, `full-dataset.zip` | 31 | Original, pre-fix. 9 captions describe the wrong image (see `caption_check_result.log`). Don't train on these. |
| `lora-dataset/` | ? | v1 training dataset, pre-caption-fix. Historical only. |
| `lora-dataset-v2/` | ? | v2 training dataset. Historical only. |
| `lora-dataset-v3/` | 24 | Stale: an incomplete subset (7 images missing) that still carries 9 wrong captions. The invalid v3 run below trained on this by mistake. |
| `clawmarks-illustrious-dataset-v2.zip` | 31 | The actually corrected dataset, despite the confusing "v2" name. Produced after `caption_check_result.log` flagged the 9 mismatches; every caption verified against its image. |
| `clawmarks-illustrious-dataset-corrected.zip` | 31 | Same content as the v2 zip, re-zipped under a clearer name and uploaded to the current retrain pod. **Use this one.** |

`caption_check_prompt.txt` / `caption_check_result.log` hold the GPT-5.5-via-opencode
caption/image consistency check that found the 9 mismatches. The log's `MISMATCH:` lines give
the correct description for each.

### Checkpoints

| Dir | Trained on | Status |
|---|---|---|
| `checkpoints/` | v1 dataset, original captions | Historical baseline. |
| `checkpoints_v2/` | v2 dataset, first caption-fix pass | Historical. |
| `checkpoints_v3/` | `lora-dataset-v3` (stale) | Invalid. Don't use for generation. Kept for reference. |
| `checkpoints_v3_fixed/` | `clawmarks-illustrious-dataset-corrected.zip` (31 verified images) | **The valid run.** unet_lr 1e-4, text_encoder_lr 5e-5, min_snr_gamma 5, final loss 0.106. Epoch 4 is current-best (see Section 1). |

Each checkpoint directory holds epoch snapshots `-000002` / `-000004` / `-000006` / `-000008` /
`-final`.

### Comparison sheets already built

- `train_compare_sheet_3way.png`: v1 vs. v2 vs. v3 (invalid) side by side.
- `v3_epoch_compare_sheet.png`: epoch 2/4/6/8/final grid for the invalid v3 run. Useful only as
  a reference for what overfitting or undertraining looks like, not for picking a real
  checkpoint.
- `epoch_sheet_full.png`, `epoch_sheet_full_res.png`: supporting full-resolution crops for the
  above.
- Raw per-checkpoint batches backing these sheets live in `gen/`, `gen_train/` (v1),
  `gen_train_v2/`, `gen_train_v3/` (invalid), plus kohya's own in-training sample dumps in
  `samples/` (v1) and `training_samples_v3/` (invalid run).

`gpt55_prompt.md` / `gpt55_review.log`: an independent overfitting assessment on the invalid v3
run. It recommended epoch 4 as generally safest and proposed the validation-grid method (8-12
prompts × 3-4 seeds across epoch 2/4/final) that the real retrain later used.

### Infra and access

- SSH key: `runpod-ssh/id_ed25519(.pub)`, reused across every pod recreation via the `PUBLIC_KEY`
  env var.
- Helper scripts `rpssh.py` / `rpget.py` / `rpsftp.py`: edit the `HOST`/`PORT` constants at the
  top of each whenever a new pod exists:
  `sed -i 's/HOST = ".*"/HOST = "x.x.x.x"/; s/PORT = .*/PORT = NNNN/'`.
- Base checkpoint: Illustrious SDXL v0.1. Civitai model ID in `clawmarks_model.json`, downloaded
  via `https://civitai.com/api/download/models/889818?token=...`.
- Dependency gotcha, hit twice already: pin `torch==2.4.1` and
  `xformers==0.0.28.post1 --no-deps`, both from `--index-url https://download.pytorch.org/whl/cu124`,
  installed with `uv`. An unpinned `xformers` install lets the resolver silently upgrade torch to
  an incompatible version.
- kohya dataset convention: `--train_data_dir` takes the *parent* of a repeat-count-prefixed
  subfolder (e.g. `img/10_trentbuckle/`), not the image folder itself.
- v3 hyperparameters: `network_dim 32`, `network_alpha 16`, `unet_lr 1e-4`, `text_encoder_lr 5e-5`,
  `lr_scheduler cosine` (3 cycles), `min_snr_gamma 5`, `clip_skip 2`, AdamW8bit, bf16,
  `max_train_epochs 10`, `train_batch_size 4`, 1024² bucketed, `seed 42`.
- Serverless endpoint `uix4vdb2cec7sb` (RunPod, EU-RO-1) runs `runpod/worker-comfyui:5.8.6-base`
  (template `u45jy611b1`) with network volume `pwkmq2gjhw` (20GB) holding
  `illustrious_v0.1.safetensors` (base) and `clawmarks-illustrious-v3-epoch4.safetensors` (LoRA)
  under `/models/checkpoints` and `/models/loras`. ComfyUI auto-detects these at
  `/runpod-volume/models/...` on serverless workers. Submit jobs via
  `POST https://api.runpod.ai/v2/uix4vdb2cec7sb/run` with a ComfyUI API-format workflow JSON (see
  `gen_batch.py`). The endpoint scales to zero and costs little to leave idle, so nothing forces
  its teardown; terminate it only when no more generation is planned.
- RunPod API key: not stored in any durable config file. It has lived only in session
  transcripts; grep prior `.jsonl` transcripts to recover it if lost.
- **Pod idle policy: pause (`podStop`), don't terminate, once a batch finishes and the pod is
  sitting idle.** Pausing keeps the pod and its disk intact and drops billing to storage-only
  cost; terminating destroys the pod and its ephemeral disk permanently and changes the next
  pod's SSH host/port, requiring `rpssh.py`/`rpssh2.py`-style helpers to be re-pointed. Reserve
  terminate for a pod that's genuinely done and won't be reused. The `runpod-status` skill
  (`~/.claude/skills/runpod-status/`) and this project's `CLAUDE.md` both encode this now; check
  idle status proactively (GPU utilization + matching process, not just the dashboard label)
  any time pod state is even tangentially relevant, not only when asked.

### Gotcha log

- **Duplicate job submission (~500 jobs instead of 250).** Killing a backgrounded Python script
  because its log looked empty (stdout buffering, not an actual stall) let it finish posting most
  of its 250 jobs before it died. Relaunching then double-submitted, leaving roughly 500 total
  jobs queued. RunPod bills GPU-seconds actually run, not queue depth, so the cost impact stayed
  small (about $1-3 wasted), but draining the duplicate backlog before the tracked batch could
  start cost real wall-clock time. Lesson: launch background Python with `-u` (or otherwise force
  unbuffered output) before treating "no log output yet" as a stall.
- **`transformers.CLIPModel.get_image_features()` returns `BaseModelOutputWithPooling`**, not a
  raw tensor, on `transformers` 5.12.0. Fix: append `.pooler_output` before normalizing.
- **`AutoImageProcessor.from_pretrained("facebook/dinov2-base")` raises
  `ImportError: requires Torchvision`.** Rather than install torchvision and risk an unpinned
  package silently touching the pinned CPU-only `torch==2.12.0+cu130` build, fetch the model's
  `preprocessor_config.json` via `huggingface_hub.hf_hub_download` and reimplement the
  resize/crop/normalize pipeline by hand with PIL, numpy, and torch. This avoids the dependency
  entirely.
- **Dataset folder names lie.** A folder named `-v3` held a stale, incomplete (24 of 31 images)
  pre-fix snapshot, while the real corrected captions ended up in a zip confusingly named
  `clawmarks-illustrious-dataset-v2.zip`. Always diff a dataset's actual caption content against
  `caption_check_result.log`'s `MISMATCH:` entries, or rerun the consistency check, before reusing
  a cached dataset folder for training. Never trust the folder name alone.
- **`api.runpod.io/graphql` 403s bare `urllib` requests.** Python's `urllib.request` with no
  explicit `User-Agent` gets a Cloudflare block (`403`, body `error code: 1010`) on this host, even
  though the identical query succeeds via `curl`. The serverless REST host (`api.runpod.ai`)
  hasn't shown this. Fix: always set `User-Agent` (e.g. `"curl/8.0"`) on GraphQL requests.
- **`runpod/pytorch` dropped its old version-numbered image tags.** `2.4.1-py3.11-cuda12.4.1-...`
  (used in `rp_bring_up.py`'s first draft) no longer resolves; Docker Hub now serves mostly
  `1.0.7-rc.*` tags, though some old-style tags (e.g. `2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
  still exist. Check `https://hub.docker.com/v2/repositories/runpod/pytorch/tags/<tag>` for a 200
  before trusting an image tag in a pod-creation script; the base image's preinstalled torch
  version doesn't matter anyway since `remote_setup.sh` installs its own pinned venv.
- **kohya's default `caption_extension` is `.caption`, not `.txt`.** This dataset's captions are
  `.txt` files; without `--caption_extension .txt` explicitly, training would silently run with
  empty captions (no error, just an unconditioned LoRA) rather than failing loudly.
- **kohya_ss (`sd-scripts`) imports `torchvision` at module load time**
  (`library/utils.py` -> `library/original_unet.py`), even though this project's own DINOv2
  scoring scripts deliberately avoid it. `remote_setup.sh`'s pinned venv step only installed
  `torch`/`xformers`, so the first real training run failed immediately with
  `ModuleNotFoundError: No module named 'torchvision'`. Fixed by pinning
  `torchvision==0.19.1` (the release matched to `torch==2.4.1`) alongside the other two.
- **`rpssh.py` runs a non-login, non-interactive shell**, so `~/.local/bin` (where `uv` installs)
  isn't on `PATH` even though it's in `.bashrc`. Any one-off remote command that calls `uv`
  directly (outside `remote_setup.sh`, which exports `PATH` itself) needs
  `export PATH=$HOME/.local/bin:$PATH` prepended explicitly.
- **The RunPod `runpod/pytorch` base image has no `unzip`, and `remote_setup.sh`'s dataset step
  ran without `set -e`.** The `unzip` call failed silently, every subsequent `mv`/`rmdir` in that
  step failed too, but the script still reached `touch dataset.done` at the end, marking a failed
  extraction as complete. Fixed by installing `unzip` first and wrapping the extraction step in
  `set -e`/`set +e` so a real failure stops the script before the marker is written, rather than
  silently leaving `/workspace/training/img/` empty on a "successfully" set-up pod.
- **The 780-step full-length figure assumes `train_batch_size 4`, not 1.** 31 images x 10 repeats
  / batch 4 = ceil(310/4) = 78 steps/epoch x 10 epochs = 780. With the outlier now down-weighted
  (30 images x10 repeats + 1 x3 repeats = 303 image-repeats), that's ~76 steps/epoch, close enough
  that probe/calibration runs pass `--max_train_steps` explicitly rather than deriving it from
  epoch count.
- **`train_probe.py`'s remote command redirects all training output to a log file on the pod**
  (`> train.log 2>&1`), so the SSH channel carries zero bytes for the entire ~20-minute training
  run. With no keepalive configured, `dim64_780`'s launch hit a paramiko `socket.timeout` on the
  read side well before the training itself finished, even though the remote process kept running
  unaffected (it was writing to a local file on the pod, not blocked on the client reading
  anything) and completed successfully. The wrapper script's crash meant its checkpoints sat
  un-downloaded until recovered manually. This isn't the `timeout=3600` parameter's intended
  meaning kicking in (the run finished in ~20 min, well under that); it looks like an idle
  network path (a NAT or proxy somewhere between here and the pod) dropping a connection with no
  traffic on it. Fixed by calling `client.get_transport().set_keepalive(30)` in `ssh_client()` so
  the connection sends periodic traffic and doesn't look idle, even while the actual command
  output is silent on the channel.

### Unrelated material in this directory

- `fal-workflow.md`, `fal-workflow-review/`: an unrelated fal.ai workflow-skill review, no part of
  the CLAWMARKS LoRA work.
- `art-style-prompts.md`, `clawmarks-evolve/` (brief.md plus per-model variants and contact
  sheets): an earlier brainstorming round on hand-written per-image captions across multiple LLMs
  (GPT, Opus, Fable, DeepSeek), upstream of the current caption set.
- `clawmarks.safetensors`, `clawmarks_model.json`: an early prototype LoRA and its Civitai listing
  metadata, predating the v1/v2/v3 runs above. Superseded.

---

## 6. Lab log

*(Dated entries go here as rounds run: status, decisions, numbers, surprises.)*

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

### 2026-07-09: Noise floor measured, replicate count (n) derived, effect-size floor revised from 0.02 to 0.05

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

Derived the actual replicate count via simulation (sign-flip permutation test, 4000 simulated
trials x 2000 permutations each, per-prompt noise variance from the table above, delta variance
= 2x single-run variance since a direction-vs-control delta carries noise from both sides):

| true effect (cosine) | n=3 | n=4 | n=6 | n=8 |
|---|---|---|---|---|
| 0.02 | 11% | 15% | 19% | 24% |
| 0.05 | 44% | 54% | 74% | 84% |
| 0.08 | 78% | 89% | 98% | 100% |

0.02 is undetectable at any practical n (24% power even at n=8; adding more replicates helps
only slowly). **Decision: raise the effect-size floor from 0.02 to 0.05 cosine, and set round
1's replicate count to n=8** (84% power at 0.05, 98%+ at the scale of gaps actually seen in the
calibration table: dim64's ~0.06 gap, constlr's ~0.10 swing). Updated Section 3 steps 2-3 and
the budget estimate accordingly: n=8 x ~10 directions per round raises probing alone to 8-13
GPU-hours per round, not the earlier 80-110 minutes, so total budget across 5 rounds plus
calibration moves from an estimated 11-13 hours to **45-70 hours**, worth running on two pods in
parallel as calibration already did.

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
