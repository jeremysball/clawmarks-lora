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
   real effect (most of the bootstrap mass positive) and a practically meaningful size (at least
   0.05 DINOv2 cosine, not just statistically nonzero). With ~10 probes tested per round, apply
   a multiple-comparisons correction (Benjamini-Hochberg). Taking the first direction that
   clears a loose bar, with this many tested per round, guarantees some spurious wins by chance
   alone.

   **Corrected, 2026-07-13:** the sentence above originally treated "rank every direction and
   only ever advance the single best one" as an equivalent alternative to Benjamini-Hochberg.
   It is not; see the external-review entry below ("'Advance only the single best direction' is
   not a substitute for a real multiple-comparisons correction"). Taking the argmax of noisy
   statistics controls no false-discovery rate and is subject to winner's curse. The corrected
   rule: keep the significance-plus-effect-size gate first (screen every direction, advance
   nothing if none pass), use "take the single best" only as a tie-break among directions that
   already passed the gate, and reserve full BH correction for any whitepaper claim of the form
   "we tested N, M improved."

   **Note on step 3 in practice: the noise floor has to be measured before it can be used.**
   "Beats the noise floor" only means something once the floor itself is a real number, not an
   assumption. The floor comes from the pooled control-only probes: score each one, take
   pairwise deltas between them (same fixed prompt/seed slots as everything else), and the
   spread of those deltas, which should average to zero since they're all the same config, is
   the noise floor's empirical standard deviation. Every direction's delta against control has
   to clear this floor, both statistically (permutation test or bootstrap CI, above) and in
   practical size (at least 0.05 cosine).

   This same number also decides how many replicates (n) round 1 actually needs, which the
   "3-4 replicates" starting assumption above was a guess at, not a derived figure. Once the
   noise floor's spread is measured from real control probes, simulate: generate synthetic
   paired deltas with that measured spread plus an injected effect, run the same permutation
   test at a few candidate n values (3, 4, 6, 8), and see which n detects the injected effect
   at least 80% of the time. That n, not the guess, is what round 1 should use.

   **Corrected, 2026-07-13** (see the lab log entry below for the reproducible results): the
   exact sign-flip test makes rejection impossible at alpha=0.05 for n=3 and n=4 because their
    one-sided p-value floors are 0.125 and 0.0625. With the calibration-noise proxy, the
   sign-flip test itself (p<=alpha, ignoring effect size) gives n=8 a power of 79.90% at a 0.05
   effect and 99.20% at 0.08.

   **Corrected again, 2026-07-13:** those two numbers describe the sign-flip test's own power,
   not round 1's actual decision rule, which requires p<=alpha **and** an observed mean delta
   >= 0.05 (see the Selection rule above). Requiring both is strictly more conservative. The
   full gate's power at n=8 is **49.42%** at a 0.05 effect (not 79.90%) and **95.34%** at 0.08
   (not 99.20%), reproducible via `probe_power.py`'s "Round-1 gate power" table. Round 1
   therefore keeps the **eight canonical paired training seeds**, uses 0.05 as an exploratory
   practical threshold, and prespecifies 0.08 as the effect size for an 80%-power
   per-direction screening claim, since 0.08 is the effect size where the full gate, not just
   the test, clears 80% power. A result at the 0.05 threshold has only coin-flip power to be
   detected at all and cannot be presented as confirmatory evidence under this planning model.

   **Note added 2026-07-09: paired seeds make a separately-measured noise floor optional, not
   required.** A sign-flip permutation test builds its null distribution from the very deltas
   under test. Given a direction's 8 paired deltas (`direction_score[seed_i] -
   control_score[seed_i]`, same seed on both sides via `CANONICAL_SEEDS`), the test asks how
   often random sign-flips of those same 8 numbers would produce a mean this large. It never
   needs an externally-measured floor constant to run. That constant was only ever a stand-in for
   two side calculations: the practical exploratory effect-size floor (0.05 cosine, but it can
   come from round 1's own paired deltas rather than a dedicated batch) and the replicate count n
   (n=8 was derived from unpaired noise; pairing may shrink variance, but that reduction has not
   yet been measured). Practical
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

**Budget, revised 2026-07-13 after the corrected power analysis:** with n=8 replicates (up from
the earlier 3-4 guess) across roughly 10 directions per round, that's ~80 probes per round at
6-10 minutes each, 8-13 hours of probing alone, plus one 34-minute commit run scored at 5
checkpoints. Call it **9-14 hours per round**, not 2.2-2.5. Five rounds plus the one-time
calibration check: **45-70 hours of GPU time**, a large jump from the original 11-13 hour
estimate, and worth running two pods in parallel (as calibration already did) rather than
serially. The corrected analysis does not claim 80% power at the exploratory 0.05 threshold;
it uses 0.08 as the prespecified 80%-power per-direction screening effect while retaining 0.05 as a practical
screen (see the 2026-07-13 log entry).

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

**Honest read for the whitepaper:** the novelty trajectory is exploratory and selection-biased, not
evidence that reinforcement was real rather than noise. The cumulative best can rise under pure
noise because each generation gets another chance to find a favorable fluctuation, and the exploit
pool repeatedly reuses selected images. Two candidate explanations, neither confirmed: (1) this
specific style, at these settings, may simply have a fairly low novelty ceiling under the current
LoRA, the liminal band it can support without breaking faithfulness is narrower than the original
proposal's illustrative numbers assumed; or (2) the exploit/explore split (half of every batch
mutates near existing high scorers) may have been too exploit-heavy once the pool filled with
similar winners, starving exploration of the room it needed to find a genuinely different region
rather than a refinement of the same one. A per-generation cohort statistic or an untouched replay
comparison is required before making a reinforcement claim. Neither has been run.

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
- **A keyword-scoped `grep` over a multi-section design spec silently drops sections it wasn't
  told to look for.** The `2026-07-16-sulfur-proof-shared-shell.md` plan was written by grepping
  the Sulfur Proof spec for depth-grammar vocabulary ("raised", "recessed", shadow strengths) to
  extract `CONTROL_CSS`'s five classes. That grep never matched the spec's separate "Controls"
  section (primary actions, selected controls, workflow stepper), so the plan silently omitted a
  `.primary-action` component, and the taskferry implementer prompt it produced went further and
  mislabeled `.raised-control` as "the spec's primary interactive control," turning an omission
  into something that read as intentional to the dispatched worker. Every button in the app was
  stuck rendering as `.raised-control` (flat paper-on-paper) with no way to ever hit the spec's
  black-fill/sulfur-underline treatment, until this surfaced as "the theme looks wrong" days
  later. Lesson: when extracting a plan from a multi-section spec, read the whole spec's section
  headers first and account for every section explicitly (even a one-line "N/A, out of scope for
  this task"), rather than grepping for the vocabulary you already expect to find. A scoped grep
  can only ever confirm what you already knew to look for.

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

> Entries before 2026-07-16 were moved to `notes/lab_notebook_archive.md` on 2026-07-18 to keep this file scannable. Nothing was deleted, only relocated; see that file for the earlier dated history.

### 2026-07-16 (session 18): live Playwright pass for Tasks 1-4, then merged the whole stack

Ran the full suite first (423 passed) and started `curation_server.py` from this worktree, the
tip of the entire reconciled stack. The real leg `trent_v3_epoch4/freeform1` (50 scored entries)
has no expedition metadata inside this worktree, since `expeditions/trent_v3_epoch4/` is untracked
in the main checkout; copied its two small JSON config files in for the duration of the check
(no generation output touched) and deleted the copy afterward.

Verified Task 3 (undo recovery) end to end in the browser: opened the lightbox on `scan.html`,
favorited an image (counter went to "1 favorited", button read "favorited (click to remove)"),
removed it (counter dropped to 0, an "Undo" button appeared with "Removed favorite. Undo?"), then
clicked Undo and confirmed the favorite came back ("1 favorited" again). No new console errors
from any of these actions.

Verified Task 1 (favorite mutation leg-binding) at the API level: `POST /api/favorite` with an
`expedition`/`leg` pair that didn't match the server's actual active selection
(`uncanny_frontier/cockpit` while `trent_v3_epoch4/freeform1` was active) correctly returned 409
`"favorite mutation targets a stale expedition/leg"` instead of silently writing to the wrong
leg's favorites file.

Verified Task 4's sibling branch (a leg that was never launched, `n_entries == 0`) renders the
existing "launch a round" advice rather than an error, confirming the never-launched/damaged
distinction didn't regress the common case. Did not fabricate a damaged-leg (`n_entries > 0`,
files missing) test fixture against real state directories to check the other branch; that
scenario is unit-tested (422 passed per session 17) and manufacturing fake corruption in
`$XDG_STATE_HOME/clawmarks/` risked exactly the kind of accidental data-integrity incident this
project's CLAUDE.md exists to prevent.

Did not exercise Task 2's stop-run identity mismatch live either: with no search run active,
`POST /api/searchrun/stop` with a bogus PID correctly no-ops (`{"running": false}`), but the real
mismatch path (a stale PID rejected against a genuinely running process) requires an actual paid
RunPod search run to trigger, and starting one solely to click a stop button isn't a reasonable
use of budget. Relying on the existing unit coverage (419 passed at Task 2) for that path.

Restored all state touched during the check: unfavorited the test image, switched the active leg
back to `uncanny_frontier/cockpit` (the value present before this check began), removed the copied
`expeditions/trent_v3_epoch4/` directory, and stopped the temporary server.

One unrelated finding, out of scope for this pass: `scan.html` requests thumbnails as bare
filenames (e.g. `/gen1_explore_23_seed194542.png`, 404) instead of `/thumbs/<tag>.jpg` for any leg
whose thumbnails haven't been generated yet. `scan_gallery.compute_data` (`build/scan_gallery.py:63`)
falls back to the raw basename when `os.path.exists(thumb_path)` is false at manifest-compute time,
rather than always pointing at the `/thumbs/` route so the server's on-demand thumbnail generation
(`curation_server.py:1486-1495`) ever gets a chance to run. Confirmed on `trent_v3_epoch4/freeform1`,
a real leg with no `thumbs/` directory yet. Not a Tasks 1-4 regression (reproduces on data, not on
new code), but real: any freshly-scored leg without pre-generated thumbnails currently shows broken
images in the scan grid until something else populates `thumbs/`. Worth its own follow-up task.

With the live check clean, merged the whole reconciled stack (phases 1-8 plus review-fix tasks 0-6,
39 commits ahead of `main`) into `main` via PR #47, retargeted from its previous base
(`pr41-task4-integrity-error`) to `main` directly, since `pr41-task6-lowseverity-fixes` (PR #47's
head) was confirmed a strict git ancestor superset of every intermediate phase and task branch.
PRs #34-46 were closed as superseded rather than merged separately, since every commit they
contain already rides along inside PR #47's merge.

### 2026-07-16: expedition/leg config moved from git to `$XDG_STATE_HOME` (ADR 0001)

Jeremy decided the `trent_v3_epoch4/expedition.json` placeholder-field question from the prior
entry should be answered by removing the underlying split entirely, not just this one file:
expedition/leg config should never have been git-tracked in the first place, since it was the
`ROOT/expeditions/` vs. `STATE_DIR/expeditions/` split itself that let `trent_v3_epoch4` end up
with real generation output but no config anywhere in git. Wrote up the reasoning as
`docs/adr/0001-expedition-config-lives-in-state-dir.md`, the project's first ADR.

`config.EXPEDITIONS_DIR` now resolves to `STATE_DIR / "expeditions"`, the same root
`config.leg_dir()` already used, so a given expedition's config and its per-leg output now live
side by side in one directory, distinguished by `legs/<leg>.json` (config) versus a bare `<leg>/`
(output). Migrated both existing expeditions by copying their `expedition.json`/`legs/*.json` into
the matching `$XDG_STATE_HOME/clawmarks/expeditions/<name>/` directory (content-diffed clean
against the originals first), `git rm`'d the previously-tracked `uncanny_frontier` config, deleted
the untracked `trent_v3_epoch4` config from the repo tree, and added `/expeditions/` to
`.gitignore` so the split can't quietly reappear. `trent_v3_epoch4`'s placeholder fields
(`textures`, `fallback_subjects`, budget/generation-count, still borrowed from `uncanny_frontier`)
are unchanged by this move; only their storage location changed, so that specific review question
is still open, just no longer blocking anything on the git side.

Updated `test_config.py::test_expeditions_dir_is_repo_relative` (renamed
`test_expeditions_dir_is_state_dir_relative`) to assert the new location; every other test already
used `monkeypatch.setattr(config, "EXPEDITIONS_DIR", ...)` and needed no change. Full suite: 427
passed. Ruff, MyPy, and `git diff --check` all clean.

### 2026-07-16: GLM review of the ADR 0001 diff found a real regression it introduced

Dispatched the ADR 0001 diff (config.py, curation_server.py, test_curation_server_
expedition_routes.py, .gitignore) to `cheapestinference/glm-5.2` at max effort via taskferry
(delegate-code-review's single-dispatch path, since the diff was self-contained at ~285 lines).
The prompt named the specific claim to break: that config and per-leg output paths never
collide now that `EXPEDITIONS_DIR` and `leg_dir()` share one root.

The model broke it. Before this migration, `EXPEDITIONS_DIR` (`ROOT/expeditions`, git-tracked)
and `leg_dir()` (`STATE_DIR/expeditions/...`) were two separate filesystem trees, so a leg named
`"legs"` never collided with anything. Moving both onto `STATE_DIR/expeditions` means
`leg_dir(expedition, "legs")` now resolves to the exact same directory that holds every other
leg's `legs/<leg>.json` config file. `_create_leg` only rejected a blank name, so nothing stopped
this; `_list_expeditions()`'s `legs_dir.glob("*.json")` would then list every generation artifact
written into that directory (`scored_manifest.json`, `user_favorites.json`, etc.) as a bogus leg.
A real regression introduced by this migration, not a pre-existing bug.

Fixed by adding `_validate_expedition_or_leg_name()`, called from both `_create_expedition` and
`_create_leg`, rejecting a path separator or `..` in the name (also closes a pre-existing,
lower-severity path-traversal gap the model flagged in the same pass, present before this
migration too) and rejecting the reserved leg name `"legs"`. Four new regression tests cover the
reserved-name rejection and the path-separator rejection for both expeditions and legs.

The model's third finding also held up: the docker-compose bind mount (`./state:/app/state/
clawmarks`) means a host running docker-compose from a repo checkout gets expedition config
(and generation output, already true before this migration) landing directly in `<repo>/state/`,
which `.gitignore` never covered. Added `/state/` to `.gitignore`.

Verified every claim by hand before fixing anything (per delegate-code-review's rule not to
trust a model's mechanism claim at face value): read `_create_leg`, `_create_expedition`, and
`_list_expeditions()` directly, confirmed the collision is real by tracing `config.leg_dir()`
against the new `EXPEDITIONS_DIR` value, and confirmed `git check-ignore` returned nothing for
`state/` before the fix. Full suite after the fix: 430 passed. Ruff, MyPy, and `git diff --check`
clean.

### 2026-07-16: exploration workflow review exposed the missing map-to-generation handoff

Created a durable visual review at `docs/design/explore-workflow-review/index.html` and a
continuation brief at `docs/design/explore-workflow-continuation-prompt.md`. The review preserves
the approved route and navigation decisions: Explore becomes `/`, the current status and leg
picker move to `/status.html`, every tool uses a shared sticky context header, and clicking the
active leg opens an in-place selection modal rather than navigating away. It also defines the
workflow-card pattern for future documentation and in-product guidance: show the user's goal,
route/context, concrete action, and visible outcome, using real CLAWMARKS art or live UI captures
instead of generic placeholder shapes. Scrollbars are part of the themed dark UI, not browser
defaults.

Live Playwright inspection against `trent_v3_epoch4/freeform1` confirmed the exploration tools
are useful for evidence navigation but do not yet make a true embedding-space navigator. The
Solution map lets a researcher replay generations, inspect a point, and filter by a nearest real
training-image anchor. Coverage highlights empty but adjacent-to-dense frontier cells, and
Redundancy exposes near-copy clusters. The essential transition is broken: Coverage's “Target
this gap in cockpit” link passes no selected cell, while opening `/cockpit.html` mutates the
global active leg to the expedition's empty `cockpit` leg. A researcher therefore loses the exact
region they inspected and cannot direct a trial from it. The next product-design task is an
explicit expedition/leg-scoped selected-region object that survives across map, coverage,
comparison, and Cockpit, then maps a resulting batch back to the originating region.

Backed up and content-diffed `$XDG_STATE_HOME/clawmarks/` before restoring `freeform1` for the
live inspection (161 files in both copies). The Redundancy page also produced four real thumbnail
404s because it used bare filenames instead of `/thumbs/<tag>.jpg`; this matches the existing
fresh-leg thumbnail-route follow-up and is recorded in `TODO.txt`.

### 2026-07-16: Focus dossier architecture selected; LLM chat and visual alternatives checkpointed

Continued the selected-region-to-Cockpit design from
`docs/design/explore-workflow-continuation-prompt.md`. Compared three interaction models: a
persistent Focus dossier, a dedicated region workspace, and an immediate Cockpit draft. Jeremy
approved the Focus dossier. A Focus is the durable, expedition/leg-scoped evidence object; a trial
is a bounded test derived from one Focus revision. The intended handoff preserves fixed member
tags, score ranges, nearby real anchors, human judgments, source view, and a working hypothesis.
Cockpit receives the Focus explicitly, stays on the originating leg, snapshots the tested revision,
and writes `focus_id` and `trial_id` onto generated results so Coverage, DINOv2 neighbors,
redundancy, and human review can compare the batch with its originating region. Raw UMAP screen
coordinates remain illustrative; stable member tags and high-dimensional image neighbors define
the durable region.

The review also established that the LLM belongs between evidence and action, and again after a
trial. It should propose competing interpretations, challenge a draft hypothesis, help turn it
into a testable brief, and compare completed results with the predeclared expectation. It must not
select the region, alter evidence, launch paid work, or decide whether the hypothesis won. This
requires a small Focus-scoped chat interface. The server will shell out to OpenCode for the actual
conversation; conversation persistence, context assembly, image input, and safe apply-actions still
need their own design pass. The current Autopilot receives score and prompt metadata but no image
pixels, so it can honestly act as a brief coach, not a visual interpreter. Any feature labeled
visual interpretation must send representative generated images and real-art anchors to an
image-capable model.

Three hypothesis representations remain under review: natural language alone, a fixed structured
form, and a hybrid research brief. The current recommendation keeps the researcher's natural
language verbatim, then asks the researcher to confirm a small contract: intention, evidence scope,
changed variable, held-constant settings, expected visual move, and evidence that would count
against the idea. This preserves visual nuance while preventing the LLM or UI from silently moving
the goalposts. Jeremy has not approved that representation yet.

Built a Lavish review surface at `.lavish/focus-dossier-design.html`. Its product architecture was
useful, but Jeremy rejected the visual treatment: too many nested boxes, text too small, weak
typography, and no clear path for the eye. The next design step is to show several substantially
different compositions with fewer enclosing panels and stronger editorial hierarchy. No production
implementation has started.

The composition follow-up clarified the chat's product role. It must be accessible from every tool
page, receive the current page plus active expedition/leg, Focus, and local selection as context,
close without discarding its conversation, and have a first-class mobile treatment, most likely a
pull-up bottom sheet. This rules out making chat a permanent notebook margin or a fixed left column:
those layouts give the LLM too much ownership of the screen and consume map space even when the
researcher is not using it. The current direction is a shared Guide control in the cohesive header,
opening a dismissible desktop drawer or overlay and a mobile pull-up sheet, with the same persisted
Focus-scoped OpenCode conversation behind both.

The second Lavish pass also confirmed that composition alone does not fix the visual problem. Jeremy
wants explicit theme alternatives, a new color system, fewer enclosing panels, and clearer labels
for map-selection graphics. The unlabeled dashed ellipses used to suggest a lasso were ambiguous in
all three mockups. The next visual review will hold the interaction pattern constant, label selected
regions directly, and compare typography, palette, surface treatment, and desktop/mobile Guide
states across several themes.

Jeremy then corrected the theme study's scope: the redesign covers the entire curation site, not
only the new Focus and Guide feature. He liked the warm-gray-paper and dense-black-ink direction but
rejected the oxide-red and ultramarine colors used in that draft. The next review must demonstrate a
coherent site system across Explore, Solution Map, Coverage, Compare, Cockpit, Runs, and the Guide,
not repeat one Focus screen in different palettes. The current visual throughline is a working print
proof: real CLAWMARKS art and research evidence share one warm gray table; hierarchy comes from
dense black ink, scale, rules, registration marks, and annotation layers rather than nested cards.
Color, if used, behaves as a functional annotation ink for selection, warnings, or Guide context,
not as dashboard decoration.

The interactive site-system pass applied that throughline across seven representative routes and
made the shared Guide open from each page with a page-specific context receipt. Jeremy selected the
Sulfur marker direction, but did not approve its first palette: the warm gray and yellow-green did
not mesh, and the system still needs more design attention. The refinement rule is to stop treating
sulfur as a second brand color. Dense black remains the hierarchy; sulfur becomes a physical
annotation material used as translucent overprint, hatching, underlines, and active registration
marks. The paper should shift cooler with a slight olive cast, and large filled sulfur controls
should become black controls with restrained sulfur marks.

### 2026-07-16: Sulfur Proof v2 approved; workflow cards replaced with a stepper

The refined site-system review cooled the paper to olive-gray, muted the sulfur pigment, and used
sulfur only as annotation overprint on a dense process-black hierarchy. Jeremy approved this
Sulfur Proof v2 direction for the whole curation site. The approved palette uses paper `#C3C5BA`,
deep paper `#B3B5A9`, ink `#11120F`, rules `#898D81`, sulfur `#CBD63F`, Guide surface `#20251B`,
and Guide ink `#ECEFDF`. The secondary text token moved from `#55594F` to `#4D5048` after approval
to raise normal-text contrast against paper from 4.10:1 to 4.70:1, above the WCAG AA threshold.

Jeremy then rejected the Explore page's Orient, Scout, Explain, Act, and Learn treatment because
the five stages still looked like cards rather than controls. The review surface now renders them
as one connected navigation rail with real button elements, one strong active state, keyboard
focus, and one shared detail line below the rail. This preserves the research loop without turning
five short verbs into five equal content panels. The written interaction spec must describe this
as a compact stepper or tab bar, never a row of workflow cards.

### 2026-07-16: Explore changed from a landing page to an active research desk

The connected Orient, Scout, Explain, Act, and Learn button treatment worked, but Jeremy correctly
identified a larger composition failure: the oversized question, decorative image collage, broad
hero spacing, and recent-Focus strip still looked like a product or SaaS homepage. He selected an
active research desk instead. Explore now places the workflow control directly below the shared
header, then shows the current Focus as a compact working heading over one ruled surface. Saved
evidence and observations occupy the main width; the next decision sits in a narrow side column;
a chronological Focus ledger closes the page. Images appear as labeled evidence, not decoration.
The corresponding specs now prohibit welcome heroes, marketing copy, feature cards, and automatic
selection of a recent Focus. A bare Explore URL shows a ruled Focus ledger; an explicit Focus URL
opens the desk. The revised Lavish surface produced no severe layout warning during five minutes
of browser-side auditing.

### 2026-07-16: first independent review hardened the Focus and Guide specifications

An independent GPT-5.6 Sol advisor reviewed the four design specs and ADR 0002 against the current
server and data-integrity rules. It found 14 concrete gaps. The most serious were real: the draft
applied backup and spend rules only to Focus trials even though standalone Cockpit and
counterfactual endpoints can also spend money; it froze an unverified safety receipt before launch;
it had no atomic `launching` claim, so concurrent requests could duplicate paid work; and it required
a spend-cap value without proving the worst-case estimate fit under it. The current code confirmed
those mechanisms: Cockpit and counterfactual generation check only account balance, Cockpit marks a
trial running before preflight, and downloaded images can precede any durable provenance record.

The revised design routes every paid generation endpoint through one gate, claims launches with an
idempotent `confirmed` to `launching` transition, enforces conservative cost inequalities, and
backs up the complete leg under a dedicated state-level backup tree while holding cross-process
locks. A durable per-result receipt now records provider identity before download and image checksum
before manifest update, allowing repair without deleting an orphan image. Trial creation requires
the reviewed Focus revision and snapshots manifest, image, human-judgment, and calculation digests.
The Guide now runs OpenCode in pure mode from an empty temporary directory with discovery, mutation,
delegation, web, and MCP tools denied and secrets removed. URL parameters, not another global
pointer, define the active Focus. The revision also added source-specific Focus validation,
file-plus-parent `fsync`, and explicit image, map, dialog, and screen-reader requirements. A
follow-up advisor pass is checking these corrections before the specs reach Jeremy for review.

The follow-up review found seven remaining crash and concurrency edges, then five narrow wording
contradictions. The specs now give every paid image route a generic `launch_id`, write a durable
submission intent before each provider call, forbid automatic resubmission unless the provider
enforces idempotency, and leave ambiguous query-only jobs in `needs_reconciliation`. Numbered launch
attempts preserve every preflight failure. Account-wide durable reservations prevent two legs from
spending the same balance, while stable job slots cap accepted jobs across workers and retries.
Backups `fsync` every copied file and directory before recording success. Generated evidence binds
to that backup; real anchors copy into a content-addressed evidence bundle; complete judgment and
derived values accompany their digests. Focus sources now reject cross-leg tags, unknown real-art
tags, inverted ranges, and values outside the true cosine-derived domains (faithfulness `[-1, 1]`,
novelty `[0, 2]`). After these corrections, the final independent closure check reported no
remaining defect in the reviewed architecture and returned `Approved`.

### 2026-07-16: abstract skeuomorphism selected as the site-wide depth language

Further Lavish review refined the active research desk. The initial attempt to strengthen
clickability made the workflow rail too dark and unbalanced. Jeremy's annotations clarified that
flat evidence rows needed subtle source labels, while controls needed stronger affordance. The
workflow now uses light raised keys with one black active key; sulfur appears only in hard offset
shadows, switch marks, and small registration ticks. The research question moved from a large
article-like headline into a compact labeled Focus field, and the heavy divider above Evidence
became a faint rule.

Jeremy strongly approved the raised Next Decision plate, especially its small inner light edge, and
selected **abstract skeuomorphism** for the whole site. The durable design rule is tactile and hard,
not soft neumorphism: raised controls use crisp inner highlights, darker opposite edges, and
unblurred offset shadows; recessed instrument areas reverse those edges. The palette stays paper,
deep paper, ink, and restrained sulfur. Flat prose and data rows remain flat so depth marks actions,
decisions, callouts, payload confirmation, and instrument readouts rather than becoming another
all-panels treatment. The site-system specification now records this dimensional grammar.

### 2026-07-16: tactile depth moved from component fragments to whole-page comparisons

The first tactile study showed isolated tabs, readouts, and image frames. Jeremy correctly noted
that those fragments did not reveal how each depth rule would shape Explore as a whole. The revised
study holds one complete miniature Explore page constant, including its shared header, five-stage
workflow, Focus tabs, research question, five-image evidence wall, activity history, and Next
Decision. Three controls now switch that same page between edge hierarchy, instrument bay, and
evidence wall treatments. This makes the tradeoff visible as a page-level hierarchy rather than a
collection of unrelated component samples.

The raised Next Decision plate had already moved back to the paper plane, but its first full-height
recessed gutter resembled a scrollbar. The current version uses a narrow cut seam with three black
registration ticks and a sulfur offset on the left. The decision content remains flat. Desktop and
mobile Playwright checks confirmed all three study modes, the Focus tabs, workflow controls, Guide
drawer, and zero page-level horizontal overflow. That audit also found a pre-existing hit-target
bug in the Guide drawer: the visible close label sat beneath the context receipt. Raising the close
control in the stacking order restored its click target. Jeremy's tactile-direction selection
remains open before the durable design-system specification changes again.

Jeremy's next annotation pass selected a hybrid rather than one untouched direction. He liked the
Instrument Bay's mounted image frames and raised Next Decision, but found its activity wells too
deep and its recessed research-question strip visually awkward. The revised Instrument Bay keeps
the image mounts and decision plate, changes the activity history to Edge Hierarchy's shallower
detents, and turns the question into one lightly raised readout. The main Explore page now uses the
same raised Next Decision treatment. Paper grain is stronger across both the full prototype and the
miniature study, following his request for a less subtle background texture. This hybrid awaits one
more visual confirmation before it becomes the durable site rule.

Jeremy approved the revised hybrid without further changes. The final Explore depth allocation is
now explicit: light raised workflow keys with one black active key; two dossier tabs; a shallow
raised research-question readout; hard-edged mounted evidence images; shallow activity detents with
no outer shadow; and one raised Next Decision plate with a crisp inner light edge and hard black
shadow. Fine ruling, faint cross-grain, and broad tonal variation make the olive-gray background
read as proofing paper rather than a flat application fill. Sulfur remains limited to active edges,
registration marks, and evidence-role accents.

The final write-up updated the Sulfur Proof site design system, research-workspace navigation, and
Explore route specifications. It resolves two older ambiguities: evidence images may use functional
mounts even though ornamental frames remain prohibited, and the five-stage workflow is a compact row
of light raised keys rather than an all-black rail. The specs also define three depth strengths so
implementation cannot flatten the approved hierarchy or turn every section into a panel.

### 2026-07-16: durable paid-result storage design and implementation planning completed

RunPod retains asynchronous results for only 30 minutes, so the paid-work design could still lose a
successfully generated image after a CLAWMARKS crash or network outage. Jeremy selected a separate
Docker Compose project containing VersityGW 1.7.0 and a dedicated Tailscale 1.98.9 sidecar. Tailscale
Funnel exposes one public HTTPS hostname; VersityGW stores buckets as ordinary POSIX files. The
sidecar uses a tagged OAuth client and persistent non-ephemeral identity. Jeremy accepts Funnel's
beta status and fixed bandwidth limits.

The approved design pins both container images by manifest digest, publishes no host port, uses
Tailscale's default userspace networking, and requires path-style SigV4. Root S3 credentials stay on
the host. RunPod receives a restricted VersityGW user that can upload, read, and list current and
next UTC `MM-YY` buckets but cannot delete objects, buckets, policies, or ACLs. The exact pinned
`worker-comfyui:5.8.6-base` upload helper must pass an upload, presigned read, prefix list, and delete
denial test through Funnel before production use. A narrowly derived path-style worker image is the
defined fallback if upstream boto3 chooses virtual-host addressing.

Every paid dispatch now requires two verified complete mirrors: the expedition leg and the full
object plus IAM roots. It also requires bucket ownership, capacity, TLS, SigV4, and a public
write/read canary before `/run`. Known provider job IDs recover by listing their frozen two-bucket
set even after RunPod expires its response. An ambiguous submission without a provider ID reports
unclaimed prefixes for manual review and never resubmits or auto-adopts them. Remote objects remain
indefinitely.

The implementation plan is
`docs/superpowers/plans/2026-07-16-s3-funnel-durable-results.md`. It separates XDG paths and CLI,
the independent pinned stack, IAM and bucket bootstrap, complete mirrors, exact worker compatibility,
no-clobber S3 import, paid-gate integration, and one explicitly approved live recovery probe. The
paid-work plan now also specifies a durable global request-ID index and exact finite
`data.myself.clientBalance` parsing. The Guide plan sets `OPENCODE_DISABLE_EXTERNAL_SKILLS=1` and
tests that project, `.opencode`, `.claude`, and `.agents` sentinel skills remain undiscovered.

Local placeholder, hardcoded-path, interface, prose, and `git diff --check` reviews passed. Two
independent cheapestinference closure attempts produced no verdict: Kimi K2.7 crashed after starting
its reads, and GLM 5.2 stalled until cancellation. No reviewer edited the workspace. One successful
independent closure review remains before implementation.
### 2026-07-17: Strengthened Task 2's opposite-order lock contention test

The second independent review pass found that the opposite-order multiprocessing test only
observed a worker announcing its intent to call `flock` before the syscall could block. That
assertion could pass through scheduling luck without proving cross-process exclusion. The test now
waits for one worker to enter while both first lock attempts have started, then requires the other
worker's entry event to remain unset for 0.2 seconds before releasing the winner. This positively
proves the loser is blocked while the lock is held. A temporary no-op `flock` mutation failed at
the strengthened assertion, while the real implementation passed all 11 durable-record tests in
five consecutive runs. Ruff and MyPy remained clean. Production code was unchanged.

### 2026-07-17: Implemented Task 3 map-member Focus persistence

Added `src/clawmarks/focus_store.py` with frozen expedition/leg `Scope`, map-member source
validation, direct real-anchor validation, durable state-directory storage, revision-checked
updates, archive transitions, status-filtered listing, and readable corruption errors. Generated
manifest members must resolve exactly once and their resolved files must remain inside the scoped
`config.leg_dir()`; duplicate source tags are deduplicated in input order while natural-language
fields remain unchanged. Every create, update, and archive writes through `atomic_json_write()`
under the durable per-record `fcntl` lock. The existing Focus test file was left behaviorally
unchanged apart from removing two unused imports required by Ruff.

The focused suite passed 10 tests, the full suite passed 453 tests, Ruff passed, and MyPy passed.

### 2026-07-17: Implemented Task 4 coverage-frontier Focus validation

Extended `src/clawmarks/focus_store.py` so `FocusStore.create()` dispatches on `source.kind`.
The existing `map_members` branch keeps its behavior; a new `coverage_frontier` branch validates
`score_ranges` as two finite numbers per metric with `min < max`, faithfulness within `[-1.0, 1.0]`,
novelty within `[0.0, 2.0]`, deduplicated `adjacent_member_tags` resolving exactly once in the
scoped manifest via the same leg-containment path as member tags, real anchors via the existing
real-art validation, and a caller-supplied `coverage_cells` list whose bin exactly matches the
requested score range with `count == 0` and `frontier is True`. `coverage_hint` is preserved
opaquely as the design spec allows; the canonical score ranges, deduplicated adjacent tags, and
real anchors overwrite the deep-copied source. Map-member and frontier validation share manifest,
real-anchor, deduplication, and per-record locking plumbing, so the discriminated union stays a
single create path with no duplicate record-emission code.

The focused suite passed 18 tests, the full suite passed 461 tests, Ruff passed, and MyPy passed.

### 2026-07-17: Implemented Task 5 dossier HTTP APIs

Exposed the five Focus routes from the design spec on the curation server:
`GET /api/foci`, `POST /api/foci`, `GET /api/foci/<id>`, `PATCH /api/foci/<id>`, and
`POST /api/foci/<id>/archive`. Added a `do_PATCH`/`_do_PATCH` pair mirroring the existing
`do_POST`/`_do_POST` JSON error boundary so the new verb reuses `_send_json_error`,
`_json_response`, and the `NoActiveLegError` plumbing instead of inventing a parallel
dispatch. Each handler builds `Scope(expedition, leg)` from explicit query/body params, never
from `_active_selection`, so a request that targets one scope cannot silently resolve to the
active leg (the brief's "scope mismatch even when the mismatched pair equals the global active
selection" rule). Per-request `FocusStore(config.STATE_DIR, Path(REAL_DIR))` instantiation keeps
test-time `config.STATE_DIR` monkeypatches working without a captured-at-import-time value.
The scoped manifest is loaded directly from `config.leg_dir(expedition, leg)/scored_manifest.json`
with a missing file → empty list, so create against a leg with no scored images yet still works
(relying on `FocusStore.create`'s own `FocusValidationError` for member-tag resolution).

For `coverage_frontier` create, the server recomputes Coverage via
`clawmarks.build.coverage_map.compute_data(str(leg_dir))` and passes `data["cells"]` as the
authoritative `coverage_cells`. The client cannot smuggle a synthetic count/frontier/adjacency
claim past the recompute; an unmatched score_range returns 400. The `coverage_hint` inside
`source` (row/column/binning_version) is opaque display data and still passes through from the
client, matching Task 4's existing behavior. Error mapping is `FocusValidationError` → 400,
`FocusNotFound` → 404, `FocusConflict` → 409 with `current`, `FocusIntegrityError` → 500 with
no disk mutation (FocusStore never touches disk on a read-time integrity failure, so catching
and reporting satisfies the "never delete the corrupt file" rule). Create returns 201 with the
full record; GET, PATCH, and archive return 200.

Added `tests/test_curation_server_focus_routes.py` (20 tests): the brief's two exact snippets
(round-trip and scope-mismatch), list status filtering, invalid status 400, stale PATCH and
archive both → 409 with the current record, malformed JSON on POST and PATCH → 400, missing
query scope on list, single, PATCH, and archive → 400, unknown member tag → 400, unknown real
anchor → 400, unknown focus id → 404, PATCH missing `changes` or `expected_revision` → 400, a
`coverage_frontier` create whose score_range actually matches a recomputed frontier cell → 201,
a `coverage_frontier` create with a synthetic range that does not match any cell → 400 (proves
the server, not the client, decides), and `cs._active_selection` unchanged across all five
verbs. The full suite passed 481 tests, Ruff passed, MyPy passed, and `git diff --check` passed.
Committed as `feat(focus): expose dossier APIs`.

### 2026-07-17: Final whole-branch review of feat/focus-persistence, approved

Reviewed the full branch (8 commits, `5d3bdfe..12944e7`) against the Focus Persistence plan's
global constraints: storage location, no-delete-before-replace, fsync-then-replace-then-fsync-
parent on every write, one-level-at-a-time durable directory creation, `focus_`-prefixed UUID
record IDs, `fcntl.flock` plus revision checks on every mutation, and untouched corrupt records.
Every constraint holds. The reentrant cross-process lock test in `test_durable_records.py` uses
real multiprocessing with opposite-order lock acquisition, not a same-process simulation, and a
no-op flock mutation fails it, so the concurrency guarantee is genuinely exercised. No code path
in `atomic_io.py`, `durable_records.py`, or `focus_store.py` deletes a file before its
replacement succeeds.

Triaged the ten non-blocking findings accumulated across Tasks 3-5. Nine closed as either
working-as-designed or cosmetic, most notably the canonical-JSON tension: `atomic_json_write`
writes records with `indent=1`, not the plan's canonical `sort_keys`/compact form, but tracing
the actual call sites showed canonical-JSON encoding is only used for content-addressed digests
(`sha256_json`, not yet called by any Focus write path), not for the on-disk record file itself.
The plan's canonical-JSON line defines what canonical JSON is, not that every write must use it.

One finding carries forward as a real, non-blocking follow-up: `FocusStore.list()`
(`focus_store.py:80-85`) aborts entirely on the first corrupt record found in a scope, so one
bad file 500s the whole list endpoint for that scope rather than skipping it and returning the
rest. Worth fixing in a follow-up (log the corrupt file, return the remaining valid records),
not worth blocking this merge over.

Fresh verification run (not just re-trusting per-task reports): full suite 481 passed, Ruff
clean, MyPy clean across 48 source files, `git diff --check` clean across the whole branch.
Status: Approved. Proceeding to `superpowers:finishing-a-development-branch`.

### 2026-07-17: Sulfur Proof shared shell migration complete, PR opened

`feat/sulfur-proof-shared-shell` applies the approved Sulfur Proof typography, tokens, tactile
controls, shared context header, and mobile touch-target rules to every live curation page, per
`docs/superpowers/plans/2026-07-16-sulfur-proof-shared-shell.md`. Tasks 1-5 (bundled fonts and
package assets, the shared foundation and header, and the migration of all evidence, curation,
action, status, and error pages) ran task-by-task via `subagent-driven-development`, each
implemented and reviewed independently through taskferry (`opencode-go/minimax-m3`); several
implementer sessions crashed mid-task (`no_output_timeout`, bare `SIGTERM`) without reaching their
own commit, and in each case the controller inspected the resulting diff directly, confirmed it
materially complete, ran the full verification gate itself, and committed rather than retrying
past the two-consecutive-crash threshold. Full suite grew from 430 to 494 passing tests across the
five tasks with Ruff/MyPy clean throughout.

Task 6 (live Playwright verification, which cannot be taskferry-dispatched and must run directly
against a server the controller starts) exercised Explore, Map, Coverage, Compare, Cockpit, Runs,
the shared status page, and a 404 empty state at 1440px desktop and 390px mobile against a
disposable copy of the real `demo_expedition/cockpit` leg (backed up and file-count-verified, 65/65
files, before use). Zero console errors, zero horizontal overflow on any page, bundled fonts with
no runtime font request, correct focus-trap and Escape-to-close-with-focus-restoration on the
context-switch dialog, and `prefers-reduced-motion: reduce` present and universal. The audit found
one real gap: at 390px, the header's `.wordmark` and `.session-status` links fell under the 44px
touch-target minimum, because `MOBILE_BASE_CSS`'s min-height rule only covers form controls and
explicitly excludes plain anchors (`a:not(.navlink):not(.nav-activeleg)`). Fixed with a two-line
addition to the existing mobile media query in `shared_ui.py`; re-verified zero small targets
afterward. Final gate: `uv run pytest -q` 494 passed, Ruff and MyPy clean, `git diff --check`
clean. Opened as PR #51 against `main`.

### 2026-07-18: missing `.primary-action` component found and fixed site-wide; root swapped to Explore

Jeremy reported the live site "looks so bad" and asked where the Sulfur Proof theme went. Root
cause: `curation_server.py` never imported `BTN_CSS`, so three status-page renderers still used a
dead `btn btn--primary` class, rendering flat and unstyled. Fixing that exposed a deeper gap: no
button anywhere in the app could ever render the spec's "Primary actions" treatment (black fill,
sulfur underline), because `CONTROL_CSS` only ever got the spec's Dimensional Grammar section, not
its separate Controls section. See the gotcha log entry above for the traced root cause (a
keyword-scoped grep over the design spec).

Added `.primary-action` to `CONTROL_CSS` in `shared_ui.py` (TDD: two failing tests first, then the
CSS) and applied it to all 5 genuine primary actions site-wide: the three leg-picker buttons and
`createExpBtn`/`createLegBtn` in `curation_server.py`'s status-page renderers, `genBtn` in
`seed_browser.py`, and `launchBtn` in `runs_page.py`. Verified live via Playwright against the
decided `sulfur-proof-v2` Lavish artifact on `/`, `/runs.html`, and `/seeds.html`: correct black
fill, sulfur underline, and press-state on every primary button, zero console errors. Full suite
547 passed, Ruff clean.

Also made the homepage change from the `explore-root-review` Lavish artifact's route contract:
`/` and `/explore.html` now both render the Explore workbench (`explore_hub.render_html`), and
`/status.html` alone keeps the expedition/leg picker (previously `/` served the picker and
`/explore.html` was a separate route). Updated the test suite for the new contract
(`test_curation_server_expedition_routes.py`, `test_curation_server_startup.py`); one test
originally asserted byte-identical bodies between `/` and `/explore.html`, which is flaky since
`shared_ui.info_btn()` increments a process-wide tooltip-id counter on every render call, so two
separate render calls are never byte-identical even when they're the same template. Fixed to
compare the stable `<title>` instead. Verified live via Playwright: `/` renders the Explore hub,
`/status.html` renders the picker, zero console errors. Full suite 547 passed, Ruff clean.

Archived the lab log's 2026-07-08 through 2026-07-15 entries (68 of 85) to
`notes/lab_notebook_archive.md` to keep this file scannable, per Jeremy's request. Straight
relocation, not a summary: verified by character-count equality between the original lab-log text
and the archived+kept split before writing either file, and confirmed via `git diff` that sections
1-5 (everything before "## 6. Lab log") were untouched.

### 2026-07-18: expedition/leg creation folded into the shared context dialog; status.html shrunk

Jeremy reviewed the homepage/primary-action fix via Lavish and suggested the leg picker on
`/status.html` could move into the shared header's `contextDialog`, since that dialog already
handled switching but not creating. Built a mockup in the same Lavish artifact and got explicit
approval ("Approved: fold expedition/leg creation into the shared contextDialog, shrink
/status.html to a pure status/health page").

Implemented with TDD: added tests in `test_shared_ui.py` for new markup ids
(`contextNewExpToggle`/`contextNewExpForm`/`contextNewExpName`,
`contextNewLegToggle`/`contextNewLegForm`/`contextNewLegExpedition`/`contextNewLegName`), new
`.context-create*` CSS, and the JS wiring, before writing any implementation. Added two
collapsible forms to `nav_bar_html()`'s `context_dialog` in `shared_ui.py`, POSTing to the
existing `/api/expeditions` and `/api/legs` endpoints and refreshing the switcher list in place on
success (no page reload). Removed the now-redundant leg-btn picker rows and create-expedition/
create-leg forms (and their JS) from all three `/status.html` render branches in
`curation_server.py` (`_status_page_no_selection_body`, `_status_page_selected_empty_body`,
`_status_page_data_integrity_error_body`), replacing them with a one-line pointer to the header's
"choose context" button.

Two bugs caught during the process, both fixed before calling it done:

- A test (`test_render_html_keeps_choices_outside_the_shared_nav` in `test_compare_page.py`) split
  the page HTML on the first `</header>` to check no real `<button>` elements exist in the page
  body. The context dialog's own inner header (`<header class="context-dialog-header">`, holding
  its close button) matched that split before the outer nav header did, so the new create-form
  buttons landed on the wrong side of the split and failed the test. Fixed by changing that inner
  element from `<header>` to a plain `<div>` (the CSS already targeted it by class, not tag), so
  there is exactly one `<header>` in the page.
- Live Playwright verification against the real running server showed both create forms rendered
  open by default, even though they carry the `hidden` attribute. Root cause: the `.context-
  create-form { display:flex; }` class rule has higher CSS specificity than the browser's default
  `[hidden] { display:none; }` user-agent style, so the class rule won. Fixed with an explicit
  `.context-create-form[hidden] { display:none; }` rule ahead of it. This is a reminder that
  `hidden` is not automatically respected once any class rule sets `display` on the same element;
  the fix generalizes to any future toggle-by-`hidden` element that also gets a `display` value
  from a class.

Verified live: opened the dialog on the running server, created a real expedition
(`lavish_smoke_test`) and leg (`smoke_leg`) end to end through the new forms, confirmed both
appeared in the switcher immediately with no page reload, zero browser console errors, then
deleted the smoke-test scaffold from `$XDG_STATE_HOME/clawmarks/expeditions/` (config only, no
generation output, so no backup needed first). Full suite 550 passed, Ruff clean. Updated the
Lavish artifact with a live screenshot of the built dialog and replied via `lavish-axi poll
--agent-reply`.

### 2026-07-17: Research workspace navigation Task 2 complete

Task 2 of the Research Workspace Navigation plan made live page data scope-explicit. Manifest
lookups and every live-cache key now include the requested expedition and leg, so two browser tabs
cannot reuse one another's computed page data. Focus-scoped GET routes resolve one immutable
`WorkspaceContext` per request and leave the global active-leg selection and `active_leg.json`
untouched. The new `/generated/<tag>` route and the scoped `/thumbs/<tag>.jpg` path resolve the tag
inside the requested leg's manifest and reject manifest file paths outside that leg directory.

TDD verification added real two-leg HTTP fixtures, full-image reads, thumbnail generation, and
outside-leg path rejection. The focused suite passed 13 tests, the full suite passed 554 tests,
Ruff passed, and MyPy passed across 49 source files. The full suite still reports existing
third-party scikit-learn and UMAP warnings.

The follow-up review found that the first implementation scoped data and direct image endpoints but
left renderer-generated relative thumbnail URLs on the global active-leg directory. The fix passes
explicit `WorkspaceContext` into scan, map, redundancy, coverage, archive, and preference-rank
renderers, which now emit scope-bearing generated-image URLs while preserving bare legacy URLs. The
thumbnail route also preserves blank query keys during parsing, so blank expedition, leg, and
Focus values now receive the same HTTP 400 validation as other malformed workspace queries. The
focused suite passed 16 tests and the full suite passed 557 tests; Ruff and MyPy remained clean.

### 2026-07-18: Focus navigation review fixes

A whole-branch review found six navigation regressions, and this pass fixed each one. The legacy
thumbnail fallback now validates the decoded tag before constructing a cache path, so traversal
requests cannot write outside the active leg. Shared navigation links preserve expedition and leg
scope even when no Focus is selected. The Seeds and Runs pages now embed their rendered scope and
forward it through every leg-scoped API call; scoped seed generation also writes to the requested
leg rather than reusing the global active leg. Trial records retain `focus_id`, and the Cockpit
queue displays that provenance. Favorite and unfavorite mutations resolve a supplied Focus through
the FocusStore and reject references that are missing or belong to another leg before writing.

Regression coverage includes the traversal rejection, no-Focus navigation URL, scoped Seeds and
Runs renderer calls, requested-leg seed generation, Focus trial provenance, queue display, and both
favorite mutation endpoints. The final verification passed 602 tests, Ruff across `src tests`,
MyPy across `src`, and `git diff --check`. Existing scikit-learn optimization warnings remain.

### 2026-07-18: Follow-up review closed the remaining workspace authorization gaps

The follow-up review of commit `4671456` found two regression tests that could pass through
unrelated 404 paths and one paid-generation authorization gap. The legacy `/thumbs/` regression
now adds an `outside` manifest tag to the active fixture, matching the tag that the old
`os.path.basename()` extraction would have produced from `/thumbs/../../outside.jpg`, and checks
the actual escaped path outside the active leg. The `/generated/` regression uses a manifest
lookup sentinel so a traversal-shaped URL tag must be rejected before manifest contents are read.

The counterfactual POST handler now validates an optional `focus_id` against the resolved
expedition and leg before checking the RunPod balance or submitting a generation. Its regression
creates a Focus in `round2`, requests a counterfactual for `round1`, and confirms both mocked
RunPod calls remain untouched while the endpoint returns HTTP 400.

The focused tests passed 20 tests. Final verification passed 604 tests, Ruff across `src tests`,
MyPy across `src`, and `git diff --check`. The full suite emitted the existing scikit-learn
`OptimizeWarning` messages about the `iprint` solver option.

### 2026-07-18: Persona audit led to an image-first information architecture specification

Three independent usability walkthroughs, framed as a grandmother, an impaired casual visitor,
and a child, converged on the same structural failures: the Explore homepage showed no artwork;
the Orient/Scout/Explain/Act/Learn method looked like unexplained navigation; the flat tool index
led with research jargon; `faith` read as religious language; and users could not distinguish inert
help from controls that submit paid work. All three understood the shared page-jump select, and the
image gallery and head-to-head comparison were the clearest destinations.

An independent UX review concluded that labels alone cannot repair this shape. Jeremy approved its
full direction: make the existing gallery the homepage, group destinations by user task, demote the
five-stage loop to a `How a search round works` explainer, defer formal vocabulary behind plain
labels, replace `?` tips with accessible `i` controls, and mark every billable action explicitly.
The approved specification is
`docs/superpowers/specs/2026-07-18-image-first-information-architecture-design.md`; the TDD plan is
`docs/superpowers/plans/2026-07-18-image-first-information-architecture.md`. The spec reserves deep
aubergine `#5B3A63` as a restrained cost signal that complements sulfur while remaining distinct
from selection and error colors. No implementation or generation operation occurred in this step.

### 2026-07-18: Task 5 regression gate and live accessibility verification

Executed Task 5 of the image-first IA plan. Before the gate, closed two Task 4 review findings with
TDD: the Cockpit `Review and run` button (cockpit.py:601) now carries `primary-action billable-action`
(was bare `billable-action`), matching the pattern used by `seed_browser.py`'s Generate and
`runs_page.py`'s Back up and launch. Removed the redundant `.billable-action::before` pseudo-element
from `shared_ui.py`'s CONTROL_CSS while keeping the aubergine `border-left:3px solid var(--cost)`.
Both changes passed their covering tests (RED then GREEN evidence recorded: the first run of the
updated assertions failed because `class="primary-action billable-action"` was not yet on the Cockpit
button and `::before` was still present; after the code changes both passed). All 105 related tests
passed in the second run.

Ran the static check gate: `uv run ruff check src tests` passed, `uv run mypy src` passed, and
`uv run pytest -q` passed 639 tests (6396 sklearn warnings, no failures).

Started `curation_server.py` bound to `0.0.0.0:8420` with `CLAWMARKS_HOST=0.0.0.0` against the
existing read-only leg `trent_v3_epoch4/freeform1` (50 scored images, all present on disk). No
generation state was created, deleted, overwritten, or transformed. No billable action was clicked.

Live Playwright MCP verification at 1440x900 desktop covered `/`, `/scan.html`, `/explore.html`,
`/coverage.html`, `/seeds.html`, and `/runs.html`, each with workspace-scope query parameters:

- All routes loaded without real console errors (only benign `Transition was skipped` messages).
- Thumbnails appeared above the fold: 64 of 100 image elements visible at 1440x900.
- Task groups and plain labels present throughout.
- The research-loop disclosure (`How a search round works`) opened on click and showed Orient/Scout/
  Explain/Act/Learn content.
- Information buttons opened glossary popovers with definition text; Escape closed the popover and
  focus returned to the info button (`aria-label="More information about ..."`).
- All paid actions visibly carry `Spends money` badges (seeds Generate, runs Back up and launch).
- No horizontal page overflow on any desktop route.

At 390x844 mobile, checked `/`, `/scan.html`, `/seeds.html`, and `/runs.html`:

- Zero horizontal overflow on all routes (scrollWidth == clientWidth).
- All visible info buttons measure exactly 44x44px (meeting the 44px touch-target minimum).
- All visible `select` elements measure at least 44px tall.
- Escape closes glossary popovers and focus restores to the triggering info button.
- `Spends money` badge remains visible at 200% zoom on seeds and runs pages.
- The `.billable-action::before` pseudo-element confirmed absent from live server CSS.

No defects fixed beyond the two Task 4 review findings. No new issues found during the gate.

Stopped the server after verification (`kill 857060`). Post-task `pgrep -af "clawmarks serve|clawmarks.curation_server" || true` confirmed no server process remained running. No generation state was created, deleted, overwritten, or transformed during the entire gate.

Note: I was unable to find the `.claude/skills/run/SKILL.md` file referenced in the task brief. The
project's `cli.py` already has a `clawmarks serve` command that calls `curation_server.main([])`,
using `CLAWMARKS_HOST` env var for binding. This workaround was used without issue.
