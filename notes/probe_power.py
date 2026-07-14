"""Power analysis for the paired-seed probe design.

The independent unit is one canonical training seed.  Each candidate direction is
compared with its control at the same seed, producing one paired delta per seed.
Prompt rows and mirrored deltas are measurements within a seed, not additional
independent observations.

For the small sample sizes used here, the one-sided sign-flip test enumerates every
possible sign pattern.  This makes the attainable p-value floor explicit and avoids
the duplicate sign patterns that a Monte Carlo sampler can produce.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np


CANONICAL_SEEDS = (
    20260709,
    8675309,
    271828,
    141421,
    314159,
    161803,
    57721,
    30103,
)
N_SIMULATIONS = 10_000
ALPHA = 0.05
MIN_EFFECT = 0.05
EFFECTS = (0.05, 0.08)
CHECKPOINT_MEAN_SD = 0.0354
# The available calibration batch was unpaired.  Use sqrt(2) times its
# checkpoint-mean SD as an unverified planning proxy until paired round-one
# data can replace it; the test itself does not change.
DELTA_SD = float(np.sqrt(2.0) * CHECKPOINT_MEAN_SD)


@lru_cache(maxsize=None)
def sign_flip_matrix(n):
    """Return all 2**n sign patterns in deterministic order."""
    if n < 1:
        raise ValueError("n must be positive")
    flips = np.asarray(tuple(product((-1.0, 1.0), repeat=n)))
    flips.setflags(write=False)
    return flips


def attainable_one_sided_floor(n):
    """Return the smallest exact one-sided sign-flip p-value at sample size n."""
    return 1.0 / len(sign_flip_matrix(n))


def exact_sign_flip_pvalue(deltas):
    """Compute the exact one-sided p-value for positive mean paired deltas."""
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 1 or not len(deltas):
        raise ValueError("deltas must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("deltas must contain only finite values")
    observed = deltas.mean()
    shuffled_means = sign_flip_matrix(len(deltas)) @ deltas / len(deltas)
    return float(np.count_nonzero(shuffled_means >= observed) / len(shuffled_means))


@dataclass(frozen=True)
class SimulationResult:
    n: int
    effect: float
    simulations: int
    rejections: int
    rate: float
    standard_error: float


def simulate_rejection_rate(
    n,
    effect,
    *,
    simulations=N_SIMULATIONS,
    delta_sd=DELTA_SD,
    alpha=ALPHA,
    min_effect=None,
    seed=20260713,
):
    """Estimate rejection rate for normal paired deltas at a fixed effect.

    With `min_effect=None`, this reports the sign-flip test's own calibration
    (does p<=alpha fire at rate alpha under the null, and does it rise with
    effect). That is a check on the test, not on the round-1 decision rule.

    The actual round-1 gate (lab_notebook.md, "Selection rule") requires both
    p<=alpha and an observed mean delta >= 0.05 DINOv2 cosine. Pass
    `min_effect=MIN_EFFECT` to report power for that full gate instead of the
    test alone; the two can differ substantially, since requiring a minimum
    observed effect makes the gate strictly more conservative than the
    p-value test by itself.
    """
    if n < 1 or simulations < 1 or delta_sd <= 0:
        raise ValueError("n, simulations, and delta_sd must be positive")
    rng = np.random.default_rng(seed)
    flips = sign_flip_matrix(n)
    rejections = 0
    for deltas in rng.normal(effect, delta_sd, size=(simulations, n)):
        observed = deltas.mean()
        shuffled_means = flips @ deltas / n
        p_value = np.count_nonzero(shuffled_means >= observed) / len(flips)
        passes = p_value <= alpha
        if min_effect is not None:
            passes = passes and observed >= min_effect
        rejections += passes
    rate = rejections / simulations
    standard_error = float(np.sqrt(rate * (1.0 - rate) / simulations))
    return SimulationResult(n, effect, simulations, rejections, rate, standard_error)


def _print_result(label, result):
    print(
        f"{label:>6} n={result.n} effect={result.effect:.2f} "
        f"rejections={result.rejections}/{result.simulations} "
        f"rate={result.rate:.4f} +/- {result.standard_error:.4f}"
    )


def main():
    print("Paired unit: one canonical training seed; n=8 planned seeds")
    print(f"Canonical seeds: {', '.join(str(seed) for seed in CANONICAL_SEEDS)}")
    print(f"Assumed paired-delta SD: {DELTA_SD:.6f} (sqrt(2) * {CHECKPOINT_MEAN_SD:.4f})")
    print(f"One-sided alpha: {ALPHA:.2f}; simulations per row: {N_SIMULATIONS}")
    print("Attainable one-sided exact sign-flip p-value floors:")
    for n in (3, 4, 6, 8):
        print(f"  n={n}: {attainable_one_sided_floor(n):.6f}")

    print("Sign-flip test calibration, null rejection rate (effect=0.00, p<=alpha only):")
    for n in (3, 4, 6, 8):
        _print_result(
            "null",
            simulate_rejection_rate(n, 0.0, seed=20260713 + n),
        )

    print("Sign-flip test calibration, positive-control rejection rate (p<=alpha only):")
    for effect in EFFECTS:
        for n in (3, 4, 6, 8):
            _print_result(
                "power",
                simulate_rejection_rate(n, effect, seed=20260713 + n + int(effect * 1000)),
            )

    print(
        f"\nRound-1 gate power (p<=alpha AND observed mean delta >= {MIN_EFFECT:.2f}), "
        "the actual decision rule from the lab notebook's Selection rule step:"
    )
    for effect in EFFECTS:
        for n in (3, 4, 6, 8):
            _print_result(
                "gate",
                simulate_rejection_rate(
                    n,
                    effect,
                    min_effect=MIN_EFFECT,
                    seed=20260713 + n + int(effect * 1000),
                ),
            )


if __name__ == "__main__":
    main()
