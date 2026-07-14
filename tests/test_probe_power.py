import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "notes"))

from probe_power import (
    CANONICAL_SEEDS,
    MIN_EFFECT,
    attainable_one_sided_floor,
    exact_sign_flip_pvalue,
    sign_flip_matrix,
    simulate_rejection_rate,
)


def test_canonical_seeds_are_the_eight_planned_independent_units():
    assert len(CANONICAL_SEEDS) == 8
    assert len(set(CANONICAL_SEEDS)) == 8


def test_sign_flip_enumeration_has_no_duplicate_patterns():
    patterns = sign_flip_matrix(4)

    assert patterns.shape == (16, 4)
    assert len(np.unique(patterns, axis=0)) == 16


def test_exact_sign_flip_pvalue_reaches_one_sided_floor():
    deltas = np.ones(8)

    assert attainable_one_sided_floor(8) == 1 / 256
    assert exact_sign_flip_pvalue(deltas) == 1 / 256


def test_null_simulation_is_reproducible_and_positive_control_has_power():
    null = simulate_rejection_rate(8, 0.0, simulations=10_000, seed=123)
    positive = simulate_rejection_rate(8, 0.08, simulations=10_000, seed=456)

    assert null == simulate_rejection_rate(8, 0.0, simulations=10_000, seed=123)
    assert abs(null.rate - 0.05) <= 3 * null.standard_error + 0.001
    assert positive.rate > 0.8


def test_gate_power_is_lower_than_test_only_power_at_the_same_effect():
    # The round-1 decision rule requires p<=alpha AND observed >= MIN_EFFECT, not p<=alpha
    # alone. Requiring both can only be more conservative than requiring the p-value alone.
    test_only = simulate_rejection_rate(8, 0.05, simulations=10_000, seed=789)
    gate = simulate_rejection_rate(8, 0.05, simulations=10_000, seed=789, min_effect=MIN_EFFECT)

    assert gate.rate <= test_only.rate
    assert 0.3 < gate.rate < 0.7


def test_gate_power_rejects_nothing_below_the_effect_floor_even_with_alpha_met():
    # At n=8 the exact sign-flip floor is 1/256, well under alpha=0.05, so p<=alpha is easy
    # to satisfy on its own; the effect-size condition still has to do real work.
    gate = simulate_rejection_rate(8, 0.0, simulations=10_000, seed=321, min_effect=MIN_EFFECT)

    assert gate.rate < 0.01
