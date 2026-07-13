import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "notes"))

from probe_power import (
    CANONICAL_SEEDS,
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
