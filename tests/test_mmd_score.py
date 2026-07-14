import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "notes"))

from mmd_score import mmd2_unbiased, monte_carlo_p_value


def test_monte_carlo_p_value_has_finite_sample_floor():
    shuffled = np.zeros(10)

    assert monte_carlo_p_value(1.0, shuffled) == 1 / 11


def test_monte_carlo_p_value_counts_at_least_as_extreme_statistics():
    shuffled = np.array([0.1, 0.4, 0.8, 0.9])

    assert monte_carlo_p_value(0.8, shuffled) == 3 / 5


def test_monte_carlo_p_value_rejects_non_finite_observed():
    shuffled = np.array([0.1, 0.4, 0.8])

    with pytest.raises(ValueError, match="observed statistic must be finite"):
        monte_carlo_p_value(float("nan"), shuffled)


def test_monte_carlo_p_value_rejects_non_finite_shuffled():
    shuffled = np.array([0.1, float("inf"), 0.8])

    with pytest.raises(ValueError, match="shuffled statistics must all be finite"):
        monte_carlo_p_value(0.5, shuffled)


def test_mmd2_unbiased_rejects_group_smaller_than_two():
    K = torch.eye(4)
    idx_a = torch.tensor([0])
    idx_b = torch.tensor([1, 2, 3])

    with pytest.raises(ValueError, match="at least 2 items per group"):
        mmd2_unbiased(K, idx_a, idx_b)


def test_mmd2_unbiased_accepts_two_item_groups():
    K = torch.eye(4)
    idx_a = torch.tensor([0, 1])
    idx_b = torch.tensor([2, 3])

    mmd2, term_aa, term_bb, term_ab = mmd2_unbiased(K, idx_a, idx_b)

    assert np.isfinite(mmd2)
