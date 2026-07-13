import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "notes"))

from mmd_score import monte_carlo_p_value


def test_monte_carlo_p_value_has_finite_sample_floor():
    shuffled = np.zeros(10)

    assert monte_carlo_p_value(1.0, shuffled) == 1 / 11


def test_monte_carlo_p_value_counts_at_least_as_extreme_statistics():
    shuffled = np.array([0.1, 0.4, 0.8, 0.9])

    assert monte_carlo_p_value(0.8, shuffled) == 3 / 5
