# tests/test_scoring.py
from clawmarks.search.scoring import bin_edges, bin_of, novelty_from_similarity


def test_bin_edges_splits_sorted_values_into_n_groups():
    vals = sorted([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    edges = bin_edges(vals, 4)
    assert len(edges) == 3
    assert edges == sorted(edges)


def test_bin_of_returns_last_bin_for_max_value():
    edges = [0.25, 0.5, 0.75]
    assert bin_of(0.9, edges) == 3


def test_bin_of_returns_first_bin_for_min_value():
    edges = [0.25, 0.5, 0.75]
    assert bin_of(0.1, edges) == 0


def test_novelty_from_similarity_inverts_similarity():
    assert novelty_from_similarity(0.3) == 1 - 0.3
    assert novelty_from_similarity(1.0) == 0.0
