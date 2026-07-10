# tests/test_rating_sampler.py
import random

from clawmarks.search import rating_sampler


def _manifest(n):
    return [{"tag": f"t{i}", "centroid_sim": i / n, "novelty": 1 - i / n} for i in range(n)]


def test_bin_manifest_splits_into_n_bins_by_bin_count():
    manifest = _manifest(16)
    grid = rating_sampler.bin_manifest(manifest)
    assert len(grid) <= rating_sampler.N_BINS * rating_sampler.N_BINS
    assert sum(len(v) for v in grid.values()) == 16


def test_eligible_grid_excludes_reviewed_tags():
    manifest = _manifest(8)
    reviewed = {"t0", "t1", "t2"}
    grid = rating_sampler.eligible_grid(manifest, reviewed)
    all_tags = {m["tag"] for items in grid.values() for m in items}
    assert reviewed.isdisjoint(all_tags)
    assert len(all_tags) == 5


def test_pick_next_returns_none_when_everything_reviewed():
    manifest = _manifest(4)
    reviewed = {m["tag"] for m in manifest}
    assert rating_sampler.pick_next(manifest, reviewed) is None


def test_pick_next_only_returns_eligible_items():
    manifest = _manifest(20)
    reviewed = {f"t{i}" for i in range(15)}
    rng = random.Random(0)
    for _ in range(30):
        item = rating_sampler.pick_next(manifest, reviewed, rng=rng)
        assert item is not None
        assert item["tag"] not in reviewed


def test_pick_next_can_return_from_a_sparsely_populated_bin():
    # one bin has a single eligible item, another has many; over enough draws the sparse
    # bin's item should still turn up, proving bins are chosen uniformly, not by size.
    manifest = _manifest(100)
    reviewed = {m["tag"] for m in manifest[1:50]}  # leaves t0 alone in its low bin
    rng = random.Random(1)
    seen = set()
    for _ in range(200):
        item = rating_sampler.pick_next(manifest, reviewed, rng=rng)
        seen.add(item["tag"])
    assert "t0" in seen
