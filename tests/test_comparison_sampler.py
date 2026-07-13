import itertools
import random

from clawmarks.search import comparison_sampler as cs


def _manifest(n):
    return [{"tag": f"t{i}", "centroid_sim": i / n, "novelty": 1 - i / n} for i in range(n)]


def test_bin_manifest_splits_into_n_bins_by_bin_count():
    manifest = _manifest(16)
    grid = cs.bin_manifest(manifest)
    assert len(grid) <= cs.N_BINS * cs.N_BINS
    assert sum(len(v) for v in grid.values()) == 16


def test_stratified_random_pair_returns_two_distinct_items():
    manifest = _manifest(20)
    rng = random.Random(0)
    for _ in range(30):
        pair = cs.stratified_random_pair(manifest, rng=rng)
        assert pair is not None
        a, b = pair
        assert a["tag"] != b["tag"]


def test_stratified_random_pair_returns_none_with_fewer_than_two_images():
    assert cs.stratified_random_pair(_manifest(1)) is None
    assert cs.stratified_random_pair(_manifest(0)) is None


def test_most_uncertain_pair_picks_the_closest_scored_candidates():
    manifest = _manifest(10)
    scores_by_tag = {f"t{i}": float(i) for i in range(10)}
    # t4 and t5 are adjacent (gap 1.0); every other adjacent gap is also 1.0, so force a
    # tighter gap between two items to make the expected winner unambiguous.
    scores_by_tag["t4"] = 5.0
    scores_by_tag["t5"] = 5.05

    def score_fn(model, embeddings):
        return [scores_by_tag[tag] for tag in embeddings]

    def embeddings_for(items):
        return [it["tag"] for it in items]

    pair = cs.most_uncertain_pair(manifest, model=object(), score_fn=score_fn,
                                  embeddings_for=embeddings_for, rng=random.Random(0))
    assert pair is not None
    tags = {pair[0]["tag"], pair[1]["tag"]}
    assert tags == {"t4", "t5"}


def test_most_uncertain_pair_returns_none_with_fewer_than_two_images():
    assert cs.most_uncertain_pair(_manifest(1), object(), lambda m, e: [], lambda items: []) is None


def test_stratified_random_pair_never_returns_an_excluded_pair():
    manifest = _manifest(4)  # t0..t3: only 6 possible pairs
    exclude = {
        frozenset(("t0", "t1")), frozenset(("t0", "t2")), frozenset(("t0", "t3")),
        frozenset(("t1", "t2")), frozenset(("t1", "t3")),
    }
    rng = random.Random(0)
    for _ in range(20):
        pair = cs.stratified_random_pair(manifest, rng=rng, exclude=exclude)
        assert pair is not None
        assert frozenset((pair[0]["tag"], pair[1]["tag"])) == frozenset(("t2", "t3"))


def test_stratified_random_pair_returns_none_when_every_pair_is_excluded():
    manifest = _manifest(3)
    exclude = {frozenset(p) for p in itertools.combinations([m["tag"] for m in manifest], 2)}
    assert cs.stratified_random_pair(manifest, rng=random.Random(0), exclude=exclude) is None


def test_most_uncertain_pair_skips_excluded_gaps():
    manifest = _manifest(4)
    scores_by_tag = {"t0": 0.0, "t1": 1.0, "t2": 1.05, "t3": 5.0}

    def score_fn(model, embeddings):
        return [scores_by_tag[tag] for tag in embeddings]

    def embeddings_for(items):
        return [it["tag"] for it in items]

    exclude = {frozenset(("t1", "t2"))}
    pair = cs.most_uncertain_pair(manifest, model=object(), score_fn=score_fn,
                                  embeddings_for=embeddings_for, rng=random.Random(0), exclude=exclude)
    assert pair is not None
    assert frozenset((pair[0]["tag"], pair[1]["tag"])) != frozenset(("t1", "t2"))


def test_pick_next_pair_falls_back_to_stratified_when_most_uncertain_is_exhausted():
    manifest = _manifest(2)
    exclude = {frozenset(("t0", "t1"))}

    def score_fn(model, embeddings):
        return [0.0, 1.0]

    pair = cs.pick_next_pair(manifest, n_comparisons=60, model=object(), score_fn=score_fn,
                              embeddings_for=lambda items: [it["tag"] for it in items],
                              rng=random.Random(0), exclude=exclude)
    assert pair is None


def test_pick_next_pair_uses_stratified_below_min_comparisons():
    manifest = _manifest(20)
    calls = {"most_uncertain": 0}

    def score_fn(model, embeddings):
        calls["most_uncertain"] += 1
        return [0.0] * len(embeddings)

    pair = cs.pick_next_pair(manifest, n_comparisons=10, model=object(), score_fn=score_fn,
                              embeddings_for=lambda items: items, rng=random.Random(0))
    assert pair is not None
    assert calls["most_uncertain"] == 0


def test_pick_next_pair_uses_stratified_when_no_model_even_above_floor():
    manifest = _manifest(20)
    pair = cs.pick_next_pair(manifest, n_comparisons=60, model=None, rng=random.Random(0))
    assert pair is not None


def test_pick_next_pair_uses_most_uncertain_at_or_above_floor_with_a_model():
    manifest = _manifest(20)
    calls = {"n": 0}

    def score_fn(model, embeddings):
        calls["n"] += 1
        return list(range(len(embeddings)))

    pair = cs.pick_next_pair(manifest, n_comparisons=60, model=object(), score_fn=score_fn,
                              embeddings_for=lambda items: items, rng=random.Random(0))
    assert pair is not None
    assert calls["n"] == 1
