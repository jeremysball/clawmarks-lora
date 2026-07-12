"""
Pair sampler for the compare UI (compare.html / GET /api/compare/next): picks two images to
show side by side next. Replaces search/rating_sampler.py's role for head-to-head comparisons.

Below MIN_COMPARISONS, picks a stratified-random pair: two bins chosen independently at random
from the existing faithfulness x novelty grid (also used by build/elite_archive.py), one random
image from each, so an early comparison session doesn't over-sample whichever region happens to
dominate the pool. At or above the floor, switches to model-uncertainty-guided selection: a
random candidate set of images is scored by the current model, and the two whose scores are
closest together are returned, since that pair is the model's best approximation of "least sure
which one wins" without enumerating every possible pair across a pool of thousands of images.
See docs/superpowers/specs/2026-07-11-head-to-head-preference-design.md.
"""
import random

from clawmarks.search.scoring import bin_edges, bin_of

N_BINS = 4  # matches build/elite_archive.py's grid
MIN_COMPARISONS = 50
RETRAIN_EVERY = 10
CANDIDATE_POOL_SIZE = 200


def bin_manifest(manifest):
    faith_vals = sorted(m["centroid_sim"] for m in manifest)
    novelty_vals = sorted(m["novelty"] for m in manifest)
    faith_edges = bin_edges(faith_vals, N_BINS)
    novelty_edges = bin_edges(novelty_vals, N_BINS)
    grid = {}
    for m in manifest:
        fb = bin_of(m["centroid_sim"], faith_edges)
        nb = bin_of(m["novelty"], novelty_edges)
        grid.setdefault((fb, nb), []).append(m)
    return grid


def stratified_random_pair(manifest, seen=None, rng=random):
    """Returns two distinct manifest items, or None if the manifest has fewer than 2 images.

    `seen` maps a tag to how many past comparisons it appears in. Picking a bin uniformly and
    then an image within it (the original behavior) over-samples images in sparse bins: an image
    alone in its bin is drawn with probability 1/n_bins, one of twenty images in a dense bin only
    1/n_bins/20, so a lone-bin image reappears far more often than a person perceives as random.
    To keep the grid spread without that skew, restrict each draw to the least-covered frontier:
    the bins whose least-shown image has the lowest seen-count, then the least-shown image inside
    the chosen bin (random tie-break). An image thus never reappears until every other bin's
    least-shown image has been shown as often, which spreads coverage evenly across the archive."""
    if len(manifest) < 2:
        return None
    seen = seen or {}
    grid = bin_manifest(manifest)
    nonempty = [items for items in grid.values() if items]
    if not nonempty:
        return None

    def seen_of(item):
        return seen.get(item["tag"], 0)

    def pick(exclude_tag=None):
        pools = [[it for it in items if it["tag"] != exclude_tag] for items in nonempty]
        pools = [p for p in pools if p]
        if not pools:
            return None
        frontier_cov = min(min(seen_of(it) for it in p) for p in pools)
        frontier = [p for p in pools if min(seen_of(it) for it in p) == frontier_cov]
        chosen = rng.choice(frontier)
        least = min(seen_of(it) for it in chosen)
        return rng.choice([it for it in chosen if seen_of(it) == least])

    item_a = pick()
    if item_a is None:
        return None
    item_b = pick(exclude_tag=item_a["tag"])
    if item_b is None:
        return None
    return (item_a, item_b)


def most_uncertain_pair(manifest, model, score_fn, embeddings_for, rng=random):
    """Returns the two manifest items whose model scores are closest together, out of a random
    candidate set of up to CANDIDATE_POOL_SIZE images. `score_fn(model, embeddings) -> sequence`
    and `embeddings_for(items) -> sequence` let callers plug in
    preference_pairwise_model.score and an embedding lookup without this module importing the
    embedding cache directly. Returns None if fewer than 2 candidates are available."""
    if len(manifest) < 2:
        return None
    candidates = rng.sample(manifest, min(CANDIDATE_POOL_SIZE, len(manifest)))
    scores = score_fn(model, embeddings_for(candidates))
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1])
    best_gap, best_pair = None, None
    for i in range(len(ranked) - 1):
        gap = abs(ranked[i][1] - ranked[i + 1][1])
        if best_gap is None or gap < best_gap:
            best_gap, best_pair = gap, (ranked[i][0], ranked[i + 1][0])
    return best_pair


def pick_next_pair(manifest, n_comparisons, model=None, score_fn=None, embeddings_for=None,
                   seen=None, rng=random):
    """Top-level entry point used by curation_server.py. Below MIN_COMPARISONS, or when no model
    is available yet, falls back to stratified_random_pair (coverage-balanced via `seen`, a
    tag->appearance-count map from the comparison history). At/above the floor with a model
    available, uses most_uncertain_pair."""
    if n_comparisons < MIN_COMPARISONS or model is None:
        return stratified_random_pair(manifest, seen=seen, rng=rng)
    return most_uncertain_pair(manifest, model, score_fn, embeddings_for, rng=rng)
