"""
Stratified sampler for the ratings UI (rate.html / GET /api/rate/next): picks an unreviewed
image to show next, spread across the faithfulness x novelty grid build/elite_archive.py
already uses, so an early rating session doesn't over-sample whichever region happens to
dominate the pool. See docs/superpowers/specs/2026-07-09-preference-classifier-design.md,
Component 2.
"""
import random

from clawmarks.search.scoring import bin_edges, bin_of

N_BINS = 4  # matches build/elite_archive.py's grid


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


def eligible_grid(manifest, reviewed_tags):
    grid = bin_manifest(manifest)
    return {key: [m for m in items if m["tag"] not in reviewed_tags]
            for key, items in grid.items()}


def pick_next(manifest, reviewed_tags, rng=random):
    """Returns a random manifest item from a randomly chosen non-empty bin, or None if every
    image is already reviewed. Choosing the bin uniformly at random (not weighted by how many
    eligible images remain in it) is what makes this stratified rather than plain random: a bin
    with 5 eligible images is exactly as likely to be sampled from as a bin with 500."""
    grid = eligible_grid(manifest, reviewed_tags)
    nonempty = [items for items in grid.values() if items]
    if not nonempty:
        return None
    bin_items = rng.choice(nonempty)
    return rng.choice(bin_items)
