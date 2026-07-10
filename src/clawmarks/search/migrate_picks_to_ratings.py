"""One-time migration: user_picks.json entries become user_ratings.json entries with
label: "yes", so "pick as winner" can be retired without losing the existing picks. Safe to
rerun: any tag that already has a rating is left alone. Not wired into `clawmarks` as a
permanent CLI subcommand since it's a one-shot migration, not a recurring operation. See
docs/superpowers/specs/2026-07-09-preference-classifier-design.md, Component 2a.

Run with: python -m clawmarks.search.migrate_picks_to_ratings
"""
import json
import os

from clawmarks.config import USER_PICKS_FILE, USER_RATINGS_FILE


def merge_picks_into_ratings(picks, ratings):
    """Returns (updated_ratings, migrated_tags). Does not overwrite a tag that already has a
    rating, whatever its label, since a rating recorded through rate.html reflects a more
    deliberate, later judgment than an old pick."""
    updated = dict(ratings)
    migrated = []
    for tag, pick in picks.items():
        if tag in updated:
            continue
        updated[tag] = {"label": "yes", "rated_at": pick.get("picked_at")}
        migrated.append(tag)
    return updated, migrated


def main(argv=None):
    picks = {}
    if USER_PICKS_FILE.exists():
        with open(USER_PICKS_FILE) as f:
            picks = json.load(f)
    ratings = {}
    if USER_RATINGS_FILE.exists():
        with open(USER_RATINGS_FILE) as f:
            ratings = json.load(f)

    updated, migrated = merge_picks_into_ratings(picks, ratings)
    tmp = str(USER_RATINGS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(updated, f, indent=1)
    os.replace(tmp, USER_RATINGS_FILE)

    print(f"migrated {len(migrated)} picks into {USER_RATINGS_FILE} as yes-ratings "
          f"({len(picks) - len(migrated)} already had a rating and were left alone)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
