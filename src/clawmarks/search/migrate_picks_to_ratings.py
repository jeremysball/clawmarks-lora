"""One-time migration: user_picks.json entries become user_ratings.json entries with
label: "yes", so "pick as winner" can be retired without losing the existing picks. Safe to
rerun: any tag that already has a rating is left alone. Not wired into `clawmarks` as a
permanent CLI subcommand since it's a one-shot migration, not a recurring operation. See
docs/superpowers/specs/2026-07-09-preference-classifier-design.md, Component 2a.

Run with: python -m clawmarks.search.migrate_picks_to_ratings --expedition <name> --leg <name>
"""
import argparse
import json
import os

from clawmarks import config


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--expedition", required=True)
    parser.add_argument("--leg", required=True)
    args = parser.parse_args(argv)
    out_dir = config.leg_dir(args.expedition, args.leg)

    user_picks_file = out_dir / "user_picks.json"
    user_ratings_file = out_dir / "user_ratings.json"

    picks = {}
    if user_picks_file.exists():
        with open(user_picks_file) as f:
            picks = json.load(f)
    ratings = {}
    if user_ratings_file.exists():
        with open(user_ratings_file) as f:
            ratings = json.load(f)

    updated, migrated = merge_picks_into_ratings(picks, ratings)
    tmp = str(user_ratings_file) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(updated, f, indent=1)
    os.replace(tmp, user_ratings_file)

    print(f"migrated {len(migrated)} picks into {user_ratings_file} as yes-ratings "
          f"({len(picks) - len(migrated)} already had a rating and were left alone)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
