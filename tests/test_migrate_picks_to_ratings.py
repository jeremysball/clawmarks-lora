# tests/test_migrate_picks_to_ratings.py
from clawmarks.search.migrate_picks_to_ratings import merge_picks_into_ratings


def test_migrates_picks_not_already_rated():
    picks = {"a": {"picked_at": "t0"}, "b": {"picked_at": "t1"}}
    ratings = {}
    updated, migrated = merge_picks_into_ratings(picks, ratings)
    assert migrated == ["a", "b"]
    assert updated["a"] == {"label": "yes", "rated_at": "t0"}
    assert updated["b"] == {"label": "yes", "rated_at": "t1"}


def test_does_not_overwrite_an_existing_rating():
    picks = {"a": {"picked_at": "t0"}}
    ratings = {"a": {"label": "no", "rated_at": "t9"}}
    updated, migrated = merge_picks_into_ratings(picks, ratings)
    assert migrated == []
    assert updated["a"] == {"label": "no", "rated_at": "t9"}


def test_leaves_existing_ratings_not_derived_from_picks_untouched():
    picks = {}
    ratings = {"c": {"label": "yes", "rated_at": "t5"}}
    updated, migrated = merge_picks_into_ratings(picks, ratings)
    assert migrated == []
    assert updated == {"c": {"label": "yes", "rated_at": "t5"}}
