# tests/test_seed_pool.py
from clawmarks.search import seed_pool


def test_merge_adds_new_subjects_and_reports_them():
    existing = {"close-up cat portrait": {"source": "fallback", "created_at": "t0"}}
    updated, added = seed_pool.merge(
        existing, ["airport baggage carousel", "close-up cat portrait"],
        source="gpt5.5", created_at="t1",
    )
    assert added == ["airport baggage carousel"]
    assert "airport baggage carousel" in updated
    assert updated["airport baggage carousel"] == {"source": "gpt5.5", "created_at": "t1"}


def test_merge_dedupes_case_insensitively():
    existing = {"Close-Up Cat Portrait": {"source": "fallback", "created_at": "t0"}}
    updated, added = seed_pool.merge(
        existing, ["close-up cat portrait"], source="gpt5.5", created_at="t1",
    )
    assert added == []
    assert len(updated) == 1


def test_merge_dedupes_within_the_new_batch_itself():
    updated, added = seed_pool.merge(
        {}, ["glass office atrium", "Glass Office Atrium"], source="gpt5.5", created_at="t1",
    )
    assert added == ["glass office atrium"]
    assert len(updated) == 1


def test_load_missing_file_returns_empty_dict(tmp_path):
    assert seed_pool.load(tmp_path / "does_not_exist.json") == {}


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "seeds.json"
    seeds = {"roadwork cones": {"source": "gpt5.5", "created_at": "t1"}}
    seed_pool.save(path, seeds)
    assert seed_pool.load(path) == seeds
