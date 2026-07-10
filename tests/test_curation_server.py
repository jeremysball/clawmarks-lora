# tests/test_curation_server.py
from clawmarks import curation_server as cs


def test_next_rating_response_returns_item_summary_shape():
    manifest = [
        {"tag": "a", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.5,
         "novelty": 0.3, "strength": 1.0, "cfg": 7.0, "file": "a.png"},
    ]
    result = cs.next_rating_response(manifest, reviewed_tags=set())
    assert result["tag"] == "a"
    assert result["faith"] == 0.5
    assert "done" not in result


def test_next_rating_response_reports_done_when_all_reviewed():
    manifest = [{"tag": "a", "centroid_sim": 0.5, "novelty": 0.3, "prompt_name": "p",
                 "prompt_type": "style", "strength": 1.0, "cfg": 7.0, "file": "a.png"}]
    result = cs.next_rating_response(manifest, reviewed_tags={"a"})
    assert result == {"done": True}


def test_record_rating_upserts_with_timestamp():
    ratings = {}
    updated = cs.record_rating(ratings, "a", "yes", now="2026-07-10T00:00:00Z")
    assert updated["a"] == {"label": "yes", "rated_at": "2026-07-10T00:00:00Z"}


def test_record_rating_overwrites_not_duplicates():
    ratings = {"a": {"label": "no", "rated_at": "t0"}}
    updated = cs.record_rating(ratings, "a", "yes", now="t1")
    assert updated == {"a": {"label": "yes", "rated_at": "t1"}}
    assert len(updated) == 1


def test_record_rating_rejects_invalid_label():
    try:
        cs.record_rating({}, "a", "maybe", now="t0")
        assert False, "expected ValueError"
    except ValueError:
        pass
