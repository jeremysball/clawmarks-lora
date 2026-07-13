# tests/test_curation_server.py
from clawmarks import curation_server as cs


def test_next_compare_response_returns_two_item_summaries():
    manifest = [
        {"tag": "a", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.5,
         "novelty": 0.3, "strength": 1.0, "cfg": 7.0, "file": "a.png"},
        {"tag": "b", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.6,
         "novelty": 0.4, "strength": 1.0, "cfg": 7.0, "file": "b.png"},
    ]
    result = cs.next_compare_response(manifest, comparisons=[])
    assert {result["img1"]["tag"], result["img2"]["tag"]} == {"a", "b"}
    assert result["img1"]["faith"] in (0.5, 0.6)
    assert "done" not in result


def test_next_compare_response_reports_done_with_one_image():
    manifest = [{"tag": "a", "centroid_sim": 0.5, "novelty": 0.3, "prompt_name": "p",
                 "prompt_type": "style", "strength": 1.0, "cfg": 7.0, "file": "a.png"}]
    result = cs.next_compare_response(manifest, comparisons=[])
    assert result == {"done": True}


def test_next_compare_response_never_repeats_the_only_already_judged_pair():
    manifest = [
        {"tag": "a", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.5,
         "novelty": 0.3, "strength": 1.0, "cfg": 7.0, "file": "a.png"},
        {"tag": "b", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.6,
         "novelty": 0.4, "strength": 1.0, "cfg": 7.0, "file": "b.png"},
        {"tag": "c", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.7,
         "novelty": 0.5, "strength": 1.0, "cfg": 7.0, "file": "c.png"},
    ]
    comparisons = [{"winner": "a", "loser": "b", "compared_at": "t0"}] * 5
    for _ in range(20):
        result = cs.next_compare_response(manifest, comparisons)
        assert {result["img1"]["tag"], result["img2"]["tag"]} != {"a", "b"}


def test_record_comparison_appends_with_timestamp():
    updated = cs.record_comparison([], "a", "b", now="2026-07-10T00:00:00Z")
    assert updated == [{"winner": "a", "loser": "b", "compared_at": "2026-07-10T00:00:00Z"}]


def test_record_comparison_preserves_existing_records():
    comparisons = [{"winner": "a", "loser": "b", "compared_at": "t0"}]
    updated = cs.record_comparison(comparisons, "b", "a", now="t1")
    assert updated == [
        {"winner": "a", "loser": "b", "compared_at": "t0"},
        {"winner": "b", "loser": "a", "compared_at": "t1"},
    ]


def test_record_comparison_does_not_mutate_input():
    comparisons = [{"winner": "a", "loser": "b", "compared_at": "t0"}]
    cs.record_comparison(comparisons, "b", "a", now="t1")
    assert len(comparisons) == 1
