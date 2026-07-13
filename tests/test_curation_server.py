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


def _manifest_entry(tag, prompt):
    return {"tag": tag, "prompt": prompt, "prompt_name": tag, "prompt_type": "style",
            "centroid_sim": 0.5, "novelty": 0.4, "strength": 1.0, "cfg": 7.0, "file": f"{tag}.png"}


def test_cockpit_evidence_ranks_by_word_overlap():
    manifest = [
        _manifest_entry("a", "trentbuckle style, weathered owl portrait, pale paper, ink wash"),
        _manifest_entry("b", "trentbuckle style, empty parking garage at night"),
    ]
    result = cs.cockpit_evidence(manifest, "weathered owl portrait, ink wash", favorites={}, comparisons=[])
    assert [r["tag"] for r in result] == ["a", "b"]
    assert result[0]["similarity"] > result[1]["similarity"]


def test_cockpit_evidence_empty_prompt_returns_nothing():
    manifest = [_manifest_entry("a", "trentbuckle style, weathered owl portrait")]
    assert cs.cockpit_evidence(manifest, "", favorites={}, comparisons=[]) == []
    assert cs.cockpit_evidence(manifest, "   ", favorites={}, comparisons=[]) == []


def test_cockpit_evidence_status_kept_when_favorited():
    manifest = [_manifest_entry("a", "weathered owl portrait")]
    result = cs.cockpit_evidence(manifest, "weathered owl portrait", favorites={"a": {}}, comparisons=[])
    assert result[0]["status"] == "kept"


def test_cockpit_evidence_status_rejected_when_never_won():
    manifest = [_manifest_entry("a", "weathered owl portrait")]
    comparisons = [{"winner": "b", "loser": "a"}, {"winner": "c", "loser": "a"}]
    result = cs.cockpit_evidence(manifest, "weathered owl portrait", favorites={}, comparisons=comparisons)
    assert result[0]["status"] == "rejected"


def test_cockpit_evidence_status_unrated_by_default():
    manifest = [_manifest_entry("a", "weathered owl portrait")]
    result = cs.cockpit_evidence(manifest, "weathered owl portrait", favorites={}, comparisons=[])
    assert result[0]["status"] == "unrated"


def test_cockpit_evidence_favorite_wins_over_loss_record():
    # A tag that lost every comparison but was later favorited anyway should read "kept": the
    # human's explicit bookmark outranks the automatic loss tally.
    manifest = [_manifest_entry("a", "weathered owl portrait")]
    comparisons = [{"winner": "b", "loser": "a"}]
    result = cs.cockpit_evidence(manifest, "weathered owl portrait", favorites={"a": {}}, comparisons=comparisons)
    assert result[0]["status"] == "kept"


def test_cockpit_evidence_respects_top_n():
    manifest = [_manifest_entry(f"t{i}", "weathered owl portrait ink wash") for i in range(5)]
    result = cs.cockpit_evidence(manifest, "weathered owl portrait ink wash", favorites={}, comparisons=[], top_n=2)
    assert len(result) == 2


def test_build_trial_requires_prompt():
    try:
        cs.build_trial({}, now="t0", trial_id="x")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "prompt" in str(e)


def test_build_trial_rejects_bad_seed_strategy():
    try:
        cs.build_trial({"prompt": "p", "seed_strategy": "list"}, now="t0", trial_id="x")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "seed_strategy" in str(e)


def test_build_trial_clamps_batch_size():
    trial = cs.build_trial({"prompt": "p", "n": 99}, now="t0", trial_id="x")
    assert trial["n"] == 6
    trial = cs.build_trial({"prompt": "p", "n": 0}, now="t0", trial_id="x")
    assert trial["n"] == 1


def test_build_trial_defaults_and_shape():
    trial = cs.build_trial({"prompt": " owl portrait ", "mission": "gap"}, now="t0", trial_id="trial_1")
    assert trial["status"] == "draft"
    assert trial["prompt"] == "owl portrait"
    assert trial["seed_strategy"] == "random"
    assert trial["n"] == 4
    assert trial["sampler"] == "ddim"
    assert trial["result_tags"] == []
    assert trial["queue_title"]  # resolved from cockpit.MISSIONS["gap"]


def _trial(**overrides):
    trial = cs.build_trial({"prompt": "owl portrait", "mission": "gap", "n": 3}, now="t0", trial_id="trial_x")
    trial.update(overrides)
    return trial


def test_build_generation_jobs_respects_batch_size():
    jobs = cs.build_generation_jobs(_trial())
    assert len(jobs) == 3
    assert {j["tag"] for j in jobs} == {j["tag"] for j in jobs}  # tags unique
    assert len({j["tag"] for j in jobs}) == 3


def test_build_generation_jobs_random_strategy_varies_seeds():
    jobs = cs.build_generation_jobs(_trial(seed_strategy="random", n=8))
    assert len({j["seed"] for j in jobs}) > 1


def test_build_generation_jobs_fixed_strategy_reuses_one_seed():
    jobs = cs.build_generation_jobs(_trial(seed_strategy="fixed", n=5))
    assert len({j["seed"] for j in jobs}) == 1


def test_build_generation_jobs_carries_prompt_and_generation_params():
    trial = _trial(strength=0.8, cfg=6.0, steps=20, sampler="euler")
    job = cs.build_generation_jobs(trial)[0]
    assert job["prompt"] == "owl portrait"
    assert job["strength"] == 0.8
    assert job["cfg"] == 6.0
    assert job["steps"] == 20
    assert job["sampler"] == "euler"
    assert job["prompt_name"] == "cockpit_gap_trial_x"


def test_build_autopilot_context_pulls_frontier_cells():
    coverage_data = {
        "cells": [
            {"fb": 0, "nb": 0, "count": 5, "frontier": False, "faith_lo": 0.0, "faith_hi": 0.1,
             "novelty_lo": 0.0, "novelty_hi": 0.1, "thumb": "t.jpg", "best_tag": "a",
             "items": [{"tag": "a", "faith": 0.5, "novelty": 0.5, "thumb": "t.jpg", "prompt_name": "p"}]},
            {"fb": 1, "nb": 0, "count": 0, "frontier": True, "faith_lo": 0.1, "faith_hi": 0.2,
             "novelty_lo": 0.0, "novelty_hi": 0.1, "thumb": None, "best_tag": None, "items": []},
        ],
        "max_count": 5,
    }
    context = cs.build_autopilot_context(coverage_data, manifest=[], favorites={}, comparisons=[])
    assert len(context["cells"]) == 1
    assert context["cells"][0]["fb"] == 1


def test_build_autopilot_context_separates_kept_and_rejected_prompts():
    manifest = [_manifest_entry("a", "kept prompt"), _manifest_entry("b", "rejected prompt")]
    favorites = {"a": {}}
    comparisons = [{"winner": "c", "loser": "b"}]
    context = cs.build_autopilot_context({"cells": [], "max_count": 0}, manifest, favorites, comparisons)
    assert context["kept_prompts"] == ["kept prompt"]
    assert context["rejected_prompts"] == ["rejected prompt"]


def test_suggestion_has_numeric_forecast_catches_percentages_and_score_words():
    assert cs.suggestion_has_numeric_forecast("this has a 70% chance of working")
    assert cs.suggestion_has_numeric_forecast("high confidence this will land")
    assert cs.suggestion_has_numeric_forecast("estimated score of 0.8")
    assert not cs.suggestion_has_numeric_forecast("fills a gap near the existing owl cluster")


def test_filter_autopilot_suggestions_drops_numeric_and_incomplete():
    suggestions = [
        {"title": "A", "mission": "gap", "prompt": "p1", "rationale": "clean, no numbers here"},
        {"title": "B", "mission": "gap", "prompt": "p2", "rationale": "85% likely to work"},
        {"title": "C", "mission": "gap", "prompt": "p3"},  # missing rationale
        "not a dict",
    ]
    kept = cs.filter_autopilot_suggestions(suggestions)
    assert [s["title"] for s in kept] == ["A"]
