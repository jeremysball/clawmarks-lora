import json
from pathlib import Path

import pytest

from clawmarks.compute import comfyui
from clawmarks.search import driver


def test_load_leg_config_merges_expedition_and_leg_fields(tmp_path, monkeypatch):
    from clawmarks.search import driver
    from clawmarks import config

    expeditions_dir = tmp_path / "expeditions"
    (expeditions_dir / "demo" / "legs").mkdir(parents=True)
    (expeditions_dir / "demo" / "expedition.json").write_text(json.dumps({
        "wall_clock_cap_hours": 7.5, "budget_usd_cap": 10.0, "budget_safety_margin": 1.5,
        "gen_batch_size": 60, "explore_fraction": 0.5, "max_generations": 400,
        "textures": ["tex-a"], "fallback_subjects": ["subj-a"], "seed_from_start": False,
    }))
    (expeditions_dir / "demo" / "legs" / "leg1.json").write_text(json.dumps({
        "explore_fraction": 0.85, "seed_from_start": True,
    }))
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", expeditions_dir)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")

    cfg = driver.load_leg_config("demo", "leg1")

    assert cfg.expedition == "demo"
    assert cfg.leg == "leg1"
    assert cfg.explore_fraction == 0.85  # leg override wins
    assert cfg.seed_from_start is True  # leg override wins
    assert cfg.wall_clock_cap_hours == 7.5  # inherited from expedition.json
    assert cfg.textures == ["tex-a"]  # inherited, no leg override present
    assert cfg.dir == config.leg_dir("demo", "leg1")


def test_load_leg_config_empty_leg_file_inherits_everything(tmp_path, monkeypatch):
    from clawmarks.search import driver
    from clawmarks import config

    expeditions_dir = tmp_path / "expeditions"
    (expeditions_dir / "demo" / "legs").mkdir(parents=True)
    (expeditions_dir / "demo" / "expedition.json").write_text(json.dumps({
        "wall_clock_cap_hours": 1.0, "budget_usd_cap": 1.0, "budget_safety_margin": 0.1,
        "gen_batch_size": 20, "explore_fraction": 0.85, "max_generations": 60,
        "textures": [], "fallback_subjects": [], "seed_from_start": True,
    }))
    (expeditions_dir / "demo" / "legs" / "cockpit.json").write_text("{}")
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", expeditions_dir)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")

    cfg = driver.load_leg_config("demo", "cockpit")

    assert cfg.explore_fraction == 0.85
    assert cfg.seed_from_start is True


def test_load_leg_config_ignores_expedition_metadata_fields(tmp_path, monkeypatch):
    from clawmarks import config

    expeditions_dir = tmp_path / "expeditions"
    (expeditions_dir / "demo" / "legs").mkdir(parents=True)
    (expeditions_dir / "demo" / "expedition.json").write_text(json.dumps({
        "trigger_word": "trentbuckle style, ",
        "negative_prompt": "low quality, blurry, watermark",
        "wall_clock_cap_hours": 1.0, "budget_usd_cap": 1.0, "budget_safety_margin": 0.1,
        "gen_batch_size": 20, "explore_fraction": 0.5, "max_generations": 60,
        "textures": [], "fallback_subjects": [], "seed_from_start": False,
    }))
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", expeditions_dir)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")

    cfg = driver.load_leg_config("demo", "leg1")

    assert cfg.explore_fraction == 0.5


def test_load_leg_config_missing_expedition_is_a_clear_error(tmp_path, monkeypatch):
    from clawmarks.search import driver
    from clawmarks import config

    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")

    with pytest.raises(RuntimeError, match="unknown expedition"):
        driver.load_leg_config("does_not_exist", "leg1")


def test_load_leg_config_missing_leg_file_is_ok(tmp_path, monkeypatch):
    """A leg file can be entirely absent (not just empty): inherit everything."""
    from clawmarks.search import driver
    from clawmarks import config

    expeditions_dir = tmp_path / "expeditions"
    (expeditions_dir / "demo" / "legs").mkdir(parents=True)
    (expeditions_dir / "demo" / "expedition.json").write_text(json.dumps({
        "wall_clock_cap_hours": 1.0, "budget_usd_cap": 1.0, "budget_safety_margin": 0.1,
        "gen_batch_size": 20, "explore_fraction": 0.5, "max_generations": 60,
        "textures": [], "fallback_subjects": [], "seed_from_start": False,
    }))
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", expeditions_dir)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")

    cfg = driver.load_leg_config("demo", "brand_new_leg")

    assert cfg.explore_fraction == 0.5


def test_load_leg_config_does_not_share_mutable_default_lists(tmp_path, monkeypatch):
    from clawmarks import config

    expeditions_dir = tmp_path / "expeditions"
    (expeditions_dir / "demo" / "legs").mkdir(parents=True)
    (expeditions_dir / "demo" / "expedition.json").write_text(json.dumps({
        "wall_clock_cap_hours": 1.0, "budget_usd_cap": 1.0, "budget_safety_margin": 0.1,
        "gen_batch_size": 20, "explore_fraction": 0.5, "max_generations": 60,
        "textures": [], "fallback_subjects": [], "seed_from_start": False,
    }))
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", expeditions_dir)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")

    cfg1 = driver.load_leg_config("demo", "leg1")
    cfg2 = driver.load_leg_config("demo", "leg2")

    cfg1.widened_textures.append("x")

    assert cfg2.widened_textures == []


def _manifest_entry(tag):
    return {
        "tag": tag,
        "centroid_sim": 0.5,
        "novelty": 0.3,
        "file": "a.png",
        "prompt_name": "p",
        "prompt": "a prompt",
        "seed": 1,
        "strength": 1.0,
        "cfg": 7.0,
        "steps": 28,
        "sampler": "ddim",
        "negative": "bad",
        "prompt_type": "style",
    }


def test_state_file_uses_the_shared_filename_regardless_of_leg(tmp_path):
    """The merged driver uses allnight_state.json for every leg."""
    cfg = driver.LegConfig(
        expedition="demo", leg="leg1", dir=tmp_path,
        wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
        gen_batch_size=60, explore_fraction=0.5, max_generations=400,
        textures=[], fallback_subjects=[], seed_from_start=False,
    )
    path = driver._state_file(cfg)
    assert path == tmp_path / "allnight_state.json"


def test_state_file_second_leg_uses_shared_filename(tmp_path):
    cfg = driver.LegConfig(
        expedition="demo", leg="leg2", dir=tmp_path,
        wall_clock_cap_hours=1.0, budget_usd_cap=1.0, budget_safety_margin=0.1,
        gen_batch_size=20, explore_fraction=0.85, max_generations=60,
        textures=[], fallback_subjects=[], seed_from_start=True,
    )
    path = driver._state_file(cfg)
    assert path == tmp_path / "allnight_state.json"


def test_load_state_resumes_from_the_correctly_named_round_one_file(tmp_path):
    cfg = driver.LegConfig(
        expedition="demo", leg="leg1", dir=tmp_path,
        wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
        gen_batch_size=60, explore_fraction=0.5, max_generations=400,
        textures=[], fallback_subjects=[], seed_from_start=False,
    )
    (tmp_path / "allnight_state.json").write_text(json.dumps({
        "generation": 49, "stage": 1, "plateau_count": 2,
        "novelty_history": [0.1] * 49,
        "gpt55_subjects": [], "start_balance": 5.0, "start_time": 100.0,
    }))
    state = driver.load_state(cfg)
    assert state["generation"] == 49
    assert len(state["novelty_history"]) == 49
    assert state["novelty_history"][0] == 0.1


def test_validate_state_rejects_history_shorter_than_generation(tmp_path):
    state = {
        "generation": 50, "stage": 1, "plateau_count": 2,
        "novelty_history": [0.1] * 49,
        "gpt55_subjects": [], "start_balance": 5.0, "start_time": 100.0,
    }

    with pytest.raises(RuntimeError, match="generation/history mismatch"):
        driver._validate_state(state, tmp_path / "allnight_state.json")


def test_validate_state_rejects_the_old_off_by_one_history_length_the_shim_used_to_tolerate(tmp_path):
    state = {
        "generation": 50, "stage": 1, "plateau_count": 2,
        "novelty_history": [0.1] * 51,
        "gpt55_subjects": [], "start_balance": 5.0, "start_time": 100.0,
    }

    # The removed round-1 shim explicitly tolerated generation + 1 history entries.
    with pytest.raises(RuntimeError, match="generation/history mismatch"):
        driver._validate_state(state, tmp_path / "allnight_state.json")


def test_load_state_resumes_from_the_correctly_named_second_leg_file(tmp_path):
    cfg = driver.LegConfig(
        expedition="demo", leg="leg2", dir=tmp_path,
        wall_clock_cap_hours=1.0, budget_usd_cap=1.0, budget_safety_margin=0.1,
        gen_batch_size=20, explore_fraction=0.85, max_generations=60,
        textures=[], fallback_subjects=[], seed_from_start=True,
    )
    (tmp_path / "allnight_state.json").write_text(json.dumps({
        "generation": 14, "stage": 0, "plateau_count": 0, "novelty_history": [0.1] * 14,
        "gpt55_subjects": [], "start_balance": 1.0, "start_time": 100.0,
    }))
    state = driver.load_state(cfg)
    assert state["generation"] == 14
    assert state["stage"] == 0
    assert "stage" in json.loads((tmp_path / "allnight_state.json").read_text())


def test_second_leg_rejects_missing_required_field(tmp_path):
    cfg = driver.LegConfig(
        expedition="demo", leg="leg2", dir=tmp_path,
        wall_clock_cap_hours=1.0, budget_usd_cap=1.0, budget_safety_margin=0.1,
        gen_batch_size=20, explore_fraction=0.85, max_generations=60,
        textures=[], fallback_subjects=[], seed_from_start=True,
    )
    (tmp_path / "allnight_state.json").write_text(json.dumps({
        "generation": 14, "stage": 0, "novelty_history": [0.1] * 14,
        "gpt55_subjects": [], "start_balance": 1.0, "start_time": 100.0,
    }))
    with pytest.raises(RuntimeError, match="missing required fields"):
        driver.load_state(cfg)


def test_load_resumable_manifest_returns_empty_when_no_manifest_exists(tmp_path):
    assert driver._load_resumable_manifest(tmp_path) == []


def test_load_resumable_manifest_resumes_prior_persisted_images(tmp_path):
    """Regression test for issue #15: the main loop used to always start from manifest = [] and
    then overwrite scored_manifest.json with only the new run's images, so a restart permanently
    discarded every previously persisted record. Loading the existing file first prevents that.
    The fixture carries every field a real persisted entry has (see score_batch/submit_and_collect
    in driver.py), not just the three fields build_gallery reads, so this also proves the shape
    survives round-tripping intact."""
    prior = [
        _manifest_entry("grid_0"),
        _manifest_entry("truncated_0"),
        _manifest_entry("r2_gen1_explore_0_seed1"),
        _manifest_entry("gen1_explore_0_seed1"),
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(prior))
    assert driver._load_resumable_manifest(tmp_path) == prior


def test_load_resumable_manifest_fails_closed_on_corrupt_json(tmp_path):
    """A truncated manifest must stop a paid restart rather than silently discarding its history."""
    (tmp_path / "scored_manifest.json").write_text("{not valid json")
    with pytest.raises(RuntimeError, match="unreadable"):
        driver._load_resumable_manifest(tmp_path)


def test_resume_fails_closed_when_manifest_is_ahead_of_state(tmp_path):
    cfg = driver.LegConfig(
        expedition="demo", leg="leg1", dir=tmp_path,
        wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
        gen_batch_size=60, explore_fraction=0.5, max_generations=400,
        textures=[], fallback_subjects=[], seed_from_start=False,
    )
    state = {
        "generation": 1, "stage": 0, "plateau_count": 0, "novelty_history": [0.1],
        "gpt55_subjects": [], "start_balance": 1.0, "start_time": 100.0,
    }
    driver.save_state(cfg, state)
    manifest = [_manifest_entry("grid_0"), _manifest_entry("truncated_0"),
                _manifest_entry("r2_gen1_explore_0_seed1"), _manifest_entry("gen2_explore_0_seed1")]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="behind manifest"):
        driver._validate_resume_agreement(
            state, driver._load_resumable_manifest(tmp_path),
            driver._state_file(cfg), tmp_path / "scored_manifest.json",
        )


def test_resume_fails_closed_when_state_is_ahead_of_manifest(tmp_path):
    state = {
        "generation": 2, "stage": 0, "plateau_count": 0, "novelty_history": [0.1, 0.2],
        "gpt55_subjects": [], "start_balance": 1.0, "start_time": 100.0,
    }
    manifest = [_manifest_entry("grid_0"), _manifest_entry("truncated_0"),
                _manifest_entry("r2_gen1_explore_0_seed1"), _manifest_entry("gen1_explore_0_seed1")]
    with pytest.raises(RuntimeError, match="ahead of manifest"):
        driver._validate_resume_agreement(
            state, manifest, tmp_path / "allnight_state.json", tmp_path / "scored_manifest.json",
        )


def test_resume_allows_historical_manifest_at_generation_zero(tmp_path):
    state = {
        "generation": 0, "stage": 0, "plateau_count": 0, "novelty_history": [],
        "gpt55_subjects": [], "start_balance": None, "start_time": 100.0,
    }
    manifest = [_manifest_entry("grid_0"), _manifest_entry("truncated_0"),
                _manifest_entry("r2_gen1_explore_0_seed1")]
    driver._validate_resume_agreement(
        state, manifest, tmp_path / "allnight_state.json", tmp_path / "scored_manifest.json",
    )


@pytest.mark.parametrize("tag", ["", 42])
def test_manifest_rejects_empty_or_non_string_tags(tmp_path, tag):
    with pytest.raises(RuntimeError, match="invalid tag"):
        driver._validate_manifest([_manifest_entry(tag)], tmp_path / "scored_manifest.json")


def test_manifest_rejects_duplicate_tags(tmp_path):
    entries = [_manifest_entry("grid_0"), _manifest_entry("grid_0")]
    with pytest.raises(RuntimeError, match="invalid tag"):
        driver._validate_manifest(entries, tmp_path / "scored_manifest.json")


def test_state_and_manifest_writes_replace_sibling_temporary_files(tmp_path, monkeypatch):
    replacements = []
    original_replace = driver.os.replace
    monkeypatch.setattr(driver.os, "replace", lambda source, target: (
        replacements.append((source, target)), original_replace(source, target)
    )[1])
    state = driver._new_state()
    cfg = driver.LegConfig(
        expedition="demo", leg="leg1", dir=tmp_path,
        wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
        gen_batch_size=60, explore_fraction=0.5, max_generations=400,
        textures=[], fallback_subjects=[], seed_from_start=False,
    )
    driver.save_state(cfg, state)
    driver._save_manifest(tmp_path, [])
    assert len(replacements) == 2
    assert all(Path(source).parent == target.parent for source, target in replacements)
    assert all(source != target for source, target in replacements)


def test_build_gallery_skips_manifest_entries_whose_file_no_longer_exists(tmp_path):
    """A resumed manifest (see _load_resumable_manifest) can reference a PNG that was deleted
    since it was persisted -- exactly what happened to notes/uncanny_sweep/ per this project's
    CLAUDE.md data-integrity incident log. Before this fix, build_gallery's thumb_data_uri call
    would crash trying to open the missing file, taking down the whole generation loop; it must
    instead render the gallery using only the entries whose files still exist."""
    existing_png = tmp_path / "exists.png"
    from PIL import Image
    Image.new("RGB", (4, 4)).save(existing_png)

    manifest = [
        {"tag": "gone", "file": str(tmp_path / "missing.png"), "centroid_sim": 0.5, "novelty": 0.3,
         "prompt_name": "p", "prompt_type": "style"},
        {"tag": "here", "file": str(existing_png), "centroid_sim": 0.5, "novelty": 0.4,
         "prompt_name": "p", "prompt_type": "style"},
    ]
    cfg = driver.LegConfig(
        expedition="demo", leg="leg1", dir=tmp_path,
        wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
        gen_batch_size=60, explore_fraction=0.5, max_generations=400,
        textures=[], fallback_subjects=[], seed_from_start=False,
    )
    best_novelty = driver.build_gallery(cfg, manifest, real_ref=(0.5, 0.0, 1.0))
    assert best_novelty == 0.4
    assert (tmp_path / "gallery.html").exists()


def test_spent_or_none_returns_spend_when_balance_check_succeeds(monkeypatch):
    monkeypatch.setattr(driver, "get_balance", lambda: 7.0)
    assert driver._spent_or_none(10.0) == 3.0


def test_spent_or_none_fails_closed_when_balance_check_raises(monkeypatch):
    """Regression test for issue #16: a balance-check failure used to be treated as $0 spent
    (fail-open), letting further paid batches start unchecked. It must instead signal the caller
    to stop."""
    def _raise():
        raise RuntimeError("network blip")
    monkeypatch.setattr(driver, "get_balance", _raise)
    assert driver._spent_or_none(10.0) is None


def test_build_gallery_returns_zero_when_every_file_is_missing(tmp_path):
    """bin_edges indexes into the sorted centroid_sim/novelty lists, which would raise
    IndexError on an empty list if every manifest entry's file were missing. Must bail out the
    same way the caller already does for a genuinely empty manifest, not crash."""
    manifest = [{"tag": "gone", "file": str(tmp_path / "missing.png"), "centroid_sim": 0.5,
                 "novelty": 0.3, "prompt_name": "p", "prompt_type": "style"}]
    cfg = driver.LegConfig(
        expedition="demo", leg="leg1", dir=tmp_path,
        wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
        gen_batch_size=60, explore_fraction=0.5, max_generations=400,
        textures=[], fallback_subjects=[], seed_from_start=False,
    )
    assert driver.build_gallery(cfg, manifest, real_ref=(0.5, 0.0, 1.0)) == 0.0


def test_submit_and_collect_cancels_jobs_still_pending_at_timeout(tmp_path, monkeypatch):
    """Regression test for issue #16: a job still pending when submit_and_collect gives up used
    to be abandoned client-side only, so it kept running (and billing) on the provider side.
    It must be actively cancelled instead."""
    monkeypatch.setattr(comfyui, "api_post", lambda path, payload: {"id": "job-1"} if path == "/run" else {})
    monkeypatch.setattr(comfyui, "api_get", lambda path: {"status": "IN_PROGRESS"})
    cancelled = []
    monkeypatch.setattr(comfyui, "cancel_job", lambda job_id: cancelled.append(job_id))

    jobs = [{"tag": "t0", "prompt": "p", "seed": 1, "strength": 1.0, "cfg": 7.0, "steps": 28,
             "sampler": "ddim", "negative": "bad"}]
    cfg = driver.LegConfig(
        expedition="demo", leg="leg1", dir=tmp_path,
        wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
        gen_batch_size=60, explore_fraction=0.5, max_generations=400,
        textures=[], fallback_subjects=[], seed_from_start=False,
    )
    driver.submit_and_collect(cfg, jobs, tmp_path, "gen1", timeout_s=0)

    assert cancelled == ["job-1"]


def test_submit_and_collect_surfaces_cancel_failure_in_summary(tmp_path, monkeypatch, capsys):
    """Regression test for issue #16 review response: a cancel_job failure was silently
    swallowed with no trace in the printed summary line, so an operator skimming logs had no
    way to notice a job that kept running (and billing) because its cancel call itself failed.
    The summary line must surface a cancel_failed count."""
    monkeypatch.setattr(comfyui, "api_post", lambda path, payload: {"id": "job-1"} if path == "/run" else {})
    monkeypatch.setattr(comfyui, "api_get", lambda path: {"status": "IN_PROGRESS"})

    def _raise(job_id):
        raise RuntimeError("cancel endpoint 500")
    monkeypatch.setattr(comfyui, "cancel_job", _raise)

    jobs = [{"tag": "t0", "prompt": "p", "seed": 1, "strength": 1.0, "cfg": 7.0, "steps": 28,
             "sampler": "ddim", "negative": "bad"}]
    cfg = driver.LegConfig(
        expedition="demo", leg="leg1", dir=tmp_path,
        wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
        gen_batch_size=60, explore_fraction=0.5, max_generations=400,
        textures=[], fallback_subjects=[], seed_from_start=False,
    )
    driver.submit_and_collect(cfg, jobs, tmp_path, "gen1", timeout_s=0)

    out = capsys.readouterr().out
    assert "CANCEL_FAIL job-1" in out
    assert "cancel_failed=1" in out


def test_submit_and_collect_does_not_cancel_completed_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(comfyui, "api_post", lambda path, payload: {"id": "job-1"} if path == "/run" else {})
    from PIL import Image
    from io import BytesIO
    import base64
    buf = BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    monkeypatch.setattr(comfyui, "api_get", lambda path: {"status": "COMPLETED", "output": {"images": [{"data": encoded}]}})
    cancelled = []
    monkeypatch.setattr(comfyui, "cancel_job", lambda job_id: cancelled.append(job_id))

    jobs = [{"tag": "t0", "prompt": "p", "seed": 1, "strength": 1.0, "cfg": 7.0, "steps": 28,
             "sampler": "ddim", "negative": "bad"}]
    cfg = driver.LegConfig(
        expedition="demo", leg="leg1", dir=tmp_path,
        wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
        gen_batch_size=60, explore_fraction=0.5, max_generations=400,
        textures=[], fallback_subjects=[], seed_from_start=False,
    )
    manifest = driver.submit_and_collect(cfg, jobs, tmp_path, "gen1", timeout_s=5)

    assert len(manifest) == 1
    assert cancelled == []
