import json

from clawmarks.compute import comfyui
from clawmarks.search import driver


def test_state_file_round_one_has_no_round_suffix(tmp_path, monkeypatch):
    """Regression test for issue #15: round 1's original script wrote allnight_state.json (no
    round-number suffix). The merged driver must keep reading/writing that same filename, or
    resuming round 1 silently starts over at generation 0 instead of finding its existing state."""
    monkeypatch.setattr(driver, "SWEEP_DIR", tmp_path)
    path = driver._state_file(driver.ROUND_CONFIGS[1])
    assert path == tmp_path / "allnight_state.json"


def test_state_file_round_two_keeps_its_round_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SWEEP2_DIR", tmp_path)
    path = driver._state_file(driver.ROUND_CONFIGS[2])
    assert path == tmp_path / "allnight2_state.json"


def test_load_state_resumes_from_the_correctly_named_round_one_file(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "SWEEP_DIR", tmp_path)
    (tmp_path / "allnight_state.json").write_text(json.dumps({
        "generation": 7, "stage": 1, "plateau_count": 2, "novelty_history": [0.1],
        "gpt55_subjects": [], "start_balance": 5.0, "start_time": 100.0,
    }))
    state = driver.load_state(driver.ROUND_CONFIGS[1])
    assert state["generation"] == 7


def test_load_state_resumes_from_the_correctly_named_round_two_file(tmp_path, monkeypatch):
    """Round-trip test for round 2, mirroring the round 1 test above: proves load_state actually
    reads a file written under the on-disk name _state_file returns for round 2
    (allnight2_state.json in SWEEP2_DIR), not just that the helper computes the right path in
    isolation."""
    monkeypatch.setattr(driver, "SWEEP2_DIR", tmp_path)
    (tmp_path / "allnight2_state.json").write_text(json.dumps({
        "generation": 3, "stage": 0, "plateau_count": 0, "novelty_history": [],
        "gpt55_subjects": [], "start_balance": 1.0, "start_time": 100.0,
    }))
    state = driver.load_state(driver.ROUND_CONFIGS[2])
    assert state["generation"] == 3


def test_load_resumable_manifest_returns_empty_when_no_manifest_exists(tmp_path):
    assert driver._load_resumable_manifest(tmp_path) == []


def test_load_resumable_manifest_resumes_prior_persisted_images(tmp_path):
    """Regression test for issue #15: the main loop used to always start from manifest = [] and
    then overwrite scored_manifest.json with only the new run's images, so a restart permanently
    discarded every previously persisted record. Loading the existing file first prevents that.
    The fixture carries every field a real persisted entry has (see score_batch/submit_and_collect
    in driver.py), not just the three fields build_gallery reads, so this also proves the shape
    survives round-tripping intact."""
    prior = [{
        "tag": "a", "centroid_sim": 0.5, "novelty": 0.3, "file": "a.png",
        "prompt_name": "p", "prompt": "a prompt", "strength": 1.0, "cfg": 7.0,
        "prompt_type": "style",
    }]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(prior))
    assert driver._load_resumable_manifest(tmp_path) == prior


def test_load_resumable_manifest_falls_back_to_empty_on_corrupt_json(tmp_path):
    """A process killed mid-write (before the write became atomic) can leave a truncated
    scored_manifest.json. Losing the resume is acceptable; crashing the whole restart isn't."""
    (tmp_path / "scored_manifest.json").write_text("{not valid json")
    assert driver._load_resumable_manifest(tmp_path) == []


def test_build_gallery_skips_manifest_entries_whose_file_no_longer_exists(tmp_path, monkeypatch):
    """A resumed manifest (see _load_resumable_manifest) can reference a PNG that was deleted
    since it was persisted -- exactly what happened to notes/uncanny_sweep/ per this project's
    CLAUDE.md data-integrity incident log. Before this fix, build_gallery's thumb_data_uri call
    would crash trying to open the missing file, taking down the whole generation loop; it must
    instead render the gallery using only the entries whose files still exist."""
    monkeypatch.setattr(driver, "SWEEP_DIR", tmp_path)
    existing_png = tmp_path / "exists.png"
    from PIL import Image
    Image.new("RGB", (4, 4)).save(existing_png)

    manifest = [
        {"tag": "gone", "file": str(tmp_path / "missing.png"), "centroid_sim": 0.5, "novelty": 0.3,
         "prompt_name": "p", "prompt_type": "style"},
        {"tag": "here", "file": str(existing_png), "centroid_sim": 0.5, "novelty": 0.4,
         "prompt_name": "p", "prompt_type": "style"},
    ]
    best_novelty = driver.build_gallery(driver.ROUND_CONFIGS[1], manifest, real_ref=(0.5, 0.0, 1.0))
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


def test_build_gallery_returns_zero_when_every_file_is_missing(tmp_path, monkeypatch):
    """bin_edges indexes into the sorted centroid_sim/novelty lists, which would raise
    IndexError on an empty list if every manifest entry's file were missing. Must bail out the
    same way the caller already does for a genuinely empty manifest, not crash."""
    monkeypatch.setattr(driver, "SWEEP_DIR", tmp_path)
    manifest = [{"tag": "gone", "file": str(tmp_path / "missing.png"), "centroid_sim": 0.5,
                 "novelty": 0.3, "prompt_name": "p", "prompt_type": "style"}]
    assert driver.build_gallery(driver.ROUND_CONFIGS[1], manifest, real_ref=(0.5, 0.0, 1.0)) == 0.0


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
    driver.submit_and_collect(driver.ROUND_CONFIGS[1], jobs, tmp_path, "gen1", timeout_s=0)

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
    driver.submit_and_collect(driver.ROUND_CONFIGS[1], jobs, tmp_path, "gen1", timeout_s=0)

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
    manifest = driver.submit_and_collect(driver.ROUND_CONFIGS[1], jobs, tmp_path, "gen1", timeout_s=5)

    assert len(manifest) == 1
    assert cancelled == []
