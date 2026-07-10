from clawmarks import curation_server as cs


def test_get_manifest_cached_reuses_cache_across_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")

    calls = []

    def compute(sweep_dir):
        calls.append(1)
        return {"n": len(calls)}

    first = cs._get_manifest_cached("coverage", compute)
    second = cs._get_manifest_cached("coverage", compute)

    assert first == {"n": 1}
    assert second is first
    assert len(calls) == 1


def test_get_manifest_cached_keeps_targets_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")

    a = cs._get_manifest_cached("novelty_decay", lambda sweep_dir: {"which": "novelty_decay"})
    b = cs._get_manifest_cached("lineage", lambda sweep_dir: {"which": "lineage"})

    assert a == {"which": "novelty_decay"}
    assert b == {"which": "lineage"}


def test_archive_route_caches_actual_and_predicted_preference_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")

    calls = []

    def fake_compute_data(sweep_dir, use_predicted_preference=False):
        calls.append(use_predicted_preference)
        return {"use_predicted_preference": use_predicted_preference}

    monkeypatch.setattr(cs.elite_archive, "compute_data", fake_compute_data)

    actual = cs._get_manifest_cached(
        "archive_actual", lambda sd: cs.elite_archive.compute_data(sd, use_predicted_preference=False),
    )
    predicted = cs._get_manifest_cached(
        "archive_predicted", lambda sd: cs.elite_archive.compute_data(sd, use_predicted_preference=True),
    )
    actual_again = cs._get_manifest_cached(
        "archive_actual", lambda sd: cs.elite_archive.compute_data(sd, use_predicted_preference=False),
    )

    assert actual == {"use_predicted_preference": False}
    assert predicted == {"use_predicted_preference": True}
    assert actual_again is actual
    assert calls == [False, True]  # only two real computes, not three
