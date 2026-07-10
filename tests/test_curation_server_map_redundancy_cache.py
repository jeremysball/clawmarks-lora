from clawmarks import curation_server as cs


def test_get_map_data_is_cached_and_depends_on_solution_map(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")

    solution_map_calls = []
    map_calls = []
    monkeypatch.setattr(
        cs.solution_map, "compute_data",
        lambda sweep_dir: solution_map_calls.append(1) or {"points": []},
    )
    monkeypatch.setattr(
        cs.map_view, "compute_data",
        lambda sweep_dir, deps: map_calls.append(1) or {"from_solution_map": deps["solution-map"]},
    )

    first = cs._get_map_data()
    second = cs._get_map_data()

    assert first == {"from_solution_map": {"points": []}}
    assert second is first
    assert len(map_calls) == 1
    assert len(solution_map_calls) == 1


def test_get_redundancy_data_is_cached_and_depends_on_solution_map(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")

    monkeypatch.setattr(cs.solution_map, "compute_data", lambda sweep_dir: {"points": []})
    redundancy_calls = []
    monkeypatch.setattr(
        cs.redundancy_view, "compute_data",
        lambda sweep_dir, deps: redundancy_calls.append(1) or {"from_solution_map": deps["solution-map"]},
    )

    first = cs._get_redundancy_data()
    second = cs._get_redundancy_data()

    assert first == {"from_solution_map": {"points": []}}
    assert second is first
    assert len(redundancy_calls) == 1
