from clawmarks import curation_server as cs


def test_get_solution_map_data_uses_live_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")

    sentinel = {"solution_map_data": {"points": [], "real_points": []}, "similarity_scored": {}}
    monkeypatch.setattr(cs.solution_map, "compute_data", lambda sweep_dir: sentinel)

    assert cs._get_solution_map_data() is sentinel


def test_get_solution_map_data_watches_the_final_embeddings_file_too(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    (tmp_path / "scored_manifest.json").write_text("[]")
    embs_file = tmp_path / "solution_map_final_embs.pt"
    embs_file.write_text("v1")

    calls = []
    monkeypatch.setattr(
        cs.solution_map, "compute_data",
        lambda sweep_dir: calls.append(1) or {"n": len(calls)},
    )

    first = cs._get_solution_map_data()
    assert first == {"n": 1}

    # Second call with nothing changed should hit the cache.
    second = cs._get_solution_map_data()
    assert second == {"n": 1}

    # Swap the embeddings file (simulating merge_round2.py overwriting it) without touching
    # scored_manifest.json. Without watching this file, the cache would never notice.
    import os, time
    new_mtime = os.path.getmtime(embs_file) + 5
    embs_file.write_text("v2")
    os.utime(embs_file, (new_mtime, new_mtime))

    third = cs._get_solution_map_data()
    assert third == {"n": 2}
