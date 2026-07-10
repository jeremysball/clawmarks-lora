import json

from clawmarks.build import solution_map


def test_compute_data_returns_both_outputs(monkeypatch, tmp_path):
    manifest = [
        {"file": f"/tmp/{i}.png", "tag": f"gen0_{i}", "category": "seedrun1", "prompt_name": "p",
         "prompt_type": "conflict", "centroid_sim": 0.5, "novelty": 0.4}
        for i in range(10)
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))

    import torch
    monkeypatch.setattr(solution_map, "TOP_K", 1)

    def fake_embed(paths, model, label, sweep_dir):
        n = len(paths)
        torch.manual_seed(0)
        vecs = torch.rand(n, 8) + 1
        return vecs / vecs.norm(dim=1, keepdim=True)

    monkeypatch.setattr(solution_map, "embed_with_progress", fake_embed)
    fake_model = type("FakeModel", (), {"eval": lambda self: None})()
    monkeypatch.setattr(
        solution_map, "AutoModel",
        type("M", (), {"from_pretrained": staticmethod(lambda *_: fake_model)}),
    )
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    for i in range(6):
        (real_dir / f"r{i}.jpg").write_bytes(b"fake")
    monkeypatch.setattr(solution_map, "REAL_DIR", str(real_dir))

    data = solution_map.compute_data(str(tmp_path))
    assert "solution_map_data" in data
    assert "similarity_scored" in data
    assert len(data["solution_map_data"]["points"]) == 10
    assert len(data["solution_map_data"]["real_points"]) == 6
    assert set(data["similarity_scored"].keys()) == {f"gen0_{i}" for i in range(10)}
