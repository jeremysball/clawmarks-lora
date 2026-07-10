import json

from clawmarks.build import similarity_index


def test_compute_data_returns_tag_to_neighbors_mapping(monkeypatch, tmp_path):
    manifest = [
        {"file": "/tmp/a.png", "tag": "a"},
        {"file": "/tmp/b.png", "tag": "b"},
        {"file": "/tmp/c.png", "tag": "c"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))

    import torch
    fake_embs = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    monkeypatch.setattr(similarity_index, "TOP_K", 2)
    monkeypatch.setattr(similarity_index, "embed_with_progress", lambda paths, model, sweep_dir: fake_embs)
    fake_model = type("FakeModel", (), {"eval": lambda self: None})()
    monkeypatch.setattr(
        similarity_index, "AutoModel",
        type("M", (), {"from_pretrained": staticmethod(lambda *_: fake_model)}),
    )

    data = similarity_index.compute_data(str(tmp_path))
    assert set(data.keys()) == {"a", "b", "c"}
    assert data["a"][0] == "b"
