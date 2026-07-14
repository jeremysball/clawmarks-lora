import json
import os

import pytest

from clawmarks.build import merge_round2


def test_manifest_write_failure_leaves_original_manifest_intact(tmp_path, monkeypatch):
    sweep_dir = tmp_path / "uncanny_sweep"
    sweep2_dir = tmp_path / "uncanny_sweep2"
    sweep_dir.mkdir()
    sweep2_dir.mkdir()

    manifest1 = [{"file": "a.png", "tag": "gen1_x"}]
    manifest2 = [{"file": "b.png", "tag": "gen1_y"}]

    manifest_file = sweep_dir / "scored_manifest.json"
    manifest_file.write_text(json.dumps(manifest1))
    (sweep2_dir / "scored_manifest.json").write_text(json.dumps(manifest2))

    monkeypatch.setattr(merge_round2, "SWEEP_DIR", sweep_dir)
    monkeypatch.setattr(merge_round2, "SWEEP2_DIR", sweep2_dir)
    monkeypatch.setattr(merge_round2, "MANIFEST_FILE", str(manifest_file))

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("simulated crash mid-write")))

    with pytest.raises(OSError):
        merge_round2.main()

    assert json.loads(manifest_file.read_text()) == manifest1


def test_embeddings_write_failure_leaves_original_embeddings_cache_intact(tmp_path, monkeypatch):
    """The manifest write (first) is allowed to succeed; only the embeddings-cache write
    (second, protecting the irreplaceable solution_map_final_embs.pt) fails."""
    import torch
    from PIL import Image

    sweep_dir = tmp_path / "uncanny_sweep"
    sweep2_dir = tmp_path / "uncanny_sweep2"
    sweep_dir.mkdir()
    sweep2_dir.mkdir()

    img1 = sweep_dir / "a.png"
    img2 = sweep2_dir / "b.png"
    Image.new("RGB", (32, 32), color="red").save(img1)
    Image.new("RGB", (32, 32), color="blue").save(img2)

    manifest1 = [{"file": str(img1), "tag": "gen1_x"}]
    manifest2 = [{"file": str(img2), "tag": "gen1_y"}]
    manifest_file = sweep_dir / "scored_manifest.json"
    manifest_file.write_text(json.dumps(manifest1))
    (sweep2_dir / "scored_manifest.json").write_text(json.dumps(manifest2))

    embs_file = sweep_dir / "solution_map_final_embs.pt"
    original_cache = {
        "paths": [str(img1)], "real_paths": [], "real_embs": torch.zeros(0, 8),
        "gen_embs": torch.rand(1, 8),
    }
    torch.save(original_cache, embs_file)
    original_bytes = embs_file.read_bytes()

    monkeypatch.setattr(merge_round2, "SWEEP_DIR", sweep_dir)
    monkeypatch.setattr(merge_round2, "SWEEP2_DIR", sweep2_dir)
    monkeypatch.setattr(merge_round2, "MANIFEST_FILE", str(manifest_file))
    monkeypatch.setattr(merge_round2, "EMBS_FILE", str(embs_file))

    def fake_forward(self, pixel_values):
        n = pixel_values.shape[0]
        return type("Out", (), {"pooler_output": torch.rand(n, 8)})()
    fake_model = type("FakeModel", (), {"eval": lambda self: None, "__call__": fake_forward})()
    monkeypatch.setattr(
        merge_round2, "AutoModel",
        type("M", (), {"from_pretrained": staticmethod(lambda *_: fake_model)}),
    )

    real_replace = os.replace
    calls = {"n": 0}

    def replace_second_call_fails(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_replace(*a, **k)
        raise OSError("simulated crash mid-write")
    monkeypatch.setattr(os, "replace", replace_second_call_fails)

    with pytest.raises(OSError):
        merge_round2.main()

    assert json.loads(manifest_file.read_text()) == [
        dict(manifest1[0], round=1),
        dict(manifest2[0], round=2, tag="r2_gen1_y"),
    ]
    assert embs_file.read_bytes() == original_bytes


def test_already_merged_manifest_is_a_noop_and_does_not_touch_replace(tmp_path, monkeypatch):
    sweep_dir = tmp_path / "uncanny_sweep"
    sweep2_dir = tmp_path / "uncanny_sweep2"
    sweep_dir.mkdir()
    sweep2_dir.mkdir()

    manifest1 = [{"file": "a.png", "tag": "gen1_x", "round": 2}]
    manifest_file = sweep_dir / "scored_manifest.json"
    manifest_file.write_text(json.dumps(manifest1))
    (sweep2_dir / "scored_manifest.json").write_text(json.dumps([{"file": "b.png", "tag": "gen1_y"}]))

    monkeypatch.setattr(merge_round2, "SWEEP_DIR", sweep_dir)
    monkeypatch.setattr(merge_round2, "SWEEP2_DIR", sweep2_dir)
    monkeypatch.setattr(merge_round2, "MANIFEST_FILE", str(manifest_file))

    def boom(*a, **k):
        raise AssertionError("should not be called: manifest already merged")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(SystemExit) as exc:
        merge_round2.main()
    assert exc.value.code == 0
