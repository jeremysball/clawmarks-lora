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
