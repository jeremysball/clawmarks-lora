# tests/test_manifest_index.py
from clawmarks.search import manifest_index


def test_index_by_tag_builds_lookup():
    manifest = [{"tag": "a", "x": 1}, {"tag": "b", "x": 2}]
    idx = manifest_index.index_by_tag(manifest)
    assert idx == {"a": {"tag": "a", "x": 1}, "b": {"tag": "b", "x": 2}}


def test_item_summary_falls_back_to_basename_when_no_thumb(tmp_path):
    m = {"tag": "t1", "prompt_name": "style_x", "prompt_type": "style",
         "centroid_sim": 0.5, "novelty": 0.25, "strength": 1.0, "cfg": 7.0,
         "file": str(tmp_path / "images" / "t1.png")}
    summary = manifest_index.item_summary(m, tmp_path)
    assert summary["thumb"] == "t1.png"
    assert summary["file"] == "t1.png"
    assert summary["faith"] == 0.5
    assert summary["novelty"] == 0.25


def test_item_summary_uses_thumb_when_present(tmp_path):
    thumbs_dir = tmp_path / "thumbs"
    thumbs_dir.mkdir()
    (thumbs_dir / "t1.jpg").write_bytes(b"x")
    m = {"tag": "t1", "prompt_name": "style_x", "prompt_type": "style",
         "centroid_sim": 0.5, "novelty": 0.25, "strength": 1.0, "cfg": 7.0,
         "file": str(tmp_path / "t1.png")}
    summary = manifest_index.item_summary(m, tmp_path)
    assert summary["thumb"] == "thumbs/t1.jpg"
