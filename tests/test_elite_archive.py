# tests/test_elite_archive.py
import json
import re

from clawmarks.build import elite_archive


def test_main_uses_yes_rated_images_not_user_picks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(elite_archive, "SWEEP_DIR", tmp_path)
    # Force every image into a single cell, regardless of its faith/novelty values, so the test
    # doesn't depend on how a 2-item manifest happens to quantile-split across N_BINS x N_BINS
    # cells (bin_edges(vals, 1) always returns [], so bin_of always returns 0).
    monkeypatch.setattr(elite_archive, "N_BINS", 1)
    manifest = [
        {"tag": "a", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.9,
         "novelty": 0.1, "strength": 1.0, "cfg": 7.0, "file": "a.png"},
        {"tag": "b", "prompt_name": "p", "prompt_type": "style", "centroid_sim": 0.9,
         "novelty": 0.9, "strength": 1.0, "cfg": 7.0, "file": "b.png"},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    # "a" has lower novelty than "b" but is yes-rated: it should win the cell despite that,
    # exactly the behavior user_picks.json used to provide.
    (tmp_path / "user_ratings.json").write_text(json.dumps({"a": {"label": "yes", "rated_at": "t0"}}))
    # a stale user_picks.json should be ignored entirely
    (tmp_path / "user_picks.json").write_text(json.dumps({"b": {"picked_at": "t0"}}))

    elite_archive.main([])

    captured = capsys.readouterr()
    assert "1 occupied cells, 1 human-picked elites" in captured.out

    html = (tmp_path / "archive.html").read_text()
    match = re.search(r"const CELLS = (\[.+?\]);\nlet picks", html)
    assert match is not None, "could not find 'const CELLS = [...]; let picks' in archive.html"
    cells = json.loads(match.group(1))
    assert len(cells) == 1
    tags_in_cell = {item["tag"] for item in cells[0]["items"]}
    assert tags_in_cell == {"a", "b"}
