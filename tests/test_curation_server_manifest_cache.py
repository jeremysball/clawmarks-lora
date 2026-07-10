# tests/test_curation_server_manifest_cache.py
import json
import os
import threading
import time

from clawmarks import curation_server as cs


def test_load_manifest_re_reads_when_file_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_manifest_cache", {"manifest": None, "mtime": None})

    path = tmp_path / "scored_manifest.json"
    path.write_text(json.dumps([{"tag": "a"}]))
    first = cs.load_manifest()
    assert first == [{"tag": "a"}]

    # Same content + same mtime should hit the cache (no re-read).
    second = cs.load_manifest()
    assert second is first

    # Bump mtime and rewrite. Without invalidation, the cache would still return
    # the old in-memory manifest; with mtime invalidation it sees the new content.
    new_mtime = os.path.getmtime(path) + 5
    path.write_text(json.dumps([{"tag": "a"}, {"tag": "b"}]))
    os.utime(path, (new_mtime, new_mtime))
    third = cs.load_manifest()
    assert third == [{"tag": "a"}, {"tag": "b"}]
    assert third is not first


def test_load_manifest_parses_only_once_under_concurrent_access(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_manifest_cache", {"manifest": None, "mtime": None})

    path = tmp_path / "scored_manifest.json"
    path.write_text(json.dumps([{"tag": "a"}]))

    parse_calls = []
    real_json_load = json.load

    def counting_load(f):
        parse_calls.append(1)
        time.sleep(0.02)
        return real_json_load(f)

    monkeypatch.setattr(cs.json, "load", counting_load)

    threads = [threading.Thread(target=cs.load_manifest) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)

    assert len(parse_calls) == 1
