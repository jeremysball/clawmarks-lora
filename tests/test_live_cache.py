import os
import time

import pytest

from clawmarks.live_cache import LiveCache


def test_computes_once_and_caches_when_files_unchanged(tmp_path):
    watched = tmp_path / "manifest.json"
    watched.write_text("[]")
    calls = []

    def compute(sweep_dir):
        calls.append(1)
        return {"n": len(calls)}

    cache = LiveCache()
    first = cache.get("scan", compute, watched_files=[str(watched)])
    second = cache.get("scan", compute, watched_files=[str(watched)])

    assert first == {"n": 1}
    assert second is first
    assert len(calls) == 1


def test_recomputes_when_watched_file_mtime_changes(tmp_path):
    watched = tmp_path / "manifest.json"
    watched.write_text("[]")
    calls = []

    def compute(sweep_dir):
        calls.append(1)
        return {"n": len(calls)}

    cache = LiveCache()
    cache.get("scan", compute, watched_files=[str(watched)])

    new_mtime = os.path.getmtime(watched) + 5
    watched.write_text('[{"tag": "a"}]')
    os.utime(watched, (new_mtime, new_mtime))

    second = cache.get("scan", compute, watched_files=[str(watched)])
    assert second == {"n": 2}
    assert len(calls) == 2


def test_depends_on_passes_dependency_data_and_propagates_invalidation(tmp_path):
    watched = tmp_path / "scored_manifest.json"
    watched.write_text("[]")
    base_calls, dependent_calls = [], []

    def compute_base(sweep_dir):
        base_calls.append(1)
        return {"base_n": len(base_calls)}

    def compute_dependent(sweep_dir, deps):
        dependent_calls.append(1)
        return {"from_base": deps["solution-map"]["base_n"], "dependent_n": len(dependent_calls)}

    cache = LiveCache()

    def get_dependent():
        cache.get("solution-map", compute_base, watched_files=[str(watched)])
        return cache.get(
            "map", compute_dependent, watched_files=[], depends_on=["solution-map"],
        )

    first = get_dependent()
    assert first == {"from_base": 1, "dependent_n": 1}

    new_mtime = os.path.getmtime(watched) + 5
    watched.write_text('[{"tag": "a"}]')
    os.utime(watched, (new_mtime, new_mtime))

    second = get_dependent()
    assert second == {"from_base": 2, "dependent_n": 2}


def test_concurrent_get_only_computes_once(tmp_path):
    import threading

    watched = tmp_path / "manifest.json"
    watched.write_text("[]")
    calls = []

    def slow_compute(sweep_dir):
        time.sleep(0.05)
        calls.append(1)
        return {"n": len(calls)}

    cache = LiveCache()
    threads = [
        threading.Thread(target=lambda: cache.get("scan", slow_compute, watched_files=[str(watched)]))
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)

    assert len(calls) == 1


def test_get_reads_dependency_data_and_mtimes_from_a_single_snapshot():
    """Regression test for a race where two separate self._entries[dep_name] reads (one for
    data, one for mtimes) could straddle a concurrent update to the dependency, pairing old
    data with new mtimes and caching a stale result that never self-invalidates. The fix reads
    each dependency's entry once and takes both fields from that single snapshot."""

    class _CountingDict(dict):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.access_count = 0

        def __getitem__(self, key):
            self.access_count += 1
            return super().__getitem__(key)

    cache = LiveCache()
    cache.get("dep", lambda sweep_dir: "dep-data-v1", watched_files=[])

    cache._entries = _CountingDict(cache._entries)
    result = cache.get(
        "target", lambda sweep_dir, deps: deps["dep"], watched_files=[], depends_on=["dep"],
    )

    assert result == "dep-data-v1"
    assert cache._entries.access_count == 1
