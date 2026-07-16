"""
Generalizes the mtime-invalidated cache curation_server.py already used just for
scored_manifest.json (see the old _manifest_cache/load_manifest()) to every live-rendered
tool-page target: one cache entry per target name, invalidated when any of that target's
declared watched files change mtime, with support for one target's compute_fn depending on
another's already-cached data (e.g. "map" and "redundancy" both need "solution-map"'s output).
"""
import os
import threading


class LiveCache:
    def __init__(self):
        self._entries = {}
        self._locks = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, target_name):
        with self._locks_guard:
            if target_name not in self._locks:
                self._locks[target_name] = threading.Lock()
            return self._locks[target_name]

    def _current_mtimes(self, watched_files):
        mtimes = {}
        for path in watched_files:
            try:
                mtimes[path] = os.path.getmtime(path)
            except FileNotFoundError:
                # A new leg has no manifest or model yet. Record that absence so creating the
                # watched file later invalidates this cached empty state.
                mtimes[path] = None
        return mtimes

    def get(self, target_name, compute_fn, watched_files, depends_on=(), sweep_dir=None):
        with self._lock_for(target_name):
            deps = None
            dep_mtimes = {}
            if depends_on:
                deps = {}
                for dep_name in depends_on:
                    if dep_name not in self._entries:
                        raise KeyError(
                            f"target {target_name!r} depends on {dep_name!r}, "
                            f"but {dep_name!r} has never been computed yet. "
                            f"Call cache.get({dep_name!r}, ...) before {target_name!r}."
                        )
                    dep_entry = self._entries[dep_name]
                    deps[dep_name] = dep_entry["data"]
                    dep_mtimes[dep_name] = dep_entry["mtimes"]

            mtimes = self._current_mtimes(watched_files)
            entry = self._entries.get(target_name)
            if entry is not None and entry["mtimes"] == mtimes and entry["dep_mtimes"] == dep_mtimes:
                return entry["data"]

            data = compute_fn(sweep_dir, deps) if depends_on else compute_fn(sweep_dir)
            self._entries[target_name] = {"data": data, "mtimes": mtimes, "dep_mtimes": dep_mtimes}
            return data
