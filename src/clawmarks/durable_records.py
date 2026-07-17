"""Durable record primitives: canonical JSON, digests, identifiers, and ordered
reentrant cross-process file locks.

These primitives are intentionally low-level. Focus records, leg transitions, and
later storage code compose `file_locks`, `record_locks`, and `leg_write_lock` to
guarantee that multi-path transitions always acquire locks in the same order and
that a single process can re-enter a lock it already holds without deadlocking."""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import re
import stat
import threading
import uuid
from pathlib import Path
from typing import Iterable, Iterator

from clawmarks.atomic_io import durable_makedirs, fsync_directory


# Only characters that are safe in a single path component.
_COMPONENT_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")

# Process-local reentrant locks, one per absolute lock path.
_rlock_lock = threading.Lock()
_rlocks: dict[Path, threading.RLock] = {}

# Per-thread nesting count for each lock path.
_thread_local = threading.local()


def _get_rlock(path: Path) -> threading.RLock:
    """Return the process-local RLock for ``path``, creating it if necessary."""
    with _rlock_lock:
        lock = _rlocks.get(path)
        if lock is None:
            lock = threading.RLock()
            _rlocks[path] = lock
        return lock


def _depths() -> dict[Path, int]:
    """Return this thread's map of lock path -> nesting depth."""
    try:
        return _thread_local.depths  # type: ignore[return-value]
    except AttributeError:
        _thread_local.depths = {}
        return _thread_local.depths  # type: ignore[return-value]


def _open_lock_file(path: Path) -> int:
    """Open the lock file at ``path``, creating it durably if it does not exist.

    The file is created with ``O_CREAT | O_EXCL`` so two racing processes never
    both truncate an existing lock file. The parent directory is created with
    ``durable_makedirs`` and both the new file and its parent directory are
    fsynced before any ``flock`` call is made. Symlinks and non-regular files
    are rejected.
    """
    parent = path.parent
    durable_makedirs(parent)

    if os.path.islink(path):
        raise ValueError(f"lock path is a symlink: {path}")

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow

    try:
        fd = os.open(path, create_flags, 0o644)
    except FileExistsError:
        fd = os.open(path, os.O_RDWR | nofollow)

    try:
        st = os.fstat(fd)
    except OSError:
        os.close(fd)
        raise

    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise ValueError(f"lock path is not a regular file: {path}")

    os.fsync(fd)
    fsync_directory(parent)
    return fd


@contextlib.contextmanager
def file_locks(paths: Iterable[Path]) -> Iterator[None]:
    """Acquire exclusive flocks on all ``paths`` in a deterministic order.

    Parent directories are resolved to absolute form while each final path
    component remains literal, then paths are deduplicated and sorted before
    any lock is taken. This preserves the symlink check for the lock file
    itself. A process-local ``threading.RLock`` plus a thread-local nesting
    count make the lock reentrant within a process: only the outermost
    acquisition opens the file and calls ``flock(LOCK_EX)``; only the
    outermost release calls ``flock(LOCK_UN)`` and closes the descriptor.
    """
    resolved = sorted({Path(p).parent.resolve() / Path(p).name for p in paths})
    acquired: list[tuple[Path, threading.RLock]] = []
    opened: list[tuple[Path, int]] = []
    depths = _depths()

    try:
        for path in resolved:
            rlock = _get_rlock(path)
            rlock.acquire()
            acquired.append((path, rlock))
            if depths.get(path, 0) == 0:
                fd = _open_lock_file(path)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except BaseException:
                    os.close(fd)
                    raise
                opened.append((path, fd))
                depths[path] = 1
            else:
                depths[path] += 1
        yield
    finally:
        for path, rlock in reversed(acquired):
            count = depths.get(path, 0)
            if count == 1:
                for idx, (opened_path, fd) in enumerate(opened):
                    if opened_path == path:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                        os.close(fd)
                        opened.pop(idx)
                        break
                depths.pop(path, None)
            elif count > 1:
                depths[path] = count - 1
            rlock.release()


@contextlib.contextmanager
def record_locks(lock_root: Path, identities: Iterable[str]) -> Iterator[None]:
    """Acquire the per-identity lock files under ``lock_root``."""
    paths = [record_lock_path(lock_root, identity) for identity in identities]
    with file_locks(paths):
        yield


@contextlib.contextmanager
def leg_write_lock(
    state_dir: Path, expedition: str, leg: str
) -> Iterator[None]:
    """Acquire the single write lock for an expedition leg."""
    with file_locks([leg_lock_path(state_dir, expedition, leg)]):
        yield


def canonical_json_bytes(value: object) -> bytes:
    """Return a canonical, compact JSON byte string for ``value``.

    Mapping keys are sorted, whitespace is removed, and non-ASCII characters are
    escaped so that the same logical value always produces the same bytes.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    """Return the SHA-256 hex digest of the canonical JSON for ``value``."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at ``path``."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    """Return a new identifier such as ``focus_<32-hex-chars>``."""
    return f"{prefix}_{uuid.uuid4().hex}"


def validate_component(value: str, kind: str) -> str:
    """Return ``value`` if it is a safe single path component, else raise.

    Rejects empty strings, ``.`` and ``..``, and any character outside
    ``[A-Za-z0-9_.-]``.
    """
    if not isinstance(value, str):
        raise TypeError(f"{kind} must be a string, got {type(value).__name__}")
    if not value or not _COMPONENT_RE.match(value):
        raise ValueError(f"{kind} contains invalid characters: {value!r}")
    if value in (".", ".."):
        raise ValueError(f"{kind} cannot be {value!r}")
    return value


def record_lock_path(state_dir: Path, identity: str) -> Path:
    """Return the lock file path for ``identity`` under ``state_dir``."""
    safe = validate_component(identity, "identity")
    return Path(state_dir) / f"{safe}.lock"


def leg_lock_path(state_dir: Path, expedition: str, leg: str) -> Path:
    """Return the leg write lock path under ``state_dir``."""
    exp = validate_component(expedition, "expedition")
    lg = validate_component(leg, "leg")
    return Path(state_dir) / "locks" / "expeditions" / exp / "legs" / lg / "write.lock"
