"""Crash-safe file writes: write to a temp file in the target's own directory, then
os.replace (atomic on POSIX) into place. A process killed mid-write leaves the temp file
truncated and the original untouched, instead of leaving the only copy corrupt."""
import json
import os
import tempfile
from pathlib import Path


def fsync_directory(path):
    fd = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_makedirs(path):
    path = Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir():
        raise NotADirectoryError(cursor)
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        fsync_directory(directory)
        fsync_directory(directory.parent)


def _replace_via_temp_file(path, mode, write_fn):
    path = Path(path)
    durable_makedirs(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, mode) as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_json_write(path, value):
    _replace_via_temp_file(path, "w", lambda f: json.dump(value, f, indent=1))


def atomic_write(path, write_fn):
    """`write_fn(file_obj)` performs the write (e.g. torch.save(obj, file_obj)) against a
    binary-mode temp file that then atomically replaces `path`."""
    _replace_via_temp_file(path, "wb", write_fn)
