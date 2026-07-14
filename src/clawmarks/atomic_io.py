"""Crash-safe file writes: write to a temp file in the target's own directory, then
os.replace (atomic on POSIX) into place. A process killed mid-write leaves the temp file
truncated and the original untouched, instead of leaving the only copy corrupt."""
import json
import os
import tempfile
from pathlib import Path


def _replace_via_temp_file(path, mode, write_fn):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, mode) as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json_write(path, value):
    _replace_via_temp_file(path, "w", lambda f: json.dump(value, f, indent=1))


def atomic_write(path, write_fn):
    """`write_fn(file_obj)` performs the write (e.g. torch.save(obj, file_obj)) against a
    binary-mode temp file that then atomically replaces `path`."""
    _replace_via_temp_file(path, "wb", write_fn)
