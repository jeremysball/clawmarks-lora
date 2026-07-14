import json
import os

import pytest

from clawmarks.atomic_io import atomic_json_write, atomic_write


def test_atomic_json_write_creates_readable_file(tmp_path):
    target = tmp_path / "manifest.json"
    atomic_json_write(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}


def test_atomic_json_write_leaves_no_temp_file_behind(tmp_path):
    target = tmp_path / "manifest.json"
    atomic_json_write(target, {"a": 1})
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_json_write_replaces_existing_file(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps({"a": 1}))
    atomic_json_write(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}


def test_atomic_json_write_failure_leaves_original_intact(tmp_path, monkeypatch):
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps({"a": 1}))

    def boom(*args, **kwargs):
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(OSError):
        atomic_json_write(target, {"a": 2})

    assert json.loads(target.read_text()) == {"a": 1}


def test_atomic_json_write_failure_leaves_no_temp_file_behind(tmp_path, monkeypatch):
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps({"a": 1}))

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))

    with pytest.raises(OSError):
        atomic_json_write(target, {"a": 2})

    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_binary_creates_readable_file(tmp_path):
    target = tmp_path / "data.bin"
    atomic_write(target, lambda f: f.write(b"hello"))
    assert target.read_bytes() == b"hello"


def test_atomic_write_binary_failure_leaves_original_intact(tmp_path, monkeypatch):
    target = tmp_path / "data.bin"
    target.write_bytes(b"original")

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))

    with pytest.raises(OSError):
        atomic_write(target, lambda f: f.write(b"new"))

    assert target.read_bytes() == b"original"
