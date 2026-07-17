import hashlib
import multiprocessing
import re
from datetime import datetime

import pytest

from clawmarks import durable_records


def test_canonical_json_digest_ignores_mapping_order():
    assert durable_records.sha256_json({"b": 2, "a": 1}) == durable_records.sha256_json(
        {"a": 1, "b": 2}
    )
    assert durable_records.canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_record_locks_sort_paths_before_flock(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        durable_records.fcntl, "flock", lambda fd, mode: seen.append((fd, mode))
    )
    with durable_records.record_locks(tmp_path, ["trial_b", "focus_a"]):
        assert (tmp_path / "focus_a.lock").exists()
        assert (tmp_path / "trial_b.lock").exists()
    assert len(seen) == 4


def test_file_locks_rejects_symlink_lock_path(tmp_path):
    target = tmp_path / "target.lock"
    target.touch()
    link = tmp_path / "link.lock"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        with durable_records.file_locks([link]):
            pass


def lock_in_order(
    paths, ready, start, first_flock_attempted, entered, entered_any, release, done
):
    original_flock = durable_records.fcntl.flock
    first_attempt = True

    def observed_flock(fd, mode):
        nonlocal first_attempt
        if mode == durable_records.fcntl.LOCK_EX and first_attempt:
            first_attempt = False
            first_flock_attempted.set()
        return original_flock(fd, mode)

    durable_records.fcntl.flock = observed_flock
    ready.set()
    if not start.wait(2):
        raise RuntimeError("start event timed out")
    with durable_records.file_locks(paths):
        entered.set()
        entered_any.set()
        if not release.wait(2):
            raise RuntimeError("release event timed out")
    done.set()


def test_file_locks_complete_when_processes_supply_opposite_orders(tmp_path):
    ready_a = multiprocessing.Event()
    ready_b = multiprocessing.Event()
    start = multiprocessing.Event()
    first_flock_a = multiprocessing.Event()
    first_flock_b = multiprocessing.Event()
    entered_a = multiprocessing.Event()
    entered_b = multiprocessing.Event()
    entered_any = multiprocessing.Event()
    release = multiprocessing.Event()
    done_a = multiprocessing.Event()
    done_b = multiprocessing.Event()
    paths = [tmp_path / "focus.lock", tmp_path / "leg.lock"]
    a = multiprocessing.Process(
        target=lock_in_order,
        args=(
            paths,
            ready_a,
            start,
            first_flock_a,
            entered_a,
            entered_any,
            release,
            done_a,
        ),
    )
    b = multiprocessing.Process(
        target=lock_in_order,
        args=(
            list(reversed(paths)),
            ready_b,
            start,
            first_flock_b,
            entered_b,
            entered_any,
            release,
            done_b,
        ),
    )
    try:
        a.start()
        b.start()
        assert ready_a.wait(2)
        assert ready_b.wait(2)
        start.set()
        assert first_flock_a.wait(2)
        assert first_flock_b.wait(2)
        assert entered_any.wait(2)
        assert entered_a.is_set() != entered_b.is_set()
        if entered_a.is_set():
            assert not entered_b.wait(0.2)
        else:
            assert not entered_a.wait(0.2)
        release.set()
        assert done_a.wait(2)
        assert done_b.wait(2)
        a.join(2)
        b.join(2)
        assert a.exitcode == b.exitcode == 0
    finally:
        for process in (a, b):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(2)


def _hold_lock(path, started, release, done):
    with durable_records.file_locks([path]):
        started.set()
        if not release.wait(2):
            raise RuntimeError("release event timed out")
    done.set()


def _try_enter(path, entered, done):
    with durable_records.file_locks([path]):
        entered.set()
    done.set()


def test_file_locks_serialize_access_across_processes(tmp_path):
    path = tmp_path / "serialize.lock"
    started = multiprocessing.Event()
    release = multiprocessing.Event()
    done_a = multiprocessing.Event()
    done_b = multiprocessing.Event()
    entered_b = multiprocessing.Event()

    holder = multiprocessing.Process(
        target=_hold_lock, args=(path, started, release, done_a)
    )
    waiter = multiprocessing.Process(
        target=_try_enter, args=(path, entered_b, done_b)
    )
    try:
        holder.start()
        assert started.wait(2)
        waiter.start()
        assert not entered_b.wait(0.2)
        release.set()
        assert entered_b.wait(2)
        assert done_a.wait(2)
        assert done_b.wait(2)
        holder.join(2)
        waiter.join(2)
        assert holder.exitcode == waiter.exitcode == 0
    finally:
        for process in (holder, waiter):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(2)


def test_file_locks_closes_fd_when_flock_fails(tmp_path, monkeypatch):
    path = tmp_path / "flock-failure.lock"
    opened = []
    closed = []
    real_open_lock_file = durable_records._open_lock_file
    real_close = durable_records.os.close

    def open_lock_file(lock_path):
        fd = real_open_lock_file(lock_path)
        opened.append(fd)
        return fd

    def close(fd):
        closed.append(fd)
        real_close(fd)

    def fail_flock(fd, mode):
        raise OSError("flock failed")

    monkeypatch.setattr(durable_records, "_open_lock_file", open_lock_file)
    monkeypatch.setattr(durable_records.os, "close", close)
    monkeypatch.setattr(durable_records.fcntl, "flock", fail_flock)

    with pytest.raises(OSError, match="flock failed"):
        with durable_records.file_locks([path]):
            pass

    assert opened
    assert opened[0] in closed


def test_validate_component_rejects_path_escape():
    with pytest.raises(ValueError):
        durable_records.validate_component("../other", "expedition")


def test_validate_component_rejects_trailing_newline():
    with pytest.raises(ValueError):
        durable_records.validate_component("focus_ok\n", "expedition")


def test_sha256_file_matches_content_digest(tmp_path):
    data = b"hello world"
    path = tmp_path / "sample.bin"
    path.write_bytes(data)
    assert durable_records.sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_new_id_format():
    focus_id = durable_records.new_id("focus")
    assert focus_id.startswith("focus_")
    hex_part = focus_id.split("_", 1)[1]
    assert re.fullmatch(r"[0-9a-f]{32}", hex_part)


def test_utc_now_is_iso_utc():
    parsed = datetime.fromisoformat(durable_records.utc_now())
    assert parsed.tzinfo is not None
    assert parsed.tzinfo.utcoffset(parsed).total_seconds() == 0
