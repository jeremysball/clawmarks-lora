import json
import os
import subprocess
import sys
import time

import pytest

from clawmarks.search import run_manager


def test_is_process_alive_true_for_current_process():
    assert run_manager.is_process_alive(os.getpid()) is True


def test_is_process_alive_false_for_nonexistent_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert run_manager.is_process_alive(proc.pid) is False


def test_current_run_returns_none_when_no_lock_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "LOCK_FILE", tmp_path / ".searchrun.lock")
    assert run_manager.current_run() is None


def test_current_run_clears_stale_lock_for_dead_pid(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    lock_file.write_text(json.dumps({"pid": proc.pid, "expedition": "demo", "leg": "leg1", "started_at": 1.0, "out_dir": "x"}))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    assert run_manager.current_run() is None
    assert not lock_file.exists()


def test_current_run_returns_info_for_live_pid(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    info = {"pid": os.getpid(), "expedition": "demo", "leg": "leg1", "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    assert run_manager.current_run() == info


def test_backup_out_dir_mirrors_all_files(tmp_path):
    out_dir = tmp_path / "sweep"
    out_dir.mkdir()
    (out_dir / "a.json").write_text("1")
    (out_dir / "sub").mkdir()
    (out_dir / "sub" / "b.json").write_text("2")

    backup_dir = run_manager.backup_out_dir(out_dir)

    assert (backup_dir / "a.json").read_text() == "1"
    assert (backup_dir / "sub" / "b.json").read_text() == "2"


def test_verify_backup_passes_when_file_counts_match(tmp_path):
    out_dir = tmp_path / "sweep"
    out_dir.mkdir()
    (out_dir / "a.json").write_text("1")
    backup_dir = run_manager.backup_out_dir(out_dir)

    run_manager.verify_backup(out_dir, backup_dir)  # should not raise


def test_verify_backup_fails_closed_on_count_mismatch(tmp_path):
    out_dir = tmp_path / "sweep"
    out_dir.mkdir()
    (out_dir / "a.json").write_text("1")
    backup_dir = run_manager.backup_out_dir(out_dir)
    (backup_dir / "a.json").unlink()

    with pytest.raises(run_manager.LaunchError):
        run_manager.verify_backup(out_dir, backup_dir)


def test_launch_run_refuses_when_a_run_is_already_in_progress(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    info = {"pid": os.getpid(), "expedition": "demo", "leg": "leg1", "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    calls = []
    with pytest.raises(run_manager.LaunchError):
        run_manager.launch_run("demo", "leg1", tmp_path / "sweep", "fake-key",
                                popen_fn=lambda *a, **k: calls.append((a, k)),
                                balance_fn=lambda key: 100.0)
    assert calls == []


def test_launch_run_refuses_when_balance_below_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "LOCK_FILE", tmp_path / ".searchrun.lock")
    calls = []

    with pytest.raises(run_manager.LaunchError):
        run_manager.launch_run("demo", "leg1", tmp_path / "sweep", "fake-key",
                                popen_fn=lambda *a, **k: calls.append((a, k)),
                                balance_fn=lambda key: 0.01)
    assert calls == []


def test_launch_run_backs_up_verifies_and_writes_lock(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)
    out_dir = tmp_path / "sweep"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text("[]")

    class FakeProc:
        pid = 12345

    info = run_manager.launch_run("demo", "leg1", out_dir, "fake-key",
                                   popen_fn=lambda *a, **k: FakeProc(),
                                   balance_fn=lambda key: 100.0)

    assert info["pid"] == 12345
    assert info["expedition"] == "demo"
    assert info["leg"] == "leg1"
    assert info["out_dir"] == str(out_dir)
    assert json.loads(lock_file.read_text()) == info
    backups = list(tmp_path.glob("sweep_backup_*"))
    assert len(backups) == 1
    assert (backups[0] / "manifest.json").read_text() == "[]"


def test_launch_run_skips_backup_when_out_dir_does_not_exist_yet(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)
    out_dir = tmp_path / "sweep_not_yet_created"

    class FakeProc:
        pid = 99999

    info = run_manager.launch_run("demo", "leg1", out_dir, "fake-key",
                                   popen_fn=lambda *a, **k: FakeProc(),
                                   balance_fn=lambda key: 100.0)

    assert info["pid"] == 99999
    assert list(tmp_path.glob("sweep_not_yet_created_backup_*")) == []


def test_status_reports_not_running_with_no_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "LOCK_FILE", tmp_path / ".searchrun.lock")
    assert run_manager.status() == {"running": False}


def test_status_reports_running_with_live_lock(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    info = {"pid": os.getpid(), "expedition": "demo", "leg": "leg2", "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    result = run_manager.status()
    assert result["running"] is True
    assert result["expedition"] == "demo"
    assert result["leg"] == "leg2"


def test_stop_run_is_noop_when_nothing_running(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "LOCK_FILE", tmp_path / ".searchrun.lock")
    assert run_manager.stop_run() == {"running": False}


def test_stop_run_rejects_a_run_that_does_not_match_confirmation(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    lock_file.write_text(json.dumps({"pid": 12345, "start_time_ticks": 999}))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)
    monkeypatch.setattr(run_manager, "is_process_alive", lambda pid: True)
    monkeypatch.setattr(run_manager, "_process_start_time", lambda pid: 999)
    signaled = []
    monkeypatch.setattr(run_manager, "_signal_run", lambda pid, sig: signaled.append((pid, sig)))

    result = run_manager.stop_run(pid=67890, start_time_ticks=999)

    assert result == {"running": True, "error": "run changed since confirmation"}
    assert signaled == []


def test_stop_run_sends_sigterm_and_removes_lock(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    time.sleep(0.2)
    info = {"pid": proc.pid, "expedition": "demo", "leg": "leg1", "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    result = run_manager.stop_run(grace_s=5)

    assert result == {"running": False}
    assert not lock_file.exists()
    proc.wait(timeout=5)
    assert proc.returncode is not None


def test_stop_run_kills_the_whole_process_group_not_just_the_leader(tmp_path, monkeypatch):
    """launch_run starts the driver with start_new_session=True so it can outlive a request
    thread; if the driver later shells out to a slow child (e.g. an opencode subprocess call),
    stop must reach that child too, not just the driver's own pid."""
    lock_file = tmp_path / ".searchrun.lock"
    proc = subprocess.Popen([
        sys.executable, "-c",
        "import subprocess, time, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "print(child.pid, flush=True); "
        "time.sleep(30)",
    ], start_new_session=True, stdout=subprocess.PIPE, text=True)
    child_pid = int(proc.stdout.readline().strip())
    info = {"pid": proc.pid, "expedition": "demo", "leg": "leg1", "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    result = run_manager.stop_run(grace_s=5)

    assert result == {"running": False}
    proc.wait(timeout=5)
    deadline = time.time() + 5
    while time.time() < deadline and run_manager.is_process_alive(child_pid):
        time.sleep(0.1)
    assert not run_manager.is_process_alive(child_pid)


def test_build_report_with_no_state_or_manifest_yet(tmp_path):
    out_dir = tmp_path / "sweep"
    out_dir.mkdir()

    report = run_manager.build_report(out_dir)

    assert report["novelty_trajectory"] == []
    assert report["plateau_count"] == 0
    assert report["total_images"] == 0
    assert report["pick_rate_by_category"] == {}
    assert report["explore_exploit_split"] == {}


def test_build_report_reads_novelty_trajectory_and_plateau_count_from_state(tmp_path):
    out_dir = tmp_path / "sweep"
    out_dir.mkdir()
    state = {
        "generation": 3, "stage": 0, "plateau_count": 2,
        "novelty_history": [0.5, 0.6, 0.61], "gpt55_subjects": [],
        "start_balance": 10.0, "start_time": 1.0,
    }
    (out_dir / "allnight_state.json").write_text(json.dumps(state))

    report = run_manager.build_report(out_dir)

    assert report["novelty_trajectory"] == [0.5, 0.6, 0.61]
    assert report["plateau_count"] == 2
    assert report["generation"] == 3
    assert report["start_balance"] == 10.0


def test_build_report_reads_allnight_state_file_name(tmp_path):
    out_dir = tmp_path / "sweep"
    out_dir.mkdir()
    state = {
        "generation": 1, "stage": 0, "plateau_count": 0,
        "novelty_history": [0.4], "gpt55_subjects": [],
        "start_balance": 1.0, "start_time": 1.0,
    }
    (out_dir / "allnight_state.json").write_text(json.dumps(state))

    report = run_manager.build_report(out_dir)

    assert report["novelty_trajectory"] == [0.4]


def test_build_report_computes_pick_rate_and_explore_exploit_split_by_category(tmp_path):
    out_dir = tmp_path / "sweep"
    out_dir.mkdir()
    manifest = [
        {"tag": "gen1_explore_0", "category": "r2_explore"},
        {"tag": "gen1_explore_1", "category": "r2_explore"},
        {"tag": "gen1_exploit_0", "category": "r2_exploit"},
    ]
    (out_dir / "scored_manifest.json").write_text(json.dumps(manifest))
    favorites = {"gen1_explore_0": {}}

    report = run_manager.build_report(out_dir, favorites=favorites)

    assert report["total_images"] == 3
    assert report["explore_exploit_split"] == {"r2_explore": 2, "r2_exploit": 1}
    assert report["pick_rate_by_category"]["r2_explore"] == pytest.approx(0.5)
    assert report["pick_rate_by_category"]["r2_exploit"] == pytest.approx(0.0)


def test_build_report_computes_spend_when_current_balance_given(tmp_path):
    out_dir = tmp_path / "sweep"
    out_dir.mkdir()
    state = {
        "generation": 1, "stage": 0, "plateau_count": 0,
        "novelty_history": [0.1], "gpt55_subjects": [],
        "start_balance": 10.0, "start_time": 1.0,
    }
    (out_dir / "allnight_state.json").write_text(json.dumps(state))

    report = run_manager.build_report(out_dir, current_balance=7.5)

    assert report["spend"] == pytest.approx(2.5)


def test_stop_run_sigkills_after_grace_period_if_process_ignores_sigterm(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    proc = subprocess.Popen([sys.executable, "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"],
        start_new_session=True)
    time.sleep(0.2)
    info = {"pid": proc.pid, "expedition": "demo", "leg": "leg1", "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    result = run_manager.stop_run(grace_s=1)

    assert result == {"running": False}
    proc.wait(timeout=5)
    assert proc.returncode is not None


def test_launch_run_is_race_free_under_concurrent_calls(tmp_path, monkeypatch):
    """The production server is threaded, so two POSTs can race here. launch_run must let only
    one of two near-simultaneous callers through, not both -- two concurrent driver.py processes
    writing to the same out_dir would corrupt it."""
    import threading

    lock_file = tmp_path / ".searchrun.lock"
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)
    out_dir = tmp_path / "sweep_not_yet_created"

    results = []

    class FakeProc:
        pid = os.getpid()

    def popen_fn(*a, **k):
        return FakeProc()

    def attempt():
        try:
            results.append(("ok", run_manager.launch_run(
                "demo", "leg1", out_dir, "fake-key", popen_fn=popen_fn, balance_fn=lambda key: 100.0)))
        except run_manager.LaunchError as e:
            results.append(("error", str(e)))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    oks = [r for r in results if r[0] == "ok"]
    errors = [r for r in results if r[0] == "error"]
    assert len(oks) == 1
    assert len(errors) == 1
    assert "already in progress" in errors[0][1]


def test_launch_run_reaps_the_child_and_clears_the_lock_if_writing_the_lock_fails(tmp_path, monkeypatch):
    """If the lock write fails after the subprocess is already spawned, the child must not be
    left running with no lock file -- that would silently break the one-run-at-a-time guarantee
    for every launch after it. Uses a real spawned process (not a pid-recording mock) so the
    assertion proves the process actually dies, not just that .kill() was called."""
    lock_file = tmp_path / ".searchrun.lock"
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)
    out_dir = tmp_path / "sweep_not_yet_created"

    spawned = []

    def popen_fn(*a, **k):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        spawned.append(proc)
        return proc

    def broken_dump(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(run_manager.json, "dump", broken_dump)

    with pytest.raises(OSError):
        run_manager.launch_run("demo", "leg1", out_dir, "fake-key",
                                popen_fn=popen_fn, balance_fn=lambda key: 100.0)

    assert not lock_file.exists()
    pid = spawned[0].pid
    deadline = time.time() + 5
    while time.time() < deadline and run_manager.is_process_alive(pid):
        time.sleep(0.1)
    assert not run_manager.is_process_alive(pid)


def test_current_run_treats_a_pid_reused_by_an_unrelated_process_as_stale(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    info = {
        "pid": os.getpid(), "expedition": "demo", "leg": "leg1", "started_at": 1.0, "out_dir": "x",
        "start_time_ticks": -1,  # cannot match this process's real start time
    }
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    assert run_manager.current_run() is None
    assert not lock_file.exists()


def test_stop_run_reaps_the_process_promptly_instead_of_blocking_the_full_grace_period(tmp_path, monkeypatch):
    """A process that has already exited but was never wait()'d by its parent is a zombie:
    os.kill(pid, 0) still succeeds on it, so without reaping, is_process_alive reports it as
    alive and stop_run spins its full SIGTERM-then-SIGKILL grace periods for nothing. run_manager
    (this process) is the real parent here, so it can and should reap it."""
    lock_file = tmp_path / ".searchrun.lock"
    # Deliberately never call proc.poll()/.wait() here -- either would reap it itself (Python's
    # subprocess module wait()s under the hood) and defeat the point of this test: the process
    # must exit and become a zombie *before* run_manager ever touches it.
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    time.sleep(0.5)
    info = {"pid": proc.pid, "expedition": "demo", "leg": "leg1", "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    start = time.time()
    result = run_manager.stop_run(grace_s=10)
    elapsed = time.time() - start

    assert result == {"running": False}
    assert elapsed < 3


def test_launch_run_records_pid_start_time_for_reuse_detection(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)
    out_dir = tmp_path / "sweep_not_yet_created"

    class FakeProc:
        pid = os.getpid()

    info = run_manager.launch_run("demo", "leg1", out_dir, "fake-key",
                                   popen_fn=lambda *a, **k: FakeProc(),
                                   balance_fn=lambda key: 100.0)

    assert info["start_time_ticks"] == run_manager._process_start_time(os.getpid())
    assert info["start_time_ticks"] is not None
