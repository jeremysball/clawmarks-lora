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
    lock_file.write_text(json.dumps({"pid": proc.pid, "round": 1, "started_at": 1.0, "out_dir": "x"}))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    assert run_manager.current_run() is None
    assert not lock_file.exists()


def test_current_run_returns_info_for_live_pid(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    info = {"pid": os.getpid(), "round": 1, "started_at": 1.0, "out_dir": "x"}
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
    info = {"pid": os.getpid(), "round": 1, "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    calls = []
    with pytest.raises(run_manager.LaunchError):
        run_manager.launch_run(1, tmp_path / "sweep", "fake-key",
                                popen_fn=lambda *a, **k: calls.append((a, k)),
                                balance_fn=lambda key: 100.0)
    assert calls == []


def test_launch_run_refuses_when_balance_below_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "LOCK_FILE", tmp_path / ".searchrun.lock")
    calls = []

    with pytest.raises(run_manager.LaunchError):
        run_manager.launch_run(1, tmp_path / "sweep", "fake-key",
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

    info = run_manager.launch_run(1, out_dir, "fake-key",
                                   popen_fn=lambda *a, **k: FakeProc(),
                                   balance_fn=lambda key: 100.0)

    assert info["pid"] == 12345
    assert info["round"] == 1
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

    info = run_manager.launch_run(1, out_dir, "fake-key",
                                   popen_fn=lambda *a, **k: FakeProc(),
                                   balance_fn=lambda key: 100.0)

    assert info["pid"] == 99999
    assert list(tmp_path.glob("sweep_not_yet_created_backup_*")) == []


def test_status_reports_not_running_with_no_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "LOCK_FILE", tmp_path / ".searchrun.lock")
    assert run_manager.status() == {"running": False}


def test_status_reports_running_with_live_lock(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    info = {"pid": os.getpid(), "round": 2, "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    result = run_manager.status()
    assert result["running"] is True
    assert result["round"] == 2


def test_stop_run_is_noop_when_nothing_running(tmp_path, monkeypatch):
    monkeypatch.setattr(run_manager, "LOCK_FILE", tmp_path / ".searchrun.lock")
    assert run_manager.stop_run() == {"running": False}


def test_stop_run_sends_sigterm_and_removes_lock(tmp_path, monkeypatch):
    lock_file = tmp_path / ".searchrun.lock"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    time.sleep(0.2)
    info = {"pid": proc.pid, "round": 1, "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    result = run_manager.stop_run(grace_s=5)

    assert result == {"running": False}
    assert not lock_file.exists()
    proc.wait(timeout=5)
    assert proc.returncode is not None


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


def test_build_report_reads_round2_state_file_name(tmp_path):
    out_dir = tmp_path / "sweep2"
    out_dir.mkdir()
    state = {
        "generation": 1, "stage": 0, "plateau_count": 0,
        "novelty_history": [0.4], "gpt55_subjects": [],
        "start_balance": 1.0, "start_time": 1.0,
    }
    (out_dir / "allnight2_state.json").write_text(json.dumps(state))

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
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"])
    time.sleep(0.2)
    info = {"pid": proc.pid, "round": 1, "started_at": 1.0, "out_dir": "x"}
    lock_file.write_text(json.dumps(info))
    monkeypatch.setattr(run_manager, "LOCK_FILE", lock_file)

    result = run_manager.stop_run(grace_s=1)

    assert result == {"running": False}
    proc.wait(timeout=5)
    assert proc.returncode is not None
