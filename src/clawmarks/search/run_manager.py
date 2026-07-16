"""Launch, monitor, and stop an overnight search run (search/driver.py) from the curation
server UI, with the safety rails from docs/superpowers/specs/2026-07-12-overnight-search-launch-design.md:
backup+verify before every launch (fail closed), a balance-floor check, a one-run-at-a-time
lock file, a detached subprocess, and SIGTERM-then-SIGKILL stop."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from clawmarks.config import NOTES_DIR
from clawmarks.runpod_client import runpod_balance

LOCK_FILE = NOTES_DIR / ".searchrun.lock"
BALANCE_FLOOR_USD = 0.05
STOP_GRACE_S = 10


class LaunchError(Exception):
    pass


def is_process_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _process_start_time(pid):
    """Linux process start time in clock ticks since boot (field 22 of /proc/<pid>/stat), used
    to tell a still-live launched driver apart from an unrelated process that later reused the
    same PID. Best-effort: returns None if /proc is unavailable or the process is gone, in which
    case callers should skip the reuse check rather than treat every lock as reused."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
    except OSError:
        return None
    # comm (arg 2) is parenthesized and may itself contain spaces/parens, so split on the last ')'.
    fields = stat[stat.rfind(")") + 1:].split()
    return int(fields[19])  # field 22 overall, index 19 after dropping pid+comm+state


def read_lock():
    if not LOCK_FILE.exists():
        return None
    try:
        with open(LOCK_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _write_lock(info):
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        json.dump(info, f)


def _remove_lock():
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass


def current_run():
    """The lock info if a run is genuinely live (lock exists, its PID is alive, and -- when
    recorded -- that PID's start time still matches the one launch_run observed), else None --
    clearing a stale lock left behind by a crashed process, or one whose PID has since been
    reused by an unrelated process, rather than refusing every future launch forever."""
    info = read_lock()
    if info is None:
        return None
    if not is_process_alive(info["pid"]):
        _remove_lock()
        return None
    recorded = info.get("start_time_ticks")
    if recorded is not None and _process_start_time(info["pid"]) != recorded:
        _remove_lock()
        return None
    return info


def backup_out_dir(out_dir):
    out_dir = Path(out_dir)
    backup_dir = out_dir.parent / f"{out_dir.name}_backup_{int(time.time())}"
    shutil.copytree(out_dir, backup_dir)
    return backup_dir


def verify_backup(out_dir, backup_dir):
    out_dir = Path(out_dir)
    backup_dir = Path(backup_dir)
    orig_count = sum(1 for p in out_dir.rglob("*") if p.is_file())
    backup_count = sum(1 for p in backup_dir.rglob("*") if p.is_file())
    if orig_count != backup_count:
        raise LaunchError(
            f"backup verification failed: {out_dir} has {orig_count} files but "
            f"backup {backup_dir} has {backup_count}"
        )


def launch_run(expedition, leg, out_dir, api_key, popen_fn=subprocess.Popen, balance_fn=runpod_balance):
    # current_run() also clears a stale lock (dead or PID-reused) left by a prior crash, so a
    # fresh atomic create below isn't blocked by garbage from a run that never really finished.
    current_run()

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        # O_CREAT|O_EXCL is an atomic claim: under concurrent callers (the production server is
        # threaded) exactly one open() succeeds and the rest see FileExistsError, closing the
        # check-then-write race that a separate current_run()-then-_write_lock() would leave open.
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise LaunchError("a search run is already in progress")

    proc = None
    try:
        balance = balance_fn(api_key)
        if balance < BALANCE_FLOOR_USD:
            raise LaunchError(f"balance ${balance:.2f} is below floor ${BALANCE_FLOOR_USD:.2f}")

        out_dir = Path(out_dir)
        if out_dir.exists():
            backup_dir = backup_out_dir(out_dir)
            verify_backup(out_dir, backup_dir)

        proc = popen_fn(
            [sys.executable, "-m", "clawmarks.search.driver",
             "--expedition", expedition, "--leg", leg],
            start_new_session=True,
        )
        info = {
            "pid": proc.pid, "expedition": expedition, "leg": leg,
            "started_at": time.time(), "out_dir": str(out_dir),
            "start_time_ticks": _process_start_time(proc.pid),
        }
        # os.fdopen takes ownership of fd immediately: it will be closed when f is (even via
        # this function's own except below), so drop our own claim on it right away to avoid a
        # double-close.
        f = os.fdopen(fd, "w")
        fd = None
        with f:
            json.dump(info, f)
        return info
    except BaseException:
        if fd is not None:
            os.close(fd)
        _remove_lock()
        if proc is not None:
            # The subprocess was already spawned when the lock write failed; without this it
            # would keep running with no lock file, silently breaking the one-run-at-a-time
            # guarantee for every launch after it. wait() (not just kill()) so it's actually
            # reaped rather than left as a zombie that is_process_alive still reports as alive.
            proc.kill()
            proc.wait()
        raise


def status():
    info = current_run()
    if info is None:
        return {"running": False}
    return {"running": True, **info}


def build_report(out_dir, favorites=None, current_balance=None):
    """The per-run report the notebook keeps asking for: novelty trajectory, plateau count,
    spend, pick rate by category, explore-vs-exploit split. Reads directly off disk (state
    file + scored_manifest.json) rather than through driver.load_state, since a report is a
    read-only summary and shouldn't require constructing leg configuration or pay load_state's
    resume-validation cost."""
    out_dir = Path(out_dir)
    favorites = favorites or {}

    state = {}
    state_file = out_dir / "allnight_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    manifest = []
    manifest_file = out_dir / "scored_manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)

    by_category = {}
    for m in manifest:
        cat = m.get("category", "unknown")
        counts = by_category.setdefault(cat, {"count": 0, "picked": 0})
        counts["count"] += 1
        if m.get("tag") in favorites:
            counts["picked"] += 1

    report = {
        "novelty_trajectory": state.get("novelty_history", []),
        "plateau_count": state.get("plateau_count", 0),
        "generation": state.get("generation", 0),
        "start_balance": state.get("start_balance"),
        "total_images": len(manifest),
        "explore_exploit_split": {cat: v["count"] for cat, v in by_category.items()},
        "pick_rate_by_category": {
            cat: (v["picked"] / v["count"] if v["count"] else 0.0)
            for cat, v in by_category.items()
        },
    }
    if current_balance is not None and state.get("start_balance") is not None:
        report["spend"] = state["start_balance"] - current_balance
    return report


def _reap_if_exited(pid):
    """Best-effort zombie reap. launch_run's Popen object goes out of scope right after
    spawning (only the pid is kept in the lock), so nothing else in this process ever wait()s on
    the driver; once it exits it sits as a zombie, and os.kill(pid, 0) -- what is_process_alive
    checks -- reports zombies as alive. Without this, stop_run's SIGTERM/SIGKILL grace loops
    would spin their full duration on every stop, even for a driver that exited immediately.
    ChildProcessError means pid isn't (or is no longer) our child; safe to ignore."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


def _signal_run(pid, sig):
    """launch_run starts the driver with start_new_session=True, making it a process-group
    leader (pgid == pid at spawn time), so killpg reaches any child it has shelled out to (e.g.
    driver.py's blocking opencode subprocess call) as well as the driver itself. Falls back to
    signaling just the pid if the group is already gone."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except ProcessLookupError:
        pass


def stop_run(grace_s=STOP_GRACE_S, sleep_fn=time.sleep, pid=None, start_time_ticks=None):
    info = current_run()
    if info is None:
        return {"running": False}

    # Reject a confirmation for a different run instead of signaling the live one.
    if pid is not None and pid != info["pid"]:
        return {"running": True, "error": "run changed since confirmation"}
    recorded = info.get("start_time_ticks")
    if start_time_ticks is not None and recorded is not None and start_time_ticks != recorded:
        return {"running": True, "error": "run changed since confirmation"}

    pid = info["pid"]
    _signal_run(pid, signal.SIGTERM)
    _reap_if_exited(pid)
    deadline = time.time() + grace_s
    while time.time() < deadline and is_process_alive(pid):
        sleep_fn(0.2)
        _reap_if_exited(pid)
    if is_process_alive(pid):
        _signal_run(pid, signal.SIGKILL)
        _reap_if_exited(pid)
        deadline = time.time() + grace_s
        while time.time() < deadline and is_process_alive(pid):
            sleep_fn(0.2)
            _reap_if_exited(pid)
    _remove_lock()
    return {"running": False}
