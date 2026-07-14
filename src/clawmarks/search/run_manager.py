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
    """The lock info if a run is genuinely live (lock exists and its PID is alive), else
    None -- clearing a stale lock left behind by a crashed process rather than refusing every
    future launch forever."""
    info = read_lock()
    if info is None:
        return None
    if not is_process_alive(info["pid"]):
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


def launch_run(round_num, out_dir, api_key, popen_fn=subprocess.Popen, balance_fn=runpod_balance):
    if current_run() is not None:
        raise LaunchError("a search run is already in progress")

    balance = balance_fn(api_key)
    if balance < BALANCE_FLOOR_USD:
        raise LaunchError(f"balance ${balance:.2f} is below floor ${BALANCE_FLOOR_USD:.2f}")

    out_dir = Path(out_dir)
    if out_dir.exists():
        backup_dir = backup_out_dir(out_dir)
        verify_backup(out_dir, backup_dir)

    proc = popen_fn(
        [sys.executable, "-m", "clawmarks.search.driver", "--round", str(round_num)],
        start_new_session=True,
    )
    info = {"pid": proc.pid, "round": round_num, "started_at": time.time(), "out_dir": str(out_dir)}
    _write_lock(info)
    return info


def status():
    info = current_run()
    if info is None:
        return {"running": False}
    return {"running": True, **info}


def build_report(out_dir, favorites=None, current_balance=None):
    """The per-run report the notebook keeps asking for: novelty trajectory, plateau count,
    spend, pick rate by category, explore-vs-exploit split. Reads directly off disk (state
    file + scored_manifest.json) rather than through driver.load_state, since a report is a
    read-only summary and shouldn't require constructing a RoundConfig or pay load_state's
    resume-validation cost."""
    out_dir = Path(out_dir)
    favorites = favorites or {}

    state = {}
    for name in ("allnight_state.json", "allnight2_state.json"):
        state_file = out_dir / name
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            break

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


def stop_run(grace_s=STOP_GRACE_S, sleep_fn=time.sleep):
    info = current_run()
    if info is None:
        return {"running": False}

    pid = info["pid"]
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + grace_s
    while time.time() < deadline and is_process_alive(pid):
        sleep_fn(0.2)
    if is_process_alive(pid):
        os.kill(pid, signal.SIGKILL)
        deadline = time.time() + grace_s
        while time.time() < deadline and is_process_alive(pid):
            sleep_fn(0.2)
    _remove_lock()
    return {"running": False}
