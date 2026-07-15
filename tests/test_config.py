import os
import subprocess
import sys

from clawmarks import config


def test_repo_root_finds_pyproject():
    root = config.repo_root()
    assert (root / "pyproject.toml").exists()


def test_repo_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAWMARKS_ROOT", str(tmp_path))
    assert config.repo_root() == tmp_path


def test_derived_paths_under_state_dir():
    assert config.PROBE_DIR == config.STATE_DIR / "probe_uncanny"
    assert config.PROBE_STRENGTH_DIR == config.STATE_DIR / "probe_strength"


def test_expeditions_dir_is_repo_relative():
    assert config.EXPEDITIONS_DIR == config.ROOT / "expeditions"


def test_active_leg_file_is_state_dir_relative():
    assert config.ACTIVE_LEG_FILE == config.STATE_DIR / "active_leg.json"


def test_leg_dir_resolves_under_state_dir_expeditions():
    assert config.leg_dir("uncanny_frontier", "round1") == (
        config.STATE_DIR / "expeditions" / "uncanny_frontier" / "round1"
    )


def test_state_dir_defaults_to_xdg_state_home(tmp_path):
    env = dict(os.environ, XDG_STATE_HOME=str(tmp_path), PYTHONPATH="src")
    env.pop("CLAWMARKS_STATE_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", "from clawmarks.config import STATE_DIR; print(STATE_DIR)"],
        env=env, capture_output=True, text=True, cwd=str(config.ROOT), check=True,
    )
    assert result.stdout.strip() == str(tmp_path / "clawmarks")


def test_state_dir_env_override(tmp_path):
    env = dict(os.environ, CLAWMARKS_STATE_DIR=str(tmp_path), PYTHONPATH="src")
    result = subprocess.run(
        [sys.executable, "-c",
         "from clawmarks.config import leg_dir; print(leg_dir('e', 'l'))"],
        env=env, capture_output=True, text=True, cwd=str(config.ROOT), check=True,
    )
    assert result.stdout.strip() == str(tmp_path / "expeditions" / "e" / "l")
