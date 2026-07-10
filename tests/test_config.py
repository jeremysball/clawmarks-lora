# tests/test_config.py
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


def test_derived_paths_under_repo_root():
    root = config.repo_root()
    assert config.SWEEP_DIR == root / "notes" / "uncanny_sweep"
    assert config.SWEEP2_DIR == root / "notes" / "uncanny_sweep2"


def test_user_ratings_file_path():
    assert config.USER_RATINGS_FILE == config.SWEEP_DIR / "user_ratings.json"


def test_sweep_dir_env_override(tmp_path):
    # SWEEP_DIR is a module-level constant computed at import time, so the override has to be
    # exercised in a fresh subprocess rather than monkeypatched onto the already-imported module.
    env = dict(os.environ, CLAWMARKS_SWEEP_DIR=str(tmp_path), PYTHONPATH="src")
    result = subprocess.run(
        [sys.executable, "-c", "from clawmarks.config import SWEEP_DIR; print(SWEEP_DIR)"],
        env=env, capture_output=True, text=True, cwd=str(config.ROOT), check=True,
    )
    assert result.stdout.strip() == str(tmp_path)
