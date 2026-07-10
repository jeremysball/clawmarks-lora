# tests/test_config.py
import os
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
