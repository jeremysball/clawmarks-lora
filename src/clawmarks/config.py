import os
from pathlib import Path


def repo_root() -> Path:
    env_override = os.environ.get("CLAWMARKS_ROOT")
    if env_override:
        return Path(env_override)
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(
        "Could not find pyproject.toml by walking up from clawmarks/config.py. "
        "If clawmarks is installed outside its source checkout, set CLAWMARKS_ROOT "
        "to the repo root explicitly."
    )


ROOT = repo_root()
NOTES_DIR = ROOT / "notes"
SWEEP_DIR = NOTES_DIR / "uncanny_sweep"
SWEEP2_DIR = NOTES_DIR / "uncanny_sweep2"
PROBE_DIR = NOTES_DIR / "probe_uncanny"
PROBE_STRENGTH_DIR = NOTES_DIR / "probe_strength"
SEEDS_FILE = SWEEP_DIR / "candidate_seeds.json"
USER_PICKS_FILE = SWEEP_DIR / "user_picks.json"
USER_RATINGS_FILE = SWEEP_DIR / "user_ratings.json"
