import json
from pathlib import Path


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save(path: Path, seeds: dict) -> None:
    with open(path, "w") as f:
        json.dump(seeds, f, indent=1)


def merge(existing: dict, new_subjects: list, source: str, created_at: str) -> tuple:
    seeds = dict(existing)
    existing_lower = {s.lower().strip() for s in seeds}
    added = []
    for s in new_subjects:
        s = str(s).strip()
        if s and s.lower() not in existing_lower:
            seeds[s] = {"source": source, "created_at": created_at}
            existing_lower.add(s.lower())
            added.append(s)
    return seeds, added
