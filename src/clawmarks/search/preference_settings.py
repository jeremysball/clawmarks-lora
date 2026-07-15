"""
Single persisted setting shared by archive.html's rendering and `clawmarks run allnight`'s
exploit-pool source, so flipping predicted-preference on or off happens in one place instead
of two independent controls (a query param and a CLI flag). See
docs/superpowers/specs/2026-07-10-preference-toggle-design.md.

Takes an explicit out_dir (the active leg's directory) rather than a fixed module constant,
since there is no longer one process-wide sweep directory.
"""
import json
import os


def load(out_dir):
    """Returns {"use_predicted_preference": bool}. Missing file means the default, False."""
    path = out_dir / "preference_settings.json"
    if not os.path.exists(path):
        return {"use_predicted_preference": False}
    with open(path) as f:
        return json.load(f)


def save(enabled, out_dir):
    path = out_dir / "preference_settings.json"
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"use_predicted_preference": bool(enabled)}, f)
    os.replace(tmp, path)
