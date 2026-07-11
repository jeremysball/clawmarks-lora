"""
Single persisted setting shared by archive.html's rendering and `clawmarks run allnight`'s
exploit-pool source, so flipping predicted-preference on or off happens in one place instead
of two independent controls (a query param and a CLI flag). See
docs/superpowers/specs/2026-07-10-preference-toggle-design.md.
"""
import json
import os

from clawmarks.config import PREFERENCE_SETTINGS_FILE


def load():
    """Returns {"use_predicted_preference": bool}. Missing file means the default, False."""
    if not os.path.exists(PREFERENCE_SETTINGS_FILE):
        return {"use_predicted_preference": False}
    with open(PREFERENCE_SETTINGS_FILE) as f:
        return json.load(f)


def save(enabled):
    tmp = f"{PREFERENCE_SETTINGS_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump({"use_predicted_preference": bool(enabled)}, f)
    os.replace(tmp, PREFERENCE_SETTINGS_FILE)
