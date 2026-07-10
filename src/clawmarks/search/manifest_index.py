"""Shared helpers for looking up scored_manifest.json entries by tag, and for building the
small per-image summary dict several tool pages need. Extracted out of build/elite_archive.py
so curation_server.py's ratings endpoints and build/preference_rank.py can reuse the exact same
summary shape instead of re-deriving it."""
import os


def index_by_tag(manifest):
    return {m["tag"]: m for m in manifest}


def item_summary(m, sweep_dir):
    thumb_path = os.path.join(str(sweep_dir), "thumbs", f"{m['tag']}.jpg")
    return {
        "tag": m["tag"], "prompt_name": m["prompt_name"], "prompt_type": m["prompt_type"],
        "faith": round(m["centroid_sim"], 4), "novelty": round(m["novelty"], 4),
        "strength": m["strength"], "cfg": m["cfg"],
        "thumb": (f"thumbs/{m['tag']}.jpg" if os.path.exists(thumb_path)
                  else os.path.basename(m["file"])),
        "file": os.path.basename(m["file"]),
    }
