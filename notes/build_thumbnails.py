"""
Generates small JPEG thumbnails for every image in scored_manifest.json. scan.html was
referencing the original full-resolution PNGs (1024x1024, 1.5-2MB each) directly for its grid
thumbnails, which is why the page was laggy: even with lazy loading, each thumbnail that
scrolled into view downloaded a multi-megabyte file to display it at ~160px. This writes
resized JPEGs into a thumbs/ subdirectory instead, idempotently (skips any thumbnail whose file
already exists), so build_scan_gallery.py can point the grid at those instead of the originals.

Run: python3 notes/build_thumbnails.py [sweep_dir]
"""
import os, sys, json
from PIL import Image

SWEEP_DIR = sys.argv[1] if len(sys.argv) > 1 else "/workspace/trent-with-smart-prompts/notes/uncanny_sweep"
THUMB_DIR = f"{SWEEP_DIR}/thumbs"
THUMB_SIZE = 220
QUALITY = 78

os.makedirs(THUMB_DIR, exist_ok=True)

with open(f"{SWEEP_DIR}/scored_manifest.json") as f:
    manifest = json.load(f)

made, skipped, failed = 0, 0, 0
for i, m in enumerate(manifest):
    src = m["file"]
    tag = m["tag"]
    dst = f"{THUMB_DIR}/{tag}.jpg"
    if os.path.exists(dst):
        skipped += 1
        continue
    try:
        img = Image.open(src).convert("RGB")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.BICUBIC)
        img.save(dst, format="JPEG", quality=QUALITY)
        made += 1
    except Exception as e:
        print(f"FAILED {tag}: {e}", flush=True)
        failed += 1
    if (i + 1) % 500 == 0:
        print(f"{i + 1}/{len(manifest)} processed (made={made} skipped={skipped} failed={failed})", flush=True)

print(f"DONE: made={made} skipped={skipped} failed={failed}, thumbs in {THUMB_DIR}", flush=True)
