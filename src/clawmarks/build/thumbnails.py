"""
Generates small JPEG thumbnails for every image in scored_manifest.json. scan.html was
referencing the original full-resolution PNGs (1024x1024, 1.5-2MB each) directly for its grid
thumbnails, which is why the page was laggy: even with lazy loading, each thumbnail that
scrolled into view downloaded a multi-megabyte file to display it at ~160px. This writes
resized JPEGs into a thumbs/ subdirectory instead, idempotently (skips any thumbnail whose file
already exists), so build_scan_gallery.py can point the grid at those instead of the originals.

Run: python3 -m clawmarks.build.thumbnails [sweep_dir]
"""
import argparse
import os, json
from PIL import Image

from clawmarks.config import SWEEP_DIR

THUMB_SIZE = 220
QUALITY = 78


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir", nargs="?", default=str(SWEEP_DIR),
                        help=f"sweep directory containing scored_manifest.json (default: {SWEEP_DIR})")
    args = parser.parse_args(argv)
    sweep_dir = args.sweep_dir
    thumb_dir = f"{sweep_dir}/thumbs"
    os.makedirs(thumb_dir, exist_ok=True)

    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    made, skipped, failed = 0, 0, 0
    for i, m in enumerate(manifest):
        src = m["file"]
        tag = m["tag"]
        dst = f"{thumb_dir}/{tag}.jpg"
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

    print(f"DONE: made={made} skipped={skipped} failed={failed}, thumbs in {thumb_dir}", flush=True)


if __name__ == "__main__":
    main()
