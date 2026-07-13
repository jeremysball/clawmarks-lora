"""
Resizes a single source image into a small JPEG thumbnail. Used by curation_server.py to
lazily generate notes/<sweep>/thumbs/<tag>.jpg on first request instead of pre-generating
every thumbnail in a batch step; once made, a thumbnail never goes stale (its source image
doesn't change after generation), so there's nothing to invalidate.
"""
import base64
import os
import threading
from io import BytesIO

from PIL import Image

THUMB_SIZE = 220
QUALITY = 78


def generate_thumbnail(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    img = Image.open(src_path).convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.BICUBIC)
    tmp_path = f"{dst_path}.tmp-{os.getpid()}-{threading.get_ident()}"
    img.save(tmp_path, format="JPEG", quality=QUALITY)
    os.replace(tmp_path, dst_path)


def thumb_data_uri(path, size=192):
    """Inline base64 JPEG data URI, used by search/driver.py's per-round offline gallery.html
    archive (a static file written to the sweep dir, distinct from curation_server.py's live
    rendering, which uses generate_thumbnail's on-disk cache instead)."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
