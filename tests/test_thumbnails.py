import pytest
from PIL import Image

from clawmarks.build.thumbnails import generate_thumbnail


def test_generate_thumbnail_produces_a_valid_small_jpeg(tmp_path):
    src = tmp_path / "src.png"
    Image.new("RGB", (500, 500), color="red").save(src)
    dst = tmp_path / "thumb.jpg"

    generate_thumbnail(str(src), str(dst))

    img = Image.open(dst)
    assert img.format == "JPEG"
    assert max(img.size) <= 220


def test_generate_thumbnail_never_leaves_a_corrupt_file_at_dst_on_write_failure(tmp_path, monkeypatch):
    """Regression test: the old implementation wrote directly to dst_path via img.save(dst_path,
    ...), so a write failure partway through (disk full, process killed) could leave a
    truncated/corrupt JPEG at dst_path. Since thumbnails are never re-validated once dst_path
    exists, that corruption would be permanent. Writing to a temp file first and only
    os.replace-ing it into place on success means dst_path is never touched unless the write
    fully succeeded."""
    src = tmp_path / "src.png"
    Image.new("RGB", (256, 256), color="blue").save(src)
    dst = tmp_path / "thumb.jpg"

    def failing_save(self, fp, *a, **k):
        f = open(fp, "wb") if isinstance(fp, str) else fp
        f.write(b"partial-jpeg-bytes-then-crash")
        if isinstance(fp, str):
            f.close()
        raise IOError("simulated disk-full mid-write")

    monkeypatch.setattr(Image.Image, "save", failing_save)

    with pytest.raises(IOError):
        generate_thumbnail(str(src), str(dst))

    assert not dst.exists()
