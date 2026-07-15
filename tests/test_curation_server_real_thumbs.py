import threading
from http.server import HTTPServer

from PIL import Image
import pytest
import urllib.error
import urllib.request

from clawmarks import curation_server as cs


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    real_dir = tmp_path / "real_images"
    real_dir.mkdir()
    monkeypatch.setattr(cs, "_active_out_dir", lambda: tmp_path)
    monkeypatch.setattr(cs, "REAL_DIR", str(real_dir))
    (tmp_path / "scored_manifest.json").write_text("[]")
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path, real_dir
    server.shutdown()
    thread.join(timeout=2)


def test_real_thumb_generated_on_first_request(running_server):
    server, tmp_path, real_dir = running_server
    Image.new("RGB", (500, 500), color="blue").save(real_dir / "cat_001.jpg")
    port = server.server_address[1]

    assert not (tmp_path / "real_thumbs" / "cat_001.jpg").exists()
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/real_thumbs/cat_001.jpg") as resp:
        assert resp.status == 200
    assert (tmp_path / "real_thumbs" / "cat_001.jpg").exists()
    img = Image.open(tmp_path / "real_thumbs" / "cat_001.jpg")
    assert max(img.size) <= 220


def test_real_thumb_does_not_write_into_real_dir(running_server):
    server, tmp_path, real_dir = running_server
    Image.new("RGB", (500, 500), color="blue").save(real_dir / "cat_002.jpg")
    port = server.server_address[1]

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/real_thumbs/cat_002.jpg") as resp:
        resp.read()

    assert sorted(p.name for p in real_dir.iterdir()) == ["cat_002.jpg"]


def test_real_thumb_reused_on_second_request(running_server, monkeypatch):
    server, tmp_path, real_dir = running_server
    Image.new("RGB", (500, 500), color="blue").save(real_dir / "cat_003.jpg")
    port = server.server_address[1]

    calls = []
    real_generate = cs.generate_thumbnail

    def counting_generate(src, dst):
        calls.append((src, dst))
        return real_generate(src, dst)

    monkeypatch.setattr(cs, "generate_thumbnail", counting_generate)

    for _ in range(2):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/real_thumbs/cat_003.jpg") as resp:
            resp.read()

    assert len(calls) == 1


def test_real_thumb_404s_for_missing_real_image(running_server):
    server, tmp_path, real_dir = running_server
    port = server.server_address[1]

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/real_thumbs/does_not_exist.jpg")
    assert exc_info.value.code == 404


def test_real_thumb_route_serves_only_basenames_from_real_dir(running_server):
    server, tmp_path, real_dir = running_server
    port = server.server_address[1]

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/real_thumbs/..%2F..%2F..%2Fetc%2Fpasswd")
    assert exc_info.value.code == 404
