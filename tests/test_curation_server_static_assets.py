import threading
from http.server import HTTPServer
import urllib.error
import urllib.request

import pytest

from clawmarks import curation_server as cs


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_active_out_dir", lambda: tmp_path)
    (tmp_path / "scored_manifest.json").write_text("[]")
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=2)


def test_lightbox_js_served_without_being_written_to_disk(running_server, tmp_path):
    port = running_server.server_address[1]
    assert not (tmp_path / "lightbox.js").exists()
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/lightbox.js") as resp:
        body = resp.read().decode()
        assert resp.headers["Content-Type"] == "application/javascript"
    assert "window.Lightbox" in body
    assert not (tmp_path / "lightbox.js").exists()


def test_infotip_js_served_without_being_written_to_disk(running_server, tmp_path):
    port = running_server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/infotip.js") as resp:
        body = resp.read().decode()
        assert resp.headers["Content-Type"] == "application/javascript"
    assert "infobtn" in body
    assert not (tmp_path / "infotip.js").exists()


def test_favicon_served(running_server):
    port = running_server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/favicon.ico") as resp:
        assert resp.headers["Content-Type"] == "image/png"
        assert len(resp.read()) == len(cs._FAVICON_PNG)


def test_real_route_serves_only_basenames_from_real_dir(running_server, tmp_path, monkeypatch):
    real_dir = tmp_path / "real_images"
    real_dir.mkdir()
    (real_dir / "cat_001.jpg").write_bytes(b"fake-jpeg-bytes")
    monkeypatch.setattr(cs, "REAL_DIR", str(real_dir))
    port = running_server.server_address[1]

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/real/cat_001.jpg") as resp:
        assert resp.headers["Content-Type"] == "image/jpeg"
        assert resp.read() == b"fake-jpeg-bytes"

    # Path traversal in the requested name can only ever resolve to a REAL_DIR child, since the
    # handler takes basename() of it; an escape attempt 404s instead of reading outside REAL_DIR.
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/real/..%2F..%2F..%2Fetc%2Fpasswd")
    assert exc_info.value.code == 404
