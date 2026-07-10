import threading
from http.server import HTTPServer
import urllib.request

import pytest

from clawmarks import curation_server as cs


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
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
