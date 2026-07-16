import json
import threading
from http.server import HTTPServer
import urllib.error
import urllib.request

import pytest

from clawmarks import curation_server as cs


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_active_out_dir", lambda: tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=2)


def _fetch_error_page(port, path):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}{path}")
        raise AssertionError("expected a 500 response")
    except urllib.error.HTTPError as e:
        assert e.code == 500
        return e.read().decode()


def test_error_page_missing_manifest_hint(running_server, monkeypatch):
    """A FileNotFoundError for scored_manifest.json itself should point the user at picking or
    launching a leg, not at the stale-image-path advice (which is the wrong fix for this case)."""
    server = running_server
    port = server.server_address[1]

    def raise_missing_manifest():
        raise FileNotFoundError("[Errno 2] No such file or directory: '/x/scored_manifest.json'")

    monkeypatch.setattr(cs, "_get_map_data", raise_missing_manifest)

    body = _fetch_error_page(port, "/map.html")
    assert "no scored manifest yet" in body
    assert "Pick a leg" in body
    assert "old absolute path" not in body


def test_error_page_stale_image_path_hint(running_server, monkeypatch):
    """A FileNotFoundError for an image file behind a valid manifest should get the
    re-point-the-paths advice, not the missing-manifest advice."""
    server = running_server
    port = server.server_address[1]

    def raise_missing_image():
        raise FileNotFoundError("[Errno 2] No such file or directory: '/x/thumbs/gen0_a.jpg'")

    monkeypatch.setattr(cs, "_get_map_data", raise_missing_image)

    body = _fetch_error_page(port, "/map.html")
    assert "old absolute path" in body
    assert "no scored manifest yet" not in body


def test_error_page_shows_request_path(running_server, monkeypatch):
    server = running_server
    port = server.server_address[1]
    monkeypatch.setattr(cs, "_get_map_data", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    body = _fetch_error_page(port, "/map.html")
    assert "/map.html" in body


def test_missing_route_uses_the_styled_404_page(running_server):
    port = running_server.server_address[1]

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/missing-page.html")

    assert exc_info.value.code == 404
    body = exc_info.value.read().decode()
    assert "Nothing here" in body
    assert "/missing-page.html" in body


def test_missing_api_route_returns_json_404(running_server):
    port = running_server.server_address[1]

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/nonexistent-route-xyz")

    assert exc_info.value.code == 404
    assert exc_info.value.headers.get_content_type() == "application/json"
    body = json.loads(exc_info.value.read().decode())
    assert body["error"] == "unknown route: /api/nonexistent-route-xyz"
