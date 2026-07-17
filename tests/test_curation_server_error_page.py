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


def test_404_page_uses_sulfur_proof_shell(running_server):
    """Task 5 (404 page) render contract: the 404 sits on the Sulfur Proof foundation, has no
    prefers-color-scheme: dark branch (Sulfur Proof is the only theme), includes the shared
    header's context-switcher script, and ships a semantic <header> from the shared topnav.
    The legacy 404 had its own bespoke :root dark-theme tokens (--bg/--panel/--border/--text/
    --dim/--accent) plus border-radius:10px on the main card; both must be gone, and the
    Sulfur foundation tokens plus CONTROL_CSS's flat-border treatment must be present
    instead."""
    port = running_server.server_address[1]
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/missing-page.html")
    assert exc_info.value.code == 404
    body = exc_info.value.read().decode()
    assert "--paper:#C3C5BA" in body
    assert "shared-ui.js" in body
    assert "<header" in body
    assert "prefers-color-scheme: dark" not in body
    # The legacy 404's bespoke :root block of dark tokens is gone.
    assert "--bg:#0b0b0d" not in body
    # The legacy border-radius on the main card is gone.
    assert "border-radius:10px" not in body


def test_500_error_page_uses_sulfur_proof_shell(running_server, monkeypatch):
    """Task 5 (500 page) render contract: the error page sits on the Sulfur Proof foundation,
    has no prefers-color-scheme: dark branch, includes the shared header's context-switcher
    script, and ships a semantic <header> from the shared topnav. The legacy 500 had inline
    style= attributes only (no <style> block at all) with a 4px border-radius on the stack
    trace <pre>; both are gone, replaced by a <style> block with the Sulfur foundation and a
    flat-bordered stack trace block."""
    server = running_server
    port = server.server_address[1]
    monkeypatch.setattr(cs, "_get_map_data", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    body = _fetch_error_page(port, "/map.html")
    assert "--paper:#C3C5BA" in body
    assert "shared-ui.js" in body
    assert "<header" in body
    assert "prefers-color-scheme: dark" not in body
    # The legacy inline style="background:#f3f4f6;padding:1rem;border-radius:4px" on the
    # stack trace <pre> is gone, replaced by a class-based rule in the page-local <style>.
    assert 'style="white-space:pre-wrap;font-family:monospace;background:#f3f4f6' not in body
    assert "border-radius:4px" not in body
