import json
import threading
from http.server import HTTPServer
import urllib.request

import pytest

from clawmarks import curation_server as cs


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_active_out_dir", lambda: tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    manifest = [
        {"file": "/x/a.png", "tag": "a", "category": "seedrun1", "prompt_name": "fox",
         "prompt_type": "conflict", "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1,
         "steps": 28, "sampler": "ddim", "negative": "n", "centroid_sim": 0.5, "novelty": 0.5},
    ]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown()
    thread.join(timeout=2)


def test_scan_html_reflects_manifest_change_without_rebuild(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    monkeypatch.setattr(cs.similarity_index, "compute_data", lambda sweep_dir: {})

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/scan.html") as resp:
        first = resp.read().decode()
    assert '"prompt_name": "fox"' in first

    manifest = json.loads((tmp_path / "scored_manifest.json").read_text())
    manifest[0]["prompt_name"] = "wolf"
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    import os
    new_mtime = os.path.getmtime(tmp_path / "scored_manifest.json") + 5
    os.utime(tmp_path / "scored_manifest.json", (new_mtime, new_mtime))

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/scan.html") as resp:
        second = resp.read().decode()
    assert '"prompt_name": "wolf"' in second
    assert '"prompt_name": "fox"' not in second


def test_scan_html_serves_with_filter_state_query_string(running_server, monkeypatch):
    """scan.html mirrors its filter/sort controls into the URL's query string (so a reload or
    back-navigation restores them); the route must still match with that query string attached
    instead of 404ing on anything but a bare /scan.html."""
    server, tmp_path = running_server
    port = server.server_address[1]
    monkeypatch.setattr(cs.similarity_index, "compute_data", lambda sweep_dir: {})

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/scan.html?sortKey=gen_desc&typeFilter=style") as resp:
        html = resp.read().decode()
        assert resp.headers["Cache-Control"] == "no-cache, must-revalidate"
    assert '"prompt_name": "fox"' in html


def test_scan_data_json_route(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    monkeypatch.setattr(cs.similarity_index, "compute_data", lambda sweep_dir: {})

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/scan_data.json") as resp:
        assert resp.headers["Cache-Control"] == "no-cache, must-revalidate"
        data = json.loads(resp.read().decode())
    assert data[0]["tag"] == "a"


def test_scan_html_shows_the_active_expedition_and_leg(running_server, monkeypatch):
    server, _tmp_path = running_server
    port = server.server_address[1]
    monkeypatch.setitem(cs._active_selection, "expedition", "demo")
    monkeypatch.setitem(cs._active_selection, "leg", "leg-b")
    monkeypatch.setattr(cs.similarity_index, "compute_data", lambda sweep_dir: {})

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/scan.html") as resp:
        body = resp.read().decode()

    assert 'href="/"' in body
    assert "demo/leg-b" in body
