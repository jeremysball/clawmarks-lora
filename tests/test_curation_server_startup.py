import threading
from http.server import HTTPServer
import urllib.error
import urllib.request

import pytest

from clawmarks import curation_server as cs
from clawmarks import config


def test_check_manifest_images_is_a_noop_with_no_active_leg(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None

    cs._check_manifest_images()  # should not raise, print a warning, or sys.exit


def test_check_manifest_images_warns_to_stderr_when_selected_leg_has_no_manifest(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cs, "_active_out_dir", lambda: tmp_path)
    monkeypatch.setitem(cs._active_selection, "expedition", "uncanny_frontier")
    monkeypatch.setitem(cs._active_selection, "leg", "round3")

    cs._check_manifest_images()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "uncanny_frontier/round3" in captured.err
    assert "launch a round" in captured.err


def test_reconcile_stuck_trials_is_a_noop_with_no_active_leg(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None

    cs._reconcile_stuck_trials()  # should not raise (no leg selected, nothing to reconcile)


@pytest.fixture
def running_server_no_leg(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=2)


def test_status_page_shows_no_leg_selected_without_error_string(running_server_no_leg):
    port = running_server_no_leg.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
        body = resp.read().decode()
    assert resp.status == 200
    assert "no expedition/leg selected" in body.lower()
    assert "could not read manifest" not in body


# Regression tests for the None-guard crash class: _require_out_dir() (curation_server.py) turns
# a bare _active_out_dir() dereference (a NoneType/str TypeError -> 500 stack trace) into a clean
# 400 with a "no expedition/leg selected" message. These three routes were confirmed to 500
# before the fix: a page-render route (scan.html), a JSON API route (favorites), and the /thumbs/
# static-file route, which together cover every response shape _do_GET can produce.

def test_scan_html_returns_a_clean_400_with_no_active_leg(running_server_no_leg):
    port = running_server_no_leg.server_address[1]
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/scan.html")
    assert exc_info.value.code == 400
    body = exc_info.value.read().decode()
    assert "no expedition/leg selected" in body
    assert "Something went wrong" not in body


def test_favorites_api_returns_a_clean_400_with_no_active_leg(running_server_no_leg):
    port = running_server_no_leg.server_address[1]
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/favorites")
    assert exc_info.value.code == 400
    body = exc_info.value.read().decode()
    assert "no expedition/leg selected" in body


def test_thumbs_route_returns_a_clean_400_with_no_active_leg(running_server_no_leg):
    port = running_server_no_leg.server_address[1]
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/thumbs/some_tag.jpg")
    assert exc_info.value.code == 400
    body = exc_info.value.read().decode()
    assert "no expedition/leg selected" in body


def test_cockpit_run_post_returns_a_clean_400_with_no_active_leg(running_server_no_leg, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "fake-key-for-this-test")
    port = running_server_no_leg.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/cockpit/queue/some-trial-id/run",
        data=b"{}", headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400
    body = exc_info.value.read().decode()
    assert "no expedition/leg selected" in body
