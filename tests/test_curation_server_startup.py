import threading
from http.server import HTTPServer
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
    assert "no expedition/leg selected" in body
    assert "could not read manifest" not in body
