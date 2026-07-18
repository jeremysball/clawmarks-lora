import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from clawmarks import curation_server as cs
from clawmarks import config
from clawmarks.workspace_context import WorkspaceContext


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "ACTIVE_LEG_FILE", tmp_path / "state" / "active_leg.json")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None
    yield


@pytest.fixture
def running_server_with_leg():
    """Stand up a real Handler instance on a free port with a leg selected that has no scored
    manifest. Used by test_status_page_shows_selected_leg_with_no_data to assert the root page
    distinguishes 'no leg selected' from 'leg selected, no scored data yet'."""
    cs._create_expedition({"name": "uncanny_frontier", "textures": [], "fallback_subjects": []})
    cs._set_active_selection("uncanny_frontier", "cockpit")
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=2)


def test_list_expeditions_empty_when_none_exist():
    assert cs._list_expeditions() == []


def test_create_leg_rejects_unsafe_expedition_name():
    with pytest.raises(ValueError, match="path separator"):
        cs._create_leg({"expedition": "../escape", "name": "new-leg"})


def test_create_expedition_writes_config_and_scaffolds_cockpit_leg():
    payload = {
        "name": "demo", "trigger_word": "trentbuckle style, ",
        "negative_prompt": "low quality, blurry, watermark",
        "textures": ["tex-a"], "fallback_subjects": ["subj-a"],
        "budget_usd_cap": 5.0, "budget_safety_margin": 0.5,
        "gen_batch_size": 20, "explore_fraction": 0.5, "max_generations": 100,
    }
    result = cs._create_expedition(payload)

    assert result == {"ok": True, "name": "demo"}
    expedition_file = config.EXPEDITIONS_DIR / "demo" / "expedition.json"
    assert json.loads(expedition_file.read_text())["trigger_word"] == "trentbuckle style, "
    cockpit_leg_file = config.EXPEDITIONS_DIR / "demo" / "legs" / "cockpit.json"
    assert cockpit_leg_file.exists()
    assert config.leg_dir("demo", "cockpit").exists()


def test_create_expedition_rejects_a_name_that_already_exists():
    payload = {"name": "demo", "textures": [], "fallback_subjects": []}
    cs._create_expedition(payload)

    with pytest.raises(ValueError, match="already exists"):
        cs._create_expedition(payload)


def test_list_expeditions_reports_every_leg():
    cs._create_expedition({"name": "demo", "textures": [], "fallback_subjects": []})
    (config.EXPEDITIONS_DIR / "demo" / "legs" / "round1.json").write_text("{}")

    expeditions = cs._list_expeditions()

    assert len(expeditions) == 1
    assert expeditions[0]["name"] == "demo"
    assert set(expeditions[0]["legs"]) == {"cockpit", "round1"}


def test_status_page_shows_selected_leg_with_no_data(running_server_with_leg):
    port = running_server_with_leg.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status.html") as resp:
        body = resp.read()

    assert b"no expedition/leg selected" not in body.lower()
    assert b"uncanny_frontier" in body
    assert b"cockpit" in body
    assert b"no scored" in body.lower() or b"no search data" in body.lower()


def test_status_page_warns_when_manifest_images_are_missing(running_server_with_leg):
    leg_dir = config.leg_dir("uncanny_frontier", "cockpit")
    manifest = [
        {"tag": f"gen1_{index}", "file": str(leg_dir / f"gen1_{index}.png")}
        for index in range(3)
    ]
    (leg_dir / "scored_manifest.json").write_text(json.dumps(manifest))

    port = running_server_with_leg.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status.html") as resp:
        body = resp.read().decode()

    assert "missing" in body.lower() or "data integrity" in body.lower()
    assert "Launch a round" not in body


def test_active_leg_selection_warns_when_manifest_images_are_missing(
    running_server_with_leg, capsys
):
    leg_dir = config.leg_dir("uncanny_frontier", "round1")
    leg_dir.mkdir(parents=True, exist_ok=True)
    manifest = [{"tag": "gen1_a", "file": str(leg_dir / "gen1_a.png")}]
    (leg_dir / "scored_manifest.json").write_text(json.dumps(manifest))

    port = running_server_with_leg.server_address[1]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/active-leg",
        data=json.dumps({"expedition": "uncanny_frontier", "leg": "round1"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as resp:
        assert resp.status == 200

    assert "warning" in capsys.readouterr().err.lower()


@pytest.fixture
def running_server_with_leg_and_data(tmp_path):
    """Stand up a real Handler with a leg selected that has at least one scored manifest
    image present on disk, so the root page renders the 'has data' branch
    (_status_page_data_body). Used by test_status_page_data_branch_surfaces_comparison_count."""
    cs._create_expedition({"name": "uncanny_frontier", "textures": [], "fallback_subjects": []})
    cs._set_active_selection("uncanny_frontier", "cockpit")
    leg_dir = config.leg_dir("uncanny_frontier", "cockpit")
    leg_dir.mkdir(parents=True, exist_ok=True)
    image_path = leg_dir / "gen1_a.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    manifest = [{"tag": "gen1_a", "file": str(image_path)}]
    (leg_dir / "scored_manifest.json").write_text(json.dumps(manifest))
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=2)


def test_status_page_data_branch_surfaces_comparison_count(running_server_with_leg_and_data):
    port = running_server_with_leg_and_data.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status.html") as resp:
        body = resp.read().decode()

    assert 'id="cmpStat"' in body
    assert "fetch('/api/preference_status')" in body


def test_status_page_data_body_uses_sulfur_proof_shell(running_server_with_leg_and_data):
    """Task 5 (status data branch) render contract: the 'has data' status view sits on the
    Sulfur Proof foundation, has no prefers-color-scheme: dark branch, includes the shared
    header's context-switcher script, and ships a semantic <header>. The legacy
    DARK_TOKENS/BTN_CSS imports are gone from the page-local <style>."""
    port = running_server_with_leg_and_data.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status.html") as resp:
        body = resp.read().decode()
    assert "--paper:#C3C5BA" in body
    assert "shared-ui.js" in body
    assert "<header" in body
    assert "prefers-color-scheme: dark" not in body
    assert "DARK_TOKENS" not in body
    assert "BTN_CSS" not in body


def test_status_page_no_selection_body_uses_sulfur_proof_shell(running_server_with_leg):
    """Task 5 (status no-selection branch) render contract: same as the data branch -- Sulfur
    foundation, no dark theme, shared-ui.js, semantic <header>. The legacy .panel
    border-radius:8px on the three pickers panels is gone (replaced by a flat bordered
    treatment or a CONTROL_CSS depth class)."""
    port = running_server_with_leg.server_address[1]
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status.html") as resp:
        body = resp.read().decode()
    assert "--paper:#C3C5BA" in body
    assert "shared-ui.js" in body
    assert "<header" in body
    assert "prefers-color-scheme: dark" not in body
    assert "DARK_TOKENS" not in body
    assert "BTN_CSS" not in body
    assert "border-radius:8px" not in body


def test_status_page_selected_empty_body_uses_sulfur_proof_shell(running_server_with_leg):
    """Task 5 (status selected-empty branch) render contract: same as the data branch. The
    page must also expose a link to /status.html so the brief's Step 1 assertion
    'href=\"/status.html\"' in empty_state_html is satisfied (the shared header's
    session-status link is the natural place; the existing /runs.html prose link stays as
    additional, still-valid guidance)."""
    port = running_server_with_leg.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status.html") as resp:
        body = resp.read().decode()
    assert "--paper:#C3C5BA" in body
    assert "shared-ui.js" in body
    assert "<header" in body
    assert "prefers-color-scheme: dark" not in body
    assert 'href="/status.html?expedition=uncanny_frontier&amp;leg=cockpit"' in body
    # The legacy /runs.html prose link is still present as additional guidance.
    assert 'href="/runs.html"' in body
    assert "DARK_TOKENS" not in body
    assert "BTN_CSS" not in body
    assert "border-radius:8px" not in body


def test_status_page_data_integrity_error_body_uses_sulfur_proof_shell(running_server_with_leg):
    """Task 5 (status data-integrity-error branch) render contract: same as the other three
    branches. The brief's Step 1 'role=\"alert\"' assertion is satisfied by the warning
    paragraph that says 'Data integrity warning' / 'Do not launch a new round'; that single
    <p> carries role='alert' so screen readers announce it as an urgent live region. The
    legacy DARK_TOKENS/BTN_CSS imports and the .panel border-radius:8px are gone."""
    leg_dir = config.leg_dir("uncanny_frontier", "cockpit")
    manifest = [
        {"tag": f"gen1_{index}", "file": str(leg_dir / f"gen1_{index}.png")}
        for index in range(3)
    ]
    (leg_dir / "scored_manifest.json").write_text(json.dumps(manifest))

    port = running_server_with_leg.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status.html") as resp:
        body = resp.read().decode()
    assert "--paper:#C3C5BA" in body
    assert "shared-ui.js" in body
    assert "<header" in body
    assert "prefers-color-scheme: dark" not in body
    assert 'role="alert"' in body
    # The role="alert" must be on the warning paragraph specifically, not the whole body.
    # Verify the substring "Data integrity warning" sits inside a tag carrying role="alert".
    assert 'role="alert"' in body
    warning_idx = body.find("Data integrity warning")
    assert warning_idx != -1
    # Walk back to the nearest <p ...> tag and confirm it carries role="alert".
    snippet = body[max(0, warning_idx - 200):warning_idx]
    assert 'role="alert"' in snippet
    assert "DARK_TOKENS" not in body
    assert "BTN_CSS" not in body
    assert "border-radius:8px" not in body


def test_root_serves_scan_gallery_and_explore_stays_on_its_own_route(
    running_server_with_leg, monkeypatch,
):
    """The image gallery is the homepage: "/" serves the scan gallery while "/explore.html"
    stays on the Focus research desk. "/status.html" remains a pure status route."""
    monkeypatch.setattr(cs, "_get_scan_items", lambda expedition, leg: [
        {"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0, "sort_gen": 1,
         "category": "test", "prompt_name": "fox", "prompt_type": "conflict",
         "prompt": "p", "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28, "sampler": "ddim",
         "negative": "n", "faith": 0.5, "novelty": 0.5, "sim": []}
    ])
    port = running_server_with_leg.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as root_resp:
        root_body = root_resp.read().decode()
        assert root_resp.status == 200
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/explore.html") as explore_resp:
        explore_body = explore_resp.read().decode()
        assert explore_resp.status == 200
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status.html") as status_resp:
        status_body = status_resp.read().decode()
        assert status_resp.status == 200
    assert "<title>CLAWMARKS uncanny scan</title>" in root_body
    assert "<title>CLAWMARKS research desk</title>" in explore_body
    assert "<title>CLAWMARKS research desk</title>" not in status_body
    assert root_body != status_body


def test_explore_foci_retains_present_and_missing_evidence(monkeypatch, tmp_path):
    focus = {
        "focus_id": "focus_11111111111111111111111111111111",
        "source": {
            "member_tags": ["generated-present", "generated-missing"],
            "real_anchor_tags": ["anchor-present", "anchor-missing"],
        },
    }

    class Store:
        def list(self, scope, status=None):
            return [focus]

    monkeypatch.setattr(cs.Handler, "_focus_store", lambda _self: Store())
    monkeypatch.setattr(cs, "load_manifest", lambda _expedition, _leg: [{"tag": "generated-present"}])
    monkeypatch.setattr(cs, "REAL_DIR", tmp_path)
    (tmp_path / "anchor-present").write_bytes(b"real")

    handler = object.__new__(cs.Handler)
    enriched = handler._explore_foci(WorkspaceContext("demo", "round1"))[0]

    assert enriched["evidence"]["generated_members"] == [
        {"tag": "generated-present", "record": {"tag": "generated-present"}},
        {"tag": "generated-missing", "missing": True},
    ]
    assert enriched["evidence"]["real_anchors"] == [
        {"tag": "anchor-present"},
        {"tag": "anchor-missing", "missing": True},
    ]


def test_unfavorite_rejects_payload_for_stale_leg(running_server_with_leg):
    leg_a = config.leg_dir("uncanny_frontier", "cockpit")
    leg_b = config.leg_dir("uncanny_frontier", "round2")
    leg_b.mkdir(parents=True, exist_ok=True)
    (leg_b / "round2.json").write_text("{}")
    favorites_a = {"gen1_a": {"tag": "gen1_a"}}
    (leg_a / "user_favorites.json").write_text(json.dumps(favorites_a))

    port = running_server_with_leg.server_address[1]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/unfavorite",
        data=json.dumps({
            "tag": "gen1_a",
            "expedition": "uncanny_frontier",
            "leg": "round2",
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request)

    assert exc_info.value.code == 409
    assert json.loads((leg_a / "user_favorites.json").read_text()) == favorites_a


@pytest.mark.parametrize("endpoint", ["/api/favorite", "/api/unfavorite"])
def test_favorite_mutation_rejects_focus_from_a_different_leg(running_server_with_leg, endpoint):
    cs._create_leg({"expedition": "uncanny_frontier", "name": "round2"})
    leg_dir = config.leg_dir("uncanny_frontier", "cockpit")
    image_path = leg_dir / "gen1_a.png"
    image_path.write_bytes(b"image")
    focus = cs.FocusStore(config.STATE_DIR, cs.REAL_DIR).create(
        cs.Scope("uncanny_frontier", "cockpit"),
        {"label": "Cockpit focus", "source": {"view": "map", "kind": "map_members",
         "member_tags": ["gen1_a"], "real_anchor_tags": []}, "question": "q",
         "observation": "o", "hypothesis_text": "h", "test_contract": None},
        [{"tag": "gen1_a", "file": str(image_path)}],
    )
    payload = {
        "tag": "gen1_a",
        "expedition": "uncanny_frontier",
        "leg": "round2",
        "focus_id": focus["focus_id"],
    }
    port = running_server_with_leg.server_address[1]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request)

    assert exc_info.value.code == 400
    assert not (config.leg_dir("uncanny_frontier", "round2") / "user_favorites.json").exists()


def test_create_leg_writes_empty_overrides_and_scaffolds_its_dir():
    cs._create_expedition({"name": "demo", "textures": [], "fallback_subjects": []})

    result = cs._create_leg({"expedition": "demo", "name": "round1"})

    assert result == {"ok": True, "expedition": "demo", "name": "round1"}
    leg_file = config.EXPEDITIONS_DIR / "demo" / "legs" / "round1.json"
    assert json.loads(leg_file.read_text()) == {}
    assert config.leg_dir("demo", "round1").exists()


def test_create_leg_rejects_an_unknown_expedition():
    with pytest.raises(ValueError, match="unknown expedition"):
        cs._create_leg({"expedition": "nope", "name": "round1"})


def test_create_leg_rejects_a_name_that_already_exists():
    cs._create_expedition({"name": "demo", "textures": [], "fallback_subjects": []})
    cs._create_leg({"expedition": "demo", "name": "round1"})

    with pytest.raises(ValueError, match="already exists"):
        cs._create_leg({"expedition": "demo", "name": "round1"})


def test_create_leg_rejects_a_blank_name():
    cs._create_expedition({"name": "demo", "textures": [], "fallback_subjects": []})

    with pytest.raises(ValueError, match="'name' is required"):
        cs._create_leg({"expedition": "demo", "name": "  "})


def test_create_leg_rejects_the_reserved_name_legs():
    # A leg literally named "legs" would resolve config.leg_dir("demo", "legs") to the same
    # directory that holds every other leg's legs/<leg>.json config file, since EXPEDITIONS_DIR
    # and leg_dir() share one root as of ADR 0001. Must be rejected, not silently collided.
    cs._create_expedition({"name": "demo", "textures": [], "fallback_subjects": []})

    with pytest.raises(ValueError, match="reserved"):
        cs._create_leg({"expedition": "demo", "name": "legs"})


def test_create_leg_rejects_a_name_with_a_path_separator():
    cs._create_expedition({"name": "demo", "textures": [], "fallback_subjects": []})

    with pytest.raises(ValueError, match="path separator"):
        cs._create_leg({"expedition": "demo", "name": "../escape"})


def test_create_expedition_rejects_a_name_with_a_path_separator():
    with pytest.raises(ValueError, match="path separator"):
        cs._create_expedition({"name": "../escape", "textures": [], "fallback_subjects": []})
