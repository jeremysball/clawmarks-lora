from clawmarks import curation_server as cs
from clawmarks import config


def test_check_manifest_images_is_a_noop_with_no_active_leg(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EXPEDITIONS_DIR", tmp_path / "expeditions")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    cs._active_selection["expedition"] = None
    cs._active_selection["leg"] = None

    cs._check_manifest_images()  # should not raise, print a warning, or sys.exit
