import json

from clawmarks.search import driver


def _cfg(dir_path, expedition="demo", leg="leg1"):
    return driver.LegConfig(
        expedition=expedition, leg=leg, dir=dir_path,
        wall_clock_cap_hours=1.0, budget_usd_cap=1.0, budget_safety_margin=0.1,
        gen_batch_size=1, explore_fraction=0.5, max_generations=1,
        textures=[], fallback_subjects=[], seed_from_start=False,
    )


def test_new_expedition_has_no_sibling_leg_data_yet(tmp_path):
    expedition_root = tmp_path / "demo"
    leg_dir = expedition_root / "leg1"
    leg_dir.mkdir(parents=True)

    cfg = _cfg(leg_dir)
    assert driver._load_sibling_leg_manifests(cfg) == []


def test_sibling_leg_manifests_are_pooled_but_not_the_leg_s_own(tmp_path):
    expedition_root = tmp_path / "demo"
    leg1_dir = expedition_root / "leg1"
    leg2_dir = expedition_root / "leg2"
    leg1_dir.mkdir(parents=True)
    leg2_dir.mkdir(parents=True)
    (leg1_dir / "scored_manifest.json").write_text(json.dumps([{"tag": "leg1_a", "file": "a.png"}]))
    (leg2_dir / "scored_manifest.json").write_text(json.dumps([{"tag": "leg2_a", "file": "b.png"}]))

    cfg = _cfg(leg1_dir, leg="leg1")
    pooled = driver._load_sibling_leg_manifests(cfg)

    assert pooled == [{"tag": "leg2_a", "file": "b.png"}]


def test_sibling_leg_without_a_manifest_yet_is_skipped(tmp_path):
    expedition_root = tmp_path / "demo"
    leg1_dir = expedition_root / "leg1"
    leg2_dir = expedition_root / "leg2"  # no manifest written yet
    leg1_dir.mkdir(parents=True)
    leg2_dir.mkdir(parents=True)

    cfg = _cfg(leg1_dir, leg="leg1")
    assert driver._load_sibling_leg_manifests(cfg) == []


def test_third_leg_pools_both_earlier_siblings(tmp_path):
    expedition_root = tmp_path / "demo"
    for name in ("leg1", "leg2", "leg3"):
        (expedition_root / name).mkdir(parents=True)
    (expedition_root / "leg1" / "scored_manifest.json").write_text(json.dumps([{"tag": "l1"}]))
    (expedition_root / "leg2" / "scored_manifest.json").write_text(json.dumps([{"tag": "l2"}]))

    cfg = _cfg(expedition_root / "leg3", leg="leg3")
    pooled = driver._load_sibling_leg_manifests(cfg)

    assert {m["tag"] for m in pooled} == {"l1", "l2"}


def test_cli_requires_expedition_and_leg():
    # smoke-checks the real parser via main() itself rather than re-deriving its structure
    import pytest
    with pytest.raises(SystemExit):
        driver.main([])  # missing --expedition/--leg
