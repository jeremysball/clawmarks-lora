from clawmarks.search.score_manifest import (
    preprocess, MODEL_ID, REAL_DIR, _default_manifest, partition_by_existing_file,
    merge_quarantine_entries,
)


def test_preprocess_and_constants_importable_from_new_location():
    assert MODEL_ID == "facebook/dinov2-base"
    assert REAL_DIR.endswith("corrected_dataset_extract")
    assert callable(preprocess)


def test_default_manifest_uses_output_directory(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")

    assert _default_manifest(tmp_path) == str(manifest)


def test_partition_by_existing_file_keeps_entries_whose_file_exists(tmp_path):
    present_file = tmp_path / "a.png"
    present_file.write_bytes(b"x")
    manifest = [{"file": str(present_file)}]

    present, missing = partition_by_existing_file(manifest)

    assert present == manifest
    assert missing == []


def test_partition_by_existing_file_quarantines_entries_whose_file_is_missing(tmp_path):
    missing_entry = {"file": str(tmp_path / "does_not_exist.png")}
    manifest = [missing_entry]

    present, missing = partition_by_existing_file(manifest)

    assert present == []
    assert missing == [missing_entry]


def test_partition_by_existing_file_does_not_mutate_input(tmp_path):
    present_file = tmp_path / "a.png"
    present_file.write_bytes(b"x")
    missing_entry = {"file": str(tmp_path / "gone.png")}
    manifest = [{"file": str(present_file)}, missing_entry]
    original = list(manifest)

    partition_by_existing_file(manifest)

    assert manifest == original


def test_merge_quarantine_entries_accumulates_across_runs():
    prior = [{"file": "a.png", "tag": "gen1_a"}]
    new = [{"file": "b.png", "tag": "gen1_b"}]

    merged = merge_quarantine_entries(prior, new)

    assert merged == prior + new


def test_merge_quarantine_entries_dedupes_by_file_keeping_latest():
    prior = [{"file": "a.png", "tag": "gen1_a", "novelty": 0.1}]
    new = [{"file": "a.png", "tag": "gen1_a", "novelty": 0.9}]

    merged = merge_quarantine_entries(prior, new)

    assert merged == [{"file": "a.png", "tag": "gen1_a", "novelty": 0.9}]
