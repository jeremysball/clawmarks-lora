from clawmarks.search.score_manifest import preprocess, MODEL_ID, REAL_DIR


def test_preprocess_and_constants_importable_from_new_location():
    assert MODEL_ID == "facebook/dinov2-base"
    assert REAL_DIR.endswith("corrected_dataset_extract")
    assert callable(preprocess)
