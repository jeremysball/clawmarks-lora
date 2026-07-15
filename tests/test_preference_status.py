# tests/test_preference_status.py
import json

import numpy as np

from clawmarks.build import preference_status
from clawmarks.search import embed_cache, preference_pairwise_model


def _write_comparisons(tmp_path, n):
    comparisons = [{"winner": f"w{i}", "loser": f"l{i}", "compared_at": "t"} for i in range(n)]
    (tmp_path / "user_comparisons.json").write_text(json.dumps(comparisons))
    return comparisons


def _write_embeddings(tmp_path, tags):
    embeddings = np.random.RandomState(0).normal(size=(len(tags), 2)).astype(np.float32)
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)


def test_compute_data_with_no_comparisons_file_reports_zero_count(tmp_path):
    data = preference_status.compute_data(tmp_path)
    assert data["n_comparisons"] == 0
    assert data["has_model"] is False
    assert data["model_meta"] is None
    assert data["use_predicted_preference"] is False
    assert "50" in data["comparisons_gate_message"]


def test_compute_data_below_min_comparisons_reports_count_gate(tmp_path):
    _write_comparisons(tmp_path, 15)
    data = preference_status.compute_data(tmp_path)
    assert data["n_comparisons"] == 15
    assert "15" in data["comparisons_gate_message"] and "50" in data["comparisons_gate_message"]


def test_compute_data_at_min_comparisons_has_no_gate_message(tmp_path):
    comparisons = _write_comparisons(tmp_path, 50)
    tags = sorted({t for c in comparisons for t in (c["winner"], c["loser"])})
    _write_embeddings(tmp_path, tags)
    data = preference_status.compute_data(tmp_path)
    assert data["comparisons_gate_message"] == ""


def test_compute_data_reads_model_meta_and_toggle_when_model_exists(tmp_path):
    model_path = tmp_path / "preference_pairwise_model.joblib"
    meta_path = tmp_path / "preference_pairwise_model_meta.json"
    model_path.write_text("fake model bytes")
    meta = {"trained_at": "2026-07-11T00:00:00+00:00", "n_comparisons": 60, "cv_accuracy": 0.8}
    meta_path.write_text(json.dumps(meta))
    preference_status.preference_settings.save(True, tmp_path)

    data = preference_status.compute_data(tmp_path)
    assert data["has_model"] is True
    assert data["model_meta"] == meta
    assert data["new_comparisons_since_train"] == 0
    assert data["use_predicted_preference"] is True


def test_compute_data_counts_new_comparisons_since_last_train(tmp_path):
    model_path = tmp_path / "preference_pairwise_model.joblib"
    meta_path = tmp_path / "preference_pairwise_model_meta.json"
    comparisons = _write_comparisons(tmp_path, 65)
    tags = sorted({t for c in comparisons for t in (c["winner"], c["loser"])})
    _write_embeddings(tmp_path, tags)
    model_path.write_text("fake model bytes")
    meta_path.write_text(json.dumps({
        "trained_at": "2026-07-11T00:00:00+00:00", "n_comparisons": 60, "cv_accuracy": 0.8,
    }))

    data = preference_status.compute_data(tmp_path)

    assert data["new_comparisons_since_train"] == 5
    assert data["comparisons_changed_since_train"] is True


def test_compute_data_detects_swap_via_fingerprint_with_same_count(tmp_path):
    """A comparison's winner/loser being replaced by a different pair doesn't change
    n_comparisons, but it does change which images the model would train on. The fingerprint
    should catch this even though a bare comparison-count diff (the pre-fix behavior) would
    report zero new comparisons."""
    model_path = tmp_path / "preference_pairwise_model.joblib"
    meta_path = tmp_path / "preference_pairwise_model_meta.json"
    comparisons = _write_comparisons(tmp_path, 60)
    tags = sorted({t for c in comparisons for t in (c["winner"], c["loser"])})
    _write_embeddings(tmp_path, tags)
    tags_arr, embeddings = embed_cache.load_cache(tmp_path / "embeddings.npz")
    trained_fingerprint = preference_pairwise_model.comparisons_fingerprint(tags_arr, embeddings, comparisons)
    model_path.write_text("fake model bytes")
    meta_path.write_text(json.dumps({
        "trained_at": "2026-07-11T00:00:00+00:00", "n_comparisons": 60, "cv_accuracy": 0.8,
        "comparisons_fingerprint": trained_fingerprint,
    }))

    # Replace one comparison's loser; n_comparisons stays at 60.
    comparisons[0]["loser"] = comparisons[1]["loser"]
    (tmp_path / "user_comparisons.json").write_text(json.dumps(comparisons))

    data = preference_status.compute_data(tmp_path)

    assert data["new_comparisons_since_train"] == 0
    assert data["comparisons_changed_since_train"] is True


def test_render_html_disables_toggle_when_no_model():
    data = {"n_comparisons": 0, "n_usable": 0, "min_comparisons": 50, "comparisons_gate_message": "not enough comparisons",
            "has_model": False, "model_meta": None, "new_comparisons_since_train": 0,
            "comparisons_changed_since_train": False, "use_predicted_preference": False}
    html = preference_status.render_html(data)
    assert "disabled" in html
    assert "/api/preference_toggle" in html
    assert "/api/preference_retrain" in html


def test_render_html_enables_toggle_when_model_exists():
    meta = {"trained_at": "2026-07-11T00:00:00+00:00", "n_comparisons": 60, "cv_accuracy": 0.8,
            "baseline_accuracy": 0.5, "p_value": 0.03, "n_permutations": 200}
    data = {"n_comparisons": 60, "n_usable": 60, "min_comparisons": 50, "comparisons_gate_message": "",
            "has_model": True, "model_meta": meta, "new_comparisons_since_train": 0,
            "comparisons_changed_since_train": False, "use_predicted_preference": True}
    html = preference_status.render_html(data)
    assert "checked" in html
    assert "0.8" in html
    assert "majority-class baseline accuracy" in html
    assert "50.0%" in html
    assert "permutation p-value" in html
    assert "p &lt; 0.05: unlikely to be chance" in html
    assert "Retraining…" in html


def test_render_html_interprets_non_significant_p_value():
    meta = {"trained_at": "2026-07-11T00:00:00+00:00", "n_comparisons": 60, "cv_accuracy": 0.8,
            "baseline_accuracy": 0.5, "p_value": 0.4, "n_permutations": 200}
    data = {"n_comparisons": 60, "n_usable": 60, "min_comparisons": 50, "comparisons_gate_message": "",
            "has_model": True, "model_meta": meta, "new_comparisons_since_train": 0,
            "comparisons_changed_since_train": False, "use_predicted_preference": True}

    html = preference_status.render_html(data)

    assert "p &gt;= 0.05: not distinguishable from chance" in html


def test_render_html_omits_statistical_rows_for_old_model_meta():
    meta = {"trained_at": "2026-07-11T00:00:00+00:00", "n_comparisons": 60, "cv_accuracy": 0.8}
    data = {"n_comparisons": 60, "n_usable": 60, "min_comparisons": 50, "comparisons_gate_message": "",
            "has_model": True, "model_meta": meta, "new_comparisons_since_train": 0,
            "comparisons_changed_since_train": False, "use_predicted_preference": True}

    html = preference_status.render_html(data)

    assert "majority-class baseline accuracy" not in html
    assert "permutation p-value" not in html


def test_render_html_shows_staleness_banner_when_new_comparisons_exist():
    meta = {"trained_at": "2026-07-11T00:00:00+00:00", "n_comparisons": 60, "cv_accuracy": 0.8}
    data = {"n_comparisons": 65, "n_usable": 65, "min_comparisons": 50, "comparisons_gate_message": "",
            "has_model": True, "model_meta": meta, "new_comparisons_since_train": 5,
            "comparisons_changed_since_train": True, "use_predicted_preference": True}

    html = preference_status.render_html(data)

    assert "5 new comparisons since last train (2026-07-11T00:00:00+00:00)" in html
    assert "Retrain to include them" in html


def test_render_html_shows_generic_staleness_banner_when_comparisons_changed_but_count_is_same():
    meta = {"trained_at": "2026-07-11T00:00:00+00:00", "n_comparisons": 60, "cv_accuracy": 0.8}
    data = {"n_comparisons": 60, "n_usable": 60, "min_comparisons": 50, "comparisons_gate_message": "",
            "has_model": True, "model_meta": meta, "new_comparisons_since_train": 0,
            "comparisons_changed_since_train": True, "use_predicted_preference": True}

    html = preference_status.render_html(data)

    assert "comparisons have changed since last train (2026-07-11T00:00:00+00:00)" in html
    assert "Retrain to include them" in html


def test_render_html_omits_staleness_banner_when_model_is_current_or_missing():
    current_meta = {"trained_at": "2026-07-11T00:00:00+00:00", "n_comparisons": 60, "cv_accuracy": 0.8}
    current_data = {"n_comparisons": 60, "n_usable": 60, "min_comparisons": 50, "comparisons_gate_message": "",
                    "has_model": True, "model_meta": current_meta, "new_comparisons_since_train": 0,
                    "comparisons_changed_since_train": False, "use_predicted_preference": True}
    missing_data = {"n_comparisons": 60, "n_usable": 60, "min_comparisons": 50, "comparisons_gate_message": "",
                    "has_model": False, "model_meta": None, "new_comparisons_since_train": 0,
                    "comparisons_changed_since_train": False, "use_predicted_preference": False}

    assert "Retrain to include them" not in preference_status.render_html(current_data)
    assert "Retrain to include them" not in preference_status.render_html(missing_data)


def test_render_html_uses_panel_token_for_secondary_button():
    data = {"n_comparisons": 0, "n_usable": 0, "min_comparisons": 50, "comparisons_gate_message": "not enough comparisons",
            "has_model": False, "model_meta": None, "new_comparisons_since_train": 0,
            "comparisons_changed_since_train": False, "use_predicted_preference": False}
    html = preference_status.render_html(data)

    assert ".secondary { background:var(--panel-2);" in html
