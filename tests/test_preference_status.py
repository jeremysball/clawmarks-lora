# tests/test_preference_status.py
import json

from clawmarks.build import preference_status


def _write_ratings(tmp_path, n_yes, n_no):
    ratings = {}
    for i in range(n_yes):
        ratings[f"y{i}"] = {"label": "yes", "rated_at": "t"}
    for i in range(n_no):
        ratings[f"n{i}"] = {"label": "no", "rated_at": "t"}
    (tmp_path / "user_ratings.json").write_text(json.dumps(ratings))


def test_compute_data_with_no_ratings_file_reports_zero_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(preference_status.preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(preference_status.preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    data = preference_status.compute_data(tmp_path)
    assert data["n_yes"] == 0 and data["n_no"] == 0 and data["n_total"] == 0
    assert data["has_model"] is False
    assert data["model_meta"] is None
    assert data["new_labels_since_train"] == 0
    assert data["use_predicted_preference"] is False
    assert "50" in data["labels_gate_message"]


def test_compute_data_below_min_labels_reports_count_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(preference_status.preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(preference_status.preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    _write_ratings(tmp_path, n_yes=10, n_no=5)
    data = preference_status.compute_data(tmp_path)
    assert data["n_yes"] == 10 and data["n_no"] == 5 and data["n_total"] == 15
    assert "15" in data["labels_gate_message"] and "50" in data["labels_gate_message"]


def test_compute_data_at_min_labels_but_imbalanced_reports_balance_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(preference_status.preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(preference_status.preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    _write_ratings(tmp_path, n_yes=58, n_no=2)
    data = preference_status.compute_data(tmp_path)
    assert "5-fold" in data["labels_gate_message"]


def test_compute_data_well_balanced_above_min_labels_has_no_gate_message(tmp_path, monkeypatch):
    monkeypatch.setattr(preference_status.preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(preference_status.preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    _write_ratings(tmp_path, n_yes=30, n_no=30)
    data = preference_status.compute_data(tmp_path)
    assert data["labels_gate_message"] == ""


def test_compute_data_reads_model_meta_and_toggle_when_model_exists(tmp_path, monkeypatch):
    settings_path = tmp_path / "preference_settings.json"
    model_path = tmp_path / "preference_model.joblib"
    meta_path = tmp_path / "preference_model_meta.json"
    monkeypatch.setattr(preference_status.preference_settings, "PREFERENCE_SETTINGS_FILE", settings_path)
    monkeypatch.setattr(preference_status.preference_model, "MODEL_FILE", model_path)
    monkeypatch.setattr(preference_status.preference_model, "MODEL_META_FILE", meta_path)
    model_path.write_text("fake model bytes")
    meta = {"trained_at": "2026-07-10T00:00:00+00:00", "n_labels": 60, "n_yes": 30, "n_no": 30, "cv_accuracy": 0.8}
    meta_path.write_text(json.dumps(meta))
    preference_status.preference_settings.save(True)

    data = preference_status.compute_data(tmp_path)
    assert data["has_model"] is True
    assert data["model_meta"] == meta
    assert data["new_labels_since_train"] == 0
    assert data["use_predicted_preference"] is True


def test_compute_data_counts_new_labels_since_last_train(tmp_path, monkeypatch):
    settings_path = tmp_path / "preference_settings.json"
    model_path = tmp_path / "preference_model.joblib"
    meta_path = tmp_path / "preference_model_meta.json"
    monkeypatch.setattr(preference_status.preference_settings, "PREFERENCE_SETTINGS_FILE", settings_path)
    monkeypatch.setattr(preference_status.preference_model, "MODEL_FILE", model_path)
    monkeypatch.setattr(preference_status.preference_model, "MODEL_META_FILE", meta_path)
    _write_ratings(tmp_path, n_yes=35, n_no=30)
    model_path.write_text("fake model bytes")
    meta_path.write_text(json.dumps({
        "trained_at": "2026-07-10T00:00:00+00:00", "n_labels": 60,
        "n_yes": 30, "n_no": 30, "cv_accuracy": 0.8,
    }))

    data = preference_status.compute_data(tmp_path)

    assert data["new_labels_since_train"] == 5


def test_render_html_disables_toggle_when_no_model():
    data = {"n_yes": 0, "n_no": 0, "n_total": 0, "min_labels": 50, "labels_gate_message": "not enough labels",
            "has_model": False, "model_meta": None, "new_labels_since_train": 0, "use_predicted_preference": False}
    html = preference_status.render_html(data)
    assert "disabled" in html
    assert "/api/preference_toggle" in html
    assert "/api/preference_retrain" in html


def test_render_html_enables_toggle_when_model_exists():
    meta = {"trained_at": "2026-07-10T00:00:00+00:00", "n_labels": 60, "n_yes": 30, "n_no": 30, "cv_accuracy": 0.8,
            "baseline_accuracy": 0.5, "p_value": 0.03, "n_permutations": 200}
    data = {"n_yes": 30, "n_no": 30, "n_total": 60, "min_labels": 50, "labels_gate_message": "",
            "has_model": True, "model_meta": meta, "new_labels_since_train": 0, "use_predicted_preference": True}
    html = preference_status.render_html(data)
    assert 'id="toggle" checked  onchange' in html
    assert "checked" in html
    assert "0.8" in html
    assert "majority-class baseline accuracy" in html
    assert "50.0%" in html
    assert "permutation p-value" in html
    assert "p &lt; 0.05: unlikely to be chance" in html
    assert "Retraining…" in html


def test_render_html_interprets_non_significant_p_value():
    meta = {"trained_at": "2026-07-10T00:00:00+00:00", "n_labels": 60, "n_yes": 30, "n_no": 30, "cv_accuracy": 0.8,
            "baseline_accuracy": 0.5, "p_value": 0.4, "n_permutations": 200}
    data = {"n_yes": 30, "n_no": 30, "n_total": 60, "min_labels": 50, "labels_gate_message": "",
            "has_model": True, "model_meta": meta, "new_labels_since_train": 0, "use_predicted_preference": True}

    html = preference_status.render_html(data)

    assert "p &gt;= 0.05: not distinguishable from chance" in html


def test_render_html_omits_statistical_rows_for_old_model_meta():
    meta = {"trained_at": "2026-07-10T00:00:00+00:00", "n_labels": 60, "n_yes": 30, "n_no": 30, "cv_accuracy": 0.8}
    data = {"n_yes": 30, "n_no": 30, "n_total": 60, "min_labels": 50, "labels_gate_message": "",
            "has_model": True, "model_meta": meta, "new_labels_since_train": 0, "use_predicted_preference": True}

    html = preference_status.render_html(data)

    assert "majority-class baseline accuracy" not in html
    assert "permutation p-value" not in html


def test_render_html_shows_staleness_banner_when_new_labels_exist():
    meta = {"trained_at": "2026-07-10T00:00:00+00:00", "n_labels": 60, "n_yes": 30, "n_no": 30, "cv_accuracy": 0.8}
    data = {"n_yes": 35, "n_no": 30, "n_total": 65, "min_labels": 50, "labels_gate_message": "",
            "has_model": True, "model_meta": meta, "new_labels_since_train": 5, "use_predicted_preference": True}

    html = preference_status.render_html(data)

    assert "5 new ratings since last train (2026-07-10T00:00:00+00:00)" in html
    assert "retrain to include them" in html


def test_render_html_omits_staleness_banner_when_model_is_current_or_missing():
    current_meta = {"trained_at": "2026-07-10T00:00:00+00:00", "n_labels": 60, "n_yes": 30, "n_no": 30, "cv_accuracy": 0.8}
    current_data = {"n_yes": 30, "n_no": 30, "n_total": 60, "min_labels": 50, "labels_gate_message": "",
                    "has_model": True, "model_meta": current_meta, "new_labels_since_train": 0,
                    "use_predicted_preference": True}
    missing_data = {"n_yes": 30, "n_no": 30, "n_total": 60, "min_labels": 50, "labels_gate_message": "",
                    "has_model": False, "model_meta": None, "new_labels_since_train": 0,
                    "use_predicted_preference": False}

    assert "new ratings since last train" not in preference_status.render_html(current_data)
    assert "new ratings since last train" not in preference_status.render_html(missing_data)
