import json

from clawmarks.build import preference_rank


def test_compute_data_returns_no_model_state_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(preference_rank, "MODEL_FILE", str(tmp_path / "does_not_exist.joblib"))
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))

    data = preference_rank.compute_data(str(tmp_path))
    assert data["has_model"] is False

    html = preference_rank.render_html(data)
    assert "no trained model" in html.lower() or "not enough" in html.lower()
