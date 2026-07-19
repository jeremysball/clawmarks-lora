import json
import re

from clawmarks.build import preference_rank


def test_compute_data_returns_no_model_state_when_model_missing(tmp_path):
    manifest = [{"file": "/x/a.png", "tag": "a", "prompt_name": "p", "centroid_sim": 0.5, "novelty": 0.5}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))

    data = preference_rank.compute_data(str(tmp_path))
    assert data["has_model"] is False

    html = preference_rank.render_html(data)
    assert "no trained model" in html.lower() or "not enough" in html.lower()
    assert "topnav" in html


def test_rank_page_has_bounded_review_mode_and_rank_ordinals():
    data = {"has_model": True, "items": [
        {"tag": "a", "thumb": "a.jpg", "faith": 0.5, "novelty": 0.4,
         "predicted_preference": 0.8},
    ]}

    html = preference_rank.render_html(data)

    assert "Review top, middle, and bottom" in html
    assert "Rank #" in html
    assert "/api/preference_rank/flag" in html


def test_rank_page_renders_persisted_flag_objects_as_selected_buttons():
    data = {"has_model": True, "items": [
        {"tag": "model-controlled<tag>", "thumb": "a.jpg", "faith": 0.5, "novelty": 0.4,
         "predicted_preference": 0.8},
    ]}

    html = preference_rank.render_html(data)

    assert "flags[tag]?.flag === flag" in html
    assert "aria-pressed=\"${flags[it.tag]?.flag === 'matches'}\"" in html
    assert "class=\"flag-button ${flagSelected(it.tag, 'matches')}\"" in html
    assert "flags[tag] = flag" not in html


def test_rank_page_reports_flag_save_failures_without_mutating_state():
    data = {"has_model": True, "items": [
        {"tag": "a", "thumb": "a.jpg", "faith": 0.5, "novelty": 0.4,
         "predicted_preference": 0.8},
    ]}

    html = preference_rank.render_html(data)

    assert "if (!r.ok) throw new Error('flag save failed')" in html
    assert "flags[tag] = {flag: flag, flagged_at: flags[tag]?.flagged_at ?? null}" in html
    assert "Could not save this flag." in html
    assert "id=\"flagError\"" in html


def test_render_html_never_emits_a_literal_closing_script_tag():
    """A literal "</script>" substring anywhere before the real closing tag truncates the
    browser's HTML parse of the whole <script> block early -- everything after it is dropped
    silently, with no console error. This bit six pages via a copy-pasted comment; guard
    against it coming back."""
    data = {"has_model": True, "items": [
        {"tag": "a", "prompt_name": "p", "prompt_type": "style", "faith": 0.5, "novelty": 0.5,
         "strength": 1.0, "cfg": 7.0, "thumb": "a.png", "file": "a.png", "predicted_preference": 0.9},
    ]}
    html = preference_rank.render_html(data)
    script_start = html.index("<script>")
    script_end = html.index("</script>", script_start + len("<script>"))
    body = html[script_start + len("<script>"):script_end]
    assert "</script" not in body


def test_render_html_uses_sulfur_proof_shell():
    """Task 4 render contract: the page sits on the Sulfur Proof foundation, includes the
    shared header's context-switcher script, ships a semantic <header>, and has no
    prefers-color-scheme: dark branch (Sulfur Proof is the only theme)."""
    data = {"has_model": True, "items": [
        {"tag": "a", "thumb": "a.jpg", "faith": 0.5, "novelty": 0.4,
         "predicted_preference": 0.8},
    ]}
    html = preference_rank.render_html(data)
    assert "--paper:#C3C5BA" in html
    assert "shared-ui.js" in html
    assert "<header" in html
    assert "prefers-color-scheme: dark" not in html


def test_render_html_no_model_state_uses_sulfur_proof_shell():
    """The 'no trained model' empty-state page must also sit on the Sulfur Proof
    foundation and ship the shared header's context-switcher script. Task 4's universal
    render contract applies to every variant the page renders, not only the populated
    one. The empty state was historically a single inline string with its own
    `:root { color-scheme: dark }` block, which the migration removes."""
    data = {"has_model": False, "model_file": "/tmp/missing.joblib"}
    html = preference_rank.render_html(data)
    assert "--paper:#C3C5BA" in html
    assert "shared-ui.js" in html
    assert "<header" in html
    assert "prefers-color-scheme: dark" not in html


def test_render_html_model_evidence_is_ruled_rows_not_stat_card_grid():
    """Task 4 brief, Step 3 (Preference pages): the list of model evidence the preference
    model produces (the ranked images, with their predicted score / faith / novelty and
    review-mode flag buttons) renders as flat rows separated by `border-bottom:1px solid
    var(--rule)`, matching the "no statistic-card grid" pattern used in part C's
    redundancy/novelty-decay/lineage migration. The legacy '#grid' with rounded bordered
    '.cell' cards must be gone. The shared INFOTIP_CSS carve-out (the infopop has
    border-radius:8px) is allowed by the brief, so we check the page-local portion only."""
    data = {"has_model": True, "items": [
        {"tag": "a", "thumb": "a.jpg", "faith": 0.5, "novelty": 0.4,
         "predicted_preference": 0.8},
    ]}
    html = preference_rank.render_html(data)

    # The page-local CSS lives in the <style> block before the first .infobtn rule, which
    # is the start of the shared INFOTIP_CSS.
    infobtn_start = html.index(".infobtn")
    page_local = html[:infobtn_start]

    # The new ruled-row treatment: each row has a `border-bottom:1px solid var(--rule)`.
    assert "border-bottom:1px solid var(--rule)" in page_local
    # No border-radius in the page-local CSS (only the shared INFOTIP_CSS carve-out may
    # have it). The legacy '.cell { border-radius:10px; }' and '.cell .review button
    # { border-radius:4px; }' are gone.
    assert "border-radius" not in page_local
    # The legacy filled-card treatment on the page-local .cell selector is gone; the row
    # is flat on paper.
    assert "background:var(--panel)" not in page_local
    # No hex-coded dark theme color on the page-local selectors; everything routes through
    # the shared Sulfur tokens.
    assert "color:#eaeaee" not in page_local
    assert "color:#9a9aa4" not in page_local
    assert "color:#0b0b0d" not in page_local
    # The legacy cell `border:1px solid var(--border)` filled-card wrapper is gone.
    assert "border:1px solid var(--border)" not in page_local


def test_render_html_uses_plain_metric_labels():
    """No user-facing text uses faith=, f=, n= as unexplained labels."""
    data = {"has_model": True, "items": [
        {"tag": "a", "thumb": "a.jpg", "faith": 0.5, "novelty": 0.4,
         "predicted_preference": 0.8},
    ]}
    html = preference_rank.render_html(data)
    assert "faithfulness=${it.faith} novelty=${it.novelty}" in html
    assert 'f=${' not in html
    assert re.search(r'(?<!fulness)faith=', html) is None
