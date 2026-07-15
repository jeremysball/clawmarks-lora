from clawmarks.build import compare_page


def test_render_html_includes_compare_api_calls():
    html = compare_page.render_html()
    assert "/api/compare/next" in html
    assert "/api/compare" in html


def test_render_html_has_two_panes():
    html = compare_page.render_html()
    assert 'id="pane1"' in html
    assert 'id="pane2"' in html
    assert 'id="img1"' in html
    assert 'id="img2"' in html


def test_render_html_has_no_button_elements():
    html = compare_page.render_html()
    assert "<button" not in html


def test_render_html_has_zoom_icons_and_overlay():
    html = compare_page.render_html()
    assert 'id="zoom1"' in html
    assert 'id="zoom2"' in html
    assert 'id="zoom-overlay"' in html
    assert "function openZoom(" in html
    assert "function closeZoom(" in html


def test_render_html_has_arrow_key_handling():
    html = compare_page.render_html()
    assert "ArrowLeft" in html
    assert "ArrowRight" in html


def test_render_html_has_session_count():
    html = compare_page.render_html()
    assert 'id="count"' in html
    assert "comparedThisSession" in html


def test_render_html_has_done_state():
    html = compare_page.render_html()
    assert 'id="done"' in html


def test_render_html_has_per_pane_captions_not_shared_meta():
    html = compare_page.render_html()
    assert 'id="cap1"' in html
    assert 'id="cap2"' in html
    # The old single shared caption row is gone; captions now live inside each pane so they
    # stay with their image when the panes stack on mobile.
    assert 'id="meta"' not in html


def test_render_html_has_progress_bar_driven_by_status():
    html = compare_page.render_html()
    assert 'id="prog-fill"' in html
    assert "function renderProgress(" in html
    # The bar reflects a real signal: the model's cross-validated accuracy from the status API.
    assert "/api/preference_status" in html
    assert "cv_accuracy" in html


def test_render_html_captions_avoid_innerhtml_injection():
    html = compare_page.render_html()
    # prompt_name is model-controlled; captions must be set via textContent, never innerHTML.
    assert "cap1').textContent" in html
    assert "cap2').textContent" in html


def test_render_html_supports_keyboard_choice_and_metric_blinding():
    html = compare_page.render_html()

    assert 'role="button"' in html
    assert 'tabindex="0"' in html
    assert "function revealSamplingDetails(" in html
    assert "Image A" in html
    assert "Image B" in html
    assert "e.key !== 'Enter'" in html
    assert "e.key !== ' '" in html
    assert "e.key === 'Escape'" in html


def test_render_html_accepts_at_most_one_successful_choice_per_pair():
    html = compare_page.render_html()
    choose_body = html.split("function choose(side) {", 1)[1].split(
        "document.getElementById('pane1')", 1
    )[0]

    assert "let choiceSubmitted = false;" in html
    assert "if (!current || choiceSubmitted || zoomOpen) return;" in choose_body
    assert choose_body.index("choiceSubmitted = true;") < choose_body.index("fetch('/api/compare'")
    assert "choiceSubmitted = false;" in choose_body
    assert "choiceSubmitted = false;" in html.split("function loadNext() {", 1)[1]


def test_render_html_blocks_choices_while_zoom_is_open():
    html = compare_page.render_html()
    choose_body = html.split("function choose(side) {", 1)[1].split(
        "document.getElementById('pane1')", 1
    )[0]

    assert "if (!current || choiceSubmitted || zoomOpen) return;" in choose_body
    assert choose_body.index("zoomOpen") < choose_body.index("fetch('/api/compare'")
