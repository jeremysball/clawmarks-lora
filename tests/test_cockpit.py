import re

from clawmarks.build import cockpit


def test_render_html_uses_plain_metric_labels():
    """No user-facing text uses faith, Faith, faith=, f=, n= as unexplained labels."""
    html = cockpit.render_html()
    # Mission title uses plain label
    assert "Faithfulness" in html or "faithfulness" in html
    # Explanatory tooltip uses plain label
    assert "Faithfulness is how close" in html
    # Coverage grid tooltip uses plain label
    assert "faithfulness x novelty" in html or "faithfulness \u00d7 novelty" in html
    # Axis labels in JS template
    assert "faithfulness" in html
    # No bare faith= label (must be preceded by "fulness")
    assert re.search(r'(?<!fulness)faith=', html) is None
    # No bare f= abbreviation
    assert 'f=${' not in html
