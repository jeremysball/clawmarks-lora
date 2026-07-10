# tests/test_rate_page.py
from clawmarks.build import rate_page


def test_main_writes_rate_html(tmp_path, monkeypatch):
    monkeypatch.setattr(rate_page, "SWEEP_DIR", tmp_path)
    rate_page.main([])
    out = tmp_path / "rate.html"
    assert out.exists()
    content = out.read_text()
    assert "/api/rate/next" in content
    assert "/api/rate" in content
