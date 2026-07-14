import json
from urllib.request import Request

from clawmarks import runpod_client


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_runpod_balance_sends_api_key_as_a_header_not_a_query_param(monkeypatch):
    """A URL query string ends up in places a header doesn't: process listings if this were ever
    shelled out, server-side access logs, browser/proxy history. Assert the API key travels as an
    Authorization header, and never appears in the request URL."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return _FakeResponse(json.dumps({"data": {"myself": {"clientBalance": 12.5}}}).encode())

    monkeypatch.setattr(runpod_client.urllib.request, "urlopen", fake_urlopen)

    balance = runpod_client.runpod_balance("secret-key-123")

    assert balance == 12.5
    req = captured["req"]
    assert isinstance(req, Request)
    assert "secret-key-123" not in req.full_url
    assert req.get_header("Authorization") == "Bearer secret-key-123"
