from clawmarks.compute import comfyui


def test_cancel_job_posts_to_cancel_endpoint(monkeypatch):
    calls = []
    monkeypatch.setattr(comfyui, "api_post", lambda path, payload: calls.append((path, payload)))

    comfyui.cancel_job("job-1")

    assert calls == [("/cancel/job-1", {})]
