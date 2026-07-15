import json
import re
import threading
from http.server import HTTPServer
from pathlib import Path
import urllib.request

import pytest

from clawmarks import curation_server as cs


def test_save_store_accepts_path_object(tmp_path):
    # SEEDS_FILE (clawmarks.config) is a pathlib.Path, unlike every other *_FILE constant in
    # curation_server.py (which are f-strings). save_store used to do `path + ".tmp"`, a TypeError
    # on Path, which silently discarded the seeds a real GPT-5.5 call had already produced.
    path = tmp_path / "seed_pool.json"
    cs.save_store(path, {"a scene": {"source": "gpt5.5"}})
    assert cs.load_store(path) == {"a scene": {"source": "gpt5.5"}}


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_active_out_dir", lambda: tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown()
    thread.join(timeout=2)


def test_seeds_generate_persists_across_the_real_save_path(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]

    def fake_run(cmd, capture_output, text, timeout):
        # Mirror what _handle_seed_generate expects: opencode writes a JSON array to the tmp_path
        # baked into the prompt string (cmd[-1]).
        prompt = cmd[-1]
        assert "Write ONLY a JSON array" in prompt
        out_path = re.search(r"the file (\S+\.json)", prompt).group(1)
        Path(out_path).write_text(json.dumps(["a quiet flooded parking lot at dusk"]))

        class Result:
            returncode = 0
            stdout = "=== DONE ==="
        return Result()

    monkeypatch.setattr(cs.subprocess, "run", fake_run)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/seeds/generate",
        data=json.dumps({"n": 1}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode())
    assert body["ok"] is True
    assert body["added"] == ["a quiet flooded parking lot at dusk"]

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/seeds") as resp:
        seeds = json.loads(resp.read().decode())
    assert "a quiet flooded parking lot at dusk" in seeds
