import json
import threading
from http.server import HTTPServer
from PIL import Image
import urllib.request

from clawmarks import curation_server as cs


def test_thumb_generated_on_first_request(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    manifest = [{"file": str(tmp_path / "a.png"), "tag": "a"}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    Image.new("RGB", (500, 500), color="red").save(tmp_path / "a.png")

    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        assert not (tmp_path / "thumbs" / "a.jpg").exists()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/thumbs/a.jpg") as resp:
            assert resp.status == 200
        assert (tmp_path / "thumbs" / "a.jpg").exists()
        img = Image.open(tmp_path / "thumbs" / "a.jpg")
        assert max(img.size) <= 220
    finally:
        server.shutdown()
        thread.join(timeout=2)
