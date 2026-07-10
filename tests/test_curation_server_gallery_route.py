import json
import threading
from http.server import HTTPServer
import urllib.request

from PIL import Image

from clawmarks import curation_server as cs


def test_gallery_html_served_live(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    a_path = tmp_path / "a.png"
    Image.new("RGB", (32, 32), color="red").save(a_path)
    manifest = [{"file": str(a_path), "tag": "a", "prompt_name": "fox", "prompt_type": "conflict",
                 "centroid_sim": 0.5, "novelty": 0.5, "strength": 1.0, "cfg": 5.0, "steps": 28, "sampler": "ddim"}]
    (tmp_path / "scored_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "real_ref.json").write_text(json.dumps({"mean": 0.8, "min": 0.7, "max": 0.9}))
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/gallery.html") as resp:
            html = resp.read().decode()
        assert "CLAWMARKS uncanny frontier atlas" in html
    finally:
        server.shutdown()
        thread.join(timeout=2)
