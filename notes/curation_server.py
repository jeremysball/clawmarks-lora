"""
Static file server + tiny pick/unpick API for the uncanny-frontier scan gallery. Replaces the
plain `python3 -m http.server` that was serving notes/uncanny_sweep/ read-only: a plain static
server can't accept writes, and the whole point of this is letting a human mark "this is the
one that should win this bin" from the browser, which needs somewhere to persist that choice.

Picks are stored in notes/uncanny_sweep/user_picks.json, keyed by image tag, with the full
image metadata the browser already has (so this server doesn't need to load scored_manifest.json
itself to fill in details) plus a picked_at timestamp. A future search run reads this file and
prefers user-picked images over the automated novelty ranking when choosing which images to
mutate near ("exploit"), only falling back to raw novelty for anything the user hasn't reviewed.

Favorites (notes/uncanny_sweep/user_favorites.json) are the same shape but never read by the
search: a plain bookmark for images worth keeping (e.g. for the writeup) without steering where
the search goes next, for when "I like this" and "build more like this" are different judgments.

Counterfactuals (notes/uncanny_sweep/user_counterfactuals.json, images in
notes/uncanny_sweep/counterfactuals/) are on-demand single generations: pick an existing image,
change whichever of prompt/strength/cfg/seed you want, submit, and this server calls the same
serverless ComfyUI endpoint the search itself uses (uix4vdb2cec7sb), waits synchronously for the
one job to finish (a few seconds if a worker is already warm, up to several minutes if the
endpoint scaled to zero and needs to cold-start one), and saves the result. These are NOT scored against
the DINOv2 centroid/novelty metrics and are NOT fed back into the search; they're a quick "what
if" comparison tool, not part of the MAP-Elites archive. A RunPod balance check runs before every
submission and refuses below a safety floor rather than risk the silent-stall failure mode this
project hit once already with a negative balance.

Candidate seeds (notes/uncanny_sweep/candidate_seeds.json) are the pool of subject/texture
descriptions "explore" jobs draw from. The search driver (run_uncanny_allnight2.py) escalates to
GPT-5.5 for fresh ones on plateau, via a subprocess call to `opencode run`; this server exposes
the same mechanism on demand so the pool can be reviewed and topped up between runs, not just
mid-run. Generation is synchronous (up to 5 minutes) and calls out to opencode/GPT-5.5, so it
costs real API time but no RunPod spend.

API:
  GET  /api/picks             -> {tag: {...metadata, picked_at}}
  POST /api/pick               body: full item object (must include "tag") -> upserts, returns ok
  POST /api/unpick              body: {"tag": "..."}                        -> removes, returns ok
  GET  /api/favorites          -> {tag: {...metadata, favorited_at}}
  POST /api/favorite           body: full item object (must include "tag") -> upserts, returns ok
  POST /api/unfavorite          body: {"tag": "..."}                        -> removes, returns ok
  GET  /api/counterfactuals    -> {tag: {...record}}
  POST /api/counterfactual      body: {origin_tag, prompt, strength, cfg, seed, steps, sampler,
                                        negative, overridden: [field names]}
                                 -> generates synchronously, returns {ok, tag, file, ...record}
                                    or {error} on failure/timeout/low balance
  GET  /api/seeds              -> {text: {source, created_at}}
  POST /api/seeds/generate      body: {n: int (default 20)}
                                 -> calls GPT-5.5 for n new subjects excluding existing ones,
                                    returns {ok, added: [text, ...], count} or {error}
Everything else falls through to normal static file serving.

Run with: python3 notes/curation_server.py [port]
"""
import base64, json, os, random, subprocess, sys, threading, time
import urllib.request
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from datetime import datetime, timezone

SC = "/workspace/trent-with-smart-prompts"
SWEEP_DIR = f"{SC}/notes/uncanny_sweep"
PICKS_FILE = f"{SWEEP_DIR}/user_picks.json"
FAVORITES_FILE = f"{SWEEP_DIR}/user_favorites.json"
COUNTERFACTUALS_DIR = f"{SWEEP_DIR}/counterfactuals"
COUNTERFACTUALS_FILE = f"{SWEEP_DIR}/user_counterfactuals.json"
SEEDS_FILE = f"{SWEEP_DIR}/candidate_seeds.json"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8420

COMFY_ENDPOINT_ID = "uix4vdb2cec7sb"  # same serverless endpoint the search uses
COMFY_BASE = f"https://api.runpod.ai/v2/{COMFY_ENDPOINT_ID}"
GRAPHQL_URL = "https://api.runpod.io/graphql"
BALANCE_FLOOR_USD = 0.05  # refuse to submit below this rather than risk a silent stall
GENERATION_TIMEOUT_S = 330  # a cold endpoint (scaled to zero) took ~215s to spin up a worker in testing
SEED_GEN_TIMEOUT_S = 300  # matches run_uncanny_allnight2.py's request_gpt55_subjects timeout
NEG_DEFAULT = "low quality, blurry, watermark"

os.makedirs(COUNTERFACTUALS_DIR, exist_ok=True)
_lock = threading.Lock()


def build_workflow(prompt, seed, strength=1.0, cfg=7.5, steps=28, sampler="ddim", negative=NEG_DEFAULT):
    return {
        "input": {
            "workflow": {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "illustrious_v0.1.safetensors"}},
                "2": {"class_type": "LoraLoader", "inputs": {
                    "lora_name": "clawmarks-illustrious-v3-epoch4.safetensors",
                    "strength_model": strength, "strength_clip": strength,
                    "model": ["1", 0], "clip": ["1", 1]}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 1]}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 1]}},
                "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
                "6": {"class_type": "KSampler", "inputs": {
                    "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": "normal",
                    "denoise": 1.0, "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0],
                    "latent_image": ["5", 0]}},
                "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
                "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "counterfactual"}}
            }
        }
    }


def comfy_post(path, payload, api_key):
    req = urllib.request.Request(f"{COMFY_BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def comfy_get(path, api_key):
    req = urllib.request.Request(f"{COMFY_BASE}{path}", headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def runpod_balance(api_key):
    req = urllib.request.Request(
        f"{GRAPHQL_URL}?api_key={api_key}",
        data=json.dumps({"query": "query { myself { clientBalance } }"}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    return res["data"]["myself"]["clientBalance"]


def load_store(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_store(path, store):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f, indent=1)
    os.replace(tmp, path)


def load_picks():
    return load_store(PICKS_FILE)


def save_picks(picks):
    save_store(PICKS_FILE, picks)


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so 3392 grid thumbnails don't reopen a
                                     # connection per image

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SWEEP_DIR, **kwargs)

    def end_headers(self):
        if self.path.endswith((".jpg", ".jpeg", ".png")):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        elif self.path.endswith(".html"):
            self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def _json_response(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/picks":
            with _lock:
                self._json_response(200, load_picks())
            return
        if self.path == "/api/favorites":
            with _lock:
                self._json_response(200, load_store(FAVORITES_FILE))
            return
        if self.path == "/api/counterfactuals":
            with _lock:
                self._json_response(200, load_store(COUNTERFACTUALS_FILE))
            return
        if self.path == "/api/seeds":
            with _lock:
                self._json_response(200, load_store(SEEDS_FILE))
            return
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/scan.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._json_response(400, {"error": "invalid JSON body"})
            return

        if self.path == "/api/pick":
            tag = payload.get("tag")
            if not tag:
                self._json_response(400, {"error": "missing 'tag'"})
                return
            with _lock:
                picks = load_picks()
                payload["picked_at"] = datetime.now(timezone.utc).isoformat()
                picks[tag] = payload
                save_picks(picks)
            self._json_response(200, {"ok": True, "count": len(picks)})
            return

        if self.path == "/api/unpick":
            tag = payload.get("tag")
            with _lock:
                picks = load_picks()
                picks.pop(tag, None)
                save_picks(picks)
            self._json_response(200, {"ok": True, "count": len(picks)})
            return

        if self.path == "/api/favorite":
            tag = payload.get("tag")
            if not tag:
                self._json_response(400, {"error": "missing 'tag'"})
                return
            with _lock:
                favorites = load_store(FAVORITES_FILE)
                payload["favorited_at"] = datetime.now(timezone.utc).isoformat()
                favorites[tag] = payload
                save_store(FAVORITES_FILE, favorites)
            self._json_response(200, {"ok": True, "count": len(favorites)})
            return

        if self.path == "/api/unfavorite":
            tag = payload.get("tag")
            with _lock:
                favorites = load_store(FAVORITES_FILE)
                favorites.pop(tag, None)
                save_store(FAVORITES_FILE, favorites)
            self._json_response(200, {"ok": True, "count": len(favorites)})
            return

        if self.path == "/api/counterfactual":
            self._handle_counterfactual(payload)
            return

        if self.path == "/api/seeds/generate":
            self._handle_seed_generate(payload)
            return

        self._json_response(404, {"error": "unknown endpoint"})

    def _handle_counterfactual(self, payload):
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            self._json_response(400, {"error": "RUNPOD_API_KEY not set in server environment"})
            return

        origin_tag = payload.get("origin_tag")
        prompt = payload.get("prompt")
        if not origin_tag or not prompt:
            self._json_response(400, {"error": "missing 'origin_tag' or 'prompt'"})
            return

        try:
            balance = runpod_balance(api_key)
        except Exception as e:
            self._json_response(502, {"error": f"balance check failed: {e}"})
            return
        if balance < BALANCE_FLOOR_USD:
            self._json_response(402, {
                "error": f"RunPod balance ${balance:.4f} is below the ${BALANCE_FLOOR_USD:.2f} "
                         "safety floor. Add funds before generating (a negative/near-zero balance "
                         "has previously caused jobs to silently stall in queue instead of erroring)."
            })
            return

        strength = float(payload.get("strength", 1.0))
        cfg = float(payload.get("cfg", 7.5))
        seed = int(payload.get("seed") or random.randint(1, 999999))
        steps = int(payload.get("steps", 28))
        sampler = payload.get("sampler", "ddim")
        negative = payload.get("negative", NEG_DEFAULT)

        wf = build_workflow(prompt, seed, strength, cfg, steps, sampler, negative)
        try:
            res = comfy_post("/run", wf, api_key)
            jid = res.get("id")
        except Exception as e:
            self._json_response(502, {"error": f"submit failed: {e}"})
            return
        if not jid:
            self._json_response(502, {"error": f"submit failed: {res}"})
            return

        t0 = time.time()
        while time.time() - t0 < GENERATION_TIMEOUT_S:
            try:
                res = comfy_get(f"/status/{jid}", api_key)
            except Exception:
                time.sleep(2)
                continue
            status = res.get("status")
            if status == "COMPLETED":
                images = res.get("output", {}).get("images", [])
                if not images:
                    self._json_response(502, {"error": "job completed with no image output"})
                    return
                new_tag = f"cf_{int(time.time())}_{origin_tag[:30]}"
                fname = f"{COUNTERFACTUALS_DIR}/{new_tag}.png"
                with open(fname, "wb") as f:
                    f.write(base64.b64decode(images[0]["data"]))
                record = {
                    "tag": new_tag, "origin_tag": origin_tag, "prompt": prompt,
                    "strength": strength, "cfg": cfg, "seed": seed, "steps": steps,
                    "sampler": sampler, "negative": negative,
                    "file": f"counterfactuals/{new_tag}.png",
                    "overridden": payload.get("overridden", []),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                with _lock:
                    records = load_store(COUNTERFACTUALS_FILE)
                    records[new_tag] = record
                    save_store(COUNTERFACTUALS_FILE, records)
                self._json_response(200, {"ok": True, **record})
                return
            if status in ("FAILED", "CANCELLED"):
                self._json_response(502, {"error": f"generation job {status.lower()}: {res}"})
                return
            time.sleep(2)

        self._json_response(504, {"error": f"generation timed out after {GENERATION_TIMEOUT_S}s"})

    def _handle_seed_generate(self, payload):
        n = int(payload.get("n", 20))
        n = max(1, min(n, 40))
        with _lock:
            seeds = load_store(SEEDS_FILE)
        existing = list(seeds.keys())

        tmp_path = f"{SWEEP_DIR}/candidate_seeds_gen_{int(time.time())}.json"
        prompt = (
            f"Write {n} short, vivid, concrete visual scene or subject descriptions (5-15 words "
            f"each, no artist-style words, no medium words) suitable for testing where a "
            f"fine-tuned image-generation style survives on unfamiliar subject matter, versus "
            f"where it breaks down into visual noise. Favor liminal, uncanny, quietly unsettling "
            f"everyday scenes over gore or fantasy creatures. Prioritize genuinely different "
            f"categories of scene from each other (spaces, objects, weather, crowds, machines, "
            f"architecture), not variations on the same idea. Do not repeat or closely paraphrase "
            f"any of these already-used subjects: {json.dumps(existing)}. "
            f"Write ONLY a JSON array of {n} strings to the file {tmp_path}, nothing else in that "
            f"file. When done, print exactly: === DONE ==="
        )
        try:
            result = subprocess.run(
                ["opencode", "run", "--dir", SC, "--dangerously-skip-permissions",
                 "-m", "openai/gpt-5.5", "--", prompt],
                capture_output=True, text=True, timeout=SEED_GEN_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            self._json_response(504, {"error": f"opencode call timed out after {SEED_GEN_TIMEOUT_S}s"})
            return
        except Exception as e:
            self._json_response(502, {"error": f"failed to invoke opencode: {e}"})
            return

        if not os.path.exists(tmp_path):
            self._json_response(502, {
                "error": f"opencode exit={result.returncode}, no output file produced: "
                         f"{result.stdout[-300:]!r}"
            })
            return
        try:
            with open(tmp_path) as f:
                new_subjects = json.load(f)
        except Exception as e:
            self._json_response(502, {"error": f"couldn't parse opencode output: {e}"})
            return
        finally:
            os.remove(tmp_path)

        if not isinstance(new_subjects, list) or not new_subjects:
            self._json_response(502, {"error": f"opencode returned no usable subjects: {new_subjects!r}"})
            return

        existing_lower = {s.lower().strip() for s in existing}
        added = []
        now = datetime.now(timezone.utc).isoformat()
        with _lock:
            seeds = load_store(SEEDS_FILE)
            for s in new_subjects:
                s = str(s).strip()
                if not s or s.lower() in existing_lower:
                    continue
                seeds[s] = {"source": "gpt5.5", "created_at": now}
                existing_lower.add(s.lower())
                added.append(s)
            save_store(SEEDS_FILE, seeds)
        self._json_response(200, {"ok": True, "added": added, "count": len(seeds)})

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            print(f"{self.address_string()} - {fmt % args}", flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"serving {SWEEP_DIR} + pick API on 0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
