"""
Static file server + tiny comparison API for the uncanny-frontier scan gallery. Replaces the
plain `python3 -m http.server` that was serving notes/uncanny_sweep/ read-only: a plain static
server can't accept writes, and the whole point of this is letting a human record head-to-head
preference comparisons from the browser, which needs somewhere to persist that choice.

Comparisons are stored in notes/uncanny_sweep/user_comparisons.json, a list of
{winner, loser, compared_at} records. search/preference_pairwise_model.py trains a Bradley-
Terry-style model on this data (see
docs/superpowers/specs/2026-07-11-head-to-head-preference-design.md). The selection of which
pair to compare next is stratified across the faithfulness x novelty grid below
comparison_sampler.MIN_COMPARISONS, then model-uncertainty-guided above it; this server retrains
the model every comparison_sampler.RETRAIN_EVERY comparisons once the floor is cleared.

Favorites (notes/uncanny_sweep/user_favorites.json) are a separate store: a plain bookmark
for images worth keeping (e.g. for the writeup) without steering where the search goes next,
for when "I like this" and "build more like this" are different judgments.

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
descriptions "explore" jobs draw from. The search driver (search/driver.py) escalates to
GPT-5.5 for fresh ones on plateau, via a subprocess call to `opencode run`; this server exposes
the same mechanism on demand so the pool can be reviewed and topped up between runs, not just
mid-run. Generation is synchronous (up to 5 minutes) and calls out to opencode/GPT-5.5, so it
costs real API time but no RunPod spend.

API:
  GET  /api/compare/next       -> {"img1": item_summary, "img2": item_summary} for the next
                                   pair to compare, or {"done": true} if fewer than 2 images
                                   exist in the pool
  POST /api/compare             body: {"winner": tag, "loser": tag} -> appends a comparison
                                 record, returns {"ok": true, "count": n}
  POST /api/preference_retrain  body: {} -> trains the pairwise model on demand (same gates as
                                 preference_pairwise_model.train_and_save), returns the refreshed
                                 preference-status payload or {error} if the comparisons aren't
                                 ready yet or training crashes
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

Run with: clawmarks serve [port]
"""
import base64, json, os, random, subprocess, sys, threading, time
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from clawmarks.config import ROOT, SEEDS_FILE, SWEEP_DIR
from clawmarks.search.seed_pool import merge as seed_pool_merge
from clawmarks.search import comparison_sampler, preference_settings, preference_pairwise_model
from clawmarks.search import embed_cache
from clawmarks.search.manifest_index import item_summary
from clawmarks.shared_ui import _LIGHTBOX_JS, SCROLLNAV_JS, INFOTIP_JS
from clawmarks.live_cache import LiveCache
from clawmarks.build import (
    scan_gallery, similarity_index, solution_map, map_view, redundancy_view, coverage_map,
    novelty_decay, lineage_view, elite_archive, preference_rank, uncanny_gallery, explore_hub,
    seed_browser, compare_page, preference_status,
)
from clawmarks.build.thumbnails import generate_thumbnail

_live_cache = LiveCache()


def _manifest_path():
    return f"{SWEEP_DIR}/scored_manifest.json"


def _get_scan_items():
    _live_cache.get(
        "similarity", similarity_index.compute_data,
        watched_files=[_manifest_path()], sweep_dir=str(SWEEP_DIR),
    )
    return _live_cache.get(
        "scan", scan_gallery.compute_data,
        watched_files=[_manifest_path()], depends_on=["similarity"], sweep_dir=str(SWEEP_DIR),
    )


def _solution_map_watched_files():
    files = [_manifest_path()]
    embs_file = f"{SWEEP_DIR}/solution_map_final_embs.pt"
    if os.path.exists(embs_file):
        files.append(embs_file)
    return files


def _get_solution_map_data():
    return _live_cache.get(
        "solution-map", solution_map.compute_data,
        watched_files=_solution_map_watched_files(), sweep_dir=str(SWEEP_DIR),
    )


def _get_map_data():
    _get_solution_map_data()
    return _live_cache.get(
        "map", map_view.compute_data,
        watched_files=[], depends_on=["solution-map"], sweep_dir=str(SWEEP_DIR),
    )


def _get_redundancy_data():
    _get_solution_map_data()
    return _live_cache.get(
        "redundancy", redundancy_view.compute_data,
        watched_files=[], depends_on=["solution-map"], sweep_dir=str(SWEEP_DIR),
    )


def _get_manifest_cached(target_name, compute_fn):
    return _live_cache.get(
        target_name, compute_fn,
        watched_files=[_manifest_path()], sweep_dir=str(SWEEP_DIR),
    )


def _prediction_watched_files():
    """Like _manifest_path() alone, but also watches the trained pairwise model so a retrain
    actually invalidates any cached page whose rendering depends on the model's predictions
    (predicted archive.html, preference_rank.html) instead of serving stale predictions until
    the manifest next changes or the server restarts."""
    files = [_manifest_path()]
    for f in (preference_pairwise_model.MODEL_FILE, preference_pairwise_model.MODEL_META_FILE):
        if os.path.exists(f):
            files.append(str(f))
    return files


def _preference_status_watched_files():
    files = []
    for f in (COMPARISONS_FILE, preference_pairwise_model.MODEL_FILE,
              preference_pairwise_model.MODEL_META_FILE, preference_settings.PREFERENCE_SETTINGS_FILE):
        if os.path.exists(f):
            files.append(str(f))
    return files


def _get_preference_status_data():
    return _live_cache.get(
        "preference-status", preference_status.compute_data,
        watched_files=_preference_status_watched_files(), sweep_dir=str(SWEEP_DIR),
    )


def _preference_retrain_gate_error():
    """Mirrors preference_pairwise_model.train_and_save's own gates exactly, using
    build_training_set so a comparison referencing a tag without a cached embedding can't make
    this check pass while the real training call still has too few usable rows. Distinguishes
    "not enough comparisons yet" from "comparisons exist but their embeddings aren't cached",
    since the first is fixed by comparing more images and the second isn't, and pointing someone
    at compare.html for the wrong problem wastes their time."""
    comparisons = load_comparisons()
    n_raw_comparisons = len(comparisons)
    tags, embeddings = embed_cache.load_cache(embed_cache.EMBEDDINGS_FILE)
    _, y = preference_pairwise_model.build_training_set(tags, embeddings, comparisons)
    n_usable = len(y) // 2
    if n_usable < preference_pairwise_model.MIN_COMPARISONS:
        if n_raw_comparisons >= preference_pairwise_model.MIN_COMPARISONS:
            return (f"only {n_usable} of {n_raw_comparisons} comparisons reference images with a "
                    f"cached embedding (need {preference_pairwise_model.MIN_COMPARISONS} usable); "
                    f"run `python -m clawmarks.search.embed_cache` to refresh the embedding cache.")
        return (f"only {n_usable} usable comparisons (need "
                f"{preference_pairwise_model.MIN_COMPARISONS}); not training. Compare more images "
                f"via compare.html first.")
    return ""

FAVORITES_FILE = f"{SWEEP_DIR}/user_favorites.json"
COMPARISONS_FILE = f"{SWEEP_DIR}/user_comparisons.json"
COUNTERFACTUALS_DIR = f"{SWEEP_DIR}/counterfactuals"
COUNTERFACTUALS_FILE = f"{SWEEP_DIR}/user_counterfactuals.json"
DEFAULT_PORT = 8420

COMFY_ENDPOINT_ID = "uix4vdb2cec7sb"  # same serverless endpoint the search uses
COMFY_BASE = f"https://api.runpod.ai/v2/{COMFY_ENDPOINT_ID}"
GRAPHQL_URL = "https://api.runpod.io/graphql"
BALANCE_FLOOR_USD = 0.05  # refuse to submit below this rather than risk a silent stall
GENERATION_TIMEOUT_S = 330  # a cold endpoint (scaled to zero) took ~215s to spin up a worker in testing
SEED_GEN_TIMEOUT_S = 300  # matches search/driver.py's request_gpt55_subjects timeout
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


def load_comparisons():
    if os.path.exists(COMPARISONS_FILE):
        with open(COMPARISONS_FILE) as f:
            return json.load(f)
    return []


def save_comparisons(comparisons):
    tmp = f"{COMPARISONS_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump(comparisons, f, indent=1)
    os.replace(tmp, COMPARISONS_FILE)


def record_comparison(comparisons, winner, loser, now):
    updated = list(comparisons)
    updated.append({"winner": winner, "loser": loser, "compared_at": now})
    return updated


_pairwise_model_cache = {"model": None}


def _embeddings_for(items):
    tags, embeddings = embed_cache.load_cache(embed_cache.EMBEDDINGS_FILE)
    tag_to_row = {t: i for i, t in enumerate(tags)}
    idx = [tag_to_row[m["tag"]] for m in items if m["tag"] in tag_to_row]
    return embeddings[idx]


def _maybe_retrain_pairwise_model(comparisons):
    """Retrains and refreshes the pairwise model cache at each training interval. Training is
    best-effort: the comparison has already been saved by the caller, so a training failure (e.g.
    a corrupt embedding cache) must not fail the comparison write. On failure the old cached model
    stays in place and the next interval retries."""
    n = len(comparisons)
    if n < comparison_sampler.MIN_COMPARISONS or n % comparison_sampler.RETRAIN_EVERY != 0:
        return
    try:
        result = preference_pairwise_model.train_and_save(comparisons)
    except Exception as e:
        print(f"pairwise model auto-retrain failed at n={n}, keeping previous model: {e}",
              file=sys.stderr, flush=True)
        return
    if result is not None:
        _pairwise_model_cache["model"] = result["model"]


def next_compare_response(manifest, comparisons):
    """Returns a pair of item summaries, or {"done": True} when fewer than two images exist."""
    model = _pairwise_model_cache["model"]
    candidate_manifest = manifest
    if model is not None:
        tags, _ = embed_cache.load_cache(embed_cache.EMBEDDINGS_FILE)
        embedded = set(tags)
        embedded_manifest = [m for m in manifest if m["tag"] in embedded]
        if embedded_manifest:
            candidate_manifest = embedded_manifest
        else:
            # A model exists but no current manifest image has a cached embedding (e.g.
            # embeddings were rebuilt with new tags). The uncertainty path can score nothing,
            # so drop to stratified-random over the full manifest instead of returning done.
            model = None
    pair = comparison_sampler.pick_next_pair(
        candidate_manifest, len(comparisons), model=model,
        score_fn=preference_pairwise_model.score, embeddings_for=_embeddings_for,
    )
    if pair is None:
        return {"done": True}
    a, b = pair
    return {"img1": item_summary(a, SWEEP_DIR), "img2": item_summary(b, SWEEP_DIR)}


_manifest_cache = {"manifest": None, "mtime": None, "by_tag": None}
_manifest_cache_lock = threading.Lock()


def load_manifest():
    path = f"{SWEEP_DIR}/scored_manifest.json"
    with _manifest_cache_lock:
        mtime = os.path.getmtime(path)
        if _manifest_cache["manifest"] is None or _manifest_cache["mtime"] != mtime:
            with open(path) as f:
                manifest = json.load(f)
            _manifest_cache["manifest"] = manifest
            _manifest_cache["by_tag"] = {m["tag"]: m for m in manifest}
            _manifest_cache["mtime"] = mtime
        return _manifest_cache["manifest"]


def manifest_entry_by_tag(tag):
    load_manifest()
    return _manifest_cache["by_tag"].get(tag)


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
        if self.path == "/api/compare/next":
            with _lock:
                comparisons = load_comparisons()
                response = next_compare_response(load_manifest(), comparisons)
            self._json_response(200, response)
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

        _JS_ASSETS = {"/lightbox.js": _LIGHTBOX_JS, "/scrollnav.js": SCROLLNAV_JS, "/infotip.js": INFOTIP_JS}
        if self.path in _JS_ASSETS:
            body = _JS_ASSETS[self.path].encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/scan.html":
            html = scan_gallery.render_html(_get_scan_items())
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/scan_data.json":
            self._json_response(200, _get_scan_items())
            return

        if self.path == "/map.html":
            html = map_view.render_html(_get_map_data())
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/redundancy.html":
            html = redundancy_view.render_html(_get_redundancy_data())
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/coverage.html":
            html = coverage_map.render_html(_get_manifest_cached("coverage", coverage_map.compute_data))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/novelty_decay.html":
            html = novelty_decay.render_html(_get_manifest_cached("novelty_decay", novelty_decay.compute_data))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/lineage.html":
            html = lineage_view.render_html(_get_manifest_cached("lineage", lineage_view.compute_data))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/archive.html"):
            use_predicted = preference_settings.load()["use_predicted_preference"]
            target_name = "archive_predicted" if use_predicted else "archive_actual"
            watched = _prediction_watched_files() if use_predicted else [_manifest_path()]
            data = _live_cache.get(
                target_name,
                lambda sd: elite_archive.compute_data(sd, use_predicted_preference=use_predicted),
                watched_files=watched, sweep_dir=str(SWEEP_DIR),
            )
            html = elite_archive.render_html(data)
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/preference_rank.html":
            data = _live_cache.get(
                "preference_rank", preference_rank.compute_data,
                watched_files=_prediction_watched_files(), sweep_dir=str(SWEEP_DIR),
            )
            html = preference_rank.render_html(data)
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/preference_status.html":
            html = preference_status.render_html(_get_preference_status_data())
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/preference_status":
            self._json_response(200, _get_preference_status_data())
            return

        if self.path == "/gallery.html":
            html = uncanny_gallery.render_html(_get_manifest_cached("gallery", uncanny_gallery.compute_data))
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/explore.html":
            body = explore_hub.render_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/seeds.html":
            body = seed_browser.render_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/compare.html":
            body = compare_page.render_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/thumbs/") and self.path.endswith(".jpg"):
            thumb_path = f"{SWEEP_DIR}{self.path}"
            if not os.path.exists(thumb_path):
                tag = os.path.basename(self.path)[: -len(".jpg")]
                match = manifest_entry_by_tag(tag)
                if match is None:
                    self.send_error(404, "no manifest entry for this tag")
                    return
                os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                generate_thumbnail(match["file"], thumb_path)
            # fall through to super().do_GET() below, which now finds the file on disk

        super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._json_response(400, {"error": "invalid JSON body"})
            return

        if self.path == "/api/compare":
            winner = payload.get("winner")
            loser = payload.get("loser")
            if not winner or not loser:
                self._json_response(400, {"error": "missing 'winner' or 'loser'"})
                return
            if winner == loser:
                self._json_response(400, {"error": "'winner' and 'loser' must be different images"})
                return
            with _lock:
                comparisons = load_comparisons()
                comparisons = record_comparison(comparisons, winner, loser, datetime.now(timezone.utc).isoformat())
                save_comparisons(comparisons)
                _maybe_retrain_pairwise_model(comparisons)
            self._json_response(200, {"ok": True, "count": len(comparisons)})
            return

        if self.path == "/api/preference_toggle":
            enabled = payload.get("enabled")
            if not isinstance(enabled, bool):
                self._json_response(400, {"error": "missing or non-boolean 'enabled'"})
                return
            if enabled and not os.path.exists(preference_pairwise_model.MODEL_FILE):
                self._json_response(400, {"error": "no trained model yet; cannot enable predicted preference"})
                return
            preference_settings.save(enabled)
            self._json_response(200, _get_preference_status_data())
            return

        if self.path == "/api/preference_retrain":
            try:
                with _lock:
                    gate_error = _preference_retrain_gate_error()
                    if gate_error:
                        self._json_response(400, {"error": gate_error})
                        return
                    comparisons = load_comparisons()
                    result = preference_pairwise_model.train_and_save(comparisons)
                    if result is None:
                        self._json_response(500, {"error": "preference retrain failed: no usable comparisons"})
                        return
                    _pairwise_model_cache["model"] = result["model"]
            except Exception as e:
                self._json_response(500, {"error": f"preference retrain crashed: {e}"})
                return
            self._json_response(200, _get_preference_status_data())
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
                ["opencode", "run", "--dir", str(ROOT), "--dangerously-skip-permissions",
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

        with _lock:
            seeds = load_store(SEEDS_FILE)
            updated, added = seed_pool_merge(
                seeds, new_subjects,
                source="gpt5.5",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            save_store(SEEDS_FILE, updated)
        self._json_response(200, {"ok": True, "added": added, "count": len(updated)})

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            print(f"{self.address_string()} - {fmt % args}", flush=True)


def main(argv=None):
    port = DEFAULT_PORT
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        port = int(argv[0])
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"serving {SWEEP_DIR} + ratings API on 0.0.0.0:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
