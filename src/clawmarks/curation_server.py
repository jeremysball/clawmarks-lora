"""
Dynamic page server + tiny comparison API for the uncanny-frontier scan gallery. Replaces the
plain `python3 -m http.server` that was serving notes/uncanny_sweep/ read-only: a plain static
server can't accept writes, and the whole point of this is letting a human record head-to-head
preference comparisons from the browser, which needs somewhere to persist that choice.

Rendering model (do NOT mistake this for a static-file server, despite the file paths below):
every .html route builds its page in-process at request time from the live manifest, via
`<view>.render_html(<view>.compute_data(...))`, and injects the view's data straight into the
returned HTML. There are no static .html files on disk and no per-page data .json to 404 on. The
one companion-JSON route is /scan_data.json, which scan.html alone fetches client-side to redraw
its grid without a full reload; every other view (map, redundancy, coverage, lineage,
novelty_decay, archive) embeds its data inline at render, so a page that looks empty is
either legitimately empty for this dataset (e.g. lineage on a single-generation seed run with no
parent_tag chains) or a client-side threshold/filter issue, never a missing file. The old data
build artifacts (solution_map_data.json, similarity.json) are gone; nothing here reads them.

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

Binds the tailscale0 interface's own IP by default (falling back to 0.0.0.0 if
tailscale isn't running), not every interface the host has — if the host also
has a LAN/Wi-Fi interface, 0.0.0.0 would let anything on that network reach
this unauthenticated server too. Set CLAWMARKS_HOST to override the default:
127.0.0.1 to front it with `tailscale serve` instead of exposing this process
directly, or 0.0.0.0 explicitly if you really want every interface (e.g. a
Docker sidecar topology where tailscale0 lives in a different container's
netns and auto-detection would otherwise fall back to 0.0.0.0 anyway).
"""
import base64
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from clawmarks.config import ROOT, SEEDS_FILE, SWEEP2_DIR, SWEEP_DIR
from clawmarks.runpod_client import runpod_balance
from clawmarks.search import run_manager
from clawmarks.search.driver import ROUND_CONFIGS
from clawmarks.search.score_manifest import REAL_DIR
from clawmarks.search.seed_pool import merge as seed_pool_merge
from clawmarks.search import comparison_sampler, preference_settings, preference_pairwise_model
from clawmarks.search import embed_cache
from clawmarks.search.manifest_index import item_summary
from clawmarks.shared_ui import _LIGHTBOX_JS, SCROLLNAV_JS, INFOTIP_JS
from clawmarks.live_cache import LiveCache
from clawmarks.build import (
    scan_gallery, similarity_index, solution_map, map_view, redundancy_view, coverage_map,
    novelty_decay, lineage_view, elite_archive, preference_rank, explore_hub,
    seed_browser, compare_page, preference_status, cockpit, runs_page,
)
from clawmarks.build.thumbnails import generate_thumbnail

with open(os.path.join(os.path.dirname(__file__), "static", "favicon.png"), "rb") as _f:
    _FAVICON_PNG = _f.read()

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
    three cases: "not enough comparisons yet", "comparisons exist but their embeddings aren't
    cached", and "comparisons exist and are cached but repeated judgments on the same pairs
    consolidated below the floor" (see preference_pairwise_model.n_consolidated_pairs) -- each has
    a different fix, and pointing someone at the wrong one wastes their time."""
    comparisons = load_comparisons()
    n_raw_comparisons = len(comparisons)
    tags, embeddings = embed_cache.load_cache(embed_cache.EMBEDDINGS_FILE)
    _, y = preference_pairwise_model.build_training_set(tags, embeddings, comparisons)
    n_usable = len(y) // 2
    if n_usable < preference_pairwise_model.MIN_COMPARISONS:
        if n_raw_comparisons >= preference_pairwise_model.MIN_COMPARISONS:
            n_consolidated = preference_pairwise_model.n_consolidated_pairs(comparisons)
            if n_consolidated < preference_pairwise_model.MIN_COMPARISONS:
                return (f"only {n_usable} distinct usable pairs after consolidating repeated "
                        f"judgments on the same pair (need {preference_pairwise_model.MIN_COMPARISONS}); "
                        f"compare more distinct pairs via compare.html first.")
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
COCKPIT_QUEUE_FILE = f"{SWEEP_DIR}/cockpit_queue.json"
DEFAULT_PORT = 8420

COMFY_ENDPOINT_ID = "uix4vdb2cec7sb"  # same serverless endpoint the search uses
COMFY_BASE = f"https://api.runpod.ai/v2/{COMFY_ENDPOINT_ID}"
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


def load_store(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_store(path, store):
    tmp = str(path) + ".tmp"
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
        with _lock:
            _pairwise_model_cache["model"] = result["model"]


def _compared_pair_keys(comparisons):
    return {frozenset((c["winner"], c["loser"])) for c in comparisons
            if c.get("winner") and c.get("loser")}


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
    seen = {}
    for c in comparisons:
        for tag in (c.get("winner"), c.get("loser")):
            if tag:
                seen[tag] = seen.get(tag, 0) + 1
    pair = comparison_sampler.pick_next_pair(
        candidate_manifest, len(comparisons), model=model,
        score_fn=preference_pairwise_model.score, embeddings_for=_embeddings_for, seen=seen,
        exclude=_compared_pair_keys(comparisons),
    )
    if pair is None:
        return {"done": True}
    a, b = pair
    return {"img1": item_summary(a, SWEEP_DIR), "img2": item_summary(b, SWEEP_DIR)}


SAMPLERS = ("ddim", "dpmpp_2m", "euler")


def build_trial(payload, now, trial_id):
    """Validates and shapes a draft trial for cockpit_queue.json. Raises ValueError with a
    user-facing message on any invalid field, so the POST handler can turn it straight into a
    400 without duplicating validation logic."""
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("missing 'prompt'")
    seed_strategy = payload.get("seed_strategy") or "random"
    if seed_strategy not in ("random", "fixed"):
        raise ValueError("seed_strategy must be 'random' or 'fixed'")
    sampler = payload.get("sampler") or "ddim"
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}")
    try:
        n = max(1, min(int(payload.get("n", 4)), 6))
        strength = float(payload.get("strength", 1.0))
        steps = int(payload.get("steps", 28))
        cfg = float(payload.get("cfg", 7.5))
    except (TypeError, ValueError):
        raise ValueError("n/strength/steps/cfg must be numbers")
    mission = payload.get("mission") or "freeform"
    queue_title = cockpit.MISSIONS.get(mission, {}).get("queue", mission)
    return {
        "id": trial_id, "status": "draft", "mission": mission, "queue_title": queue_title,
        "prompt": prompt, "hypothesis": (payload.get("hypothesis") or "").strip(),
        "target": payload.get("target") or "", "target_cell": payload.get("target_cell"),
        "seed_strategy": seed_strategy, "n": n, "strength": strength,
        "sampler": sampler, "steps": steps, "cfg": cfg,
        "negative": payload.get("negative") or NEG_DEFAULT,
        "created_at": now, "result_tags": [], "error": None,
    }


def build_generation_jobs(trial):
    """Builds one ComfyUI job per requested image. "random" draws a fresh seed per job; "fixed"
    reuses one seed across the whole batch, so a user can isolate the effect of strength/cfg
    without seed noise."""
    base_seed = trial.get("seed") or random.randint(1, 999999)
    prompt_name = f"cockpit_{trial['mission']}_{trial['id']}"
    jobs = []
    for i in range(trial["n"]):
        seed = base_seed if trial["seed_strategy"] == "fixed" else random.randint(1, 999999)
        jobs.append({
            "tag": f"{trial['id']}_{i}_{seed}", "prompt_name": prompt_name,
            "prompt": trial["prompt"], "seed": seed, "strength": trial["strength"],
            "cfg": trial["cfg"], "steps": trial["steps"], "sampler": trial["sampler"],
            "negative": trial["negative"],
        })
    return jobs


_cockpit_scoring_state = {"model": None, "real_embs": None, "real_centroid": None}
_cockpit_scoring_lock = threading.Lock()


def _cockpit_scoring_context():
    """Lazily loads DINOv2 and embeds every real training image once per server process, then
    reuses both for every later cockpit trial run. search/driver.py recomputes this fresh per
    offline sweep invocation; here that cost would otherwise repeat on every single trial run
    within a live server session, which is wasteful since the real image set doesn't change."""
    with _cockpit_scoring_lock:
        if _cockpit_scoring_state["model"] is None:
            from transformers import AutoModel
            from clawmarks.search.score_manifest import MODEL_ID, embed_images

            model = AutoModel.from_pretrained(MODEL_ID)
            model.eval()
            real_paths = sorted(
                os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            real_embs = embed_images(real_paths, model=model)
            real_centroid = real_embs.mean(dim=0)
            real_centroid = real_centroid / real_centroid.norm()
            _cockpit_scoring_state["model"] = model
            _cockpit_scoring_state["real_embs"] = real_embs
            _cockpit_scoring_state["real_centroid"] = real_centroid
        return (_cockpit_scoring_state["model"], _cockpit_scoring_state["real_embs"],
                _cockpit_scoring_state["real_centroid"])


def score_cockpit_batch(results, trial):
    from clawmarks.search.driver import score_batch

    model, real_embs, real_centroid = _cockpit_scoring_context()
    scored = score_batch(model, real_embs, real_centroid, results, prev_embs=None)
    for m in scored:
        m["prompt_type"] = "cockpit"
        m["category"] = "cockpit"
        m["round"] = 0
        m["trial_id"] = trial["id"]
        m["mission"] = trial["mission"]
    return scored


def _load_scored_manifest():
    path = f"{SWEEP_DIR}/scored_manifest.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_scored_manifest(manifest):
    path = f"{SWEEP_DIR}/scored_manifest.json"
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, path)


def _run_cockpit_trial(trial_id, api_key):
    """Runs entirely in a background thread, spawned by _handle_cockpit_run after that request
    already returned a "running" response: submits every job, polls until each completes or the
    batch times out, scores and thumbnails the results, appends them to scored_manifest.json,
    and marks the trial completed or failed. Any exception here becomes the trial's stored
    error string rather than an unhandled thread crash, since nothing is left to catch it."""
    with _lock:
        trials = load_store(COCKPIT_QUEUE_FILE)
        trial = trials[trial_id]

    try:
        jobs = build_generation_jobs(trial)
        for job in jobs:
            wf = build_workflow(job["prompt"], job["seed"], job["strength"], job["cfg"],
                                 job["steps"], job["sampler"], job["negative"])
            res = comfy_post("/run", wf, api_key)
            jid = res.get("id")
            if not jid:
                raise RuntimeError(f"submit failed for {job['tag']}: {res}")
            job["job_id"] = jid

        pending = {j["tag"]: j for j in jobs}
        results = []
        t0 = time.time()
        while pending and time.time() - t0 < GENERATION_TIMEOUT_S:
            for tag, job in list(pending.items()):
                try:
                    res = comfy_get(f"/status/{job['job_id']}", api_key)
                except Exception:
                    continue
                status = res.get("status")
                if status == "COMPLETED":
                    images = res.get("output", {}).get("images", [])
                    if not images:
                        raise RuntimeError(f"job {tag} completed with no image output")
                    fname = f"{SWEEP_DIR}/{tag}.png"
                    with open(fname, "wb") as f:
                        f.write(base64.b64decode(images[0]["data"]))
                    job["file"] = fname
                    results.append(job)
                    del pending[tag]
                elif status in ("FAILED", "CANCELLED"):
                    raise RuntimeError(f"job {tag} {status.lower()}: {res}")
            if pending:
                time.sleep(2)
        if pending:
            raise RuntimeError(f"{len(pending)} job(s) timed out after {GENERATION_TIMEOUT_S}s")

        scored = score_cockpit_batch(results, trial)
        for m in scored:
            generate_thumbnail(m["file"], f"{SWEEP_DIR}/thumbs/{m['tag']}.jpg")

        with _lock:
            manifest = _load_scored_manifest()
            manifest.extend(scored)
            _save_scored_manifest(manifest)

            trials = load_store(COCKPIT_QUEUE_FILE)
            trials[trial_id]["status"] = "completed"
            trials[trial_id]["result_tags"] = [m["tag"] for m in scored]
            save_store(COCKPIT_QUEUE_FILE, trials)
    except Exception as e:
        with _lock:
            trials = load_store(COCKPIT_QUEUE_FILE)
            trials[trial_id]["status"] = "failed"
            trials[trial_id]["error"] = str(e)
            save_store(COCKPIT_QUEUE_FILE, trials)


def build_autopilot_context(coverage_data, manifest, favorites, comparisons, n_cells=3):
    """Gathers grounding data for the Autopilot subprocess call: the most attractive frontier
    cells and a handful of recent kept/rejected prompts, so the model proposes trials grounded
    in real coverage gaps and real preference signal instead of inventing plausible ideas from
    nothing."""
    cells = coverage_map.top_frontier_cells(coverage_data, n=n_cells)
    kept_prompts = [m["prompt"] for m in manifest if m["tag"] in favorites][-5:]
    winners, losers = set(), set()
    for c in comparisons:
        if c.get("winner"):
            winners.add(c["winner"])
        if c.get("loser"):
            losers.add(c["loser"])
    rejected_tags = [t for t in losers if t not in winners]
    by_tag = {m["tag"]: m for m in manifest}
    rejected_prompts = [by_tag[t]["prompt"] for t in rejected_tags if t in by_tag][-5:]
    return {"cells": cells, "kept_prompts": kept_prompts, "rejected_prompts": rejected_prompts}


_NUMERIC_FORECAST_RE = re.compile(r"\d+(\.\d+)?\s*%|\bscore\b|\bconfidence\b|\bprobability\b", re.IGNORECASE)


def suggestion_has_numeric_forecast(rationale):
    return bool(_NUMERIC_FORECAST_RE.search(rationale or ""))


def filter_autopilot_suggestions(suggestions):
    """Drops any suggestion missing a required field, whose mission isn't one the cockpit UI
    actually knows about, or whose rationale smuggles in a numeric score/confidence/percentage:
    a cheap guard against the model ignoring the "no numeric forecast" instruction (or inventing
    a mission name) in its own prompt."""
    out = []
    for s in suggestions:
        if not isinstance(s, dict):
            continue
        if not s.get("title") or not s.get("prompt") or not isinstance(s.get("rationale"), str):
            continue
        if s.get("mission") not in cockpit.MISSIONS:
            continue
        if suggestion_has_numeric_forecast(s["rationale"]):
            continue
        out.append(s)
    return out


def cockpit_evidence(manifest, prompt, favorites, comparisons, top_n=3, cell_tags=None):
    """Ranks manifest entries by how much their prompt text overlaps with `prompt` (a plain
    difflib.SequenceMatcher ratio, since this repo has no text-embedding model, only DINOv2 image
    embeddings), and attaches an honest status: "kept" if the tag is in `favorites`, "rejected" if
    it appeared in at least one head-to-head comparison and never won one, else "unrated". Never
    fabricates a numeric preference score.

    `cell_tags`, if given, restricts the search to that set of tags (the gap mission's selected
    target cell's neighboring items, from coverage_map.neighbor_tags), so "nearby work" is
    spatially relevant to the frontier cell the curator is aiming for, not just wording-similar
    prompts from anywhere in the manifest."""
    import difflib

    wins, appearances = {}, {}
    for c in comparisons:
        winner, loser = c.get("winner"), c.get("loser")
        if winner:
            appearances[winner] = appearances.get(winner, 0) + 1
            wins[winner] = wins.get(winner, 0) + 1
        if loser:
            appearances[loser] = appearances.get(loser, 0) + 1

    def status_of(tag):
        if tag in favorites:
            return "kept"
        if appearances.get(tag, 0) > 0 and wins.get(tag, 0) == 0:
            return "rejected"
        return "unrated"

    prompt_lower = (prompt or "").lower().strip()
    if not prompt_lower:
        return []
    pool = manifest if cell_tags is None else [m for m in manifest if m["tag"] in cell_tags]
    scored = []
    for m in pool:
        ratio = difflib.SequenceMatcher(None, prompt_lower, m["prompt"].lower()).ratio()
        if ratio <= 0:
            continue
        scored.append((ratio, m))
    scored.sort(key=lambda pair: -pair[0])
    out = []
    for ratio, m in scored[:top_n]:
        summary = item_summary(m, SWEEP_DIR)
        summary["similarity"] = round(ratio, 4)
        summary["status"] = status_of(m["tag"])
        out.append(summary)
    return out


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
        if self.path == "/api/searchrun/status":
            self._json_response(200, run_manager.status())
            return
        if self.path.startswith("/api/searchrun/report"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                round_num = int((query.get("round") or ["1"])[0])
            except ValueError:
                self._json_response(400, {"error": "'round' must be an integer"})
                return
            if round_num not in ROUND_CONFIGS:
                self._json_response(400, {"error": f"unknown round {round_num!r}"})
                return
            cfg = ROUND_CONFIGS[round_num]
            out_dir = SWEEP_DIR if cfg.out_dir_name == "uncanny_sweep" else SWEEP2_DIR
            favorites = load_store(FAVORITES_FILE)
            self._json_response(200, run_manager.build_report(out_dir, favorites=favorites))
            return
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
        if self.path == "/api/cockpit/target_cells":
            coverage_data = _get_manifest_cached("coverage", coverage_map.compute_data)
            cells = coverage_map.top_frontier_cells(coverage_data, n=3)
            self._json_response(200, {"cells": cells})
            return
        if self.path.startswith("/api/cockpit/evidence"):
            self._handle_cockpit_evidence()
            return
        if self.path == "/api/cockpit/queue":
            with _lock:
                trials = load_store(COCKPIT_QUEUE_FILE)
            self._json_response(200, {"trials": sorted(trials.values(), key=lambda t: t["created_at"])})
            return
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/scan.html")
            self.end_headers()
            return

        if self.path in ("/favicon.ico", "/favicon.png"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_FAVICON_PNG)))
            self.end_headers()
            self.wfile.write(_FAVICON_PNG)
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

        if self.path == "/cockpit.html":
            body = cockpit.render_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/runs.html":
            body = runs_page.render_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/real/"):
            # basename() strips any path components a malicious/malformed request tried to smuggle
            # in (e.g. /real/../../etc/passwd), so this can only ever resolve to a direct child of
            # REAL_DIR, read-only.
            name = os.path.basename(urllib.parse.unquote(self.path[len("/real/"):]))
            real_path = os.path.join(REAL_DIR, name)
            if not name or not os.path.isfile(real_path):
                self.send_error(404, "no such real training image")
                return
            with open(real_path, "rb") as f:
                body = f.read()
            content_type = "image/png" if real_path.lower().endswith(".png") else "image/jpeg"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
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

        if self.path.startswith("/real_thumbs/"):
            # Mirrors /thumbs/ above but for REAL_DIR (corrected_dataset_extract/, read-only
            # reference data): cache writes go to SWEEP_DIR/real_thumbs/, never into REAL_DIR
            # itself. basename() strips any path traversal the same way /real/ does.
            name = os.path.basename(urllib.parse.unquote(self.path[len("/real_thumbs/"):]))
            thumb_path = f"{SWEEP_DIR}/real_thumbs/{name}"
            if not name or not os.path.exists(thumb_path):
                real_path = os.path.join(REAL_DIR, name) if name else ""
                if not name or not os.path.isfile(real_path):
                    self.send_error(404, "no such real training image")
                    return
                os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                generate_thumbnail(real_path, thumb_path)
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
            # Outside _lock: a full model fit can take a while, and every other route (favorites,
            # compare, cockpit) shares this same lock, so retraining here would block them for the
            # fit's whole duration instead of just the comparison write above.
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
                # Fit outside _lock: a full model fit can take a while, and every other route
                # (favorites, compare, cockpit) shares this same lock, so holding it here blocks
                # them for the fit's whole duration instead of just the state swap below.
                result = preference_pairwise_model.train_and_save(comparisons)
                if result is None:
                    self._json_response(500, {"error": "preference retrain failed: no usable comparisons"})
                    return
                with _lock:
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

        if self.path == "/api/cockpit/queue":
            trial_id = f"trial_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            try:
                trial = build_trial(payload, datetime.now(timezone.utc).isoformat(), trial_id)
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            with _lock:
                trials = load_store(COCKPIT_QUEUE_FILE)
                trials[trial_id] = trial
                save_store(COCKPIT_QUEUE_FILE, trials)
            self._json_response(200, {"ok": True, "id": trial_id})
            return

        if self.path.startswith("/api/cockpit/queue/") and self.path.endswith("/run"):
            trial_id = self.path[len("/api/cockpit/queue/"):-len("/run")]
            self._handle_cockpit_run(trial_id)
            return

        if self.path == "/api/cockpit/autopilot":
            self._handle_cockpit_autopilot()
            return

        if self.path == "/api/counterfactual":
            self._handle_counterfactual(payload)
            return

        if self.path == "/api/seeds/generate":
            self._handle_seed_generate(payload)
            return

        if self.path == "/api/searchrun/launch":
            self._handle_searchrun_launch(payload)
            return

        if self.path == "/api/searchrun/stop":
            self._json_response(200, run_manager.stop_run())
            return

        self._json_response(404, {"error": "unknown endpoint"})

    def _handle_searchrun_launch(self, payload):
        try:
            round_num = int(payload.get("round"))
        except (TypeError, ValueError):
            self._json_response(400, {"error": "'round' must be an integer"})
            return
        if round_num not in ROUND_CONFIGS:
            self._json_response(400, {"error": f"unknown round {round_num!r}"})
            return

        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            self._json_response(400, {"error": "RUNPOD_API_KEY not set in server environment"})
            return

        cfg = ROUND_CONFIGS[round_num]
        # Use this module's own SWEEP_DIR/SWEEP2_DIR (both overridable via CLAWMARKS_SWEEP_DIR
        # for tests) rather than driver._out_dir, which resolves against clawmarks.config
        # directly and would ignore a monkeypatch made only on this module.
        out_dir = SWEEP_DIR if cfg.out_dir_name == "uncanny_sweep" else SWEEP2_DIR
        try:
            info = run_manager.launch_run(
                round_num, out_dir, api_key,
                popen_fn=subprocess.Popen, balance_fn=run_manager.runpod_balance,
            )
        except run_manager.LaunchError as e:
            message = str(e)
            if "already in progress" in message:
                self._json_response(409, {"error": message})
            elif "below floor" in message:
                self._json_response(402, {"error": message})
            else:
                self._json_response(400, {"error": message})
            return
        self._json_response(200, {"ok": True, **info})

    def _handle_cockpit_evidence(self):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        prompt = (query.get("prompt") or [""])[0]
        cell = (query.get("cell") or [""])[0]
        cell_tags = None
        if cell:
            try:
                fb, nb = (int(x) for x in cell.split(","))
            except ValueError:
                fb = nb = None
            if fb is not None:
                coverage_data = _get_manifest_cached("coverage", coverage_map.compute_data)
                cell_tags = coverage_map.neighbor_tags(coverage_data, fb, nb)
        with _lock:
            favorites = load_store(FAVORITES_FILE)
            comparisons = load_comparisons()
        nearest = cockpit_evidence(load_manifest(), prompt, favorites, comparisons, cell_tags=cell_tags)
        self._json_response(200, {"nearest": nearest})

    def _handle_cockpit_run(self, trial_id):
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            self._json_response(400, {"error": "RUNPOD_API_KEY not set in server environment"})
            return

        with _lock:
            trials = load_store(COCKPIT_QUEUE_FILE)
            trial = trials.get(trial_id)
            if trial is None:
                self._json_response(404, {"error": f"no such trial {trial_id!r}"})
                return
            if trial["status"] == "running":
                self._json_response(409, {"error": "trial is already running"})
                return
            prev_status = trial["status"]
            trial["status"] = "running"
            trial["error"] = None
            save_store(COCKPIT_QUEUE_FILE, trials)

        def _revert(error):
            with _lock:
                trials = load_store(COCKPIT_QUEUE_FILE)
                trials[trial_id]["status"] = prev_status
                trials[trial_id]["error"] = error
                save_store(COCKPIT_QUEUE_FILE, trials)

        try:
            balance = runpod_balance(api_key)
        except Exception as e:
            _revert(f"balance check failed: {e}")
            self._json_response(502, {"error": f"balance check failed: {e}"})
            return
        if balance < BALANCE_FLOOR_USD:
            error = (
                f"RunPod balance ${balance:.4f} is below the ${BALANCE_FLOOR_USD:.2f} safety "
                "floor. Add funds before generating (a negative/near-zero balance has "
                "previously caused jobs to silently stall in queue instead of erroring)."
            )
            _revert(error)
            self._json_response(402, {"error": error})
            return

        self._json_response(200, {"ok": True, "status": "running"})

        threading.Thread(target=_run_cockpit_trial, args=(trial_id, api_key), daemon=True).start()

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
            n = int(payload.get("n", 1))
        except (TypeError, ValueError):
            self._json_response(400, {"error": "'n' must be an integer"})
            return
        n = max(1, min(n, 6))

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
        pinned_seed = payload.get("seed")
        # A pinned seed makes every job in the batch byte-identical (same prompt/strength/cfg
        # too), so honoring n>1 here would just pay for n copies of one image. Only n's random-
        # seed path benefits from batching.
        seeds = [int(pinned_seed)] if pinned_seed else [random.randint(1, 999999) for _ in range(n)]
        steps = int(payload.get("steps", 28))
        sampler = payload.get("sampler", "ddim")
        negative = payload.get("negative", NEG_DEFAULT)

        results = []
        for i, seed in enumerate(seeds):
            try:
                record = self._submit_and_wait_for_counterfactual(
                    api_key, origin_tag, prompt, strength, cfg, seed, steps, sampler, negative,
                    payload.get("overridden", []), i,
                )
            except (RuntimeError, TimeoutError) as e:
                self._json_response(502, {"ok": False, "error": str(e), "results": results})
                return
            results.append(record)

        self._json_response(200, {"ok": True, "results": results})

    def _submit_and_wait_for_counterfactual(self, api_key, origin_tag, prompt, strength, cfg,
                                             seed, steps, sampler, negative, overridden, batch_index):
        wf = build_workflow(prompt, seed, strength, cfg, steps, sampler, negative)
        try:
            res = comfy_post("/run", wf, api_key)
            jid = res.get("id")
        except Exception as e:
            raise RuntimeError(f"submit failed: {e}") from e
        if not jid:
            raise RuntimeError(f"submit failed: {res}")

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
                    raise RuntimeError("job completed with no image output")
                # uuid suffix (not just batch_index) avoids two concurrent requests for the same
                # origin_tag racing on the same filename and corrupting each other's PNG.
                new_tag = f"cf_{int(time.time())}_{batch_index}_{uuid.uuid4().hex[:8]}_{origin_tag[:30]}"
                fname = f"{COUNTERFACTUALS_DIR}/{new_tag}.png"
                with open(fname, "wb") as f:
                    f.write(base64.b64decode(images[0]["data"]))
                record = {
                    "tag": new_tag, "origin_tag": origin_tag, "prompt": prompt,
                    "strength": strength, "cfg": cfg, "seed": seed, "steps": steps,
                    "sampler": sampler, "negative": negative,
                    "file": f"counterfactuals/{new_tag}.png",
                    "overridden": overridden,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                with _lock:
                    records = load_store(COUNTERFACTUALS_FILE)
                    records[new_tag] = record
                    save_store(COUNTERFACTUALS_FILE, records)
                return record
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"generation job {status.lower()}: {res}")
            time.sleep(2)

        raise TimeoutError(f"generation timed out after {GENERATION_TIMEOUT_S}s")

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

    def _handle_cockpit_autopilot(self):
        with _lock:
            favorites = load_store(FAVORITES_FILE)
            comparisons = load_comparisons()
        manifest = load_manifest()
        coverage_data = _get_manifest_cached("coverage", coverage_map.compute_data)
        context = build_autopilot_context(coverage_data, manifest, favorites, comparisons)

        tmp_path = f"{SWEEP_DIR}/cockpit_autopilot_{int(time.time())}.json"
        prompt = (
            "You are proposing 2-3 next generation trials for a LoRA style-transfer search tool "
            "called CLAWMARKS. Ground every suggestion ONLY in the data below; do not invent "
            "scores, confidences, or percentages anywhere in your response.\n\n"
            f"Frontier coverage gaps (empty regions of the faithfulness x novelty grid, bordering "
            f"populated territory): {json.dumps(context['cells'])}\n\n"
            f"Recently kept (favorited) prompts: {json.dumps(context['kept_prompts'])}\n\n"
            "Recently rejected prompts (lost every head-to-head comparison they appeared in): "
            f"{json.dumps(context['rejected_prompts'])}\n\n"
            f"Write ONLY a JSON array of 2-3 objects to the file {tmp_path}, each shaped exactly "
            'as {"title": short label, "mission": one of "gap"/"candidate"/"lineage"/"freeform", '
            '"target_cell": [fb, nb] or null, "prompt": a full trentbuckle-style prompt string, '
            '"rationale": 1-2 sentences explaining why, with NO numbers, percentages, scores, or '
            "confidence levels}. Nothing else in that file. When done, print exactly: === DONE ==="
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
                raw_suggestions = json.load(f)
        except Exception as e:
            self._json_response(502, {"error": f"couldn't parse opencode output: {e}"})
            return
        finally:
            os.remove(tmp_path)

        if not isinstance(raw_suggestions, list):
            self._json_response(502, {
                "error": f"opencode returned no usable suggestions: {raw_suggestions!r}"
            })
            return

        self._json_response(200, {"suggestions": filter_autopilot_suggestions(raw_suggestions)})

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            print(f"{self.address_string()} - {fmt % args}", flush=True)


def tailscale_ip():
    """Reads the tailscale0 interface's IPv4 address so the server binds to the tailnet instead
    of every interface. Falls back to 0.0.0.0 if tailscale isn't up, so a laptop without
    tailscale running can still serve locally."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "tailscale0"],
                              capture_output=True, text=True, timeout=5)
        match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out.stdout)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "0.0.0.0"


def _reconcile_stuck_trials():
    """A trial's status=running is persisted before its generation thread starts (see
    _handle_cockpit_run), so a server crash or restart mid-generation leaves it stuck: no thread
    is running to ever move it to completed/failed, and the UI's 409 "already running" check
    blocks every retry forever. Called once at startup to fail those out so they're retriable."""
    with _lock:
        trials = load_store(COCKPIT_QUEUE_FILE)
        changed = False
        for trial in trials.values():
            if trial.get("status") == "running":
                trial["status"] = "failed"
                trial["error"] = "interrupted by a server restart"
                changed = True
        if changed:
            save_store(COCKPIT_QUEUE_FILE, trials)


def main(argv=None):
    port = DEFAULT_PORT
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        port = int(argv[0])
    _reconcile_stuck_trials()
    host = os.environ.get("CLAWMARKS_HOST") or tailscale_ip()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"serving {SWEEP_DIR} + ratings API on {host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
