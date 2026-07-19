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

Candidate seeds (the active leg's out_dir/seed_pool.json, shared with search/driver.py) are the
pool of subject/texture
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
import html
import importlib.resources
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from clawmarks.atomic_io import atomic_json_write

from clawmarks import config
from clawmarks.config import ROOT
from clawmarks.focus_store import (
    FocusConflict,
    FocusIntegrityError,
    FocusNotFound,
    FocusStore,
    FocusValidationError,
    Scope,
)
from clawmarks.runpod_client import runpod_balance
from clawmarks.search import run_manager
from clawmarks.search.score_manifest import REAL_DIR
from clawmarks.search.seed_pool import merge as seed_pool_merge
from clawmarks.search import comparison_sampler, preference_settings, preference_pairwise_model
from clawmarks.search import embed_cache
from clawmarks.search.manifest_index import item_summary
from clawmarks.shared_ui import (
    CONTROL_CSS,
    INFOTIP_JS,
    MOBILE_BASE_CSS,
    SCROLLNAV_JS,
    SHARED_UI_JS,
    SULFUR_CSS,
    SULFUR_FONT_CSS,
    TOPNAV_CSS,
    _LIGHTBOX_JS,
    nav_bar_html,
)
from clawmarks.live_cache import LiveCache
from clawmarks.workspace_context import (
    ContextQueryError,
    WorkspaceContext,
    resolve_workspace_context,
)
from clawmarks.build import (
    scan_gallery, similarity_index, solution_map, map_view, redundancy_view, coverage_map,
    novelty_decay, lineage_view, elite_archive, preference_rank, explore_hub,
    seed_browser, compare_page, preference_status, cockpit, runs_page,
)
from clawmarks.build.thumbnails import generate_thumbnail

with open(os.path.join(os.path.dirname(__file__), "static", "favicon.png"), "rb") as _f:
    _FAVICON_PNG = _f.read()

_FONT_ASSETS = frozenset({
    "BarlowCondensed-SemiBold.ttf",
    "BarlowCondensed-ExtraBold.ttf",
    "IBMPlexSans-Variable.ttf",
    "IBMPlexMono-Regular.ttf",
    "IBMPlexMono-SemiBold.ttf",
    "LICENSE-Barlow.txt",
    "LICENSE-IBM-Plex.txt",
})

_live_cache = LiveCache()
_logger = logging.getLogger(__name__)

_active_selection = {"expedition": None, "leg": None}


def _load_active_selection():
    if config.ACTIVE_LEG_FILE.exists():
        with open(config.ACTIVE_LEG_FILE) as f:
            data = json.load(f)
        _active_selection["expedition"] = data.get("expedition")
        _active_selection["leg"] = data.get("leg")


_load_active_selection()


def _active_out_dir():
    expedition = _active_selection["expedition"]
    leg = _active_selection["leg"]
    if expedition is None or leg is None:
        return None
    try:
        _validate_expedition_or_leg_name(expedition, "expedition")
        _validate_expedition_or_leg_name(leg, "leg", reserved={"legs"})
    except (TypeError, ValueError):
        return None
    return config.leg_dir(expedition, leg)


class NoActiveLegError(Exception):
    """Raised by _require_out_dir() so do_GET/do_POST's catch-all can turn it into a clean 400
    instead of a 500 TypeError stack trace. _active_out_dir() legitimately returns None at call
    sites that already check for it (the status page, the picker); _require_out_dir() is for the
    many call sites that don't make sense without a leg selected at all."""


def _require_out_dir():
    out_dir = _active_out_dir()
    if out_dir is None:
        raise NoActiveLegError("no expedition/leg selected")
    return out_dir


def _set_active_selection(expedition, leg):
    _request_scope(expedition, leg)
    _active_selection["expedition"] = expedition
    _active_selection["leg"] = leg
    atomic_json_write(config.ACTIVE_LEG_FILE, dict(_active_selection))


def _list_expeditions():
    if not config.EXPEDITIONS_DIR.exists():
        return []
    result = []
    for expedition_dir in sorted(config.EXPEDITIONS_DIR.iterdir()):
        if not (expedition_dir / "expedition.json").exists():
            continue
        legs_dir = expedition_dir / "legs"
        legs = sorted(p.stem for p in legs_dir.glob("*.json")) if legs_dir.exists() else []
        result.append({"name": expedition_dir.name, "legs": legs})
    return result


def _validate_expedition_or_leg_name(name, kind, reserved=()):
    """Reject a name that would escape EXPEDITIONS_DIR/<expedition>/ (path separators, '..')
    or collide with a reserved path segment that directory uses for its own config
    (e.g. a leg literally named "legs" would resolve config.leg_dir() to the same directory
    that holds every other leg's legs/<leg>.json config file)."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{kind} name must be a non-empty string")
    if "\x00" in name:
        raise ValueError(f"{kind} name {name!r} may not contain NUL")
    if os.sep in name or (os.altsep and os.altsep in name) or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"{kind} name {name!r} may not contain a path separator or '..'")
    if name in reserved:
        raise ValueError(f"{kind} name {name!r} is reserved")


def _create_expedition(payload):
    name = (payload.get("name") or "").strip()
    if not name:
        raise ValueError("'name' is required")
    _validate_expedition_or_leg_name(name, "expedition")
    expedition_dir = config.EXPEDITIONS_DIR / name
    if expedition_dir.exists():
        raise ValueError(f"expedition {name!r} already exists")

    expedition_fields = {
        "trigger_word": payload.get("trigger_word", ""),
        "negative_prompt": payload.get("negative_prompt", ""),
        "textures": payload.get("textures", []),
        "fallback_subjects": payload.get("fallback_subjects", []),
        "budget_usd_cap": payload.get("budget_usd_cap", 1.0),
        "budget_safety_margin": payload.get("budget_safety_margin", 0.1),
        "gen_batch_size": payload.get("gen_batch_size", 20),
        "explore_fraction": payload.get("explore_fraction", 0.5),
        "max_generations": payload.get("max_generations", 60),
    }
    (expedition_dir / "legs").mkdir(parents=True)
    atomic_json_write(expedition_dir / "expedition.json", expedition_fields)
    atomic_json_write(expedition_dir / "legs" / "cockpit.json", {})
    _validate_expedition_or_leg_name("cockpit", "leg", reserved={"legs"})
    config.leg_dir(name, "cockpit").mkdir(parents=True, exist_ok=True)
    return {"ok": True, "name": name}


def _create_leg(payload):
    expedition = (payload.get("expedition") or "").strip()
    name = (payload.get("name") or "").strip()
    if not expedition:
        raise ValueError("'expedition' is required")
    if not name:
        raise ValueError("'name' is required")
    _validate_expedition_or_leg_name(expedition, "expedition")
    _validate_expedition_or_leg_name(name, "leg", reserved={"legs"})
    expedition_dir = config.EXPEDITIONS_DIR / expedition
    if not (expedition_dir / "expedition.json").exists():
        raise ValueError(f"unknown expedition {expedition!r}")
    leg_file = expedition_dir / "legs" / f"{name}.json"
    if leg_file.exists():
        raise ValueError(f"leg {name!r} already exists in expedition {expedition!r}")
    # Empty overrides, same as the cockpit leg _create_expedition auto-scaffolds: a leg with no
    # overrides file simply inherits every field from the expedition's own defaults.
    atomic_json_write(leg_file, {})
    config.leg_dir(expedition, name).mkdir(parents=True, exist_ok=True)
    return {"ok": True, "expedition": expedition, "name": name}


def _scope_out_dir(expedition, leg):
    if expedition is None or leg is None:
        if expedition is not None or leg is not None:
            raise ValueError("'expedition' and 'leg' must be provided together")
        legacy_dir = _active_out_dir()
        if legacy_dir is not None:
            return legacy_dir
        raise NoActiveLegError("no expedition/leg selected")
    _validate_expedition_or_leg_name(expedition, "expedition")
    _validate_expedition_or_leg_name(leg, "leg", reserved={"legs"})
    return config.leg_dir(expedition, leg)


def _request_scope(expedition, leg):
    """Validate an optional request scope before resolving any filesystem path."""
    if expedition is None and leg is None:
        active_expedition, active_leg = _active_scope()
        if active_expedition is None or active_leg is None:
            return active_expedition, active_leg
        return _request_scope(active_expedition, active_leg)
    if not isinstance(expedition, str) or not isinstance(leg, str):
        raise ValueError("'expedition' and 'leg' must be strings")
    _validate_scope_names(expedition, leg)
    expedition_dir = config.EXPEDITIONS_DIR / expedition
    if not (expedition_dir / "expedition.json").exists():
        raise ValueError(f"unknown expedition {expedition!r}")
    if not (config.EXPEDITIONS_DIR / expedition / "legs" / f"{leg}.json").exists() and not config.leg_dir(expedition, leg).exists():
        raise ValueError(f"unknown leg {leg!r} in expedition {expedition!r}")
    return expedition, leg


def _validate_scope_names(expedition, leg):
    if not isinstance(expedition, str) or not isinstance(leg, str):
        raise ValueError("'expedition' and 'leg' must be strings")
    _validate_expedition_or_leg_name(expedition, "expedition")
    _validate_expedition_or_leg_name(leg, "leg", reserved={"legs"})
    return expedition, leg


def _active_scope():
    expedition = _active_selection["expedition"]
    leg = _active_selection["leg"]
    if expedition is not None and leg is not None:
        _validate_scope_names(expedition, leg)
    active_dir = _active_out_dir()
    if active_dir is not None and (
        expedition is None
        or leg is None
        or active_dir.resolve() != config.leg_dir(expedition, leg).resolve()
    ):
        return None, None
    return expedition, leg


def _manifest_path(expedition, leg):
    return _scope_out_dir(expedition, leg) / "scored_manifest.json"


def _get_scan_items(expedition, leg):
    scope = f"{expedition}:{leg}"
    _live_cache.get(
        f"similarity:{scope}", similarity_index.compute_data,
        watched_files=[str(_manifest_path(expedition, leg))],
        sweep_dir=str(_scope_out_dir(expedition, leg)),
    )
    return _live_cache.get(
        f"scan:{scope}", scan_gallery.compute_data,
        watched_files=[str(_manifest_path(expedition, leg))],
        depends_on=[f"similarity:{scope}"],
        sweep_dir=str(_scope_out_dir(expedition, leg)),
    )


def _solution_map_watched_files(expedition, leg):
    out_dir = _scope_out_dir(expedition, leg)
    files = [str(_manifest_path(expedition, leg))]
    embs_file = str(out_dir / "solution_map_final_embs.pt")
    if os.path.exists(embs_file):
        files.append(embs_file)
    return files


def _get_solution_map_data(expedition, leg):
    scope = f"{expedition}:{leg}"
    return _live_cache.get(
        f"solution-map:{scope}", solution_map.compute_data,
        watched_files=_solution_map_watched_files(expedition, leg),
        sweep_dir=str(_scope_out_dir(expedition, leg)),
    )


def _get_map_data(expedition, leg):
    scope = f"{expedition}:{leg}"
    _get_solution_map_data(expedition, leg)
    return _live_cache.get(
        f"map:{scope}", map_view.compute_data,
        watched_files=[], depends_on=[f"solution-map:{scope}"],
        sweep_dir=str(_scope_out_dir(expedition, leg)),
    )


def _get_redundancy_data(expedition, leg):
    scope = f"{expedition}:{leg}"
    _get_solution_map_data(expedition, leg)
    return _live_cache.get(
        f"redundancy:{scope}", redundancy_view.compute_data,
        watched_files=[], depends_on=[f"solution-map:{scope}"],
        sweep_dir=str(_scope_out_dir(expedition, leg)),
    )


def _get_manifest_cached(target_name, compute_fn, expedition, leg):
    scope = f"{expedition}:{leg}"
    return _live_cache.get(
        f"{target_name}:{scope}", compute_fn,
        watched_files=[str(_manifest_path(expedition, leg))],
        sweep_dir=str(_scope_out_dir(expedition, leg)),
    )


def _prediction_watched_files(expedition=None, leg=None):
    """Like _manifest_path() alone, but also watches the trained pairwise model so a retrain
    actually invalidates any cached page whose rendering depends on the model's predictions
    (predicted archive.html, preference_rank.html) instead of serving stale predictions until
    the manifest next changes or the server restarts."""
    if expedition is None or leg is None:
        expedition, leg = _active_scope()
    files = [str(_manifest_path(expedition, leg))]
    out_dir = _scope_out_dir(expedition, leg)
    model_files = (
        preference_pairwise_model.model_file(out_dir),
        preference_pairwise_model.model_meta_file(out_dir),
    )
    files.extend(str(f) for f in model_files)
    return files


def _preference_status_watched_files(expedition, leg):
    files = []
    out_dir = _scope_out_dir(expedition, leg)
    leg_files = (
        preference_pairwise_model.model_file(out_dir),
        preference_pairwise_model.model_meta_file(out_dir),
        out_dir / "preference_settings.json",
    ) if out_dir else ()
    for f in (_comparisons_file(expedition, leg), *leg_files):
        if os.path.exists(f):
            files.append(str(f))
    return files


def _get_preference_status_data(expedition, leg):
    scope = f"{expedition}:{leg}"
    return _live_cache.get(
        f"preference-status:{scope}", preference_status.compute_data,
        watched_files=_preference_status_watched_files(expedition, leg),
        sweep_dir=str(_scope_out_dir(expedition, leg)),
    )


def _preference_retrain_gate_error(expedition=None, leg=None):
    """Mirrors preference_pairwise_model.train_and_save's own gates exactly, using
    build_training_set so a comparison referencing a tag without a cached embedding can't make
    this check pass while the real training call still has too few usable rows. Distinguishes
    three cases: "not enough comparisons yet", "comparisons exist but their embeddings aren't
    cached", and "comparisons exist and are cached but repeated judgments on the same pairs
    consolidated below the floor" (see preference_pairwise_model.n_consolidated_pairs) -- each has
    a different fix, and pointing someone at the wrong one wastes their time."""
    comparisons = load_comparisons(expedition, leg)
    n_raw_comparisons = len(comparisons)
    tags, embeddings = embed_cache.load_cache(embed_cache.embeddings_file(_scope_out_dir(expedition, leg)))
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

def _favorites_file(expedition=None, leg=None):
    if expedition is None or leg is None:
        return _require_out_dir() / "user_favorites.json"
    return _scope_out_dir(expedition, leg) / "user_favorites.json"


def _comparisons_file(expedition=None, leg=None):
    return _scope_out_dir(expedition, leg) / "user_comparisons.json"


def _preference_rank_flags_file(expedition=None, leg=None):
    return _scope_out_dir(expedition, leg) / "preference_rank_flags.json"


def _counterfactuals_dir(expedition=None, leg=None):
    return _scope_out_dir(expedition, leg) / "counterfactuals"


def _counterfactuals_file(expedition=None, leg=None):
    return _scope_out_dir(expedition, leg) / "user_counterfactuals.json"


def _cockpit_queue_file(expedition=None, leg=None):
    return _scope_out_dir(expedition, leg) / "cockpit_queue.json"


def _seeds_file(expedition=None, leg=None):
    # search/driver.py reads/writes this same file (out_dir / "seed_pool.json") as the shared
    # subject pool a leg draws from on plateau; using a different filename here silently
    # disconnects seeds topped up from this UI from anything the driver ever reads.
    return _scope_out_dir(expedition, leg) / "seed_pool.json"


DEFAULT_PORT = 8420

COMFY_ENDPOINT_ID = "uix4vdb2cec7sb"  # same serverless endpoint the search uses
COMFY_BASE = f"https://api.runpod.ai/v2/{COMFY_ENDPOINT_ID}"
BALANCE_FLOOR_USD = 0.05  # refuse to submit below this rather than risk a silent stall
GENERATION_TIMEOUT_S = 330  # a cold endpoint (scaled to zero) took ~215s to spin up a worker in testing
SEED_GEN_TIMEOUT_S = 300  # matches search/driver.py's request_gpt55_subjects timeout
NEG_DEFAULT = "low quality, blurry, watermark"

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


def load_comparisons(expedition=None, leg=None):
    if os.path.exists(_comparisons_file(expedition, leg)):
        with open(_comparisons_file(expedition, leg)) as f:
            return json.load(f)
    return []


def save_comparisons(comparisons, expedition=None, leg=None):
    target = _comparisons_file(expedition, leg)
    tmp = f"{target}.tmp"
    with open(tmp, "w") as f:
        json.dump(comparisons, f, indent=1)
    os.replace(tmp, target)


def record_comparison(comparisons, winner, loser, now):
    updated = list(comparisons)
    updated.append({"winner": winner, "loser": loser, "compared_at": now})
    return updated


_pairwise_model_cache: dict[str, Any] = {"model": None, "by_scope": {}}


def _cache_model(expedition, leg, model):
    _pairwise_model_cache["by_scope"][(expedition, leg)] = model
    # Keep the old inspection slot for existing callers and tests. Scoped reads never use it.
    _pairwise_model_cache["model"] = model


def _model_for_scope(expedition=None, leg=None):
    if expedition is None or leg is None:
        return _pairwise_model_cache["model"]
    return _pairwise_model_cache["by_scope"].get((expedition, leg))


def _embeddings_for(items, expedition=None, leg=None):
    tags, embeddings = embed_cache.load_cache(
        embed_cache.embeddings_file(_scope_out_dir(expedition, leg))
    )
    tag_to_row = {t: i for i, t in enumerate(tags)}
    idx = [tag_to_row[m["tag"]] for m in items if m["tag"] in tag_to_row]
    return embeddings[idx]


def _maybe_retrain_pairwise_model(comparisons, expedition=None, leg=None):
    """Retrains and refreshes the pairwise model cache at each training interval. Training is
    best-effort: the comparison has already been saved by the caller, so a training failure (e.g.
    a corrupt embedding cache) must not fail the comparison write. On failure the old cached model
    stays in place and the next interval retries."""
    n = len(comparisons)
    if n < comparison_sampler.MIN_COMPARISONS or n % comparison_sampler.RETRAIN_EVERY != 0:
        return
    try:
        result = preference_pairwise_model.train_and_save(
            comparisons, _scope_out_dir(expedition, leg)
        )
    except Exception as e:
        print(f"pairwise model auto-retrain failed at n={n}, keeping previous model: {e}",
              file=sys.stderr, flush=True)
        return
    if result is not None:
        with _lock:
            _cache_model(expedition, leg, result["model"])


def _compared_pair_keys(comparisons):
    return {frozenset((c["winner"], c["loser"])) for c in comparisons
            if c.get("winner") and c.get("loser")}


def next_compare_response(manifest, comparisons, expedition=None, leg=None):
    """Returns a pair of item summaries, or {"done": True} when fewer than two images exist."""
    model = _model_for_scope(expedition, leg)
    candidate_manifest = manifest
    if model is not None:
        tags, _ = embed_cache.load_cache(
            embed_cache.embeddings_file(_scope_out_dir(expedition, leg))
        )
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
         score_fn=preference_pairwise_model.score,
         embeddings_for=lambda items: _embeddings_for(items, expedition, leg), seen=seen,
        exclude=_compared_pair_keys(comparisons),
    )
    if pair is None:
        return {"done": True}
    a, b = pair
    out_dir = _scope_out_dir(expedition, leg)
    return {"img1": item_summary(a, out_dir), "img2": item_summary(b, out_dir)}


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
    focus_id = payload.get("focus_id")
    return {
        "id": trial_id, "status": "draft", "mission": mission, "queue_title": queue_title,
        "prompt": prompt, "hypothesis": (payload.get("hypothesis") or "").strip(),
        "target": payload.get("target") or "", "target_cell": payload.get("target_cell"),
        "focus_id": focus_id,
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


def _sibling_leg_exclusion_embeddings(expedition, leg, model):
    """Mirrors driver.py's _load_sibling_leg_manifests + embedding step, but scoped to the
    curation server's own long-lived DINOv2 instance (see _cockpit_scoring_context) instead
    of loading a fresh model per call."""
    from clawmarks.search.driver import _load_sibling_leg_manifests
    from clawmarks.search.score_manifest import embed_images

    class _Cfg:
        pass
    fake_cfg = _Cfg()
    fake_cfg.dir = _scope_out_dir(expedition, leg)
    fake_cfg.leg = leg

    sibling_manifest = _load_sibling_leg_manifests(fake_cfg)
    paths = [m["file"] for m in sibling_manifest if os.path.exists(m["file"])]
    if not paths:
        return None
    return embed_images(paths, model=model)


def score_cockpit_batch(results, trial, expedition, leg):
    from clawmarks.search.driver import score_batch

    model, real_embs, real_centroid = _cockpit_scoring_context()
    prev_embs = _sibling_leg_exclusion_embeddings(expedition, leg, model)
    scored = score_batch(model, real_embs, real_centroid, results, prev_embs=prev_embs)
    for m in scored:
        m["prompt_type"] = "cockpit"
        m["category"] = "cockpit"
        m["round"] = 0
        m["trial_id"] = trial["id"]
        m["mission"] = trial["mission"]
    return scored


def _load_scored_manifest(out_dir):
    path = out_dir / "scored_manifest.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_scored_manifest(out_dir, manifest):
    path = out_dir / "scored_manifest.json"
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, path)


def _run_cockpit_trial(trial_id, api_key, out_dir, queue_file, expedition, leg):
    """Runs entirely in a background thread, spawned by _handle_cockpit_run after that request
    already returned a "running" response: submits every job, polls until each completes or the
    batch times out, scores and thumbnails the results, appends them to scored_manifest.json,
    and marks the trial completed or failed. Any exception here becomes the trial's stored
    error string rather than an unhandled thread crash, since nothing is left to catch it."""
    with _lock:
        trials = load_store(queue_file)
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
                    fname = str(out_dir / f"{tag}.png")
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

        scored = score_cockpit_batch(results, trial, expedition, leg)
        thumbs_dir = out_dir / "thumbs"
        os.makedirs(thumbs_dir, exist_ok=True)
        for m in scored:
            generate_thumbnail(m["file"], thumbs_dir / f"{m['tag']}.jpg")

        with _lock:
            manifest = _load_scored_manifest(out_dir)
            manifest.extend(scored)
            _save_scored_manifest(out_dir, manifest)

            trials = load_store(queue_file)
            trials[trial_id]["status"] = "completed"
            trials[trial_id]["result_tags"] = [m["tag"] for m in scored]
            save_store(queue_file, trials)
    except Exception as e:
        with _lock:
            trials = load_store(queue_file)
            trials[trial_id]["status"] = "failed"
            trials[trial_id]["error"] = str(e)
            save_store(queue_file, trials)


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
        summary = item_summary(m, _active_out_dir())
        summary["similarity"] = round(ratio, 4)
        summary["status"] = status_of(m["tag"])
        out.append(summary)
    return out


_manifest_cache: dict[
    tuple[str | None, str | None], dict[str, object]
] = {}
_manifest_cache_lock = threading.Lock()


def load_manifest(expedition, leg):
    path = _manifest_path(expedition, leg)
    cache_key = (expedition, leg)
    with _manifest_cache_lock:
        mtime = os.path.getmtime(path)
        cached = _manifest_cache.get(cache_key)
        if cached is None or cached["mtime"] != mtime:
            with open(path) as f:
                manifest = json.load(f)
            cached = {
                "manifest": manifest,
                "by_tag": {m["tag"]: m for m in manifest},
                "mtime": mtime,
            }
            _manifest_cache[cache_key] = cached
        return cached["manifest"]


def manifest_entry_by_tag(tag, expedition, leg):
    load_manifest(expedition, leg)
    return _manifest_cache[(expedition, leg)]["by_tag"].get(tag)


def _manifest_file_in_scope(entry, out_dir):
    candidate = Path(entry["file"])
    if not candidate.is_absolute():
        candidate = out_dir / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(out_dir.resolve())
    except ValueError as exc:
        raise FileNotFoundError("manifest image is outside its leg directory") from exc
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


_ROUTES = [
    ("/scan.html", "scan gallery"),
    ("/map.html", "solution map"),
    ("/redundancy.html", "redundancy clustering"),
    ("/coverage.html", "coverage map"),
    ("/novelty_decay.html", "novelty decay"),
    ("/lineage.html", "lineage view"),
    ("/archive.html", "archive"),
    ("/preference_rank.html", "preference ranking"),
    ("/preference_status.html", "preference status"),
    ("/explore.html", "explore"),
    ("/seeds.html", "seed browser"),
    ("/compare.html", "compare"),
    ("/cockpit.html", "cockpit"),
    ("/runs.html", "runs"),
]


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so 3392 grid thumbnails don't reopen a
                                     # connection per image

    def __init__(self, *args, **kwargs):
        active_dir = _active_out_dir()
        super().__init__(*args, directory=str(active_dir) if active_dir else str(config.STATE_DIR), **kwargs)

    def end_headers(self):
        path = urllib.parse.urlparse(self.path).path
        if path.endswith((".jpg", ".jpeg", ".png")):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        elif path.endswith((".html", ".json")):
            self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def _json_response(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_image_file(self, path):
        body = path.read_bytes()
        suffix = path.suffix.lower()
        content_type = "image/png" if suffix == ".png" else "image/jpeg"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_scoped_generated_image(self, tag, context, thumbnail):
        expedition, leg = self._page_scope(context)
        out_dir = _scope_out_dir(expedition, leg)
        match = manifest_entry_by_tag(tag, expedition, leg)
        if match is None:
            self.send_error(404, "no manifest entry for this tag")
            return
        try:
            image_path = _manifest_file_in_scope(match, out_dir)
        except (KeyError, FileNotFoundError):
            self.send_error(404, "manifest image is unavailable")
            return
        if not thumbnail:
            self._send_image_file(image_path)
            return

        thumbnail_path = out_dir / "thumbs" / f"{tag}.jpg"
        if not thumbnail_path.exists():
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            generate_thumbnail(str(image_path), str(thumbnail_path))
        self._send_image_file(thumbnail_path)

    def do_GET(self):
        try:
            self._do_GET()
        except ContextQueryError as e:
            self._send_context_error(e)
        except NoActiveLegError as e:
            self._send_no_active_leg_error(e)
        except Exception as e:
            if self._wants_json():
                self._send_json_error(e)
            else:
                self._send_error_page(e, traceback.format_exc())

    def _wants_json(self):
        path = self.path.split("?")[0]
        return path.startswith("/api/") or path.endswith(".json")

    def _send_no_active_leg_error(self, exc):
        # A clean 400 for _require_out_dir()'s NoActiveLegError, instead of the generic 500
        # "Something went wrong" page a raw NoneType/str TypeError would otherwise produce.
        if self._wants_json():
            try:
                self._json_response(400, {"error": str(exc)})
            except Exception:
                pass  # client already gone; nothing left to send
            return
        body = f"""<div style="font-family:sans-serif;max-width:48rem;margin:2rem auto;line-height:1.5">
<h1>No expedition/leg selected</h1>
<p>{html.escape(str(exc))}. <a href="/">Pick one from the status page</a> first.</p>
</div>""".encode()
        try:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            pass  # client already gone; nothing left to send

    def _send_context_error(self, exc):
        body = f"<h1>Invalid workspace context</h1><p>{html.escape(str(exc))}</p>".encode()
        try:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            pass

    def _send_json_error(self, exc):
        no_manifest = isinstance(exc, FileNotFoundError) and "scored_manifest.json" in str(exc)
        try:
            self._json_response(500, {
                "error": f"{type(exc).__name__}: {exc}",
                "no_manifest": no_manifest,
            })
        except Exception:
            pass  # client already gone; nothing left to send

    def _send_error_page(self, exc, detail):
        message = f"{type(exc).__name__}: {exc}"
        hint = ""
        if isinstance(exc, FileNotFoundError):
            missing_path = str(exc).split("'")[1] if "'" in str(exc) else ""
            if missing_path.endswith("scored_manifest.json"):
                hint = (
                    "<p>The active leg has no scored manifest yet. "
                    '<a href="/">Pick a leg that has completed a search round</a>, or '
                    '<a href="/runs.html">launch a new round for this leg</a>.</p>'
                )
            else:
                hint = (
                    "<p>This usually means <code>scored_manifest.json</code> still points at an "
                    "old absolute path (e.g. after the project directory was renamed or moved) "
                    "and the image no longer lives there. Re-pointing or regenerating the "
                    "manifest's <code>file</code> paths should fix it.</p>"
                )
        body = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>clawmarks curation server: error</title>
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
main {{ max-width:48rem; margin:2rem auto; line-height:1.5; padding:0 24px; }}
h1 {{ color:var(--ink); font-size:22px; margin:0 0 12px; letter-spacing:0.02em; text-transform:uppercase; }}
h1.bad {{ color:#8a3030; }}
p {{ color:var(--text-soft); font-size:13.5px; line-height:1.6; }}
p strong {{ color:var(--ink); }}
details {{ margin-top:14px; }}
summary {{ cursor:pointer; color:var(--ink); font-weight:600; }}
pre.stack {{ white-space:pre-wrap; font-family:var(--font-mono); background:var(--paper-deep);
  padding:1rem; border:1px solid var(--rule); }}
</style></head><body>

{nav_bar_html('/status.html', active_expedition=_active_selection["expedition"],
              active_leg=_active_selection["leg"],
              running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None)}
<main>
<h1 class="bad">Something went wrong</h1>
<p>Route: <code>{html.escape(self.path)}</code></p>
<p><strong>{html.escape(message)}</strong></p>
{hint}
<details>
<summary>Show stack trace</summary>
<pre class="stack">{html.escape(detail)}</pre>
</details>
<p><a href="/">&larr; back to status page</a></p>
</main>
<script src="/shared-ui.js"></script>
</body></html>""".encode()
        try:
            self.send_response(500)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception:
            pass  # client already gone; nothing left to send

    def _send_404_page(self, path):
        body = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>clawmarks curation server: 404</title>
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
main {{ max-width:42rem; margin:2rem auto; padding:2rem;
  background:var(--paper); border:1px solid var(--ink); }}
h1 {{ font-size:22px; margin:0 0 12px; letter-spacing:0.02em; text-transform:uppercase; }}
p {{ color:var(--text-soft); font-size:13.5px; line-height:1.6; }}
</style></head><body>

{nav_bar_html('/status.html', active_expedition=_active_selection["expedition"],
              active_leg=_active_selection["leg"],
              running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None)}
<main>
<h1>Nothing here</h1>
<p>Route: <code>{html.escape(path)}</code></p>
<p>Check the address or return to the status page.</p>
<p><a href="/">Back to status page</a></p>
</main>
<script src="/shared-ui.js"></script>
</body></html>""".encode()
        self.send_response(404)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error(self, code, message=None, explain=None):
        if code == 404:
            if self.path.startswith("/api/"):
                self._json_response(404, {"error": f"unknown route: {self.path}"})
                return
            self._send_404_page(self.path)
            return
        super().send_error(code, message, explain)

    def _send_status_page(self):
        selection = _active_selection
        if selection["expedition"] is None:
            body = self._status_page_no_selection_body()
        else:
            n_entries = 0
            try:
                manifest = load_manifest(selection["expedition"], selection["leg"])
                n_entries = len(manifest)
                n_present = sum(1 for m in manifest if os.path.exists(m["file"]))
                manifest_summary = f"{n_present}/{n_entries} manifest images present on disk"
                has_data = n_present > 0
            except FileNotFoundError:
                manifest_summary = (
                    f"{selection['expedition']}/{selection['leg']} has no scored manifest yet"
                )
                has_data = False
            if has_data:
                body = self._status_page_data_body(manifest_summary)
            elif n_entries > 0:
                body = self._status_page_data_integrity_error_body(selection, n_entries)
            else:
                body = self._status_page_selected_empty_body(selection, manifest_summary)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _explore_state_records(self, directory, context):
        if context.expedition is None or context.leg is None:
            return []
        root = config.STATE_DIR / directory / context.expedition / context.leg
        records = []
        if not root.is_dir():
            return records
        for path in sorted(root.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def _explore_foci(self, context):
        if context.expedition is None or context.leg is None:
            return []
        records = self._focus_store().list(Scope(context.expedition, context.leg), status="open")
        manifest_by_tag = {}
        try:
            manifest_by_tag = {record["tag"]: record for record in load_manifest(context.expedition, context.leg)}
        except FileNotFoundError:
            pass
        enriched = []
        for record in records:
            copy = dict(record)
            source = record.get("source") or {}
            generated = source.get("member_tags") or source.get("adjacent_member_tags") or []
            anchors = source.get("real_anchor_tags") or []
            copy["evidence"] = {
                "generated_members": [
                    ({"tag": tag, "record": manifest_by_tag[tag]} if tag in manifest_by_tag else {"tag": tag, "missing": True})
                    for tag in generated
                ],
                "real_anchors": [
                    ({"tag": tag} if (Path(REAL_DIR) / tag).is_file() else {"tag": tag, "missing": True})
                    for tag in anchors
                ],
            }
            enriched.append(copy)
        return enriched

    def _send_explore_page(self):
        context = self._page_context()
        trials = []
        if context.expedition is not None and context.leg is not None:
            stored_trials = load_store(
                _scope_out_dir(context.expedition, context.leg) / "cockpit_queue.json"
            )
            if isinstance(stored_trials, dict):
                trials = list(stored_trials.values())
        data = explore_hub.build_explore_data(
            context,
            self._explore_foci(context),
            trials=trials,
            guide_threads=self._explore_state_records("guide_threads", context),
            launches=self._explore_state_records("paid_launches", context),
        )
        body = explore_hub.render_html(
            active_expedition=context.expedition,
            active_leg=context.leg,
            running=((run["expedition"], run["leg"]) if (run := run_manager.current_run()) else None),
            data=data,
            context=context,
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _page_context(self):
        query = urllib.parse.parse_qs(
            urllib.parse.urlparse(self.path).query, keep_blank_values=True
        )
        if "expedition" in query and "leg" in query:
            try:
                _validate_scope_names(query["expedition"][0], query["leg"][0])
            except (IndexError, ValueError) as e:
                raise ContextQueryError(str(e)) from e
        return resolve_workspace_context(
            self.path, _active_selection, self._focus_store()
        )

    def _page_render_context(self, context: WorkspaceContext):
        query = urllib.parse.parse_qs(
            urllib.parse.urlparse(self.path).query, keep_blank_values=True
        )
        if any(key in query for key in ("expedition", "leg", "focus_id")):
            return context
        return None

    def _page_scope(self, context: WorkspaceContext):
        query = urllib.parse.parse_qs(
            urllib.parse.urlparse(self.path).query, keep_blank_values=True
        )
        if not any(key in query for key in ("expedition", "leg", "focus_id")):
            expedition, leg = _active_scope()
            if expedition is None or leg is None:
                return expedition, leg
            try:
                return _validate_scope_names(expedition, leg)
            except ValueError as e:
                raise ContextQueryError(str(e)) from e
        try:
            return _validate_scope_names(context.expedition, context.leg)
        except ValueError as e:
            raise ContextQueryError(str(e)) from e

    def _status_page_data_body(self, manifest_summary):
        links = " &middot; ".join(f'<a href="{path}">{label}</a>' for path, label in _ROUTES)
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>clawmarks curation server</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
body {{ margin:0; padding:24px; }}
h1 {{ font-size:18px; margin:0 0 4px; letter-spacing:0.02em; text-transform:uppercase; }}
p {{ color:var(--text-soft); font-size:13px; line-height:1.6; }}
</style></head><body>

{nav_bar_html('/status.html', active_expedition=_active_selection["expedition"],
              active_leg=_active_selection["leg"],
              running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None)}
<h1>clawmarks curation server</h1>
<p>sweep dir: <code>{html.escape(str(_active_out_dir() or 'none selected'))}</code></p>
<p>{html.escape(manifest_summary)}</p>
<p id="cmpStat" class="sub">&nbsp;</p>
<script>
fetch('/api/preference_status').then(r => r.json()).then(d => {{
  const el = document.getElementById('cmpStat');
  if (typeof d.n_comparisons === 'number') {{
    const acc = (d.model_meta && typeof d.model_meta.cv_accuracy === 'number')
      ? `, model at ${{(d.model_meta.cv_accuracy * 100).toFixed(0)}}%` : '';
    el.textContent = `${{d.n_comparisons}} comparisons${{acc}}`;
  }}
}}).catch(() => {{}});
</script>
<p>{links}</p>
<script src="/shared-ui.js"></script>
</body></html>""".encode()

    def _status_page_no_selection_body(self):
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>clawmarks curation server</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
body {{ margin:0; padding:24px; }}
h1 {{ font-size:18px; margin:0 0 4px; letter-spacing:0.02em; text-transform:uppercase; }}
p {{ color:var(--text-soft); font-size:13px; line-height:1.6; }}
p.sub {{ max-width:640px; }}
.panel {{ background:var(--paper); border:1px solid var(--ink); padding:16px;
  margin-top:16px; max-width:640px; }}
</style></head><body>

{nav_bar_html('/status.html', active_expedition=_active_selection["expedition"],
              active_leg=_active_selection["leg"],
              running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None)}
<h1>clawmarks curation server</h1>
<div class="panel">
<p class="sub">No expedition/leg selected. Use "choose context" in the header above to switch
to an existing leg, or create a new expedition or leg from that same dialog.</p>
</div>
<script src="/shared-ui.js"></script>
</body></html>""".encode()

    def _status_page_selected_empty_body(self, selection, manifest_summary):
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>clawmarks curation server</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
body {{ margin:0; padding:24px; }}
h1 {{ font-size:18px; margin:0 0 4px; letter-spacing:0.02em; text-transform:uppercase; }}
p {{ color:var(--text-soft); font-size:13px; line-height:1.6; }}
p.sub {{ max-width:640px; }}
</style></head><body>

{nav_bar_html('/status.html', active_expedition=_active_selection["expedition"],
              active_leg=_active_selection["leg"],
              running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None)}
<h1>clawmarks curation server</h1>
<p>Active: <code>{html.escape(selection["expedition"])}/{html.escape(selection["leg"])}</code>,
{html.escape(manifest_summary)}. Launch a round from <a href="/runs.html">runs.html</a>,
or use "choose context" in the header above to switch to a different leg.</p>
<script src="/shared-ui.js"></script>
</body></html>""".encode()

    def _status_page_data_integrity_error_body(self, selection, n_entries):
        return f"""<!doctype html><html><head><meta charset="utf-8">
<title>clawmarks curation server</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
body {{ margin:0; padding:24px; }}
h1 {{ font-size:18px; margin:0 0 4px; letter-spacing:0.02em; text-transform:uppercase; }}
p {{ color:var(--text-soft); font-size:13px; line-height:1.6; }}
p.sub {{ max-width:640px; }}
p.alert {{ background:var(--paper-deep); border:1px solid #8a3030; padding:10px 12px;
  color:var(--ink); }}
</style></head><body>

{nav_bar_html('/status.html', active_expedition=_active_selection["expedition"],
              active_leg=_active_selection["leg"],
              running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None)}
<h1>clawmarks curation server</h1>
<p class="sub alert" role="alert"><strong>Data integrity warning:</strong> active leg
<code>{html.escape(selection["expedition"])}/{html.escape(selection["leg"])}</code> lists
{n_entries} manifest images, but none are present on disk. Do not launch a new round. Check
your backup or the state directory at <code>$XDG_STATE_HOME/clawmarks/</code> before changing
this leg.</p>
<p class="sub">Use "choose context" in the header above to switch to a different leg if
needed.</p>
<script src="/shared-ui.js"></script>
</body></html>""".encode()

    def _do_GET(self):
        route_path = urllib.parse.urlparse(self.path).path
        if self.path == "/api/active-leg":
            self._json_response(200, dict(_active_selection))
            return
        if self.path == "/api/expeditions":
            self._json_response(200, {"expeditions": _list_expeditions()})
            return
        if route_path == "/api/searchrun/status":
            self._json_response(200, run_manager.status())
            return
        if self.path.startswith("/api/searchrun/report"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            expedition = (query.get("expedition") or [None])[0]
            leg = (query.get("leg") or [None])[0]
            if not expedition or not leg:
                self._json_response(400, {"error": "'expedition' and 'leg' query params are required"})
                return
            try:
                expedition, leg = _request_scope(expedition, leg)
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            out_dir = _scope_out_dir(expedition, leg)
            favorites = load_store(out_dir / "user_favorites.json")
            self._json_response(200, run_manager.build_report(out_dir, favorites=favorites))
            return
        if route_path == "/api/compare/next":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            with _lock:
                comparisons = load_comparisons(expedition, leg)
                response = next_compare_response(
                    load_manifest(expedition, leg), comparisons, expedition, leg
                )
            self._json_response(200, response)
            return
        if route_path == "/api/favorites":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            with _lock:
                self._json_response(200, load_store(_favorites_file(expedition, leg)))
            return
        if route_path == "/api/preference_rank/flags":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            with _lock:
                self._json_response(200, load_store(_preference_rank_flags_file(expedition, leg)))
            return
        if route_path == "/api/counterfactuals":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            with _lock:
                self._json_response(200, load_store(_counterfactuals_file(expedition, leg)))
            return
        if route_path == "/api/seeds":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            with _lock:
                self._json_response(200, load_store(_seeds_file(expedition, leg)))
            return
        if route_path == "/api/cockpit/target_cells":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            coverage_data = _get_manifest_cached(
                "coverage", coverage_map.compute_data, expedition, leg,
            )
            cells = coverage_map.top_frontier_cells(coverage_data, n=3)
            self._json_response(200, {"cells": cells})
            return
        if self.path.startswith("/api/cockpit/evidence"):
            self._handle_cockpit_evidence()
            return
        if route_path == "/api/cockpit/queue":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            with _lock:
                trials = load_store(_cockpit_queue_file(expedition, leg))
            self._json_response(200, {"trials": sorted(trials.values(), key=lambda t: t["created_at"])})
            return
        if self.path.startswith("/api/foci"):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/foci":
                self._handle_focus_list(parsed)
                return
            if parsed.path.startswith("/api/foci/"):
                self._handle_focus_get(parsed)
                return
        if route_path == "/status.html":
            self._send_status_page()
            return

        if route_path == "/explore.html":
            self._send_explore_page()
            return

        if route_path == "/":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            html = scan_gallery.render_html(
                _get_scan_items(expedition, leg), context.expedition, context.leg,
                context=self._page_render_context(context),
                focus=context.focus,
            )
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path in ("/favicon.ico", "/favicon.png"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_FAVICON_PNG)))
            self.end_headers()
            self.wfile.write(_FAVICON_PNG)
            return

        _JS_ASSETS = {"/lightbox.js": _LIGHTBOX_JS, "/scrollnav.js": SCROLLNAV_JS, "/infotip.js": INFOTIP_JS, "/shared-ui.js": SHARED_UI_JS}
        if self.path in _JS_ASSETS:
            body = _JS_ASSETS[self.path].encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/assets/fonts/"):
            name = self.path[len("/assets/fonts/"):].split("?", 1)[0].split("/", 1)[0]
            if name in _FONT_ASSETS:
                body = importlib.resources.files("clawmarks").joinpath(
                    "static", "fonts", name
                ).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/ttf" if name.endswith(".ttf") else "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404, "unknown font asset")
            return

        if route_path == "/scan.html":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            html = scan_gallery.render_html(
                _get_scan_items(expedition, leg), context.expedition, context.leg,
                context=self._page_render_context(context),
                focus=context.focus,
            )
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if route_path == "/scan_data.json":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            self._json_response(200, _get_scan_items(expedition, leg))
            return

        if route_path == "/map.html":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            html = map_view.render_html(
                _get_map_data(expedition, leg), active_expedition=context.expedition,
                active_leg=context.leg,
                running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None,
                context=self._page_render_context(context),
                focus=context.focus,
            )
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route_path == "/redundancy.html":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            html = redundancy_view.render_html(
                _get_redundancy_data(expedition, leg), active_expedition=context.expedition,
                active_leg=context.leg,
                running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None,
                context=self._page_render_context(context),
                focus=context.focus,
            )
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route_path == "/coverage.html":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            html = coverage_map.render_html(
                _get_manifest_cached("coverage", coverage_map.compute_data, expedition, leg),
                active_expedition=context.expedition, active_leg=context.leg,
                running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None,
                context=self._page_render_context(context),
                focus=context.focus,
            )
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route_path == "/novelty_decay.html":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            html = novelty_decay.render_html(_get_manifest_cached("novelty_decay", novelty_decay.compute_data, expedition, leg), active_expedition=context.expedition, active_leg=context.leg, running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None, focus=context.focus)
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route_path == "/lineage.html":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            html = lineage_view.render_html(_get_manifest_cached("lineage", lineage_view.compute_data, expedition, leg), active_expedition=context.expedition, active_leg=context.leg, running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None, focus=context.focus)
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route_path == "/archive.html":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            out_dir = _scope_out_dir(expedition, leg)
            use_predicted = preference_settings.load(out_dir)["use_predicted_preference"]
            target_name = "archive_predicted" if use_predicted else "archive_actual"
            watched = _prediction_watched_files(expedition, leg) if use_predicted else [str(_manifest_path(expedition, leg))]
            data = _live_cache.get(
                f"{target_name}:{expedition}:{leg}",
                lambda sd: elite_archive.compute_data(sd, use_predicted_preference=use_predicted),
                watched_files=watched, sweep_dir=str(out_dir),
            )
            html = elite_archive.render_html(
                data, active_expedition=context.expedition, active_leg=context.leg,
                running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None,
                context=self._page_render_context(context), focus=context.focus,
            )
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route_path == "/preference_rank.html":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            data = _live_cache.get(
                f"preference_rank:{expedition}:{leg}", preference_rank.compute_data,
                watched_files=_prediction_watched_files(expedition, leg),
                sweep_dir=str(_scope_out_dir(expedition, leg)),
            )
            html = preference_rank.render_html(
                data, active_expedition=context.expedition, active_leg=context.leg,
                running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None,
                context=self._page_render_context(context), focus=context.focus,
            )
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route_path == "/preference_status.html":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            html = preference_status.render_html(_get_preference_status_data(expedition, leg), active_expedition=context.expedition, active_leg=context.leg, running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None, focus=context.focus)
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route_path == "/api/preference_status":
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            self._json_response(200, _get_preference_status_data(expedition, leg))
            return

        if route_path == "/seeds.html":
            context = self._page_context()
            body = seed_browser.render_html(
                active_expedition=context.expedition,
                active_leg=context.leg,
                running=(_run["expedition"], _run["leg"])
                if (_run := run_manager.current_run()) else None,
                focus=context.focus,
                explicit_scope=self._page_render_context(context) is not None,
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route_path == "/compare.html":
            context = self._page_context()
            body = compare_page.render_html(active_expedition=context.expedition, active_leg=context.leg, running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None, focus=context.focus).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route_path == "/cockpit.html":
            context = self._page_context()
            body = cockpit.render_html(
                expeditions=[e["name"] for e in _list_expeditions()],
                current_expedition=context.expedition,
                active_expedition=context.expedition,
                active_leg=context.leg,
                running=(_run["expedition"], _run["leg"]) if (_run := run_manager.current_run()) else None,
                focus=context.focus,
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if route_path == "/runs.html":
            context = self._page_context()
            body = runs_page.render_html(
                active_expedition=context.expedition,
                active_leg=context.leg,
                running=(_run["expedition"], _run["leg"])
                if (_run := run_manager.current_run()) else None,
                focus=context.focus,
                explicit_scope=self._page_render_context(context) is not None,
            ).encode()
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

        query = urllib.parse.parse_qs(
            urllib.parse.urlparse(self.path).query, keep_blank_values=True
        )
        has_scope_query = any(key in query for key in ("expedition", "leg", "focus_id"))

        if route_path.startswith("/generated/"):
            tag = urllib.parse.unquote(route_path[len("/generated/"):])
            if not tag or "/" in tag or ".." in tag:
                self.send_error(404, "invalid generated image tag")
                return
            self._send_scoped_generated_image(tag, self._page_context(), thumbnail=False)
            return

        if route_path.startswith("/thumbs/") and route_path.endswith(".jpg"):
            if has_scope_query:
                tag = urllib.parse.unquote(route_path[len("/thumbs/"):-len(".jpg")])
                if not tag or "/" in tag:
                    self.send_error(404, "invalid thumbnail tag")
                    return
                self._send_scoped_generated_image(tag, self._page_context(), thumbnail=True)
                return

            tag = urllib.parse.unquote(route_path[len("/thumbs/"):-len(".jpg")])
            if not tag or "/" in tag or ".." in tag:
                self.send_error(404, "invalid thumbnail tag")
                return
            thumb_path = str(_require_out_dir() / route_path.lstrip("/"))
            if not os.path.exists(thumb_path):
                match = manifest_entry_by_tag(
                    tag, *_active_scope()
                )
                if match is None:
                    self.send_error(404, "no manifest entry for this tag")
                    return
                os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
                generate_thumbnail(match["file"], thumb_path)
            # fall through to super().do_GET() below, which now finds the file on disk

        if route_path.startswith("/counterfactuals/"):
            tag = urllib.parse.unquote(route_path[len("/counterfactuals/"):])
            if not tag or "/" in tag or "\\" in tag:
                self.send_error(404, "invalid counterfactual tag")
                return
            context = self._page_context()
            expedition, leg = self._page_scope(context)
            image_path = _counterfactuals_dir(expedition, leg) / f"{tag}.png"
            if not image_path.is_file():
                self.send_error(404, "counterfactual image is unavailable")
                return
            self._send_image_file(image_path)
            return

        if self.path.startswith("/real_thumbs/"):
            # Mirrors /thumbs/ above but for REAL_DIR (corrected_dataset_extract/, read-only
            # reference data): cache writes go to the active leg's real_thumbs/, never into REAL_DIR
            # itself. basename() strips any path traversal the same way /real/ does.
            name = os.path.basename(urllib.parse.unquote(self.path[len("/real_thumbs/"):]))
            thumb_path = str(_require_out_dir() / "real_thumbs" / name)
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
        try:
            self._do_POST()
        except NoActiveLegError as e:
            self._send_no_active_leg_error(e)
        except Exception as e:
            self._send_json_error(e)

    def _do_POST(self):
        route_path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._json_response(400, {"error": "invalid JSON body"})
            return

        if self.path == "/api/active-leg":
            expedition = payload.get("expedition")
            leg = payload.get("leg")
            if not expedition or not leg:
                self._json_response(400, {"error": "'expedition' and 'leg' are required"})
                return
            try:
                _set_active_selection(expedition, leg)
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            _warn_if_manifest_images_missing()
            self._json_response(200, dict(_active_selection))
            return

        if self.path == "/api/expeditions":
            try:
                result = _create_expedition(payload)
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            self._json_response(200, result)
            return

        if self.path == "/api/legs":
            try:
                result = _create_leg(payload)
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            self._json_response(200, result)
            return

        if route_path == "/api/compare":
            winner = payload.get("winner")
            loser = payload.get("loser")
            if not winner or not loser:
                self._json_response(400, {"error": "missing 'winner' or 'loser'"})
                return
            if winner == loser:
                self._json_response(400, {"error": "'winner' and 'loser' must be different images"})
                return
            try:
                expedition, leg = _request_scope(
                    payload.get("expedition"), payload.get("leg")
                )
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            with _lock:
                comparisons = load_comparisons(expedition, leg)
                comparisons = record_comparison(comparisons, winner, loser, datetime.now(timezone.utc).isoformat())
                save_comparisons(comparisons, expedition, leg)
            # Outside _lock: a full model fit can take a while, and every other route (favorites,
            # compare, cockpit) shares this same lock, so retraining here would block them for the
            # fit's whole duration instead of just the comparison write above.
            _maybe_retrain_pairwise_model(comparisons, expedition, leg)
            self._json_response(200, {"ok": True, "count": len(comparisons)})
            return

        if route_path == "/api/preference_toggle":
            enabled = payload.get("enabled")
            if not isinstance(enabled, bool):
                self._json_response(400, {"error": "missing or non-boolean 'enabled'"})
                return
            try:
                expedition, leg = _request_scope(
                    payload.get("expedition"), payload.get("leg")
                )
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            out_dir = _scope_out_dir(expedition, leg)
            if out_dir is None:
                self._json_response(400, {"error": "no active leg selected"})
                return
            if enabled and not os.path.exists(preference_pairwise_model.model_file(out_dir)):
                self._json_response(400, {"error": "no trained model yet; cannot enable predicted preference"})
                return
            preference_settings.save(enabled, out_dir)
            self._json_response(200, _get_preference_status_data(
                expedition, leg
            ))
            return

        if route_path == "/api/preference_rank/flag":
            tag = payload.get("tag")
            flag = payload.get("flag")
            if not tag or flag not in {"matches", "questionable"}:
                self._json_response(400, {"error": "'tag' and a valid 'flag' are required"})
                return
            try:
                expedition, leg = _request_scope(
                    payload.get("expedition"), payload.get("leg")
                )
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            with _lock:
                flags = load_store(_preference_rank_flags_file(expedition, leg))
                flags[tag] = {"flag": flag, "flagged_at": datetime.now(timezone.utc).isoformat()}
                save_store(_preference_rank_flags_file(expedition, leg), flags)
            self._json_response(200, {"ok": True, "tag": tag, "flag": flag})
            return

        if route_path == "/api/preference_retrain":
            try:
                expedition, leg = _request_scope(
                    payload.get("expedition"), payload.get("leg")
                )
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            try:
                with _lock:
                    gate_error = _preference_retrain_gate_error(expedition, leg)
                    if gate_error:
                        self._json_response(400, {"error": gate_error})
                        return
                    comparisons = load_comparisons(expedition, leg)
                # Fit outside _lock: a full model fit can take a while, and every other route
                # (favorites, compare, cockpit) shares this same lock, so holding it here blocks
                # them for the fit's whole duration instead of just the state swap below.
                result = preference_pairwise_model.train_and_save(
                    comparisons, _scope_out_dir(expedition, leg)
                )
                if result is None:
                    self._json_response(500, {"error": "preference retrain failed: no usable comparisons"})
                    return
                with _lock:
                    _cache_model(expedition, leg, result["model"])
            except Exception as e:
                self._json_response(500, {"error": f"preference retrain crashed: {e}"})
                return
            self._json_response(200, _get_preference_status_data(
                expedition, leg
            ))
            return

        if route_path == "/api/favorite":
            tag = payload.get("tag")
            if not tag:
                self._json_response(400, {"error": "missing 'tag'"})
                return
            expedition = payload.get("expedition")
            leg = payload.get("leg")
            if (expedition is None) != (leg is None):
                self._json_response(400, {"error": "'expedition' and 'leg' must be provided together"})
                return
            if expedition is not None:
                try:
                    expedition, leg = _request_scope(expedition, leg)
                except ValueError as e:
                    self._json_response(400, {"error": str(e)})
                    return
            if expedition is None or leg is None:
                _logger.warning("favorite mutation without expedition/leg is deprecated")
                favorites_file = _favorites_file()
            elif payload.get("focus_id") is not None:
                try:
                    self._validate_focus_for_scope(expedition, leg, payload["focus_id"])
                except ValueError as e:
                    self._json_response(400, {"error": str(e)})
                    return
                favorites_file = _favorites_file(expedition, leg)
            elif (expedition, leg) != (_active_selection["expedition"], _active_selection["leg"]):
                self._json_response(409, {"error": "favorite mutation targets a stale expedition/leg"})
                return
            else:
                favorites_file = _favorites_file(expedition, leg)
            with _lock:
                favorites = load_store(favorites_file)
                payload["favorited_at"] = datetime.now(timezone.utc).isoformat()
                favorites[tag] = payload
                save_store(favorites_file, favorites)
            self._json_response(200, {"ok": True, "count": len(favorites)})
            return

        if route_path == "/api/unfavorite":
            tag = payload.get("tag")
            expedition = payload.get("expedition")
            leg = payload.get("leg")
            if (expedition is None) != (leg is None):
                self._json_response(400, {"error": "'expedition' and 'leg' must be provided together"})
                return
            if expedition is not None:
                try:
                    expedition, leg = _request_scope(expedition, leg)
                except ValueError as e:
                    self._json_response(400, {"error": str(e)})
                    return
            if expedition is None or leg is None:
                _logger.warning("favorite mutation without expedition/leg is deprecated")
                favorites_file = _favorites_file()
            elif payload.get("focus_id") is not None:
                try:
                    self._validate_focus_for_scope(expedition, leg, payload["focus_id"])
                except ValueError as e:
                    self._json_response(400, {"error": str(e)})
                    return
                favorites_file = _favorites_file(expedition, leg)
            elif (expedition, leg) != (_active_selection["expedition"], _active_selection["leg"]):
                self._json_response(409, {"error": "favorite mutation targets a stale expedition/leg"})
                return
            else:
                favorites_file = _favorites_file(expedition, leg)
            with _lock:
                favorites = load_store(favorites_file)
                favorites.pop(tag, None)
                save_store(favorites_file, favorites)
            self._json_response(200, {"ok": True, "count": len(favorites)})
            return

        if self.path == "/api/cockpit/queue":
            trial_id = f"trial_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            expedition = payload.get("expedition")
            leg = payload.get("leg")
            try:
                expedition, leg = _request_scope(expedition, leg)
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            try:
                self._validate_focus_for_scope(expedition, leg, payload.get("focus_id"))
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            try:
                trial = build_trial(payload, datetime.now(timezone.utc).isoformat(), trial_id)
            except ValueError as e:
                self._json_response(400, {"error": str(e)})
                return
            with _lock:
                trials = load_store(_cockpit_queue_file(expedition, leg))
                trials[trial_id] = trial
                save_store(_cockpit_queue_file(expedition, leg), trials)
            self._json_response(200, {"ok": True, "id": trial_id})
            return

        if route_path.startswith("/api/cockpit/queue/") and route_path.endswith("/run"):
            trial_id = route_path[len("/api/cockpit/queue/"):-len("/run")]
            self._handle_cockpit_run(trial_id, payload.get("expedition"), payload.get("leg"))
            return

        if self.path == "/api/foci":
            self._handle_focus_create(payload)
            return
        if self.path.startswith("/api/foci/"):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path.endswith("/archive"):
                self._handle_focus_archive(parsed, payload)
                return

        if route_path == "/api/cockpit/autopilot":
            self._handle_cockpit_autopilot(payload)
            return

        if route_path == "/api/counterfactual":
            self._handle_counterfactual(payload)
            return

        if route_path == "/api/seeds/generate":
            query = urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query, keep_blank_values=True
            )
            if any(key in query for key in ("expedition", "leg", "focus_id")):
                context = self._page_context()
                expedition, leg = self._page_scope(context)
                self._handle_seed_generate(payload, expedition, leg)
            else:
                self._handle_seed_generate(payload)
            return

        if route_path == "/api/searchrun/launch":
            self._handle_searchrun_launch(payload)
            return

        if route_path == "/api/searchrun/stop":
            self._json_response(200, run_manager.stop_run(
                pid=payload.get("pid"), start_time_ticks=payload.get("start_time_ticks")))
            return

        self._json_response(404, {"error": "unknown endpoint"})

    def do_PATCH(self):
        try:
            self._do_PATCH()
        except NoActiveLegError as e:
            self._send_no_active_leg_error(e)
        except Exception as e:
            self._send_json_error(e)

    def _do_PATCH(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._json_response(400, {"error": "invalid JSON body"})
            return

        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/foci/"):
            self._handle_focus_update(parsed, payload)
            return

        self._json_response(404, {"error": "unknown endpoint"})

    def _focus_store(self):
        """Per-request FocusStore so tests monkeypatching config.STATE_DIR between requests
        see the override instead of a captured-at-import-time value. The explicit-scope Focus
        routes never touch the active leg, so they never go through _require_out_dir() or
        _active_out_dir() for their scope."""
        return FocusStore(config.STATE_DIR, Path(REAL_DIR))

    def _validate_focus_for_scope(self, expedition, leg, focus_id):
        """Ensure an optional Focus reference belongs to the request's expedition and leg."""
        if focus_id is None:
            return
        try:
            self._focus_store().get(Scope(expedition, leg), focus_id)
        except (FocusNotFound, FocusIntegrityError, FocusValidationError) as exc:
            raise ValueError(
                f"invalid focus_id for {expedition}/{leg}: {focus_id!r}"
            ) from exc

    def _handle_focus_list(self, parsed):
        query = urllib.parse.parse_qs(parsed.query)
        expedition = (query.get("expedition") or [None])[0]
        leg = (query.get("leg") or [None])[0]
        if not expedition or not leg:
            self._json_response(400, {"error": "'expedition' and 'leg' query params are required"})
            return
        try:
            expedition, leg = _validate_scope_names(expedition, leg)
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
            return
        status = (query.get("status") or [None])[0]
        scope = Scope(expedition, leg)
        try:
            records = self._focus_store().list(scope, status=status)
        except FocusValidationError as e:
            self._json_response(400, {"error": str(e)})
            return
        except FocusIntegrityError as e:
            self._json_response(500, {"error": str(e)})
            return
        self._json_response(200, records)

    def _handle_focus_get(self, parsed):
        parts = parsed.path.split("/")
        if len(parts) != 4 or not parts[3].startswith("focus_"):
            self._json_response(404, {"error": "unknown endpoint"})
            return
        focus_id = parts[3]
        query = urllib.parse.parse_qs(parsed.query)
        expedition = (query.get("expedition") or [None])[0]
        leg = (query.get("leg") or [None])[0]
        if not expedition or not leg:
            self._json_response(400, {"error": "'expedition' and 'leg' query params are required"})
            return
        try:
            expedition, leg = _validate_scope_names(expedition, leg)
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
            return
        scope = Scope(expedition, leg)
        try:
            record = self._focus_store().get(scope, focus_id)
        except FocusNotFound as e:
            self._json_response(404, {"error": str(e)})
            return
        except FocusValidationError as e:
            self._json_response(400, {"error": str(e)})
            return
        except FocusIntegrityError as e:
            self._json_response(500, {"error": str(e)})
            return
        self._json_response(200, record)

    def _handle_focus_create(self, payload):
        scope_field = payload.get("scope") or {}
        if not isinstance(scope_field, dict):
            self._json_response(400, {"error": "'scope' must be an object"})
            return
        expedition = scope_field.get("expedition")
        leg = scope_field.get("leg")
        if not expedition or not leg:
            self._json_response(400, {"error": "'scope.expedition' and 'scope.leg' are required"})
            return
        try:
            expedition, leg = _request_scope(expedition, leg)
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
            return
        scope = Scope(expedition, leg)
        leg_dir = _scope_out_dir(expedition, leg)
        manifest_path = leg_dir / "scored_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = []

        coverage_cells = None
        source = payload.get("source")
        if isinstance(source, dict) and source.get("kind") == "coverage_frontier":
            # Recompute Coverage server-side from the explicit scoped manifest, never trust a
            # browser-supplied count/frontier/adjacency claim. The hint fields inside source
            # (row/column/binning_version, opaque display data) pass through unvalidated
            # alongside the normalized cells.
            coverage_data = coverage_map.compute_data(str(leg_dir))
            coverage_cells = coverage_data.get("cells", [])

        try:
            record = self._focus_store().create(
                scope, payload, manifest, coverage_cells=coverage_cells,
            )
        except FocusValidationError as e:
            self._json_response(400, {"error": str(e)})
            return
        except FocusIntegrityError as e:
            self._json_response(500, {"error": str(e)})
            return
        self._json_response(201, record)

    def _handle_focus_update(self, parsed, payload):
        parts = parsed.path.split("/")
        if len(parts) != 4 or not parts[3].startswith("focus_"):
            self._json_response(404, {"error": "unknown endpoint"})
            return
        focus_id = parts[3]
        query = urllib.parse.parse_qs(parsed.query)
        expedition = (query.get("expedition") or [None])[0]
        leg = (query.get("leg") or [None])[0]
        if not expedition or not leg:
            self._json_response(400, {"error": "'expedition' and 'leg' query params are required"})
            return
        try:
            expedition, leg = _validate_scope_names(expedition, leg)
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
            return
        scope = Scope(expedition, leg)
        expected_revision = payload.get("expected_revision")
        changes = payload.get("changes")
        if (
            not isinstance(changes, dict)
            or isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
        ):
            self._json_response(400, {"error": "'expected_revision' (int) and 'changes' (object) are required"})
            return
        try:
            record = self._focus_store().update(scope, focus_id, expected_revision, changes)
        except FocusValidationError as e:
            self._json_response(400, {"error": str(e)})
            return
        except FocusConflict as e:
            self._json_response(409, {"error": str(e), "current": e.current})
            return
        except FocusNotFound as e:
            self._json_response(404, {"error": str(e)})
            return
        except FocusIntegrityError as e:
            self._json_response(500, {"error": str(e)})
            return
        self._json_response(200, record)

    def _handle_focus_archive(self, parsed, payload):
        parts = parsed.path.split("/")
        # parts = ['', 'api', 'foci', '<focus_id>', 'archive']
        if len(parts) != 5 or parts[4] != "archive" or not parts[3].startswith("focus_"):
            self._json_response(404, {"error": "unknown endpoint"})
            return
        focus_id = parts[3]
        query = urllib.parse.parse_qs(parsed.query)
        expedition = (query.get("expedition") or [None])[0]
        leg = (query.get("leg") or [None])[0]
        if not expedition or not leg:
            self._json_response(400, {"error": "'expedition' and 'leg' query params are required"})
            return
        try:
            expedition, leg = _validate_scope_names(expedition, leg)
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
            return
        scope = Scope(expedition, leg)
        expected_revision = payload.get("expected_revision")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            self._json_response(400, {"error": "'expected_revision' (int) is required"})
            return
        try:
            record = self._focus_store().archive(scope, focus_id, expected_revision)
        except FocusValidationError as e:
            self._json_response(400, {"error": str(e)})
            return
        except FocusConflict as e:
            self._json_response(409, {"error": str(e), "current": e.current})
            return
        except FocusNotFound as e:
            self._json_response(404, {"error": str(e)})
            return
        except FocusIntegrityError as e:
            self._json_response(500, {"error": str(e)})
            return
        self._json_response(200, record)

    def _handle_searchrun_launch(self, payload):
        expedition = payload.get("expedition")
        leg = payload.get("leg")
        if not expedition or not leg:
            self._json_response(400, {"error": "'expedition' and 'leg' are required"})
            return
        try:
            expedition, leg = _request_scope(expedition, leg)
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
            return

        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            self._json_response(400, {"error": "RUNPOD_API_KEY not set in server environment"})
            return

        out_dir = _scope_out_dir(expedition, leg)
        try:
            info = run_manager.launch_run(
                expedition, leg, out_dir, api_key,
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
        context = self._page_context()
        expedition, leg = self._page_scope(context)
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
                coverage_data = _get_manifest_cached(
                    "coverage", coverage_map.compute_data, expedition, leg,
                )
                cell_tags = coverage_map.neighbor_tags(coverage_data, fb, nb)
        with _lock:
            favorites = load_store(_favorites_file(expedition, leg))
            comparisons = load_comparisons(expedition, leg)
        nearest = cockpit_evidence(
            load_manifest(expedition, leg),
            prompt, favorites, comparisons, cell_tags=cell_tags,
        )
        self._json_response(200, {"nearest": nearest})

    def _handle_cockpit_run(self, trial_id, expedition=None, leg=None):
        try:
            expedition, leg = _request_scope(expedition, leg)
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
            return
        api_key = os.environ.get("RUNPOD_API_KEY")
        if not api_key:
            self._json_response(400, {"error": "RUNPOD_API_KEY not set in server environment"})
            return

        out_dir = _scope_out_dir(expedition, leg)
        queue_file = _cockpit_queue_file(expedition, leg)

        with _lock:
            trials = load_store(queue_file)
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
            save_store(queue_file, trials)

        def _revert(error):
            with _lock:
                trials = load_store(queue_file)
                trials[trial_id]["status"] = prev_status
                trials[trial_id]["error"] = error
                save_store(queue_file, trials)

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

        threading.Thread(
            target=_run_cockpit_trial,
            args=(trial_id, api_key, out_dir, queue_file, expedition, leg),
            daemon=True,
        ).start()

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
            expedition, leg = _request_scope(
                payload.get("expedition"), payload.get("leg")
            )
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
            return
        focus_id = payload.get("focus_id")
        try:
            self._validate_focus_for_scope(expedition, leg, focus_id)
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
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
                    expedition, leg, focus_id,
                )
            except (RuntimeError, TimeoutError) as e:
                self._json_response(502, {"ok": False, "error": str(e), "results": results})
                return
            results.append(record)

        self._json_response(200, {"ok": True, "results": results})

    def _submit_and_wait_for_counterfactual(self, api_key, origin_tag, prompt, strength, cfg,
                                             seed, steps, sampler, negative, overridden, batch_index,
                                             expedition=None, leg=None, focus_id=None):
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
                os.makedirs(_counterfactuals_dir(expedition, leg), exist_ok=True)
                fname = str(_counterfactuals_dir(expedition, leg) / f"{new_tag}.png")
                with open(fname, "wb") as f:
                    f.write(base64.b64decode(images[0]["data"]))
                image_url = f"/counterfactuals/{urllib.parse.quote(new_tag, safe='')}"
                query = {"expedition": expedition, "leg": leg}
                if focus_id:
                    query["focus_id"] = focus_id
                if expedition is not None and leg is not None:
                    image_url += "?" + urllib.parse.urlencode(query)
                record = {
                    "tag": new_tag, "origin_tag": origin_tag, "prompt": prompt,
                    "strength": strength, "cfg": cfg, "seed": seed, "steps": steps,
                    "sampler": sampler, "negative": negative,
                    "file": image_url,
                    "overridden": overridden,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                with _lock:
                    records = load_store(_counterfactuals_file(expedition, leg))
                    records[new_tag] = record
                    save_store(_counterfactuals_file(expedition, leg), records)
                return record
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"generation job {status.lower()}: {res}")
            time.sleep(2)

        raise TimeoutError(f"generation timed out after {GENERATION_TIMEOUT_S}s")

    def _handle_seed_generate(self, payload, expedition=None, leg=None):
        n = int(payload.get("n", 20))
        n = max(1, min(n, 40))
        out_dir = _scope_out_dir(expedition, leg)
        with _lock:
            seeds = load_store(_seeds_file(expedition, leg))
        existing = list(seeds.keys())

        tmp_path = str(out_dir / f"candidate_seeds_gen_{int(time.time())}.json")
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
            seeds = load_store(_seeds_file(expedition, leg))
            updated, added = seed_pool_merge(
                seeds, new_subjects,
                source="gpt5.5",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            save_store(_seeds_file(expedition, leg), updated)
        self._json_response(200, {"ok": True, "added": added, "count": len(updated)})

    def _handle_cockpit_autopilot(self, payload=None):
        payload = payload or {}
        try:
            expedition, leg = _request_scope(
                payload.get("expedition"), payload.get("leg")
            )
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
            return
        with _lock:
            favorites = load_store(_favorites_file(expedition, leg))
            comparisons = load_comparisons(expedition, leg)
        manifest = load_manifest(expedition, leg)
        coverage_data = _get_manifest_cached(
            "coverage", coverage_map.compute_data,
            expedition, leg,
        )
        context = build_autopilot_context(coverage_data, manifest, favorites, comparisons)

        tmp_path = str(_scope_out_dir(expedition, leg) / f"cockpit_autopilot_{int(time.time())}.json")
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
    if _active_out_dir() is None:
        return  # nothing selected yet; the empty-state hub handles this case
    with _lock:
        trials = load_store(_cockpit_queue_file())
        changed = False
        for trial in trials.values():
            if trial.get("status") == "running":
                trial["status"] = "failed"
                trial["error"] = "interrupted by a server restart"
                changed = True
        if changed:
            save_store(_cockpit_queue_file(), trials)


def _check_manifest_images():
    active_dir = _active_out_dir()
    if active_dir is None:
        return  # nothing selected yet; the empty-state hub handles this case
    manifest_path = active_dir / "scored_manifest.json"
    if not manifest_path.exists():
        print(
            "warning: active leg "
            f"{_active_selection['expedition']}/{_active_selection['leg']} has no scored manifest at "
            f"{manifest_path}. Switch to a completed leg or launch a round for this leg.",
            file=sys.stderr,
            flush=True,
        )
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    n_total = len(manifest)
    if n_total == 0:
        return
    n_present = sum(1 for m in manifest if os.path.exists(m["file"]))
    if n_present == 0:
        example = manifest[0]["file"]
        print(
            f"FATAL: none of {n_total} images in {manifest_path} exist on disk "
            f"(e.g. {example!r} is missing). This usually means the manifest's paths are "
            "stale, most likely from the project directory being renamed or moved. Fix the "
            "manifest's 'file' paths, or select a different expedition/leg via "
            "POST /api/active-leg, before starting the server.",
            file=sys.stderr, flush=True,
        )
        sys.exit(1)
    if n_present < n_total:
        print(f"warning: only {n_present}/{n_total} manifest images found on disk", flush=True)


def _warn_if_manifest_images_missing():
    active_dir = _active_out_dir()
    if active_dir is None:
        return
    manifest_path = active_dir / "scored_manifest.json"
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        n_total = len(manifest)
        if n_total == 0:
            return
        n_present = sum(1 for m in manifest if os.path.exists(m["file"]))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return
    if n_present == 0:
        print(
            f"warning: none of {n_total} images in {manifest_path} exist on disk after selecting "
            f"{_active_selection['expedition']}/{_active_selection['leg']}; check backups before "
            "launching a new round",
            file=sys.stderr,
            flush=True,
        )


def main(argv=None):
    port = DEFAULT_PORT
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        port = int(argv[0])
    _check_manifest_images()
    _reconcile_stuck_trials()
    host = os.environ.get("CLAWMARKS_HOST") or tailscale_ip()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"serving on {host}:{port} (active leg: {_active_out_dir() or 'none selected'})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
