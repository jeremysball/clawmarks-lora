"""
Merged all-night driver for the CLAWMARKS liminal-band / uncanny-frontier search
(lab_notebook.md Section 3b). Runs one named "leg" of generation within a named "expedition"
(see docs/superpowers/specs/2026-07-14-expedition-leg-generation-design.md): the expedition
holds shared prompt vocab and budget defaults (expedition.json); the leg holds whatever
overrides make this run different (legs/<leg>.json), merged via load_leg_config.

Novelty for a leg's new images is scored against the real training set plus every *other*
leg's already-generated images within the same expedition (see _load_sibling_leg_manifests),
so retreading a sibling leg's already-explored region no longer counts as novel.

Run with: uv run clawmarks run allnight --expedition <name> --leg <name>
"""
import argparse
import base64
import copy
import json
import math
import os
import random
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field

from clawmarks import config as clawmarks_config
from clawmarks.atomic_io import atomic_json_write as _atomic_json_write
from clawmarks.search.scoring import bin_edges, bin_of, novelty_from_similarity
from clawmarks.search.seed_pool import merge as seed_pool_merge, load as seed_pool_load, save as seed_pool_save

TRIGGER = "trentbuckle style, "
NEG_DEFAULT = "low quality, blurry, watermark"
TYPE_COLOR = {"style": "#5ec98a", "conflict": "#e0a25e"}
N_BINS = 4
PLATEAU_WINDOW = 3
PLATEAU_EPSILON = 0.01


_LEG_CONFIG_DEFAULTS = {
    "wall_clock_cap_hours": 7.5, "budget_usd_cap": 10.0, "budget_safety_margin": 1.5,
    "gen_batch_size": 60, "explore_fraction": 0.5, "max_generations": 400,
    "textures": [], "fallback_subjects": [], "seed_from_start": False,
    "style_subject_count": 4, "description": "",
    "widened_textures": [], "widened_subjects": [],
}


@dataclass
class LegConfig:
    expedition: str
    leg: str
    dir: object  # pathlib.Path; typed loosely to avoid a circular import-time annotation
    wall_clock_cap_hours: float
    budget_usd_cap: float
    budget_safety_margin: float
    gen_batch_size: int
    explore_fraction: float
    max_generations: int
    textures: list
    fallback_subjects: list
    seed_from_start: bool
    style_subject_count: int = 4
    description: str = ""
    widened_textures: list = field(default_factory=list)
    widened_subjects: list = field(default_factory=list)


def load_leg_config(expedition, leg):
    """Loads expeditions/<expedition>/expedition.json, merges legs/<leg>.json field-by-field
    on top (leg wins), and fills in any field neither file sets from _LEG_CONFIG_DEFAULTS.
    Missing expedition.json is a hard error: launching into a typo'd expedition name must
    never silently create an empty one."""
    expedition_file = clawmarks_config.EXPEDITIONS_DIR / expedition / "expedition.json"
    if not expedition_file.exists():
        raise RuntimeError(
            f"unknown expedition {expedition!r}: {expedition_file} does not exist. "
            f"Create it via the curation server's expedition-creation form first."
        )
    with open(expedition_file) as f:
        merged = json.load(f)

    leg_file = clawmarks_config.EXPEDITIONS_DIR / expedition / "legs" / f"{leg}.json"
    if leg_file.exists():
        with open(leg_file) as f:
            leg_overrides = json.load(f)
        merged.update(leg_overrides)

    fields = copy.deepcopy(_LEG_CONFIG_DEFAULTS)
    fields.update({key: value for key, value in merged.items() if key in fields})
    return LegConfig(
        expedition=expedition, leg=leg, dir=clawmarks_config.leg_dir(expedition, leg), **fields
    )


def build_generation_jobs(gen_idx, subjects, textures, elites, user_picks, batch_size,
                          explore_fraction, style_subject_count=4):
    jobs = []
    # User picks (a human judging actual quality) take absolute priority over the automated
    # novelty ranking for choosing what to mutate near: when any user picks exist, ALL exploit
    # jobs sample from them (with replacement), not from the elite pool. Only when no user
    # picks exist do we fall back to the novelty-ranked elites.
    exploit_pool = list(user_picks) if user_picks else list(elites)

    if not exploit_pool:
        # No material to mutate near: every job is a fresh explore recombination. Matches
        # round 1's original behavior (its n_exploit = batch_size // 2 if elites else 0).
        n_exploit = 0
        n_explore = batch_size
    else:
        n_explore = round(batch_size * explore_fraction)
        n_exploit = batch_size - n_explore

    for i in range(n_exploit):
        if not exploit_pool:
            break
        e = random.choice(exploit_pool)
        strength = max(0.3, min(2.2, e["strength"] + random.gauss(0, 0.2)))
        cfg = max(1.0, min(20.0, e["cfg"] + random.gauss(0, 2.0)))
        seed = random.randint(1, 999999)
        jobs.append({
            "tag": f"gen{gen_idx}_exploit_{i}_seed{seed}", "category": "r2_exploit",
            "prompt_name": e["prompt_name"], "prompt": e["prompt"],
            "seed": seed, "strength": round(strength, 3), "cfg": round(cfg, 2),
            "steps": 28, "sampler": "ddim", "negative": NEG_DEFAULT,
            # Record which image this mutated near, so a lineage tree can show whether exploit
            # steps actually improve on their parent or just wobble. Only present on jobs built
            # after this patch landed; older images have no parent_tag.
            "parent_tag": e.get("tag"),
        })

    for i in range(n_explore):
        subj = random.choice(subjects)
        tex = random.choice(textures)
        prompt = f"{TRIGGER}{subj}, {tex}"
        strength = round(random.uniform(0.5, 2.0), 3)
        cfg = round(random.uniform(2.0, 15.0), 2)
        seed = random.randint(1, 999999)
        is_style = subj in subjects[:style_subject_count]
        pname = ("style_" if is_style else "conflict_") + subj[:24].replace(" ", "_")
        jobs.append({
            "tag": f"gen{gen_idx}_explore_{i}_seed{seed}", "category": "r2_explore",
            "prompt_name": pname, "prompt": prompt,
            "seed": seed, "strength": strength, "cfg": cfg,
            "steps": 28, "sampler": "ddim", "negative": NEG_DEFAULT,
        })
    return jobs


# --- shared helpers (originally inlined in both allnight scripts) ---

def _gql(query):
    api_key = os.environ["RUNPOD_API_KEY"]
    graphql = f"https://api.runpod.io/graphql?api_key={api_key}"
    req = urllib.request.Request(graphql, data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    if "errors" in res:
        raise RuntimeError(res["errors"])
    return res["data"]


def get_balance():
    return _gql("query { myself { clientBalance } }")["myself"]["clientBalance"]


def _spent_or_none(start_balance):
    """Returns spend-so-far, or None if the balance check itself failed. Fails closed: a caller
    that can't verify how much has been spent must stop, not assume $0 spent and let further
    paid batches start unchecked (the original bug this guards against)."""
    try:
        return start_balance - get_balance()
    except Exception as e:
        print(f"STOPPING: balance check failed ({e}); failing closed instead of assuming "
              f"$0 spent, which would let further paid batches start unchecked", flush=True)
        return None


def _out_dir(cfg):
    return cfg.dir


def _state_file(cfg):
    return cfg.dir / "allnight_state.json"


def load_state(cfg):
    state_file = _state_file(cfg)
    if not state_file.exists():
        return _new_state()
    try:
        with open(state_file) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"cannot resume: persisted state {state_file} is unreadable: {e}") from e
    _validate_state(state, state_file)
    return state


def _new_state():
    return {
        "generation": 0, "stage": 0, "plateau_count": 0,
        "novelty_history": [], "gpt55_subjects": [], "start_balance": None,
        "start_time": time.time(),
    }


def _validate_state(state, state_file):
    required = {
        "generation", "stage", "plateau_count", "novelty_history", "gpt55_subjects",
        "start_balance", "start_time",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise RuntimeError(
            f"cannot resume: persisted state {state_file} is malformed or missing required fields"
        )
    for name in ("generation", "stage", "plateau_count"):
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"cannot resume: persisted state {state_file} has invalid {name}")
    if not isinstance(state["novelty_history"], list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in state["novelty_history"]
    ):
        raise RuntimeError(f"cannot resume: persisted state {state_file} has invalid novelty_history")
    if len(state["novelty_history"]) != state["generation"]:
        raise RuntimeError(
            f"cannot resume: persisted state {state_file} has generation/history mismatch"
        )
    if not isinstance(state["gpt55_subjects"], list) or any(
        not isinstance(value, str) for value in state["gpt55_subjects"]
    ):
        raise RuntimeError(f"cannot resume: persisted state {state_file} has invalid gpt55_subjects")
    balance = state["start_balance"]
    if balance is not None and (
        isinstance(balance, bool) or not isinstance(balance, (int, float)) or not math.isfinite(balance)
    ):
        raise RuntimeError(f"cannot resume: persisted state {state_file} has invalid start_balance")
    start_time = state["start_time"]
    if isinstance(start_time, bool) or not isinstance(start_time, (int, float)) or not math.isfinite(start_time):
        raise RuntimeError(f"cannot resume: persisted state {state_file} has invalid start_time")
    if state["generation"] > 0 and balance is None:
        raise RuntimeError(
            f"cannot resume: persisted state {state_file} has generations but no start_balance"
        )


def save_state(cfg, state):
    state_file = _state_file(cfg)
    _validate_state(state, state_file)
    _atomic_json_write(state_file, state)


def _load_resumable_manifest(out_dir):
    """Loads this round's own already-persisted scored_manifest.json, if any, so restarting the
    search resumes on top of prior generations instead of discarding them: the main loop used to
    always start from an empty manifest and then overwrite scored_manifest.json with only the
    new run's images, permanently losing every previously persisted record on every restart.
    Refuses to continue if the file is truncated, corrupt, or has an invalid entry. Starting a
    paid run with an unknown prior manifest could overwrite the only trustworthy record of the
    search, so resume errors fail closed and leave the original file untouched."""
    manifest_path = out_dir / "scored_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"cannot resume: persisted manifest {manifest_path} is unreadable: {e}") from e
    _validate_manifest(manifest, manifest_path)
    print(f"resuming with {len(manifest)} already-persisted images from {manifest_path}", flush=True)
    return manifest


_GENERATION_TAG = re.compile(r"^gen([1-9][0-9]*)_")


def _current_driver_generation(tag):
    match = _GENERATION_TAG.match(tag)
    return int(match.group(1)) if match else None


def _validate_manifest(manifest, manifest_path):
    if not isinstance(manifest, list):
        raise RuntimeError(f"cannot resume: persisted manifest {manifest_path} is not a list")
    required = {
        "tag", "file", "prompt_name", "prompt", "seed", "strength", "cfg", "steps",
        "sampler", "negative", "centroid_sim", "novelty", "prompt_type",
    }
    tags = set()
    for entry in manifest:
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise RuntimeError(f"cannot resume: persisted manifest {manifest_path} has malformed entry")
        tag = entry["tag"]
        if not isinstance(tag, str) or not tag or tag in tags:
            raise RuntimeError(f"cannot resume: persisted manifest {manifest_path} has invalid tag")
        tags.add(tag)
        if not isinstance(entry["file"], str) or not entry["file"]:
            raise RuntimeError(f"cannot resume: persisted manifest {manifest_path} has invalid file")
        for name in ("centroid_sim", "novelty", "strength", "cfg"):
            value = entry[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"cannot resume: persisted manifest {manifest_path} has invalid {name}")
        for name in ("prompt_name", "prompt", "sampler", "negative", "prompt_type"):
            if not isinstance(entry[name], str):
                raise RuntimeError(f"cannot resume: persisted manifest {manifest_path} has invalid {name}")
        for name in ("seed", "steps"):
            if isinstance(entry[name], bool) or not isinstance(entry[name], int):
                raise RuntimeError(f"cannot resume: persisted manifest {manifest_path} has invalid {name}")


def _validate_resume_agreement(state, manifest, state_file, manifest_path):
    _validate_state(state, state_file)
    _validate_manifest(manifest, manifest_path)
    generations = [
        generation for entry in manifest
        if (generation := _current_driver_generation(entry["tag"])) is not None
    ]
    state_generation = state["generation"]
    if state_generation == 0 and generations:
        raise RuntimeError(
            f"cannot resume: persisted state {state_file} is behind manifest {manifest_path}"
        )
    if state_generation > 0 and (not generations or max(generations) != state_generation):
        if generations and max(generations) > state_generation:
            raise RuntimeError(
                f"cannot resume: persisted state {state_file} is behind manifest {manifest_path}"
            )
        raise RuntimeError(
            f"cannot resume: persisted state {state_file} is ahead of manifest {manifest_path}"
        )


def _save_manifest(out_dir, manifest):
    _validate_manifest(manifest, out_dir / "scored_manifest.json")
    _atomic_json_write(out_dir / "scored_manifest.json", manifest)


def request_gpt55_subjects(cfg, existing_subjects, n=30):
    """Plateau escalation / seed-from-start: hand creative-idea generation to GPT-5.5 via
    opencode, so fresh prompt variety doesn't depend on this script's own fixed vocabulary
    or on the Claude session that launched it staying alive."""
    out_dir = cfg.dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gpt55_subjects.json"
    prompt = (
        f"Write {n} short, vivid, concrete visual scene or subject descriptions (5-15 words "
        f"each, no artist-style words, no medium words) suitable for testing where a "
        f"fine-tuned image-generation style survives on unfamiliar subject matter, versus "
        f"where it breaks down into visual noise. Favor liminal, uncanny, quietly unsettling "
        f"everyday scenes over gore or fantasy creatures. Prioritize genuinely different "
        f"categories of scene from each other (spaces, objects, weather, crowds, machines, "
        f"architecture), not variations on the same idea. Do not repeat or closely paraphrase "
        f"any of these already-used subjects: {json.dumps(existing_subjects)}. "
        f"Write ONLY a JSON array of {n} strings to the file {out_path}, nothing else in that "
        f"file. When done, print exactly: === DONE ==="
    )
    try:
        result = subprocess.run(
            ["opencode", "run", "--dir", str(clawmarks_config.ROOT), "--dangerously-skip-permissions",
             "-m", "openai/gpt-5.5", "--", prompt],
            capture_output=True, text=True, timeout=300,
        )
        print(f"[gpt5.5] exit={result.returncode} stdout_tail={result.stdout[-300:]!r}", flush=True)
    except Exception as e:
        print(f"[gpt5.5] FAILED to invoke opencode: {e}", flush=True)
        return []
    if out_path.exists():
        try:
            with open(out_path) as f:
                subjects = json.load(f)
            if isinstance(subjects, list) and subjects:
                print(f"[gpt5.5] got {len(subjects)} subjects", flush=True)
                return [str(s) for s in subjects]
        except Exception as e:
            print(f"[gpt5.5] couldn't parse {out_path}: {e}", flush=True)
    return []


def _load_favorited_images(out_dir):
    """Favorites supersede raw novelty for what the exploit step mutates near, the same role
    yes/no ratings used to play before head-to-head comparisons replaced them (see
    docs/superpowers/specs/2026-07-11-head-to-head-preference-design.md). Unlike the old ratings
    store, user_favorites.json already holds a full item object per tag (tag, prompt_name,
    prompt, strength, cfg, ...), so favorited items can be returned directly without joining
    against scored_manifest.json."""
    favorites_path = out_dir / "user_favorites.json"
    if not favorites_path.exists():
        return []
    with open(favorites_path) as f:
        favorites = json.load(f)
    return list(favorites.values())


def _predicted_preference_pool(manifest, model_path, embed_model, out_dir, top_n=15):
    """Stage 5b (opt-in via --use-predicted-preference): ranks this round's own generated
    images by the trained preference model's score instead of favorite membership. Extends the
    shared embedding cache with any new images first, so an image is never re-embedded across
    generations. Returns [] (callers fall back to Stage 5a's favorites) if no model has been
    trained yet or the manifest is empty."""
    if not manifest or not os.path.exists(model_path):
        return []

    import joblib

    from clawmarks.search import embed_cache
    from clawmarks.search.preference_pairwise_model import score as pairwise_score

    by_tag = {m["tag"]: m for m in manifest}

    def image_path_for(tag):
        return by_tag[tag]["file"]

    tags, embeddings = embed_cache.sync(
        manifest, embed_cache.embeddings_file(out_dir), embed_model, image_path_for
    )
    model = joblib.load(model_path)
    scores = pairwise_score(model, embeddings)
    ranked = sorted(
        ((by_tag[t], s) for t, s in zip(tags, scores) if t in by_tag),
        key=lambda pair: -pair[1],
    )
    return [m for m, _ in ranked[:top_n]]


def _load_sibling_leg_manifests(cfg):
    """Every *other* leg directory within cfg's own expedition, concatenated, as the
    "already explored" exclusion pool -- generalizes round 2's one-off "exclude round 1"
    special case to any number of sibling legs. A brand-new expedition (or a leg with no
    sibling that has produced a manifest yet) returns []."""
    expedition_root = cfg.dir.parent
    manifests = []
    if not expedition_root.exists():
        return manifests
    for sibling_dir in sorted(expedition_root.iterdir()):
        if not sibling_dir.is_dir() or sibling_dir.name == cfg.leg:
            continue
        manifest_path = sibling_dir / "scored_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifests.extend(json.load(f))
    return manifests


def submit_and_collect(cfg, jobs, out_dir, label, timeout_s=600):
    """Submit jobs to the ComfyUI serverless endpoint and poll for results."""
    from clawmarks.compute.comfyui import api_post, api_get, build_workflow, cancel_job
    job_ids = {}
    for j in jobs:
        wf = build_workflow(j["prompt"], j["seed"], j["strength"], j["cfg"], j["steps"], j["sampler"], j["negative"])
        try:
            res = api_post("/run", wf)
            job_ids[res["id"]] = j
        except Exception as e:
            print(f"SUBMIT_FAIL {j['tag']}: {e}", flush=True)

    pending = set(job_ids.keys())
    completed, failed = 0, 0
    manifest = []
    t0 = time.time()
    while pending and time.time() - t0 < timeout_s:
        for jid in list(pending):
            try:
                res = api_get(f"/status/{jid}")
            except Exception:
                continue
            status = res.get("status")
            if status == "COMPLETED":
                j = job_ids[jid]
                images = res.get("output", {}).get("images", [])
                if images:
                    fname = f"{out_dir}/{j['tag']}.png"
                    with open(fname, "wb") as f:
                        f.write(base64.b64decode(images[0]["data"]))
                    manifest.append({**j, "file": fname})
                    completed += 1
                pending.discard(jid)
            elif status in ("FAILED", "CANCELLED"):
                failed += 1
                pending.discard(jid)
        if pending:
            time.sleep(8)
    cancel_failed = 0
    if pending:
        print(f"[{label}] {len(pending)} jobs still pending after {timeout_s}s, cancelling them "
              f"so they stop billing on the provider side", flush=True)
        for jid in pending:
            try:
                cancel_job(jid)
            except Exception as e:
                cancel_failed += 1
                print(f"CANCEL_FAIL {jid}: {e}", flush=True)
    print(f"[{label}] completed={completed} failed={failed} timed_out={len(pending)} "
          f"cancel_failed={cancel_failed}", flush=True)
    return manifest


def score_batch(model, real_embs, real_centroid, manifest_batch, prev_embs=None):
    """Score a batch of new images. With prev_embs (round 2), novelty is measured against
    BOTH the real training set AND round 1's already-explored images; without (round 1), only
    the real set."""
    from clawmarks.search.score_manifest import embed_images
    manifest_batch = [m for m in manifest_batch if os.path.exists(m["file"])]
    if not manifest_batch:
        return []
    paths = [m["file"] for m in manifest_batch]
    embs = embed_images(paths, model=model)
    centroid_sim = (embs @ real_centroid).tolist()
    nn_real = (embs @ real_embs.T).max(dim=1).values
    if prev_embs is not None and prev_embs.shape[0] > 0:
        nn_prev = (embs @ prev_embs.T).max(dim=1).values
        nn_combined = _torch_maximum(nn_real, nn_prev).tolist()
    else:
        nn_combined = nn_real.tolist()
    for m, cs, ns in zip(manifest_batch, centroid_sim, nn_combined):
        m["centroid_sim"] = cs
        m["novelty"] = novelty_from_similarity(ns)
        m["prompt_type"] = "style" if m["prompt_name"].startswith("style_") else "conflict"
    return manifest_batch


def _torch_maximum(a, b):
    """torch.maximum is used by score_batch to combine real-set and prev-round nearest
    neighbors; pulled out so the import isn't required at module-load time."""
    import torch
    return torch.maximum(a, b)


def cell_html(items, faith_edges, novelty_edges, fb, nb):
    if not items:
        return '<div class="cell empty"></div>'
    from clawmarks.build.thumbnails import thumb_data_uri
    thumbs = "".join(
        f'<img style="border:2px solid {TYPE_COLOR[m["prompt_type"]]}" '
        f'src="{thumb_data_uri(m["file"])}" title="{m["tag"]} faith={m["centroid_sim"]:.3f} novelty={m["novelty"]:.3f}">'
        for m in sorted(items, key=lambda m: -m["novelty"])[:12]
    )
    return f'<div class="cell" data-count="{len(items)}"><div class="cell-label">n={len(items)}</div><div class="cell-thumbs">{thumbs}</div></div>'


def build_gallery(cfg, manifest, real_ref):
    # A resumed manifest (see _load_resumable_manifest) can reference a PNG that no longer
    # exists on disk; thumb_data_uri would crash the whole generation loop trying to open it.
    # Filter here, mirroring score_batch's identical guard for the same reason. bin_edges
    # indexes into an empty list if every entry got filtered out, so bail out the same way the
    # caller already does for an empty manifest.
    manifest = [m for m in manifest if os.path.exists(m["file"])]
    if not manifest:
        return 0.0
    faith_vals = sorted(m["centroid_sim"] for m in manifest)
    novelty_vals = sorted(m["novelty"] for m in manifest)

    faith_edges = bin_edges(faith_vals, N_BINS)
    novelty_edges = bin_edges(novelty_vals, N_BINS)

    grid = {}
    for m in manifest:
        grid.setdefault((bin_of(m["centroid_sim"], faith_edges), bin_of(m["novelty"], novelty_edges)), []).append(m)

    real_ref_min, real_ref_max = real_ref[1], real_ref[2]
    liminal_band = [m for m in manifest if real_ref_min <= m["centroid_sim"] <= real_ref_max] \
        or [m for m in manifest if faith_edges[0] <= m["centroid_sim"] <= faith_edges[-1]]
    top = sorted(liminal_band, key=lambda m: -m["novelty"])[:32]
    top.sort(key=lambda m: -m["centroid_sim"])

    rows = "".join(
        '<div class="row">' + "".join(
            cell_html(grid.get((fb, nb), []), faith_edges, novelty_edges, fb, nb) for nb in range(N_BINS)
        ) + '</div>'
        for fb in range(N_BINS)
    )
    from clawmarks.build.thumbnails import thumb_data_uri
    highlight_html = "".join(
        f'<figure><img style="border:2px solid {TYPE_COLOR[m["prompt_type"]]}" src="{thumb_data_uri(m["file"])}">'
        f'<figcaption>{m["prompt_name"]} | faith={m["centroid_sim"]:.3f} novelty={m["novelty"]:.3f}</figcaption></figure>'
        for m in top
    )
    title = f"CLAWMARKS uncanny frontier atlas: {cfg.expedition}/{cfg.leg}"
    intro = cfg.description or f"Leg {cfg.leg} of expedition {cfg.expedition}."
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ background:#111; color:#eee; font-family: -apple-system, sans-serif; margin:0; padding:24px; }}
.grid {{ display:flex; flex-direction:column; gap:6px; margin-top:24px; }}
.row {{ display:flex; gap:6px; }}
.cell {{ flex:1; background:#1c1c1c; border:1px solid #333; min-height:160px; padding:6px; }}
.cell.empty {{ background:#161616; }}
.cell-label {{ font-size:10px; color:#777; margin-bottom:4px; }}
.cell-thumbs img {{ width:56px; height:56px; object-fit:cover; margin:1px; border-radius:3px; }}
.highlight {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:16px; }}
.highlight figure {{ margin:0; width:160px; }}
.highlight img {{ width:160px; height:160px; object-fit:cover; border-radius:6px; }}
.highlight figcaption {{ font-size:11px; color:#aaa; margin-top:4px; }}
</style></head><body>
<h1>{title}</h1>
<p>{intro}</p>
<h2>Liminal band highlights</h2>
<div class="highlight">{highlight_html}</div>
<h2>Full descriptor grid</h2>
<div class="grid">{rows}</div>
</body></html>"""
    out_dir = _out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gallery.html", "w") as f:
        f.write(html)
    return max((m["novelty"] for m in liminal_band), default=0.0)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--expedition", required=True)
    parser.add_argument("--leg", required=True)
    parser.add_argument(
        "--use-predicted-preference", action="store_true", default=False,
        help="Stage 5b (opt-in, requires a trained preference_pairwise_model.joblib in this "
             "leg's directory and human validation via preference_rank.html first): rank the "
             "exploit pool by the trained model's predicted preference instead of favorited "
             "images. Defaults off; do not enable without having browsed preference_rank.html "
             "first.",
    )
    args = parser.parse_args(argv)
    cfg = load_leg_config(args.expedition, args.leg)

    import torch
    from clawmarks.search.score_manifest import (
        MODEL_ID, REAL_DIR, embed_images,
    )
    from transformers import AutoModel

    out_dir = cfg.dir
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(cfg)
    manifest = _load_resumable_manifest(out_dir)
    _validate_resume_agreement(state, manifest, _state_file(cfg), out_dir / "scored_manifest.json")
    if state["start_balance"] is None:
        state["start_balance"] = get_balance()
        save_state(cfg, state)
    start_time = state["start_time"]

    print("loading DINOv2 (once, kept warm across all generations)...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    real_paths = sorted(os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    real_embs = embed_images(real_paths, model=model)
    real_centroid = real_embs.mean(dim=0)
    real_centroid = real_centroid / real_centroid.norm()
    loo_sims = []
    for i in range(real_embs.shape[0]):
        others = torch.cat([real_embs[:i], real_embs[i + 1:]], dim=0)
        loo_c = others.mean(dim=0)
        loo_c = loo_c / loo_c.norm()
        loo_sims.append((real_embs[i] @ loo_c).item())
    real_ref = (sum(loo_sims) / len(loo_sims), min(loo_sims), max(loo_sims))
    print(f"real-image reference band: {real_ref}", flush=True)

    print("embedding every sibling leg's already-explored images as the exclusion set...", flush=True)
    prev_manifest = _load_sibling_leg_manifests(cfg)
    prev_paths = [m["file"] for m in prev_manifest if os.path.exists(m["file"])]
    prev_embs = embed_images(prev_paths, model=model) if prev_paths else None
    print(f"embedded {len(prev_paths)} sibling-leg images as the exclusion set" if prev_paths
          else "no sibling-leg manifests found yet; running without exclusion embeddings", flush=True)

    # Round 2: seed the subject pool from the shared candidate_seeds.json pool AND from
    # GPT-5.5 at startup (no plateau wait). Round 1: just start with the base vocabulary
    # and let the plateau-detection stage logic widen it later.
    if cfg.seed_from_start:
        shared_pool_dict = seed_pool_load(out_dir / "seed_pool.json")
        shared_pool = list(shared_pool_dict.keys())
        print(f"loaded {len(shared_pool)} subjects from this leg's seed pool ({out_dir / 'seed_pool.json'})", flush=True)
        if not state["gpt55_subjects"]:
            print("seeding subject pool with GPT-5.5 from generation 1 (no plateau wait)...", flush=True)
            gpt_subjects = request_gpt55_subjects(cfg, cfg.fallback_subjects + shared_pool, n=30)
            state["gpt55_subjects"] = gpt_subjects
            if gpt_subjects:
                updated, _added = seed_pool_merge(
                    shared_pool_dict, gpt_subjects,
                    source=f"gpt5.5-{cfg.expedition}-{cfg.leg}",
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                seed_pool_save(out_dir / "seed_pool.json", updated)
            save_state(cfg, state)
        subjects = cfg.fallback_subjects + state["gpt55_subjects"] + shared_pool
        if not state["gpt55_subjects"]:
            print("GPT-5.5 handoff produced nothing usable; continuing with the fallback subject list only", flush=True)
    else:
        subjects = list(cfg.fallback_subjects)
    textures = list(cfg.textures)
    while True:
        elapsed_h = (time.time() - start_time) / 3600
        if elapsed_h > cfg.wall_clock_cap_hours:
            print(f"STOPPING: wall-clock cap reached ({elapsed_h:.2f}h > {cfg.wall_clock_cap_hours}h)", flush=True)
            break
        if state["generation"] >= cfg.max_generations:
            print(f"STOPPING: hit MAX_GENERATIONS sanity ceiling ({cfg.max_generations})", flush=True)
            break
        spent = _spent_or_none(state["start_balance"])
        if spent is None:
            break
        if abs(spent) > (cfg.budget_usd_cap - cfg.budget_safety_margin):
            print(f"STOPPING: projected spend ${abs(spent):.2f} crossed the "
                  f"${cfg.budget_usd_cap - cfg.budget_safety_margin:.2f} safety threshold "
                  f"(cap ${cfg.budget_usd_cap}, margin ${cfg.budget_safety_margin})", flush=True)
            break

        state["generation"] += 1
        gen = state["generation"]
        liminal_band_all = [m for m in manifest if real_ref[1] <= m["centroid_sim"] <= real_ref[2]]
        elites = sorted(liminal_band_all, key=lambda m: -m["novelty"])[:15]
        if not elites and not cfg.seed_from_start:
            elites = manifest[-30:] if manifest else []
        user_picks = _load_favorited_images(out_dir) if cfg.seed_from_start else []
        if args.use_predicted_preference:
            predicted_pool = _predicted_preference_pool(
                manifest, out_dir / "preference_pairwise_model.joblib", model, out_dir,
            )
            if predicted_pool:
                user_picks = predicted_pool
            else:
                print("--use-predicted-preference set but no trained model found yet "
                       "(or nothing generated so far this leg); using favorited images "
                       "instead", flush=True)

        print(f"\n=== generation {gen} | elapsed {elapsed_h:.2f}h | spend ${abs(spent):.3f} | "
              f"stage {state['stage']} | plateau_count {state['plateau_count']} ===", flush=True)

        jobs = build_generation_jobs(gen, subjects, textures, elites, user_picks,
                                      cfg.gen_batch_size, cfg.explore_fraction,
                                      style_subject_count=cfg.style_subject_count)
        new_manifest = submit_and_collect(cfg, jobs, out_dir, f"gen{gen}")
        new_scored = score_batch(model, real_embs, real_centroid, new_manifest, prev_embs=prev_embs)
        manifest.extend(new_scored)
        # _save_manifest validates every entry, then writes tmp+os.replace, so a kill mid-write
        # can never leave a truncated scored_manifest.json for the next restart to trip over.
        _save_manifest(out_dir, manifest)

        best_novelty = build_gallery(cfg, manifest, real_ref) if manifest else 0.0
        state["novelty_history"].append(best_novelty)
        print(f"generation {gen}: {len(new_scored)} new images, cumulative {len(manifest)}, "
              f"liminal-band best novelty {best_novelty:.4f}", flush=True)

        hist = state["novelty_history"]
        if len(hist) > PLATEAU_WINDOW and max(hist[-PLATEAU_WINDOW:]) <= max(hist[:-PLATEAU_WINDOW]) + PLATEAU_EPSILON:
            state["plateau_count"] += 1
            if not cfg.seed_from_start:
                # Round 1's two-stage escalation: widen vocabulary, then hand off to GPT-5.5.
                print(f"PLATEAU detected (count={state['plateau_count']}): best novelty hasn't "
                      f"improved by >{PLATEAU_EPSILON} over the last {PLATEAU_WINDOW} generations", flush=True)
                if state["stage"] == 0:
                    state["stage"] = 1
                    subjects = list(cfg.fallback_subjects) + list(cfg.widened_subjects)
                    textures = list(cfg.textures) + list(cfg.widened_textures)
                    print("SELF-IMPROVE stage 1: widened subject/texture vocabulary and "
                          "strength/CFG ranges for future generations", flush=True)
                elif state["stage"] == 1:
                    state["stage"] = 2
                    print("SELF-IMPROVE stage 2: local vocabulary widening didn't help either; "
                          "handing creative-subject generation off to GPT-5.5 via opencode so "
                          "fresh variety doesn't depend on this script's fixed lists", flush=True)
                    new_subjects = request_gpt55_subjects(cfg, subjects)
                    if new_subjects:
                        subjects = subjects + new_subjects
                        state["gpt55_subjects"] = state["gpt55_subjects"] + new_subjects
                    else:
                        print("gpt5.5 handoff produced nothing usable; continuing with the "
                              "widened deterministic vocabulary instead", flush=True)
            else:
                # Round 2's plateau handling: ask GPT-5.5 for more subjects on every 3rd
                # plateau event, add them to both the run's subject pool and the shared seed pool.
                print(f"PLATEAU detected (count={state['plateau_count']})", flush=True)
                if state["plateau_count"] % 3 == 1:
                    more = request_gpt55_subjects(cfg, subjects, n=20)
                    if more:
                        subjects = subjects + more
                        state["gpt55_subjects"] = state["gpt55_subjects"] + more
                        shared_dict = seed_pool_load(out_dir / "seed_pool.json")
                        updated, _added = seed_pool_merge(
                            shared_dict, more, source=f"gpt5.5-{cfg.expedition}-{cfg.leg}",
                            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                        seed_pool_save(out_dir / "seed_pool.json", updated)
        save_state(cfg, state)

    print(f"\nLEG {cfg.expedition}/{cfg.leg} RUN ENDED at generation {state['generation']}, "
          f"{len(manifest)} total images, gallery at {out_dir / 'gallery.html'}", flush=True)


if __name__ == "__main__":
    main()
