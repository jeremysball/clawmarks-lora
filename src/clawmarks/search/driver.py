"""
Merged all-night driver for the CLAWMARKS liminal-band / uncanny-frontier search
(lab_notebook.md Section 3b). Combines notes/run_uncanny_allnight.py (round 1) and
notes/run_uncanny_allnight2.py (round 2) into a single --round-parameterized entry point so
the two scripts' 90%-duplicated bodies stop drifting apart.

Round 1 (notes/run_uncanny_allnight.py): staged-escalation plateau handling (widen vocabulary
first, then hand off to GPT-5.5), 50/50 explore/exploit, no parent_tag, no shared seed pool.

Round 2 (notes/run_uncanny_allnight2.py): explore-heavy 85/15 split, GPT-5.5 seeds from
generation 1, user-picks-first exploit pool with parent_tag, prior-round exclusion
embeddings, shared seed pool read/write.

Per-round config lives in ROUND_CONFIGS. Per the plan, round 2's build_generation_jobs is a
strict generalization of round 1's: at explore_fraction=0.5 and with empty user_picks, it
reproduces round 1's 50/50 split exactly.

Run with: uv run clawmarks run allnight --round 1
          uv run clawmarks run allnight --round 2
"""
import argparse
import base64
import json
import os
import random
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field

from clawmarks.config import SEEDS_FILE, SWEEP2_DIR, SWEEP_DIR
from clawmarks.search.scoring import bin_edges, bin_of, novelty_from_similarity
from clawmarks.search.seed_pool import merge as seed_pool_merge, load as seed_pool_load, save as seed_pool_save
from clawmarks.search.manifest_index import index_by_tag

TRIGGER = "trentbuckle style, "
NEG_DEFAULT = "low quality, blurry, watermark"
TYPE_COLOR = {"style": "#5ec98a", "conflict": "#e0a25e"}
N_BINS = 4
PLATEAU_WINDOW = 3
PLATEAU_EPSILON = 0.01


@dataclass
class RoundConfig:
    round: int
    wall_clock_cap_hours: float
    budget_usd_cap: float
    budget_safety_margin: float
    gen_batch_size: int
    explore_fraction: float
    max_generations: int
    textures: list
    fallback_subjects: list
    seed_from_start: bool
    exclude_prev_round: bool
    out_dir_name: str
    # round 1's two-stage escalation vocabularies (the merged driver only uses them when
    # cfg.seed_from_start is False, mirroring round 1's stage-0/1 plateau handling)
    widened_textures: list = field(default_factory=list)
    widened_subjects: list = field(default_factory=list)


ROUND_CONFIGS = {
    1: RoundConfig(
        round=1, wall_clock_cap_hours=7.5, budget_usd_cap=10.0, budget_safety_margin=1.5,
        gen_batch_size=60, explore_fraction=0.5, max_generations=400,
        textures=[
            "marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
            "dark-rimmed eyes glowing pale blue, dense dark-blue vertical brush-dash background, "
            "thick acrylic dry-brush texture, raw outsider-art painting",
        ],
        widened_textures=[
            "loose watercolor wash bleeding at the edges, raw sketchbook page, mixed media",
            "heavy black ink crosshatching over torn found-paper collage edges, raw "
            "outsider-art painting",
        ],
        fallback_subjects=[
            "close-up cat portrait", "close-up wolf portrait", "close-up fox portrait",
            "close-up owl portrait", "close-up horse portrait",
            "close-up human face, pale skin, hand pressed beside cheek",
            "close-up cyborg face, half exposed circuitry and wiring, clawed metal hand pressed beside cheek",
            "close-up face mid-transformation, skin splitting to reveal clawed fingers pushing through the cheek",
            "figure standing alone in an empty fluorescent-lit hallway, clawed hand pressed against the wall",
            "dental x-ray radiograph of a jaw",
            "empty concrete stairwell viewed from below",
            "television weather map with swirling storm system",
            "crowd of human faces packed close together",
        ],
        widened_subjects=[
            "dollhouse interior seen through a broken window",
            "empty parking garage at night, one flickering light",
            "wall of surveillance camera monitors, mostly static",
            "vending machine humming alone in a dark hallway",
            "mannequin display missing its head",
            "storm drain grate half-submerged in still water",
            "airport terminal at night, all gates empty",
            "abandoned playground, swing set mid-motion with no one on it",
            "elevator interior, doors closing on an empty hallway",
            "waiting room with rows of identical empty chairs",
        ],
        seed_from_start=False, exclude_prev_round=False, out_dir_name="uncanny_sweep",
    ),
    2: RoundConfig(
        round=2, wall_clock_cap_hours=1.0, budget_usd_cap=1.00, budget_safety_margin=0.10,
        gen_batch_size=20, explore_fraction=0.85, max_generations=60,
        textures=[
            "marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
            "dark-rimmed eyes glowing pale blue, dense dark-blue vertical brush-dash background, "
            "thick acrylic dry-brush texture, raw outsider-art painting",
            "loose watercolor wash bleeding at the edges, raw sketchbook page, mixed media",
            "heavy black ink crosshatching over torn found-paper collage edges, raw "
            "outsider-art painting",
        ],
        fallback_subjects=[
            "close-up cat portrait", "close-up wolf portrait", "close-up fox portrait",
            "close-up human face, pale skin, hand pressed beside cheek",
            "dollhouse interior seen through a broken window",
            "empty parking garage at night, one flickering light",
            "wall of surveillance camera monitors, mostly static",
            "vending machine humming alone in a dark hallway",
            "abandoned playground, swing set mid-motion with no one on it",
            "waiting room with rows of identical empty chairs",
        ],
        seed_from_start=True, exclude_prev_round=True, out_dir_name="uncanny_sweep2",
    ),
}


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


def _out_dir(cfg):
    return SWEEP_DIR if cfg.out_dir_name == "uncanny_sweep" else SWEEP2_DIR


def _state_file(cfg):
    return _out_dir(cfg) / f"allnight{cfg.round}_state.json"


def load_state(cfg):
    state_file = _state_file(cfg)
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {
        "generation": 0, "stage": 0, "plateau_count": 0,
        "novelty_history": [], "gpt55_subjects": [], "start_balance": None,
        "start_time": time.time(),
    }


def save_state(cfg, state):
    with open(_state_file(cfg), "w") as f:
        json.dump(state, f, indent=1)


def request_gpt55_subjects(cfg, existing_subjects, n=30):
    """Plateau escalation / seed-from-start: hand creative-idea generation to GPT-5.5 via
    opencode, so fresh prompt variety doesn't depend on this script's own fixed vocabulary
    or on the Claude session that launched it staying alive."""
    out_dir = _out_dir(cfg)
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
            ["opencode", "run", "--dir", str(SWEEP_DIR.parent), "--dangerously-skip-permissions",
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


def _load_yes_rated_images():
    """Ratings supersede picks: a human's yes/no judgment on an image, not raw novelty, decides
    what the exploit step mutates near. user_ratings.json stores only {label, rated_at} per tag
    (the image metadata already lives in scored_manifest.json), so yes-rated tags are joined
    against that manifest to recover prompt/strength/cfg for mutation."""
    ratings_path = SWEEP_DIR / "user_ratings.json"
    manifest_path = SWEEP_DIR / "scored_manifest.json"
    if not ratings_path.exists() or not manifest_path.exists():
        return []
    with open(ratings_path) as f:
        ratings = json.load(f)
    yes_tags = {tag for tag, r in ratings.items() if r.get("label") == "yes"}
    if not yes_tags:
        return []
    with open(manifest_path) as f:
        manifest = json.load(f)
    by_tag = index_by_tag(manifest)
    return [by_tag[t] for t in yes_tags if t in by_tag]


def _predicted_preference_pool(manifest, model_path, embed_model, top_n=15):
    """Stage 5b (opt-in via --use-predicted-preference): ranks this round's own generated
    images by the trained preference model's P(yes) instead of yes-rating membership. Extends
    the shared embedding cache with any new images first, so an image is never re-embedded
    across generations. Returns [] (callers fall back to Stage 5a's yes-rated pool) if no model
    has been trained yet or the manifest is empty."""
    if not manifest or not os.path.exists(model_path):
        return []

    import joblib

    from clawmarks.search import embed_cache
    from clawmarks.search.preference_model import predict_proba

    by_tag = {m["tag"]: m for m in manifest}

    def image_path_for(tag):
        return by_tag[tag]["file"]

    tags, embeddings = embed_cache.sync(manifest, embed_cache.EMBEDDINGS_FILE, embed_model, image_path_for)
    model = joblib.load(model_path)
    scores = predict_proba(model, embeddings)
    ranked = sorted(
        ((by_tag[t], s) for t, s in zip(tags, scores) if t in by_tag),
        key=lambda pair: -pair[1],
    )
    return [m for m, _ in ranked[:top_n]]


def _load_prev_round_state(cfg):
    """Round 1's body still reads its own state file (allnight_state.json) and the
    first-run fixed-sweep manifest. Round 2 reads round 1's manifest as the exclusion set.
    Returns (manifest, prev_embs_or_None) tuple."""
    if cfg.exclude_prev_round:
        prev_manifest_path = SWEEP_DIR / "scored_manifest.json"
        if prev_manifest_path.exists():
            with open(prev_manifest_path) as f:
                return json.load(f), None  # actual embeddings computed inside main()
    return [], None


def submit_and_collect(cfg, jobs, out_dir, label, timeout_s=600):
    """Submit jobs to the ComfyUI serverless endpoint and poll for results."""
    from clawmarks.compute.comfyui import api_post, api_get, build_workflow
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
    if pending:
        print(f"[{label}] {len(pending)} jobs still pending after {timeout_s}s, giving up on them", flush=True)
    print(f"[{label}] completed={completed} failed={failed} timed_out={len(pending)}", flush=True)
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
        nn_combined = torch_maximum(nn_real, nn_prev).tolist()
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
    from clawmarks.build.uncanny_gallery import thumb_data_uri
    thumbs = "".join(
        f'<img style="border:2px solid {TYPE_COLOR[m["prompt_type"]]}" '
        f'src="{thumb_data_uri(m["file"])}" title="{m["tag"]} faith={m["centroid_sim"]:.3f} novelty={m["novelty"]:.3f}">'
        for m in sorted(items, key=lambda m: -m["novelty"])[:12]
    )
    return f'<div class="cell" data-count="{len(items)}"><div class="cell-label">n={len(items)}</div><div class="cell-thumbs">{thumbs}</div></div>'


def build_gallery(cfg, manifest, real_ref):
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
        f'<div class="row">' + "".join(
            cell_html(grid.get((fb, nb), []), faith_edges, novelty_edges, fb, nb) for nb in range(N_BINS)
        ) + '</div>'
        for fb in range(N_BINS)
    )
    from clawmarks.build.uncanny_gallery import thumb_data_uri
    highlight_html = "".join(
        f'<figure><img style="border:2px solid {TYPE_COLOR[m["prompt_type"]]}" src="{thumb_data_uri(m["file"])}">'
        f'<figcaption>{m["prompt_name"]} | faith={m["centroid_sim"]:.3f} novelty={m["novelty"]:.3f}</figcaption></figure>'
        for m in top
    )
    title = f"CLAWMARKS uncanny frontier atlas (round {cfg.round})"
    intro = {
        1: "Round 1: staged-escalation plateau handling (widen vocabulary first, then GPT-5.5), "
           "50/50 explore/exploit. Novelty scored against the real training set only.",
        2: "Round 2: explore-heavy rerun (85% fresh recombination, 15% mutation), GPT-5.5-seeded "
           "from generation 1, novelty scored against both the real training set and round 1's "
           "3392 images, so retreading round 1's already-explored region no longer counts as novel.",
    }[cfg.round]
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
    parser.add_argument("--round", type=int, choices=list(ROUND_CONFIGS.keys()), required=True)
    parser.add_argument(
        "--use-predicted-preference", action="store_true", default=False,
        help="Stage 5b (opt-in, requires notes/uncanny_sweep/preference_model.joblib and "
             "human validation via preference_rank.html first): rank the exploit pool by the "
             "trained model's predicted preference instead of yes-rated images. Defaults off; "
             "do not enable without having browsed preference_rank.html first.",
    )
    args = parser.parse_args(argv)
    cfg = ROUND_CONFIGS[args.round]

    import torch
    from clawmarks.search.score_manifest import (
        MODEL_ID, REAL_DIR, embed_images,
    )
    from transformers import AutoModel

    out_dir = _out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(cfg)
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
        loo_c = others.mean(dim=0); loo_c = loo_c / loo_c.norm()
        loo_sims.append((real_embs[i] @ loo_c).item())
    real_ref = (sum(loo_sims) / len(loo_sims), min(loo_sims), max(loo_sims))
    print(f"real-image reference band: {real_ref}", flush=True)

    # Round 2's prior-round exclusion embeddings: embed round 1's already-explored images so
    # novelty for a new image can be measured against the union of (real set, round 1 set).
    # Round 1 has no prior round, so it skips this entirely.
    prev_embs = None
    if cfg.exclude_prev_round:
        print("embedding the prior round's already-explored images as the exclusion set...", flush=True)
        prev_manifest, _ = _load_prev_round_state(cfg)
        prev_paths = [m["file"] for m in prev_manifest if os.path.exists(m["file"])]
        if prev_paths:
            prev_embs = embed_images(prev_paths, model=model)
            print(f"embedded {len(prev_paths)} prior-round images as the exclusion set", flush=True)
        else:
            print("no prior-round manifest found; running without exclusion embeddings", flush=True)
            prev_embs = None

    # Round 2: seed the subject pool from the shared candidate_seeds.json pool AND from
    # GPT-5.5 at startup (no plateau wait). Round 1: just start with the base vocabulary
    # and let the plateau-detection stage logic widen it later.
    if cfg.seed_from_start:
        shared_pool_dict = seed_pool_load(SEEDS_FILE)
        shared_pool = list(shared_pool_dict.keys())
        print(f"loaded {len(shared_pool)} subjects from the shared candidate seed pool ({SEEDS_FILE})", flush=True)
        if not state["gpt55_subjects"]:
            print("seeding subject pool with GPT-5.5 from generation 1 (no plateau wait)...", flush=True)
            gpt_subjects = request_gpt55_subjects(cfg, cfg.fallback_subjects + shared_pool, n=30)
            state["gpt55_subjects"] = gpt_subjects
            if gpt_subjects:
                updated, _added = seed_pool_merge(
                    shared_pool_dict, gpt_subjects,
                    source=f"gpt5.5-round{cfg.round}", created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                seed_pool_save(SEEDS_FILE, updated)
            save_state(cfg, state)
        subjects = cfg.fallback_subjects + state["gpt55_subjects"] + shared_pool
        if not state["gpt55_subjects"]:
            print("GPT-5.5 handoff produced nothing usable; continuing with the fallback subject list only", flush=True)
    else:
        subjects = list(cfg.fallback_subjects)
    textures = list(cfg.textures)
    # round 1's main loop indexes the first 5 base subjects to decide style-vs-conflict
    # tagging; round 2 uses the first 4. The tests exercise the round 2 path with
    # style_subject_count=4 by default. Round 1's main loop calls the function with
    # style_subject_count=5 below.
    style_subject_count = 5 if cfg.round == 1 else 4

    manifest = []

    while True:
        elapsed_h = (time.time() - start_time) / 3600
        if elapsed_h > cfg.wall_clock_cap_hours:
            print(f"STOPPING: wall-clock cap reached ({elapsed_h:.2f}h > {cfg.wall_clock_cap_hours}h)", flush=True)
            break
        if state["generation"] >= cfg.max_generations:
            print(f"STOPPING: hit MAX_GENERATIONS sanity ceiling ({cfg.max_generations})", flush=True)
            break
        try:
            balance_now = get_balance()
            spent = state["start_balance"] - balance_now
        except Exception as e:
            print(f"balance check failed ({e})", flush=True)
            spent = 0
        if abs(spent) > (cfg.budget_usd_cap - cfg.budget_safety_margin):
            print(f"STOPPING: projected spend ${abs(spent):.2f} crossed the "
                  f"${cfg.budget_usd_cap - cfg.budget_safety_margin:.2f} safety threshold "
                  f"(cap ${cfg.budget_usd_cap}, margin ${cfg.budget_safety_margin})", flush=True)
            break

        state["generation"] += 1
        gen = state["generation"]
        liminal_band_all = [m for m in manifest if real_ref[1] <= m["centroid_sim"] <= real_ref[2]]
        elites = sorted(liminal_band_all, key=lambda m: -m["novelty"])[:15]
        if not elites and cfg.round == 1:
            elites = manifest[-30:] if manifest else []
        user_picks = _load_yes_rated_images() if cfg.seed_from_start else []
        if args.use_predicted_preference:
            predicted_pool = _predicted_preference_pool(
                manifest, SWEEP_DIR / "preference_model.joblib", model,
            )
            if predicted_pool:
                user_picks = predicted_pool
            else:
                print("--use-predicted-preference set but no trained model found yet "
                      "(or nothing generated so far this round); using yes-rated images "
                      "instead", flush=True)

        print(f"\n=== generation {gen} | elapsed {elapsed_h:.2f}h | spend ${abs(spent):.3f} | "
              f"stage {state['stage']} | plateau_count {state['plateau_count']} ===", flush=True)

        jobs = build_generation_jobs(gen, subjects, textures, elites, user_picks,
                                      cfg.gen_batch_size, cfg.explore_fraction,
                                      style_subject_count=style_subject_count)
        new_manifest = submit_and_collect(cfg, jobs, out_dir, f"gen{gen}")
        new_scored = score_batch(model, real_embs, real_centroid, new_manifest, prev_embs=prev_embs)
        manifest.extend(new_scored)
        with open(out_dir / "scored_manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)

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
                        shared_dict = seed_pool_load(SEEDS_FILE)
                        updated, _added = seed_pool_merge(
                            shared_dict, more, source=f"gpt5.5-round{cfg.round}",
                            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                        seed_pool_save(SEEDS_FILE, updated)
        save_state(cfg, state)

    print(f"\nROUND {cfg.round} RUN ENDED at generation {state['generation']}, "
          f"{len(manifest)} total images, gallery at {out_dir / 'gallery.html'}", flush=True)


if __name__ == "__main__":
    main()
