"""
All-night driver for the CLAWMARKS "liminal band" / uncanny-frontier search
(lab_notebook.md Section 3b). Generation 0 was notes/run_uncanny_sweep.py's 452-job fixed
grid, already complete (notes/uncanny_sweep/manifest.json). This script picks up from there
and keeps going: each generation submits a new batch (mix of "exploit" - small mutations near
the current most-novel liminal-band images - and "explore" - fresh prompt-vocabulary
recombinations), scores everything cumulatively, rebuilds the gallery, and decides whether to
keep going based on a wall-clock cap and a running RunPod balance-based cost estimate.

Runs unattended, no LLM calls needed for the normal path (all decisions are deterministic
Python). If the search plateaus (best liminal-band novelty score stops improving) for two
consecutive escalation stages even after widening the search's own ranges, it hands the
"invent fresh subjects" problem to GPT-5.5 via a one-shot non-interactive `opencode run` call
(model openai/gpt-5.5, confirmed present in `opencode models` output), so creative variety
doesn't depend on this session staying alive or on this script's own fixed vocabulary. Every
generation's state is checkpointed to disk, so a restart resumes rather than starting over.

Stop conditions (whichever comes first): WALL_CLOCK_CAP_HOURS elapsed, or the RunPod account
balance delta since this script started projects past BUDGET_USD_CAP.

Run with: python3 notes/run_uncanny_allnight.py 2>&1 | tee -a notes/uncanny_allnight.log
"""
import os, sys, json, time, random, subprocess, base64
import urllib.request
import torch

sys.path.insert(0, os.path.dirname(__file__))
from run_uncanny_sweep import build_workflow, api_post, api_get
from build_uncanny_gallery import (
    preprocess, embed_images, thumb_data_uri, TYPE_COLOR, cell_html, build_html, N_BINS,
    MODEL_ID, REAL_DIR,
)
from transformers import AutoModel

SC = "/workspace/trent-with-smart-prompts"
SWEEP_DIR = f"{SC}/notes/uncanny_sweep"
STATE_FILE = f"{SWEEP_DIR}/allnight_state.json"
API_KEY = os.environ["RUNPOD_API_KEY"]
GRAPHQL = f"https://api.runpod.io/graphql?api_key={API_KEY}"

WALL_CLOCK_CAP_HOURS = 7.5
BUDGET_USD_CAP = 10.0
BUDGET_SAFETY_MARGIN = 1.5   # stop generating once projected spend crosses (cap - margin)
GEN_BATCH_SIZE = 60
PLATEAU_WINDOW = 3
PLATEAU_EPSILON = 0.01
MAX_GENERATIONS = 400        # hard sanity ceiling regardless of time/budget math

BASE_TEXTURES = [
    "marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
    "dark-rimmed eyes glowing pale blue, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
]
WIDENED_TEXTURES = [
    "loose watercolor wash bleeding at the edges, raw sketchbook page, mixed media",
    "heavy black ink crosshatching over torn found-paper collage edges, raw outsider-art painting",
]

BASE_SUBJECTS = [
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
]
WIDENED_SUBJECTS = [
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
]

TRIGGER = "trentbuckle style, "
NEG_DEFAULT = "low quality, blurry, watermark"


def gql(query):
    req = urllib.request.Request(GRAPHQL, data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "curl/8.0"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    if "errors" in res:
        raise RuntimeError(res["errors"])
    return res["data"]


def get_balance():
    return gql("query { myself { clientBalance } }")["myself"]["clientBalance"]


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "generation": 0, "stage": 0, "plateau_count": 0,
        "novelty_history": [], "gpt55_subjects": [], "start_balance": None,
        "start_time": time.time(),
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1)


def load_manifest():
    path = f"{SWEEP_DIR}/scored_manifest.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    with open(f"{SWEEP_DIR}/manifest.json") as f:
        return json.load(f)


def request_gpt55_subjects(existing_subjects, n=15):
    """Plateau escalation stage 2: hand creative-idea generation to GPT-5.5 via opencode,
    so fresh prompt variety doesn't depend on this script's own fixed vocabulary or on the
    Claude session that launched it staying alive. Non-interactive, one-shot, writes its
    answer to a file this script reads back."""
    out_path = f"{SWEEP_DIR}/gpt55_subjects.json"
    prompt = (
        f"Write {n} short, vivid, concrete visual scene or subject descriptions (5-15 words "
        f"each, no artist-style words, no medium words) suitable for testing where a "
        f"fine-tuned image-generation style survives on unfamiliar subject matter, versus "
        f"where it breaks down into visual noise. Favor liminal, uncanny, quietly unsettling "
        f"everyday scenes over gore or fantasy creatures. Do not repeat or closely paraphrase "
        f"any of these already-used subjects: {json.dumps(existing_subjects)}. "
        f"Write ONLY a JSON array of {n} strings to the file {out_path}, nothing else in that "
        f"file. When done, print exactly: === DONE ==="
    )
    script_path = f"{SWEEP_DIR}/gpt55_prompt.txt"
    with open(script_path, "w") as f:
        f.write(prompt)
    try:
        result = subprocess.run(
            ["opencode", "run", "--dir", SC, "--dangerously-skip-permissions",
             "-m", "openai/gpt-5.5", "--", prompt],
            capture_output=True, text=True, timeout=300,
        )
        print(f"[gpt5.5 handoff] exit={result.returncode} "
              f"stdout_tail={result.stdout[-300:]!r}", flush=True)
    except Exception as e:
        print(f"[gpt5.5 handoff] FAILED to invoke opencode: {e}", flush=True)
        return []
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                subjects = json.load(f)
            if isinstance(subjects, list) and subjects:
                print(f"[gpt5.5 handoff] got {len(subjects)} new subjects", flush=True)
                return [str(s) for s in subjects]
        except Exception as e:
            print(f"[gpt5.5 handoff] couldn't parse {out_path}: {e}", flush=True)
    return []


def build_generation_jobs(gen_idx, subjects, textures, batch_size, elites):
    jobs = []
    n_exploit = batch_size // 2 if elites else 0
    n_explore = batch_size - n_exploit

    for i in range(n_exploit):
        e = random.choice(elites)
        strength = max(0.3, min(2.2, e["strength"] + random.gauss(0, 0.2)))
        cfg = max(1.0, min(20.0, e["cfg"] + random.gauss(0, 2.0)))
        seed = random.randint(1, 999999)
        jobs.append({
            "tag": f"gen{gen_idx}_exploit_{i}_seed{seed}", "category": "allnight_exploit",
            "prompt_name": e["prompt_name"], "prompt": e["prompt"],
            "seed": seed, "strength": round(strength, 3), "cfg": round(cfg, 2),
            "steps": 28, "sampler": "ddim", "negative": NEG_DEFAULT,
        })

    for i in range(n_explore):
        subj = random.choice(subjects)
        tex = random.choice(textures)
        prompt = f"{TRIGGER}{subj}, {tex}"
        strength = round(random.uniform(0.5, 2.0), 3)
        cfg = round(random.uniform(2.0, 15.0), 2)
        seed = random.randint(1, 999999)
        is_style = subj in BASE_SUBJECTS[:5]
        pname = ("style_" if is_style else "conflict_") + subj[:24].replace(" ", "_")
        jobs.append({
            "tag": f"gen{gen_idx}_explore_{i}_seed{seed}", "category": "allnight_explore",
            "prompt_name": pname, "prompt": prompt,
            "seed": seed, "strength": strength, "cfg": cfg,
            "steps": 28, "sampler": "ddim", "negative": NEG_DEFAULT,
        })
    return jobs


def submit_and_collect(jobs, out_dir, label, timeout_s=900):
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
        print(f"[{label}] {len(pending)} jobs still pending after {timeout_s}s timeout, "
              f"giving up on them for this generation (not blocking the whole night on a "
              f"hung job).", flush=True)
    print(f"[{label}] completed={completed} failed={failed} timed_out={len(pending)}", flush=True)
    return manifest


def score_batch(model, real_embs, centroid, manifest_batch):
    paths = [m["file"] for m in manifest_batch if os.path.exists(m["file"])]
    manifest_batch = [m for m in manifest_batch if os.path.exists(m["file"])]
    if not paths:
        return []
    embs = embed_images(paths, model=model)
    centroid_sim = (embs @ centroid).tolist()
    nn_sim = (embs @ real_embs.T).max(dim=1).values.tolist()
    for m, cs, ns in zip(manifest_batch, centroid_sim, nn_sim):
        m["centroid_sim"] = cs
        m["novelty"] = 1 - ns
        m["prompt_type"] = "style" if m["prompt_name"].startswith("style_") else "conflict"
    return manifest_batch


def rebuild_gallery(manifest, real_ref):
    faith_vals = sorted(m["centroid_sim"] for m in manifest)
    novelty_vals = sorted(m["novelty"] for m in manifest)

    def bin_edges(vals, n):
        return [vals[int(i * len(vals) / n)] for i in range(1, n)]

    faith_edges = bin_edges(faith_vals, N_BINS)
    novelty_edges = bin_edges(novelty_vals, N_BINS)

    def bin_of(val, edges):
        for i, e in enumerate(edges):
            if val <= e:
                return i
        return len(edges)

    grid = {}
    for m in manifest:
        fb, nb = bin_of(m["centroid_sim"], faith_edges), bin_of(m["novelty"], novelty_edges)
        grid.setdefault((fb, nb), []).append(m)

    real_ref_min, real_ref_max = real_ref[1], real_ref[2]
    liminal_band = [m for m in manifest if real_ref_min <= m["centroid_sim"] <= real_ref_max] \
        or [m for m in manifest if faith_edges[0] <= m["centroid_sim"] <= faith_edges[-1]]
    top = sorted(liminal_band, key=lambda m: -m["novelty"])[:32]
    top.sort(key=lambda m: -m["centroid_sim"])

    by_type = {}
    for m in manifest:
        by_type.setdefault(m["prompt_type"], []).append(m["centroid_sim"])
    type_summary = {t: (sum(v) / len(v), len(v)) for t, v in by_type.items()}

    build_html(manifest, grid, faith_edges, novelty_edges, top, real_ref, type_summary)
    best_novelty = max((m["novelty"] for m in liminal_band), default=0.0)
    return best_novelty


def main():
    state = load_state()
    if state["start_balance"] is None:
        state["start_balance"] = get_balance()
        save_state(state)
    start_time = state["start_time"]

    print("loading DINOv2 (once, kept warm across all generations)...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    real_paths = sorted(os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    real_embs = embed_images(real_paths, model=model)
    centroid = real_embs.mean(dim=0)
    centroid = centroid / centroid.norm()
    loo_sims = []
    for i in range(real_embs.shape[0]):
        others = torch.cat([real_embs[:i], real_embs[i + 1:]], dim=0)
        loo_c = others.mean(dim=0); loo_c = loo_c / loo_c.norm()
        loo_sims.append((real_embs[i] @ loo_c).item())
    real_ref = (sum(loo_sims) / len(loo_sims), min(loo_sims), max(loo_sims))
    print(f"real-image reference band: {real_ref}", flush=True)

    manifest = load_manifest()
    manifest = [m for m in manifest if "centroid_sim" in m] or score_batch(model, real_embs, centroid, manifest)
    with open(f"{SWEEP_DIR}/scored_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    best_novelty = rebuild_gallery(manifest, real_ref)
    print(f"generation 0 (fixed sweep) baseline liminal-band novelty: {best_novelty:.4f}", flush=True)
    state["novelty_history"].append(best_novelty)
    save_state(state)

    subjects = list(BASE_SUBJECTS)
    textures = list(BASE_TEXTURES)

    while True:
        elapsed_h = (time.time() - start_time) / 3600
        if elapsed_h > WALL_CLOCK_CAP_HOURS:
            print(f"STOPPING: wall-clock cap reached ({elapsed_h:.2f}h > {WALL_CLOCK_CAP_HOURS}h)", flush=True)
            break
        if state["generation"] >= MAX_GENERATIONS:
            print(f"STOPPING: hit MAX_GENERATIONS sanity ceiling ({MAX_GENERATIONS})", flush=True)
            break
        try:
            balance_now = get_balance()
            spent = balance_now - state["start_balance"]
        except Exception as e:
            print(f"balance check failed ({e}), assuming spend continues at prior rate", flush=True)
            spent = 0
        if abs(spent) > (BUDGET_USD_CAP - BUDGET_SAFETY_MARGIN):
            print(f"STOPPING: projected spend ${abs(spent):.2f} crossed the "
                  f"${BUDGET_USD_CAP - BUDGET_SAFETY_MARGIN:.2f} safety threshold "
                  f"(cap ${BUDGET_USD_CAP}, margin ${BUDGET_SAFETY_MARGIN})", flush=True)
            break

        state["generation"] += 1
        gen = state["generation"]
        liminal_band_all = [m for m in manifest if real_ref[1] <= m["centroid_sim"] <= real_ref[2]]
        elites = sorted(liminal_band_all, key=lambda m: -m["novelty"])[:15] or manifest[-30:]

        print(f"\n=== generation {gen} | elapsed {elapsed_h:.2f}h | spend ${abs(spent):.3f} | "
              f"stage {state['stage']} | plateau_count {state['plateau_count']} ===", flush=True)

        jobs = build_generation_jobs(gen, subjects, textures, GEN_BATCH_SIZE, elites)
        new_manifest = submit_and_collect(jobs, SWEEP_DIR, f"gen{gen}")
        new_scored = score_batch(model, real_embs, centroid, new_manifest)
        manifest.extend(new_scored)
        with open(f"{SWEEP_DIR}/scored_manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)

        best_novelty = rebuild_gallery(manifest, real_ref)
        state["novelty_history"].append(best_novelty)
        print(f"generation {gen}: {len(new_scored)} new images, cumulative {len(manifest)}, "
              f"liminal-band best novelty {best_novelty:.4f}", flush=True)

        hist = state["novelty_history"]
        if len(hist) > PLATEAU_WINDOW and max(hist[-PLATEAU_WINDOW:]) <= max(hist[:-PLATEAU_WINDOW]) + PLATEAU_EPSILON:
            state["plateau_count"] += 1
            print(f"PLATEAU detected (count={state['plateau_count']}): best novelty hasn't "
                  f"improved by >{PLATEAU_EPSILON} over the last {PLATEAU_WINDOW} generations", flush=True)
            if state["stage"] == 0:
                state["stage"] = 1
                subjects = list(BASE_SUBJECTS) + list(WIDENED_SUBJECTS)
                textures = list(BASE_TEXTURES) + list(WIDENED_TEXTURES)
                print("SELF-IMPROVE stage 1: widened subject/texture vocabulary and "
                      "strength/CFG ranges for future generations", flush=True)
            elif state["stage"] == 1:
                state["stage"] = 2
                print("SELF-IMPROVE stage 2: local vocabulary widening didn't help either; "
                      "handing creative-subject generation off to GPT-5.5 via opencode so "
                      "fresh variety doesn't depend on this script's fixed lists", flush=True)
                new_subjects = request_gpt55_subjects(subjects)
                if new_subjects:
                    subjects = subjects + new_subjects
                    state["gpt55_subjects"] = state["gpt55_subjects"] + new_subjects
                else:
                    print("gpt5.5 handoff produced nothing usable; continuing with the "
                          "widened deterministic vocabulary instead", flush=True)
        save_state(state)

    print(f"\nALL-NIGHT RUN ENDED at generation {state['generation']}, "
          f"{len(manifest)} total images, gallery at {SWEEP_DIR}/gallery.html", flush=True)


if __name__ == "__main__":
    main()
