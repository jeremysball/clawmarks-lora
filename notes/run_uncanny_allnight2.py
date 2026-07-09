"""
Round 2 of the liminal-band/uncanny-frontier search (lab_notebook.md Section 3b), rerun with
settings meant to push past the plateau the first run hit (novelty flat at 0.8396 for its last
23 generations). Three changes from notes/run_uncanny_allnight.py, per user direction after
reviewing the first run's results:

1. Explore-heavy mix: ~85% of every batch is fresh subject/texture recombination, not mutation
   near existing high scorers, since the exploit-heavy 50/50 split is one suspected cause of the
   first run's plateau (Section 3b's open question).
2. GPT-5.5 seeds the subject pool from generation 1, rather than only after two plateaus. The
   first run showed the handoff genuinely works (15 usable ideas, real novelty gain), so there's
   no reason to gate it behind a slow escalation ladder this time.
3. Excludes the already-explored region: novelty is scored against BOTH the real training set
   AND the first run's 3392 images (embedded once at startup), not just the real set. An image
   that merely resembles something the first run already generated no longer counts as novel,
   which directly targets "did we find a new region, or just refine the last one."

Small budget by design (account balance was down to ~$1.16 after the first run): stops at $0.90
cumulative spend, well inside a $1.00 safety ceiling, meant as a quick test of whether these
changes actually move the plateau before spending more.

Run with: python3 notes/run_uncanny_allnight2.py 2>&1 | tee -a notes/uncanny_allnight2.log
"""
import os, sys, json, time, random, subprocess, base64
import urllib.request
import torch

sys.path.insert(0, os.path.dirname(__file__))
from run_uncanny_sweep import build_workflow, api_post, api_get
from build_uncanny_gallery import preprocess, embed_images, thumb_data_uri, MODEL_ID, REAL_DIR
from transformers import AutoModel

SC = "/workspace/trent-with-smart-prompts"
PREV_DIR = f"{SC}/notes/uncanny_sweep"
OUT_DIR = f"{SC}/notes/uncanny_sweep2"
os.makedirs(OUT_DIR, exist_ok=True)
STATE_FILE = f"{OUT_DIR}/allnight2_state.json"
USER_PICKS_FILE = f"{PREV_DIR}/user_picks.json"
SEEDS_FILE = f"{PREV_DIR}/candidate_seeds.json"
API_KEY = os.environ["RUNPOD_API_KEY"]
GRAPHQL = f"https://api.runpod.io/graphql?api_key={API_KEY}"

WALL_CLOCK_CAP_HOURS = 1.0
BUDGET_USD_CAP = 1.00
BUDGET_SAFETY_MARGIN = 0.10   # stop once cumulative spend crosses $0.90
GEN_BATCH_SIZE = 20
EXPLORE_FRACTION = 0.85
PLATEAU_WINDOW = 3
PLATEAU_EPSILON = 0.01
MAX_GENERATIONS = 60

TEXTURES = [
    "marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
    "dark-rimmed eyes glowing pale blue, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
    "loose watercolor wash bleeding at the edges, raw sketchbook page, mixed media",
    "heavy black ink crosshatching over torn found-paper collage edges, raw outsider-art painting",
]

FALLBACK_SUBJECTS = [
    "close-up cat portrait", "close-up wolf portrait", "close-up fox portrait",
    "close-up human face, pale skin, hand pressed beside cheek",
    "dollhouse interior seen through a broken window",
    "empty parking garage at night, one flickering light",
    "wall of surveillance camera monitors, mostly static",
    "vending machine humming alone in a dark hallway",
    "abandoned playground, swing set mid-motion with no one on it",
    "waiting room with rows of identical empty chairs",
]

TRIGGER = "trentbuckle style, "
NEG_DEFAULT = "low quality, blurry, watermark"
TYPE_COLOR = {"style": "#5ec98a", "conflict": "#e0a25e"}
N_BINS = 4


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
        "generation": 0, "plateau_count": 0, "novelty_history": [],
        "gpt55_subjects": [], "start_balance": None, "start_time": time.time(),
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1)


def load_shared_seed_pool():
    """candidate_seeds.json is the run-independent pool browsable/growable from seeds.html
    (notes/curation_server.py), a dict of subject text -> {source, created_at}. Reading it here
    means a seed someone added through that UI reaches the next run instead of sitting unused
    until someone notices the gap."""
    if os.path.exists(SEEDS_FILE):
        with open(SEEDS_FILE) as f:
            return list(json.load(f).keys())
    return []


def add_to_shared_seed_pool(subjects, source):
    if not subjects:
        return
    seeds = {}
    if os.path.exists(SEEDS_FILE):
        with open(SEEDS_FILE) as f:
            seeds = json.load(f)
    existing_lower = {s.lower().strip() for s in seeds}
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for s in subjects:
        s = str(s).strip()
        if s and s.lower() not in existing_lower:
            seeds[s] = {"source": source, "created_at": now}
            existing_lower.add(s.lower())
    with open(SEEDS_FILE, "w") as f:
        json.dump(seeds, f, indent=1)


def request_gpt55_subjects(existing_subjects, n=30):
    out_path = f"{OUT_DIR}/gpt55_subjects.json"
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
            ["opencode", "run", "--dir", SC, "--dangerously-skip-permissions",
             "-m", "openai/gpt-5.5", "--", prompt],
            capture_output=True, text=True, timeout=300,
        )
        print(f"[gpt5.5] exit={result.returncode} stdout_tail={result.stdout[-300:]!r}", flush=True)
    except Exception as e:
        print(f"[gpt5.5] FAILED to invoke opencode: {e}", flush=True)
        return []
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                subjects = json.load(f)
            if isinstance(subjects, list) and subjects:
                print(f"[gpt5.5] got {len(subjects)} subjects", flush=True)
                return [str(s) for s in subjects]
        except Exception as e:
            print(f"[gpt5.5] couldn't parse {out_path}: {e}", flush=True)
    return []


def load_user_picks():
    """Human-in-the-loop MAP-Elites: this project has no automated coherence/quality scorer, so
    per lab_notebook.md Section 3b there's no way for an image to automatically "win" a bin. A
    person reviewing notes/uncanny_sweep/scan.html (served by notes/curation_server.py, which is
    what actually persists picks) can mark specific images as winners instead. When present,
    those picks anchor the exploit step's mutations in place of the raw novelty ranking, which is
    only ever a proxy for "interesting," not a verdict on it."""
    if os.path.exists(USER_PICKS_FILE):
        with open(USER_PICKS_FILE) as f:
            picks = json.load(f)
        return list(picks.values())
    return []


def build_generation_jobs(gen_idx, subjects, elites, user_picks, batch_size):
    jobs = []
    n_explore = round(batch_size * EXPLORE_FRACTION)
    n_exploit = batch_size - n_explore

    # User picks (a human judging actual quality) take priority over the automated novelty
    # ranking for choosing what to mutate near, falling back to novelty-ranked elites only for
    # however many exploit slots the picks don't fill.
    exploit_pool = list(user_picks) if user_picks else []
    if len(exploit_pool) < n_exploit:
        exploit_pool = exploit_pool + [e for e in elites if e not in exploit_pool]

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
            # Idea 7 (Fable brainstorm, 2026-07-09): record which image this mutated near, so a
            # lineage tree can show whether exploit steps actually improve on their parent or
            # just wobble. Only present on jobs built after this patch landed; older images (all
            # of round 1, and round 2 images from before a restart) have no parent_tag.
            "parent_tag": e.get("tag"),
        })

    for i in range(n_explore):
        subj = random.choice(subjects)
        tex = random.choice(TEXTURES)
        prompt = f"{TRIGGER}{subj}, {tex}"
        strength = round(random.uniform(0.5, 2.0), 3)
        cfg = round(random.uniform(2.0, 15.0), 2)
        seed = random.randint(1, 999999)
        is_style = subj in FALLBACK_SUBJECTS[:4]
        pname = ("style_" if is_style else "conflict_") + subj[:24].replace(" ", "_")
        jobs.append({
            "tag": f"gen{gen_idx}_explore_{i}_seed{seed}", "category": "r2_explore",
            "prompt_name": pname, "prompt": prompt,
            "seed": seed, "strength": strength, "cfg": cfg,
            "steps": 28, "sampler": "ddim", "negative": NEG_DEFAULT,
        })
    return jobs


def submit_and_collect(jobs, out_dir, label, timeout_s=600):
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


def score_batch(model, real_embs, real_centroid, prev_embs, manifest_batch):
    manifest_batch = [m for m in manifest_batch if os.path.exists(m["file"])]
    if not manifest_batch:
        return []
    paths = [m["file"] for m in manifest_batch]
    embs = embed_images(paths, model=model)
    centroid_sim = (embs @ real_centroid).tolist()
    nn_real = (embs @ real_embs.T).max(dim=1).values
    nn_prev = (embs @ prev_embs.T).max(dim=1).values
    nn_combined = torch.maximum(nn_real, nn_prev).tolist()
    for m, cs, ns in zip(manifest_batch, centroid_sim, nn_combined):
        m["centroid_sim"] = cs
        m["novelty"] = 1 - ns
        m["prompt_type"] = "style" if m["prompt_name"].startswith("style_") else "conflict"
    return manifest_batch


def cell_html(items, faith_edges, novelty_edges, fb, nb):
    if not items:
        return '<div class="cell empty"></div>'
    thumbs = "".join(
        f'<img style="border:2px solid {TYPE_COLOR[m["prompt_type"]]}" '
        f'src="{thumb_data_uri(m["file"])}" title="{m["tag"]} faith={m["centroid_sim"]:.3f} novelty={m["novelty"]:.3f}">'
        for m in sorted(items, key=lambda m: -m["novelty"])[:12]
    )
    return f'<div class="cell" data-count="{len(items)}"><div class="cell-label">n={len(items)}</div><div class="cell-thumbs">{thumbs}</div></div>'


def build_gallery(manifest, real_ref):
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
        grid.setdefault((bin_of(m["centroid_sim"], faith_edges), bin_of(m["novelty"], novelty_edges)), []).append(m)

    real_ref_mean, real_ref_min, real_ref_max = real_ref
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
    highlight_html = "".join(
        f'<figure><img style="border:2px solid {TYPE_COLOR[m["prompt_type"]]}" src="{thumb_data_uri(m["file"])}">'
        f'<figcaption>{m["prompt_name"]} | faith={m["centroid_sim"]:.3f} novelty={m["novelty"]:.3f}</figcaption></figure>'
        for m in top
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>CLAWMARKS uncanny frontier atlas (round 2)</title>
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
<h1>CLAWMARKS uncanny frontier atlas, round 2</h1>
<p>Explore-heavy rerun (85% fresh recombination, 15% mutation), GPT-5.5-seeded from generation 1,
novelty scored against both the real training set and round 1's 3392 images, so retreading round
1's already-explored region no longer counts as novel.</p>
<h2>Liminal band highlights</h2>
<div class="highlight">{highlight_html}</div>
<h2>Full descriptor grid</h2>
<div class="grid">{rows}</div>
</body></html>"""
    with open(f"{OUT_DIR}/gallery.html", "w") as f:
        f.write(html)
    return max((m["novelty"] for m in liminal_band), default=0.0)


def main():
    state = load_state()
    if state["start_balance"] is None:
        state["start_balance"] = get_balance()
        save_state(state)
    start_time = state["start_time"]

    print("loading DINOv2...", flush=True)
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

    print("embedding round 1's 3392 images as the already-explored reference set (one-time cost)...", flush=True)
    with open(f"{PREV_DIR}/scored_manifest.json") as f:
        prev_manifest = json.load(f)
    prev_paths = [m["file"] for m in prev_manifest if os.path.exists(m["file"])]
    prev_embs = embed_images(prev_paths, model=model)
    print(f"embedded {len(prev_paths)} round-1 images as the exclusion set", flush=True)

    shared_pool = load_shared_seed_pool()
    print(f"loaded {len(shared_pool)} subjects from the shared candidate seed pool ({SEEDS_FILE})", flush=True)

    if not state["gpt55_subjects"]:
        print("seeding subject pool with GPT-5.5 from generation 1 (no plateau wait this time)...", flush=True)
        gpt_subjects = request_gpt55_subjects(FALLBACK_SUBJECTS + shared_pool, n=30)
        state["gpt55_subjects"] = gpt_subjects
        add_to_shared_seed_pool(gpt_subjects, source="gpt5.5-round2")
        save_state(state)
    subjects = FALLBACK_SUBJECTS + state["gpt55_subjects"] + shared_pool
    if not state["gpt55_subjects"]:
        print("GPT-5.5 handoff produced nothing usable; continuing with the fallback subject list only", flush=True)

    manifest = []

    while True:
        elapsed_h = (time.time() - start_time) / 3600
        if elapsed_h > WALL_CLOCK_CAP_HOURS:
            print(f"STOPPING: wall-clock cap reached ({elapsed_h:.2f}h)", flush=True)
            break
        if state["generation"] >= MAX_GENERATIONS:
            print(f"STOPPING: hit MAX_GENERATIONS ({MAX_GENERATIONS})", flush=True)
            break
        try:
            balance_now = get_balance()
            spent = state["start_balance"] - balance_now
        except Exception as e:
            print(f"balance check failed ({e})", flush=True)
            spent = 0
        if spent > (BUDGET_USD_CAP - BUDGET_SAFETY_MARGIN):
            print(f"STOPPING: spend ${spent:.3f} crossed the ${BUDGET_USD_CAP - BUDGET_SAFETY_MARGIN:.2f} "
                  f"safety threshold (cap ${BUDGET_USD_CAP})", flush=True)
            break

        state["generation"] += 1
        gen = state["generation"]
        liminal_band_all = [m for m in manifest if real_ref[1] <= m["centroid_sim"] <= real_ref[2]]
        elites = sorted(liminal_band_all, key=lambda m: -m["novelty"])[:15]
        user_picks = load_user_picks()

        print(f"\n=== generation {gen} | elapsed {elapsed_h:.2f}h | spend ${spent:.3f} | "
              f"plateau_count {state['plateau_count']} | user_picks {len(user_picks)} ===", flush=True)

        jobs = build_generation_jobs(gen, subjects, elites, user_picks, GEN_BATCH_SIZE)
        new_manifest = submit_and_collect(jobs, OUT_DIR, f"gen{gen}")
        new_scored = score_batch(model, real_embs, real_centroid, prev_embs, new_manifest)
        manifest.extend(new_scored)
        with open(f"{OUT_DIR}/scored_manifest.json", "w") as f:
            json.dump(manifest, f, indent=1)

        best_novelty = build_gallery(manifest, real_ref) if manifest else 0.0
        state["novelty_history"].append(best_novelty)
        print(f"generation {gen}: {len(new_scored)} new images, cumulative {len(manifest)}, "
              f"liminal-band best novelty (vs. real+round1) {best_novelty:.4f}", flush=True)

        hist = state["novelty_history"]
        if len(hist) > PLATEAU_WINDOW and max(hist[-PLATEAU_WINDOW:]) <= max(hist[:-PLATEAU_WINDOW]) + PLATEAU_EPSILON:
            state["plateau_count"] += 1
            print(f"PLATEAU detected (count={state['plateau_count']})", flush=True)
            if state["plateau_count"] % 3 == 1:
                more = request_gpt55_subjects(subjects, n=20)
                if more:
                    subjects = subjects + more
                    state["gpt55_subjects"] = state["gpt55_subjects"] + more
                    add_to_shared_seed_pool(more, source="gpt5.5-round2")
        save_state(state)

    print(f"\nROUND 2 RUN ENDED at generation {state['generation']}, {len(manifest)} total images, "
          f"gallery at {OUT_DIR}/gallery.html", flush=True)


if __name__ == "__main__":
    main()
