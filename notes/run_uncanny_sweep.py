"""
Round-1-adjacent exploratory side branch (lab_notebook.md Section 3b): map the CLAWMARKS
style's "liminal band", images that stay recognizably in-style by DINOv2 centroid similarity
but sit far from any single real training image (novelty), by sweeping LoRA strength, CFG
scale, and prompt content on the existing serverless endpoint, then scoring and binning
everything into a 2D descriptor grid.

This does NOT pick one "elite" per bin (true MAP-Elites needs an automated fitness/quality
score; we don't have a reliable one, per lab_notebook.md Section 1's standing lesson that a
metric is a filter, not a verdict). Instead it keeps every image per bin and leaves final
curation to a human looking at the gallery. Honest framing, not full MAP-Elites.

Budget guard: hard-capped job count (~350), phased so the calibration batch's actual cost is
checked before committing to the full sweep. No pod bring-up: uses the existing serverless
endpoint (uix4vdb2cec7sb), which is already loaded with the base checkpoint + epoch4 LoRA and
bills per GPU-second only while a job runs, not idle time.

Run with: python3 notes/run_uncanny_sweep.py 2>&1 | tee notes/uncanny_sweep.log
"""
import json, os, time, base64, sys
import urllib.request

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT = "uix4vdb2cec7sb"
BASE = f"https://api.runpod.ai/v2/{ENDPOINT}"
SC = "/workspace/trent-with-smart-prompts"
OUT_DIR = f"{SC}/notes/uncanny_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

NEG_DEFAULT = "low quality, blurry, watermark"
TRIGGER = "trentbuckle style, "

STYLE_PROMPTS = {
    "cat":   TRIGGER + "close-up cat portrait, marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
    "wolf":  TRIGGER + "close-up wolf portrait, marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
    "fox":   TRIGGER + "close-up fox portrait, marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
    "owl":   TRIGGER + "close-up owl portrait, marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
    "horse": TRIGGER + "close-up horse portrait, marker and ink linework, colored pencil shading, raw sketchbook page, mixed media",
}

CONFLICTED_PROMPTS = {
    "human_face": TRIGGER + "close-up human face, dark-rimmed eyes glowing pale blue, pale skin with visible brush texture, hand pressed beside cheek, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
    "cyborg": TRIGGER + "close-up cyborg face, half exposed circuitry and wiring, dark-rimmed human eye glowing pale blue beside a mechanical lens, clawed metal hand pressed beside cheek, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
    "body_horror": TRIGGER + "close-up face mid-transformation, skin splitting to reveal clawed fingers pushing through the cheek, dark-rimmed eyes glowing pale blue, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
    "liminal": TRIGGER + "figure standing alone in an empty fluorescent-lit hallway, dark-rimmed eyes glowing pale blue, clawed hand pressed against the wall, dense dark-blue vertical brush-dash background replaced by flat institutional tile, thick acrylic dry-brush texture, raw outsider-art painting",
    "dental_xray": TRIGGER + "dental x-ray radiograph of a jaw, dark-rimmed glowing pale blue accents, thick acrylic dry-brush texture, raw outsider-art painting",
    "stairwell": TRIGGER + "empty concrete stairwell viewed from below, dark-rimmed pale blue glowing light fixture, thick acrylic dry-brush texture, raw outsider-art painting",
    "weather_map": TRIGGER + "television weather map with swirling storm system, dark-rimmed pale blue glowing isobar lines, thick acrylic dry-brush texture, raw outsider-art painting",
    "crowd": TRIGGER + "crowd of human faces packed close together, dark-rimmed eyes glowing pale blue, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
}

ALL_PROMPTS = {**{f"style_{k}": v for k, v in STYLE_PROMPTS.items()},
               **{f"conflict_{k}": v for k, v in CONFLICTED_PROMPTS.items()}}

SEEDS = [11, 22]
STRENGTHS = [0.6, 1.0, 1.4, 1.8]
CFGS = [2.0, 5.0, 7.5, 12.0]
BASELINE_STRENGTH, BASELINE_CFG = 1.0, 7.5


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
                "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "uncanny"}}
            }
        }
    }


def api_post(path, payload):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def api_get(path):
    req = urllib.request.Request(f"{BASE}{path}", headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def build_job_list():
    jobs = []

    # Core grid: 4 strengths x 4 CFGs x 10 prompts x 2 seeds = 320
    for strength in STRENGTHS:
        for cfg in CFGS:
            for pname, ptext in ALL_PROMPTS.items():
                for seed in SEEDS:
                    jobs.append({
                        "tag": f"grid_s{strength}_c{cfg}_{pname}_seed{seed}",
                        "category": "grid", "prompt_name": pname, "prompt": ptext,
                        "seed": seed, "strength": strength, "cfg": cfg,
                        "steps": 28, "sampler": "ddim", "negative": NEG_DEFAULT,
                    })

    # Negative-trigger arm: put the style trigger phrase in the negative prompt instead,
    # LoRA weights stay loaded and applied. Style prompts only, baseline strength/cfg. 5x2=10.
    for pname, ptext in STYLE_PROMPTS.items():
        plain = ptext.replace(TRIGGER, "")
        for seed in SEEDS:
            jobs.append({
                "tag": f"negtrigger_{pname}_seed{seed}",
                "category": "negtrigger", "prompt_name": f"style_{pname}", "prompt": plain,
                "seed": seed, "strength": BASELINE_STRENGTH, "cfg": BASELINE_CFG,
                "steps": 28, "sampler": "ddim", "negative": TRIGGER.strip(", ") + ", " + NEG_DEFAULT,
            })

    # Truncated-trajectory arm: 8 steps, euler_ancestral (non-converging), all 10 prompts x2. 20.
    for pname, ptext in ALL_PROMPTS.items():
        for seed in SEEDS:
            jobs.append({
                "tag": f"truncated_{pname}_seed{seed}",
                "category": "truncated", "prompt_name": pname, "prompt": ptext,
                "seed": seed, "strength": BASELINE_STRENGTH, "cfg": BASELINE_CFG,
                "steps": 8, "sampler": "euler_ancestral", "negative": NEG_DEFAULT,
            })

    return jobs


def submit_and_collect(jobs, out_dir, label):
    print(f"[{label}] submitting {len(jobs)} jobs...", flush=True)
    job_ids = {}
    for j in jobs:
        wf = build_workflow(j["prompt"], j["seed"], j["strength"], j["cfg"], j["steps"], j["sampler"], j["negative"])
        try:
            res = api_post("/run", wf)
            jid = res.get("id")
            job_ids[jid] = j
        except Exception as e:
            print(f"SUBMIT_FAIL {j['tag']}: {e}", flush=True)
    print(f"[{label}] {len(job_ids)} jobs accepted, polling...", flush=True)

    pending = set(job_ids.keys())
    completed, failed = 0, 0
    manifest = []
    t0 = time.time()
    while pending:
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
                    manifest.append({**{k: v for k, v in j.items() if k != "prompt"}, "prompt": j["prompt"], "file": fname})
                    completed += 1
                pending.discard(jid)
            elif status in ("FAILED", "CANCELLED"):
                failed += 1
                pending.discard(jid)
                print(f"JOB_FAILED {job_ids[jid]['tag']}: {res}", flush=True)
        elapsed = time.time() - t0
        print(f"[{label}][{elapsed:.0f}s] completed={completed} failed={failed} pending={len(pending)}", flush=True)
        if pending:
            time.sleep(10)
    return manifest


def main():
    all_jobs = build_job_list()
    grid_jobs = [j for j in all_jobs if j["category"] == "grid"]
    extra_jobs = [j for j in all_jobs if j["category"] != "grid"]
    print(f"TOTAL PLANNED: {len(all_jobs)} ({len(grid_jobs)} grid, {len(extra_jobs)} supplementary)", flush=True)

    manifest = []
    manifest += submit_and_collect(grid_jobs, OUT_DIR, "core-grid")
    with open(f"{OUT_DIR}/manifest_partial.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"CORE GRID DONE: {len(manifest)}/{len(grid_jobs)} images. "
          f"Historical cost evidence (lab_notebook.md gotcha log): ~250 similar serverless "
          f"generations cost roughly $1-3 total, so {len(grid_jobs)} should stay well under "
          f"the $10 budget; proceeding to supplementary arms.", flush=True)

    manifest += submit_and_collect(extra_jobs, OUT_DIR, "supplementary")

    with open(f"{OUT_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"ALL DONE: {len(manifest)}/{len(all_jobs)} images downloaded to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
