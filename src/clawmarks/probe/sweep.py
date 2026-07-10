"""
Probe/calibration scripts combined into one module. Each entry point matches the body of the
original standalone script (notes/probe_uncanny.py, notes/probe_strength_sweep.py,
notes/gen_samples.py), wrapped in a named function so the `clawmarks` CLI can call it
directly without a subprocess.
"""
import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request

import paramiko

from clawmarks.config import PROBE_DIR, PROBE_STRENGTH_DIR, ROOT

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT = "uix4vdb2cec7sb"
BASE = f"https://api.runpod.ai/v2/{ENDPOINT}"

KEY_PATH = f"{ROOT}/runpod-ssh/id_ed25519"
LOCAL_PROMPTS_FILE = "/tmp/art_prompts_base_v2.txt"
PROMPT_LINES = [1, 41, 47, 50]  # cat split-color, galloping horse, tiger stripe fragment, wolf-cat hybrid
SEED = 42
STEPS = 28
SCALE = 7.5
RESOLUTION = "1024,1024"


# --- probe_uncanny (originally notes/probe_uncanny.py) ---

PROBES_PROMPTS = {
    "human_face": "close-up human face, dark-rimmed eyes glowing pale blue, pale skin with visible brush texture, hand pressed beside cheek, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
    "cyborg": "close-up cyborg face, half exposed circuitry and wiring, dark-rimmed human eye glowing pale blue beside a mechanical lens, clawed metal hand pressed beside cheek, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
    "body_horror": "close-up face mid-transformation, skin splitting to reveal clawed fingers pushing through the cheek, dark-rimmed eyes glowing pale blue, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
    "liminal": "figure standing alone in an empty fluorescent-lit hallway, dark-rimmed eyes glowing pale blue, clawed hand pressed against the wall, dense dark-blue vertical brush-dash background replaced by flat institutional tile, thick acrylic dry-brush texture, raw outsider-art painting",
}
PROBES_SEEDS = [11, 22]


def _probe_build_workflow(prompt, seed):
    return {
        "input": {
            "workflow": {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "illustrious_v0.1.safetensors"}},
                "2": {"class_type": "LoraLoader", "inputs": {
                    "lora_name": "clawmarks-illustrious-v3-epoch4.safetensors",
                    "strength_model": 1.0, "strength_clip": 1.0,
                    "model": ["1", 0], "clip": ["1", 1]}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 1]}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality, blurry, watermark", "clip": ["2", 1]}},
                "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
                "6": {"class_type": "KSampler", "inputs": {
                    "seed": seed, "steps": 28, "cfg": 7.5, "sampler_name": "ddim", "scheduler": "normal",
                    "denoise": 1.0, "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0],
                    "latent_image": ["5", 0]}},
                "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
                "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "clawmarks"}}
            }
        }
    }


def _probe_api_post(path, payload):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _probe_api_get(path):
    req = urllib.request.Request(f"{BASE}{path}", headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def run_probe_uncanny():
    os.makedirs(PROBE_DIR, exist_ok=True)
    jobs = []
    for label, prompt in PROBES_PROMPTS.items():
        for seed in PROBES_SEEDS:
            jobs.append((label, prompt, seed))

    print(f"Total jobs to submit: {len(jobs)}")
    job_ids = {}
    for label, prompt, seed in jobs:
        wf = _probe_build_workflow(prompt, seed)
        res = _probe_api_post("/run", wf)
        jid = res.get("id")
        job_ids[jid] = (label, prompt, seed)
        print(f"submitted {label} seed={seed} -> {jid}")

    pending = set(job_ids.keys())
    completed = 0
    failed = 0
    manifest = []
    t0 = time.time()
    while pending:
        for jid in list(pending):
            try:
                res = _probe_api_get(f"/status/{jid}")
            except Exception:
                continue
            status = res.get("status")
            if status == "COMPLETED":
                label, prompt, seed = job_ids[jid]
                images = res.get("output", {}).get("images", [])
                if images:
                    fname = f"{PROBE_DIR}/{label}_seed{seed}.png"
                    with open(fname, "wb") as f:
                        f.write(base64.b64decode(images[0]["data"]))
                    manifest.append({"label": label, "prompt": prompt, "seed": seed, "file": fname})
                    completed += 1
                pending.discard(jid)
            elif status in ("FAILED", "CANCELLED"):
                failed += 1
                pending.discard(jid)
                print(f"JOB_FAILED {jid}: {res}")
        elapsed = time.time() - t0
        print(f"[{elapsed:.0f}s] completed={completed} failed={failed} pending={len(pending)}")
        if pending:
            time.sleep(8)

    with open(f"{PROBE_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"DONE completed={completed} failed={failed}")


# --- probe_strength_sweep (originally notes/probe_strength_sweep.py) ---

STRENGTH_PROMPTS = {
    "human_face": "close-up human face, dark-rimmed eyes glowing pale blue, pale skin with visible brush texture, hand pressed beside cheek, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
    "cyborg": "close-up cyborg face, half exposed circuitry and wiring, dark-rimmed human eye glowing pale blue beside a mechanical lens, clawed metal hand pressed beside cheek, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting",
    "liminal": "figure standing alone in an empty fluorescent-lit hallway, dark-rimmed eyes glowing pale blue, clawed hand pressed against the wall, dense dark-blue vertical brush-dash background replaced by flat institutional tile, thick acrylic dry-brush texture, raw outsider-art painting",
}
CONTROL_PROMPT = "trentbuckle, close-up cat face, dark-rimmed eyes glowing pale blue, orange fur, clawed hand pressed beside cheek, dense dark-blue vertical brush-dash background, thick acrylic dry-brush texture, raw outsider-art painting"

STRENGTHS = [0.0, 0.5, 0.75, 1.0, 1.3, 1.6]
STRENGTH_SEED = 11
CONTROL_SEEDS = [11, 22]


def _strength_build_workflow(prompt, seed, strength):
    return {
        "input": {
            "workflow": {
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "illustrious_v0.1.safetensors"}},
                "2": {"class_type": "LoraLoader", "inputs": {
                    "lora_name": "clawmarks-illustrious-v3-epoch4.safetensors",
                    "strength_model": strength, "strength_clip": strength,
                    "model": ["1", 0], "clip": ["1", 1]}},
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 1]}},
                "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality, blurry, watermark", "clip": ["2", 1]}},
                "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
                "6": {"class_type": "KSampler", "inputs": {
                    "seed": seed, "steps": 28, "cfg": 7.5, "sampler_name": "ddim", "scheduler": "normal",
                    "denoise": 1.0, "model": ["2", 0], "positive": ["3", 0], "negative": ["4", 0],
                    "latent_image": ["5", 0]}},
                "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
                "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "clawmarks"}}
            }
        }
    }


def _strength_api_post(path, payload):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _strength_api_get(path):
    req = urllib.request.Request(f"{BASE}{path}", headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def run_strength_sweep():
    os.makedirs(PROBE_STRENGTH_DIR, exist_ok=True)
    jobs = []
    for label, prompt in STRENGTH_PROMPTS.items():
        for strength in STRENGTHS:
            jobs.append((f"{label}_str{strength}", prompt, STRENGTH_SEED, strength))
    for seed in CONTROL_SEEDS:
        jobs.append((f"control_cat_str1.0_seed{seed}", CONTROL_PROMPT, seed, 1.0))

    print(f"Total jobs to submit: {len(jobs)}")
    job_ids = {}
    for label, prompt, seed, strength in jobs:
        wf = _strength_build_workflow(prompt, seed, strength)
        res = _strength_api_post("/run", wf)
        jid = res.get("id")
        job_ids[jid] = (label, prompt, seed, strength)
        print(f"submitted {label} -> {jid}")

    pending = set(job_ids.keys())
    completed = 0
    failed = 0
    manifest = []
    t0 = time.time()
    while pending:
        for jid in list(pending):
            try:
                res = _strength_api_get(f"/status/{jid}")
            except Exception:
                continue
            status = res.get("status")
            if status == "COMPLETED":
                label, prompt, seed, strength = job_ids[jid]
                images = res.get("output", {}).get("images", [])
                if images:
                    fname = f"{PROBE_STRENGTH_DIR}/{label}.png"
                    with open(fname, "wb") as f:
                        f.write(base64.b64decode(images[0]["data"]))
                    manifest.append({"label": label, "prompt": prompt, "seed": seed, "strength": strength, "file": fname})
                    completed += 1
                pending.discard(jid)
            elif status in ("FAILED", "CANCELLED"):
                failed += 1
                pending.discard(jid)
                print(f"JOB_FAILED {jid}: {res}")
        elapsed = time.time() - t0
        print(f"[{elapsed:.0f}s] completed={completed} failed={failed} pending={len(pending)}")
        if pending:
            time.sleep(8)

    with open(f"{PROBE_STRENGTH_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"DONE completed={completed} failed={failed}")


# --- gen_samples (originally notes/gen_samples.py) ---

def _read_host_port(host_module):
    text = open(host_module).read()
    host = re.search(r'HOST = "(.*)"', text).group(1)
    port = int(re.search(r"PORT = (\d+)", text).group(1))
    return host, port


def _gen_ssh_client(pod):
    host_module = f"{ROOT}/rpssh.py" if pod == 1 else f"{ROOT}/rpssh{pod}.py"
    host, port = _read_host_port(host_module)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = paramiko.Ed25519Key.from_private_key_file(KEY_PATH)
    client.connect(host, port=port, username="root", pkey=pkey, timeout=20)
    client.get_transport().set_keepalive(30)
    return client


def _gen_run_cmd(client, cmd, timeout=None):
    print(f"+ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    for line in iter(stdout.readline, ""):
        print(line, end="")
    code = stdout.channel.recv_exit_status()
    err = stderr.read().decode(errors="replace")
    if err.strip():
        print("STDERR:", err, file=sys.stderr)
    return code


def gen_samples():
    """
    Generate the fixed 4-prompt sample set from a trained checkpoint, directly on the pod, via
    kohya's sdxl_gen_img.py. Used for every probe/full checkpoint so different directions and
    lengths are visually and quantitatively comparable on identical prompts/seed/sampler settings.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="run name, e.g. controlB_156")
    ap.add_argument("--pod", type=int, default=1, choices=[1, 2])
    ap.add_argument("--remote-ckpt-dir", default=None, help="defaults to --checkpoint")
    args = ap.parse_args()

    remote_dir = args.remote_ckpt_dir or args.checkpoint
    remote_ckpt = f"/workspace/output/{remote_dir}/{args.checkpoint}.safetensors"
    remote_out = f"/workspace/samples_out/{args.checkpoint}"
    remote_prompts = f"/workspace/prompts_{args.checkpoint}.txt"

    with open(LOCAL_PROMPTS_FILE) as f:
        all_lines = f.read().splitlines()
    prompts = [all_lines[i - 1] for i in PROMPT_LINES]

    client = _gen_ssh_client(args.pod)
    sftp = client.open_sftp()
    with sftp.open(remote_prompts, "w") as rf:
        rf.write("\n".join(prompts) + "\n")
    sftp.close()

    _gen_run_cmd(client, f"mkdir -p {remote_out}")
    cmd = " ".join([
        "source /workspace/venv/bin/activate &&",
        "cd /workspace/kohya_ss &&",
        "python3 sdxl_gen_img.py",
        "--ckpt /workspace/models/illustrious_v0.1.safetensors",
        f"--network_module networks.lora --network_weights {remote_ckpt}",
        f"--from_file {remote_prompts}",
        f"--outdir {remote_out}",
        f"--seed {SEED}",
        f"--steps {STEPS}",
        f"--scale {SCALE}",
        f"--W {RESOLUTION.split(',')[0]} --H {RESOLUTION.split(',')[1]}",
        "--sampler ddim",
        "--images_per_prompt 1",
        "--xformers",
        f"> {remote_out}/gen.log 2>&1",
    ])
    code = _gen_run_cmd(client, cmd, timeout=1200)
    if code != 0:
        print(f"GENERATION FAILED (exit {code}), see remote log at {remote_out}/gen.log")
        client.close()
        sys.exit(code)

    local_dir = f"{ROOT}/notes/probe_samples/{args.checkpoint}"
    os.makedirs(local_dir, exist_ok=True)
    sftp = client.open_sftp()
    remote_files = sorted(f for f in sftp.listdir(remote_out) if f.endswith(".png"))
    for i, fname in enumerate(remote_files, start=1):
        local_name = f"im_{i:06d}.png"
        sftp.get(f"{remote_out}/{fname}", f"{local_dir}/{local_name}")
        print(f"downloaded {fname} -> {local_name}")
    sftp.close()
    client.close()
    print(f"DONE: {args.checkpoint} -> {local_dir}")
