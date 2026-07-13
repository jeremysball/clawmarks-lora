"""
ComfyUI workflow submission against a RunPod serverless endpoint. Moved verbatim from
notes/run_uncanny_sweep.py, where the build_workflow/api_post/api_get functions lived; the
search driver (search/driver.py) and the curation server (curation_server.py) both call
these to submit single jobs and poll for results.
"""
import json
import os
import urllib.request

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT = "uix4vdb2cec7sb"
BASE = f"https://api.runpod.ai/v2/{ENDPOINT}"

NEG_DEFAULT = "low quality, blurry, watermark"
TRIGGER = "trentbuckle style, "


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


def cancel_job(job_id):
    """Cancels a still-running RunPod serverless job. Used when a driver gives up polling a job
    (see search/driver.py's submit_and_collect): the job keeps running and billing on the
    provider side if it's merely abandoned client-side instead of actually cancelled."""
    return api_post(f"/cancel/{job_id}", {})
