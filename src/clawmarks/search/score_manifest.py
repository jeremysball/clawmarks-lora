"""
Scores every image from notes/uncanny_sweep/manifest.json on two axes (DINOv2 centroid
similarity = faithfulness, 1 - nearest-neighbor similarity to any single real image =
novelty) and writes notes/uncanny_sweep/scored_manifest.json plus real_ref.json (the
leave-one-out reference band computed from the real training images).

This is the expensive step (loads DINOv2, embeds every real and generated image) and stays a
standalone script, run once per search round after notes/run_uncanny_sweep.py finishes, rather
than part of the live per-request rendering path curation_server.py's page views belong to.

Run after notes/run_uncanny_sweep.py finishes (or on manifest_partial.json if run early):
    python -m clawmarks.search.score_manifest <out_dir> [manifest_path]
"""
import os
import sys
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel

from clawmarks.atomic_io import atomic_json_write
from clawmarks.config import ROOT

MODEL_ID = "facebook/dinov2-base"
REAL_DIR = f"{ROOT}/corrected_dataset_extract"

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess(img):
    img = img.convert("RGB")
    w, h = img.size
    shortest = 256
    if w < h:
        new_w, new_h = shortest, round(h * shortest / w)
    else:
        new_h, new_w = shortest, round(w * shortest / h)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left, top = (new_w - 224) // 2, (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return (t - IMAGE_MEAN) / IMAGE_STD


def embed_images(paths, batch_size=16, model=None):
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            tensors = [preprocess(Image.open(p)) for p in batch]
            pixel_values = torch.stack(tensors, dim=0)
            out = model(pixel_values=pixel_values)
            feats = out.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats)
    return torch.cat(embs, dim=0)


def partition_by_existing_file(manifest):
    """Split manifest entries into (present, missing) by whether m["file"] exists on disk,
    without dropping the missing ones: a missing file means quarantine and report the entry,
    never silently delete it from the record."""
    present = [m for m in manifest if os.path.exists(m["file"])]
    missing = [m for m in manifest if not os.path.exists(m["file"])]
    return present, missing


def merge_quarantine_entries(prior, new):
    """Accumulates quarantined entries across runs, keyed by "file", instead of each run's
    quarantine write replacing the last: a prior run's quarantined entry that isn't
    re-quarantined this run (its manifest entry no longer exists at all, say) would otherwise
    silently vanish from the record the moment a later run overwrote the file."""
    merged = {m["file"]: m for m in prior}
    merged.update({m["file"]: m for m in new})
    return list(merged.values())


def _default_manifest(out_dir):
    full = out_dir / "manifest.json"
    partial = out_dir / "manifest_partial.json"
    if full.exists():
        return str(full)
    if partial.exists():
        print(f"NOTE: {full} not found yet, building from partial results ({partial}). "
              f"Some planned jobs may still be missing; that's fine, this doesn't wait for "
              f"100% completion.", flush=True)
        return str(partial)
    raise FileNotFoundError("neither manifest.json nor manifest_partial.json exists yet")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        raise SystemExit("usage: python -m clawmarks.search.score_manifest <out_dir> [manifest_path]")
    out_dir = Path(argv[0])
    manifest_path = argv[1] if len(argv) > 1 else _default_manifest(out_dir)
    print("loading DINOv2...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    real_paths = sorted(os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    real_embs = embed_images(real_paths, model=model)
    centroid = real_embs.mean(dim=0)
    centroid = centroid / centroid.norm()

    # Leave-one-out reference: how similar does each real image score against the centroid of
    # the *other* real images? This anchors "definitely in-style" on an absolute scale, rather
    # than the generated batch's own quartiles, which are skewed by deliberately extreme
    # settings (strength up to 1.8, CFG up to 12) and out-of-domain prompts (flagged by an
    # external reviewer, Fable, 2026-07-09: batch-relative quartiles alone could put the
    # "middle faithfulness" bins in garbage territory if half the batch is fried noise).
    n_real = real_embs.shape[0]
    loo_sims = []
    for i in range(n_real):
        others = torch.cat([real_embs[:i], real_embs[i + 1:]], dim=0)
        loo_centroid = others.mean(dim=0)
        loo_centroid = loo_centroid / loo_centroid.norm()
        loo_sims.append((real_embs[i] @ loo_centroid).item())
    real_ref_mean = sum(loo_sims) / len(loo_sims)
    real_ref_min = min(loo_sims)
    real_ref_max = max(loo_sims)
    print(f"real-image leave-one-out reference: mean={real_ref_mean:.4f} "
          f"min={real_ref_min:.4f} max={real_ref_max:.4f}", flush=True)

    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"scoring {len(manifest)} generated images...", flush=True)

    manifest, quarantined = partition_by_existing_file(manifest)
    if quarantined:
        quarantine_file = f"{out_dir}/manifest_quarantine.json"
        prior_quarantined = []
        if os.path.exists(quarantine_file):
            with open(quarantine_file) as f:
                prior_quarantined = json.load(f)
        merged_quarantined = merge_quarantine_entries(prior_quarantined, quarantined)
        atomic_json_write(quarantine_file, merged_quarantined)
        print(f"WARNING: {len(quarantined)} manifest entries have no file on disk; quarantined "
              f"to {quarantine_file} ({len(merged_quarantined)} total accumulated) instead of "
              f"being dropped", flush=True)

    paths = [m["file"] for m in manifest]
    embs = embed_images(paths, model=model)

    centroid_sim = (embs @ centroid).tolist()
    nn_matrix = embs @ real_embs.T
    nn_sim = nn_matrix.max(dim=1).values.tolist()

    for m, cs, ns in zip(manifest, centroid_sim, nn_sim):
        m["centroid_sim"] = cs
        m["novelty"] = 1 - ns
        m["prompt_type"] = "style" if m["prompt_name"].startswith("style_") else "conflict"

    atomic_json_write(f"{out_dir}/scored_manifest.json", manifest)
    atomic_json_write(f"{out_dir}/real_ref.json",
                       {"mean": real_ref_mean, "min": real_ref_min, "max": real_ref_max})

    print(f"DONE: wrote {out_dir}/scored_manifest.json ({len(manifest)} images scored)", flush=True)


if __name__ == "__main__":
    main()
