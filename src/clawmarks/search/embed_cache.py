"""
DINOv2 embedding cache: computes and persists an embedding per image in scored_manifest.json so
the preference model (search/preference_model.py) can train on frozen features without
re-running the (slow) DINOv2 model every time. Runs locally, no RunPod cost. See
docs/superpowers/specs/2026-07-09-preference-classifier-design.md, Component 1.

Run with: python -m clawmarks.search.embed_cache
"""
import json
import os

import numpy as np
import torch
from PIL import Image

from clawmarks.config import SWEEP_DIR

MODEL_ID = "facebook/dinov2-base"
EMBEDDINGS_FILE = SWEEP_DIR / "embeddings.npz"

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
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = (t - IMAGE_MEAN) / IMAGE_STD
    return t


def embed_paths(paths, model, batch_size=16):
    """Returns an (N, D) float32 array of L2-normalized embeddings, one row per path, in the
    same order as `paths`."""
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            tensors = [preprocess(Image.open(p)) for p in batch_paths]
            pixel_values = torch.stack(tensors, dim=0)
            out = model(pixel_values=pixel_values)
            feats = out.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.detach().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def load_cache(path):
    """Returns (tags, embeddings). Empty list/array if the file doesn't exist yet."""
    if not os.path.exists(path):
        return [], np.zeros((0, 0), dtype=np.float32)
    data = np.load(path)
    return list(data["tags"]), data["embeddings"]


def save_cache(path, tags, embeddings):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, tags=np.array(tags), embeddings=np.asarray(embeddings, dtype=np.float32))
    os.replace(tmp, path)


def missing_tags(manifest_tags, cached_tags):
    cached = set(cached_tags)
    return [t for t in manifest_tags if t not in cached]


def sync(manifest, cache_path, model, image_path_for):
    """Loads the existing cache, embeds any manifest tag missing from it, appends, and saves.
    `image_path_for(tag)` resolves a manifest tag to its image file path. Raises
    FileNotFoundError (listing the offending tag) if a manifest tag's image file doesn't exist,
    rather than silently skipping it. Returns (tags, embeddings) for the full, updated cache."""
    tags, embeddings = load_cache(cache_path)
    manifest_tags = [m["tag"] for m in manifest]
    to_add = missing_tags(manifest_tags, tags)
    if not to_add:
        return tags, embeddings

    missing_paths = []
    for t in to_add:
        p = image_path_for(t)
        if not os.path.exists(p):
            raise FileNotFoundError(f"tag {t!r} is in the manifest but its image file is missing: {p}")
        missing_paths.append(p)

    new_embeddings = embed_paths(missing_paths, model)
    all_tags = list(tags) + to_add
    all_embeddings = new_embeddings if embeddings.size == 0 else np.concatenate([embeddings, new_embeddings], axis=0)
    save_cache(cache_path, all_tags, all_embeddings)
    return all_tags, all_embeddings


def main(argv=None):
    from transformers import AutoModel

    with open(SWEEP_DIR / "scored_manifest.json") as f:
        manifest = json.load(f)
    by_tag = {m["tag"]: m for m in manifest}

    def image_path_for(tag):
        # Falls back to the thumbnail when the full-res file is missing.
        full_res = str(SWEEP_DIR / by_tag[tag]["file"])
        if os.path.exists(full_res):
            return full_res
        return str(SWEEP_DIR / "thumbs" / f"{tag}.jpg")

    print("loading DINOv2 model...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    tags, _ = sync(manifest, EMBEDDINGS_FILE, model, image_path_for)
    print(f"embedding cache now covers {len(tags)} images at {EMBEDDINGS_FILE}", flush=True)


if __name__ == "__main__":
    main()
