"""
Re-embeds every image in scored_manifest.json with DINOv2 and computes each image's top-K
nearest neighbors by cosine similarity, so the scan gallery (notes/build_scan_gallery.py) can
offer "show me images similar to this one" instead of only sort/filter by score. Writes each
image's neighbor list straight into scan.html's embedded data (re-run build_scan_gallery.py
after this to pick it up), via a shared similarity.json intermediate.

Run: python3 notes/build_similarity_index.py
"""
import os, sys, json, time
import torch
from transformers import AutoModel

sys.path.insert(0, os.path.dirname(__file__))
from build_uncanny_gallery import preprocess, MODEL_ID

SWEEP_DIR = "/workspace/trent-with-smart-prompts/notes/uncanny_sweep"
TOP_K = 16
BATCH_SIZE = 16
CHECKPOINT_FILE = f"{SWEEP_DIR}/similarity_embed_checkpoint.pt"
CHECKPOINT_EVERY = 10  # batches

with open(f"{SWEEP_DIR}/scored_manifest.json") as f:
    manifest = json.load(f)

paths = [m["file"] for m in manifest]
tags = [m["tag"] for m in manifest]


def embed_with_progress(paths, model):
    from PIL import Image
    n = len(paths)
    done_embs = []
    start_i = 0
    if os.path.exists(CHECKPOINT_FILE):
        ckpt = torch.load(CHECKPOINT_FILE)
        if ckpt["paths"] == paths[:ckpt["n_done"]]:
            done_embs = [ckpt["embs"]]
            start_i = ckpt["n_done"]
            print(f"resuming from checkpoint: {start_i}/{n} already embedded", flush=True)
        else:
            print("checkpoint doesn't match current manifest order, starting fresh", flush=True)

    t0 = time.time()
    batches_since_checkpoint = 0
    with torch.no_grad():
        for i in range(start_i, n, BATCH_SIZE):
            batch = paths[i:i + BATCH_SIZE]
            tensors = [preprocess(Image.open(p)) for p in batch]
            pixel_values = torch.stack(tensors, dim=0)
            out = model(pixel_values=pixel_values)
            feats = out.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            done_embs.append(feats)

            n_done = min(i + BATCH_SIZE, n)
            elapsed = time.time() - t0
            rate = (n_done - start_i) / elapsed if elapsed > 0 else 0
            eta_min = (n - n_done) / rate / 60 if rate > 0 else float("inf")
            print(f"embedded {n_done}/{n} ({rate:.1f} img/s, ETA {eta_min:.1f} min)", flush=True)

            batches_since_checkpoint += 1
            if batches_since_checkpoint >= CHECKPOINT_EVERY:
                torch.save({"paths": paths[:n_done], "n_done": n_done,
                            "embs": torch.cat(done_embs, dim=0)}, CHECKPOINT_FILE)
                batches_since_checkpoint = 0

    return torch.cat(done_embs, dim=0)


print(f"loading DINOv2 and embedding {len(paths)} images...", flush=True)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()
embs = embed_with_progress(paths, model)
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
print("embeddings done, computing pairwise similarity...", flush=True)

sim = embs @ embs.T  # already L2-normalized in embed_images
sim.fill_diagonal_(-1.0)
topk = sim.topk(TOP_K, dim=1).indices.tolist()

neighbors = {tags[i]: [tags[j] for j in topk[i]] for i in range(len(tags))}

with open(f"{SWEEP_DIR}/similarity.json", "w") as f:
    json.dump(neighbors, f)

print(f"wrote {SWEEP_DIR}/similarity.json ({len(neighbors)} entries, top-{TOP_K} each)", flush=True)
