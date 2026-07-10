"""
Re-embeds every image in scored_manifest.json with DINOv2 and computes each image's top-K
nearest neighbors by cosine similarity, so scan.html can offer "show me images similar to
this one" instead of only sort/filter by score. compute_data() is a data-only live-cache
target with no route of its own; scan.html's own compute_data() depends on it.
"""
import os, json, time
import torch
from transformers import AutoModel

from clawmarks.search.score_manifest import preprocess, MODEL_ID

TOP_K = 16
BATCH_SIZE = 16
CHECKPOINT_EVERY = 10  # batches


def embed_with_progress(paths, model, sweep_dir):
    from PIL import Image
    checkpoint_file = f"{sweep_dir}/similarity_embed_checkpoint.pt"
    n = len(paths)
    done_embs = []
    start_i = 0
    if os.path.exists(checkpoint_file):
        ckpt = torch.load(checkpoint_file)
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
                            "embs": torch.cat(done_embs, dim=0)}, checkpoint_file)
                batches_since_checkpoint = 0

    return torch.cat(done_embs, dim=0)


def compute_data(sweep_dir):
    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    paths = [m["file"] for m in manifest]
    tags = [m["tag"] for m in manifest]

    print(f"loading DINOv2 and embedding {len(paths)} images...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    embs = embed_with_progress(paths, model, sweep_dir)
    checkpoint_file = f"{sweep_dir}/similarity_embed_checkpoint.pt"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print("embeddings done, computing pairwise similarity...", flush=True)

    sim = embs @ embs.T  # already L2-normalized in embed_images
    sim.fill_diagonal_(-1.0)
    topk = sim.topk(TOP_K, dim=1).indices.tolist()

    return {tags[i]: [tags[j] for j in topk[i]] for i in range(len(tags))}
