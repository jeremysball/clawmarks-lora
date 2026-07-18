"""
Re-embeds every image in scored_manifest.json (plus the real training images) with DINOv2,
then returns the data three of the exploration tools need:

  1/2. a 2D UMAP projection of the full embedding space (real images + every generated image),
       with generation number attached, for map.html's scatter plot and generation slider.
       Answers "what does the search space actually look like" and "is round N finding new
       territory or re-treading round N-1's," which the faithfulness x novelty plane (two
       derived scalars) can't show on its own.
  4.   top-K nearest-neighbor lists WITH the actual cosine similarity values attached (not just
       neighbor identity, unlike similarity_index.py), so redundancy.html can cluster
       near-duplicates at an adjustable threshold.
  6.   nearest_real_idx per image - which of the ~31 real training images each generation is
       closest to (mode-collapse check: if generations only ever anchor to a handful of the
       real images, the search is faithful to a sliver of the style, not the whole thing).

This duplicates similarity_index.py's embedding pass rather than importing its output, because
that module discards the raw embeddings once it's done with them (only top-16 neighbor *tags*
survive) and UMAP/nearest-real-image both need the actual vectors. The finished embeddings are
cached on disk at solution_map_final_embs.pt purely as an internal speed-up (skips DINOv2
entirely on a cache hit); this is an implementation detail, not an output other tools consume.

compute_data() is a data-only live-cache target with no route of its own; map.html and
redundancy.html both depend on it (target name "solution-map"), via curation_server.py calling
LiveCache.get(..., depends_on=["solution-map"]).
"""
import os
import json
import time
import re
import torch
from transformers import AutoModel

from clawmarks.search.score_manifest import preprocess, MODEL_ID, REAL_DIR
from clawmarks.durable_records import sha256_json

TOP_K = 16
BATCH_SIZE = 16
CHECKPOINT_EVERY = 10


def embed_with_progress(paths, model, label, sweep_dir):
    from PIL import Image
    checkpoint_file = f"{sweep_dir}/solution_map_embed_checkpoint.pt"
    n = len(paths)
    done_embs = []
    start_i = 0
    if os.path.exists(checkpoint_file):
        ckpt = torch.load(checkpoint_file)
        if ckpt["paths"] == paths[:ckpt["n_done"]]:
            done_embs = [ckpt["embs"]]
            start_i = ckpt["n_done"]
            print(f"[{label}] resuming from checkpoint: {start_i}/{n} already embedded", flush=True)
        else:
            print(f"[{label}] checkpoint doesn't match current manifest order, starting fresh", flush=True)

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
            print(f"[{label}] embedded {n_done}/{n} ({rate:.1f} img/s, ETA {eta_min:.1f} min)", flush=True)

            batches_since_checkpoint += 1
            if batches_since_checkpoint >= CHECKPOINT_EVERY:
                torch.save({"paths": paths[:n_done], "n_done": n_done,
                            "embs": torch.cat(done_embs, dim=0)}, checkpoint_file)
                batches_since_checkpoint = 0

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    return torch.cat(done_embs, dim=0)


def generation_of(tag):
    m = re.match(r"gen(\d+)_", tag)
    return int(m.group(1)) if m else 0


def compute_data(sweep_dir):
    with open(f"{sweep_dir}/scored_manifest.json") as f:
        manifest = json.load(f)

    paths = [m["file"] for m in manifest]
    tags = [m["tag"] for m in manifest]

    real_paths = sorted(os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    final_embs_file = f"{sweep_dir}/solution_map_final_embs.pt"
    if os.path.exists(final_embs_file):
        print(f"loading already-computed embeddings from {final_embs_file} (skipping DINOv2 entirely)...", flush=True)
        saved = torch.load(final_embs_file)
        if saved["paths"] == paths and saved["real_paths"] == real_paths:
            real_embs, gen_embs = saved["real_embs"], saved["gen_embs"]
        else:
            # Don't delete the file here: it's the only surviving embedding data if the
            # source images are ever gone by the time this runs again, and the recompute
            # below already overwrites it in place on success. A mismatch (e.g. running
            # from a different checkout path, which changes REAL_DIR's absolute prefix)
            # should never be able to destroy the last good copy.
            print("saved embeddings don't match current manifest, re-embedding from scratch", flush=True)
            real_embs = gen_embs = None
    else:
        real_embs = gen_embs = None

    if real_embs is None:
        print("loading DINOv2...", flush=True)
        model = AutoModel.from_pretrained(MODEL_ID)
        model.eval()

        print(f"embedding {len(real_paths)} real training images...", flush=True)
        real_embs = embed_with_progress(real_paths, model, "real", sweep_dir)

        print(f"embedding {len(paths)} generated images...", flush=True)
        gen_embs = embed_with_progress(paths, model, "gen", sweep_dir)

        # Persist the finished embeddings immediately, before UMAP runs: UMAP/sklearn version
        # mismatches are a real failure mode we've already hit once, and re-embedding 3392+31 images
        # takes ~28 minutes of wall-clock, so a UMAP-only crash should never force redoing this part.
        torch.save({"paths": paths, "real_paths": real_paths, "real_embs": real_embs, "gen_embs": gen_embs},
                   final_embs_file)
        print(f"saved final embeddings to {final_embs_file}", flush=True)

    # --- Idea 4: pairwise similarity WITH scores, for redundancy clustering ---
    print("computing pairwise similarity (with scores) for redundancy clustering...", flush=True)
    sim = gen_embs @ gen_embs.T
    sim.fill_diagonal_(-1.0)
    topk_vals, topk_idx = sim.topk(TOP_K, dim=1)
    similarity_scored = {
        tags[i]: [[tags[j], round(topk_vals[i][k].item(), 4)] for k, j in enumerate(topk_idx[i].tolist())]
        for i in range(len(tags))
    }

    # --- Idea 6: nearest real image per generated image ---
    nn_real_matrix = gen_embs @ real_embs.T
    nearest_real_idx = nn_real_matrix.argmax(dim=1).tolist()
    nearest_real_sim = nn_real_matrix.max(dim=1).values.tolist()
    real_names = [os.path.basename(p) for p in real_paths]

    # --- Ideas 1/2: UMAP projection of the joint space (real + generated) ---
    print("fitting UMAP on the joint real+generated embedding space...", flush=True)
    import umap

    all_embs = torch.cat([real_embs, gen_embs], dim=0).numpy()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42, n_jobs=1)
    coords = reducer.fit_transform(all_embs)
    real_coords = coords[:len(real_paths)]
    gen_coords = coords[len(real_paths):]
    print("UMAP projection done.", flush=True)

    points = []
    for i, m in enumerate(manifest):
        points.append({
            "tag": m["tag"],
            "x": round(float(gen_coords[i][0]), 4),
            "y": round(float(gen_coords[i][1]), 4),
            "gen": generation_of(m["tag"]),
            "category": m["category"],
            "prompt_type": m["prompt_type"],
            "prompt_name": m["prompt_name"],
            "faith": round(m["centroid_sim"], 4),
            "novelty": round(m["novelty"], 4),
            "nearest_real": real_names[nearest_real_idx[i]],
            "nearest_real_sim": round(nearest_real_sim[i], 4),
            "thumb": f"thumbs/{m['tag']}.jpg" if os.path.exists(f"{sweep_dir}/thumbs/{m['tag']}.jpg") else os.path.basename(m["file"]),
        })

    real_points = [
        {"name": real_names[i], "x": round(float(real_coords[i][0]), 4), "y": round(float(real_coords[i][1]), 4)}
        for i in range(len(real_paths))
    ]

    return {
        "solution_map_data": {
            "points": points,
            "real_points": real_points,
            "projection_version": sha256_json({"points": points, "real_points": real_points}),
        },
        "similarity_scored": similarity_scored,
    }
