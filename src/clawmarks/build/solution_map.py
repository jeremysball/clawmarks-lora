"""
Re-embeds every image in scored_manifest.json (plus the real training images) with DINOv2,
then builds the data three of the new exploration tools need:

  1/2. solution_map_data.json  - a 2D UMAP projection of the full embedding space (real images
       + every generated image), with generation number attached, for map.html's scatter plot
       and generation slider. Answers "what does the search space actually look like" and
       "is round N finding new territory or re-treading round N-1's," which the faithfulness x
       novelty plane (two derived scalars) can't show on its own.
  4.   similarity_scored.json  - same top-K nearest-neighbor lists as build_similarity_index.py,
       but with the actual cosine similarity values attached (not just neighbor identity), so
       redundancy.html can cluster near-duplicates at an adjustable threshold.
  6.   nearest_real_idx per image - which of the ~31 real training images each generation is
       closest to, folded into solution_map_data.json so map.html and real_anchor.html can both
       use it (mode-collapse check: if generations only ever anchor to a handful of the real
       images, the search is faithful to a sliver of the style, not the whole thing).

This duplicates build_similarity_index.py's embedding pass rather than importing its output,
because that script discards the raw embeddings once it's done with them (only top-16 neighbor
*tags* survive to disk) and UMAP/nearest-real-image both need the actual vectors.

Run: python3 -m clawmarks.build.solution_map
"""
import os, sys, json, time, re
import torch
from transformers import AutoModel

from clawmarks.config import SWEEP_DIR
from clawmarks.build.uncanny_gallery import preprocess, MODEL_ID, REAL_DIR

TOP_K = 16
BATCH_SIZE = 16
CHECKPOINT_FILE = f"{SWEEP_DIR}/solution_map_embed_checkpoint.pt"
CHECKPOINT_EVERY = 10
FINAL_EMBS_FILE = f"{SWEEP_DIR}/solution_map_final_embs.pt"


def main(argv=None):
    with open(f"{SWEEP_DIR}/scored_manifest.json") as f:
        manifest = json.load(f)

    paths = [m["file"] for m in manifest]
    tags = [m["tag"] for m in manifest]

    def embed_with_progress(paths, model, label):
        from PIL import Image
        n = len(paths)
        done_embs = []
        start_i = 0
        if os.path.exists(CHECKPOINT_FILE):
            ckpt = torch.load(CHECKPOINT_FILE)
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
                                "embs": torch.cat(done_embs, dim=0)}, CHECKPOINT_FILE)
                    batches_since_checkpoint = 0

        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        return torch.cat(done_embs, dim=0)

    real_paths = sorted(os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    if os.path.exists(FINAL_EMBS_FILE):
        print(f"loading already-computed embeddings from {FINAL_EMBS_FILE} (skipping DINOv2 entirely)...", flush=True)
        saved = torch.load(FINAL_EMBS_FILE)
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
        real_embs = embed_with_progress(real_paths, model, "real")

        print(f"embedding {len(paths)} generated images...", flush=True)
        gen_embs = embed_with_progress(paths, model, "gen")

        # Persist the finished embeddings immediately, before UMAP runs: UMAP/sklearn version
        # mismatches are a real failure mode we've already hit once, and re-embedding 3392+31 images
        # takes ~28 minutes of wall-clock, so a UMAP-only crash should never force redoing this part.
        torch.save({"paths": paths, "real_paths": real_paths, "real_embs": real_embs, "gen_embs": gen_embs},
                   FINAL_EMBS_FILE)
        print(f"saved final embeddings to {FINAL_EMBS_FILE}", flush=True)

    # --- Idea 4: pairwise similarity WITH scores, for redundancy clustering ---
    print("computing pairwise similarity (with scores) for redundancy clustering...", flush=True)
    sim = gen_embs @ gen_embs.T
    sim.fill_diagonal_(-1.0)
    topk_vals, topk_idx = sim.topk(TOP_K, dim=1)
    neighbors_scored = {
        tags[i]: [[tags[j], round(topk_vals[i][k].item(), 4)] for k, j in enumerate(topk_idx[i].tolist())]
        for i in range(len(tags))
    }
    with open(f"{SWEEP_DIR}/similarity_scored.json", "w") as f:
        json.dump(neighbors_scored, f)
    print(f"wrote similarity_scored.json ({len(neighbors_scored)} entries, top-{TOP_K} each, with cosine scores)", flush=True)

    # --- Idea 6: nearest real image per generated image ---
    nn_real_matrix = gen_embs @ real_embs.T
    nearest_real_idx = nn_real_matrix.argmax(dim=1).tolist()
    nearest_real_sim = nn_real_matrix.max(dim=1).values.tolist()
    real_names = [os.path.basename(p) for p in real_paths]

    # --- Ideas 1/2: UMAP projection of the joint space (real + generated) ---
    print("fitting UMAP on the joint real+generated embedding space...", flush=True)
    import numpy as np
    import umap

    all_embs = torch.cat([real_embs, gen_embs], dim=0).numpy()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(all_embs)
    real_coords = coords[:len(real_paths)]
    gen_coords = coords[len(real_paths):]
    print("UMAP projection done.", flush=True)

    def generation_of(tag):
        m = re.match(r"gen(\d+)_", tag)
        return int(m.group(1)) if m else 0

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
            "thumb": f"thumbs/{m['tag']}.jpg" if os.path.exists(f"{SWEEP_DIR}/thumbs/{m['tag']}.jpg") else os.path.basename(m["file"]),
        })

    real_points = [
        {"name": real_names[i], "x": round(float(real_coords[i][0]), 4), "y": round(float(real_coords[i][1]), 4)}
        for i in range(len(real_paths))
    ]

    with open(f"{SWEEP_DIR}/solution_map_data.json", "w") as f:
        json.dump({"points": points, "real_points": real_points}, f)

    print(f"wrote solution_map_data.json ({len(points)} generated points, {len(real_points)} real points)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
