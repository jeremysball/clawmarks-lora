import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


MODEL_ID = "facebook/dinov2-base"
ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "corrected_dataset_extract"
GEN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "art_batch"
N_PERMUTATIONS = 2000

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


def monte_carlo_p_value(observed, shuffled):
    """Return a finite-sample corrected one-sided Monte Carlo p-value."""
    shuffled = np.asarray(shuffled)
    if shuffled.ndim != 1:
        raise ValueError("shuffled statistics must be one-dimensional")
    if not len(shuffled):
        raise ValueError("at least one shuffled statistic is required")
    if not np.isfinite(observed):
        raise ValueError(f"observed statistic must be finite, got {observed!r}")
    if not np.all(np.isfinite(shuffled)):
        raise ValueError("shuffled statistics must all be finite")
    b = np.count_nonzero(shuffled >= observed)
    return (b + 1) / (len(shuffled) + 1)


def embed_images(model, paths, batch_size=16):
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            tensors = [preprocess(Image.open(p)) for p in batch_paths]
            pixel_values = torch.stack(tensors, dim=0)
            out = model(pixel_values=pixel_values)
            feats = out.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats)
            print(f"  embedded {i + len(batch_paths)}/{len(paths)}", flush=True)
    return torch.cat(embs, dim=0)


def mmd2_unbiased(K, idx_a, idx_b):
    a, b = len(idx_a), len(idx_b)
    if a < 2 or b < 2:
        raise ValueError(
            f"mmd2_unbiased needs at least 2 items per group (the unbiased "
            f"estimator excludes the diagonal), got a={a}, b={b}"
        )
    Kaa = K[idx_a][:, idx_a]
    Kbb = K[idx_b][:, idx_b]
    Kab = K[idx_a][:, idx_b]
    term_aa = (Kaa.sum() - Kaa.diag().sum()) / (a * (a - 1))
    term_bb = (Kbb.sum() - Kbb.diag().sum()) / (b * (b - 1))
    term_ab = Kab.sum() / (a * b)
    mmd2 = term_aa + term_bb - 2 * term_ab
    return mmd2.item(), term_aa.item(), term_bb.item(), term_ab.item()


def main():
    print("loading model...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    real_paths = sorted(
        p for p in REAL_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    gen_paths = sorted(
        p for p in GEN_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"real images: {len(real_paths)}, generated images: {len(gen_paths)} (from {GEN_DIR})")

    real_embs = embed_images(model, real_paths)
    gen_embs = embed_images(model, gen_paths)

    m, n = real_embs.shape[0], gen_embs.shape[0]
    all_embs = torch.cat([real_embs, gen_embs], dim=0)  # [m+n, 768], unit vectors

    # For unit vectors, squared Euclidean distance equals 2 - 2 * cosine similarity.
    cos_sim = all_embs @ all_embs.T
    sq_dist = (2 - 2 * cos_sim).clamp(min=0)

    # Median heuristic bandwidth: median of the off-diagonal squared distances.
    N = m + n
    off_diag_mask = ~torch.eye(N, dtype=torch.bool)
    median_sq_dist = sq_dist[off_diag_mask].median().item()
    sigma2 = median_sq_dist
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError(
            f"median-heuristic bandwidth is degenerate (sigma^2={sigma2!r}); this happens "
            "when most embeddings are near-identical, and the RBF kernel below would divide "
            "by zero or produce non-finite values"
        )
    print(f"\nbandwidth (median heuristic): sigma^2={sigma2:.4f}")

    K = torch.exp(-sq_dist / (2 * sigma2))

    real_idx = torch.arange(0, m)
    gen_idx = torch.arange(m, m + n)
    mmd2, real_real, gen_gen, real_gen = mmd2_unbiased(K, real_idx, gen_idx)

    print("\nkernel terms:")
    print(f"  real-real avg similarity (self-cohesion of the 31 real images): {real_real:.4f}")
    print(f"  gen-gen avg similarity  (self-cohesion of the {n} generated images): {gen_gen:.4f}")
    print(f"  real-gen avg similarity (cross term, how alike the two piles are): {real_gen:.4f}")
    print(f"\nMMD^2 = {mmd2:.4f}  (0 = indistinguishable distributions, larger = more different)")

    # Permutation test: reshuffle labels and recompute MMD^2 from the fixed K.
    # Assumes every image is an exchangeable unit; not yet checked for images that share a
    # prompt, seed, or checkpoint, which would correlate their embeddings and bias the p-value.
    rng = np.random.default_rng(0)
    perm_scores = np.empty(N_PERMUTATIONS)
    all_idx = np.arange(N)
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(all_idx)
        p_real, p_gen = torch.from_numpy(perm[:m]), torch.from_numpy(perm[m:])
        perm_scores[_] = mmd2_unbiased(K, p_real, p_gen)[0]

    p_value = monte_carlo_p_value(mmd2, perm_scores)
    print(f"\npermutation test ({N_PERMUTATIONS} shuffles): p-value={p_value:.4f}")
    print("(low p-value = the real/generated split is a genuinely more different pairing than random splits of the same pool)")

    # Split the 31 real images into two random halves as a noise-floor baseline.
    N_SELF_SPLITS = 200
    self_mmd2 = np.empty(N_SELF_SPLITS)
    real_idx_np = real_idx.numpy()
    for i in range(N_SELF_SPLITS):
        shuffled = rng.permutation(real_idx_np)
        half = m // 2
        a, b = torch.from_numpy(shuffled[:half]), torch.from_numpy(shuffled[half:])
        self_mmd2[i] = mmd2_unbiased(K, a, b)[0]

    print(f"\nreal-vs-real self-split baseline ({N_SELF_SPLITS} random halvings of the 31 real images):")
    print(f"  mean={self_mmd2.mean():.4f}  min={self_mmd2.min():.4f}  max={self_mmd2.max():.4f}")
    if self_mmd2.mean() > 0:
        print(f"  observed real-vs-generated MMD^2 ({mmd2:.4f}) is {mmd2 / self_mmd2.mean():.1f}x the self-split mean")

    with (ROOT / "notes" / "mmd_result.json").open("w") as f:
        json.dump(
            {
                "gen_dir": GEN_DIR,
                "n_real": m,
                "n_gen": n,
                "sigma2": sigma2,
                "real_real": real_real,
                "gen_gen": gen_gen,
                "real_gen": real_gen,
                "mmd2": mmd2,
                "p_value": float(p_value),
                "n_permutations": N_PERMUTATIONS,
                "self_split_mean": float(self_mmd2.mean()),
                "self_split_min": float(self_mmd2.min()),
                "self_split_max": float(self_mmd2.max()),
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
