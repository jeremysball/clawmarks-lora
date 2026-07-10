"""
Merges round 2's 280 images (notes/uncanny_sweep2/) into round 1's dataset
(notes/uncanny_sweep/), so every exploration tool (scan.html, map.html, coverage.html,
archive.html, redundancy.html, novelty_decay.html) shows both rounds as one population instead
of round 2 being stranded in its own directory with only its own gallery.html.

Round 2 tags reuse round 1's "genN_..." naming scheme, so they're prefixed "r2_" here to
guarantee uniqueness (collision by coincidence is astronomically unlikely given the random seed
suffix, but not worth relying on). File paths are NOT copied; scored_manifest.json just points
at each image's original absolute path, since every downstream script already reads m["file"]
directly.

Reuses build_solution_map.py's cached embeddings (solution_map_final_embs.pt) for round
1's 3392 images and the 31 real images, so this only needs to run DINOv2 on the 280 NEW images,
not re-embed everything. After merging, the cache is overwritten with the full 3672-image set,
so a future round 3 merge only re-embeds round 3's images, same pattern.

Run: python3 -m clawmarks.build.merge_round2
Then rebuild every tool against the merged data (this script does that at the end automatically).
"""
import json, os, sys, shutil
import torch
from transformers import AutoModel

from clawmarks.config import SWEEP_DIR, SWEEP2_DIR
from clawmarks.search.score_manifest import preprocess, MODEL_ID

EMBS_FILE = f"{SWEEP_DIR}/solution_map_final_embs.pt"
MANIFEST_FILE = f"{SWEEP_DIR}/scored_manifest.json"


def main(argv=None):
    with open(MANIFEST_FILE) as f:
        manifest1 = json.load(f)

    with open(f"{SWEEP2_DIR}/scored_manifest.json") as f:
        manifest2_raw = json.load(f)

    if any(m.get("round") == 2 for m in manifest1):
        print("round 2 already merged into scored_manifest.json, nothing to do", flush=True)
        sys.exit(0)

    # Guard against double-running on a manifest that's already been merged once and re-saved.
    already_merged_tags = {m["tag"] for m in manifest1}
    manifest2 = []
    for m in manifest2_raw:
        m = dict(m)
        m["tag"] = "r2_" + m["tag"]
        m["round"] = 2
        manifest2.append(m)
    for m in manifest1:
        m.setdefault("round", 1)

    backup_path = f"{SWEEP_DIR}/scored_manifest_round1_only.json"
    if not os.path.exists(backup_path):
        shutil.copy(MANIFEST_FILE, backup_path)
        print(f"backed up round-1-only manifest to {backup_path}", flush=True)

    merged_manifest = manifest1 + manifest2
    with open(MANIFEST_FILE, "w") as f:
        json.dump(merged_manifest, f, indent=1)
    print(f"merged manifest: {len(manifest1)} round-1 + {len(manifest2)} round-2 = {len(merged_manifest)} total", flush=True)

    # --- Embed only the new round-2 images, reusing the cached round-1 + real embeddings ---
    saved = torch.load(EMBS_FILE)
    cached_paths, real_paths = saved["paths"], saved["real_paths"]
    real_embs, gen_embs = saved["real_embs"], saved["gen_embs"]

    new_paths = [m["file"] for m in manifest2]
    print(f"embedding {len(new_paths)} new round-2 images (round 1's {len(cached_paths)} are cached)...", flush=True)

    print("loading DINOv2...", flush=True)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    from PIL import Image

    def embed(paths):
        embs = []
        with torch.no_grad():
            for i in range(0, len(paths), 16):
                batch = paths[i:i + 16]
                tensors = [preprocess(Image.open(p)) for p in batch]
                pixel_values = torch.stack(tensors, dim=0)
                out = model(pixel_values=pixel_values)
                feats = out.pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
                embs.append(feats)
                print(f"  embedded {min(i + 16, len(paths))}/{len(paths)}", flush=True)
        return torch.cat(embs, dim=0)

    new_embs = embed(new_paths)
    merged_gen_embs = torch.cat([gen_embs, new_embs], dim=0)
    merged_paths = cached_paths + new_paths

    torch.save({"paths": merged_paths, "real_paths": real_paths, "real_embs": real_embs, "gen_embs": merged_gen_embs},
               EMBS_FILE)
    print(f"updated embedding cache: {merged_gen_embs.shape[0]} generated images total", flush=True)

    # Similarity and solution-map data are no longer written to disk here: since PR #7,
    # curation_server.py computes both live (similarity_index.compute_data /
    # solution_map.compute_data) from scored_manifest.json and solution_map_final_embs.pt,
    # neither of which any code still reads these two files for.
    tags = [m["tag"] for m in merged_manifest]
    assert [m["file"] for m in merged_manifest] == merged_paths, "manifest/embedding order mismatch after merge"

    # UMAP is no longer refit and written here either: solution_map.compute_data() refits it
    # live off solution_map_final_embs.pt (updated above) the next time map.html or
    # redundancy.html is requested, and curation_server.py's LiveCache keeps that result cached
    # across requests until the manifest or embeddings file changes again.

    print("DONE merging round 2", flush=True)


if __name__ == "__main__":
    main()
