import os, json, sys
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-large-patch14"
SC = "/workspace/trent-with-smart-prompts"

print("loading model...", flush=True)
model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval()

def embed_images(paths, batch_size=16):
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=imgs, return_tensors="pt")
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats)
            print(f"  embedded {i+len(batch_paths)}/{len(paths)}", flush=True)
    return torch.cat(embs, dim=0)

real_dir = f"{SC}/corrected_dataset_extract"
real_paths = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
print(f"real training images: {len(real_paths)}")

gen_dir = f"{SC}/art_batch"
gen_paths = sorted([os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith(".png")])
print(f"generated images: {len(gen_paths)}")

real_embs = embed_images(real_paths)
centroid = real_embs.mean(dim=0)
centroid = centroid / centroid.norm()

gen_embs = embed_images(gen_paths)

sims = (gen_embs @ centroid).tolist()

results = []
for p, s in zip(gen_paths, sims):
    results.append({"file": p, "score": s})

results.sort(key=lambda r: -r["score"])

with open(f"{SC}/clip_scores.json", "w") as f:
    json.dump({
        "model": MODEL_ID,
        "real_images": len(real_paths),
        "centroid_intra_real_mean_sim": None,
        "results": results
    }, f, indent=1)

# also compute intra-real similarity spread as a reference baseline
intra = (real_embs @ centroid).tolist()
print(f"real-image self-similarity to centroid: mean={sum(intra)/len(intra):.4f} min={min(intra):.4f} max={max(intra):.4f}")
print(f"generated-image similarity to centroid: mean={sum(sims)/len(sims):.4f} min={min(sims):.4f} max={max(sims):.4f}")
print("Top 10:")
for r in results[:10]:
    print(f"  {r['score']:.4f}  {os.path.basename(r['file'])}")
print("Bottom 10:")
for r in results[-10:]:
    print(f"  {r['score']:.4f}  {os.path.basename(r['file'])}")
