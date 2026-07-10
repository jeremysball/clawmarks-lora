"""
Scores every image from notes/uncanny_sweep/manifest.json on two axes (DINOv2 centroid
similarity = faithfulness, 1 - nearest-neighbor similarity to any single real image =
novelty), bins them into a 2D descriptor grid (lab_notebook.md Section 3b's "liminal band"
idea), and writes an HTML gallery/atlas to notes/uncanny_sweep/gallery.html.

This keeps every image per bin rather than picking one "elite" per bin (no automated
coherence/quality scorer exists to pick with); final curation is left to a human looking at
the gallery, per this project's standing rule that a metric is a filter, not a verdict.

Run after notes/run_uncanny_sweep.py finishes (or on manifest_partial.json if run early):
    python3 -m clawmarks.build.uncanny_gallery [manifest_path]
"""
import os, sys, json, base64
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel

from clawmarks.config import ROOT, SWEEP_DIR

MODEL_ID = "facebook/dinov2-base"
REAL_DIR = f"{ROOT}/corrected_dataset_extract"
N_BINS = 4

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


def thumb_data_uri(path, size=192):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _default_manifest():
    full = f"{SWEEP_DIR}/manifest.json"
    partial = f"{SWEEP_DIR}/manifest_partial.json"
    if os.path.exists(full):
        return full
    if os.path.exists(partial):
        print(f"NOTE: {full} not found yet, building from partial results ({partial}). "
              f"Some planned jobs may still be missing; that's fine, this doesn't wait for "
              f"100% completion.", flush=True)
        return partial
    raise FileNotFoundError("neither manifest.json nor manifest_partial.json exists yet")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    manifest_path = argv[0] if argv else _default_manifest()
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

    paths = [m["file"] for m in manifest if os.path.exists(m["file"])]
    manifest = [m for m in manifest if os.path.exists(m["file"])]
    embs = embed_images(paths, model=model)

    centroid_sim = (embs @ centroid).tolist()
    nn_matrix = embs @ real_embs.T
    nn_sim = nn_matrix.max(dim=1).values.tolist()

    for m, cs, ns in zip(manifest, centroid_sim, nn_sim):
        m["centroid_sim"] = cs
        m["novelty"] = 1 - ns
        m["prompt_type"] = "style" if m["prompt_name"].startswith("style_") else "conflict"

    faith_vals = sorted(m["centroid_sim"] for m in manifest)
    novelty_vals = sorted(m["novelty"] for m in manifest)

    def bin_edges(vals, n):
        return [vals[int(i * len(vals) / n)] for i in range(1, n)]

    faith_edges = bin_edges(faith_vals, N_BINS)
    novelty_edges = bin_edges(novelty_vals, N_BINS)

    def bin_of(val, edges):
        for i, e in enumerate(edges):
            if val <= e:
                return i
        return len(edges)

    grid = {}
    for m in manifest:
        fb = bin_of(m["centroid_sim"], faith_edges)
        nb = bin_of(m["novelty"], novelty_edges)
        grid.setdefault((fb, nb), []).append(m)

    with open(f"{SWEEP_DIR}/scored_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)

    liminal_lo, liminal_hi = faith_edges[0], faith_edges[-1]
    liminal_band = [m for m in manifest if liminal_lo <= m["centroid_sim"] <= liminal_hi]
    # Select by novelty (this is the whole point: find what's far from any real image while
    # still in-band), but *display* sorted by faithfulness descending, per Fable's flag that
    # pure novelty-first ordering front-loads broken/incoherent images ahead of plausible ones.
    liminal_band_top = sorted(liminal_band, key=lambda m: -m["novelty"])[:32]
    liminal_band_top.sort(key=lambda m: -m["centroid_sim"])

    by_type = {}
    for m in manifest:
        by_type.setdefault(m["prompt_type"], []).append(m["centroid_sim"])
    type_summary = {t: (sum(v) / len(v), len(v)) for t, v in by_type.items()}
    print(f"faithfulness by prompt type: {type_summary}", flush=True)

    print("building gallery...", flush=True)
    build_html(manifest, grid, faith_edges, novelty_edges, liminal_band_top,
               (real_ref_mean, real_ref_min, real_ref_max), type_summary)
    print(f"DONE: {SWEEP_DIR}/gallery.html ({len(manifest)} images, {len(grid)} occupied bins)", flush=True)


TYPE_COLOR = {"style": "#5ec98a", "conflict": "#e0a25e"}


def cell_html(items, faith_edges, novelty_edges, fb, nb):
    lo_f = faith_edges[fb - 1] if fb > 0 else "-inf"
    hi_f = faith_edges[fb] if fb < len(faith_edges) else "+inf"
    lo_n = novelty_edges[nb - 1] if nb > 0 else "-inf"
    hi_n = novelty_edges[nb] if nb < len(novelty_edges) else "+inf"
    label = f"faith [{lo_f}, {hi_f}) x novelty [{lo_n}, {hi_n})" if items else ""
    if not items:
        return '<div class="cell empty"></div>'
    thumbs = "".join(
        f'<img style="border:2px solid {TYPE_COLOR[m["prompt_type"]]}" '
        f'src="{thumb_data_uri(m["file"])}" title="{m["tag"]} | type={m["prompt_type"]} prompt={m["prompt_name"]} strength={m["strength"]} cfg={m["cfg"]} steps={m["steps"]} sampler={m["sampler"]} faith={m["centroid_sim"]:.3f} novelty={m["novelty"]:.3f}">'
        for m in sorted(items, key=lambda m: -m["novelty"])[:12]
    )
    return f'<div class="cell" data-count="{len(items)}"><div class="cell-label">{label}<br>n={len(items)}</div><div class="cell-thumbs">{thumbs}</div></div>'


def build_html(manifest, grid, faith_edges, novelty_edges, highlight, real_ref, type_summary):
    real_ref_mean, real_ref_min, real_ref_max = real_ref
    rows = []
    for fb in range(N_BINS):
        cols = []
        for nb in range(N_BINS):
            cols.append(cell_html(grid.get((fb, nb), []), faith_edges, novelty_edges, fb, nb))
        rows.append(f'<div class="row">{"".join(cols)}</div>')

    highlight_html = "".join(
        f'<figure><img style="border:2px solid {TYPE_COLOR[m["prompt_type"]]}" src="{thumb_data_uri(m["file"])}">'
        f'<figcaption>{m["prompt_name"]} ({m["prompt_type"]}) | s={m["strength"]} cfg={m["cfg"]}<br>faith={m["centroid_sim"]:.3f} novelty={m["novelty"]:.3f}</figcaption></figure>'
        for m in highlight
    )

    type_rows = "".join(
        f"<tr><td style='color:{TYPE_COLOR.get(t, '#ccc')}'>{t}</td><td>{mean:.3f}</td><td>{n}</td></tr>"
        for t, (mean, n) in sorted(type_summary.items())
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS uncanny frontier atlas</title>
<style>
body {{ background:#111; color:#eee; font-family: -apple-system, sans-serif; margin:0; padding:24px; }}
h1 {{ font-weight:600; }}
p.sub {{ color:#999; max-width:800px; }}
.grid {{ display:flex; flex-direction:column; gap:6px; margin-top:24px; }}
.row {{ display:flex; gap:6px; }}
.cell {{ flex:1; background:#1c1c1c; border:1px solid #333; min-height:160px; padding:6px; }}
.cell.empty {{ background:#161616; }}
.cell-label {{ font-size:10px; color:#777; margin-bottom:4px; }}
.cell-thumbs img {{ width:56px; height:56px; object-fit:cover; margin:1px; border-radius:3px; }}
.axis-label {{ font-size:12px; color:#888; margin:8px 0; }}
.highlight {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:16px; }}
.highlight figure {{ margin:0; width:160px; }}
.highlight img {{ width:160px; height:160px; object-fit:cover; border-radius:6px; }}
.highlight figcaption {{ font-size:11px; color:#aaa; margin-top:4px; }}
table.summary {{ border-collapse:collapse; margin:8px 0; font-size:12px; }}
table.summary td, table.summary th {{ padding:3px 10px; border:1px solid #333; text-align:left; }}
.legend {{ font-size:12px; color:#aaa; margin:6px 0; }}
.legend span {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:middle; }}
.refband {{ font-size:12px; color:#aaa; background:#1a1a1a; border-left:3px solid #666; padding:6px 10px; margin:8px 0; max-width:800px; }}
</style></head><body>
<h1>CLAWMARKS uncanny frontier atlas</h1>
<p class="sub">Every generated image plotted on two axes: faithfulness (DINOv2 cosine similarity
to the real-art centroid, x-axis, left=less faithful) and novelty (1 minus similarity to the
single nearest real training image, y-axis, top=most novel). Each grid cell shows up to 12
images that landed there, most-novel first, hover for prompt/strength/CFG metadata. This is
descriptor binning, not full MAP-Elites: every image per bin is kept rather than one automated
"elite," since no reliable automated coherence score exists yet, per this project's standing
rule that a metric filters, it doesn't verdict. Final curation is a human call.</p>

<div class="refband"><b>Reference anchor</b>: the 31 real training images score
{real_ref_min:.3f}-{real_ref_max:.3f} (mean {real_ref_mean:.3f}) against each other's own
centroid (leave-one-out). Treat that range as "definitely in-style" on an absolute scale, since
this batch's own bin edges are relative to a deliberately extreme sweep (LoRA strength up to
1.8x, CFG up to 12) and can drift the "middle" bins toward garbage if enough of the batch is
fried. A cell's faithfulness range well below this reference band is more likely off-style than
liminal.</div>

<div class="legend"><span style="background:{TYPE_COLOR['style']}"></span>style-typical prompt
&nbsp;&nbsp;<span style="background:{TYPE_COLOR['conflict']}"></span>conflicted prompt (content
the LoRA never saw in training)</div>
<table class="summary"><tr><th>prompt type</th><th>mean faithfulness</th><th>n</th></tr>{type_rows}</table>
<p class="sub">Conflicted prompts are expected to skew toward lower faithfulness by construction,
not necessarily because the style broke: DINOv2's centroid similarity can't distinguish "lost
the style" from "kept the style, changed the subject" (flagged in review). Read faithfulness
comparisons within a prompt type, not across types.</p>

<h2>Liminal band highlights</h2>
<p class="sub">Images in the middle faithfulness band (still reads as CLAWMARKS, not off-style
noise) with the highest novelty (furthest from any single real training image), the region this
whole search exists to find. Selected by novelty, but shown most-faithful-first so the more
plausible candidates surface before likely-broken ones.</p>
<div class="highlight">{highlight_html}</div>

<h2>Full descriptor grid</h2>
<p class="axis-label">Rows: faithfulness bins, low (top) to high (bottom). Columns: novelty
bins, low (left) to high (right).</p>
<div class="grid">{"".join(rows)}</div>
</body></html>"""

    with open(f"{SWEEP_DIR}/gallery.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
