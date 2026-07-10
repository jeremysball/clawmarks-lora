"""
Component 4 of the preference-classifier design: ranks every embedded image by the trained
model's predicted P(yes), highest first, so the model's judgment can be eyeballed against the
user's own taste before Stage 5b lets it steer anything live. Requires
search/preference_model.py to have already produced notes/uncanny_sweep/preference_model.joblib
(needs 50+ ratings — see search/preference_model.py's MIN_LABELS).

Run with: python3 -m clawmarks.build.preference_rank (or `clawmarks build preference-rank`)
"""
import json
import os

import joblib

from clawmarks.config import SWEEP_DIR
from clawmarks.search import embed_cache
from clawmarks.search.manifest_index import index_by_tag, item_summary
from clawmarks.search.preference_model import MODEL_FILE, predict_proba
from clawmarks.shared_ui import (
    INFOTIP_CSS, MOBILE_BASE_CSS, TOPNAV_CSS, info_btn, nav_bar_html, write_infotip_asset,
    write_lightbox_asset, write_scrollnav_asset,
)


def build_ranked_items(by_tag, tags, scores, sweep_dir, limit=500):
    ranked = sorted(
        ((t, s) for t, s in zip(tags, scores) if t in by_tag),
        key=lambda pair: -pair[1],
    )[:limit]
    items = []
    for tag, score in ranked:
        summary = item_summary(by_tag[tag], sweep_dir)
        summary["predicted_preference"] = round(float(score), 4)
        items.append(summary)
    return items


def main(argv=None):
    if not os.path.exists(MODEL_FILE):
        print(f"no trained model at {MODEL_FILE}; run `python -m "
              f"clawmarks.search.preference_model` first (needs 50+ ratings)", flush=True)
        return 1

    write_lightbox_asset(SWEEP_DIR)
    write_scrollnav_asset(SWEEP_DIR)
    write_infotip_asset(SWEEP_DIR)

    with open(f"{SWEEP_DIR}/scored_manifest.json") as f:
        manifest = json.load(f)
    by_tag = index_by_tag(manifest)

    tags, embeddings = embed_cache.load_cache(embed_cache.EMBEDDINGS_FILE)
    model = joblib.load(MODEL_FILE)
    scores = predict_proba(model, embeddings)
    items = build_ranked_items(by_tag, tags, scores, SWEEP_DIR)

    rank_tip = info_btn(
        "Sorted by the trained preference model's predicted probability that you'd rate this "
        "image 'yes,' highest first. This view exists to sanity-check the model before it's "
        "allowed to steer the live search: does the top of this list actually look like things "
        "you like?"
    )
    data_json = json.dumps(items)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS predicted preference</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee; --text-dim:#9a9aa4; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
h1 {{ font-size:18px; margin:0 0 4px; }}
p.sub {{ color:var(--text-dim); max-width:760px; font-size:13px; line-height:1.6; }}
#grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap:10px; margin-top:20px; }}
.cell {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
.cell img {{ width:100%; aspect-ratio:1; object-fit:cover; display:block; cursor:pointer; }}
.cell .meta {{ padding:6px 8px; font-size:11px; color:var(--text-dim); }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('preference_rank.html')}
<h1>Predicted preference{rank_tip}</h1>
<p class="sub">Top {len(items)} images by predicted P(yes), highest first.</p>
<div id="grid"></div>
<script>
const ITEMS = {data_json};
document.getElementById('grid').innerHTML = ITEMS.map(it => `
  <div class="cell">
    <img src="${{it.thumb}}" loading="lazy" data-tag="${{it.tag}}" onclick="Lightbox.open('${{it.tag}}')">
    <div class="meta">p=${{it.predicted_preference}} | f=${{it.faith}} n=${{it.novelty}}</div>
  </div>`).join('');
</script>
<script src="scrollnav.js"></script>
<script src="lightbox.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    with open(f"{SWEEP_DIR}/preference_rank.html", "w") as f:
        f.write(html)
    print(f"wrote {SWEEP_DIR}/preference_rank.html ({len(items)} ranked images)", flush=True)
    return 0


if __name__ == "__main__":
    main()
