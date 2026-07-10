import base64, os

from clawmarks.config import PROBE_DIR

OUT = f"{PROBE_DIR}_report.html"

REAL_MEAN, REAL_MIN, REAL_MAX = 0.6127, 0.2219, 0.8416

SUBJECTS = [
    {
        "key": "human_face",
        "title": "Human face",
        "prompt": "close-up human face, dark-rimmed eyes glowing pale blue, pale skin with visible "
                  "brush texture, hand pressed beside cheek, dense dark-blue vertical brush-dash "
                  "background, thick acrylic dry-brush texture, raw outsider-art painting",
        "note": "Both seeds reverted to the model's dominant trained motif, the swirled concentric "
                "eye pattern, instead of rendering an actual human face. The subject cue got "
                "absorbed rather than obeyed.",
        "seeds": [
            {"seed": 11, "file": "human_face_seed11.jpg", "score": 0.3788},
            {"seed": 22, "file": "human_face_seed22.jpg", "score": 0.3612},
        ],
    },
    {
        "key": "cyborg",
        "title": "Cyborg",
        "prompt": "close-up cyborg face, half exposed circuitry and wiring, dark-rimmed human eye "
                  "glowing pale blue beside a mechanical lens, clawed metal hand pressed beside "
                  "cheek, dense dark-blue vertical brush-dash background, thick acrylic dry-brush "
                  "texture, raw outsider-art painting",
        "note": "The strongest result by eye, seed 22's wiring-skull reads as genuinely on-style "
                "mechanical fusion, yet it scores mid-pack and seed 11 scores the lowest of all "
                "eight probes.",
        "seeds": [
            {"seed": 11, "file": "cyborg_seed11.jpg", "score": 0.1949},
            {"seed": 22, "file": "cyborg_seed22.jpg", "score": 0.3460},
        ],
    },
    {
        "key": "body_horror",
        "title": "Body horror",
        "prompt": "close-up face mid-transformation, skin splitting to reveal clawed fingers "
                  "pushing through the cheek, dark-rimmed eyes glowing pale blue, dense dark-blue "
                  "vertical brush-dash background, thick acrylic dry-brush texture, raw "
                  "outsider-art painting",
        "note": "Seed 11 kept the eye motif with a texture shift; seed 22 abstracted into a "
                "radiating claw-burst pattern, visually striking but the least literal reading "
                "of the prompt.",
        "seeds": [
            {"seed": 11, "file": "body_horror_seed11.jpg", "score": 0.3198},
            {"seed": 22, "file": "body_horror_seed22.jpg", "score": 0.2373},
        ],
    },
    {
        "key": "liminal",
        "title": "Liminal space",
        "prompt": "figure standing alone in an empty fluorescent-lit hallway, dark-rimmed eyes "
                  "glowing pale blue, clawed hand pressed against the wall, dense dark-blue "
                  "vertical brush-dash background replaced by flat institutional tile, thick "
                  "acrylic dry-brush texture, raw outsider-art painting",
        "note": "The only prompt to produce full humanoid figures in a distinct setting rather "
                "than a face-crop. Seed 11 is the most legible \"counter-art\" result of the batch.",
        "seeds": [
            {"seed": 11, "file": "liminal_seed11.jpg", "score": 0.3690},
            {"seed": 22, "file": "liminal_seed22.jpg", "score": 0.2186},
        ],
    },
]


def data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


def score_bar(score):
    # scale 0..1 across a 0.0-0.9 visible range for legibility
    span = 0.9
    pct = max(0, min(100, score / span * 100))
    mean_pct = REAL_MEAN / span * 100
    min_pct = REAL_MIN / span * 100
    max_pct = REAL_MAX / span * 100
    below = score < REAL_MIN
    return f"""
    <div class="scorebar" title="Real-image range: {REAL_MIN:.2f}-{REAL_MAX:.2f}, mean {REAL_MEAN:.2f}">
      <div class="scorebar-track">
        <div class="scorebar-real-range" style="left:{min_pct:.1f}%; width:{max_pct-min_pct:.1f}%"></div>
        <div class="scorebar-mean" style="left:{mean_pct:.1f}%"></div>
        <div class="scorebar-fill" style="width:{pct:.1f}%"></div>
        <div class="scorebar-marker {'below' if below else ''}" style="left:{pct:.1f}%"></div>
      </div>
    </div>
    """


def main(argv=None):
    cards = []
    all_scores = []
    for subj in SUBJECTS:
        seed_blocks = []
        for s in subj["seeds"]:
            uri = data_uri(os.path.join(PROBE_DIR, s["file"]))
            all_scores.append((f'{subj["title"]} · seed {s["seed"]}', s["score"]))
            below = s["score"] < REAL_MIN
            flag = '<span class="flag">below real floor</span>' if below else ''
            seed_blocks.append(f"""
            <figure class="probe">
              <img src="{uri}" alt="{subj['title']} seed {s['seed']}" loading="lazy">
              <figcaption>
                <div class="seed-row">
                  <span class="seed-label">seed {s['seed']}</span>
                  <span class="score-num">{s['score']:.4f}</span>
                  {flag}
                </div>
                {score_bar(s['score'])}
              </figcaption>
            </figure>
            """)
        cards.append(f"""
        <section class="subject" id="{subj['key']}">
          <div class="subject-head">
            <h2>{subj['title']}</h2>
            <p class="prompt">&ldquo;{subj['prompt']}&rdquo;</p>
            <p class="note">{subj['note']}</p>
          </div>
          <div class="probe-grid">
            {''.join(seed_blocks)}
          </div>
        </section>
        """)

    all_scores.sort(key=lambda x: -x[1])
    rows = "\n".join(
        f'<tr><td>{name}</td><td class="num">{score:.4f}</td>'
        f'<td>{"below real floor (0.22)" if score < REAL_MIN else "within real range"}</td></tr>'
        for name, score in all_scores
    )

    html = f"""<title>CLAWMARKS: Uncanny Subject Probe</title>
<style>
:root {{
  --bg: #12141c;
  --panel: #1a1e2b;
  --panel-2: #212639;
  --cream: #ede6d8;
  --ink: #e9e5da;
  --accent: #6fb8e0;
  --accent-dim: #4a7f9c;
  --muted: #8b93a8;
  --rule: #2b3145;
  --danger: #d98a6b;
}}
:root[data-theme="light"] {{
  --bg: #f4f0e6;
  --panel: #fffdf8;
  --panel-2: #ece5d4;
  --cream: #1c1e26;
  --ink: #22242e;
  --accent: #2f7a9e;
  --accent-dim: #5a9bb8;
  --muted: #6b7280;
  --rule: #ddd4bf;
  --danger: #b0532f;
}}
@media (prefers-color-scheme: light) {{
  :root:not([data-theme="dark"]) {{
    --bg: #f4f0e6;
    --panel: #fffdf8;
    --panel-2: #ece5d4;
    --cream: #1c1e26;
    --ink: #22242e;
    --accent: #2f7a9e;
    --accent-dim: #5a9bb8;
    --muted: #6b7280;
    --rule: #ddd4bf;
    --danger: #b0532f;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--ink);
  font-family: Georgia, 'Iowan Old Style', 'Palatino Linotype', ui-serif, serif;
  line-height: 1.6;
  margin: 0;
  padding: 0 1.25rem 5rem;
}}
.wrap {{ max-width: 900px; margin: 0 auto; }}
header {{
  padding: 3.5rem 0 2rem;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 2.5rem;
}}
.eyebrow {{
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
}}
h1 {{
  font-size: 2.1rem;
  margin: 0.5rem 0 0.75rem;
  text-wrap: balance;
}}
.dek {{
  color: var(--muted);
  font-size: 1.02rem;
  max-width: 62ch;
  margin: 0 0 1.25rem;
}}
.method {{
  background: var(--panel);
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 1.1rem 1.4rem;
  font-size: 0.92rem;
  color: var(--muted);
}}
.method strong {{ color: var(--ink); }}
.method .num {{
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}}
.subject {{
  margin-bottom: 3.5rem;
}}
.subject-head h2 {{
  font-size: 1.45rem;
  margin: 0 0 0.5rem;
  border-left: 3px solid var(--accent);
  padding-left: 0.7rem;
}}
.prompt {{
  font-style: italic;
  color: var(--muted);
  font-size: 0.95rem;
  margin: 0 0 0.6rem 0.9rem;
  max-width: 68ch;
}}
.note {{
  font-size: 0.92rem;
  margin: 0 0 1.3rem 0.9rem;
  max-width: 68ch;
}}
.probe-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.25rem;
}}
.probe {{
  margin: 0;
  background: var(--panel);
  border: 1px solid var(--rule);
  border-radius: 8px;
  overflow: hidden;
}}
.probe img {{
  width: 100%;
  height: auto;
  display: block;
}}
.probe figcaption {{
  padding: 0.85rem 1rem 1rem;
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.82rem;
}}
.seed-row {{
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
  margin-bottom: 0.5rem;
}}
.seed-label {{
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-size: 0.72rem;
}}
.score-num {{
  color: var(--accent);
  font-size: 1rem;
  font-variant-numeric: tabular-nums;
}}
.flag {{
  color: var(--danger);
  font-size: 0.7rem;
  letter-spacing: 0.03em;
  margin-left: auto;
}}
.scorebar-track {{
  position: relative;
  height: 8px;
  background: var(--panel-2);
  border-radius: 4px;
  overflow: visible;
}}
.scorebar-real-range {{
  position: absolute;
  top: 0;
  height: 100%;
  background: var(--accent-dim);
  opacity: 0.28;
  border-radius: 4px;
}}
.scorebar-mean {{
  position: absolute;
  top: -2px;
  width: 2px;
  height: 12px;
  background: var(--accent);
}}
.scorebar-fill {{
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background: var(--accent-dim);
  border-radius: 4px;
  opacity: 0.5;
}}
.scorebar-marker {{
  position: absolute;
  top: -3px;
  width: 10px;
  height: 14px;
  border-radius: 3px;
  background: var(--accent);
  transform: translateX(-50%);
}}
.scorebar-marker.below {{ background: var(--danger); }}
table {{
  width: 100%;
  border-collapse: collapse;
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 0.85rem;
}}
th, td {{
  text-align: left;
  padding: 0.55rem 0.7rem;
  border-bottom: 1px solid var(--rule);
}}
th {{
  color: var(--muted);
  font-weight: normal;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-size: 0.72rem;
}}
td.num {{ font-variant-numeric: tabular-nums; color: var(--accent); }}
.summary {{
  background: var(--panel);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 1.5rem 1.6rem;
  margin-top: 1rem;
  overflow-x: auto;
}}
.finding {{
  max-width: 68ch;
  font-size: 0.98rem;
  margin-top: 1.4rem;
}}
footer {{
  margin-top: 3rem;
  color: var(--muted);
  font-size: 0.8rem;
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
}}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">CLAWMARKS LoRA &middot; reach probe</div>
  <h1>Does the style extend past animal portraits?</h1>
  <p class="dek">
    Eight generations from the epoch-4 checkpoint, testing four subjects the 31-image
    training set never depicted: a human face, a cyborg, body horror, and a figure in a
    liminal space. Two seeds each, scored against the real-image DINOv2 style centroid.
  </p>
  <div class="method">
    <strong>Reading the score bar:</strong> each bar spans <span class="num">0.00</span>
    to <span class="num">0.90</span> similarity to the real 31-image centroid. The shaded
    band marks the real training images' own self-similarity range
    (<span class="num">{REAL_MIN:.2f}</span>&ndash;<span class="num">{REAL_MAX:.2f}</span>,
    mean <span class="num">{REAL_MEAN:.2f}</span>). A marker landing left of that band
    means the generation scored below the real dataset's own worst-case outlier.
  </div>
</header>

{''.join(cards)}

<section class="summary">
  <h2 style="margin-top:0; font-size:1.2rem;">All eight, ranked</h2>
  <table>
    <thead><tr><th>Generation</th><th>DINOv2 score</th><th>Relative to real range</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  <p class="finding">
    <strong>The scores don't track visual impression.</strong> The two generations that read
    most successfully by eye, the cyborg wiring-skull and the liminal corridor figure, land in
    the middle of this ranking, not the top. The generation that scores highest reverted to the
    model's safest, most conservative motif (the trained eye pattern) rather than depicting an
    actual human face. Six of eight probes score at or below the real dataset's own weakest
    self-similarity outlier. This either means DINOv2 style-similarity stops discriminating
    reliably at a subject jump this large, or the checkpoint genuinely hasn't generalized the
    style's rules this far yet, or both. Human judgment, not this score, should decide what
    goes in front of the artist.
  </p>
</section>

<footer>
  epoch-4 checkpoint &middot; illustrious_v0.1 base &middot; DINOv2 (facebook/dinov2-base) centroid scoring
  &middot; see notes/lab_notebook.md and continuation_prompt.md for full context
</footer>
</div>
"""

    with open(OUT, "w") as f:
        f.write(html)
    print(f"wrote {OUT}, {os.path.getsize(OUT)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
