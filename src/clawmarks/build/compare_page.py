"""
Generates compare.html: a head-to-head comparison page. Shows two images side by side; tapping
or clicking one picks it as the winner, feeding search/preference_pairwise_model.py. The old
yes/no rating interface is gone. This page bakes in no per-image data at build time: it fetches
GET /api/compare/next itself and POSTs to
/api/compare, both served by curation_server.py, so the page never goes stale between rebuilds.

Served live at /compare.html by curation_server.py.
"""
from clawmarks.shared_ui import (
    CONTROL_CSS,
    INFOTIP_CSS,
    MOBILE_BASE_CSS,
    SULFUR_CSS,
    SULFUR_FONT_CSS,
    TOPNAV_CSS,
    info_btn,
    nav_bar_html,
    scoped_href,
)


def render_html(active_expedition=None, active_leg=None, running=None, focus=None):
    compare_tip = info_btn(
        "Trains the preference model by comparison: pick whichever of the two images you "
        "prefer, as many times as you can stand. Early comparisons are sampled to spread across "
        "the faithfulness/novelty grid; once 50+ comparisons exist, the model itself starts "
        "picking which pairs are most useful to compare next."
    )
    accuracy_tip = info_btn(
        "This is a Bradley-Terry-style classifier: logistic regression on the *difference* "
        "between two images' DINOv2 embeddings, trained on your head-to-head picks. It doesn't "
        "rate one image alone, it learns a direction in embedding space where 'further this "
        "way' means 'more preferred', so it can score any image, even one it never compared. "
        "The percentage is cross-validated accuracy: how often it names the actual winner on "
        "comparisons held out of training (leave-one-out below 50 comparisons, 5-fold beyond "
        "that), not accuracy on data it already saw. 50% is a coin flip, 100% is perfect. Tap "
        "\"show the work\" below for the exact numbers, including a permutation-test p-value "
        "that checks whether this accuracy could plausibly be noise."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS compare</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
body {{ display:flex; flex-direction:column; align-items:center; }}
h1 {{ font-size:22px; margin:24px 0 4px; align-self:flex-start; letter-spacing:0.02em;
  text-transform:uppercase; }}
p.sub {{ color:var(--text-soft); max-width:640px; font-size:13px; line-height:1.6;
  align-self:flex-start; }}
#stage {{ margin-top:20px; width:100%; max-width:1100px; display:flex; flex-direction:column; align-items:center; }}
#pair {{ display:flex; gap:18px; width:100%; justify-content:center; align-items:stretch;
  flex-wrap:wrap; }}
.pane {{ position:relative; flex:1 1 420px; max-width:520px; cursor:pointer;
  background:var(--paper); box-shadow:5px 5px 0 var(--ink);
  border:2px solid var(--ink); transition: box-shadow .12s ease; }}
.pane:hover {{ box-shadow:6px 6px 0 var(--ink); }}
.pane:focus-visible, .zoom-icon:focus-visible {{ outline:3px solid var(--sulfur); outline-offset:3px; }}
.pane.submitting {{ pointer-events:none; opacity:0.6; }}
.pane img {{ display:block; width:100%; max-height:70vh; object-fit:contain;
  background:var(--paper-deep); user-select:none; -webkit-user-drag:none; }}
.zoom-icon {{ position:absolute; top:8px; right:8px; width:30px; height:30px;
  background:var(--ink); border:1px solid var(--ink); color:var(--paper);
  font-size:15px; display:flex; align-items:center; justify-content:center; cursor:zoom-in;
  z-index:2; box-shadow:2px 2px 0 var(--ink); }}
.zoom-icon:hover {{ background:var(--sulfur); color:var(--ink); box-shadow:3px 3px 0 var(--ink); }}
.or-axis {{ display:flex; align-items:center; justify-content:center; flex:0 0 auto;
  font:800 28px/1 var(--font-display); color:var(--ink); letter-spacing:0.08em;
  padding:0 4px; user-select:none; }}
.cap {{ color:var(--text-soft); font-size:12.5px; margin-top:8px; text-align:center; padding:0 4px 2px;
  line-height:1.5; }}
#count {{ color:var(--text-soft); font-size:12px; margin-top:14px; }}
#done {{ color:var(--text-soft); font-size:14px; margin-top:40px; text-align:center; }}
#progress {{ width:100%; max-width:640px; align-self:flex-start; margin-top:14px; }}
#prog-label {{ font-size:13.5px; color:var(--ink); font-weight:600; margin-bottom:6px; }}
#prog-track {{ height:10px; background:var(--paper-deep); border:1px solid var(--ink);
  overflow:hidden; }}
#prog-fill {{ height:100%; width:0%;
  background:var(--ink); transition:width .5s cubic-bezier(.2,.7,.2,1); }}
#prog-fill.bump {{ background:var(--sulfur); }}
#prog-sub {{ font-size:11.5px; color:var(--text-soft); margin-top:6px; font-family:var(--font-mono); }}
#prog-work {{ margin-top:8px; border-top:1px solid var(--rule); padding-top:8px; }}
#prog-work summary {{ font-size:11.5px; color:var(--text-soft); cursor:pointer; user-select:none; }}
#prog-work summary:hover {{ color:var(--ink); }}
table.work-table {{ font-size:12px; border-collapse:collapse; margin-top:8px; font-family:var(--font-mono); }}
table.work-table td {{ padding:2px 10px 2px 0; color:var(--text-soft); }}
table.work-table td:first-child {{ color:var(--ink); }}
.work-note {{ color:var(--text-soft); }}
@media (max-width: 700px) {{
  #pair {{ flex-direction:column; align-items:center; }}
  .pane {{ flex:1 1 auto; width:100%; max-width:none; }}
  .pane img {{ max-height:48vh; }}
  .or-axis {{ padding:6px 0; }}
  #progress {{ align-self:stretch; max-width:none; }}
}}
#zoom-overlay {{ position:fixed; inset:0; background:rgba(8,8,10,0.94); backdrop-filter:blur(6px);
  display:none; align-items:center; justify-content:center; z-index:1000; cursor:grab; overflow:hidden; }}
#zoom-overlay.open {{ display:flex; }}
#zoom-overlay img {{ max-width:none; max-height:none; user-select:none; -webkit-user-drag:none; }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('compare.html', active_expedition=active_expedition, active_leg=active_leg, running=running, focus=focus)}
<h1>Compare{compare_tip}</h1>
<p class="sub">Tap or click the image you prefer (or press &larr;/&rarr;). Tap the magnifier in
a corner to inspect that image at full resolution; tap again to close.</p>
<p class="sub"><a href="{scoped_href('/preference_status.html', active_expedition, active_leg, focus)}">View preference status</a> or <a href="{scoped_href('/preference_rank.html', active_expedition, active_leg, focus)}">review the model's ranking</a>.</p>

<div id="progress">
  <div id="prog-label"><span id="prog-label-text">&nbsp;</span>{accuracy_tip}</div>
  <div id="prog-track"><div id="prog-fill"></div></div>
  <div id="prog-sub"></div>
  <details id="prog-work" style="display:none;">
    <summary>show the work</summary>
    <table class="work-table" id="work-table"></table>
  </details>
</div>

<div id="stage">
  <div id="pair">
    <div class="pane" id="pane1" data-side="1" role="button" tabindex="0" aria-label="Choose image A">
      <img id="img1" style="display:none;">
      <div class="zoom-icon" id="zoom1" role="button" tabindex="0" aria-label="Inspect image A at full resolution">&#128269;</div>
      <div class="cap" id="cap1"></div>
    </div>
    <div class="or-axis" aria-hidden="true">OR</div>
    <div class="pane" id="pane2" data-side="2" role="button" tabindex="0" aria-label="Choose image B">
      <img id="img2" style="display:none;">
      <div class="zoom-icon" id="zoom2" role="button" tabindex="0" aria-label="Inspect image B at full resolution">&#128269;</div>
      <div class="cap" id="cap2"></div>
    </div>
  </div>
  <div id="done" style="display:none;">Nothing left to compare right now. The pool doesn't have enough images left to form a new pair.</div>
</div>
<div id="count"></div>

<div id="zoom-overlay">
  <img id="zoom-img">
</div>

<script>
let current = null;
let choiceSubmitted = false;
let comparedThisSession = 0;
let totalCount = 0;
let rawCount = 0;
let statusStale = false;
let lastAccuracy = null;
let modelMeta = null;
const SCOPE_QUERY = new URLSearchParams(window.location.search);
function scopedApi(path) {{
  const url = new URL(path, window.location.origin);
  ['expedition', 'leg'].forEach(key => {{
    if (SCOPE_QUERY.has(key)) url.searchParams.set(key, SCOPE_QUERY.get(key));
  }});
  return url.toString();
}}

const MIN_COMPARISONS = 50;
const RETRAIN_EVERY = 10;

function caption(img) {{
  // textContent (set by the caller) keeps model-controlled prompt_name from being parsed as HTML.
  return `${{img.prompt_name}} · faith ${{img.faith}} · novelty ${{img.novelty}}`;
}}

function revealSamplingDetails() {{
  document.getElementById('cap1').textContent = caption(current.img1);
  document.getElementById('cap2').textContent = caption(current.img2);
}}

function bumpBar() {{
  const fill = document.getElementById('prog-fill');
  fill.classList.add('bump');
  setTimeout(() => fill.classList.remove('bump'), 550);
}}

function renderProgress() {{
  const labelText = document.getElementById('prog-label-text');
  const fill = document.getElementById('prog-fill');
  const sub = document.getElementById('prog-sub');
  const work = document.getElementById('prog-work');
  work.style.display = 'none';
  if (totalCount < MIN_COMPARISONS) {{
    const left = MIN_COMPARISONS - totalCount;
    labelText.textContent = `Model unlocks in ${{left}} vote${{left === 1 ? '' : 's'}}`;
    fill.style.width = (totalCount / MIN_COMPARISONS * 100) + '%';
    sub.textContent = rawCount !== totalCount
      ? `${{totalCount}} / ${{MIN_COMPARISONS}} usable comparisons (${{rawCount}} submitted)`
      : `${{totalCount}} / ${{MIN_COMPARISONS}} comparisons`;
  }} else if (lastAccuracy == null) {{
    labelText.textContent = 'Model unlocked. Training on your picks…';
    fill.style.width = '0%';
    sub.textContent = `${{totalCount}} comparisons`;
  }} else {{
    const pct = Math.round(lastAccuracy * 100);
    labelText.textContent = `Model reads your taste: ${{pct}}%`;
    // Map cross-validated accuracy (0.5 = coin flip, 1.0 = perfect) onto the bar.
    fill.style.width = Math.max(0, Math.min(100, (lastAccuracy - 0.5) / 0.5 * 100)) + '%';
    const toRefresh = RETRAIN_EVERY - (totalCount % RETRAIN_EVERY);
    const n = toRefresh === RETRAIN_EVERY ? 0 : toRefresh;
    const refresh = n === 0 ? 'just refreshed' : `refreshes in ${{n}} vote${{n === 1 ? '' : 's'}}`;
    sub.textContent = `${{refresh}} · coin-flip 50% → 100%`;
    if (modelMeta) renderWork(work);
  }}
  if (statusStale) {{
    sub.textContent += " · couldn't refresh, counts may be stale";
  }}
}}

function renderWork(work) {{
  const m = modelMeta;
  const table = document.getElementById('work-table');
  const rows = [
    ['comparisons used', m.n_comparisons],
    ['cross-validated accuracy', (m.cv_accuracy * 100).toFixed(1) + '%'],
  ];
  if (typeof m.baseline_accuracy === 'number') {{
    rows.push(['majority-class baseline', (m.baseline_accuracy * 100).toFixed(1) + '%']);
  }}
  if (typeof m.p_value === 'number') {{
    const interp = m.p_value < 0.05 ? 'unlikely to be chance' : 'not distinguishable from chance';
    rows.push(['permutation p-value', `${{m.p_value.toFixed(4)}} (${{interp}}, ${{m.n_permutations}} shuffles)`]);
  }}
  rows.push(['trained', m.trained_at]);
  table.innerHTML = rows.map(([k, v]) => `<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join('')
    + '<tr><td colspan="2" class="work-note">full breakdown + retrain: '
    + '<a href="{scoped_href('/preference_status.html', active_expedition, active_leg, focus)}" style="color:inherit;">preference_status.html</a></td></tr>';
  work.style.display = 'block';
}}

function fetchStatus(after) {{
  const prevTotalCount = totalCount;
  fetch(scopedApi('/api/preference_status')).then(r => r.ok ? r.json() : null).then(d => {{
    if (d) {{
      statusStale = false;
      if (typeof d.n_usable === 'number') totalCount = d.n_usable;
      else if (typeof d.n_comparisons === 'number') totalCount = d.n_comparisons;
      rawCount = typeof d.n_comparisons === 'number' ? d.n_comparisons : totalCount;
      if (d.model_meta && typeof d.model_meta.cv_accuracy === 'number') {{
        lastAccuracy = d.model_meta.cv_accuracy;
        modelMeta = d.model_meta;
      }}
    }} else {{
      statusStale = true;
    }}
    renderProgress();
    if (after) after(prevTotalCount);
  }}).catch(() => {{ statusStale = true; renderProgress(); if (after) after(prevTotalCount); }});
}}

function loadNext() {{
  fetch(scopedApi('/api/compare/next')).then(r => {{
    if (!r.ok) throw new Error('Could not load the next comparison');
    return r.json();
  }}).then(d => {{
    if (d.done) {{
      current = null;
      document.getElementById('pair').style.display = 'none';
      document.getElementById('done').style.display = 'block';
      return;
    }}
    current = d;
    choiceSubmitted = false;
    submitting = false;
    document.getElementById('pane1').classList.remove('submitting');
    document.getElementById('pane2').classList.remove('submitting');
    document.getElementById('pair').style.display = 'flex';
    document.getElementById('done').style.display = 'none';
    const img1 = document.getElementById('img1');
    const img2 = document.getElementById('img2');
    img1.src = d.img1.file; img1.style.display = 'block';
    img2.src = d.img2.file; img2.style.display = 'block';
    document.getElementById('cap1').textContent = 'Image A';
    document.getElementById('cap2').textContent = 'Image B';
  }}).catch(() => {{
    document.getElementById('done').textContent =
      "Couldn't reach the server. Check your connection and try again.";
    document.getElementById('done').style.display = 'block';
  }});
}}

let submitting = false;

function choose(side) {{
  if (!current || submitting) return;
  if (!current || choiceSubmitted || zoomOpen) return;
  choiceSubmitted = true;
  submitting = true;
  const winner = side === 1 ? current.img1.tag : current.img2.tag;
  const loser = side === 1 ? current.img2.tag : current.img1.tag;
  fetch('/api/compare', {{method:'POST', headers:{{'Content-Type':'application/json'}},
     body: JSON.stringify({{winner, loser, expedition: SCOPE_QUERY.get('expedition'), leg: SCOPE_QUERY.get('leg')}})}})
    .then(r => {{
      if (!r.ok) throw new Error('Could not save the comparison');
      return r.json();
    }})
    .then((res) => {{
      comparedThisSession++;
      document.getElementById('count').textContent = `${{comparedThisSession}} compared this session`;
      // res.count is the raw store size, not the usable (deduplicated) pair count the retrain
      // gate actually uses, so re-derive totalCount from /api/preference_status rather than
      // trusting it directly.
      fetchStatus((prevTotalCount) => {{
        // Compare buckets rather than a single modulo snapshot: n_usable can jump by 0 or more
        // than 1 per vote (deduplication), so a bare `totalCount % RETRAIN_EVERY === 0` check
        // can step over the exact boundary and miss a crossing entirely.
        const prevBucket = prevTotalCount >= MIN_COMPARISONS ? Math.floor(prevTotalCount / RETRAIN_EVERY) : -1;
        const bucket = totalCount >= MIN_COMPARISONS ? Math.floor(totalCount / RETRAIN_EVERY) : -1;
        const crossedRetrain = totalCount >= MIN_COMPARISONS && bucket !== prevBucket;
        if (crossedRetrain || totalCount < MIN_COMPARISONS) bumpBar();
      }});
      revealSamplingDetails();
      setTimeout(loadNext, 1000);
    }}).catch(() => {{
      choiceSubmitted = false;
      submitting = false;
      document.getElementById('pane1').classList.remove('submitting');
      document.getElementById('pane2').classList.remove('submitting');
      document.getElementById('done').textContent =
        "Couldn't reach the server. Check your connection and try again.";
      document.getElementById('done').style.display = 'block';
    }});
  document.getElementById('pane1').classList.add('submitting');
  document.getElementById('pane2').classList.add('submitting');
}}

document.getElementById('pane1').addEventListener('click', () => choose(1));
document.getElementById('pane2').addEventListener('click', () => choose(2));
document.querySelectorAll('.pane').forEach(pane => pane.addEventListener('keydown', e => {{
  if (e.target !== pane || (e.key !== 'Enter' && e.key !== ' ')) return;
  e.preventDefault();
  choose(Number(pane.dataset.side));
}}));

document.addEventListener('keydown', e => {{
  if (e.key === 'Escape' && zoomOpen) {{ closeZoom(); return; }}
  if (zoomOpen) return;
  if (e.key === 'ArrowLeft') {{ e.preventDefault(); choose(1); }}
  if (e.key === 'ArrowRight') {{ e.preventDefault(); choose(2); }}
}});

// --- zoom overlay: opens on a zoom-icon tap, closes on any tap, drag to pan while open ---

let zoomOpen = false;
let zoomControl = null;
let panX = 0, panY = 0, dragging = false, dragMoved = false, dragStartX = 0, dragStartY = 0;

function clampOffset(offset, wrapSize, imgSize) {{
  if (imgSize <= wrapSize) return (wrapSize - imgSize) / 2;
  return Math.min(0, Math.max(wrapSize - imgSize, offset));
}}

function openZoom(side, e) {{
  e.stopPropagation();
  if (!current) return;
  const src = side === 1 ? current.img1.file : current.img2.file;
  const overlay = document.getElementById('zoom-overlay');
  const zimg = document.getElementById('zoom-img');
  zimg.src = src;
  panX = 0; panY = 0;
  zimg.style.transform = 'translate(0px, 0px)';
  overlay.classList.add('open');
  zoomOpen = true;
  zoomControl = e.currentTarget;
}}

function closeZoom() {{
  document.getElementById('zoom-overlay').classList.remove('open');
  zoomOpen = false;
  if (zoomControl) zoomControl.focus();
}}

document.getElementById('zoom1').addEventListener('click', e => openZoom(1, e));
document.getElementById('zoom2').addEventListener('click', e => openZoom(2, e));
document.querySelectorAll('.zoom-icon').forEach(icon => icon.addEventListener('keydown', e => {{
  if (e.key !== 'Enter' && e.key !== ' ') return;
  e.preventDefault();
  openZoom(icon.id === 'zoom1' ? 1 : 2, e);
}}));

const overlayEl = document.getElementById('zoom-overlay');
overlayEl.addEventListener('mousedown', e => {{
  dragging = true; dragMoved = false;
  dragStartX = e.clientX - panX; dragStartY = e.clientY - panY;
}});
document.addEventListener('mousemove', e => {{
  if (!dragging) return;
  dragMoved = true;
  const zimg = document.getElementById('zoom-img');
  panX = clampOffset(e.clientX - dragStartX, overlayEl.clientWidth, zimg.naturalWidth);
  panY = clampOffset(e.clientY - dragStartY, overlayEl.clientHeight, zimg.naturalHeight);
  zimg.style.transform = `translate(${{panX}}px, ${{panY}}px)`;
}});
document.addEventListener('mouseup', () => {{
  if (dragging && !dragMoved) closeZoom();
  dragging = false;
}});
overlayEl.addEventListener('touchstart', e => {{
  // Cancel the touch default so the browser doesn't emit the compatibility
  // mousedown/mouseup/click after touchend. Without this, tapping the overlay to
  // close zoom fires a synthetic click on the pane beneath it, recording an
  // unintended comparison. Registered {{passive: false}} so preventDefault applies.
  e.preventDefault();
  const touch = e.touches[0];
  dragging = true; dragMoved = false;
  dragStartX = touch.clientX - panX; dragStartY = touch.clientY - panY;
}}, {{passive: false}});
overlayEl.addEventListener('touchmove', e => {{
  if (!dragging) return;
  e.preventDefault();
  dragMoved = true;
  const touch = e.touches[0];
  const zimg = document.getElementById('zoom-img');
  panX = clampOffset(touch.clientX - dragStartX, overlayEl.clientWidth, zimg.naturalWidth);
  panY = clampOffset(touch.clientY - dragStartY, overlayEl.clientHeight, zimg.naturalHeight);
  zimg.style.transform = `translate(${{panX}}px, ${{panY}}px)`;
}}, {{passive: false}});
overlayEl.addEventListener('touchend', () => {{
  if (dragging && !dragMoved) closeZoom();
  dragging = false;
}});

fetchStatus();
loadNext();
</script>
<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
<script src="/shared-ui.js"></script>
</body></html>"""

    return html
