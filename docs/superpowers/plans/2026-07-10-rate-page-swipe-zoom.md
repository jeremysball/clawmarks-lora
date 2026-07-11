# rate.html swipe-vote + double-click zoom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `rate.html`'s tap-to-vote buttons with a Tinder-style swipe gesture, and add a
double-click/double-tap zoom-to-full-resolution with drag-to-pan, per
`docs/superpowers/specs/2026-07-10-rate-page-swipe-zoom-design.md`.

**Architecture:** All changes are confined to `src/clawmarks/build/rate_page.py`'s
`render_html()` function (a single Python f-string producing the page's HTML/CSS/JS). No server
API changes: the page keeps using the existing `GET /api/rate/next` and `POST /api/rate`
endpoints. No changes to `shared_ui.py` or the shared `Lightbox`. Built in three layers: (1) DOM
scaffold with buttons removed, (2) double-click zoom + mouse pan, (3) touch swipe-to-vote + touch
pan (these last two share one set of touch handlers since they branch on the same `zoomed` state
and the same drag classification, so they can't be usefully split further).

**Tech Stack:** Python f-string HTML/CSS/JS generation (existing pattern in this file), pytest
for string-membership tests on the generated HTML (existing pattern in `tests/test_rate_page.py`).

## Global Constraints

- Scope is `src/clawmarks/build/rate_page.py` only. No other tool page, and not the shared
  `Lightbox` in `shared_ui.py`, changes.
- No new server routes or API endpoints. Reuse `GET /api/rate/next` and `POST /api/rate` exactly
  as they exist today.
- **Superseded by Task 5 below** (post-review revision, see the spec's 2026-07-10 revision
  note): ~~Swipe-to-vote is touch-only. Desktop keeps voting via arrow keys / `y` / `n`; mouse
  drag never votes.~~ Mouse drag now votes too, same as touch. ~~Zoom triggers on `dblclick`
  (mouse double-click or touch double-tap), not single click/tap.~~ Zoom now triggers on a
  single tap/click (no movement past the deadzone), not double. Panning while zoomed still works
  with both touch drag and mouse drag.
- Swipe commit threshold: 25% of the image's rendered width.
- Drag classification (swipe vs pan vs ignore) happens once per touch, after a ~10px deadzone,
  and is decided by direction (horizontal-dominant) when not zoomed, or always "pan" when zoomed.
- No pinch-to-zoom or scroll-wheel zoom. Zoom is exactly two states: fit-to-screen and native
  pixel size.
- No em dashes in any prose this plan or its tasks produce (project writing-style rule).

---

## Task 1: Remove tap buttons, scaffold the swipe/zoom DOM and CSS

**Files:**
- Modify: `src/clawmarks/build/rate_page.py` (whole `render_html()` function body)
- Test: `tests/test_rate_page.py`

**Interfaces:**
- Consumes: `clawmarks.shared_ui.nav_bar_html`, `TOPNAV_CSS`, `MOBILE_BASE_CSS`, `INFOTIP_CSS`,
  `info_btn` (all already imported in this file; unchanged).
- Produces: HTML elements `#imgwrap` (wraps `#img`, will host zoom/pan transforms in Task 2/3)
  and `#swipe-overlay` (will host the yes/no swipe feedback in Task 3). The `#buttons` element
  and its `.yes`/`.no` button styling are removed entirely; later tasks must not reintroduce them.
  `rate(label)` and `loadNext()` keep their existing signatures (no arguments beyond `label` on
  `rate`) so Task 2/3 can call them unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rate_page.py`:

```python
def test_render_html_has_no_tap_buttons():
    html = rate_page.render_html()
    assert "<button" not in html
    assert 'id="buttons"' not in html


def test_render_html_has_imgwrap_and_overlay():
    html = rate_page.render_html()
    assert 'id="imgwrap"' in html
    assert 'id="swipe-overlay"' in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_rate_page.py -v`
Expected: the two new tests FAIL (`<button` and `id="buttons"` are present today; `id="imgwrap"`
and `id="swipe-overlay"` don't exist yet). The original `test_render_html_includes_rate_api_calls`
still PASSES.

- [ ] **Step 3: Replace `render_html()` with the button-free scaffold**

Replace the full contents of `src/clawmarks/build/rate_page.py` with:

```python
"""
Generates rate.html: a full-screen, swipe-driven yes/no rating page. Unlike every other
build/*.py generator, this page bakes in no per-image data at build time. It fetches
GET /api/rate/next itself and POSTs to /api/rate, both served by curation_server.py, so the page
never goes stale between rebuilds. Rebuilding only matters if this file itself changes.

Served live at /rate.html by curation_server.py.
"""
from clawmarks.shared_ui import nav_bar_html, TOPNAV_CSS, MOBILE_BASE_CSS, INFOTIP_CSS, info_btn


def render_html():
    rate_tip = info_btn(
        "Rating trains the preference classifier: yes/no on as many images as you can stand "
        "to look at. Yes-rated images immediately take over the search's exploit pool (the same "
        "role picking used to play); once enough ratings exist, a model trained on them takes "
        "over ranking automatically."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS rate</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee;
  --text-dim:#9a9aa4; --yes:#5ec98a; --no:#e0605e; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px;
  display:flex; flex-direction:column; align-items:center; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
h1 {{ font-size:18px; margin:0 0 4px; align-self:flex-start; }}
p.sub {{ color:var(--text-dim); max-width:640px; font-size:13px; line-height:1.6; align-self:flex-start; }}
#stage {{ margin-top:20px; width:100%; max-width:640px; display:flex; flex-direction:column; align-items:center; }}
#imgwrap {{ position:relative; max-width:100%; max-height:78vh; overflow:hidden; touch-action:none;
  display:flex; align-items:center; justify-content:center; }}
#img {{ max-width:100%; max-height:78vh; border-radius:10px; box-shadow:0 20px 60px rgba(0,0,0,0.6);
  user-select:none; -webkit-user-drag:none; }}
#swipe-overlay {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  font-size:48px; font-weight:800; letter-spacing:0.08em; opacity:0; pointer-events:none; border-radius:10px; }}
#swipe-overlay.yes {{ color:var(--yes); background:rgba(94,201,138,0.12); }}
#swipe-overlay.no {{ color:var(--no); background:rgba(224,96,94,0.12); }}
#meta {{ color:var(--text-dim); font-size:12.5px; margin-top:10px; text-align:center; }}
#count {{ color:var(--text-dim); font-size:12px; margin-top:14px; }}
#done {{ color:var(--text-dim); font-size:14px; margin-top:40px; text-align:center; }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('rate.html')}
<h1>Rate{rate_tip}</h1>
<p class="sub">Swipe left for no, right for yes (or &larr;/&rarr;, n/y on a keyboard). Double-click
or double-tap an image to zoom to full resolution; drag to look around while zoomed.</p>

<div id="stage">
  <div id="imgwrap">
    <img id="img" style="display:none;">
    <div id="swipe-overlay"></div>
  </div>
  <div id="meta"></div>
  <div id="done" style="display:none;">Nothing left to rate right now &mdash; every image in the pool has been rated or favorited.</div>
</div>
<div id="count"></div>

<script>
let current = null;
let ratedThisSession = 0;

function loadNext() {{
  fetch('/api/rate/next').then(r => r.json()).then(d => {{
    if (d.done) {{
      current = null;
      document.getElementById('img').style.display = 'none';
      document.getElementById('done').style.display = 'block';
      return;
    }}
    current = d;
    const img = document.getElementById('img');
    img.style.transform = 'translateX(0px)';
    img.src = d.file;
    img.style.display = 'block';
    document.getElementById('swipe-overlay').style.opacity = 0;
    document.getElementById('meta').textContent =
      `${{d.prompt_name}} | faith=${{d.faith}} novelty=${{d.novelty}}`;
  }});
}}

function rate(label) {{
  if (!current) return;
  const tag = current.tag;
  fetch('/api/rate', {{method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{tag, label}})}})
    .then(r => r.json())
    .then(() => {{
      ratedThisSession++;
      document.getElementById('count').textContent = `${{ratedThisSession}} rated this session`;
      loadNext();
    }});
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft' || e.key === 'n' || e.key === 'N') rate('no');
  if (e.key === 'ArrowRight' || e.key === 'y' || e.key === 'Y') rate('yes');
}});

loadNext();
</script>
<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
</body></html>"""

    return html
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_rate_page.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-with-smart-prompts
git add src/clawmarks/build/rate_page.py tests/test_rate_page.py
git commit -m "$(cat <<'EOF'
feat(clawmarks): remove rate.html tap buttons, scaffold swipe/zoom DOM

EOF
)"
```

---

## Task 2: Double-click zoom to full resolution, with mouse drag-to-pan

**Files:**
- Modify: `src/clawmarks/build/rate_page.py`
- Test: `tests/test_rate_page.py`

**Interfaces:**
- Consumes: `#imgwrap`, `#img`, `loadNext()`, `current` from Task 1. Extends `loadNext()` to call
  the new `resetZoom()` so each new image starts unzoomed.
- Produces: `zoomed` (boolean module-level JS variable, true while zoomed), `panOffsetX` /
  `panOffsetY` (current pan translation in pixels), `clampOffset(offset, wrapSize, imgSize)`
  (pure helper, used again by Task 3's touch panning), `resetZoom()`, `zoomIn(clientX, clientY)`.
  Task 3 must reuse `clampOffset`, `zoomed`, `panOffsetX`/`panOffsetY` rather than redefining them.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rate_page.py`:

```python
def test_render_html_has_zoom_machinery():
    html = rate_page.render_html()
    assert "function zoomIn(" in html
    assert "function resetZoom(" in html
    assert "function clampOffset(" in html
    assert "dblclick" in html
    assert "classList.add('zoomed')" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_rate_page.py::test_render_html_has_zoom_machinery -v`
Expected: FAIL (none of these functions/attributes exist yet).

- [ ] **Step 3: Add zoom CSS state and JS**

In `src/clawmarks/build/rate_page.py`, in the `<style>` block, add a zoomed variant right after
the existing `#imgwrap { ... }` rule:

```css
#imgwrap.zoomed {{ height:78vh; width:100%; cursor:grab; }}
#imgwrap.zoomed #img {{ max-width:none; max-height:none; border-radius:0; box-shadow:none; }}
```

In the `<script>` block, change `let current = null;` / `let ratedThisSession = 0;` to also
declare the zoom state, and update `loadNext()` to reset zoom on every new image:

```js
let current = null;
let ratedThisSession = 0;
let zoomed = false;
let panOffsetX = 0, panOffsetY = 0;

function loadNext() {{
  fetch('/api/rate/next').then(r => r.json()).then(d => {{
    if (d.done) {{
      current = null;
      document.getElementById('img').style.display = 'none';
      document.getElementById('done').style.display = 'block';
      return;
    }}
    current = d;
    resetZoom();
    const img = document.getElementById('img');
    img.style.transition = '';
    img.style.transform = 'translateX(0px)';
    img.src = d.file;
    img.style.display = 'block';
    document.getElementById('swipe-overlay').style.opacity = 0;
    document.getElementById('meta').textContent =
      `${{d.prompt_name}} | faith=${{d.faith}} novelty=${{d.novelty}}`;
  }});
}}
```

Then, after the existing `document.addEventListener('keydown', ...)` block and before the final
`loadNext();` call, insert:

```js
// --- zoom ---

function clampOffset(offset, wrapSize, imgSize) {{
  if (imgSize <= wrapSize) return (wrapSize - imgSize) / 2;
  return Math.min(0, Math.max(wrapSize - imgSize, offset));
}}

function resetZoom() {{
  zoomed = false;
  panOffsetX = 0;
  panOffsetY = 0;
  document.getElementById('imgwrap').classList.remove('zoomed');
  document.getElementById('img').style.transform = 'translate(0px, 0px)';
}}

function zoomIn(clientX, clientY) {{
  const img = document.getElementById('img');
  const wrap = document.getElementById('imgwrap');
  const rect = img.getBoundingClientRect();
  const fracX = (clientX - rect.left) / rect.width;
  const fracY = (clientY - rect.top) / rect.height;
  wrap.classList.add('zoomed');
  zoomed = true;
  const targetX = fracX * img.naturalWidth;
  const targetY = fracY * img.naturalHeight;
  panOffsetX = clampOffset(wrap.clientWidth / 2 - targetX, wrap.clientWidth, img.naturalWidth);
  panOffsetY = clampOffset(wrap.clientHeight / 2 - targetY, wrap.clientHeight, img.naturalHeight);
  img.style.transform = `translate(${{panOffsetX}}px, ${{panOffsetY}}px)`;
}}

const imgwrapEl = document.getElementById('imgwrap');

imgwrapEl.addEventListener('dblclick', e => {{
  if (!current) return;
  if (zoomed) {{
    resetZoom();
  }} else {{
    zoomIn(e.clientX, e.clientY);
  }}
}});

// mouse panning while zoomed (desktop)
let mouseDown = false, mouseStartX = 0, mouseStartY = 0;
imgwrapEl.addEventListener('mousedown', e => {{
  if (!zoomed) return;
  mouseDown = true;
  mouseStartX = e.clientX;
  mouseStartY = e.clientY;
}});
document.addEventListener('mousemove', e => {{
  if (!mouseDown || !zoomed) return;
  const img = document.getElementById('img');
  const wrap = document.getElementById('imgwrap');
  const newX = clampOffset(panOffsetX + (e.clientX - mouseStartX), wrap.clientWidth, img.naturalWidth);
  const newY = clampOffset(panOffsetY + (e.clientY - mouseStartY), wrap.clientHeight, img.naturalHeight);
  img.style.transform = `translate(${{newX}}px, ${{newY}}px)`;
}});
document.addEventListener('mouseup', e => {{
  if (!mouseDown) return;
  mouseDown = false;
  if (!zoomed) return;
  const img = document.getElementById('img');
  const wrap = document.getElementById('imgwrap');
  panOffsetX = clampOffset(panOffsetX + (e.clientX - mouseStartX), wrap.clientWidth, img.naturalWidth);
  panOffsetY = clampOffset(panOffsetY + (e.clientY - mouseStartY), wrap.clientHeight, img.naturalHeight);
}});
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_rate_page.py -v`
Expected: all tests PASS (4 from Task 1 + 1 new = 5, since the original API test still counts).

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-with-smart-prompts
git add src/clawmarks/build/rate_page.py tests/test_rate_page.py
git commit -m "$(cat <<'EOF'
feat(clawmarks): add double-click zoom and mouse pan to rate.html

EOF
)"
```

---

## Task 3: Touch swipe-to-vote and touch pan-while-zoomed

**Files:**
- Modify: `src/clawmarks/build/rate_page.py`
- Test: `tests/test_rate_page.py`

**Interfaces:**
- Consumes: `#imgwrap`, `#img`, `#swipe-overlay`, `zoomed`, `panOffsetX`/`panOffsetY`,
  `clampOffset()`, `rate(label)` from Tasks 1 and 2.
- Produces: `DEADZONE_PX`, `SWIPE_THRESHOLD_FRACTION` (constants), `updateSwipeVisual(dx)`,
  `finishSwipe(dx)`. Nothing later in this plan depends on these; this is the final task.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rate_page.py`:

```python
def test_render_html_has_touch_swipe_machinery():
    html = rate_page.render_html()
    assert "addEventListener('touchstart'" in html
    assert "addEventListener('touchmove'" in html
    assert "addEventListener('touchend'" in html
    assert "SWIPE_THRESHOLD_FRACTION" in html
    assert "function finishSwipe(" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_rate_page.py::test_render_html_has_touch_swipe_machinery -v`
Expected: FAIL (no touch listeners exist yet).

- [ ] **Step 3: Add touch swipe/pan handling**

In `src/clawmarks/build/rate_page.py`, after the zoom block added in Task 2 (after the
`mouseup` listener, before the final `loadNext();` call), insert:

```js
// --- touch: swipe-to-vote when not zoomed, pan when zoomed ---

const DEADZONE_PX = 10;
const SWIPE_THRESHOLD_FRACTION = 0.25;
let touchActive = false, touchClassified = null; // null | 'swipe' | 'pan' | 'ignore'
let touchStartX = 0, touchStartY = 0, touchDX = 0, touchDY = 0;

function updateSwipeVisual(dx) {{
  const img = document.getElementById('img');
  const overlay = document.getElementById('swipe-overlay');
  img.style.transition = '';
  img.style.transform = `translateX(${{dx}}px)`;
  const width = img.getBoundingClientRect().width || 1;
  const frac = Math.min(1, Math.abs(dx) / (width * SWIPE_THRESHOLD_FRACTION));
  overlay.className = dx > 0 ? 'yes' : 'no';
  overlay.style.opacity = frac;
}}

function finishSwipe(dx) {{
  const img = document.getElementById('img');
  const overlay = document.getElementById('swipe-overlay');
  const width = img.getBoundingClientRect().width || 1;
  const threshold = width * SWIPE_THRESHOLD_FRACTION;
  if (Math.abs(dx) >= threshold) {{
    const label = dx > 0 ? 'yes' : 'no';
    img.style.transition = 'transform 0.2s ease';
    img.style.transform = `translateX(${{dx > 0 ? width * 1.2 : -width * 1.2}}px)`;
    overlay.style.opacity = 0;
    setTimeout(() => rate(label), 180);
  }} else {{
    img.style.transition = 'transform 0.2s ease';
    img.style.transform = 'translateX(0px)';
    overlay.style.opacity = 0;
  }}
}}

imgwrapEl.addEventListener('touchstart', e => {{
  if (!current || e.touches.length !== 1) return;
  touchActive = true;
  touchClassified = null;
  touchStartX = e.touches[0].clientX;
  touchStartY = e.touches[0].clientY;
  touchDX = 0;
  touchDY = 0;
}});

imgwrapEl.addEventListener('touchmove', e => {{
  if (!touchActive) return;
  const t = e.touches[0];
  touchDX = t.clientX - touchStartX;
  touchDY = t.clientY - touchStartY;

  if (touchClassified === null) {{
    if (Math.abs(touchDX) < DEADZONE_PX && Math.abs(touchDY) < DEADZONE_PX) return;
    if (zoomed) {{
      touchClassified = 'pan';
    }} else if (Math.abs(touchDX) > Math.abs(touchDY)) {{
      touchClassified = 'swipe';
    }} else {{
      touchClassified = 'ignore';
    }}
  }}

  if (touchClassified === 'ignore') return;
  e.preventDefault();

  if (touchClassified === 'swipe') {{
    updateSwipeVisual(touchDX);
  }} else if (touchClassified === 'pan') {{
    const img = document.getElementById('img');
    const wrap = document.getElementById('imgwrap');
    const newX = clampOffset(panOffsetX + touchDX, wrap.clientWidth, img.naturalWidth);
    const newY = clampOffset(panOffsetY + touchDY, wrap.clientHeight, img.naturalHeight);
    img.style.transform = `translate(${{newX}}px, ${{newY}}px)`;
  }}
}}, {{passive: false}});

imgwrapEl.addEventListener('touchend', () => {{
  if (!touchActive) return;
  touchActive = false;
  if (touchClassified === 'swipe') {{
    finishSwipe(touchDX);
  }} else if (touchClassified === 'pan') {{
    const img = document.getElementById('img');
    const wrap = document.getElementById('imgwrap');
    panOffsetX = clampOffset(panOffsetX + touchDX, wrap.clientWidth, img.naturalWidth);
    panOffsetY = clampOffset(panOffsetY + touchDY, wrap.clientHeight, img.naturalHeight);
  }}
  touchClassified = null;
}});
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_rate_page.py -v`
Expected: all tests PASS (7 total).

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-with-smart-prompts
git add src/clawmarks/build/rate_page.py tests/test_rate_page.py
git commit -m "$(cat <<'EOF'
feat(clawmarks): add touch swipe-to-vote and pan-while-zoomed to rate.html

EOF
)"
```

---

## Task 4: Manual verification in a live browser

String-membership tests confirm the generated HTML contains the right JS, but nothing in Tasks
1 to 3 exercises real drag/tap/double-tap physics, so this task drives the page directly.

**Files:** none (verification only).

- [ ] **Step 1: Start the server against the seed-run batch**

```bash
cd /workspace/trent-with-smart-prompts
CLAWMARKS_SWEEP_DIR=/workspace/trent-with-smart-prompts/notes/uncanny_seedrun1 PYTHONPATH=src uv run python -m clawmarks.cli serve > /tmp/rate_verify_serve.log 2>&1 &
disown
sleep 3 && cat /tmp/rate_verify_serve.log
```

Expected: a line like `serving .../notes/uncanny_seedrun1 + ratings API on 0.0.0.0:<port>`, no
traceback.

- [ ] **Step 2: Confirm the page loads**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:<port>/rate.html
```

Expected: `200`.

- [ ] **Step 3: Desktop checks in a real browser**

Open `http://<tailscale-ip>:<port>/rate.html`.
- Confirm no yes/no buttons are visible.
- Press the right arrow key: confirm a rating POSTs (network tab shows `/api/rate` with
  `{{"label": "yes"}}`) and the next image loads.
- Double-click the image: confirm it zooms to full resolution, centered near the click point.
- Click-drag while zoomed: confirm the image pans and stays within its own bounds (can't drag it
  fully off-screen).
- Double-click again while zoomed: confirm it returns to fit-to-screen.

- [ ] **Step 4: Touch checks via browser devtools touch emulation**

In Chrome DevTools, toggle device toolbar (touch emulation) on `rate.html`.
- Drag the image right past roughly a quarter of its width: confirm a green "YES" overlay
  appears and grows, and releasing past the threshold POSTs `/api/rate` with `"yes"` and
  advances to the next image.
- Repeat dragging left: confirm the same for "NO" / `"no"`.
- Drag less than the threshold and release: confirm the image snaps back to center with no
  `/api/rate` call.
- Double-tap the image: confirm it zooms in. Drag while zoomed: confirm it pans, not votes.
  Double-tap again: confirm it un-zooms.
- If double-tap doesn't reliably trigger the browser's native `dblclick` under touch emulation,
  note this so a manual double-tap detector can be added as a follow-up; do not silently accept
  broken double-tap-to-zoom on touch.

- [ ] **Step 5: Stop the verification server**

```bash
pkill -f "clawmarks.cli serve" 2>/dev/null; true
```

(Only kill this if it's the instance started in Step 1 of this task, not any other server the
user is relying on.)

---

## Task 5: Single-tap zoom, mouse drag-to-vote, rotation and thumbs overlay

Revises Tasks 1-3's gesture code per the spec's 2026-07-10 revision: zoom trigger changes from
`dblclick` to a single tap/click (classified by the same deadzone already used for drag
classification, so no new timing logic), mouse drag now votes the same way touch drag does
(rotate + colored thumbs overlay), and the swipe overlay shows a thumbs up/down icon instead of
plain color with no icon. Touch and mouse now share one classification and visual-update code
path instead of two separate implementations.

**Files:**
- Modify: `src/clawmarks/build/rate_page.py` (the `<script>` block from `// --- zoom ---` through
  the final `loadNext();` call, and the `#swipe-overlay` CSS rule's `font-size`)
- Test: `tests/test_rate_page.py`

**Interfaces:**
- Consumes: `#imgwrap`, `#img`, `#swipe-overlay`, `rate(label)`, `loadNext()`, `current` (all
  unchanged from Tasks 1-3).
- Produces: `toggleZoom(clientX, clientY)`, `classifyDrag(dx, dy)`, `updateSwipeVisual(dx)`,
  `updatePanVisual(dx, dy)`, `finishSwipe(dx)`, `finishPan(dx, dy)`, `MAX_TILT_DEG`. Removes:
  the `dblclick` listener, the separate `touchActive`/`touchClassified`/`touchStartX`/
  `touchStartY`/`touchDX`/`touchDY` and `mouseDown`/`mouseStartX`/`mouseStartY` variable sets
  (replaced by one shared `dragActive`/`dragClassified`/`dragStartX`/`dragStartY`/`dragDX`/
  `dragDY`). `clampOffset(offset, wrapSize, imgSize)`, `resetZoom()`, `zoomIn(clientX, clientY)`,
  `zoomed`, `panOffsetX`, `panOffsetY` are unchanged from Task 2 and reused as-is.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_rate_page.py`:

```python
def test_render_html_has_tap_to_zoom_and_no_dblclick():
    html = rate_page.render_html()
    assert "dblclick" not in html
    assert "function toggleZoom(" in html


def test_render_html_has_rotation_and_thumbs_overlay():
    html = rate_page.render_html()
    assert "MAX_TILT_DEG" in html
    assert "rotate(" in html
    assert "\U0001F44D" in html
    assert "\U0001F44E" in html


def test_render_html_mouse_drag_votes():
    html = rate_page.render_html()
    assert "function classifyDrag(" in html
    assert "function finishSwipe(" in html
    assert "imgwrapEl.addEventListener('mousedown'" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_rate_page.py -v`
Expected: the three new tests FAIL (`dblclick` is present today; `toggleZoom`, `MAX_TILT_DEG`,
the thumbs characters, `classifyDrag`, and a `mousedown` listener on `imgwrapEl` used for voting
don't exist yet). The five tests from Tasks 1-3 still PASS.

- [ ] **Step 3: Replace the gesture script and overlay CSS**

In `src/clawmarks/build/rate_page.py`, change the `#swipe-overlay` CSS rule's `font-size` from
`48px` to `40px` (better fit for the thumbs emoji than the old text stamp):

```css
#swipe-overlay {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  font-size:40px; font-weight:800; letter-spacing:0.08em; opacity:0; pointer-events:none; border-radius:10px; }}
```

In `loadNext()`, change the transform reset line so it also resets rotation:

```js
    img.style.transform = 'translateX(0px) rotate(0deg)';
```

Replace everything from the `// --- zoom ---` comment through the final `loadNext();` call
(i.e. all of Task 2's and Task 3's gesture code) with:

```js
// --- zoom ---

function clampOffset(offset, wrapSize, imgSize) {{
  if (imgSize <= wrapSize) return (wrapSize - imgSize) / 2;
  return Math.min(0, Math.max(wrapSize - imgSize, offset));
}}

function resetZoom() {{
  zoomed = false;
  panOffsetX = 0;
  panOffsetY = 0;
  document.getElementById('imgwrap').classList.remove('zoomed');
  document.getElementById('img').style.transform = 'translate(0px, 0px)';
}}

function zoomIn(clientX, clientY) {{
  const img = document.getElementById('img');
  const wrap = document.getElementById('imgwrap');
  const rect = img.getBoundingClientRect();
  const fracX = (clientX - rect.left) / rect.width;
  const fracY = (clientY - rect.top) / rect.height;
  wrap.classList.add('zoomed');
  zoomed = true;
  const targetX = fracX * img.naturalWidth;
  const targetY = fracY * img.naturalHeight;
  panOffsetX = clampOffset(wrap.clientWidth / 2 - targetX, wrap.clientWidth, img.naturalWidth);
  panOffsetY = clampOffset(wrap.clientHeight / 2 - targetY, wrap.clientHeight, img.naturalHeight);
  img.style.transform = `translate(${{panOffsetX}}px, ${{panOffsetY}}px)`;
}}

function toggleZoom(clientX, clientY) {{
  if (!current) return;
  if (zoomed) {{
    resetZoom();
  }} else {{
    zoomIn(clientX, clientY);
  }}
}}

const imgwrapEl = document.getElementById('imgwrap');

// --- unified drag: swipe-to-vote when not zoomed, pan when zoomed, tap toggles zoom ---

const DEADZONE_PX = 10;
const SWIPE_THRESHOLD_FRACTION = 0.25;
const MAX_TILT_DEG = 15;

let dragActive = false, dragClassified = null; // null | 'swipe' | 'pan' | 'ignore'
let dragStartX = 0, dragStartY = 0, dragDX = 0, dragDY = 0;

function classifyDrag(dx, dy) {{
  if (Math.abs(dx) < DEADZONE_PX && Math.abs(dy) < DEADZONE_PX) return null;
  if (zoomed) return 'pan';
  if (Math.abs(dx) > Math.abs(dy)) return 'swipe';
  return 'ignore';
}}

function updateSwipeVisual(dx) {{
  const img = document.getElementById('img');
  const overlay = document.getElementById('swipe-overlay');
  const width = img.getBoundingClientRect().width || 1;
  const frac = Math.min(1, Math.abs(dx) / (width * SWIPE_THRESHOLD_FRACTION));
  const deg = (dx > 0 ? 1 : -1) * frac * MAX_TILT_DEG;
  img.style.transition = '';
  img.style.transform = `translateX(${{dx}}px) rotate(${{deg}}deg)`;
  overlay.className = dx > 0 ? 'yes' : 'no';
  overlay.textContent = dx > 0 ? '\U0001F44D' : '\U0001F44E';
  overlay.style.opacity = frac;
}}

function updatePanVisual(dx, dy) {{
  const img = document.getElementById('img');
  const wrap = document.getElementById('imgwrap');
  const newX = clampOffset(panOffsetX + dx, wrap.clientWidth, img.naturalWidth);
  const newY = clampOffset(panOffsetY + dy, wrap.clientHeight, img.naturalHeight);
  img.style.transform = `translate(${{newX}}px, ${{newY}}px)`;
}}

function finishSwipe(dx) {{
  const img = document.getElementById('img');
  const overlay = document.getElementById('swipe-overlay');
  const width = img.getBoundingClientRect().width || 1;
  const threshold = width * SWIPE_THRESHOLD_FRACTION;
  if (Math.abs(dx) >= threshold) {{
    const label = dx > 0 ? 'yes' : 'no';
    const deg = (dx > 0 ? 1 : -1) * MAX_TILT_DEG;
    img.style.transition = 'transform 0.2s ease';
    img.style.transform = `translateX(${{dx > 0 ? width * 1.2 : -width * 1.2}}px) rotate(${{deg}}deg)`;
    overlay.style.opacity = 0;
    setTimeout(() => rate(label), 180);
  }} else {{
    img.style.transition = 'transform 0.2s ease';
    img.style.transform = 'translateX(0px) rotate(0deg)';
    overlay.style.opacity = 0;
  }}
}}

function finishPan(dx, dy) {{
  const img = document.getElementById('img');
  const wrap = document.getElementById('imgwrap');
  panOffsetX = clampOffset(panOffsetX + dx, wrap.clientWidth, img.naturalWidth);
  panOffsetY = clampOffset(panOffsetY + dy, wrap.clientHeight, img.naturalHeight);
}}

imgwrapEl.addEventListener('touchstart', e => {{
  if (!current || e.touches.length !== 1) return;
  dragActive = true;
  dragClassified = null;
  dragStartX = e.touches[0].clientX;
  dragStartY = e.touches[0].clientY;
  dragDX = 0;
  dragDY = 0;
}});

imgwrapEl.addEventListener('touchmove', e => {{
  if (!dragActive) return;
  const t = e.touches[0];
  dragDX = t.clientX - dragStartX;
  dragDY = t.clientY - dragStartY;
  if (dragClassified === null) dragClassified = classifyDrag(dragDX, dragDY);
  if (dragClassified === null || dragClassified === 'ignore') return;
  e.preventDefault();
  if (dragClassified === 'swipe') updateSwipeVisual(dragDX);
  else if (dragClassified === 'pan') updatePanVisual(dragDX, dragDY);
}}, {{passive: false}});

imgwrapEl.addEventListener('touchend', () => {{
  if (!dragActive) return;
  dragActive = false;
  if (dragClassified === null) {{
    toggleZoom(dragStartX, dragStartY);
  }} else if (dragClassified === 'swipe') {{
    finishSwipe(dragDX);
  }} else if (dragClassified === 'pan') {{
    finishPan(dragDX, dragDY);
  }}
  dragClassified = null;
}});

imgwrapEl.addEventListener('mousedown', e => {{
  if (!current) return;
  dragActive = true;
  dragClassified = null;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  dragDX = 0;
  dragDY = 0;
}});

document.addEventListener('mousemove', e => {{
  if (!dragActive) return;
  dragDX = e.clientX - dragStartX;
  dragDY = e.clientY - dragStartY;
  if (dragClassified === null) dragClassified = classifyDrag(dragDX, dragDY);
  if (dragClassified === null || dragClassified === 'ignore') return;
  if (dragClassified === 'swipe') updateSwipeVisual(dragDX);
  else if (dragClassified === 'pan') updatePanVisual(dragDX, dragDY);
}});

document.addEventListener('mouseup', e => {{
  if (!dragActive) return;
  dragActive = false;
  if (dragClassified === null) {{
    toggleZoom(e.clientX, e.clientY);
  }} else if (dragClassified === 'swipe') {{
    finishSwipe(dragDX);
  }} else if (dragClassified === 'pan') {{
    finishPan(dragDX, dragDY);
  }}
  dragClassified = null;
}});

loadNext();
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_rate_page.py -v`
Expected: all 8 tests PASS (5 from Tasks 1-3 + 3 new).

- [ ] **Step 5: Commit**

```bash
cd /workspace/trent-with-smart-prompts
git add src/clawmarks/build/rate_page.py tests/test_rate_page.py
git commit -m "$(cat <<'EOF'
feat(clawmarks): single-tap zoom, mouse drag-to-vote, rotation and thumbs overlay

EOF
)"
```

---

## Self-review notes

- **Spec coverage:** every row of the spec's interaction table (touch drag swipe/pan, mouse
  drag no-op/pan, single tap no-op, double click/tap zoom toggle, keyboard unchanged) is
  implemented across Tasks 1 to 3. The spec's "gesture disambiguation" requirement (one
  classification decided once per touch, reused for both gestures) is Task 3's
  `touchClassified` variable. The spec's "state reset on next image" requirement is Task 2's
  `resetZoom()` call inside `loadNext()`. The spec's testing approach (manual browser
  verification, not unit tests for gesture physics) is Task 4.
- **Placeholder scan:** no TBDs; every step shows complete, runnable code.
- **Type consistency:** `rate(label)` keeps its single-argument signature across all three
  tasks. `clampOffset(offset, wrapSize, imgSize)` is defined once in Task 2 and reused verbatim
  (same parameter order) in Task 3's pan logic on both `touchmove` and `touchend`. `zoomed`,
  `panOffsetX`, `panOffsetY` are declared once in Task 2 and read/written, never redeclared, in
  Task 3.
