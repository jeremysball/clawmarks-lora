# scan.html Grid Smooth Reflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scan.html`'s thumbnail grid animate cards into their new positions (instead of snapping instantly) when a filter/sort/favorite change reorders or removes them, using the browser's View Transitions API.

**Architecture:** `scan_gallery.py`'s `render_html()` emits a `<script>` block containing all of `scan.html`'s client-side JS as an f-string. Add a `withViewTransition(fn)` helper that calls `document.startViewTransition(fn)` when available, else calls `fn()` directly. Give each thumbnail a stable `view-transition-name` derived from its tag (sanitized to a safe CSS identifier) so the browser can match old/new DOM state across a transition. Wrap the two full-grid-rebuild call sites (`applyFilters()`'s `render()` call, and the `lightbox:favorite` listener's `render()` call) in `withViewTransition`. Leave the scroll-triggered incremental append (`renderMore()`) untouched.

**Tech Stack:** Python f-string-templated JS (no build step, no separate `.js` file for this page's own script block), pytest for string-presence assertions on the rendered HTML, Playwright MCP for live browser verification (no server-side surface to unit test).

## Global Constraints

- No em dashes in any prose written for this feature (commit messages, docs). Grep for `—` and ` -- ` before committing, per project writing-style rule.
- Conventional Commits format for every commit (`feat(scan_gallery): ...` etc.).
- Only `scan.html`'s grid is in scope. No changes to `renderMore()`, `archive.html`, `coverage.html`, or any other tool page.
- No custom transition duration/easing CSS unless live verification shows the browser default (250ms ease) looks wrong.
- Full test suite (`PYTHONPATH=src uv run pytest tests/ -v`) must stay green after each task.

---

### Task 1: Add `withViewTransition` helper, thumbnail transition names, and wrap the two render call sites

**Files:**
- Modify: `src/clawmarks/build/scan_gallery.py:294-364` (the `render()`, `thumbHtml()`, `applyFilters()`, and `lightbox:favorite` listener sections of the embedded script)
- Test: `tests/test_scan_gallery.py`

**Interfaces:**
- Consumes: nothing new from outside this file.
- Produces: `withViewTransition(fn)` (JS function, embedded in the rendered HTML's `<script>` block). Takes a zero-argument function and invokes it either inside `document.startViewTransition` or directly. Used by `applyFilters()` and the `lightbox:favorite` listener. Each `.thumb` div in the rendered grid carries `style="view-transition-name: vt-<sanitized-tag>"`.

- [ ] **Step 1: Write the failing test for `withViewTransition` presence**

Add to `tests/test_scan_gallery.py`:

```python
def test_render_html_includes_view_transition_helper():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "a", "gen": 0, "category": "seedrun1",
              "prompt_name": "fox", "prompt_type": "conflict", "prompt": "p", "strength": 1.0,
              "cfg": 5.0, "seed": 1, "steps": 28, "sampler": "ddim", "negative": "n",
              "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert "function withViewTransition(fn)" in html
    assert "document.startViewTransition" in html
    assert "withViewTransition(render)" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_scan_gallery.py::test_render_html_includes_view_transition_helper -v`
Expected: FAIL with `assert "function withViewTransition(fn)" in html` (AssertionError, string not found)

- [ ] **Step 3: Write the failing test for the thumbnail's `view-transition-name`**

Add to `tests/test_scan_gallery.py`:

```python
def test_thumb_html_has_sanitized_view_transition_name():
    items = [{"file": "a.png", "thumb": "thumbs/a.jpg", "tag": "gen3_r2/exploit#1", "gen": 3,
              "category": "seedrun1", "prompt_name": "fox", "prompt_type": "conflict", "prompt": "p",
              "strength": 1.0, "cfg": 5.0, "seed": 1, "steps": 28, "sampler": "ddim", "negative": "n",
              "faith": 0.5, "novelty": 0.5, "sim": []}]
    html = scan_gallery.render_html(items)
    assert "view-transition-name" in html
```

(This asserts the mechanism is present in the template; the sanitization itself runs client-side in the browser at render time, since `thumbHtml` builds each card from the in-page `DATA` array via JS template literals, not at Python render time. The Python-side test can only confirm the JS source contains the sanitizing call.)

- [ ] **Step 4: Run both new tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_scan_gallery.py -v -k "view_transition"`
Expected: both FAIL

- [ ] **Step 5: Implement `withViewTransition` and wrap the two call sites**

In `src/clawmarks/build/scan_gallery.py`, replace the block from the `PAGE_SIZE` comment through the `lightbox:favorite` listener (current lines 294-364) with:

```python
// Rendering all matching thumbnails in one innerHTML write is what made the page lag on every
// filter keystroke: up to 3672 <img> tags parsed/laid out at once, and on a slow connection that
// also fires a burst of thumbnail requests all at once. Instead render in chunks and grow the
// grid as the user actually scrolls near the bottom (a sentinel + IntersectionObserver), so a
// filter change repaints only a page's worth of thumbnails, not the whole result set.
const PAGE_SIZE = 150;
let shown = 0;
let sentinelObserver = null;

// Wraps a full-grid rebuild so the browser animates matching thumbnails (by
// view-transition-name) sliding from their old position to their new one, instead of snapping
// instantly. Falls back to calling fn() directly on browsers without View Transitions support.
function withViewTransition(fn) {{
  if (document.startViewTransition) document.startViewTransition(fn);
  else fn();
}}

function render() {{
  shown = 0;
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  document.getElementById('count').textContent =
    view.length + ' / ' + DATA.length + ' images | ' + Object.keys(picks).length + ' picked | ' +
    Object.keys(favorites).length + ' favorited';
  renderMore();
}}

function vtName(tag) {{
  return 'vt-' + tag.replace(/[^a-zA-Z0-9_-]/g, '_');
}}

function thumbHtml(d, i) {{
  const cls = [
    d.prompt_type + '-b',
    picks[d.tag] ? 'picked' : '',
    favorites[d.tag] ? 'favorited' : '',
  ].join(' ');
  return `
    <div class="thumb ${{cls}}" style="view-transition-name: ${{vtName(d.tag)}}"
         onclick="Lightbox.open('${{d.tag}}', view.map(v=>v.tag))" data-i="${{i}}">
      <img loading="lazy" decoding="async" src="${{d.thumb}}" data-tag="${{d.tag}}">
      ${{picks[d.tag] ? '<div class="pickbadge">&#9733;</div>' : ''}}
      ${{favorites[d.tag] ? '<div class="favbadge">&#9829;</div>' : ''}}
      <div class="meta">f=${{d.faith}} n=${{d.novelty}} ${{d.prompt_name}}</div>
    </div>`;
}}

function renderMore() {{
  const grid = document.getElementById('grid');
  const old = document.getElementById('sentinel');
  if (old) old.remove();
  const next = view.slice(shown, shown + PAGE_SIZE);
  grid.insertAdjacentHTML('beforeend', next.map((d, j) => thumbHtml(d, shown + j)).join(''));
  shown += next.length;
  if (shown < view.length) {{
    const sentinel = document.createElement('div');
    sentinel.id = 'sentinel';
    sentinel.style.gridColumn = '1 / -1';
    sentinel.style.height = '1px';
    grid.appendChild(sentinel);
    if (!sentinelObserver) {{
      sentinelObserver = new IntersectionObserver(entries => {{
        if (entries.some(e => e.isIntersecting)) renderMore();
      }}, {{rootMargin: '600px'}});
    }}
    sentinelObserver.observe(sentinel);
  }}
}}

function debounce(fn, ms) {{
  let t;
  return (...args) => {{ clearTimeout(t); t = setTimeout(() => fn(...args), ms); }};
}}
const debouncedApplyFilters = debounce(applyFilters, 250);

['sortKey', 'typeFilter', 'catFilter', 'promptFilter', 'pickedOnly', 'favoritedOnly'].forEach(id =>
  document.getElementById(id).addEventListener('input', applyFilters));
['faithMin', 'faithMax', 'search'].forEach(id =>
  document.getElementById(id).addEventListener('input', debouncedApplyFilters));

document.addEventListener('lightbox:favorite', e => {{
  if (e.detail.favorited) favorites[e.detail.tag] = true; else delete favorites[e.detail.tag];
  withViewTransition(render);
}});
```

Then change `applyFilters()`'s last line (currently `render();`, near line 291) from:

```python
  render();
}}
```

to:

```python
  withViewTransition(render);
}}
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_scan_gallery.py -v -k "view_transition"`
Expected: both PASS

- [ ] **Step 7: Run the full test suite**

Run: `PYTHONPATH=src uv run pytest tests/ -v`
Expected: all tests pass (138 total: 136 pre-existing + 2 new)

- [ ] **Step 8: Check for em dashes in the diff**

Run: `git diff src/clawmarks/build/scan_gallery.py tests/test_scan_gallery.py | rg -- "—| -- "`
Expected: no output

- [ ] **Step 9: Commit**

```bash
git add src/clawmarks/build/scan_gallery.py tests/test_scan_gallery.py
git commit -m "feat(scan_gallery): animate grid reflow on filter/sort/favorite changes

Wrap the two full-grid rebuild call sites (applyFilters, the
lightbox:favorite listener) in withViewTransition, and give each
thumbnail a stable view-transition-name derived from its tag. The
browser now slides cards from their old position to their new one
instead of snapping instantly, so it's easier to track where an image
went when the result set changes. Falls back to today's instant-snap
behavior on browsers without View Transitions support. The
scroll-triggered incremental append (renderMore) is untouched, per
the design doc's scoping: animating up to 150 newly appended cards at
once would look like a cascade, not a reflow."
```

---

### Task 2: Live verification with Playwright

**Files:** none (verification only, no code changes)

**Interfaces:** none.

- [ ] **Step 1: Start the curation server**

Run: `PYTHONPATH=src uv run python -m clawmarks.curation_server notes/uncanny_seedrun1 --port 8420 &`

Confirm it's listening: `curl -sS -o /dev/null -w '%{{http_code}}\n' http://127.0.0.1:8420/scan.html`
Expected: `200`

- [ ] **Step 2: Navigate to scan.html and spy on `document.startViewTransition`**

Use `mcp__playwright__browser_navigate` to open `http://127.0.0.1:8420/scan.html`, then use `mcp__playwright__browser_evaluate` to install a spy before triggering a filter change:

```js
() => {
  window.__vtCalls = 0;
  const orig = document.startViewTransition?.bind(document);
  document.startViewTransition = (fn) => {
    window.__vtCalls++;
    return orig ? orig(fn) : (fn(), { finished: Promise.resolve(), ready: Promise.resolve(), updateCallbackDone: Promise.resolve() });
  };
}
```

- [ ] **Step 3: Trigger a filter change that removes some cards**

Use `mcp__playwright__browser_evaluate` to set the type filter to `"style"` and dispatch an `input` event:

```js
() => {
  const sel = document.getElementById('typeFilter');
  sel.value = 'style';
  sel.dispatchEvent(new Event('input'));
}
```

- [ ] **Step 4: Confirm the spy was invoked and the page didn't error**

Use `mcp__playwright__browser_console_messages` to confirm no new errors appeared, then `mcp__playwright__browser_evaluate` with `() => window.__vtCalls` to confirm it's `>= 1`.

- [ ] **Step 5: Confirm the resulting grid matches a direct `render()` call**

Use `mcp__playwright__browser_evaluate`:

```js
() => {
  const beforeCount = document.querySelectorAll('#grid .thumb').length;
  render();
  const afterCount = document.querySelectorAll('#grid .thumb').length;
  return { beforeCount, afterCount };
}
```

Expected: `beforeCount === afterCount` (the transitioned filter change already produced the same DOM `render()` would produce directly; re-running `render()` is idempotent and shouldn't change the count).

- [ ] **Step 6: Visually confirm the animation looks right**

Use `mcp__playwright__browser_take_screenshot` immediately after triggering a second filter change (e.g. switch back to `""` for all types) to eyeball that the transition doesn't look broken (no flash-of-unstyled-grid, no overlapping cards). If the default 250ms timing looks off, add the optional CSS override from the spec's "Out of scope" section:

```python
::view-transition-group(*) {{ animation-duration: 0.2s; }}
```

placed in the `<style>` block near the other transition-speed rules in `scan_gallery.py`, then repeat steps 1-6 to reverify, then commit that CSS-only change separately (`fix(scan_gallery): tighten view-transition timing to match page's existing speed`) if it was needed. If the default timing already looks right, skip this and make no commit.

- [ ] **Step 7: Stop the server**

Run: `kill %1` (or find and kill the PID from step 1 if backgrounding via `&` isn't in the same shell session)

---

## Self-Review Notes

- **Spec coverage:** Problem/Approach/Scope/Implementation sketch/Out of scope/Testing sections of the spec are all covered: Task 1 implements the helper, the two wrapped call sites, and the transition names; Task 2 covers the spec's exact three-part Playwright verification (page doesn't error, `startViewTransition` was invoked, resulting DOM matches direct `render()`), plus the optional timing-CSS fallback the spec allows.
- **Placeholder scan:** no TBD/TODO; every step has literal code or literal commands with expected output.
- **Type consistency:** `withViewTransition(fn)` takes a zero-arg function in both its definition and both call sites (`withViewTransition(render)` twice). `vtName(tag)` is a new helper, not previously named anything else. `thumbHtml`, `render`, `renderMore`, `applyFilters` keep their existing names and signatures from the current codebase, matching what Task 1 Step 5 diffs against.
