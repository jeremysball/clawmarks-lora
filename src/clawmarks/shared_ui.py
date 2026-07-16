"""
Shared UI pieces used by every build/*.py tool-page generator, so the lightbox, the top
navigation bar, and its scroll-to-hide behavior are defined once instead of duplicated across
every page. Import and use:

    from clawmarks.shared_ui import nav_bar_html, TOPNAV_CSS, SCROLLNAV_JS, _LIGHTBOX_JS

`curation_server.py` serves `_LIGHTBOX_JS`, `SCROLLNAV_JS`, and `INFOTIP_JS` directly from
`/lightbox.js`, `/scrollnav.js`, and `/infotip.js` routes; every generated page includes them
with `<script src="lightbox.js"></script>` and opens images via `Lightbox.open(tag)` instead of
`window.open('scan.html?open=...')`: no new tab, no page load, works from any page because the
module fetches scan_data.json itself.
"""
import html
import json


def json_script(data):
    """json.dumps() output embedded raw inside a <script> tag is a stored-XSS vector: it does
    not escape the literal substring "</script>", so any string field containing
    "</script><script>...injected..." closes the intended script block early and lets whatever
    follows execute as HTML/JS. Escaping "<" as its JSON/JS-safe unicode form (which every JSON
    and JS parser reads back as a plain "<") neutralizes that and every other tag-opening
    sequence without changing the decoded value."""
    return json.dumps(data).replace("<", "\\u003c")


NAV_GROUPS = [
    ("Generate", [("cockpit.html", "generation cockpit"), ("runs.html", "search runs"),
                  ("seeds.html", "candidate seeds")]),
    ("Curate", [("compare.html", "compare images (head-to-head)"),
                ("scan.html", "scan gallery"), ("archive.html", "elite archive")]),
    ("Understand search", [("map.html", "solution map (UMAP)"),
                           ("coverage.html", "coverage / void map"),
                           ("redundancy.html", "redundancy clusters"),
                           ("novelty_decay.html", "novelty decay watchlist"),
                           ("lineage.html", "lineage tree")]),
    ("Preference model", [("preference_status.html", "preference status"),
                          ("preference_rank.html", "predicted preference")]),
]
NAV_OPTIONS = [("explore.html", "all tools (hub)")] + [
    option for _group, options in NAV_GROUPS for option in options
]


DARK_TOKENS = """
:root { color-scheme:dark; --bg:#0b0b0d; --panel:#16161a; --panel-2:#1d1d22; --border:#2a2a30;
  --text:#eaeaee; --text-dim:#9a9aa4; --text-faint:#6a6a74; --accent:#7c9eff; --pick:#f5c542;
  --up:#5ec98a; --down:#e0605e; }
"""

BTN_CSS = """
.btn { font-size:13px; padding:6px 12px; border-radius:6px; border:1px solid var(--border);
  background:var(--panel-2); color:var(--text); cursor:pointer; }
.btn--primary { background:var(--accent); color:#0b0b0d; font-weight:600; border-color:var(--accent); }
.btn--secondary { background:var(--panel-2); color:var(--text); border:1px solid var(--border); }
.btn:disabled { opacity:0.4; cursor:not-allowed; }
"""


def nav_bar_html(current, active_expedition=None, active_leg=None, running=None):
    opts = '<option value="explore.html">all tools (hub)</option>' + "".join(
        f'<optgroup label="{group}">' + "".join(
            f'<option value="{href}"{" selected" if href == current else ""}>{label}</option>'
            for href, label in options
        ) + '</optgroup>'
        for group, options in NAV_GROUPS
    )
    active_label = ""
    if active_expedition and active_leg:
        label = html.escape(f"{active_expedition}/{active_leg}")
        active_label = f'<a class="nav-activeleg" href="/">{label}</a>'
    running_label = ""
    if running:
        r_exp, r_leg = running
        running_label = (
            f'<span id="nav-running" class="nav-running" '
            f'title="an overnight search run is live">RUNNING: {r_exp}/{r_leg}</span>'
        )
    return (
        f'<div id="topnav" class="topnav" data-autohide '
        f'data-expedition="{html.escape(active_expedition or "")}" '
        f'data-leg="{html.escape(active_leg or "")}">'
        '<a class="navlink" href="explore.html">&larr; all tools</a>'
        f'{active_label}{running_label}'
        '<select onchange="if(this.value) location.href=this.value;">'
        f'<option value="">jump to...</option>{opts}</select></div>'
    )


TOPNAV_CSS = """
.topnav { position:sticky; top:0; z-index:50; background:rgba(22,22,26,0.92); backdrop-filter:blur(10px);
  border-bottom:1px solid var(--border,#2a2a30); padding:10px 16px; display:flex; gap:14px; align-items:center;
  transition: transform .18s ease; }
.topnav.navhidden { transform: translateY(-100%); }
.topnav select { background:var(--panel-2,#1d1d22); color:var(--text,#eaeaee); border:1px solid var(--border,#2a2a30);
  border-radius:6px; padding:5px 9px; font-size:12.5px; max-width:220px; }
.topnav .nav-activeleg { color:var(--text-dim,#9a9aa4); font-size:12px; font-family:monospace;
  text-decoration:none; padding:2px 8px; background:rgba(154,154,164,0.12); border-radius:5px; white-space:nowrap; }
.topnav .nav-running { color:#0b0b0d; font-size:11.5px; font-weight:700; padding:2px 8px;
  background:var(--up,#5ec98a); border-radius:5px; white-space:nowrap; letter-spacing:0.02em; }
@media (max-width: 640px) {
  .topnav { padding:8px 10px; gap:8px; font-size:12px; flex-wrap:wrap; }
  .topnav select { flex:1; min-width:0; max-width:none; }
}
"""

_infotip_counter = 0

DINO_TIP = (
    "DINOv2 is an open vision model that turns an image into about 768 numbers (an embedding) "
    "capturing style without human labels; similar style gives similar embeddings, so we measure "
    "style match without a human."
)


def info_btn(tip):
    """A small tappable (?) icon that shows `tip` in a popover. Click-based, not hover-only, so
    it works on touch: this whole project is meant to become a general tool for exploring an
    AI-generated image space, not a one-off for this dataset, so every non-obvious concept
    (faithfulness, novelty, picking, favoriting...) gets one of these next to it instead of
    assuming the reader already knows the vocabulary."""
    global _infotip_counter
    _infotip_counter += 1
    tip_escaped = tip.replace('"', "&quot;")
    return f'<span class="infobtn" data-id="tip{_infotip_counter}" data-tip="{tip_escaped}">?</span>'


INFOTIP_CSS = """
.infobtn { display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px;
  border-radius:50%; background:rgba(154,154,164,0.18); color:#9a9aa4; font-size:10.5px;
  cursor:pointer; border:1px solid rgba(154,154,164,0.35); flex-shrink:0; user-select:none;
  font-weight:600; line-height:1; }
.infobtn:hover, .infobtn.active { background:rgba(124,158,255,0.25); color:#eaeaee; border-color:#7c9eff; }
.infopop { position:fixed; z-index:2000; max-width:280px; background:#1d1d22; border:1px solid #2a2a30;
  border-radius:8px; padding:10px 12px; font-size:12px; line-height:1.55; color:#dcdce2;
  box-shadow:0 10px 30px rgba(0,0,0,0.5); display:none; }
.infopop.open { display:block; }
@media (max-width: 640px) {
  .infopop { max-width:min(280px, 82vw); }
}
"""

INFOTIP_JS = """
(function(){
  document.addEventListener('click', function(e){
    var btn = e.target.closest ? e.target.closest('.infobtn') : null;
    document.querySelectorAll('.infopop.open').forEach(function(p){
      if (!btn || p.dataset.for !== btn.dataset.id) p.classList.remove('open');
    });
    document.querySelectorAll('.infobtn.active').forEach(function(b){
      if (b !== btn) b.classList.remove('active');
    });
    if (!btn) return;
    e.stopPropagation();
    // infobtn spans are nested inside <label> elements wrapping the filter control they
    // annotate (e.g. <label>Sort<span class="infobtn">...</span> <select>...). Without this,
    // the label's default action re-fires a synthetic click at that control right after this
    // one, which bubbles to document a second time with no infobtn target and immediately
    // closes the tooltip this same click just opened.
    e.preventDefault();
    var pop = document.querySelector('.infopop[data-for="' + btn.dataset.id + '"]');
    if (!pop) {
      pop = document.createElement('div');
      pop.className = 'infopop';
      pop.dataset.for = btn.dataset.id;
      pop.textContent = btn.dataset.tip;
      document.body.appendChild(pop);
    }
    // .infopop is position:fixed, so getBoundingClientRect()'s viewport-relative coordinates
    // can be used directly: no window.scrollX/Y adjustment needed. That adjustment used to be
    // necessary for absolutely-positioned popovers, but it silently placed the popover
    // off-screen whenever the button lived inside a position:fixed ancestor (like the lightbox
    // overlay), since a fixed element's viewport rect doesn't move with page scroll.
    var rect = btn.getBoundingClientRect();
    var left = Math.min(rect.left, window.innerWidth - 300);
    pop.style.top = (rect.bottom + 6) + 'px';
    pop.style.left = Math.max(8, left) + 'px';
    var willOpen = !pop.classList.contains('open');
    pop.classList.toggle('open', willOpen);
    btn.classList.toggle('active', willOpen);
  });
})();
"""


MOBILE_BASE_CSS = """
html, body { max-width:100vw; overflow-x:hidden; }
* { -webkit-tap-highlight-color: transparent; }
@media (max-width: 640px) {
  body { padding:10px !important; }
  h1 { font-size:16px !important; }
  p.sub { font-size:12.5px !important; }
  button, select, input { font-size:14px !important; min-height:34px; }
}
"""

SCROLLNAV_JS = """
(function(){
  var bar = document.querySelector('[data-autohide]');
  if (!bar) return;
  var lastY = window.scrollY, ticking = false;
  function onScroll(){
    var y = Math.max(0, window.scrollY);
    if (y > lastY && y > 60) bar.classList.add('navhidden');
    else if (y < lastY) bar.classList.remove('navhidden');
    lastY = y; ticking = false;
  }
  window.addEventListener('scroll', function(){
    if (!ticking) { requestAnimationFrame(onScroll); ticking = true; }
  }, {passive:true});
})();
"""

_LIGHTBOX_JS = r"""(function(){
  // json_script() (see json_script() above) only protects the initial <script> data
  // declaration from a </script> breakout; it does not HTML-escape the decoded string values.
  // Any manifest field written into innerHTML or an HTML attribute below must go through
  // escHtml() first, or a model-generated tag/prompt containing e.g. "<img src=x onerror=...>"
  // executes when the lightbox renders it.
  function escHtml(s){
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  // Loads a thumb immediately (whatever's already cached/fast), then swaps to the full-res
  // image asynchronously once it's loaded, instead of every caller either blocking on a heavy
  // full-res load or being stuck on a thumb forever. Exposed on window so any page's own JS can
  // call it without importing anything beyond this lightbox.js script tag.
  let progressiveCssInjected = false;
  function ensureProgressiveCss(){
    if (progressiveCssInjected) return;
    progressiveCssInjected = true;
    const style = document.createElement('style');
    style.textContent = '.progressive-loading { filter:blur(8px); opacity:0.85; ' +
      'transition:filter .2s ease, opacity .2s ease; }';
    document.head.appendChild(style);
  }
  function mountProgressive(imgEl, thumbSrc, fullSrc){
    ensureProgressiveCss();
    imgEl.src = thumbSrc;
    imgEl.classList.add('progressive-loading');
    // imgEl is the shared lightbox <img>, reused across navigations, so a stale full-res load
    // from a previous mountProgressive call can still be in flight when the user navigates
    // again. Stamp a per-call token and check it before mutating imgEl in either callback, or
    // the old image's full-res clobbers whatever the user has since navigated to.
    const my = (imgEl._mpToken = (imgEl._mpToken || 0) + 1);
    const clearBlur = () => {
      if (imgEl._mpToken !== my) return;
      imgEl.src = fullSrc;
      imgEl.classList.remove('progressive-loading');
    };
    const full = new Image();
    full.onload = clearBlur;
    full.onerror = () => {
      if (imgEl._mpToken !== my) return;
      imgEl.classList.remove('progressive-loading');
    };
    full.src = fullSrc;
  }
  window.mountProgressive = mountProgressive;

  let DATA = null, byTag = null;
  let order = [];
  let idx = -1;
  let history = [];
  let favorites = {};
  let counterfactuals = {};
  let el = null;
  let LB_EXPEDITION = null;
  let LB_LEG = null;

  function ensureDom(){
    if (el) return;
    const style = document.createElement('style');
    style.textContent = `
#lb-overlay { position:fixed; inset:0; background:rgba(8,8,10,0.94); backdrop-filter:blur(6px);
  display:none; align-items:center; justify-content:center; z-index:1000; flex-direction:column; gap:10px; padding:20px; }
#lb-overlay.open { display:flex; }
#lb-overlay .lb-imgwrap { position:relative; display:flex; align-items:center; justify-content:center;
  max-width:92vw; max-height:58vh; }
#lb-overlay img.lb-main { max-width:92vw; max-height:58vh; object-fit:contain; border-radius:8px;
  box-shadow:0 20px 60px rgba(0,0,0,0.6); transition:opacity .15s ease; }
#lb-overlay .lb-imgwrap.loading img.lb-main { opacity:0.3; }
#lb-overlay .lb-spinner { display:none; position:absolute; width:34px; height:34px; margin:-17px 0 0 -17px;
  top:50%; left:50%; border:3px solid rgba(255,255,255,0.2); border-top-color:#7c9eff; border-radius:50%;
  animation:lb-spin 0.8s linear infinite; }
#lb-overlay .lb-imgwrap.loading .lb-spinner { display:block; }
@keyframes lb-spin { to { transform:rotate(360deg); } }
#lb-overlay .lb-info { color:#9a9aa4; font-size:12.5px; max-width:92vw; text-align:center; line-height:1.6; }
#lb-overlay .lb-nav { position:absolute; top:0; bottom:0; width:16%; cursor:pointer; z-index:1;
  display:flex; align-items:center; }
#lb-overlay .lb-nav::before { content:''; width:44px; height:44px; border-radius:50%;
  background:rgba(20,20,24,0.55); border:1px solid rgba(255,255,255,0.14); color:#eaeaee;
  display:flex; align-items:center; justify-content:center; font-size:22px; line-height:1;
  opacity:0.5; transition:opacity .15s ease, background .15s ease; }
#lb-overlay .lb-nav:hover::before { opacity:1; background:rgba(34,34,40,0.85); }
#lb-overlay .lb-prev { left:0; justify-content:flex-start; padding-left:14px; }
#lb-overlay .lb-prev::before { content:'\\2039'; }
#lb-overlay .lb-next { right:0; justify-content:flex-end; padding-right:14px; }
#lb-overlay .lb-next::before { content:'\\203A'; }
#lb-overlay .lb-close { position:absolute; top:16px; right:22px; font-size:28px; cursor:pointer; color:#9a9aa4;
  width:40px; height:40px; display:flex; align-items:center; justify-content:center; z-index:2; }
#lb-overlay .lb-close:hover { color:#eaeaee; }
#lb-overlay .lb-actions { position:relative; z-index:2; display:flex; gap:10px; align-items:center;
  flex-wrap:wrap; justify-content:center; }
#lb-overlay button { background:#1d1d22; color:#eaeaee; border:1px solid #2a2a30; border-radius:7px;
  padding:8px 16px; font-size:13px; cursor:pointer; }
#lb-overlay button.favorited { background:rgba(224,96,150,0.16); border-color:#e0609a; color:#e0609a; }
#lb-overlay .lb-actions .infobtn { background:rgba(255,255,255,0.12); border-color:rgba(255,255,255,0.3); color:#dcdce2; }
#lb-overlay .lb-simlabel { font-size:11px; color:#6a6a74; letter-spacing:0.02em; text-transform:uppercase; }
#lb-overlay .lb-simstrip { position:relative; z-index:2; display:flex; gap:7px; overflow-x:auto;
  max-width:92vw; padding:4px 2px 8px; }
#lb-overlay .lb-simstrip img { width:64px; height:64px; object-fit:cover; border-radius:6px; cursor:pointer;
  flex-shrink:0; opacity:0.7; outline:2px solid transparent; }
#lb-overlay .lb-simstrip img:hover { opacity:1; outline-color:#7c9eff; }
#lb-overlay .lb-cf-panel { position:relative; z-index:2; display:none; background:rgba(255,255,255,0.05);
  border:1px solid #2a2a30; border-radius:8px; padding:12px 14px; max-width:92vw; width:520px; text-align:left; }
#lb-overlay .lb-cf-panel.open { display:block; }
#lb-overlay .lb-cf-panel label { display:block; font-size:11px; color:#9a9aa4; margin:8px 0 3px; }
#lb-overlay .lb-cf-panel textarea, #lb-overlay .lb-cf-panel input {
  width:100%; box-sizing:border-box; background:#1d1d22; color:#eaeaee; border:1px solid #2a2a30;
  border-radius:6px; padding:6px 9px; font-size:12.5px; font-family:inherit; }
#lb-overlay .lb-cf-panel textarea { min-height:52px; resize:vertical; }
#lb-overlay .lb-cf-row { display:flex; gap:10px; }
#lb-overlay .lb-cf-row > div { flex:1; }
#lb-overlay .lb-cf-submit { margin-top:10px; background:#2a4a7c; border-color:#3a5a8c; color:#cddcff; }
#lb-overlay .lb-cf-status { font-size:11.5px; color:#9a9aa4; margin-top:8px; min-height:1.4em; }
#lb-overlay .lb-cf-status.err { color:#e0605e; }
#lb-overlay .lb-cf-result { margin-top:10px; display:none; flex-wrap:wrap; gap:6px; }
#lb-overlay .lb-cf-result.open { display:flex; }
#lb-overlay .lb-cf-result img { width:100px; height:100px; object-fit:cover; border-radius:6px; cursor:pointer;
  opacity:0.85; }
#lb-overlay .lb-cf-result img:hover { opacity:1; }
#lb-overlay .lb-cf-list { margin-top:10px; display:flex; gap:6px; overflow-x:auto; }
#lb-overlay .lb-cf-list img { width:60px; height:60px; object-fit:cover; border-radius:5px; cursor:pointer;
  flex-shrink:0; opacity:0.85; }
#lb-overlay .lb-cf-list img:hover { opacity:1; }
@media (max-width:640px) {
  #lb-overlay { padding:10px; }
  #lb-overlay img.lb-main { max-height:42vh; }
  #lb-overlay .lb-nav { width:22%; }
  #lb-overlay .lb-close { top:6px; right:6px; }
  #lb-overlay .lb-actions button { font-size:12.5px; padding:8px 14px; }
  #lb-overlay .lb-simstrip img { width:50px; height:50px; }
  #lb-overlay .lb-info { font-size:11.5px; }
  #lb-overlay .lb-cf-panel { width:100%; }
  #lb-overlay .lb-cf-row { flex-direction:column; gap:0; }
}
`;
    document.head.appendChild(style);

    el = document.createElement('div');
    el.id = 'lb-overlay';
    el.innerHTML = `
  <span class="lb-close">&times;</span>
  <div class="lb-nav lb-prev"></div>
  <div class="lb-nav lb-next"></div>
  <div class="lb-imgwrap">
    <img class="lb-main">
    <div class="lb-spinner"></div>
  </div>
  <div class="lb-info"></div>
  <div class="lb-actions">
    <button class="lb-back" style="display:none;">&#8592; back</button>
    <button class="lb-favorite">&#9825; favorite</button>
    <span class="infobtn" data-id="lb-tip-favorite" data-tip="Favoriting just bookmarks this image for your own reference (e.g. for a writeup). Unlike picking, it has no effect on the search: use it for images you like but don't want the next generation to build on.">?</span>
    <button class="lb-cf-toggle">&#8635; generate counterfactual</button>
    <span class="infobtn" data-id="lb-tip-cf" data-tip="A counterfactual asks 'what if this image had used different settings' by generating a brand-new image right now, keeping whichever fields you don't change and varying the ones you do. It costs real generation time/money (seconds if the endpoint is warm, minutes if it has to cold-start) and never feeds back into the search on its own: it's a side-by-side comparison tool, not a pick.">?</span>
  </div>
  <div class="lb-cf-panel">
    <label>Prompt</label>
    <textarea class="lb-cf-prompt"></textarea>
    <div class="lb-cf-row">
      <div><label>Strength</label><input class="lb-cf-strength" type="number" step="0.05"></div>
      <div><label>CFG</label><input class="lb-cf-cfg" type="number" step="0.5"></div>
      <div><label>Seed</label><input class="lb-cf-seed" type="number" step="1"></div>
      <div><label>Count (n)</label><input class="lb-cf-n" type="number" step="1" min="1" max="6" value="1"></div>
    </div>
    <button class="lb-cf-submit">Generate</button>
    <div class="lb-cf-status"></div>
    <div class="lb-cf-result"></div>
    <div class="lb-cf-list"></div>
  </div>
  <div class="lb-simlabel">similar images (by DINOv2 embedding)</div>
  <div class="lb-simstrip"></div>`;
    const context = document.getElementById('topnav');
    LB_EXPEDITION = context ? context.dataset.expedition : null;
    LB_LEG = context ? context.dataset.leg : null;
    el.dataset.expedition = LB_EXPEDITION || '';
    el.dataset.leg = LB_LEG || '';
    document.body.appendChild(el);

    const mainImg = el.querySelector('.lb-main');
    const imgWrap = el.querySelector('.lb-imgwrap');
    mainImg.addEventListener('load', () => imgWrap.classList.remove('loading'));
    mainImg.addEventListener('error', () => imgWrap.classList.remove('loading'));

    el.querySelector('.lb-close').onclick = close;
    el.querySelector('.lb-prev').onclick = () => step(-1);
    el.querySelector('.lb-next').onclick = () => step(1);
    el.querySelector('.lb-back').onclick = back;
    el.querySelector('.lb-favorite').onclick = toggleFavorite;
    el.querySelector('.lb-cf-toggle').onclick = toggleCfPanel;
    el.querySelector('.lb-cf-submit').onclick = submitCounterfactual;
    el.addEventListener('click', e => { if (e.target === el) close(); });
    document.addEventListener('keydown', e => {
      if (!el.classList.contains('open')) return;
      if (e.key === 'ArrowRight') step(1);
      if (e.key === 'ArrowLeft') step(-1);
      if (e.key === 'Escape') close();
      if (e.key === 'f' || e.key === 'F') { e.preventDefault(); toggleFavorite(); }
      if (e.key === 'Backspace' && history.length) { e.preventDefault(); back(); }
    });
  }

  function dataUrl(){
    // Resolve scan_data.json relative to wherever this script tag was loaded from, so the
    // module works no matter which tool page (in the same served directory) includes it.
    const scripts = document.getElementsByTagName('script');
    for (const s of scripts) {
      if (s.src && s.src.indexOf('lightbox.js') !== -1) {
        return new URL('scan_data.json', s.src).toString();
      }
    }
    return 'scan_data.json';
  }

  function loadData(){
    if (DATA) return Promise.resolve(DATA);
    return fetch(dataUrl()).then(r => r.json()).then(d => {
      DATA = d;
      byTag = {};
      d.forEach(it => byTag[it.tag] = it);
      return d;
    });
  }
  function loadFavorites(){
    return fetch('/api/favorites').then(r => r.json()).then(f => { favorites = f; }).catch(() => {});
  }
  function loadCounterfactuals(){
    return fetch('/api/counterfactuals').then(r => r.json()).then(c => { counterfactuals = c; }).catch(() => {});
  }

  // Shared full-size prefetch cache, keyed by tag: { img, done }. `done` means the load
  // finished (success or error) and the browser now has it cached, so it's never restarted.
  // While a prefetch is still in flight (`done` false, `img` set), it can be aborted by
  // clearing the Image's src, which stops the in-progress network request in every major
  // browser, and resumed later from scratch by calling prefetchImage() again.
  const prefetchState = new Map();
  function prefetchImage(d){
    if (!d || !d.file) return;
    const existing = prefetchState.get(d.tag);
    if (existing && (existing.done || existing.img)) return;
    const img = new Image();
    img.onload = img.onerror = () => {
      const st = prefetchState.get(d.tag);
      if (st) { st.done = true; st.img = null; }
    };
    img.src = d.file;
    prefetchState.set(d.tag, {img, done: false});
  }
  function abortPrefetch(tag){
    const st = prefetchState.get(tag);
    if (!st || st.done || !st.img) return;
    st.img.onload = st.img.onerror = null;
    st.img.src = ''; // aborts the in-flight request
    prefetchState.delete(tag);
  }
  function prefetchNeighbors(){
    if (order.length < 2) return;
    // Stage 1: the very next image, fetched immediately so it's usually already cached by
    // the time the user taps "next". Stage 2: a slightly wider window, delayed so it never
    // competes with stage 1 for bandwidth on a slow connection.
    prefetchImage(order[(idx + 1) % order.length]);
    setTimeout(() => {
      prefetchImage(order[(idx + 2) % order.length]);
      prefetchImage(order[(idx - 1 + order.length) % order.length]);
    }, 150);
  }

  function render(){
    const d = order[idx];
    const mainImg = el.querySelector('.lb-main');
    el.querySelector('.lb-imgwrap').classList.add('loading');
    if (d.thumb) {
      mountProgressive(mainImg, d.thumb, d.file);
    } else {
      mainImg.src = d.file;
    }
    prefetchNeighbors();
    // Counterfactual records (jumped into via the cf result grid) carry origin_tag/strength/cfg
    // but none of a search-manifest entry's gen/category/prompt_type/faith/novelty fields, so
    // the two need separate info-line formats rather than one that prints "undefined" for half
    // of them.
    el.querySelector('.lb-info').textContent = ('origin_tag' in d)
      ? `${d.tag} | counterfactual of ${d.origin_tag} | prompt=${d.prompt} | ` +
        `strength=${d.strength} cfg=${d.cfg} seed=${d.seed}`
      : `${d.tag} | gen ${d.gen} | ${d.category} | type=${d.prompt_type} | prompt=${d.prompt_name} | ` +
        `strength=${d.strength} cfg=${d.cfg} | faith=${d.faith} novelty=${d.novelty}`;
    const isFav = !!favorites[d.tag];
    const favBtn = el.querySelector('.lb-favorite');
    favBtn.textContent = isFav ? '♥ favorited (click to remove)' : '♡ favorite';
    favBtn.classList.toggle('favorited', isFav);
    el.querySelector('.lb-back').style.display = history.length ? 'inline-block' : 'none';

    const strip = el.querySelector('.lb-simstrip');
    const simTags = d.sim || [];
    if (simTags.length) {
      strip.innerHTML = simTags.map(t => {
        const n = byTag[t];
        if (!n) return '';
        return `<img loading="lazy" src="${escHtml(n.thumb)}" title="f=${n.faith} n=${n.novelty} ${escHtml(n.prompt_name)}" data-tag="${escHtml(t)}">`;
      }).join('');
      strip.querySelectorAll('img').forEach(img => { img.onclick = () => jump(img.dataset.tag); });
      strip.style.display = 'flex';
      el.querySelector('.lb-simlabel').style.display = 'block';
    } else {
      strip.innerHTML = '';
      strip.style.display = 'none';
      el.querySelector('.lb-simlabel').style.display = 'none';
    }

    el.querySelector('.lb-cf-panel').classList.remove('open');
    el.querySelector('.lb-cf-result').classList.remove('open');
    el.querySelector('.lb-cf-result').innerHTML = '';
    el.querySelector('.lb-cf-status').textContent = '';
    el.querySelector('.lb-cf-status').classList.remove('err');
    renderCfList(d);
  }

  function renderCfList(d){
    const list = el.querySelector('.lb-cf-list');
    const mine = Object.values(counterfactuals).filter(c => c.origin_tag === d.tag);
    if (!mine.length) { list.innerHTML = ''; return; }
    list.innerHTML = mine.map(c =>
      `<img loading="lazy" src="${escHtml(c.file)}" title="s=${c.strength} cfg=${c.cfg} seed=${escHtml(String(c.seed))}">`
    ).join('');
  }

  function toggleCfPanel(){
    const d = order[idx];
    const panel = el.querySelector('.lb-cf-panel');
    const opening = !panel.classList.contains('open');
    panel.classList.toggle('open', opening);
    if (opening) {
      el.querySelector('.lb-cf-prompt').value = d.prompt;
      el.querySelector('.lb-cf-strength').value = d.strength;
      el.querySelector('.lb-cf-cfg').value = d.cfg;
      el.querySelector('.lb-cf-seed').value = '';
      el.querySelector('.lb-cf-n').value = 1;
      el.querySelector('.lb-cf-status').textContent = '';
      el.querySelector('.lb-cf-status').classList.remove('err');
      el.querySelector('.lb-cf-result').classList.remove('open');
      el.querySelector('.lb-cf-result').innerHTML = '';
    }
  }

  function submitCounterfactual(){
    const d = order[idx];
    const prompt = el.querySelector('.lb-cf-prompt').value.trim();
    const strength = parseFloat(el.querySelector('.lb-cf-strength').value);
    const cfg = parseFloat(el.querySelector('.lb-cf-cfg').value);
    const seedRaw = el.querySelector('.lb-cf-seed').value.trim();
    const nRaw = parseInt(el.querySelector('.lb-cf-n').value, 10);
    const n = Number.isFinite(nRaw) && nRaw > 0 ? nRaw : 1;
    const overridden = [];
    if (prompt !== d.prompt) overridden.push('prompt');
    if (strength !== d.strength) overridden.push('strength');
    if (cfg !== d.cfg) overridden.push('cfg');
    if (seedRaw) overridden.push('seed');

    const body = {
      origin_tag: d.tag, prompt, strength, cfg,
      seed: seedRaw ? parseInt(seedRaw, 10) : null,
      steps: d.steps, sampler: d.sampler, negative: d.negative,
      overridden, n,
    };

    const submitBtn = el.querySelector('.lb-cf-submit');
    const status = el.querySelector('.lb-cf-status');
    submitBtn.disabled = true;
    status.classList.remove('err');
    status.textContent = (n > 1 ? `Generating ${n} variations...` : 'Generating...') +
      ' a few seconds if the endpoint is already warm, up to 5 minutes if it has to cold-start a worker. Keep this tab open.';

    fetch('/api/counterfactual', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
      .then(r => r.json().then(data => ({ok: r.ok, data})))
      .then(({ok, data}) => {
        submitBtn.disabled = false;
        const results = data.results || [];
        if (results.length) {
          results.forEach(r => { counterfactuals[r.tag] = r; byTag[r.tag] = r; });
          const resultDiv = el.querySelector('.lb-cf-result');
          resultDiv.innerHTML = results.map(r =>
            `<img loading="lazy" src="${escHtml(r.file)}" title="s=${r.strength} cfg=${r.cfg} seed=${escHtml(String(r.seed))}" data-tag="${escHtml(r.tag)}">`
          ).join('');
          resultDiv.querySelectorAll('img').forEach(img => { img.onclick = () => jump(img.dataset.tag); });
          resultDiv.classList.add('open');
          renderCfList(d);
        }
        if (!ok || data.error) {
          status.classList.add('err');
          status.textContent = data.error || 'generation failed';
          return;
        }
        status.textContent = results.length > 1 ? `Done: ${results.length} variations.` : 'Done.';
      })
      .catch(e => {
        submitBtn.disabled = false;
        status.classList.add('err');
        status.textContent = 'request failed: ' + e;
      });
  }

  function step(delta){ idx = (idx + delta + order.length) % order.length; render(); }
  function jump(tag){
    history.push({order, idx});
    const d = byTag[tag];
    if (!d) return;
    let i = order.indexOf(d);
    if (i === -1) { order = [d]; i = 0; }
    idx = i;
    render();
  }
  function back(){
    if (!history.length) return;
    const prev = history.pop();
    order = prev.order; idx = prev.idx;
    render();
  }
  function toggleFavorite(){
    const d = order[idx];
    const isFav = !!favorites[d.tag];
    const endpoint = isFav ? '/api/unfavorite' : '/api/favorite';
    const body = isFav
      ? {tag: d.tag, expedition: LB_EXPEDITION, leg: LB_LEG}
      : Object.assign({}, d, {expedition: LB_EXPEDITION, leg: LB_LEG});
    const removedRecord = isFav ? favorites[d.tag] : null;
    fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})
      .then(r => { if (!r.ok) throw new Error('favorite save failed'); return r.json(); })
      .then(res => {
        if (res.error) throw new Error(res.error);
        if (isFav) delete favorites[d.tag]; else favorites[d.tag] = body;
        render();
        document.dispatchEvent(new CustomEvent('lightbox:favorite', {detail: {tag: d.tag, favorited: !isFav}}));
        if (isFav && removedRecord) showUndoFavorite(d.tag, removedRecord);
      }).catch(() => {
        const status = el.querySelector('.lb-info');
        if (status) status.textContent = 'Could not save. Check connection and try again.';
      });
  }
  let undoTimer = null;
  let undoBtn = null;
  function showUndoFavorite(tag, record){
    if (undoBtn) { undoBtn.remove(); undoBtn = null; }
    clearTimeout(undoTimer);
    const status = el.querySelector('.lb-info');
    const original = status.textContent;
    status.textContent = 'Removed favorite. Undo?';
    undoBtn = document.createElement('button');
    undoBtn.textContent = 'Undo';
    undoBtn.onclick = () => {
      const body = Object.assign({}, record, {expedition: LB_EXPEDITION, leg: LB_LEG});
      fetch('/api/favorite', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})
        .then(r => { if (!r.ok) throw new Error('restore failed'); return r.json(); })
        .then(res => {
          if (res.error) throw new Error(res.error);
          favorites[tag] = record;
          render();
          document.dispatchEvent(new CustomEvent('lightbox:favorite', {detail: {tag, favorited: true}}));
          undoBtn.remove(); undoBtn = null;
          status.textContent = original;
        }).catch(() => {
          status.textContent = 'Could not restore favorite. Try again.';
        });
    };
    el.querySelector('.lb-actions').appendChild(undoBtn);
    undoTimer = setTimeout(() => { if (undoBtn) { undoBtn.remove(); undoBtn = null; } }, 10000);
  }
  function close(){ el.classList.remove('open'); }

  function open(tag, localTags){
    ensureDom();
    Promise.all([loadData(), loadFavorites(), loadCounterfactuals()]).then(() => {
      history = [];
      order = (localTags && localTags.length) ? localTags.map(t => byTag[t]).filter(Boolean) : DATA;
      const d = byTag[tag];
      if (!d) return;
      idx = order.indexOf(d);
      if (idx === -1) { order = [d]; idx = 0; }
      render();
      el.classList.add('open');
    });
  }

  window.Lightbox = { open };

  // Thumbnail grids mark their <img> tags with data-tag="<tag>" (no other wiring needed).
  // On hover or keyboard focus, fire an async request for that thumbnail's full-size image
  // too, so by the time it gets tapped the lightbox already has it cached and opens instantly
  // instead of showing its own loading spinner. Gated on hover/focus intent, not raw viewport
  // proximity: the previous IntersectionObserver-based version eagerly prefetched every
  // thumbnail within 150px of the viewport, which on a dense grid meant dozens of concurrent
  // 1-2.5MB full-res fetches from scrolling alone (gallery-archive-scale problem 3). A short
  // hover delay (150ms) avoids firing on a fast mouse pass-through; moving off or blurring
  // aborts an in-flight prefetch so bandwidth isn't wasted on images the user scrolled past.
  function wireThumbPrefetch(){
    const observed = new WeakSet();
    const hoverTimers = new Map();
    function start(tag){
      loadData().then(() => prefetchImage(byTag[tag])).catch(() => {});
    }
    function wireOne(img){
      if (observed.has(img)) return;
      observed.add(img);
      const tag = img.dataset.tag;
      if (!tag) return;
      img.addEventListener('mouseenter', () => {
        clearTimeout(hoverTimers.get(tag));
        hoverTimers.set(tag, setTimeout(() => start(tag), 150));
      });
      img.addEventListener('mouseleave', () => {
        clearTimeout(hoverTimers.get(tag));
        abortPrefetch(tag);
      });
      img.addEventListener('focus', () => start(tag));
      img.addEventListener('blur', () => abortPrefetch(tag));
    }
    function scan(){
      document.querySelectorAll('img[data-tag]').forEach(wireOne);
    }
    scan();
    // Grids re-render on filter changes / pagination / modal opens, so keep watching for
    // newly-inserted thumbnails rather than only scanning once at page load.
    new MutationObserver(scan).observe(document.body, {childList: true, subtree: true});
  }

  // Load scan_data.json eagerly (not just on first open()) so thumbnail prefetch can resolve
  // tag -> full-size file before the user ever opens the lightbox.
  loadData().catch(() => {});
  wireThumbPrefetch();

  // Any page can also be deep-linked as page.html?open=<tag> and it'll pop straight open.
  (function(){
    const params = new URLSearchParams(window.location.search);
    const tag = params.get('open');
    if (tag) open(tag);
  })();
})();
"""
