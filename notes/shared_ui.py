"""
Shared UI pieces used by every notes/build_*.py tool-page generator, so the lightbox, the
top navigation bar, and its scroll-to-hide behavior are defined once instead of duplicated
across 8 scripts. Import and use:

    from shared_ui import write_lightbox_asset, nav_bar_html, TOPNAV_CSS, SCROLLNAV_JS

`write_lightbox_asset(sweep_dir)` copies the static lightbox.js module into the sweep
directory (idempotent, safe to call from every builder). Every generated page includes it
with `<script src="lightbox.js"></script>` and opens images via `Lightbox.open(tag)` instead
of `window.open('scan.html?open=...')`: no new tab, no page load, works from any page
because the module fetches notes/uncanny_sweep/scan_data.json itself.
"""
import os

NAV_OPTIONS = [
    ("explore.html", "all tools (hub)"),
    ("scan.html", "scan gallery"),
    ("map.html", "solution map (UMAP)"),
    ("coverage.html", "coverage / void map"),
    ("archive.html", "elite archive"),
    ("redundancy.html", "redundancy clusters"),
    ("novelty_decay.html", "novelty decay watchlist"),
    ("lineage.html", "lineage tree"),
    ("gallery.html", "binned atlas (original)"),
]


def nav_bar_html(current):
    opts = "".join(
        f'<option value="{href}"{" selected" if href == current else ""}>{label}</option>'
        for href, label in NAV_OPTIONS
    )
    return (
        '<div id="topnav" class="topnav" data-autohide>'
        '<a class="navlink" href="explore.html">&larr; all tools</a>'
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
@media (max-width: 640px) {
  .topnav { padding:8px 10px; gap:8px; font-size:12px; }
  .topnav select { flex:1; min-width:0; max-width:none; }
}
"""

_infotip_counter = 0


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


def write_infotip_asset(sweep_dir):
    with open(os.path.join(sweep_dir, "infotip.js"), "w") as f:
        f.write(INFOTIP_JS)


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
  let DATA = null, byTag = null;
  let order = [];
  let idx = -1;
  let history = [];
  let picks = {};
  let favorites = {};
  let counterfactuals = {};
  let el = null;

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
#lb-overlay .lb-nav { position:absolute; top:0; bottom:0; width:16%; cursor:pointer; z-index:1; }
#lb-overlay .lb-prev { left:0; } #lb-overlay .lb-next { right:0; }
#lb-overlay .lb-close { position:absolute; top:16px; right:22px; font-size:28px; cursor:pointer; color:#9a9aa4;
  width:40px; height:40px; display:flex; align-items:center; justify-content:center; z-index:2; }
#lb-overlay .lb-close:hover { color:#eaeaee; }
#lb-overlay .lb-actions { display:flex; gap:10px; align-items:center; flex-wrap:wrap; justify-content:center; }
#lb-overlay button { background:#1d1d22; color:#eaeaee; border:1px solid #2a2a30; border-radius:7px;
  padding:8px 16px; font-size:13px; cursor:pointer; }
#lb-overlay button.picked { background:rgba(245,197,66,0.16); border-color:#f5c542; color:#f5c542; }
#lb-overlay button.favorited { background:rgba(224,96,150,0.16); border-color:#e0609a; color:#e0609a; }
#lb-overlay .lb-actions .infobtn { background:rgba(255,255,255,0.12); border-color:rgba(255,255,255,0.3); color:#dcdce2; }
#lb-overlay .lb-simlabel { font-size:11px; color:#6a6a74; letter-spacing:0.02em; text-transform:uppercase; }
#lb-overlay .lb-simstrip { display:flex; gap:7px; overflow-x:auto; max-width:92vw; padding:4px 2px 8px; }
#lb-overlay .lb-simstrip img { width:64px; height:64px; object-fit:cover; border-radius:6px; cursor:pointer;
  flex-shrink:0; opacity:0.7; outline:2px solid transparent; }
#lb-overlay .lb-simstrip img:hover { opacity:1; outline-color:#7c9eff; }
#lb-overlay .lb-cf-panel { display:none; background:rgba(255,255,255,0.05); border:1px solid #2a2a30;
  border-radius:8px; padding:12px 14px; max-width:92vw; width:520px; text-align:left; }
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
#lb-overlay .lb-cf-result { margin-top:10px; display:none; }
#lb-overlay .lb-cf-result.open { display:block; }
#lb-overlay .lb-cf-result img { max-width:100%; border-radius:6px; cursor:pointer; }
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
    <button class="lb-pick">&#9733; pick as winner</button>
    <span class="infobtn" data-id="lb-tip-pick" data-tip="Picking marks this image as a human-approved success. The next search generation uses picked images as starting points for new variations, ahead of the algorithm's own ranking: it's how your judgment steers where the search goes next.">?</span>
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
    </div>
    <button class="lb-cf-submit">Generate</button>
    <div class="lb-cf-status"></div>
    <div class="lb-cf-result"><img class="lb-cf-result-img"></div>
    <div class="lb-cf-list"></div>
  </div>
  <div class="lb-simlabel">similar images (by DINOv2 embedding)</div>
  <div class="lb-simstrip"></div>`;
    document.body.appendChild(el);

    const mainImg = el.querySelector('.lb-main');
    const imgWrap = el.querySelector('.lb-imgwrap');
    mainImg.addEventListener('load', () => imgWrap.classList.remove('loading'));
    mainImg.addEventListener('error', () => imgWrap.classList.remove('loading'));

    el.querySelector('.lb-close').onclick = close;
    el.querySelector('.lb-prev').onclick = () => step(-1);
    el.querySelector('.lb-next').onclick = () => step(1);
    el.querySelector('.lb-back').onclick = back;
    el.querySelector('.lb-pick').onclick = togglePick;
    el.querySelector('.lb-favorite').onclick = toggleFavorite;
    el.querySelector('.lb-cf-toggle').onclick = toggleCfPanel;
    el.querySelector('.lb-cf-submit').onclick = submitCounterfactual;
    el.addEventListener('click', e => { if (e.target === el) close(); });
    document.addEventListener('keydown', e => {
      if (!el.classList.contains('open')) return;
      if (e.key === 'ArrowRight') step(1);
      if (e.key === 'ArrowLeft') step(-1);
      if (e.key === 'Escape') close();
      if (e.key === ' ') { e.preventDefault(); togglePick(); }
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
  function loadPicks(){
    return fetch('/api/picks').then(r => r.json()).then(p => { picks = p; }).catch(() => {});
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
    mainImg.src = d.file;
    prefetchNeighbors();
    el.querySelector('.lb-info').textContent =
      `${d.tag} | gen ${d.gen} | ${d.category} | type=${d.prompt_type} | prompt=${d.prompt_name} | ` +
      `strength=${d.strength} cfg=${d.cfg} | faith=${d.faith} novelty=${d.novelty}`;
    const isPicked = !!picks[d.tag];
    const pickBtn = el.querySelector('.lb-pick');
    pickBtn.textContent = isPicked ? '★ picked (click to unpick)' : '☆ pick as winner';
    pickBtn.classList.toggle('picked', isPicked);
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
        return `<img loading="lazy" src="${n.thumb}" title="f=${n.faith} n=${n.novelty} ${n.prompt_name}" data-tag="${t}">`;
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
    el.querySelector('.lb-cf-status').textContent = '';
    el.querySelector('.lb-cf-status').classList.remove('err');
    renderCfList(d);
  }

  function renderCfList(d){
    const list = el.querySelector('.lb-cf-list');
    const mine = Object.values(counterfactuals).filter(c => c.origin_tag === d.tag);
    if (!mine.length) { list.innerHTML = ''; return; }
    list.innerHTML = mine.map(c =>
      `<img loading="lazy" src="${c.file}" title="s=${c.strength} cfg=${c.cfg} seed=${c.seed}">`
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
      el.querySelector('.lb-cf-status').textContent = '';
      el.querySelector('.lb-cf-status').classList.remove('err');
      el.querySelector('.lb-cf-result').classList.remove('open');
    }
  }

  function submitCounterfactual(){
    const d = order[idx];
    const prompt = el.querySelector('.lb-cf-prompt').value.trim();
    const strength = parseFloat(el.querySelector('.lb-cf-strength').value);
    const cfg = parseFloat(el.querySelector('.lb-cf-cfg').value);
    const seedRaw = el.querySelector('.lb-cf-seed').value.trim();
    const overridden = [];
    if (prompt !== d.prompt) overridden.push('prompt');
    if (strength !== d.strength) overridden.push('strength');
    if (cfg !== d.cfg) overridden.push('cfg');
    if (seedRaw) overridden.push('seed');

    const body = {
      origin_tag: d.tag, prompt, strength, cfg,
      seed: seedRaw ? parseInt(seedRaw, 10) : null,
      steps: d.steps, sampler: d.sampler, negative: d.negative,
      overridden,
    };

    const submitBtn = el.querySelector('.lb-cf-submit');
    const status = el.querySelector('.lb-cf-status');
    submitBtn.disabled = true;
    status.classList.remove('err');
    status.textContent = 'Generating... a few seconds if the endpoint is already warm, up to 5 minutes if it has to cold-start a worker. Keep this tab open.';

    fetch('/api/counterfactual', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
      .then(r => r.json().then(data => ({ok: r.ok, data})))
      .then(({ok, data}) => {
        submitBtn.disabled = false;
        if (!ok || data.error) {
          status.classList.add('err');
          status.textContent = data.error || 'generation failed';
          return;
        }
        status.textContent = 'Done.';
        counterfactuals[data.tag] = data;
        const resultImg = el.querySelector('.lb-cf-result-img');
        resultImg.src = data.file;
        el.querySelector('.lb-cf-result').classList.add('open');
        renderCfList(d);
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
  function togglePick(){
    const d = order[idx];
    const isPicked = !!picks[d.tag];
    const endpoint = isPicked ? '/api/unpick' : '/api/pick';
    const body = isPicked ? {tag: d.tag} : Object.assign({}, d);
    fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})
      .then(r => r.json())
      .then(() => {
        if (isPicked) delete picks[d.tag]; else picks[d.tag] = body;
        render();
        document.dispatchEvent(new CustomEvent('lightbox:pick', {detail: {tag: d.tag, picked: !isPicked}}));
      });
  }
  function toggleFavorite(){
    const d = order[idx];
    const isFav = !!favorites[d.tag];
    const endpoint = isFav ? '/api/unfavorite' : '/api/favorite';
    const body = isFav ? {tag: d.tag} : Object.assign({}, d);
    fetch(endpoint, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})
      .then(r => r.json())
      .then(() => {
        if (isFav) delete favorites[d.tag]; else favorites[d.tag] = body;
        render();
        document.dispatchEvent(new CustomEvent('lightbox:favorite', {detail: {tag: d.tag, favorited: !isFav}}));
      });
  }
  function close(){ el.classList.remove('open'); }

  function open(tag, localTags){
    ensureDom();
    Promise.all([loadData(), loadPicks(), loadFavorites(), loadCounterfactuals()]).then(() => {
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
  // As each thumbnail scrolls into view, fire an async request for its full-size image too,
  // so by the time a visible thumbnail gets tapped the lightbox already has it cached and
  // opens instantly instead of showing its own loading spinner. Gated on visibility (not
  // "every thumbnail on the page") since a filtered grid can hold thousands of entries and
  // downloading all of their full-size files up front would be wasteful. Unlike a one-shot
  // "load once visible" observer, this one keeps watching every thumbnail for as long as it
  // stays in the DOM: scrolling a still-loading image off-screen aborts its prefetch so the
  // bandwidth goes to whatever's actually on screen, and scrolling back re-starts it.
  function wireThumbPrefetch(){
    if (!('IntersectionObserver' in window)) return;
    const observed = new WeakSet();
    const io = new IntersectionObserver(entries => {
      entries.forEach(en => {
        const tag = en.target.dataset.tag;
        if (!tag) return;
        if (en.isIntersecting) {
          loadData().then(() => prefetchImage(byTag[tag])).catch(() => {});
        } else {
          abortPrefetch(tag);
        }
      });
    }, {rootMargin: '150px'});
    function scan(){
      document.querySelectorAll('img[data-tag]').forEach(img => {
        if (observed.has(img)) return;
        observed.add(img);
        io.observe(img);
      });
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


def write_lightbox_asset(sweep_dir):
    with open(os.path.join(sweep_dir, "lightbox.js"), "w") as f:
        f.write(_LIGHTBOX_JS)


def write_scrollnav_asset(sweep_dir):
    with open(os.path.join(sweep_dir, "scrollnav.js"), "w") as f:
        f.write(SCROLLNAV_JS)
