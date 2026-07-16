"""
Shows whether the preference classifier (search/preference_pairwise_model.py) is trained and
ready, and exposes the single persisted toggle (search/preference_settings.py) that both
archive.html and `clawmarks run allnight` read to decide whether to use its predictions. See
docs/superpowers/specs/2026-07-11-head-to-head-preference-design.md.

Served live at /preference_status.html by curation_server.py.
"""
import json
import os
from pathlib import Path

from clawmarks.search import embed_cache, preference_pairwise_model, preference_settings
from clawmarks.shared_ui import (
    BTN_CSS,
    DARK_TOKENS,
    INFOTIP_CSS,
    MOBILE_BASE_CSS,
    TOPNAV_CSS,
    info_btn,
    nav_bar_html,
)


def compute_data(sweep_dir):
    sweep_dir = Path(sweep_dir)
    comparisons_path = f"{sweep_dir}/user_comparisons.json"
    if os.path.exists(comparisons_path):
        with open(comparisons_path) as f:
            comparisons = json.load(f)
    else:
        comparisons = []
    n_comparisons = len(comparisons)

    # Loaded unconditionally (not just when a model already exists) so the gate below reflects
    # usable pairs after de-duplication, not the raw submission count: a manifest of 50 raw
    # submissions of the *same* pair now consolidates to 1 usable pair (see issue #13), and a
    # gate keyed on the raw count would tell the user they're "ready to train" right before the
    # actual retrain call refuses for the same data.
    tags, embeddings = embed_cache.load_cache(embed_cache.embeddings_file(sweep_dir))
    _, usable_y = preference_pairwise_model.build_training_set(tags, embeddings, comparisons)
    n_usable = len(usable_y) // 2

    if n_usable < preference_pairwise_model.MIN_COMPARISONS:
        gate_message = (f"only {n_usable} usable comparisons of {n_comparisons} total (need "
                         f"{preference_pairwise_model.MIN_COMPARISONS}); compare more images "
                         f"via compare.html.")
    else:
        gate_message = ""

    model_path = preference_pairwise_model.model_file(sweep_dir)
    model_meta_path = preference_pairwise_model.model_meta_file(sweep_dir)
    has_model = os.path.exists(model_path)
    model_meta = None
    if has_model and os.path.exists(model_meta_path):
        with open(model_meta_path) as f:
            model_meta = json.load(f)

    new_comparisons_since_train = 0
    comparisons_changed_since_train = False
    if model_meta:
        n_usable_at_train = model_meta.get("n_usable_comparisons", model_meta["n_comparisons"])
        new_comparisons_since_train = max(0, n_usable - n_usable_at_train)
        if "comparisons_fingerprint" in model_meta:
            current_fingerprint = preference_pairwise_model.comparisons_fingerprint(tags, embeddings, comparisons)
            comparisons_changed_since_train = current_fingerprint != model_meta["comparisons_fingerprint"]
        else:
            # Model trained before comparisons_fingerprint existed: fall back to a plain count
            # comparison, which misses a swapped comparison but is still better than nothing.
            comparisons_changed_since_train = new_comparisons_since_train > 0

    return {
        "n_comparisons": n_comparisons,
        "n_usable": n_usable,
        "min_comparisons": preference_pairwise_model.MIN_COMPARISONS,
        "comparisons_gate_message": gate_message,
        "has_model": has_model,
        "model_meta": model_meta,
        "new_comparisons_since_train": new_comparisons_since_train,
        "comparisons_changed_since_train": comparisons_changed_since_train,
        "use_predicted_preference": preference_settings.load(sweep_dir)["use_predicted_preference"],
    }


def render_html(data, active_expedition=None, active_leg=None, running=None):
    gate_html = (f'<p class="gate">{data["comparisons_gate_message"]}</p>'
                 if data["comparisons_gate_message"] else '<p class="gate ok">ready to train.</p>')

    if data["model_meta"]:
        m = data["model_meta"]
        stats_rows = ""
        if "p_value" in m:
            p_interpretation = ("p &lt; 0.05: unlikely to be chance" if m["p_value"] < 0.05
                                else "p &gt;= 0.05: not distinguishable from chance")
            stats_rows = (f'<tr><td>majority-class baseline accuracy</td><td>{m["baseline_accuracy"]:.1%}</td></tr>'
                          f'<tr><td>permutation p-value</td><td>{m["p_value"]:.4f} '
                          f'<span class="interpretation">{p_interpretation}</span></td></tr>')
        meta_html = (f'<table class="meta"><tr><td>trained</td><td>{m["trained_at"]}</td></tr>'
                     f'<tr><td>comparisons used</td><td>{m["n_comparisons"]}</td></tr>'
                     f'<tr><td>cross-validated accuracy</td><td>{m["cv_accuracy"]}</td></tr>'
                     f'{stats_rows}</table>')
    else:
        meta_html = ('<p class="meta-empty">no model trained yet. Once enough comparisons exist, run '
                     '<code>python -m clawmarks.search.preference_pairwise_model</code>.</p>')

    staleness_html = ""
    if data["comparisons_changed_since_train"]:
        trained_at = data["model_meta"]["trained_at"]
        if data["new_comparisons_since_train"] > 0:
            staleness_html = (f'<p class="stale">{data["new_comparisons_since_train"]} new comparisons since last '
                              f'train ({trained_at}). Retrain to include them.</p>')
        else:
            staleness_html = (f'<p class="stale">comparisons have changed since last train ({trained_at}). '
                              f'Retrain to include them.</p>')

    disabled_attr = "" if data["has_model"] else "disabled"
    checked_attr = "checked" if data["use_predicted_preference"] else ""

    toggle_tip = info_btn(
        "When on, archive.html's fallback champion per MAP-Elites cell and the next "
        "`clawmarks run allnight`'s exploit pool both use this trained model's predicted "
        "preference instead of raw novelty / favorited images. Off by default; only turn this "
        "on after eyeballing preference_rank.html against your own taste."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS preference status</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
{DARK_TOKENS}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
{BTN_CSS}
h1 {{ font-size:18px; margin:0 0 4px; }}
p.sub {{ color:var(--text-dim); max-width:760px; font-size:13px; line-height:1.6; }}
.panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; margin-top:16px; max-width:520px; }}
p.gate {{ color:#e0a030; }}
p.gate.ok {{ color:#5fbf6f; }}
p.stale {{ color:#e0a030; border:1px solid #6b4c16; border-radius:8px; padding:8px; }}
table.meta {{ font-size:13px; border-collapse:collapse; }}
table.meta td {{ padding:3px 10px 3px 0; color:var(--text-dim); }}
table.meta td:first-child {{ color:var(--text); }}
.interpretation {{ color:var(--text-dim); margin-left:6px; }}
.toggle-row {{ margin-top:14px; display:flex; flex-wrap:wrap; align-items:center; gap:8px; }}
.secondary {{ background:var(--panel-2); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:4px 8px; cursor:pointer; }}
#toggle-status, #retrain-status {{ font-size:12px; color:var(--text-dim); margin-left:8px; }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('preference_status.html', active_expedition=active_expedition, active_leg=active_leg, running=running)}
<h1>Preference classifier status</h1>
<p class="sub">Comparisons: {data["n_usable"]} usable of {data["n_comparisons"]} total (needs {data["min_comparisons"]}).</p>
<p class="sub"><a href="compare.html">Compare more images</a> or <a href="preference_rank.html">review the ranking</a>.</p>
<div class="panel">
{gate_html}
{staleness_html}
{meta_html}
<div class="toggle-row">
<label><input type="checkbox" id="toggle" {checked_attr} {disabled_attr} onchange="toggle(this.checked)"> use predicted preference{toggle_tip}</label>
<button class="secondary" id="retrain" onclick="retrain()">Retrain now</button>
<span id="toggle-status"></span>
<span id="retrain-status"></span>
</div>
</div>
<script>
function toggle(enabled) {{
  const status = document.getElementById('toggle-status');
  status.textContent = 'saving...';
  fetch('/api/preference_toggle', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{enabled: enabled}}),
  }}).then(r => r.json()).then(data => {{
    if (data.error) {{
      status.textContent = data.error;
      document.getElementById('toggle').checked = !enabled;
    }} else {{
      status.textContent = 'saved.';
    }}
  }});
}}
function retrain() {{
  const button = document.getElementById('retrain');
  const status = document.getElementById('retrain-status');
  button.disabled = true;
  button.textContent = 'Retraining…';
  status.textContent = '';
  fetch('/api/preference_retrain', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{}}),
  }}).then(r => r.json()).then(data => {{
    if (data.error) {{
      status.textContent = data.error;
      button.disabled = false;
      button.textContent = 'Retrain now';
    }} else {{
      location.reload();
    }}
  }}).catch(e => {{
    status.textContent = e.toString();
    button.disabled = false;
    button.textContent = 'Retrain now';
  }});
}}
</script>
<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
</body></html>"""
    return html
