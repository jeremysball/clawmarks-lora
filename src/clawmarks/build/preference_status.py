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


def render_html(data, active_expedition=None, active_leg=None, running=None, focus=None):
    gate_html = (f'<p class="gate">{data["comparisons_gate_message"]}</p>'
                 if data["comparisons_gate_message"] else '<p class="gate ok">ready to train.</p>')

    if data["model_meta"]:
        m = data["model_meta"]
        stats_rows = ""
        if "p_value" in m:
            p_interpretation = ("p &lt; 0.05: unlikely to be chance" if m["p_value"] < 0.05
                                else "p &gt;= 0.05: not distinguishable from chance")
            stats_rows = (f'<div class="evidence-row"><span class="label">majority-class baseline accuracy</span>'
                          f'<span class="value">{m["baseline_accuracy"]:.1%}</span></div>'
                          f'<div class="evidence-row"><span class="label">permutation p-value</span>'
                          f'<span class="value">{m["p_value"]:.4f} '
                          f'<span class="interpretation">{p_interpretation}</span></span></div>')
        meta_html = (f'<div class="evidence-row"><span class="label">trained</span>'
                     f'<span class="value">{m["trained_at"]}</span></div>'
                     f'<div class="evidence-row"><span class="label">comparisons used</span>'
                     f'<span class="value">{m["n_comparisons"]}</span></div>'
                     f'<div class="evidence-row"><span class="label">cross-validated accuracy</span>'
                     f'<span class="value">{m["cv_accuracy"]}</span></div>'
                     f'{stats_rows}')
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
{SULFUR_FONT_CSS}
{SULFUR_CSS}
{CONTROL_CSS}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
body {{ margin:0; padding:24px; }}
h1 {{ font-size:22px; margin:24px 0 4px; letter-spacing:0.02em; text-transform:uppercase; }}
p.sub {{ color:var(--text-soft); max-width:760px; font-size:13px; line-height:1.6;
  padding-bottom:14px; border-bottom:1px solid var(--rule); }}
.readiness {{ padding:10px 0; border-bottom:1px solid var(--rule); margin-top:16px;
  max-width:560px; font-size:13px; }}
.readiness p {{ margin:0; padding:2px 0; }}
p.gate {{ color:#a84820; }}
p.gate.ok {{ color:#3d6a26; }}
p.stale {{ color:#a84820; padding:2px 0; }}
.evidence {{ max-width:560px; margin-top:14px; padding:0; }}
.evidence-row {{ display:flex; gap:16px; padding:8px 0; border-bottom:1px solid var(--rule);
  font-size:13px; align-items:baseline; }}
.evidence-row:last-child {{ border-bottom:none; }}
.evidence-row .label {{ color:var(--text-soft); min-width:200px; flex-shrink:0; }}
.evidence-row .value {{ color:var(--ink); font-family:var(--font-mono); }}
.interpretation {{ color:var(--text-soft); margin-left:6px; font-family:var(--font-body); }}
.meta-empty {{ color:var(--text-soft); font-size:13px; padding:8px 0;
  border-bottom:1px solid var(--rule); }}
.toggle-row {{ margin-top:18px; display:flex; flex-wrap:wrap; align-items:center; gap:10px;
  font-size:13px; max-width:560px; }}
.toggle-row label {{ display:inline-flex; align-items:center; gap:6px; }}
.secondary {{ background:var(--panel-2); color:var(--text); border:1px solid var(--border);
  padding:5px 10px; cursor:pointer; font:600 12.5px/1 var(--font-body); }}
#toggle-status, #retrain-status {{ font-size:12px; color:var(--text-soft); margin-left:6px; }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('preference_status.html', active_expedition=active_expedition, active_leg=active_leg, running=running, focus=focus)}
<h1>Preference classifier status</h1>
<p class="sub">Comparisons: {data["n_usable"]} usable of {data["n_comparisons"]} total (needs {data["min_comparisons"]}). <a href="{scoped_href('/compare.html', active_expedition, active_leg, focus)}">Compare more images</a> or <a href="{scoped_href('/preference_rank.html', active_expedition, active_leg, focus)}">review the ranking</a>.</p>
<div class="readiness">
{gate_html}
{staleness_html}
</div>
{("" if not data["model_meta"] else "<div class='evidence'>") + meta_html + ("</div>" if data["model_meta"] else "")}
<div class="toggle-row">
<label><input type="checkbox" id="toggle" {checked_attr} {disabled_attr} onchange="toggle(this.checked)"> use predicted preference{toggle_tip}</label>
<button class="secondary" id="retrain" onclick="retrain()">Retrain now</button>
<span id="toggle-status"></span>
<span id="retrain-status"></span>
</div>
<script>
function toggle(enabled) {{
  const status = document.getElementById('toggle-status');
  status.textContent = 'saving...';
  const scope = new URLSearchParams(location.search);
  fetch('/api/preference_toggle', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{enabled: enabled, expedition: scope.get('expedition'), leg: scope.get('leg')}}),
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
  const scope = new URLSearchParams(location.search);
  fetch('/api/preference_retrain', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{expedition: scope.get('expedition'), leg: scope.get('leg')}}),
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
<script src="/shared-ui.js"></script>
</body></html>"""
    return html
