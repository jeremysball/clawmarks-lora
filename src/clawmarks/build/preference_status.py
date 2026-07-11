"""
Shows whether the preference classifier (search/preference_model.py) is trained and ready, and
exposes the single persisted toggle (search/preference_settings.py) that both archive.html and
`clawmarks run allnight` read to decide whether to use its predictions. See
docs/superpowers/specs/2026-07-10-preference-toggle-design.md.

Served live at /preference_status.html by curation_server.py.
"""
import json
import os

from clawmarks.search import preference_model, preference_settings
from clawmarks.shared_ui import INFOTIP_CSS, MOBILE_BASE_CSS, TOPNAV_CSS, info_btn, nav_bar_html


def compute_data(sweep_dir):
    ratings_path = f"{sweep_dir}/user_ratings.json"
    if os.path.exists(ratings_path):
        with open(ratings_path) as f:
            ratings = json.load(f)
    else:
        ratings = {}
    n_yes = sum(1 for r in ratings.values() if r.get("label") == "yes")
    n_no = sum(1 for r in ratings.values() if r.get("label") == "no")
    n_total = n_yes + n_no

    if n_total < preference_model.MIN_LABELS:
        gate_message = (f"only {n_total} labels (need {preference_model.MIN_LABELS}); "
                         f"rate more images via rate.html.")
    else:
        import numpy as np
        y = np.array([1] * n_yes + [0] * n_no, dtype=np.int64)
        gate_message = preference_model.class_balance_error(y)

    has_model = os.path.exists(preference_model.MODEL_FILE)
    model_meta = None
    if has_model and os.path.exists(preference_model.MODEL_META_FILE):
        with open(preference_model.MODEL_META_FILE) as f:
            model_meta = json.load(f)
    new_labels_since_train = max(0, n_total - model_meta["n_labels"]) if model_meta else 0

    return {
        "n_yes": n_yes, "n_no": n_no, "n_total": n_total,
        "min_labels": preference_model.MIN_LABELS,
        "labels_gate_message": gate_message,
        "has_model": has_model,
        "model_meta": model_meta,
        "new_labels_since_train": new_labels_since_train,
        "use_predicted_preference": preference_settings.load()["use_predicted_preference"],
    }


def render_html(data):
    gate_html = (f'<p class="gate">{data["labels_gate_message"]}</p>'
                 if data["labels_gate_message"] else '<p class="gate ok">ready to train.</p>')

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
                     f'<tr><td>labels used</td><td>{m["n_labels"]} ({m["n_yes"]} yes / {m["n_no"]} no)</td></tr>'
                     f'<tr><td>cross-validated accuracy</td><td>{m["cv_accuracy"]}</td></tr>'
                     f'{stats_rows}</table>')
    else:
        meta_html = (f'<p class="meta-empty">no model trained yet. Once enough labels exist, run '
                     f'<code>python -m clawmarks.search.preference_model</code>.</p>')

    staleness_html = ""
    if data["new_labels_since_train"] > 0:
        trained_at = data["model_meta"]["trained_at"]
        staleness_html = (f'<p class="stale">{data["new_labels_since_train"]} new ratings since last train '
                          f'({trained_at}) - retrain to include them.</p>')

    disabled_attr = "" if data["has_model"] else "disabled"
    checked_attr = "checked" if data["use_predicted_preference"] else ""

    toggle_tip = info_btn(
        "When on, archive.html's fallback champion per MAP-Elites cell and the next "
        "`clawmarks run allnight`'s exploit pool both use this trained model's predicted "
        "preference instead of raw novelty / yes-rated images. Off by default; only turn this "
        "on after eyeballing preference_rank.html against your own taste."
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CLAWMARKS preference status</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; --bg:#0b0b0d; --panel:#16161a; --border:#2a2a30; --text:#eaeaee; --text-dim:#9a9aa4; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,sans-serif; margin:0; padding:24px; }}
{TOPNAV_CSS}
{MOBILE_BASE_CSS}
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
.toggle-row {{ margin-top:14px; display:flex; align-items:center; gap:8px; }}
.secondary {{ background:#24242a; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:4px 8px; cursor:pointer; }}
#toggle-status, #retrain-status {{ font-size:12px; color:var(--text-dim); margin-left:8px; }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('preference_status.html')}
<h1>Preference classifier status</h1>
<p class="sub">Labels: {data["n_yes"]} yes / {data["n_no"]} no ({data["n_total"]} total, needs {data["min_labels"]}).</p>
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
