# Preference Classifier Status and Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the web UI one place to see preference-classifier readiness (label counts,
training gate, model metadata) and one shared toggle that both `archive.html` and
`clawmarks run allnight` read, replacing the undiscoverable query param and CLI flag.

**Architecture:** A new persisted JSON setting (`preference_settings.json`) is the single
source of truth for the toggle. A new `build/preference_status.py` view module +
`preference_status.html` route surfaces readiness and the toggle UI. `preference_model.py`
gains a metadata sidecar so the status page can show real numbers without recomputing
cross-validation. `archive.html`'s route and `cli.py`'s `run allnight` both read the persisted
setting instead of their current independent controls.

**Tech Stack:** Python stdlib `http.server` (existing `curation_server.py`), no new
dependencies.

## Global Constraints

- No em dashes (`—`) or ` -- ` anywhere in code, comments, docstrings, or commit messages.
  Grep the diff for both before every commit.
- Conventional Commits format for every commit message.
- Tests run via: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/ -v`
- Every new on-disk write uses the atomic tmp-then-`os.replace` pattern already used by
  `search/embed_cache.py:save_cache` and `curation_server.py:save_store` (write to
  `f"{path}.tmp"`, then `os.replace(tmp, path)`).
- Follow existing module shape exactly: `build/*.py` view modules export `compute_data(sweep_dir)`
  and `render_html(data)`; routes in `curation_server.py` go through `_live_cache.get(...)`,
  never call a view module's `compute_data` directly from a route handler.
- Design doc: `docs/superpowers/specs/2026-07-10-preference-toggle-design.md`. Read it before
  Task 4 if anything below is ambiguous; this plan implements it exactly.

---

### Task 1: `search/preference_settings.py` + `config.py` setting path

**Files:**
- Create: `src/clawmarks/search/preference_settings.py`
- Modify: `src/clawmarks/config.py`
- Test: `tests/test_preference_settings.py`

**Interfaces:**
- Consumes: `clawmarks.config.PREFERENCE_SETTINGS_FILE` (new).
- Produces: `load() -> {"use_predicted_preference": bool}`, `save(enabled: bool) -> None`. Later
  tasks (4, 5, 6) call both.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preference_settings.py
import json

from clawmarks.search import preference_settings


def test_load_returns_false_default_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    assert preference_settings.load() == {"use_predicted_preference": False}


def test_save_then_load_round_trips_true(tmp_path, monkeypatch):
    path = tmp_path / "preference_settings.json"
    monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", path)
    preference_settings.save(True)
    assert preference_settings.load() == {"use_predicted_preference": True}


def test_save_writes_atomically_no_tmp_file_left_behind(tmp_path, monkeypatch):
    path = tmp_path / "preference_settings.json"
    monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", path)
    preference_settings.save(True)
    assert not (tmp_path / "preference_settings.json.tmp").exists()
    assert json.loads(path.read_text()) == {"use_predicted_preference": True}


def test_save_false_then_load_round_trips_false(tmp_path, monkeypatch):
    path = tmp_path / "preference_settings.json"
    monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", path)
    preference_settings.save(True)
    preference_settings.save(False)
    assert preference_settings.load() == {"use_predicted_preference": False}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_preference_settings.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'clawmarks.search.preference_settings'`)

- [ ] **Step 3: Add the config path**

In `src/clawmarks/config.py`, immediately after the existing `USER_RATINGS_FILE = SWEEP_DIR / "user_ratings.json"` line, add:

```python
PREFERENCE_SETTINGS_FILE = SWEEP_DIR / "preference_settings.json"
```

- [ ] **Step 4: Write the module**

```python
# src/clawmarks/search/preference_settings.py
"""
Single persisted setting shared by archive.html's rendering and `clawmarks run allnight`'s
exploit-pool source, so flipping predicted-preference on or off happens in one place instead
of two independent controls (a query param and a CLI flag). See
docs/superpowers/specs/2026-07-10-preference-toggle-design.md.
"""
import json
import os

from clawmarks.config import PREFERENCE_SETTINGS_FILE


def load():
    """Returns {"use_predicted_preference": bool}. Missing file means the default, False."""
    if not os.path.exists(PREFERENCE_SETTINGS_FILE):
        return {"use_predicted_preference": False}
    with open(PREFERENCE_SETTINGS_FILE) as f:
        return json.load(f)


def save(enabled):
    tmp = f"{PREFERENCE_SETTINGS_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump({"use_predicted_preference": bool(enabled)}, f)
    os.replace(tmp, PREFERENCE_SETTINGS_FILE)
```

Note: `load()`/`save()` read the module-level `PREFERENCE_SETTINGS_FILE` name at call time (not
a captured default argument), so the tests above can `monkeypatch.setattr(preference_settings,
"PREFERENCE_SETTINGS_FILE", ...)` and have both functions pick it up.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_preference_settings.py -v`
Expected: PASS (4/4)

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/search/preference_settings.py src/clawmarks/config.py tests/test_preference_settings.py
git commit -m "feat(clawmarks): add persisted preference-toggle setting"
```

---

### Task 2: `preference_model.py` writes a metadata sidecar

**Files:**
- Modify: `src/clawmarks/search/preference_model.py`
- Test: `tests/test_preference_model.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `MODEL_META_FILE` path constant and a metadata JSON file written by `main()` on
  every successful train. Task 4's `build/preference_status.py` reads this file.

**Context:** `main()` (lines 89-116 as of this plan's writing; locate by the `def main(argv=None):`
signature and the `joblib.dump(model, MODEL_FILE)` line if line numbers have drifted) currently
computes `acc` via `cross_validate(X, y)`, prints it, then discards it after writing the model.
This task adds a sidecar write right after `joblib.dump`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_preference_model.py
import json

import numpy as np


def test_main_writes_metadata_sidecar_on_successful_train(tmp_path, monkeypatch):
    from clawmarks.search import embed_cache

    rng = np.random.RandomState(0)
    yes_cluster = rng.normal(loc=5.0, scale=0.1, size=(30, 2))
    no_cluster = rng.normal(loc=-5.0, scale=0.1, size=(30, 2))
    embeddings = np.vstack([yes_cluster, no_cluster]).astype(np.float32)
    tags = [f"t{i}" for i in range(60)]
    embed_cache.save_cache(tmp_path / "embeddings.npz", tags, embeddings)

    ratings = {tags[i]: {"label": "yes" if i < 30 else "no", "rated_at": "t"} for i in range(60)}
    (tmp_path / "user_ratings.json").write_text(json.dumps(ratings))

    monkeypatch.setattr(preference_model, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(preference_model.embed_cache, "EMBEDDINGS_FILE", tmp_path / "embeddings.npz")
    monkeypatch.setattr(preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    monkeypatch.setattr(preference_model, "MODEL_META_FILE", tmp_path / "preference_model_meta.json")

    rc = preference_model.main([])
    assert rc == 0

    meta = json.loads((tmp_path / "preference_model_meta.json").read_text())
    assert meta["n_labels"] == 60
    assert meta["n_yes"] == 30
    assert meta["n_no"] == 30
    assert 0.0 <= meta["cv_accuracy"] <= 1.0
    assert "trained_at" in meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_preference_model.py::test_main_writes_metadata_sidecar_on_successful_train -v`
Expected: FAIL (`AttributeError: module 'clawmarks.search.preference_model' has no attribute 'MODEL_META_FILE'`)

- [ ] **Step 3: Implement**

In `src/clawmarks/search/preference_model.py`:

Add near the top, after `import sys`:

```python
from datetime import datetime, timezone
```

Add next to the existing `MODEL_FILE = SWEEP_DIR / "preference_model.joblib"` line:

```python
MODEL_META_FILE = SWEEP_DIR / "preference_model_meta.json"
```

In `main()`, replace:

```python
    model = train(X, y)
    joblib.dump(model, MODEL_FILE)
    print(f"wrote {MODEL_FILE}", flush=True)
    return 0
```

with:

```python
    model = train(X, y)
    joblib.dump(model, MODEL_FILE)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_labels": len(y),
        "n_yes": int(y.sum()),
        "n_no": len(y) - int(y.sum()),
        "cv_accuracy": round(acc, 4),
    }
    tmp = f"{MODEL_META_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, MODEL_META_FILE)
    print(f"wrote {MODEL_FILE} and {MODEL_META_FILE}", flush=True)
    return 0
```

`os` is not currently imported in this file; add `import os` alongside the existing `import
json` / `import sys` lines at the top.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_preference_model.py -v`
Expected: PASS (all tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/search/preference_model.py tests/test_preference_model.py
git commit -m "feat(clawmarks): write preference model metadata sidecar on train"
```

---

### Task 3: `shared_ui.py` nav entry for the new page

**Files:**
- Modify: `src/clawmarks/shared_ui.py`
- Test: `tests/test_shared_ui.py` (create if it doesn't exist; check first with
  `fd -a test_shared_ui tests` since it may already exist for other nav assertions)

**Interfaces:**
- Consumes: nothing.
- Produces: `"preference_status.html"` present in `shared_ui.NAV_OPTIONS`, so every existing
  page's nav dropdown (rendered via `nav_bar_html`) picks it up automatically. Task 4's
  `render_html` calls `nav_bar_html('preference_status.html')`.

- [ ] **Step 1: Check whether a nav test already exists**

Run: `fd -a test_shared_ui tests`

If it exists, read it and add a test in the same style as Step 2 below instead of creating a
new file. If it doesn't exist, create `tests/test_shared_ui.py` with the content in Step 2.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_shared_ui.py (or appended to the existing file)
from clawmarks.shared_ui import NAV_OPTIONS, nav_bar_html


def test_nav_options_includes_preference_status_page():
    hrefs = [href for href, _label in NAV_OPTIONS]
    assert "preference_status.html" in hrefs


def test_nav_bar_html_marks_preference_status_selected_when_current():
    html = nav_bar_html("preference_status.html")
    assert 'value="preference_status.html" selected' in html
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_shared_ui.py -v`
Expected: FAIL (`assert "preference_status.html" in hrefs`)

- [ ] **Step 4: Add the nav entry**

In `src/clawmarks/shared_ui.py`, add a new tuple to `NAV_OPTIONS`, right after the existing
`("preference_rank.html", "predicted preference")` entry:

```python
    ("preference_status.html", "preference status"),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_shared_ui.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/shared_ui.py tests/test_shared_ui.py
git commit -m "feat(clawmarks): add preference status page to the shared nav bar"
```

---

### Task 4: `build/preference_status.py` (compute_data + render_html)

**Files:**
- Create: `src/clawmarks/build/preference_status.py`
- Test: `tests/test_preference_status.py`

**Interfaces:**
- Consumes: `clawmarks.config.SWEEP_DIR`/`PREFERENCE_SETTINGS_FILE`,
  `clawmarks.search.preference_settings.load`, `clawmarks.search.preference_model` (`MODEL_FILE`,
  `MODEL_META_FILE`, `MIN_LABELS`, `class_balance_error`), `clawmarks.shared_ui`
  (`nav_bar_html`, `TOPNAV_CSS`, `MOBILE_BASE_CSS`, `INFOTIP_CSS`, `info_btn`).
- Produces: `compute_data(sweep_dir) -> dict`, `render_html(data) -> str`. Task 5's
  `curation_server.py` route calls both.

**`compute_data` return shape** (all keys always present):

```python
{
    "n_yes": int, "n_no": int, "n_total": int,      # from user_ratings.json
    "min_labels": int,                               # preference_model.MIN_LABELS, for display
    "labels_gate_message": str,                       # "" if training could proceed right now
    "has_model": bool,                                 # MODEL_FILE exists
    "model_meta": dict | None,                        # MODEL_META_FILE contents, or None
    "use_predicted_preference": bool,                  # current persisted toggle value
}
```

`labels_gate_message` covers two failure modes with distinct text: below `MIN_LABELS` entirely
("only N labels (need 50); rate more images via rate.html.") and `class_balance_error`'s message
when at or above `MIN_LABELS` but imbalanced. Below `MIN_LABELS`, skip the balance check
entirely (matching `preference_model.main`'s own order: it checks `len(y) < MIN_LABELS` before
`class_balance_error`), since balance is irrelevant until the count gate is cleared.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preference_status.py
import json

from clawmarks.build import preference_status


def _write_ratings(tmp_path, n_yes, n_no):
    ratings = {}
    for i in range(n_yes):
        ratings[f"y{i}"] = {"label": "yes", "rated_at": "t"}
    for i in range(n_no):
        ratings[f"n{i}"] = {"label": "no", "rated_at": "t"}
    (tmp_path / "user_ratings.json").write_text(json.dumps(ratings))


def test_compute_data_with_no_ratings_file_reports_zero_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(preference_status.preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(preference_status.preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    data = preference_status.compute_data(tmp_path)
    assert data["n_yes"] == 0 and data["n_no"] == 0 and data["n_total"] == 0
    assert data["has_model"] is False
    assert data["model_meta"] is None
    assert data["use_predicted_preference"] is False
    assert "50" in data["labels_gate_message"]


def test_compute_data_below_min_labels_reports_count_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(preference_status.preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(preference_status.preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    _write_ratings(tmp_path, n_yes=10, n_no=5)
    data = preference_status.compute_data(tmp_path)
    assert data["n_yes"] == 10 and data["n_no"] == 5 and data["n_total"] == 15
    assert "15" in data["labels_gate_message"] and "50" in data["labels_gate_message"]


def test_compute_data_at_min_labels_but_imbalanced_reports_balance_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(preference_status.preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(preference_status.preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    _write_ratings(tmp_path, n_yes=58, n_no=2)
    data = preference_status.compute_data(tmp_path)
    assert "5-fold" in data["labels_gate_message"]


def test_compute_data_well_balanced_above_min_labels_has_no_gate_message(tmp_path, monkeypatch):
    monkeypatch.setattr(preference_status.preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(preference_status.preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    _write_ratings(tmp_path, n_yes=30, n_no=30)
    data = preference_status.compute_data(tmp_path)
    assert data["labels_gate_message"] == ""


def test_compute_data_reads_model_meta_and_toggle_when_model_exists(tmp_path, monkeypatch):
    settings_path = tmp_path / "preference_settings.json"
    model_path = tmp_path / "preference_model.joblib"
    meta_path = tmp_path / "preference_model_meta.json"
    monkeypatch.setattr(preference_status.preference_settings, "PREFERENCE_SETTINGS_FILE", settings_path)
    monkeypatch.setattr(preference_status.preference_model, "MODEL_FILE", model_path)
    monkeypatch.setattr(preference_status.preference_model, "MODEL_META_FILE", meta_path)
    model_path.write_text("fake model bytes")
    meta = {"trained_at": "2026-07-10T00:00:00+00:00", "n_labels": 60, "n_yes": 30, "n_no": 30, "cv_accuracy": 0.8}
    meta_path.write_text(json.dumps(meta))
    preference_status.preference_settings.save(True)

    data = preference_status.compute_data(tmp_path)
    assert data["has_model"] is True
    assert data["model_meta"] == meta
    assert data["use_predicted_preference"] is True


def test_render_html_disables_toggle_when_no_model():
    data = {"n_yes": 0, "n_no": 0, "n_total": 0, "min_labels": 50, "labels_gate_message": "not enough labels",
            "has_model": False, "model_meta": None, "use_predicted_preference": False}
    html = preference_status.render_html(data)
    assert "disabled" in html
    assert "/api/preference_toggle" in html


def test_render_html_enables_toggle_when_model_exists():
    meta = {"trained_at": "2026-07-10T00:00:00+00:00", "n_labels": 60, "n_yes": 30, "n_no": 30, "cv_accuracy": 0.8}
    data = {"n_yes": 30, "n_no": 30, "n_total": 60, "min_labels": 50, "labels_gate_message": "",
            "has_model": True, "model_meta": meta, "use_predicted_preference": True}
    html = preference_status.render_html(data)
    assert "disabled" not in html
    assert "checked" in html
    assert "0.8" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_preference_status.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'clawmarks.build.preference_status'`)

- [ ] **Step 3: Implement**

```python
# src/clawmarks/build/preference_status.py
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

    return {
        "n_yes": n_yes, "n_no": n_no, "n_total": n_total,
        "min_labels": preference_model.MIN_LABELS,
        "labels_gate_message": gate_message,
        "has_model": has_model,
        "model_meta": model_meta,
        "use_predicted_preference": preference_settings.load()["use_predicted_preference"],
    }


def render_html(data):
    gate_html = (f'<p class="gate">{data["labels_gate_message"]}</p>'
                 if data["labels_gate_message"] else '<p class="gate ok">ready to train.</p>')

    if data["model_meta"]:
        m = data["model_meta"]
        meta_html = (f'<table class="meta"><tr><td>trained</td><td>{m["trained_at"]}</td></tr>'
                     f'<tr><td>labels used</td><td>{m["n_labels"]} ({m["n_yes"]} yes / {m["n_no"]} no)</td></tr>'
                     f'<tr><td>cross-validated accuracy</td><td>{m["cv_accuracy"]}</td></tr></table>')
    else:
        meta_html = (f'<p class="meta-empty">no model trained yet. Once enough labels exist, run '
                     f'<code>python -m clawmarks.search.preference_model</code>.</p>')

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
table.meta {{ font-size:13px; border-collapse:collapse; }}
table.meta td {{ padding:3px 10px 3px 0; color:var(--text-dim); }}
table.meta td:first-child {{ color:var(--text); }}
.toggle-row {{ margin-top:14px; display:flex; align-items:center; gap:8px; }}
#toggle-status {{ font-size:12px; color:var(--text-dim); margin-left:8px; }}
{INFOTIP_CSS}
</style></head><body>

{nav_bar_html('preference_status.html')}
<h1>Preference classifier status</h1>
<p class="sub">Labels: {data["n_yes"]} yes / {data["n_no"]} no ({data["n_total"]} total, needs {data["min_labels"]}).</p>
<div class="panel">
{gate_html}
{meta_html}
<div class="toggle-row">
<label><input type="checkbox" id="toggle" {checked_attr} {disabled_attr} onchange="toggle(this.checked)"> use predicted preference{toggle_tip}</label>
<span id="toggle-status"></span>
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
</script>
<script src="scrollnav.js"></script>
<script src="infotip.js"></script>
</body></html>"""
    return html
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_preference_status.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/clawmarks/build/preference_status.py tests/test_preference_status.py
git commit -m "feat(clawmarks): add preference classifier status view module"
```

---

### Task 5: Wire `preference_status.html` + `/api/preference_status` + `/api/preference_toggle` into `curation_server.py`, and switch `archive.html` to the persisted setting

**Files:**
- Modify: `src/clawmarks/curation_server.py`
- Test: `tests/test_curation_server_preference_status_route.py`

**Interfaces:**
- Consumes: `build.preference_status.compute_data`/`render_html` (Task 4),
  `search.preference_settings.load`/`save` (Task 1), `search.preference_model.MODEL_FILE`
  (existing).
- Produces: three new routes; `archive.html` no longer parses a query string.

**Context:** `archive.html`'s current route (locate by `self.path.startswith("/archive.html")`,
not by line number, since earlier tasks in unrelated plans have shifted line numbers before):

```python
        if self.path.startswith("/archive.html"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            use_predicted = query.get("use_predicted_preference", ["0"])[0] == "1"
            target_name = "archive_predicted" if use_predicted else "archive_actual"
            data = _get_manifest_cached(
                target_name,
                lambda sd: elite_archive.compute_data(sd, use_predicted_preference=use_predicted),
            )
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_curation_server_preference_status_route.py
import json
import threading
from http.server import HTTPServer
import urllib.request
import urllib.error

import pytest

from clawmarks import curation_server as cs
from clawmarks.search import preference_settings


@pytest.fixture
def running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_DIR", tmp_path)
    monkeypatch.setattr(cs, "_live_cache", cs.LiveCache())
    monkeypatch.setattr(preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(cs.preference_settings, "PREFERENCE_SETTINGS_FILE", tmp_path / "preference_settings.json")
    monkeypatch.setattr(cs.preference_model, "MODEL_FILE", tmp_path / "preference_model.joblib")
    (tmp_path / "scored_manifest.json").write_text(json.dumps([]))
    (tmp_path / "user_ratings.json").write_text(json.dumps({}))
    server = HTTPServer(("127.0.0.1", 0), cs.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, tmp_path
    server.shutdown()
    thread.join(timeout=2)


def test_preference_status_html_route_serves_page(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/preference_status.html") as resp:
        body = resp.read().decode()
    assert "Preference classifier status" in body


def test_api_preference_status_route_returns_json(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/preference_status") as resp:
        data = json.loads(resp.read().decode())
    assert data["has_model"] is False
    assert data["use_predicted_preference"] is False


def test_post_preference_toggle_rejects_enable_without_model(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/preference_toggle", method="POST",
        data=json.dumps({"enabled": True}).encode(), headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_post_preference_toggle_accepts_enable_with_model_and_persists(running_server):
    server, tmp_path = running_server
    port = server.server_address[1]
    (tmp_path / "preference_model.joblib").write_text("fake model")

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/preference_toggle", method="POST",
        data=json.dumps({"enabled": True}).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    assert data["use_predicted_preference"] is True
    assert preference_settings.load()["use_predicted_preference"] is True


def test_archive_html_uses_persisted_setting_not_query_param(running_server, monkeypatch):
    server, tmp_path = running_server
    port = server.server_address[1]
    calls = []
    monkeypatch.setattr(cs.elite_archive, "compute_data", lambda sd, use_predicted_preference=False: calls.append(use_predicted_preference) or {"cells": []})

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/archive.html?use_predicted_preference=1") as resp:
        resp.read()
    assert calls == [False]

    preference_settings.save(True)
    (tmp_path / "preference_model.joblib").write_text("fake model")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/archive.html") as resp:
        resp.read()
    assert calls == [False, True]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_curation_server_preference_status_route.py -v`
Expected: FAIL (404s / AttributeError, since none of these routes or imports exist yet)

- [ ] **Step 3: Add imports**

`curation_server.py` does not currently import `preference_model` directly (only
`elite_archive.compute_data`, which imports it internally). Add all three needed imports next
to the existing `from clawmarks.search import rating_sampler` line:

```python
from clawmarks.search import rating_sampler, preference_settings, preference_model
```

And add `preference_status` to the existing `from clawmarks.build import (...)` tuple (the one
starting `scan_gallery, similarity_index, solution_map, ...`):

```python
from clawmarks.build import (
    scan_gallery, similarity_index, solution_map, map_view, redundancy_view, coverage_map,
    novelty_decay, lineage_view, elite_archive, preference_rank, uncanny_gallery, explore_hub,
    seed_browser, rate_page, preference_status,
)
```

- [ ] **Step 4: Add a watched-files helper and the two GET routes**

Add this helper near the other `_*_watched_files` / `_get_*_data` helpers (e.g. right after
`_get_manifest_cached`):

```python
def _preference_status_watched_files():
    files = []
    for f in (f"{SWEEP_DIR}/user_ratings.json", preference_model.MODEL_FILE,
              preference_model.MODEL_META_FILE, preference_settings.PREFERENCE_SETTINGS_FILE):
        if os.path.exists(f):
            files.append(str(f))
    return files


def _get_preference_status_data():
    return _live_cache.get(
        "preference-status", preference_status.compute_data,
        watched_files=_preference_status_watched_files(), sweep_dir=str(SWEEP_DIR),
    )
```

`_current_mtimes` (in `live_cache.py`) requires every watched file to exist (it calls
`os.path.getmtime` unconditionally), so this helper only includes files that already exist,
same pattern as the existing `_solution_map_watched_files`. `user_ratings.json` isn't written
until the first rating happens, so it's guarded the same way as the other three rather than
listed unconditionally.

Add the two GET routes. Locate the existing `/preference_rank.html` route (`if self.path ==
"/preference_rank.html":`) and add these two new blocks immediately after it:

```python
        if self.path == "/preference_status.html":
            html = preference_status.render_html(_get_preference_status_data())
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/preference_status":
            self._json_response(200, _get_preference_status_data())
            return
```

- [ ] **Step 5: Rewrite the `archive.html` route to use the persisted setting**

Replace:

```python
        if self.path.startswith("/archive.html"):
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            use_predicted = query.get("use_predicted_preference", ["0"])[0] == "1"
            target_name = "archive_predicted" if use_predicted else "archive_actual"
```

with:

```python
        if self.path.startswith("/archive.html"):
            use_predicted = preference_settings.load()["use_predicted_preference"]
            target_name = "archive_predicted" if use_predicted else "archive_actual"
```

Keep `startswith`, not an exact `==` match: `self.path` includes any query string, so a stray
`?use_predicted_preference=1` from an old bookmark must still reach this route (and is now
simply ignored, since `query` is no longer parsed at all) rather than 404ing.

- [ ] **Step 6: Add the POST route**

Locate `do_POST`'s dispatch chain (the `if self.path == "/api/rate":` block and its siblings).
Add a new block anywhere among them:

```python
        if self.path == "/api/preference_toggle":
            enabled = payload.get("enabled")
            if not isinstance(enabled, bool):
                self._json_response(400, {"error": "missing or non-boolean 'enabled'"})
                return
            if enabled and not os.path.exists(preference_model.MODEL_FILE):
                self._json_response(400, {"error": "no trained model yet; cannot enable predicted preference"})
                return
            preference_settings.save(enabled)
            self._json_response(200, _get_preference_status_data())
            return
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_curation_server_preference_status_route.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 8: Run the full suite to confirm no regression**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/ -v`
Expected: PASS (all tests, including the pre-existing archive-route and manifest-cache tests)

- [ ] **Step 9: Commit**

```bash
git add src/clawmarks/curation_server.py tests/test_curation_server_preference_status_route.py
git commit -m "feat(clawmarks): serve preference status page and shared toggle endpoint"
```

---

### Task 6: `cli.py` reads the persisted setting as `run allnight`'s default

**Files:**
- Modify: `src/clawmarks/cli.py`
- Test: `tests/test_cli.py` (check first with `fd -a test_cli tests`; extend if it exists)

**Interfaces:**
- Consumes: `search.preference_settings.load` (Task 1).
- Produces: nothing new consumed elsewhere; this is the last consumer of the shared setting.

**Context:** Locate by the `--use-predicted-preference` string and the `if args.command ==
"run":` block, not by line number.

- [ ] **Step 1: Check for an existing CLI test file**

Run: `fd -a test_cli tests`

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_cli.py (create, or append if it already exists)
from clawmarks import cli
from clawmarks.search import preference_settings


def test_run_allnight_forwards_explicit_true_flag_regardless_of_setting(monkeypatch):
    monkeypatch.setattr(preference_settings, "load", lambda: {"use_predicted_preference": False})
    captured = {}
    monkeypatch.setattr(cli, "_dispatch_run_allnight", lambda run_argv: captured.setdefault("argv", run_argv))
    cli.main(["run", "allnight", "--round", "1", "--use-predicted-preference"])
    assert "--use-predicted-preference" in captured["argv"]


def test_run_allnight_uses_persisted_setting_when_flag_omitted_and_true(monkeypatch):
    monkeypatch.setattr(preference_settings, "load", lambda: {"use_predicted_preference": True})
    captured = {}
    monkeypatch.setattr(cli, "_dispatch_run_allnight", lambda run_argv: captured.setdefault("argv", run_argv))
    cli.main(["run", "allnight", "--round", "1"])
    assert "--use-predicted-preference" in captured["argv"]


def test_run_allnight_uses_persisted_setting_when_flag_omitted_and_false(monkeypatch):
    monkeypatch.setattr(preference_settings, "load", lambda: {"use_predicted_preference": False})
    captured = {}
    monkeypatch.setattr(cli, "_dispatch_run_allnight", lambda run_argv: captured.setdefault("argv", run_argv))
    cli.main(["run", "allnight", "--round", "1"])
    assert "--use-predicted-preference" not in captured["argv"]
```

These tests monkeypatch a new `cli._dispatch_run_allnight` seam (Step 3 introduces it) instead
of the real `driver.main`, so they don't need a real search run's dependencies.

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_cli.py -v`
Expected: FAIL (`AttributeError: module 'clawmarks.cli' has no attribute '_dispatch_run_allnight'`)

- [ ] **Step 4: Implement**

In `src/clawmarks/cli.py`, change the argparse default:

```python
    allnight_p.add_argument(
        "--use-predicted-preference", action="store_true", default=None,
        help="Stage 5b: build the exploit pool from the trained preference model's top picks "
             "instead of yes-rated images. Defaults to the persisted toggle set on "
             "preference_status.html; pass this flag explicitly to force it on for one run "
             "regardless of that setting.",
    )
```

Add a small dispatch seam right above `main()`, so tests can monkeypatch it without importing
`driver.py`'s real dependencies:

```python
def _dispatch_run_allnight(run_argv):
    from clawmarks.search.driver import main as driver_main
    return driver_main(run_argv)
```

In `main()`, replace:

```python
    if args.command == "run":
        from clawmarks.search.driver import main as driver_main
        run_argv = ["--round", str(args.round)]
        if args.use_predicted_preference:
            run_argv.append("--use-predicted-preference")
        return driver_main(run_argv)
```

with:

```python
    if args.command == "run":
        run_argv = ["--round", str(args.round)]
        if args.use_predicted_preference is None:
            from clawmarks.search import preference_settings
            effective = preference_settings.load()["use_predicted_preference"]
        else:
            effective = args.use_predicted_preference
        if effective:
            run_argv.append("--use-predicted-preference")
        return _dispatch_run_allnight(run_argv)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/test_cli.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 6: Run the full suite**

Run: `cd /workspace/trent-with-smart-prompts && PYTHONPATH=src uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/clawmarks/cli.py tests/test_cli.py
git commit -m "feat(clawmarks): default run allnight's predicted-preference flag to the persisted toggle"
```

---

## Self-Review Notes

- **Spec coverage:** every architecture section of
  `docs/superpowers/specs/2026-07-10-preference-toggle-design.md` maps to exactly one task:
  §1 persisted setting -> Task 1, §2 metadata sidecar -> Task 2, §3 status module -> Task 4,
  §4 routes -> Task 5, §5 CLI resolution -> Task 6. Task 3 (nav entry) is a small addition the
  design doc implied ("other pages link to it") but didn't spell out as its own step; it's
  cheap and self-contained, right-sized as its own task rather than folded into Task 4 since
  `shared_ui.py` and `build/preference_status.py` are different files with different tests.
- **Ordering:** Task 4 imports `preference_settings` (Task 1) and reads `preference_model`'s
  `MODEL_FILE`/`MODEL_META_FILE`/`MIN_LABELS`/`class_balance_error` (Task 2 adds
  `MODEL_META_FILE`; the rest already existed), so both must land first. Task 5 imports Task 4's
  module directly, so it must come after. Task 6 only touches `cli.py` and is independent of
  Tasks 3-5; it's ordered last only because it's the least central consumer, not because of a
  real dependency edge (Task 3 could run any time after Task 1).
- **No placeholders:** every step shows exact code, exact commands, and exact expected output.
- **Type consistency check:** `preference_settings.load()` always returns
  `{"use_predicted_preference": bool}` (never `None`, never missing the key) in every call site
  across Tasks 4, 5, and 6, matching Task 1's `load()` contract exactly.
