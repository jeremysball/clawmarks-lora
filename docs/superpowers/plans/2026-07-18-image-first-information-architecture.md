# Image-First Information Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CLAWMARKS's jargon-first research desk homepage with the existing image gallery, plain task-based navigation, progressive vocabulary disclosure, and explicit billable-action signals.

**Architecture:** Reuse `scan_gallery.render_html()` for both `/` and `/scan.html`; keep the Focus desk at `/explore.html` without treating its research stages as global navigation. Centralize destination labels, glossary definitions, information-button markup, and billable-action CSS in `shared_ui.py`, then apply those shared contracts to the gallery and every paid launch control.

**Tech Stack:** Python 3.12, server-rendered HTML/CSS/JavaScript, `BaseHTTPRequestHandler`, pytest, Playwright MCP.

## Global Constraints

- Never write to or delete irreplaceable generation output during tests or live verification.
- Preserve Focus query propagation and every server-side backup, balance-floor, spend-cap, and scope-validation guard.
- Keep exact research metrics available behind plain-language labels; do not rename persisted fields.
- Use `--cost: #5B3A63` only with a visible billable label or icon; color cannot carry meaning alone.
- Use `uv run` for every Python or pytest command.
- Verify live UI changes at desktop and 390px mobile with Playwright MCP.

---

### Task 1: Shared Task Navigation

**Files:**
- Modify: `src/clawmarks/shared_ui.py:215-331`
- Modify: `tests/test_shared_ui.py`
- Modify: `tests/test_explore_hub.py`

**Interfaces:**
- Produces: `NAV_GROUPS` with `Look at images`, `Make new images`, `Understand the search`, and `Preference model` groups.
- Produces: destination labels consumed by `nav_bar_html()` and `_page_name_for()`.

- [ ] **Step 1: Write failing navigation-contract tests**

```python
def test_nav_groups_use_plain_task_labels():
    groups = dict(NAV_GROUPS)
    assert tuple(groups) == (
        "Look at images", "Make new images", "Understand the search", "Preference model",
    )
    assert groups["Look at images"] == (
        ("/scan.html", "Browse all images"),
        ("/archive.html", "Best images by area"),
        ("/compare.html", "Choose between two images"),
    )
    assert ("/runs.html", "Run or monitor a search") in groups["Make new images"]
    assert ("/coverage.html", "Find gaps in the image space") in groups["Understand the search"]


def test_nav_has_no_workflow_stage_group():
    html = nav_bar_html("/")
    assert 'optgroup label="Explore"' not in html
    for label in ("Orient", "Scout", "Explain", "Act", "Learn"):
        assert f">{label}<" not in html
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest -q tests/test_shared_ui.py tests/test_explore_hub.py`
Expected: FAIL because `NAV_GROUPS` still exposes the old groups and stage-prefixed destinations.

- [ ] **Step 3: Replace `NAV_GROUPS` labels and simplify `_page_name_for()`**

Define the four groups exactly as specified. Keep Compare in both `Look at images` and
`Preference model`. Update `_page_name_for()` to return the first matching plain destination label;
remove the old detailed-versus-Explore precedence branch.

- [ ] **Step 4: Update Explore tests that assumed the old tool-index group names**

Delete assertions that require the homepage to mirror every navigation item. Keep tests for Focus
desk rendering at `/explore.html`, context propagation, and Sulfur Proof shell behavior.

- [ ] **Step 5: Run targeted tests**

Run: `uv run pytest -q tests/test_shared_ui.py tests/test_explore_hub.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/shared_ui.py tests/test_shared_ui.py tests/test_explore_hub.py
git commit -m "feat(ui): group navigation by user task"
```

### Task 2: Image-First Root Route

**Files:**
- Modify: `src/clawmarks/curation_server.py`
- Modify: `src/clawmarks/build/scan_gallery.py:95-474`
- Modify: `src/clawmarks/build/explore_hub.py:186-216`
- Modify: `tests/test_curation_server_startup.py`
- Modify: `tests/test_scan_gallery.py`
- Modify: `tests/test_explore_hub.py`

**Interfaces:**
- Consumes: current route context and cached Scan data already used by `/scan.html`.
- Produces: `/` and `/scan.html` through the same `scan_gallery.render_html()` path.
- Preserves: `/explore.html` as the Focus research desk.

- [ ] **Step 1: Write failing route and gallery-copy tests**

```python
def test_root_and_scan_render_the_same_image_gallery(live_server):
    root = live_server.get("/").text
    scan = live_server.get("/scan.html").text
    assert 'id="grid"' in root
    assert 'id="grid"' in scan
    assert "Browse and curate AI-generated artwork from this LoRA search." in root
    assert 'id="workflowStepper"' not in root


def test_explore_route_keeps_focus_desk_without_primary_stepper(live_server):
    explore = live_server.get("/explore.html").text
    assert "Open Foci" in explore or "Focus evidence" in explore
    assert 'id="workflowStepper"' not in explore
    assert "How a search round works" in explore
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest -q tests/test_curation_server_startup.py tests/test_scan_gallery.py tests/test_explore_hub.py`
Expected: FAIL because `/` currently renders `explore_hub` and Explore contains the stage stepper.

- [ ] **Step 3: Route `/` through the existing Scan renderer**

Change only the root route selection. Use the same cached data, `WorkspaceContext`, active
expedition/leg, Focus, and running-state inputs as `/scan.html`; do not add a second gallery builder.

- [ ] **Step 4: Add homepage orientation to Scan without duplicating the page**

Add the exact sentence `Browse and curate AI-generated artwork from this LoRA search.` near the
gallery heading. Keep it on `/scan.html` too so both routes remain one presentation contract.

- [ ] **Step 5: Demote the research loop inside Explore**

Remove `workflowStepper`, `workflowExplanation`, `workflowActions`, `STAGES`, and `ACTIONS` from
`render_html()`. Add a native `<details>` headed `How a search round works` containing the five
plain-language steps from the spec. Keep `derive_next_decision()` and the Focus desk's next-decision
status because they describe current research state rather than global navigation.

- [ ] **Step 6: Run targeted tests**

Run: `uv run pytest -q tests/test_curation_server_startup.py tests/test_scan_gallery.py tests/test_explore_hub.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/clawmarks/curation_server.py src/clawmarks/build/scan_gallery.py src/clawmarks/build/explore_hub.py tests/test_curation_server_startup.py tests/test_scan_gallery.py tests/test_explore_hub.py
git commit -m "feat(ui): make the image gallery the homepage"
```

### Task 3: Shared Glossary And Information Controls

**Files:**
- Modify: `src/clawmarks/shared_ui.py:423-489`
- Modify: `src/clawmarks/build/scan_gallery.py:112-145,229-262,410-423`
- Modify: `tests/test_shared_ui.py`
- Modify: `tests/test_scan_gallery.py`

**Interfaces:**
- Produces: `GLOSSARY: dict[str, tuple[str, str]]` mapping a key to plain label and formal definition.
- Produces: `info_btn(key: str) -> str` returning an accessible information button.

- [ ] **Step 1: Write failing shared-glossary tests**

```python
def test_glossary_keeps_plain_and_formal_metric_names_together():
    assert GLOSSARY["faithfulness"][0] == "Similarity to real art"
    assert "DINOv2 cosine similarity" in GLOSSARY["faithfulness"][1]
    assert GLOSSARY["novelty"][0] == "How new or different"


def test_info_button_is_an_accessible_inert_i_control():
    markup = info_btn("novelty")
    assert markup.startswith('<button type="button"')
    assert '>i</button>' in markup
    assert 'aria-label="More information about How new or different"' in markup
    assert "?" not in markup
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest -q tests/test_shared_ui.py tests/test_scan_gallery.py`
Expected: FAIL because `info_btn()` accepts raw text and emits a `?` span.

- [ ] **Step 3: Add the glossary and accessible button contract**

Create entries for `faithfulness`, `novelty`, `map_elites_cell`, `umap`, and `redundancy`. Render a
native button with `type="button"`, visible `i`, `aria-label`, `aria-expanded="false"`, and the
definition in `data-tip`. Update tooltip JavaScript to maintain `aria-expanded`, close on Escape,
and restore focus to the triggering button.

- [ ] **Step 4: Replace Scan's raw labels and abbreviations**

Use `Similarity to real art` in filter options and range labels. Use `How new or different` for
novelty sorting. Replace thumbnail overlay text:

```javascript
<div class="meta">${escHtml(d.prompt_name)}</div>
```

Keep exact values and formal definitions in the existing Lightbox metadata rather than deleting
them. Update hard-coded Lightbox `?` spans in `shared_ui.py` to the same button contract.

- [ ] **Step 5: Run targeted tests**

Run: `uv run pytest -q tests/test_shared_ui.py tests/test_scan_gallery.py`
Expected: PASS, including assertions that rendered gallery controls contain no `Faith`, `f=`, or
`n=` labels.

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/shared_ui.py src/clawmarks/build/scan_gallery.py tests/test_shared_ui.py tests/test_scan_gallery.py
git commit -m "feat(ui): add plain-language metric glossary"
```

### Task 4: Billable Action Affordances

**Files:**
- Modify: `src/clawmarks/shared_ui.py`
- Modify: `src/clawmarks/build/seed_browser.py`
- Modify: `src/clawmarks/build/runs_page.py`
- Modify: `src/clawmarks/build/cockpit.py`
- Modify: `tests/test_shared_ui.py`
- Modify: `tests/test_seed_browser.py`
- Modify: `tests/test_runs_page.py`
- Modify: `tests/test_curation_server_cockpit_scoring.py`
- Modify: `tests/test_curation_server_counterfactual_route.py`

**Interfaces:**
- Produces: `.billable-action` and `.cost-badge` shared styles using `--cost:#5B3A63`.
- Produces: `billable_badge(estimate: str | None = None) -> str` with visible `Spends money` text.
- Preserves: each action's existing API endpoint, validation, confirmation, and cost guard.

- [ ] **Step 1: Write failing shared and page-level tests**

```python
def test_billable_badge_never_invents_an_estimate():
    assert billable_badge() == '<span class="cost-badge">Spends money</span>'
    assert "~$2.00" in billable_badge("~$2.00")


def test_paid_pages_mark_their_commit_actions():
    assert 'id="genBtn" class="primary-action billable-action"' in seed_browser.render_html([])
    assert "Spends money" in seed_browser.render_html([])
    assert 'id="launchBtn"' in runs_page.render_html()
    assert "Spends money" in runs_page.render_html()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `uv run pytest -q tests/test_shared_ui.py tests/test_seed_browser.py tests/test_runs_page.py tests/test_curation_server_cockpit_scoring.py tests/test_curation_server_counterfactual_route.py`
Expected: FAIL because no shared billable contract exists.

- [ ] **Step 3: Add restrained billable styles and helper markup**

Add `--cost:#5B3A63` to Sulfur tokens. Keep paid buttons black-led; use aubergine for the visible
badge and a registration edge. The helper emits `Spends money` and appends an estimate only when the
caller provides one.

- [ ] **Step 4: Apply the contract to every paid action**

Mark candidate-seed generation, search launch, Cockpit generation, and counterfactual generation.
Do not mark filters, navigation, favorites, picks, Compare votes, model retraining, or stop actions.
Keep existing confirmation dialogs; revise their copy to name the selected expedition/leg and cost
or spend cap already known to the page.

- [ ] **Step 5: Run targeted tests**

Run: `uv run pytest -q tests/test_shared_ui.py tests/test_seed_browser.py tests/test_runs_page.py tests/test_curation_server_cockpit_scoring.py tests/test_curation_server_counterfactual_route.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/clawmarks/shared_ui.py src/clawmarks/build/seed_browser.py src/clawmarks/build/runs_page.py src/clawmarks/build/cockpit.py tests/test_shared_ui.py tests/test_seed_browser.py tests/test_runs_page.py tests/test_curation_server_cockpit_scoring.py tests/test_curation_server_counterfactual_route.py
git commit -m "feat(ui): identify billable generation actions"
```

### Task 5: Regression And Live Accessibility Gate

**Files:**
- Modify: `notes/lab_notebook.md`
- Modify if defects surface: files from Tasks 1-4 and their tests

**Interfaces:**
- Verifies: route identity, context propagation, safe-versus-paid semantics, keyboard behavior,
  desktop/mobile layout, and data-integrity preservation.

- [ ] **Step 1: Run focused static checks**

Run: `uv run ruff check src tests`
Expected: PASS.

Run: `uv run mypy src`
Expected: PASS.

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS with no regressions.

- [ ] **Step 3: Start the server without touching generation data**

Use the project `run` skill to start the curation server bound to `0.0.0.0`. Use an existing
read-only leg for gallery checks. Do not click a confirmed billable action.

- [ ] **Step 4: Verify desktop with Playwright MCP**

At 1440x900, inspect `/`, `/scan.html`, `/explore.html`, `/coverage.html`, `/seeds.html`, and
`/runs.html`. Confirm thumbnails appear above the fold, task groups and plain labels are present,
the research-loop disclosure works, information buttons open and close without state mutation, and
all paid actions visibly say `Spends money`. Confirm zero console errors.

- [ ] **Step 5: Verify mobile and keyboard behavior**

At 390x844, repeat `/`, `/scan.html`, `/seeds.html`, and `/runs.html`. Confirm no horizontal page
overflow, at least 44px information-button and select targets, Escape closes glossary popovers,
focus returns to the information button, and billable labels remain visible at 200% zoom.

- [ ] **Step 6: Record the redesign and verification**

Append a dated lab entry naming the persona evidence, structural IA decision, exact routes changed,
tests run, Playwright pages and viewport sizes, and any defects found. Do not summarize this plan as
completed until every check above has passed.

- [ ] **Step 7: Commit**

```bash
git add notes/lab_notebook.md
git commit -m "docs: record image-first navigation verification"
```
