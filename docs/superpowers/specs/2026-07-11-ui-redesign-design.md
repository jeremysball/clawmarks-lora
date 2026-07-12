# CLAWMARKS UI Redesign: Three-Pillar Navigation + Grayscale Glass Theme

## Problem

The tool's navigation is a flat, 13-entry `<select>` dropdown (`NAV_OPTIONS` in `shared_ui.py`)
with no structure: `compare.html` and `preference_rank.html` sit next to `lineage.html` and
`gallery.html` with no indication of how they relate. A separate, hand-maintained hub page
(`explore_hub.py`'s `TOOLS` list) tries to group and describe the tools, but it's already
drifted out of sync with `NAV_OPTIONS` (missing `compare.html`, `preference_rank.html`, and
`preference_status.html` entirely), so there's no single reliable place that shows how the
toolset fits together.

Visually, the interface is a plain dark theme, and every page duplicates the same `:root` CSS
block with the same hardcoded colors (`--bg:#0b0b0d`, `--panel:#16161a`, `--border:#2a2a30`,
`--text:#eaeaee`, etc.) instead of sharing one definition, so changing the look means editing
every page identically.

## Design

### Three pillars

The 13 tools split into three top-level pillars, reflecting the actual workflow (browse
generations, judge them head-to-head, dig into the analysis):

- **Scan**: `scan.html`, `gallery.html` (binned atlas)
- **Rate**: `compare.html`, `preference_rank.html`, `preference_status.html`, `archive.html`
- **Explore**: `map.html`, `coverage.html`, `redundancy.html`, `novelty_decay.html`,
  `lineage.html`, `seeds.html`

### Navigation component

`shared_ui.py`'s `NAV_OPTIONS` (a flat list of `(href, label)` tuples) is replaced by a
pillar-structured data shape:

```python
NAV_PILLARS = [
    ("scan", "Scan", [
        ("scan.html", "scan gallery"),
        ("gallery.html", "binned atlas"),
    ]),
    ("rate", "Rate", [
        ("compare.html", "compare images (head-to-head)"),
        ("preference_rank.html", "predicted preference"),
        ("preference_status.html", "preference status"),
        ("archive.html", "elite archive"),
    ]),
    ("explore", "Explore", [
        ("map.html", "solution map (UMAP)"),
        ("coverage.html", "coverage / void map"),
        ("redundancy.html", "redundancy clusters"),
        ("novelty_decay.html", "novelty decay watchlist"),
        ("lineage.html", "lineage tree"),
        ("seeds.html", "candidate seeds"),
    ]),
]
```

`nav_bar_html(current)` renders three persistent top-level tabs (Scan / Rate / Explore). The
tab containing `current`'s page is marked active and its dropdown is open by default; the other
two pillars are collapsed dropdowns, expandable on hover or click. This replaces the single
`<select>` plus the old "&larr; all tools" link back to `explore.html`.

`explore_hub.py`'s independent `TOOLS` list is deleted. `NAV_PILLARS` becomes the single source
of truth for both the nav bar and the hub page: `explore_hub.py` renders its landing cards by
iterating `NAV_PILLARS` instead of maintaining its own copy, so the two can no longer drift
apart.

`explore.html` itself keeps its current role as an overview/landing page (now grouped into the
three pillar sections instead of one flat list), reachable from the nav bar as before.

### Theme: grayscale glass

Every page currently declares its own `:root` block with the same dark palette. This design
centralizes that into one `THEME_CSS` constant in `shared_ui.py`, which every page includes
instead of re-declaring `:root` itself. Visual direction: a desaturated take on Frutiger Aero's
glossy glass-panel look (2000s "Aero"/Luna software chrome), built entirely from the Nord color
palette, with no accent color beyond gray and frost blue.

This is a **light theme**: backgrounds come from Nord's lighter grays (darker-toned than white,
but still a light theme, not the dark Polar Night register Nord is usually used for), with
Polar Night reserved for text and dark accents.

**Palette** (Nord, https://www.nordtheme.com):

| Role | Tokens |
|---|---|
| Backgrounds (Snow Storm, darkest-first) | `#d8dee9` (page background) → `#e5e9f0` → `#eceff4` (lightest panel layer) |
| Text (Polar Night) | `#2e3440` primary, `#3b4252` / `#4c566a` secondary/dim |
| Glass tint (Frost) | `#88c0d0`, `#8fbcbb` (used as low-alpha `rgba()` panel backgrounds with `backdrop-filter: blur()`) |

No green, no other accent hue: state and emphasis (active nav tab, "ready" status, hover) are
expressed through the Frost blues and lightness/contrast only, not a separate accent color.

**Glass panel treatment:**

- Panels (nav bar, cards, modals) use a translucent Frost-tinted background
  (`rgba(136, 192, 208, 0.10)`-ish over the Snow Storm base) with `backdrop-filter: blur(12px)`.
- A single faint top-edge highlight gradient per panel (a restrained echo of Luna's glossy
  highlight band: one soft light-to-transparent gradient at the top few pixels, nothing
  elsewhere on the panel).
- **Sharp corners throughout (0 border-radius)**, no rounding anywhere, a deliberate contrast
  with Frutiger Aero's usual bubbly rounded chrome.
- Restraint rule: no gradients beyond the one glass-highlight band, no drop shadows beyond a
  single soft ambient shadow, no skeuomorphic bevels/reflections beyond that highlight. The goal
  is quiet and editorial, not the busy, bubbly look Frutiger Aero had at the time.

### Migration

- `shared_ui.py` gains `THEME_CSS` (the `:root` variable block above) and `GLASS_PANEL_CSS` (the
  shared panel/blur/highlight rules), both included once by every page alongside the existing
  `TOPNAV_CSS` / `MOBILE_BASE_CSS` / `INFOTIP_CSS`.
- Every page currently declaring its own `:root { color-scheme: dark; --bg:...; }` block (all
  ~13 tool pages, e.g. `preference_status.py`'s `render_html`) has that block deleted and
  replaced by including `THEME_CSS`. No page keeps a locally hardcoded color.
- `nav_bar_html()` changes signature only in its output (still takes `current`); every call site
  stays the same, so this is a drop-in replacement everywhere `nav_bar_html(...)` is already
  called.
- `explore_hub.py`'s hardcoded `TOOLS` list is deleted; its rendering logic is rewritten to
  iterate `NAV_PILLARS` from `shared_ui.py`.

### What's explicitly not changing

- No new JS framework or client-side router: navigation stays server-rendered HTML with a CSS
  dropdown (`:hover` / a small vanilla-JS toggle for touch), consistent with the rest of the
  toolchain's plain-HTML-page approach.
- No per-tool visual redesign beyond adopting the shared theme: each tool's own layout,
  controls, and content stay as they are today. This project reskins the chrome (nav + palette),
  not each page's internals.
- No accent/brand color beyond the grayscale-plus-frost palette (greenery considered and
  explicitly dropped).

## Testing

- A test asserting `NAV_PILLARS` contains every one of the 13 known tool hrefs exactly once (no
  tool dropped, none duplicated across pillars), a direct regression test for the
  `explore_hub.py`/`NAV_OPTIONS` drift that motivated this change.
- A `nav_bar_html()` test confirming the pillar containing the given `current` page is marked
  active in the output.
- A snapshot-style test confirming no page's rendered HTML contains a second, page-local
  `:root {` declaration (i.e., every page relies on the shared `THEME_CSS` instead of
  redeclaring its own).
