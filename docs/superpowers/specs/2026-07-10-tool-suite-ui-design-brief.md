# CLAWMARKS Tool Suite: UI/UX Design Brief

**Purpose:** a handoff brief for an external design pass (not a Claude-authored redesign). Documents
the current de facto design system, its inconsistencies, the technical constraints any design must
respect, and what a design pass should produce.

## Project context

CLAWMARKS is a hyperparameter/style search tool for a LoRA fine-tune. A MAP-Elites search generates
batches of images against a faithfulness x novelty grid, and a single non-academic researcher uses a
suite of browser tool pages to inspect, rate, and steer that search between generations. The tool
suite is also the source of figures for an eventual whitepaper, so pages should look presentable
enough to screenshot directly, not just be functional.

There are 13 tool pages today (`scan`, `gallery`, `map`, `coverage`, `archive`, `redundancy`,
`novelty_decay`, `lineage`, `preference_rank`, `seeds`, `rate`, `explore` as the hub, plus a
standalone historical `probe_report`), all served live by a single Python process
(`curation_server.py`) rendering plain HTML/CSS/JS strings, no client framework, no build step.

## Current system (as it exists today)

**Base palette** (CSS custom properties, used via `:root { ... }` in most pages):

| Variable | Value | Meaning |
|---|---|---|
| `--bg` | `#0b0b0d` | page background |
| `--panel` | `#16161a` | card/panel background |
| `--panel-2` | `#1d1d22` | secondary panel (used in `topnav`, some inputs) |
| `--border` | `#2a2a30` | hairline borders |
| `--text` | `#eaeaee` | primary text |
| `--text-dim` | `#9a9aa4` | secondary/caption text |

**Semantic accent colors**, layered on top per page, with a consistent meaning across pages that use
them (this pattern is implicit, never written down anywhere):

| Color | Hex | Used for |
|---|---|---|
| Gold | `#f5c542` | "notable" - human picks (`--pick`), frontier cells (`--frontier`) |
| Green | `#5ec98a` | "positive" - yes ratings (`--yes`), rising novelty (`--up`), style-type prompts (`--style`) |
| Red | `#e0605e` | "negative" - no ratings (`--no`), falling novelty (`--down`) |
| Blue | `#7c9eff` | accent/interactive - links, hover states, predicted-preference (`--predicted`) |

Typography: `-apple-system, sans-serif` everywhere, no other font ever loaded. Body padding is
`24px` on most pages, but `20px` on `map_view.py` and `32px` on `explore_hub.py` with no evident
reason for the difference. `h1` is `font-size:18px` on most pages, but `20px` on `explore_hub.py`,
and `uncanny_gallery.py`'s `h1` sets `font-weight:600` with no explicit `font-size` at all (falls
back to the browser default, visibly larger and inconsistent with every other page).

**Shared components** (`src/clawmarks/shared_ui.py`, imported by every page):

- **Top nav** (`nav_bar_html` + `TOPNAV_CSS`): sticky header, a "&larr; all tools" link back to the
  hub, and a jump-to `<select>` listing every other tool page. Auto-hides on scroll-down, reappears
  on scroll-up (`SCROLLNAV_JS`).
- **Info tooltips** (`info_btn` + `INFOTIP_CSS` + `INFOTIP_JS`): a small tappable "?" badge next to
  any non-obvious term (faithfulness, novelty, picking, MAP-Elites, ...), opening a popover on
  click (not hover-only, so it works on touch). Used heavily and deliberately, since the project's
  own conventions call for explaining domain vocabulary the first time it appears.
- **Lightbox** (`_LIGHTBOX_JS` + inline CSS in the same file): a full-screen image viewer usable
  from any page via `Lightbox.open(tag)`. Supports keyboard nav (arrows, escape, "f" for favorite),
  a similar-images strip (DINOv2 nearest neighbors), favoriting, and generating "counterfactual"
  variants (re-running generation with one parameter changed) inline.
- Mobile breakpoint at `640px` (`MOBILE_BASE_CSS`) shrinks padding, font sizes, and touch targets
  uniformly.

**Known inconsistencies** (the actual bugs/drift a design pass should resolve, not just aesthetic
opinion):

1. `lineage_view.py` and `uncanny_gallery.py` hardcode hex values directly (`#0b0b0d`, `#111`,
   `#eee`) instead of the shared `--bg`/`--panel`/`--text` custom properties every other page uses.
   `uncanny_gallery.py`'s values (`#111`/`#eee`) don't even match the standard palette exactly.
2. Body padding varies (`20px`/`24px`/`32px`) and `h1` sizing varies (`18px`/`20px`/browser-default)
   across pages with no functional reason for the difference; it reads as accidental drift, not
   intentional hierarchy.
3. `probe_report.py` is the one page with a light-theme option (`:root[data-theme="light"]`, a
   different, warmer palette entirely: `--bg:#f4f0e6`, `--panel:#fffdf8`). It's a standalone
   historical report, not part of the shared `shared_ui.py` system, and its palette has never been
   reconciled with the dark-only system every other page uses.
4. No shared spacing scale, type scale, or component-sizing tokens exist beyond the color
   variables; every page hand-tunes `font-size`/`padding`/`gap` values independently, which is how
   #1 and #2 happened in the first place.

## Technical constraints any design must respect

- **Plain HTML/CSS/vanilla JS only.** Every page is a Python f-string returning a complete HTML
  document. No React/Vue/build step, no CSS-in-JS framework, no npm dependency. Deliverables need
  to be usable as literal CSS custom properties and plain markup/class names Claude can translate
  into these Python string templates directly.
- **Dark-first.** Every page but the historical report assumes `color-scheme: dark` and is never
  viewed any other way in practice (this is a solo research tool, not a public product with
  varied viewers). A light theme is not a requirement unless explicitly decided otherwise.
- **Data-dense, not marketing-polished.** These are working instruments (scatter plots, heatmaps,
  filterable grids, a rating queue), not landing pages. Legibility and information density matter
  more than whitespace-heavy "product" aesthetics.
- **13 pages share a handful of archetypes**, not 13 unique layouts: a filterable image grid
  (`scan`, `gallery`), a chart/visualization (`map`, `coverage`, `novelty_decay`), a hub of cards
  (`explore`), a single-queue action page (`rate`), a tree/list view (`lineage`, `preference_rank`,
  `redundancy`, `archive`). A design pass should target these archetypes plus the shared nav/
  tooltip/lightbox components, not redline all 13 pages individually.
- Must keep the lightbox, info-tooltip, and top-nav components' existing *behavior* intact
  (keyboard nav, auto-hide-on-scroll, click-to-open tooltips): this brief is about visual
  consistency and polish, not a functional rewrite.

## What we want out of this

1. **A resolved, named palette**: the existing gold/green/red/blue semantic set, either confirmed
   as-is or refined, expressed as a definitive list of CSS custom properties with names and hex
   values, so every page (including `lineage_view.py`, `uncanny_gallery.py`, and `probe_report.py`)
   can reference the same variables instead of hardcoding or drifting.
2. **A small type and spacing scale**: 2-3 heading sizes, a body text size, a caption/dim text
   size, and a consistent spacing unit (or short scale) for padding/gaps, so page-to-page variation
   like the `20px`/`24px`/`32px` drift and the `h1` size drift stops happening by construction.
3. **Component-level visual direction** for the nav bar, info tooltip popover, and lightbox chrome:
   confirm the current look is right, or propose specific refinements (e.g. is the sticky
   auto-hiding nav still the right pattern at 13+ pages, does the tooltip's popover styling need
   work, does the lightbox's action-button row need a clearer visual hierarchy between favorite/
   counterfactual/back).
4. **One worked example per archetype** (grid, chart, hub, queue, tree/list), not all 13 pages,
   showing the palette/type/spacing decisions applied concretely enough that Claude can implement
   the same treatment across every page sharing that archetype.

## Non-goals

- A full visual rebrand or a light theme.
- Redesigning the underlying data visualizations themselves (the UMAP scatter, the heatmap grid,
  etc.) beyond the chrome around them, this is about the shared shell, not the domain-specific
  charts.
- Any change to page behavior/interaction, only to visual presentation.
- Implementation. This document is the brief handed to the external design pass; turning its
  output into actual `build/*.py` template changes is separate follow-up work once a design comes
  back.
