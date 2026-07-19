# Image-First Information Architecture Design

## Goal

Make CLAWMARKS understandable on first contact without hiding its research depth. The site opens
on the generated artwork, names destinations by the task they support, explains technical terms in
plain language before exposing formal names, and distinguishes safe controls from billable actions.

This specification supersedes the primary-navigation and Explore-homepage requirements in
`2026-07-16-research-workspace-navigation-design.md`,
`2026-07-16-explore-root-status-page-design.md`, and the Explore/workflow requirements in
`2026-07-16-sulfur-proof-design-system.md`. Those specifications still govern Focus persistence,
scope propagation, evidence handling, accessibility, and the Sulfur Proof visual language.

## Evidence

Three independent persona walkthroughs produced the same failures:

- the homepage showed no artwork;
- `Orient`, `Scout`, `Explain`, `Act`, and `Learn` looked like navigation but named no destination;
- the flat 13-tool index presented builder vocabulary instead of user tasks;
- the plain page-jump select was the only navigation every persona trusted immediately;
- Scan Gallery and Compare were the clearest, most satisfying pages;
- abbreviated labels such as `faith` were mistaken for unrelated ordinary words;
- users avoided billable actions by intuition because the interface provided no cost signal; and
- users avoided `?` documentation controls because they feared changing something.

These are structural information-architecture and affordance failures. Renaming five buttons or
adding more tooltips cannot fix them.

## Homepage

`GET /` renders the same image-browsing experience as `GET /scan.html`. The homepage must show real
generated thumbnails above the fold whenever the selected leg has images. It includes this concise
orientation before or within the gallery controls:

> Browse and curate AI-generated artwork from this LoRA search.

The homepage keeps the shared expedition/leg context control, Guide, session status, and page-jump
navigation. `/scan.html` remains a valid stable route and renders the same gallery. Neither route
duplicates gallery behavior or data-loading logic.

When the selected leg has no scored images, the page uses the existing Sulfur Proof empty-state
grammar: name the missing input, identify the selected scope, and link to the nearest useful safe
action. It must not replace missing evidence with a marketing hero or a wall of tool descriptions.

## Navigation

Primary navigation names destinations and groups them by user task:

### Look at images

- **Browse all images** (`/scan.html`)
- **Best images by area** (`/archive.html`)
- **Choose between two images** (`/compare.html`)

### Make new images

- **Build one image trial** (`/cockpit.html`)
- **Run or monitor a search** (`/runs.html`)
- **Edit candidate ideas** (`/seeds.html`)

### Understand the search

- **Explore image neighborhoods** (`/map.html`)
- **Find gaps in the image space** (`/coverage.html`)
- **Find near-duplicate groups** (`/redundancy.html`)
- **See which prompts are running out** (`/novelty_decay.html`)
- **Trace image ancestry** (`/lineage.html`)

### Preference model

- **Choose between two images** (`/compare.html`)
- **Check taste-model readiness** (`/preference_status.html`)
- **See predicted favorites** (`/preference_rank.html`)

Compare intentionally appears in both `Look at images` and `Preference model`: it is both a direct
curation activity and the input to the preference model. The shared page-jump control remains
visible because the walkthroughs validated it as an effective direct-navigation escape hatch.

The homepage does not repeat a full 13-item inventory below the gallery. Grouped destinations live
in the shared navigation, where they remain available on every page.

## Research Loop

`Orient`, `Scout`, `Explain`, `Act`, and `Learn` remain the project's research method, not primary
navigation. Remove the connected stepper and its changing action strip from the homepage.

A secondary **How a search round works** disclosure explains the loop in one place:

1. **Set the question:** choose the expedition and leg, then state the visual question.
2. **Find evidence:** inspect images, neighborhoods, or reachable gaps.
3. **Explain the pattern:** separate the observation from a possible explanation.
4. **Run one bounded test:** change one variable, hold the others fixed, and set a spend cap.
5. **Judge the result:** compare the result with the evidence and record what changed.

The formal stage names may appear as secondary labels inside this explainer. Focus-scoped pages may
still show the current method stage as status, but must not present the stages as destinations.

## Vocabulary

Use progressive disclosure: plain language on the default surface, formal research language in a
nearby detail, glossary entry, or evidence panel. Never truncate a technical term into another real
word.

| Default surface | Technical detail |
| --- | --- |
| Similarity to real art | Faithfulness: DINOv2 cosine similarity to the real-image centroid |
| How new or different | Novelty: one minus the nearest prior-image similarity |
| Image-space area | MAP-Elites cell or bin |
| Image neighborhood map | UMAP projection of DINOv2 embeddings |
| Near-duplicate group | Redundancy cluster at the selected similarity threshold |

Gallery filters use `Similarity to real art`, never `Faith`. Thumbnail overlays do not show
unexplained `f=` or `n=` abbreviations. Exact values remain available in the image lightbox, where
each value has its plain label and formal definition.

Definitions come from one shared glossary mapping so pages do not drift. Information controls use a
lowercase `i`, an accessible `button`, and an explicit accessible name such as
`More information about novelty`. Their treatment must look informational and inert. Opening one
changes no research or generation state.

## Billable Actions

Generation and search-launch actions carry a shared billable-action treatment distinct from safe
filters, navigation, and ordinary primary actions:

- a visible `Spends money` label;
- an estimated cost when the application has a defensible estimate, otherwise no invented number;
- a confirmation step that names the operation, scope, and known cost or cap; and
- a reserved deep aubergine token, `--cost: #5B3A63`, paired with paper text and a text/icon label.

Aubergine complements sulfur while remaining semantically separate from sulfur selection and from
error red. Color never carries the warning alone. Billable controls retain the black-led primary
action structure from Sulfur Proof; the aubergine appears as a badge, registration edge, or confirm
surface rather than a large decorative fill.

`Generate`, counterfactual generation, search launch, and any future action that submits paid work
use this treatment. Safe actions must not use it. Existing server-side balance floors, spend caps,
backup checks, and confirmation rules remain mandatory.

## Responsive And Accessible Behavior

- At 390px, generated images remain visible above the tool inventory and no control causes
  horizontal page overflow.
- Navigation groups retain readable task labels; the page-jump select remains at least 44px tall.
- Information buttons provide at least a 44px touch target on mobile.
- Glossary popovers use dialog/popover semantics, keyboard activation, Escape dismissal, focus
  restoration, and a readable mobile position.
- Billable badges meet WCAG AA contrast and include text; color is supplemental.
- The homepage, gallery filters, glossary controls, and billable confirmation flow remain usable at
  200% zoom and with reduced motion.

## Non-Goals

- This redesign does not remove Focus dossiers or automate research judgment.
- It does not rename persisted schema fields such as `faith` or `novelty`; it changes presented
  language at the UI boundary.
- It does not merge the analytical tools into one large page.
- It does not hide exact scores, DINOv2 methodology, MAP-Elites structure, or statistical evidence.
- It does not add a marketing homepage, decorative image collage, or generic dashboard cards.
- It does not estimate generation cost when the application lacks enough information.

## Acceptance Criteria

- `/` and `/scan.html` render one shared, image-first gallery implementation.
- A selected leg with images shows real thumbnails above the fold on desktop and 390px mobile.
- Primary navigation uses the four approved task groups and plain destination labels.
- The five-stage loop is absent from primary navigation and available through
  `How a search round works`.
- No user-facing gallery control or thumbnail uses `Faith`, `faith`, `f=`, or `n=` as an
  unexplained label.
- Shared information controls render as accessible `i` buttons and describe themselves as
  information-only.
- Every billable UI action displays `Spends money` and requires a scope-aware confirmation.
- Safe navigation, filtering, favoriting, and comparison actions carry no billable treatment.
- Existing Focus query propagation, data-integrity guards, backup checks, spend caps, and RunPod
  balance floors continue to work.
- Targeted tests and the full suite pass; Playwright verifies `/`, `/scan.html`, one analysis page,
  `/seeds.html`, `/runs.html`, and counterfactual generation at desktop and 390px mobile with no
  console errors or horizontal overflow.
