# Sulfur Proof Site Design System

## Goal

Apply one distinctive visual language across the complete curation site: an olive-gray working
proof, dense process-black hierarchy, and muted sulfur annotation. The system must improve reading
order and evidence legibility without wrapping every item in a card.

This specification supersedes the visual tokens, dark-theme default, typography, and component
surface rules in `2026-07-10-tool-suite-ui-design-brief.md` and other earlier page specifications.
Those documents still govern behavior that this specification does not replace.

## Visual Principle

The page is a working print proof, not a paper-themed dashboard.

- Olive-gray paper forms one continuous workspace.
- Dense black ink creates hierarchy through type scale, rules, reversal, and negative space.
- Sulfur behaves like translucent annotation material: underlines, hatching, registration halos,
  selected states, and compact active marks.
- Real CLAWMARKS images carry visual weight. Interface chrome remains restrained.
- Sections use spacing and rules before containers. A bordered panel needs a functional reason,
  such as clipping a map or separating the Guide from its source page.

## Color Tokens

```css
:root {
  --paper: #C3C5BA;
  --paper-deep: #B3B5A9;
  --ink: #11120F;
  --text-soft: #4D5048;
  --rule: #898D81;
  --sulfur: #CBD63F;
  --guide-surface: #20251B;
  --guide-ink: #ECEFDF;
}
```

Measured contrast ratios:

| Pair | Ratio | Use |
| --- | ---: | --- |
| `--ink` on `--paper` | 10.75:1 | primary text |
| `--text-soft` on `--paper` | 4.70:1 | normal secondary text |
| `--guide-ink` on `--guide-surface` | 13.40:1 | Guide text |
| `--sulfur` on `--guide-surface` | 9.86:1 | Guide highlights |
| `--ink` on `--sulfur` | 11.84:1 | compact selected controls |

The original `#55594F` secondary token reached only 4.10:1 on paper and is not approved for normal
text. Rules and decorative marks need not meet text contrast, but they cannot carry meaning alone.

Status colors must remain distinct from sulfur selection. Error, warning, success, and running
states require text or an icon in addition to color. Their final accessible tokens may extend this
palette during implementation.

## Typography

- **Display and condensed headings:** Barlow Condensed, weights 600 through 800.
- **Body and controls:** IBM Plex Sans, weights 400 through 700.
- **Metadata and receipts:** IBM Plex Mono, weights 400 through 600.

Production assets must be self-hosted or bundled. Pages must not depend on Google Fonts or another
runtime network request. Fallback stacks are:

```css
--font-display: "Barlow Condensed", "Arial Narrow", sans-serif;
--font-body: "IBM Plex Sans", Arial, sans-serif;
--font-mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
```

Headings may use uppercase when they remain short. Body text and research questions use sentence
case. Metadata labels may use uppercase with tracking. Normal body text starts at 14px on desktop
and 15px on mobile; explanatory text must not shrink into 10px dashboard copy.

## Layout

- Desktop content uses the full useful width with 22px to 30px side gutters.
- Major sections separate with 1px or 2px ink/rule lines and 24px to 54px vertical space.
- Editorial headlines establish the first eye path. Evidence, action, and metadata follow in that
  order.
- Maps and large images may use dark reversed surfaces when contrast helps the evidence.
- Three-column evidence layouts collapse to one column before text becomes cramped.
- Horizontal overflow is reserved for image strips, data grids, and the compact workflow stepper.

## Shared Header

The shared header contains, in order:

1. CLAWMARKS wordmark;
2. current page;
3. active expedition and leg;
4. active Focus and revision, when present;
5. running-search state, when present;
6. Guide button.

It uses one strong bottom rule, not a floating card. On narrow screens, expedition/leg and Focus
collapse into one labeled context control rather than disappearing without replacement. The Guide
button remains visible.

## Controls

### Primary actions

Primary actions use black fill with paper text. Sulfur appears as a bottom registration mark or
short underline. Large sulfur-filled call-to-action blocks are prohibited.

### Selected controls

A compact selected tab or step may use sulfur fill with black text. The selected state also uses
shape, weight, or `aria-current`; color is not the only cue.

### Workflow stepper

Orient, Scout, Explain, Act, and Learn form one connected black navigation rail with real button
elements. The active stage has one strong selected state. One shared detail/action strip sits below
the rail. Five bordered cards or five repeated descriptions are prohibited.

### Links and secondary actions

Text links use ink, weight, and an underline. Focus links may use a thicker sulfur underline.
Secondary buttons use a clear ink rule or black fill according to hierarchy. Controls never rely
on paper texture for their affordance.

### Focus and keyboard states

Interactive controls use a visible 3px sulfur or ink focus outline with sufficient offset. Hover,
focus, selected, disabled, working, and error states remain visually distinct.

## Evidence Treatments

### Map selection

A selected map region uses an ink boundary plus sulfur hatch or registration halo. It includes a
direct label such as `SELECTED REGION` and a member count. An unlabeled dashed ellipse is
prohibited.

### Coverage frontier

A frontier cell uses diagonal sulfur hatching, a strong ink border, and a visible `F` or text label.
The side explanation states that the cell is empty but adjacent to populated evidence.

### Images

Images avoid decorative frames. Use consistent cropping only in grids and preserve access to the
full image. Focus members, real anchors, and trial results receive text labels or patterned marks,
not color-only borders. Evidence images need meaningful `alt` text or an adjacent caption that names
their evidence role. Decorative texture uses empty alt text. Solution Map and Coverage provide an
accessible list or table equivalent for every selected point, region, frontier, and value exposed
only through the visual canvas.

### Data and metadata

Use aligned rows, rules, and mono labels before table-like cards. Keep units, score definitions, and
uncertainty close to their values. Charts include text summaries and do not depend on sulfur alone.

## Guide Surface

The Guide uses `--guide-surface` and `--guide-ink` as one continuous dark layer. Sulfur marks the
context receipt, assistant label, focus state, and active composer controls. Messages separate with
rules and spacing rather than chat bubbles.

The desktop drawer casts one restrained left shadow. The mobile sheet uses rounded top corners and
a drag handle because those shapes communicate the sheet interaction; this exception does not
license rounded cards elsewhere.

## Page Applications

### Explore

Place the connected workflow stepper directly below the shared header. Follow with the current
Focus as a compact working heading, then a continuous ruled research surface: evidence and saved
observations across the main width, one next-decision column, and a chronological activity ledger.
Use images as labeled evidence, not a decorative collage. Prohibit an oversized welcome hero,
marketing copy, feature cards, and broad empty space that makes the tool resemble a SaaS homepage.

Without a selected Focus, show a compact ruled Focus ledger and direct evidence-creation actions.
Do not replace the missing work with a promotional empty state.

### Solution Map

Give the map most of the viewport. Place interpretation directly over or beside the relevant
region, with a labeled selection and one Create Focus action.

### Coverage

Pair the grid with one editorial explanation column. Use ruled evidence rows, not statistic cards.

### Compare

Let the two images dominate. Separate them with one clear `OR` axis and keep model readiness in one
thin progress row.

### Cockpit

Present the research brief, fixed generation settings, and existing evidence as one ruled recipe.
Place paid payload review in one full-width final strip.

### Runs

Lead with run outcome and three inline statistics. Show trajectory and event log as continuous
evidence, not a dashboard of tiles.

### Error and empty states

Keep errors and empty states within the same paper/ink system. Use a clear heading, concrete cause,
affected scope, and one recovery action. Data-integrity errors must remain visually stronger than
ordinary empty states.

## Motion and Texture

Paper grain may use subtle CSS gradients. It must not reduce text contrast or create large paint
effects. Motion is limited to drawer/sheet transitions, progress updates, and direct manipulation.
Honor `prefers-reduced-motion` by removing nonessential transitions.

## Responsive Behavior

At 700px and below:

- the header preserves page, compact context, and Guide;
- editorial two-column layouts become one column;
- the workflow stepper scrolls horizontally and keeps each label readable;
- map annotations remain inside the viewport;
- Compare stacks images with a horizontal `OR` divider;
- Cockpit's recipe becomes one ruled sequence;
- payload actions become a full-width final row;
- Guide opens as a pull-up sheet no taller than the available viewport.

Mobile buttons, inputs, and icon controls provide at least a 44px by 44px touch target. Inline text
links remain exempt but use at least a 24px line box. Horizontal scrolling must show a visible
clipped edge or other cue that more content exists.

Drawers, sheets, lightboxes, and selection dialogs use named native dialog semantics or equivalent
`role="dialog"`, `aria-modal`, accessible name, focus trap, Escape behavior, and focus restoration.
Screen-reader order follows the visual evidence order rather than hidden layout columns.

## Non-Goals

- The system does not imitate torn-paper scrapbook decoration on every surface.
- Sulfur is not a general brand fill or decorative accent.
- The redesign does not erase dense research data to create marketing-page whitespace.
- Cards, pills, shadows, and rounded corners are not default grouping tools.

## Acceptance Criteria

- Shared tokens, typography, header, controls, and Guide treatment apply across every live tool.
- Explore's workflow reads and operates as a connected stepper, not five cards.
- Explore reads as an active research desk, not a product or SaaS landing page.
- Secondary normal text meets WCAG AA contrast on paper.
- Every selected map region and frontier has a direct non-color label.
- Primary actions remain black-led with restrained sulfur annotation.
- No production page fetches fonts from an external runtime service.
- Keyboard focus, reduced motion, 200% zoom, and 390px mobile layouts remain usable.
- Screen-reader checks cover the shared header, workflow stepper, map and Coverage equivalents,
  evidence images, image detail view, active-leg dialog, and Guide.
- Playwright live checks cover Explore, Map, Coverage, Compare, Cockpit, Runs, one empty state, one
  integrity error, and desktop/mobile Guide states before release.
