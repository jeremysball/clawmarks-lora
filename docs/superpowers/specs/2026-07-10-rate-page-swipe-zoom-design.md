# rate.html: swipe-to-vote and double-click zoom

> **Revision (2026-07-10, after Tasks 1-3 landed):** the interaction model below is superseded
> by three changes made after manual review of the live page: zoom now triggers on a single
> tap/click instead of double, mouse drag now votes the same way touch drag does (rotate +
> color overlay), and the swipe overlay uses a thumbs up/down icon instead of a text stamp. The
> "Interaction model" table and the "Zoom trigger" and "Swipe-to-vote" sections below reflect
> the revised design; the rest of the document (scope, state reset, non-goals) still holds.

## Background

`rate.html` (`src/clawmarks/build/rate_page.py`) is the full-screen yes/no rating page that
trains the preference classifier (`search/preference_model.py`). Today it shows one image at a
time with two buttons ("no" / "yes") and keyboard shortcuts (arrow keys, `y`/`n`). Clicking the
image does nothing.

Two gaps prompted this design:

- Tapping small on-screen buttons is slower and less natural on a phone than a swipe gesture,
  the standard interaction for this exact yes/no pattern (Tinder and similar apps).
- The image shown is already the full-resolution PNG (`item_summary()` in
  `search/manifest_index.py` returns `os.path.basename(m["file"])`, not a thumbnail), but the
  page displays it scaled down to fit the screen with no way to inspect it at native size. This
  matters because some rating decisions hinge on fine detail (linework quality, artifacts) that
  a fit-to-screen view can hide.

## Scope

Changes are confined to `src/clawmarks/build/rate_page.py` (the HTML/CSS/JS this module
generates). No server-side API changes: the page continues to use the existing
`GET /api/rate/next` and `POST /api/rate` endpoints. No changes to the shared `Lightbox`
component in `shared_ui.py`: `rate.html` doesn't use it today and won't start now. This is a
page-local interaction, not a shared one.

Out of scope: any other tool page (`gallery.html`, `archive.html`, `preference_rank.html`, etc.)
keeps its current click behavior unchanged.

## Interaction model

One image is on screen at a time, same as today. Two independent gestures apply to it:

| Gesture | Not zoomed | Zoomed |
|---|---|---|
| Touch drag or mouse drag, horizontal-dominant, past a deadzone | Swipe-vote: image follows the pointer and rotates (tilt proportional to drag distance, capped at 15deg); a colored overlay (green/red) with a thumbs up/down icon fades in with drag distance; releasing past a threshold votes and advances, releasing short of it snaps back to center with rotation reset | Pans the full-resolution image under the pointer (no rotation, no overlay) |
| Single click/tap, no movement | Toggles zoom in, centered on the click point | Toggles zoom back out to fit-to-screen |
| Arrow keys / `y` / `n` | Vote, unchanged from today | Unchanged; zoom state resets when the next image loads |

### Swipe-to-vote

Replaces the "no" / "yes" buttons entirely (removed from the DOM). Keyboard shortcuts remain as
an alternative way to vote on desktop. Mouse drag now votes the same way touch drag does (this
supersedes the original "mouse drag never votes" rule); only mouse/touch drag *while zoomed*
stays pan-only, since a vote gesture on an already-zoomed image would be ambiguous with panning.

Mechanics: `touchstart`/`touchmove`/`touchend` (touch) and `mousedown`/`mousemove`/`mouseup`
(mouse) listeners on `#imgwrap`, sharing the same classification and visual-update logic. A drag
is classified once movement exceeds a small deadzone (~10px): horizontal-dominant movement while
not zoomed classifies as swipe; any movement while zoomed classifies as pan. This single
classification, decided once per pointer-down, is what keeps a drag from also registering as a
tap (see "Gesture disambiguation" below).

While dragging (swipe): the image translates horizontally with the pointer and rotates
(`rotate(deg)` composed with the existing `translate`), tilt magnitude scaling with
`|dx| / threshold` capped at 1 and mapped to a max 15deg. A colored overlay (reusing the page's
existing `--yes`/`--no` CSS variables for the background tint) fades in on the corresponding
side, opacity scaling the same way, showing a thumbs up (👍) or thumbs down (👎) icon instead of
a text stamp.

Commit threshold: 25% of the image's rendered width. At or past it on release, animate the image
off-screen in the drag direction (continuing its rotation), then call the existing
`rate()` → `loadNext()` flow (unchanged API calls). Short of threshold, animate back to center
(rotation back to 0) via CSS transition; no vote is recorded.

### Tap/click zoom

Triggered by a single tap or click that stays within the drag deadzone (no `dblclick`, no
timer-based double-tap detection). This is only reachable when not zoomed and the pointer never
crossed the ~10px deadzone; anything that moves past it is a swipe or pan instead, never a zoom
toggle, per "Gesture disambiguation" below. Tapping again while zoomed (same no-movement rule)
zooms back out to fit-to-screen.

Zoom states are two CSS states on `#img`'s wrapper, toggled by class:

- **Fit** (current behavior): `max-width:100%; max-height:78vh; object-fit:contain`, centered.
- **Zoomed**: the image at natural pixel size, inside an `overflow:hidden` wrapper, positioned
  via `transform: translate(x, y)`. The initial offset centers the wrapper on the click
  coordinate (translated into image-pixel space). No load spinner or fetch is needed on zoom-in
  since the browser already has the full-resolution bytes decoded from the initial `<img>` load.

While zoomed, drag (touch or mouse) updates the translate offset to pan, clamped so the image
never pans past its own edges (the wrapper can't show past the image bounds on any side).

Double-click/double-tap again while zoomed toggles back to fit, discarding the pan offset.

### Gesture disambiguation

A single pointer interaction (touch or mouse) must resolve to exactly one outcome: a completed
swipe-vote, a snap-back (aborted swipe), a pan, or a tap/click that toggles zoom. This is why
drag classification happens once, early, on the same deadzone (~10px) used for both purposes:
a release that never exceeded the deadzone is a tap and toggles zoom; a release that did exceed
it was already classified as swipe or pan and never also triggers a zoom toggle.

## State reset

Zoom state (and any in-progress drag transform) resets whenever `loadNext()` swaps in a new
image: the next image always starts at fit-to-screen, non-zoomed, non-panned. This is a small
addition to the existing `loadNext()` function, not new machinery.

## Testing approach

This is gesture-driven client-side JS operating on CSS transforms and native touch events, with
no backend logic to unit-test meaningfully. Verification happens by exercising it directly
(the `verify` skill): real drag/tap/double-tap interaction in a browser (touch-emulated via
devtools where real touch hardware isn't available), rather than by writing tests for CSS
transform math.

## Non-goals

- No changes to the shared `Lightbox` or any page besides `rate.html`.
- No continuous pinch-to-zoom or scroll-wheel zoom: zoom is a single double-click/double-tap
  toggle between exactly two states (fit, native size), per the approved design choice.
- No server or API changes.
