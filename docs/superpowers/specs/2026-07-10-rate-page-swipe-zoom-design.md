# rate.html: swipe-to-vote and double-click zoom

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
| Touch drag, horizontal-dominant, past a deadzone | Swipe-vote: image follows the finger, a colored yes/no overlay fades in with drag distance; releasing past a threshold votes and advances, releasing short of it snaps back to center | Pans the full-resolution image under the finger |
| Mouse drag | No-op (desktop keeps voting via arrow keys / `y` / `n`) | Pans the full-resolution image (kept symmetric with zoom being mouse-accessible) |
| Single click/tap, no movement | No-op | No-op |
| Double click/tap | Toggles zoom in, centered on the click point | Toggles zoom back out to fit-to-screen |
| Arrow keys / `y` / `n` | Vote, unchanged from today | Unchanged; zoom state resets when the next image loads |

### Swipe-to-vote

Replaces the "no" / "yes" buttons entirely (removed from the DOM). Keyboard shortcuts remain as
the non-touch/desktop way to vote.

Mechanics: `touchstart` / `touchmove` / `touchend` listeners on `#img`. A touch is classified as
a drag once movement exceeds a small deadzone (~10px) and horizontal movement dominates
vertical. This single classification, decided once per touch, is what keeps a drag from also
registering as a tap (see "Gesture disambiguation" below).

While dragging: the image translates horizontally with the finger; a colored overlay (reusing
the page's existing `--yes` / `--no` CSS variables) fades in on the corresponding side, opacity
scaling with `|dx| / threshold` capped at 1, with a "YES"/"NO" stamp.

Commit threshold: 25% of the image's rendered width. At or past it on release, animate the image
off-screen in the drag direction, then call the existing `rate()` → `loadNext()` flow (unchanged
API calls). Short of threshold, animate back to center via CSS transition; no vote is recorded.

### Double-click/double-tap zoom

Triggered by the native `dblclick` event, not a hand-rolled timer: `dblclick` fires for both
mouse double-click and, on most touch browsers, double-tap, so this avoids reimplementing tap-
timing logic. Because some mobile browsers are inconsistent about firing `dblclick` for touch
(particularly interactions with `touch-action`), implementation must verify double-tap-to-zoom
actually works under touch emulation; if `dblclick` doesn't fire reliably there, fall back to a
manual double-tap detector (two `touchend`s within ~300ms and close together in position) purely
for the touch path, keeping `dblclick` for mouse.

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

A single touch interaction must resolve to exactly one outcome: a completed swipe-vote, a
snap-back (aborted swipe), or a tap contributing to double-click detection. This is why drag
classification happens once, early (deadzone + direction), and is reused as the gate for
whether a touch's `touchend` should also be eligible to combine into a `dblclick`/double-tap:
a touch classified as a drag never contributes to double-tap detection, and a touch that never
exceeded the deadzone is never treated as a drag.

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
