# Explore Root And Status Page Design

## Goal

Make the exploration workbench the curation server's landing page. Preserve the current
status and expedition/leg picker on a dedicated page, so users can select the data context
without confusing it for the application home.

## Routes

- `GET /` renders the same Explore workbench returned by `GET /explore.html`.
- `GET /explore.html` remains supported for existing bookmarks.
- `GET /status.html` renders the current status, data-integrity warning, no-selection, and
  expedition/leg-picker variants now served at `GET /`.

## Navigation

- Every tool page, including the Explore workbench, renders the shared sticky header. Its active
  expedition/leg control opens a native modal that lists expeditions and legs, so users can switch
  context without leaving the page they were using. The modal includes a link to `status.html` for
  the complete status and data-integrity explanation.
- The shared navigation bar labels the hub link as `all tools` and points it at `/`.
- The tool jump menu groups options as Explore, Generate, Curate, Understand search, and
  Preference model. The status page appears as `session status` in Explore and in the modal.

## Behavior

Selecting a leg remains a `POST /api/active-leg` operation followed by a page reload. Once a
user selects `trent_v3_epoch4/freeform1` on `/status.html`, Scan Gallery receives that leg's
manifest. Selecting an empty `cockpit` leg still presents the existing status explanation.

## Error Handling

The route split does not change selection validation or data-integrity behavior. A missing
manifest remains visible on `status.html`; it must not be mistaken for a broken Explore hub.

## Tests

- Verify `/` and `/explore.html` both contain the Explore workbench.
- Verify `/status.html` contains the status page and its active-leg controls.
- Verify the shared navigation points the hub link at `/`, groups tool destinations, and opens the
  active-leg modal.
- Verify Explore receives the active expedition/leg and renders the shared header.
- Run the relevant route, Explore hub, and shared UI tests, then the full test suite.
