# OpenCode Guide Design

## Goal

Provide a dismissible research conversation on every curation page. The Guide helps the researcher
interpret evidence, challenge explanations, structure trials, and debrief results while leaving
all evidence changes, paid actions, and conclusions under human control.

## Role

The Guide may:

- explain the current page in plain language;
- summarize Focus evidence;
- propose competing interpretations;
- challenge a hypothesis with contradictory evidence;
- draft a six-part test contract;
- compare completed results with a frozen expectation;
- propose a Focus text edit for human review.

The Guide may not:

- select or change Focus members;
- alter source evidence or human judgments;
- update a Focus without a separate reviewed UI action;
- launch, stop, or retry paid work;
- decide that a hypothesis won;
- claim visual interpretation when it received metadata only.

## Presentation

Every shared header contains a Guide button. It opens:

- a right-side drawer or overlay on desktop, preserving enough width to inspect the source page;
- a pull-up sheet on mobile, with a visible drag handle and close control.

Closing the Guide hides the surface but preserves its thread, current response, and scroll position.
Reopening it on another page resumes the same Focus-scoped conversation and sends fresh page
context with the next message.

The drawer contains:

- a context receipt naming page, expedition, leg, Focus, Focus revision, and local selection;
- the persisted message history;
- clear speaker labels;
- a composer and send button;
- a working state that survives dismissal;
- readable retry text after a failed request;
- a metadata-only or image-grounded capability label;
- reviewed proposal controls when a response includes a Focus edit.

The Guide must never become a permanent page column. Maps, comparisons, and evidence views retain
their full width when the Guide is closed.

## Thread Scope and Storage

A Focus has one primary Guide thread. Pages without a Focus use one temporary leg-scoped thread and
offer Attach to Focus after the researcher creates a Focus.

Thread records live at:

```text
$CLAWMARKS_STATE_DIR/guide_threads/<expedition>/<leg>/<thread-id>.json
```

Each record contains:

```json
{
  "schema_version": 1,
  "thread_id": "guide_<uuid>",
  "scope": {"expedition": "...", "leg": "..."},
  "focus_id": "focus_<uuid>",
  "opencode_session_id": "...",
  "messages": [
    {
      "message_id": "message_<uuid>",
      "role": "user",
      "text": "...",
      "context_receipt": {},
      "created_at": "..."
    }
  ],
  "created_at": "...",
  "updated_at": "..."
}
```

Writes use the same file `fsync`, atomic replace, and parent-directory `fsync` pattern as Focus
records. The server persists the user message before starting OpenCode and persists the assistant
message before reporting completion to the browser. A failed or interrupted response remains
visible as request state, not a fabricated assistant message.

## Context Receipt

The browser sends IDs and local UI state. The server resolves authoritative records and builds the
model context. A receipt contains:

```json
{
  "page": "redundancy",
  "scope": {"expedition": "trent_v3_epoch4", "leg": "freeform1"},
  "focus": {"focus_id": "focus_<uuid>", "revision": 3},
  "local_selection": {"threshold": 0.74, "cluster_ids": [1, 4]},
  "evidence": {
    "member_count": 18,
    "missing_member_count": 0,
    "real_anchor_count": 1,
    "summary": "six effective clusters at threshold 0.74"
  },
  "visual_input": {
    "mode": "images",
    "generated_count": 6,
    "real_anchor_count": 1
  }
}
```

Each turn includes fresh context. The OpenCode session supplies conversational continuity; it does
not replace the current receipt.

The server limits context to the active Focus and local selection. It does not send secrets, API
keys, raw embedding tensors, unrelated state directories, or arbitrary browser-supplied paths.

## Visual Grounding

The Guide has two honest capability modes:

- **Metadata context only:** receives tags, prompts, scalar scores, comparisons, and summaries. It
  may reason about those records but must not say it inspected image content.
- **Image grounded:** receives representative Focus members, contradictory examples, and real-art
  anchors as actual image inputs to a model that supports images.

Image-grounded requests use a bounded sample, initially at most six generated images and two real
anchors. The server selects paths from validated Focus tags. The browser cannot submit arbitrary
filesystem paths.

If the configured OpenCode model cannot accept images, the server uses metadata mode and labels it
in the prompt, persisted message receipt, and UI. Every metadata-only response displays a fixed
system-controlled disclaimer: "This response used metadata, not image pixels." Guide prose never
updates a Focus automatically, so an unreliable semantic classifier is not required to decide
whether the model sounded visual.

An image-grounded structured proposal cites the image tags that support each visual claim. The
server verifies that every citation was an attached image in that request. Metadata-only proposals
may cite scores, prompts, comparisons, and summaries, but cannot carry image citations.

## OpenCode Process Boundary

The server invokes OpenCode with `--pure --agent <guide-agent>` from a fresh empty temporary
directory. The dedicated agent denies every tool that could discover or change state, including
`read`, `glob`, `grep`, `list`, `edit`, `bash`, `task`, `skill`, web tools, and every
mutation-capable MCP tool. It receives validated images only through explicit `--file` attachments.
The subprocess environment removes `RUNPOD_API_KEY`, `CIVITAI_TOKEN`, and unrelated credentials.
The current `--dangerously-skip-permissions` Autopilot invocation is not an acceptable Guide
boundary.

The server starts a new OpenCode session for the first message and stores the returned session ID.
Later messages resume that session and prepend the fresh context receipt. Pure mode, the empty
working directory, denied discovery tools, a minimal environment, and explicit attachments confine
the process to the prompt and validated images required for that turn.

The implementation must prove the restricted profile in a test or integration fixture before the
Guide ships. If the installed OpenCode version cannot enforce the restriction, the Guide remains
disabled and reports the missing safety capability.

## Asynchronous Request API

OpenCode responses may outlive one HTTP request. The Guide uses background request records:

```text
GET  /api/guide/thread?expedition=<name>&leg=<name>&focus_id=<id>
POST /api/guide/messages
GET  /api/guide/requests/<request-id>
POST /api/guide/threads/<thread-id>/attach
```

`POST /api/guide/messages` validates the scope and Focus, persists the user message, starts one
background OpenCode process, and returns HTTP 202 with a request ID. One thread may have only one
active request.

`GET /api/guide/requests/<request-id>` returns `queued`, `running`, `completed`, `failed`, or
`interrupted`. Completion includes the saved assistant message. Closing the drawer does not cancel
the request. After a server restart, any process without a live identity becomes `interrupted` and
the UI offers Retry from the saved user message.

The server enforces a timeout, captures bounded stderr for diagnosis, and never returns raw secrets
or an unrestricted stack trace to the browser.

Attach requires explicit expedition, leg, Focus ID, and `expected_focus_revision`. The server
accepts only an unattached temporary thread from the same scope. If the thread was already attached,
the Focus changed, or that Focus already has a primary thread, it returns HTTP 409 and preserves
both records. It never merges histories implicitly.

## Reviewed Focus Proposals

A Guide response may include one optional structured proposal:

```json
{
  "kind": "focus_text_patch",
  "focus_id": "focus_<uuid>",
  "expected_revision": 3,
  "changes": {
    "hypothesis_text": "...",
    "test_contract": {}
  },
  "reason": "..."
}
```

The server validates the shape but does not apply it. The UI displays old and proposed values and
offers Apply or Dismiss. Apply calls the normal revision-checked Focus PATCH endpoint. Membership,
anchors, score ranges, human judgments, trial payloads, and paid actions are never valid proposal
fields.

## Failure Behavior

- OpenCode unavailable: preserve the message and show Retry.
- Timeout: mark the request failed and preserve bounded diagnostics.
- Drawer closed: continue the request and restore its state on reopen.
- Scope changed mid-request: save the response to its original thread and label the old scope when
  shown later.
- Focus revision changed mid-request: keep the response, mark its receipt stale, and require a
  fresh proposal before Apply.
- Invalid structured output: show the prose response without proposal controls.
- Image attachment missing: fall back to metadata only and state the change before dispatch.

## Accessibility and Mobile

- Focus moves into the drawer when it opens and returns to the Guide trigger when it closes.
- Escape closes the desktop drawer. The close button remains visible and labeled.
- The drawer traps keyboard focus while open; the source page becomes inert.
- New messages use a polite live region. The working indicator has text, not animation alone.
- The mobile composer remains visible above the software keyboard.
- Drawer and sheet buttons, inputs, and icon controls provide at least a 44px by 44px touch target.
  The send action has a text label.

## Acceptance Criteria

- Every tool page opens the same Focus-scoped thread from the shared header.
- Closing and reopening the Guide preserves the thread and in-flight request.
- Every turn stores and displays an explicit context receipt.
- Every response stores and displays its system-controlled metadata-only or image-grounded mode;
  metadata-only prose cannot enter a Focus without a reviewed Apply action.
- Image-grounded requests use validated bounded image attachments and a capable model.
- OpenCode runs in pure mode from an empty directory with discovery, mutation, delegation, web, and
  MCP tools denied.
- The Guide cannot alter evidence or start paid work.
- Focus text proposals require a visible diff and explicit revision-checked Apply action.
- Desktop and 390px mobile layouts remain usable with keyboard, touch, and screen reader controls.
