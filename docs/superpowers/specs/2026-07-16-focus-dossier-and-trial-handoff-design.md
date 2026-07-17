# Focus Dossier And Trial Handoff Design

## Goal

Persist a researcher's selected evidence, interpretation, and test intention as one explicit,
expedition/leg-scoped Focus. Carry an immutable snapshot of that Focus into Cockpit, attach every
result to its paid launch, and attach every Focus-derived result to the trial that produced it.

## Terms

- A **Focus** is a durable evidence dossier for one visual question.
- A **Focus revision** is the saved state that existed after one successful edit.
- A **trial** is one bounded test derived from a specific Focus revision.
- A **result** is one generated image and its manifest record from a paid launch. Focus-derived
  results also belong to a trial.

The Focus is the primary object. A trial cannot redefine its source Focus after confirmation.

## Storage

Focus records live at:

```text
$CLAWMARKS_STATE_DIR/foci/<expedition>/<leg>/<focus-id>.json
```

Trial records live at:

```text
$CLAWMARKS_STATE_DIR/trials/<expedition>/<leg>/<trial-id>.json
```

Every paid route has a generic launch record at:

```text
$CLAWMARKS_STATE_DIR/paid_launches/<expedition>/<leg>/<launch-id>.json
```

Per-result recovery receipts live at:

```text
$CLAWMARKS_STATE_DIR/result_receipts/<launch-id>/<job-key>.json
```

Immutable real-art evidence copies live by content hash at:

```text
$CLAWMARKS_STATE_DIR/evidence_bundles/sha256/<digest>
```

Verified generation backups live outside the expedition namespace at:

```text
$CLAWMARKS_STATE_DIR/backups/<expedition>/<leg>/<timestamp>-<launch-id>/
```

`CLAWMARKS_STATE_DIR` keeps its existing XDG default. These directories remain separate from
RunPod-billed generation output, so saving a note cannot overwrite image data. Every write uses a
temporary file in the same directory, flushes and `fsync`s the file, calls `os.replace()` only
after successful serialization, then `fsync`s the parent directory. Code must never delete a good
record before writing its replacement. The implementation extends `atomic_io.py` to provide this
durability rule in one shared helper.

The server creates IDs with UUIDs:

```text
focus_<uuid-hex>
trial_<uuid-hex>
```

The UI may display a short suffix, but URLs, API payloads, manifests, and logs use the complete ID.

## Focus Schema

Version 1 stores this shape:

```json
{
  "schema_version": 1,
  "focus_id": "focus_<uuid>",
  "label": "Ink anchor",
  "revision": 3,
  "status": "open",
  "scope": {
    "expedition": "trent_v3_epoch4",
    "leg": "freeform1"
  },
  "source": {
    "view": "map",
    "kind": "map_members",
    "member_tags": ["image-tag"],
    "real_anchor_tags": ["real-tag"],
    "projection_hint": {
      "projection_version": "sha256:<digest>",
      "polygon": [[0.21, 0.38], [0.44, 0.52]]
    }
  },
  "question": "Can the ink-heavy family survive unfamiliar subjects?",
  "observation": "Six distinct clusters share one real-art anchor.",
  "hypothesis_text": "Hard contour and paper texture will persist when the subject changes.",
  "test_contract": {
    "intention": "Separate subject from mark-making.",
    "evidence_scope": "The 18 saved members and their nearest real-art anchor.",
    "changed_variable": "Animal subject",
    "held_constant": ["composition", "strength", "CFG", "sampler", "steps"],
    "expected_move": "Results remain near the saved members and real-art anchor.",
    "evidence_against": "Results leave the neighborhood, lose the anchor, or collapse into one composition."
  },
  "created_at": "2026-07-16T00:00:00Z",
  "updated_at": "2026-07-16T00:00:00Z"
}
```

Allowed `status` values are `open` and `archived`. Archiving preserves the record and every linked
trial.

### Selection authority

`source` is a discriminated union. `kind: "map_members"` requires at least one validated
`member_tag`. Every generated member must resolve exactly once in the scored manifest for the
Focus's expedition and leg; a tag from another leg is invalid. `member_tags` and high-dimensional
neighbor calculations define the durable evidence. `projection_hint` only redraws the approximate
lasso and may drift when the projection is rebuilt.

`kind: "coverage_frontier"` requires faithfulness and novelty ranges plus at least one adjacent
member tag that resolves exactly once in the scoped manifest. Each range contains exactly two
finite values, `min < max`, within that Coverage metric's declared domain. The server also verifies
that the saved bin is empty and adjacent to a populated bin at creation time. `score_ranges`,
`adjacent_member_tags`, and saved real anchors define the durable evidence. `coverage_hint` may
store grid row, column, binning version, metric domains, and display bounds, but the UI must not
treat a grid coordinate as stable across a rebuilt Coverage map.

```json
{
  "view": "coverage",
  "kind": "coverage_frontier",
  "score_ranges": {
    "faithfulness": [0.48, 0.55],
    "novelty": [0.61, 0.69]
  },
  "adjacent_member_tags": ["image-tag"],
  "real_anchor_tags": ["real-tag"],
  "coverage_hint": {
    "binning_version": "sha256:<digest>",
    "metric_domains": {
      "faithfulness": [-1.0, 1.0],
      "novelty": [0.0, 2.0]
    },
    "row": 4,
    "column": 3
  }
}
```

Every real-art tag must resolve exactly once in the configured real-art source. The server preserves
tag order while removing duplicates. It applies the required fields for the selected source kind
rather than accepting any generic combination of tags and ranges.

### Natural language and test contract

The server preserves `question`, `observation`, and `hypothesis_text` exactly as the researcher
saved them. The Guide may propose edits, but it cannot silently rewrite these fields.

`test_contract` remains nullable while the Focus is in Scout or Explain. Cockpit requires all six
contract fields before a Focus-derived trial can reach confirmed status.

## Focus API

All mutating requests use explicit expedition and leg values. The server rejects a scope mismatch
even when the mismatched pair equals the global active selection.

```text
GET    /api/foci?expedition=<name>&leg=<name>&status=open
POST   /api/foci
GET    /api/foci/<focus-id>?expedition=<name>&leg=<name>
PATCH  /api/foci/<focus-id>?expedition=<name>&leg=<name>
POST   /api/foci/<focus-id>/archive?expedition=<name>&leg=<name>
```

`POST /api/foci` accepts scope, source selection, and optional text fields. The server assigns ID,
revision 1, status, and timestamps.

`PATCH /api/foci/<focus-id>` requires `expected_revision`. A successful edit increments the
revision and returns the complete new record. A stale edit returns HTTP 409 with the current
record; it never overwrites a newer revision.

Archive requires `expected_revision`, changes status to `archived`, increments the revision, and
returns the complete record. It follows the same HTTP 409 conflict rule as PATCH.

Map, Coverage, and evidence pages fetch live derived evidence separately. A Focus response may
include a non-persisted `derived` object with missing-member count, current human judgments,
redundancy summary, and current nearest neighbors. The API labels these values with the calculation
time and never folds them into a saved revision without an explicit edit.

## Trial Schema

A trial starts as a draft copied from one Focus revision:

```json
{
  "schema_version": 1,
  "trial_id": "trial_<uuid>",
  "status": "draft",
  "scope": {
    "expedition": "trent_v3_epoch4",
    "leg": "freeform1"
  },
  "focus_id": "focus_<uuid>",
  "focus_revision": 3,
  "focus_snapshot": {},
  "evidence_snapshot": {
    "source_manifest_sha256": "sha256:<digest>",
    "generated_members": [
      {
        "tag": "image-tag",
        "file_sha256": "sha256:<digest>",
        "manifest_record": {"tag": "image-tag", "centroid_sim": 0.55, "novelty": 0.61},
        "manifest_record_sha256": "sha256:<canonical-record-digest>",
        "human_judgments": {
          "favorite": true,
          "rating": null,
          "comparisons": [
            {"comparison_id": "comparison-id", "winner": "image-tag", "loser": "other-tag"}
          ]
        }
      }
    ],
    "real_anchors": [
      {
        "tag": "real-tag",
        "file_sha256": "sha256:<digest>",
        "evidence_bundle_sha256": "sha256:<digest>"
      }
    ],
    "derived_records": [
      {
        "kind": "redundancy_summary",
        "inputs": {"threshold": 0.74, "member_tags": ["image-tag"]},
        "value": {"cluster_count": 6},
        "calculator_version": "sha256:<digest>"
      }
    ],
    "snapshot_sha256": "sha256:<canonical-snapshot-digest>",
    "captured_at": "2026-07-16T00:00:00Z"
  },
  "payload": {
    "prompt": "...",
    "negative_prompt": "...",
    "strength": 1.0,
    "cfg": 7.5,
    "sampler": "DDIM",
    "steps": 28,
    "seeds": [1, 2, 3, 4]
  },
  "payload_sha256": "sha256:<digest>",
  "paid_launch_id": null,
  "result_tags": [],
  "evaluation": null,
  "created_at": "2026-07-16T00:00:00Z",
  "confirmed_at": null,
  "completed_at": null,
  "failure": null
}
```

`focus_snapshot` contains the complete Focus record at `focus_revision`. `evidence_snapshot`
captures the exact manifest, image content, human judgments, and derived calculations shown before
the paid action. It stores canonical values, not IDs alone. At confirmation, the server copies every
real anchor into the content-addressed evidence bundle and verifies its digest. Paths inside the
snapshot are relative to the validated leg, backup, or evidence-bundle root. Later Focus, manifest,
image, preference, or calculator changes do not change the trial's question, evidence, or success
criteria.

Trial creation takes the leg write lock while reading the manifest, images, judgments, and derived
records, so the snapshot cannot combine two concurrent states. At launch, the complete verified leg
backup makes every generated member recoverable. The safety receipt binds each generated evidence
digest to its relative path in that backup and each real anchor to its evidence-bundle digest. The
server creates a missing evidence-bundle object through a temporary file, file `fsync`, atomic
replace, and parent-directory `fsync` before confirmation succeeds.

Allowed trial statuses are `draft`, `confirmed`, `launching`, `running`, `completed`, `failed`, and
`cancelled`. The server may edit a draft. Confirmation freezes `scope`, `focus_id`,
`focus_revision`, `focus_snapshot`, `evidence_snapshot`, `payload`, and `payload_sha256`. It
recomputes and verifies every evidence and payload digest before changing status. Launch creates a
generic paid-launch record and stores its ID on the trial. That launch record is the source of truth
for preflight, dispatch, running, and recovery state; trial lifecycle fields are a reconciled UI
summary and never authorize dispatch. Later writes may change only lifecycle fields, the paid-launch
link, result tags, failure details, and the human evaluation. An evaluation records one judgment from
`supports`, `challenges`, `inconclusive`, or `not_tested`, freeform notes, and a review timestamp.
Only the researcher can submit or revise it.

## Cockpit Handoff

`GET /cockpit.html?expedition=<name>&leg=<name>&focus_id=<id>` loads the Focus from the explicit
scope, then verifies that the persisted record matches it. This GET request must not call
`_set_active_selection`, create a cockpit leg, or write `active_leg.json`.

Cockpit shows:

- Focus label, revision, expedition, and leg;
- the natural-language hypothesis;
- the six-part test contract;
- representative members and real anchors;
- the exact changed variable and held constants;
- the exact generation payload;
- job count, cost estimate, spend cap, output path, and backup status;
- a return link to the source evidence view.

Cockpit can create and edit a draft through:

```text
POST   /api/foci/<focus-id>/trials?expedition=<name>&leg=<name>
GET    /api/trials/<trial-id>?expedition=<name>&leg=<name>
PATCH  /api/trials/<trial-id>?expedition=<name>&leg=<name>
POST   /api/trials/<trial-id>/confirm?expedition=<name>&leg=<name>
POST   /api/trials/<trial-id>/launch?expedition=<name>&leg=<name>
```

Trial creation requires `expected_focus_revision`. If the Focus changed after the researcher loaded
it, creation returns HTTP 409 with the current Focus and creates no draft. Confirmation and launch
remain separate actions. Confirmation validates the frozen research brief and evidence snapshot.
Launch performs live safety checks and starts paid work.

The existing standalone Cockpit remains available for shipped freeform generation. It must label
those jobs as having no Focus provenance, must not impersonate a Focus-derived trial, and must use
the same paid-launch safety gate described below.

## Paid Launch Record

Every paid route creates one durable launch record, whether or not it came from a Focus:

```json
{
  "schema_version": 1,
  "launch_id": "launch_<uuid>",
  "launch_request_id": "request_<uuid>",
  "route_kind": "focus_trial",
  "status": "preflight",
  "scope": {"expedition": "trent_v3_epoch4", "leg": "freeform1"},
  "focus_id": "focus_<uuid>",
  "focus_revision": 3,
  "trial_id": "trial_<uuid>",
  "payload_sha256": "sha256:<digest>",
  "output_path": "expeditions/trent_v3_epoch4/freeform1",
  "maximum_job_count": 4,
  "accepted_job_count": 0,
  "attempt_number": 1,
  "attempts": [
    {
      "number": 1,
      "external_dispatch_started": false,
      "started_at": "2026-07-16T00:00:00Z",
      "preflight_error": null
    }
  ],
  "job_slots": [],
  "account_reservation": null,
  "safety_receipt": null,
  "created_at": "2026-07-16T00:00:00Z",
  "updated_at": "2026-07-16T00:00:00Z"
}
```

Allowed `route_kind` values initially cover `focus_trial`, `standalone_cockpit`,
`counterfactual`, and `overnight_search`. Future paid endpoints must add an explicit kind before
they can use the gate. `focus_id`, `focus_revision`, and `trial_id` are non-null only for
Focus-derived work. `launch_id` is the universal paid-work and recovery identity.

Allowed launch statuses are `preflight`, `dispatching`, `running`, `needs_reconciliation`,
`completed`, `failed`, and `cancelled`. Every non-Focus route returns its `launch_id` and links its
status UI to this record. A Focus trial also stores `paid_launch_id`; reconciliation can repair that
link from the launch record's `trial_id` after a crash between the two atomic file writes.

## Paid-Work Safety

One shared paid-image-launch gate covers Focus trials, standalone Cockpit jobs, counterfactual
batches, overnight search, retries, and every future image-generation endpoint. A route cannot
dispatch RunPod work directly. It submits an explicit scope, immutable payload digest, maximum job
count, user spend cap, and idempotency key to this gate. OpenCode text assistance uses the separate
Guide or existing prompt-generation boundary and does not consume this RunPod reservation.

### Lock and idempotency

The gate uses two protections:

1. a cross-process paid-work lease for the expedition and leg, held from launch claim through result
   reconciliation, which prevents two generators from writing to the same leg;
2. a shorter cross-process leg write lock, respected by every application write under the leg
   directory, held while the complete backup and verification run.

Locks live under `$CLAWMARKS_STATE_DIR/locks/`, outside the expedition namespace. A Python thread
lock alone does not satisfy this requirement. The paid-work lease records launch ID, process ID,
process start time, and heartbeat. A dead process with `external_dispatch_started: true` or unknown
does not release the lease automatically; it moves the launch to `needs_reconciliation`. A dead
process with a durable pre-dispatch state may release the lease and retry safely.

Every route accepts `launch_request_id`. For a Focus trial, the server holds the trial-record lock,
verifies `confirmed`, creates and durably writes the paid-launch record with that request ID first,
then atomically writes `status: launching` and `paid_launch_id` together on the trial before any
backup or external request. A crash between those file writes leaves an identifiable preflight
launch, not a stranded trial. A same-ID retry searches the scoped paid-launch records and repairs
that trial link before creating anything. The trial lock prevents two concurrent requests from
creating two launch records. A different request ID conflicts while the route object links an active
or unreconciled launch.

Each launch retains every attempt. A durable preflight failure records
`external_dispatch_started: false`, returns the Focus trial to `confirmed` when applicable, and
permits the same request ID to start the next numbered attempt. It never erases the failed receipt.
A crash or failure after dispatch starts must reconcile provider submission intents and receipts;
it cannot return to preflight or create another launch.

The gate also uses an account-wide cross-process reservation lock. Under that lock it fetches the
current provider balance, subtracts every non-terminal launch reservation, verifies the new
worst-case reservation and balance floor, persists the reservation, and only then releases the
lock. A retry keeps the same reservation identity. If the failed attempt retained an active
reservation, the retry does not add another; if the attempt released it before dispatch, the retry
reacquires and revalidates it under the account lock. Terminal reconciliation releases it under the
same lock. `$CLAWMARKS_STATE_DIR/paid_launches/account_reservations.json` keys reservations by a
non-secret provider/account identifier so separate provider accounts do not share one balance.

### Complete backup

The backup source is the complete `config.leg_dir(expedition, leg)`, including images, manifests,
embeddings, thumbnails, curation records, and caches. The destination is the dedicated
`$CLAWMARKS_STATE_DIR/backups/` tree, never a sibling of a leg. The gate holds the leg write lock,
copies every file, `fsync`s every copied file and directory through the backup root, and verifies
both file count and content before persisting success. A narrowed trial subdirectory backup does
not satisfy this requirement.

### Enforced spend cap

The gate calculates a conservative worst-case estimate before dispatch. The receipt records the
price source, unit price, runtime or per-job assumption, maximum job count, estimate, account
balance, balance floor, retrieval timestamp, and spend cap. If the provider offers no current value
that produces a defensible upper bound, launch fails closed.

While holding the account reservation lock, the gate requires:

```text
worst_case_estimate_usd <= spend_cap_usd
account_balance_usd - active_reservations_usd - worst_case_estimate_usd >= balance_floor_usd
dispatch_job_count <= floor(spend_cap_usd / worst_case_unit_cost_usd)
```

Dynamic searches use their maximum dispatch count and hard stop budget, not the first batch size.
The worker receives that maximum and cannot submit more jobs than the receipt allows. Before any
network request, a worker claims one stable job slot under the launch-record lock. Retries reuse the
same slot and deterministic job key. The number of claimed slots and accepted provider jobs can
never exceed `maximum_job_count`, even with several worker threads or process restarts.

### Launch sequence

Before dispatch, the gate must:

1. create or resume the idempotent paid-launch record and claim the paid-work lease;
2. resolve and display the immutable output path;
3. verify that the live payload matches the frozen digest;
4. verify that the frozen evidence snapshot still matches its files and manifest;
5. reserve the worst-case cost under the account-wide lock;
6. take, `fsync`, and verify the complete leg mirror while holding the leg write lock;
7. bind generated evidence to that backup and real anchors to the evidence bundle;
8. persist and `fsync` the complete safety receipt;
9. freeze the receipt and dispatch no more than its allowed jobs;
10. move the launch to `running` after the first accepted provider job is durable.

A failed lock, backup, digest, cost lookup, balance check, payload comparison, or receipt write
blocks dispatch and releases any reservation acquired by that attempt. The worker receives the
exact launch ID, expedition, leg, output path, nullable Focus and trial provenance, payload digest,
and safety receipt. It must not re-resolve them from mutable global state.

## Result Provenance

Every generated result manifest entry records:

```json
{
  "launch_id": "launch_<uuid>",
  "route_kind": "focus_trial",
  "focus_id": "focus_<uuid>",
  "focus_revision": 3,
  "trial_id": "trial_<uuid>"
}
```

`launch_id` and `route_kind` are present for every paid result. Focus and trial fields are nullable
for standalone Cockpit, counterfactual, and overnight-search work; they are required for
Focus-derived work.

Before each provider request, the worker claims a stable job slot and atomically writes and `fsync`s
a submission-intent receipt with launch ID, nullable Focus and trial provenance, intended tag and
relative path, payload digest, attempt number, and a deterministic provider idempotency key. The
provider must support idempotent submission by that key or querying jobs by that key. If it supports
neither, the gate fails closed before dispatch.

After the provider accepts the job, the worker extends the same receipt with provider job ID and
submission time. A crash between the network response and local update reconciles by the persisted
key. Automatic resubmission is allowed only when the provider enforces idempotency for that key. A
query-only provider may submit again only after an authoritative result guarantees the first
request was not accepted; an inconclusive or eventually consistent lookup leaves the launch in
`needs_reconciliation`. It never creates a new key. After download, the worker writes the image
through a temporary file, `fsync`s and atomically replaces it, `fsync`s the parent directory, then
extends the receipt with the image checksum and completion time. Manifest, launch, and trial updates
happen only after that receipt is durable.

If image generation succeeds but a later manifest or trial update fails, recovery preserves the
image. An idempotent provenance-repair endpoint consumes the receipts, verifies checksums, and adds
only missing manifest or trial links. It reports conflicts for human review. It never deletes an
unregistered image or replaces a conflicting record to make the files look consistent.

Scan, Archive, Solution Map, Coverage, and Runs always use `launch_id` to reach the paid-work receipt.
When Focus and trial provenance is present, they also mark result images and link back to the source
Focus. Learn records the researcher's judgment in the trial evaluation. A later Focus revision may
summarize that outcome, but neither action mutates the frozen trial snapshot.

## Stale and Missing Data

- Missing member images remain listed by tag with a visible missing count.
- A rebuilt projection may redraw a lasso differently, but it cannot change membership.
- A changed human preference or redundancy threshold appears as live derived evidence with a
  timestamp.
- An archived Focus remains readable from every linked trial and result.
- A malformed record returns a readable integrity error and preserves the file for recovery.

## Acceptance Criteria

- Focus records persist outside the repository and outside generation output directories.
- Every update is atomic and revision-checked.
- Solution Map and Coverage create valid source-specific Focus records.
- Every evidence tool can open a Focus without relying on the global active leg.
- Cockpit GET requests perform no selection mutation.
- A confirmed trial preserves the complete source Focus revision and exact payload.
- Every paid generation route uses one idempotent launch gate and cannot launch without a verified
  durable complete-leg backup, conservative estimate, enforced spend cap, account-wide reservation,
  and available balance.
- Every successful paid result carries launch provenance; Focus-derived results also carry Focus and
  trial provenance.
- Every provider request has a durable intent with an idempotency key before dispatch, and every
  accepted job has a durable provider receipt before image download.
- Every partial failure preserves images and the last good JSON record.
