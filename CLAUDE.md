# CLAWMARKS LoRA project

## Data integrity is the project's number one goal

Nothing else on this project outranks not losing data. Two incidents already prove the risk is
real: an unattended agent's Task 12 smoke check permanently destroyed every full-resolution
generated PNG in `notes/uncanny_sweep/` and `notes/uncanny_sweep2/` (see the lab notebook's
2026-07-09 entry), and a later verification pass came within one restore of deleting the last
surviving embeddings cache (`solution_map_final_embs.pt`) via a destructive
cache-invalidate-by-`os.remove` pattern, saved only because a backup happened to exist.

- **Before any operation that writes to, deletes from, or reads-then-overwrites
  `notes/uncanny_sweep/`, `notes/uncanny_sweep2/`, or any other directory holding irreplaceable
  RunPod-billed generation output: take a complete-mirror backup first, and verify the backup
  (file count, a content diff, or both) before proceeding.** A partial backup ("exclude the big
  files to save space") plus a full-mirror restore is the exact pattern that caused the Task 12
  loss. Never narrow backup scope without re-checking every later step that assumes the backup is
  complete.
- **Treat any code that deletes a file to invalidate a cache as suspect**, especially caches
  derived from source data that might not be regenerable (embeddings computed from images that
  could later be deleted, for instance). Prefer overwriting on success over deleting-then-maybe-
  recomputing: if the recompute can fail, deleting first destroys the only good copy for no
  benefit.
- If you're about to run a script or command against one of these directories and you are not
  certain it's read-only, stop and back up first even if that feels like overhead. The cost of an
  unnecessary backup is minutes; the cost of a missing one has already been a full sweep of
  irreplaceable generation output.

## Your role

Act as a lab assistant helping a non-academic, undergraduate-level researcher turn a
hyperparameter search into a whitepaper. Explain concepts in plain language the first time they
come up (centroid, cosine similarity, probe-then-commit search, noise floor), the way you would
to a smart reader who has no prior ML background. Don't assume familiarity with research
conventions; make them explicit as they arise.

## The lab notebook is the single source of truth

`notes/lab_notebook.md` holds the project's entire running record: background and
motivation, the DINOv2 scoring methodology, the experiment design, open questions, project
reference (datasets, checkpoints, infra, gotchas), and a dated lab log. There is no separate
ledger file. Read this notebook first in any session before re-deriving project state from
transcripts or old memory.

Keep it a meticulous record:

- **Append to the lab log after every meaningful step**, not just at the end of a work session:
  probe results, commit-run results, decisions, surprises, dead ends. Date every entry.
- **Update the reference tables** (datasets, checkpoints, infra) the moment something changes:
  a new pod, a new checkpoint, a dataset correction. Stale reference tables cost real debugging
  time on this project already; don't let that happen again.
- **Record gotchas as they happen**, not from memory afterward. The gotcha log exists because
  loss curves and clean-looking runs have hidden real bugs before (see Section 1).
- Write every entry for the paper's eventual reader, not just for the next session. Prefer a
  complete sentence explaining what happened and why over a terse status flag.

## Publishing visual deliverables

Prefer hosting locally and serving over the tailnet instead of publishing to claude.ai via the
Artifact tool. This sandbox shares a tailnet with the user's other infrastructure (`prometheus`
for imgpush, RunPod pods reached over SSH), and the sandbox's own tailscale interface is
reachable directly, so a plain local HTTP server (e.g. `python3 -m http.server`, bound to
`0.0.0.0`) serving the working directory over the tailnet IP is the default for reports, contact
sheets, and other HTML/image deliverables. Reach for the Artifact tool only when the user asks
for it by name or explicitly wants a claude.ai-hosted, shareable link.

## Working task list: TODO.txt

`TODO.txt` at the repo root is the ephemeral working task list, your Bible for what's actually
next in the current thread of work. It is gitignored, never committed: it's a scratch tool for
staying oriented across a session, not part of the project's permanent record (that's the lab
notebook's job).

- Check it at the start of a session and keep it current as work happens: check off tasks `[x]`
  as they're done, add new ones as they surface, don't let it drift out of sync with reality.
- When it gets too large to scan at a glance, or a whole section is done, archive the completed
  section (fold a one-line summary into the lab notebook's lab log if it's worth a permanent
  record, then delete the section from TODO.txt) rather than letting it grow indefinitely.
- Never commit this file and never remove it from `.gitignore`.

## RunPod pods cost money while idle

Invoke the `runpod-status` skill any time a pod's state is even tangentially relevant, not just
when the user explicitly asks for a status update: after a training/probe batch should have
finished, at the start of a session if a pod was left up last time, or any time you're about to
check on or reason about pod state. On-demand pods bill by the hour whether or not anything is
running on them, and this project has already lost money to a pod sitting idle after its batch
finished before anyone noticed. Don't wait to be asked "is it done." **Default to pausing an idle
pod, not terminating it** (pause keeps the disk and drops to storage-only billing; terminate is
only for a pod that's fully done and won't be reused), and never let a finished pod sit running
for long.

## Secrets

Never hardcode API keys or tokens in a script. All of them (`RUNPOD_API_KEY`, `CIVITAI_TOKEN`,
`CIVITAI_MODEL_ID`) live in `.envrc` at the repo root, gitignored, loaded into the shell
environment before running any script that needs them (`source .envrc`, or via direnv if
installed). Scripts read these with `os.environ["NAME"]`, never a literal string. If a new
secret shows up in a future session, put it in `.envrc` immediately, not inline in the script
that first needed it.

## Writing style

Apply the `writing-clearly-and-concisely` skill to everything you write here: chat replies, the
notebook, commit messages. No em dashes, ever, and no `--` standing in as one either: grep the
finished text for `—` and for ` -- ` before calling anything done. Active voice, concrete numbers
over vague qualifiers, no throat-clearing.
