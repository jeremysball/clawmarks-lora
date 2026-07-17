# ADR 0002: Focus is explicit persisted research state

**Status:** Accepted
**Date:** 2026-07-16

## Context

The curation tools expose useful evidence but do not preserve the researcher's subject across
pages. Solution Map can identify a visual neighborhood and Coverage can identify an empty frontier,
yet Coverage's current Cockpit link carries no selected cell. Opening `/cockpit.html` also changes
the global active leg to an empty `cockpit` leg during a GET request. The researcher loses the
exact evidence, source scope, and question that motivated the trial.

Several representations could carry this state. A 2D lasso is easy to draw but its UMAP coordinates
can move when the projection rebuilds. A mutable global selection is convenient for rendering but
changes underneath tabs, bookmarks, background work, and verification. An immediate Cockpit draft
preserves the next action but gives the evidence no durable identity after the trial ends.

The workspace needs a stable object that survives tool navigation, supports several interpretations
and trials over time, and records which evidence existed before each paid action.

## Decision

The application will persist a **Focus** as an explicit expedition/leg-scoped research record under
`$CLAWMARKS_STATE_DIR/foci/`. A Focus stores stable generated-image tags, adjacent tags or metric
ranges for a frontier, nearby real-art anchors, source view, research question, observation,
natural-language hypothesis, and an optional test contract while the researcher scouts and
explains. Cockpit requires a complete contract before trial confirmation.

Routes and API requests carry the Focus ID, expedition, and leg explicitly. The server loads the
record from that scope and validates that the persisted values match. Reading a page, including
Cockpit, must not mutate the global active leg.

Stable member tags and high-dimensional image relationships define a map Focus. Metric ranges,
adjacent member tags, and real anchors define a Coverage frontier. Saved 2D polygons and grid
coordinates are display hints, not authoritative membership.

Each paid trial snapshots one complete Focus revision plus checksums and records for the evidence
shown before confirmation. The worker receives immutable scope, output path, payload digest, Focus
ID, and trial ID. Generated result records carry those IDs back to the evidence tools.

## Consequences

- A researcher can inspect one question across Map, Coverage, Scan, Compare, Redundancy, Novelty
  Decay, Lineage, Cockpit, and Runs without reconstructing the selection.
- Trial provenance survives later edits to the Focus because each trial stores its source revision.
- Bookmarks, multiple tabs, and background work no longer depend on one mutable active selection.
- The application gains new state that requires schema versioning, atomic writes, revision conflict
  handling, backup coverage, and readable corruption errors.
- Rebuilt projections may draw a saved lasso differently. The UI must explain that drift while
  preserving authoritative members.
- Existing leg-wide tools remain usable without a Focus, but Focus-derived trials require explicit
  provenance.
- The global active expedition/leg remains useful as a browsing default. It no longer serves as a
  research record or an implicit Cockpit input.

## Alternatives considered

- **Keep one mutable global selected region.** Rejected because tabs, bookmarks, GET requests, and
  background work can overwrite it. It also cannot preserve the evidence used by an earlier trial.
- **Carry selection only in URL parameters.** Rejected because complete member lists and research
  notes make fragile URLs, and projection coordinates still drift. URLs will carry the Focus ID,
  not the dossier itself.
- **Create a dedicated region workspace without a durable cross-tool object.** Rejected because it
  duplicates evidence pages and still needs identity and provenance once Cockpit or Runs opens.
- **Create an immediate Cockpit draft from every selection.** Rejected because it makes paid action
  the primary object, skips explanation, and cannot support several competing trials from one body
  of evidence.
- **Treat 2D UMAP coordinates as the durable region.** Rejected because UMAP is a lossy,
  non-deterministic display projection. Recomputed coordinates can move while image identity stays
  unchanged.

## Revisit this decision if

The project adopts a transactional database or a multi-user research service. The Focus concept and
explicit provenance should remain, but JSON file layout, revision handling, and ownership rules may
move into that storage layer.
