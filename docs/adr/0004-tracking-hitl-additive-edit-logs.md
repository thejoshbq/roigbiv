# ADR-0004 — Cross-session tracking HITL as two additive edit logs

- **Status:** Accepted
- **Date:** 2026-08-12
- **Deciders:** Josh Boquiren
- **Relates to:** [ADR-0003](0003-centroid-canonical-roi-stamps.md) (canonical stamps,
  and its "HITL corrections are exempt" clause, whose reasoning this ADR extends to
  centroids); `roigbiv/pipeline/corrections.py` (the pre-existing ROI-polygon HITL pattern
  this design mirrors)

## Context

Cross-session tracking (`roigbiv/pipeline/workspace.py::run_tracking`) produces two kinds
of error a human can see on the `/cells` page but, until now, could not correct: a missed
or misplaced soma centroid, and a cell the matcher failed to (or wrongly did) link across
sessions. On the reference three-session prism FOV, recall against a ground-truth
correspondence set was 0.778 — roughly one in five true cell correspondences invisible on
screen with no way to fix it short of re-running the whole pipeline and hoping.

Two constraints shaped the design:

1. **`merged_masks.tif` labels were positional.** `stamp_centroids` assigned label `1..N`
   by a centroid's position in `centroids.json`, and that label is exactly what
   `cell_observation.local_label_id` stores. Deleting one centroid would renumber every
   later one and silently invalidate every observation referencing them.
2. **Observations are fully derived data.** `_register_auto_match` always does
   `delete_observations_for_session` + `insert_observations` from ROICaT's cluster labels
   — nothing about a registration is additive. A centroid edit also changes the FOV
   fingerprint (`fingerprint.py`), so the next `run_tracking` pass misses the idempotency
   guard and fully re-registers, which would silently discard any human edit made since the
   last run unless something explicitly replayed it back in.

## Decision

**Labels became explicit rather than positional.** `stamp_labeled_centroids(labeled: dict[int,
(y, x)], shape, radius)` stamps a caller-supplied `{label: (y, x)}` mapping, iterating labels
in ascending order — the same order positional stamping always produced, so an unedited
workspace's `merged_masks.tif` is byte-identical to before. `stamp_centroids` is now a
one-line wrapper: `stamp_labeled_centroids({i: p for i, p in enumerate(points, 1)}, ...)`.
No migration, no behavior change, and now every op and every observation can name a label
that survives a delete elsewhere.

**Two new append-only JSONL logs**, following `corrections.py`'s established pattern —
pure replay, undo by dropping the tail and rewriting:

| Log | Scope | Path | Ops |
| --- | --- | --- | --- |
| Centroid edits | per session | `{output_dir}/corrections/centroids.jsonl` | `add{label,y,x}` · `delete{label}` · `move{label,y,x}` |
| Correspondence edits | per FOV | `{input_root}/corrections/matches/{fov_id}.jsonl` | `link{members}` · `unlink{member}` |

`centroids.json` itself is **never rewritten** — same reasoning as ADR-0003's "HITL
corrections are exempt" clause, extended to centroids: it is detector output,
`run_centroid_discovery` short-circuits on a params/schema match, and a hand-placed
centroid has no meaningful `npix`/`cellpose_prob`/`activity_support` to fabricate.
`centroids.json` = base, the log = the only mutation record, `merged_masks.tif` = the
materialization.

Match ops key on `(session_stem, local_label_id)`, never on `global_cell_id` — a fresh
ROICaT run mints new cell ids from scratch, so a log that named ids would go stale on
every re-match. One log **per FOV** (not a single workspace-wide log) so an "undo last"
on one FOV can never touch another's.

**A single DB materializer, `roigbiv.registry.cell_edits.apply_tracking_edits`, is the one
place both a fresh `run_tracking` pass and an interactive `/cells` edit reach the same end
state.** It re-stamps every session (replaying the centroid log), builds a
`(stem, label) -> global_cell_id` assignment restricted to labels the stamp actually
produced — never to what an op merely asked for, since a `move` can bury one label
completely under another — replays the match log over that assignment, and writes the
result with one `store.replace_observations(...)` call. No ROICaT match runs: a moved
centroid keeps its cell, a deleted one drops its observation, an added one gets a
deterministic new cell (`uuid5`, not `uuid4`, so replay is idempotent) the human can then
link. `run_tracking` calls it once per FOV after the session loop — not a nicety, since
without it a centroid edit followed by a re-run would be actively destructive (see
constraint 2 above).

**`link` merges whole cells, associatively, and rejects two members from one session
outright** rather than applying a merge that would corrupt "one member per session"
readers downstream (`ui/services/tracked_cells.py`'s label index, `anomalies.py`'s
`CellTimeline.local_label_ids`).

## Consequences

- **A new store method, `replace_observations`, is the only atomic write in the store.**
  Every other method opens its own session and commits; a relink is necessarily a
  delete-then-insert under `UniqueConstraint(session_id, local_label_id)`, and the store's
  SQLite engine (`check_same_thread=False`, served by threaded Dash callbacks) can
  genuinely interleave two writes. This did not retrofit onto `orchestrator.py`'s existing
  (non-atomic) delete+insert pair — that stays as-is, a separate concern.
- **Centroid edits only apply to centroid-stamped sessions.** `write_merged_masks` already
  refuses to overwrite a full-cascade FOV's `merged_masks.tif` (real per-stage detections
  outrank centroid stamps); `apply_tracking_edits` reads `StampedMasks.written` to tell
  "wrote it" from "computed it but a cascade outranks it" and surfaces that as a note
  rather than a silent no-op. Link/unlink has no such restriction — it works on any FOV's
  observations regardless of how its masks were produced.
- **Cross-session linking is structurally impossible without ROICaT installed** (the
  `/cells` page is FOV-scoped, and without ROICaT every session becomes its own
  single-session FOV per `run_tracking`'s existing fallback) — not something this ADR
  changes, but worth naming since a link control with nothing to link against would
  otherwise look broken rather than absent for a stated reason.
- **`_MAX_STAMP_RADIUS = 20` remains untouched.** It exists to protect ROICaT's ROInet/SWT
  embeddings from a stamp that fills its fixed 36×36 crop; nothing here changes what those
  embeddings see, so revisiting the cap is a separate decision, not a consequence of this
  one.

## References

- `roigbiv/pipeline/centroid_masks.py` — `stamp_labeled_centroids`, `StampedMasks.written` /
  `.present_labels`, `load_effective_centroids`.
- `roigbiv/pipeline/centroid_edits.py` — the centroid op log.
- `roigbiv/registry/cell_edits.py` — the match op log and `apply_tracking_edits`.
- `roigbiv/registry/store/sqlalchemy_store.py::replace_observations`.
- `roigbiv/pipeline/workspace.py::run_tracking`'s post-loop replay call.
- `roigbiv/ui/pages/cells.py` — the `/cells` edit-mode UI this log design serves.
- [ADR-0003](0003-centroid-canonical-roi-stamps.md) — the "HITL corrections are exempt"
  precedent this ADR extends from ROI polygons to centroids.
