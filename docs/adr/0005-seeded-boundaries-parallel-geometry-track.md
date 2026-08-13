# ADR-0005 — Seeded cell boundaries as a second geometry track

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** Josh Boquiren
- **Relates to:** [ADR-0003](0003-centroid-canonical-roi-stamps.md) (canonical ROI stamps —
  amended in scope, not reversed), [ADR-0004](0004-tracking-hitl-additive-edit-logs.md)
  (the edit logs this reads through)

## Context

ADR-0003 replaced every detector-native boundary with a fixed-radius disk, because
session-to-session segmentation shape variance was leaking into the ROICaT embeddings that
decide cross-session cell identity. That decision stands and its mechanism is unchanged.

What it also did — listed under its own "costs" section — was leave the project with no real
cell boundary anywhere. `centroid_masks.py` stamps disks; `centroids.py:322` computes
Cellpose's masks and then discards everything about them except `center_of_mass`. A disk is
the right input for ROICaT and the wrong thing to show a human reviewing a soma, and it is a
poor spatial profile for subtraction and trace extraction.

Meanwhile ADR-0004 produced something that did not exist when ADR-0003 was written: a set of
soma centroids a human has confirmed and linked across sessions. Cellpose still segments these
FOVs free-hand, with no knowledge of where those confirmed cells are.

## Decision

Add a **second geometry track** rather than changing the first.

| Artifact | Geometry | Consumer |
| --- | --- | --- |
| `merged_masks.tif` | fixed-radius disks (ADR-0003, unchanged) | cross-session registry / ROICaT |
| `boundaries.tif` | seeded segmentation | `/cells` page, humans, future trace work |

Both are stamped from the same effective centroids — `centroids.json` replayed through
`corrections/centroids.jsonl` — and carry the **same label ids**, so
`CellObservation.local_label_id` addresses a cell in either image and the registry needs no
knowledge that the second one exists. ADR-0003's `_MAX_STAMP_RADIUS = 20` cap is a property of
ROICaT's fixed 36×36 ROI crop and stays exactly where it is; boundaries never reach ROICaT.

### Mechanism: flow field for extent, seeds for identity

Cellpose forms masks in two steps — a learned flow field `dP` plus `cellprob`, then pixel
dynamics followed by histogram-peak clustering of the converged pixels. Step one is learned;
step two is a heuristic with no idea which cells are real. `seeded_masks.py` keeps step one and
replaces step two:

1. **extent** — a pixel is cell material if its trajectory converges within `capture_px` of
   *some* confirmed seed. Basins attracting no seed are dropped.
2. **identity** — within that extent, `watershed(-cellprob, markers=seeds)` decides which cell
   each pixel belongs to.
3. **cleanup** — per label, keep the connected component holding its own seed, fill holes,
   apply area bounds.
4. **fallback** — a seed owning no pixels gets the canonical disk, so a confirmed cell can
   never vanish from the output.

**Why step 2 is not just "nearest seed in converged space."** Measured on two synthetic somata
36 px apart: Cellpose emitted a single 3198-px label whose flow field has *one* attractor at
(99.7, 99.9), ~18 px from both true centroids. Every pixel converges to the same point, so a
nearest-seed rule assigns nothing at any sane `capture_px`. The network merged the cells in
flow space and no partition of that space can undo it. The watershed does: 1334 + 1864 px
(= 3198 exactly), centroids recovered at (100.5, 80.7) and (98.4, 115.9) against truth
(100, 82) and (100, 118). Extent and identity are different questions; the flow field only
answers the first.

### Flow cache

Boundaries must redraw on every HITL centroid edit, and re-running inference per edit is far
too slow for that loop. Centroid discovery therefore persists `dP`/`cellprob` under
`flows/`, keyed on the same resolved-params + schema dict `centroids.json` already uses for
resume. Redrawing is then pure numpy/skimage — no GPU, no Cellpose import. A missing or stale
cache reads as absent and the FOV keeps its disks.

## Measured outcome

`scripts/boundary_bakeoff/`, 5 cranial-window FOVs, 782 hand-drawn ImageJ somata (median
equivalent diameter 18.1 px), deployed checkpoint at diameter 18, `capture_px` 12, stamp
radius 9, IoU ≥ 0.3:

| arm | precision | recall | mean IoU | mean IoU (flow-derived only) |
| --- | --- | --- | --- | --- |
| free Cellpose | 1.000 | **0.143** | 0.714 | — |
| disk stamps (today) | 0.996 | 0.996 | 0.640 | — |
| seeded | 0.981 | 0.977 | 0.636 | **0.668** on 251/782 |

Read carefully, because three of those numbers are traps:

- **Free Cellpose's 0.714 is not a win.** It is measured over the 14% of cells it segments at
  all — a self-selected easy subset. Its recall is the headline finding here, and it is about
  Stage 1, not about this change.
- **The seeded arm's 0.636 mostly reports the disk arm back.** 531 of 782 seeds captured no
  flow basin (the detector never fired there) and fell back to a disk, so the aggregate is
  dominated by disks. `mean_iou_flow` exists specifically to separate these.
- **The real comparison is 0.668 vs 0.640** — seeded boundaries beat disks by ~0.03 IoU on the
  cells where a flow field exists, while recall stays at 0.977 because of the fallback.

Verified separately that the partition is faithful: on cells where Cellpose fires and one seed
sits in one basin, seeded and free-Cellpose boundaries score *identically* (0.661 and 0.719 on
two FOVs). The seeded path does not degrade the model's own boundary; it re-partitions it.

## Consequences

**Benefits**

- Every confirmed cell gets a boundary, always — recall 0.977 against a detector whose own
  recall is 0.143 on the same data.
- Basins nobody confirmed are dropped rather than shown as cells.
- Merged somata are split by the humans who marked them, which the flow field alone cannot do.
- ADR-0003's registry-stability argument is untouched: no real boundary reaches ROICaT.
- The registry required zero changes — shared label ids carry the whole integration.

**Costs / accepted tradeoffs**

- **The measured win is small and this dataset cannot show a large one.** These somata are
  near-circular with ~18 px equivalent diameter, so a radius-9 disk is already close to the
  ideal shape and there is little headroom. The motivating case is prism FOVs with larger and
  less regular somata, and **no boundary ground truth exists for those** — the bake-off cannot
  currently evaluate the case this was built for. That is the main open risk.
- **The fallback rate is high and is a detector problem, not a boundary problem.** 68% of seeds
  on the bake-off FOVs had no flow basin. Seeded boundaries make this visible
  (`boundaries.json:n_disk_fallback`) rather than fixing it.
- **A second on-disk geometry.** Two label images per FOV that must agree on label ids. The
  invariant is enforced by both reading `load_effective_centroids`, and tested, but it is a new
  way for the workspace to become internally inconsistent.
- **~6 MB/FOV at 512², ~24 MB at 1024²** for the flow cache. `centroid_persist_flows=False`
  opts out, at the cost of boundaries.
- **`capture_px` is a new knob.** Measured on the bake-off FOVs it barely matters (fallback
  rate moved 419→393 across 6→45 px, saturating), because fallbacks are driven by the detector
  not firing rather than by the capture radius. It may matter more on FOVs where it does fire.

  *Resolved.* The Boundaries page (`roigbiv/ui/pages/boundaries.py`) is the surface: it
  redraws live as the knob moves, and reports seeds / disk fallbacks / orphan pixels beside
  the picture — plus the disk area each boundary replaces, so "this FOV gains nothing from
  seeding" is readable rather than inferred. It says outright that a majority-fallback FOV is
  a detector problem, since that is the conclusion the measurement above supports and the one
  a tuning control would otherwise invite the user to disbelieve. Live tuning is only possible
  because `converge_pixels` does not depend on `capture_px` and is cached per FOV
  (`roigbiv/ui/services/boundary_preview.py`); an explicit save pins the setting into
  `boundaries.json`'s `settings` block, which later automatic redraws honour.
- **cpsam backend gets no boundaries.** The sidecar returns no flow field across the process
  boundary; `run_cellpose_flows` raises and the FOV keeps disks.

**Explicitly out of scope**

Stage 1 seeding, any change to ADR-0003's post-gate canonicalization, re-extracting traces or
neuropil from real boundaries, and hand-drawing a boundary on `/cells` (centroid edits redraw
boundaries; drawing one directly is a separate feature).

## References

- `roigbiv/pipeline/seeded_masks.py` — `converge_pixels`, `seeded_labels`.
- `roigbiv/pipeline/boundaries.py` — `write_boundaries`, the artifact contract.
- `roigbiv/pipeline/stage1.py` — `run_cellpose_flows`, `CellposeFlows`.
- `roigbiv/pipeline/centroids.py` — flow cache write + recompute key.
- `roigbiv/ui/services/tracked_cells.py` — `_render_geometry`.
- `roigbiv/ui/pages/boundaries.py` + `roigbiv/ui/services/boundary_preview.py` — the tuning
  surface, and the cache that makes it live.
- `roigbiv/registry/cell_edits.py` — `_redraw_boundaries`, which keeps the two geometry
  tracks from disagreeing after a HITL edit.
- `scripts/boundary_bakeoff/` — the measurement above.
- [ADR-0003](0003-centroid-canonical-roi-stamps.md) §Costs — "subtraction/residual fidelity"
  and "trace/neuropil precision", the costs this ADR starts paying back.
