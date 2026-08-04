# ADR-0003 — Canonical fixed-radius ROI stamps replace detector-native boundaries

- **Status:** Accepted
- **Date:** 2026-08-03
- **Deciders:** Josh Boquiren
- **Supersedes / relates to:** [ADR-0001](0001-non-destructive-candidate-union.md) (targets
  `cascade_legacy`, the pipeline mode ADR-0001/[ADR-0002](0002-cascade-default-deprecation-criteria.md)
  keep as default and selectable indefinitely); [`docs/design/OVERVIEW.md`](../design/OVERVIEW.md)
  §9 "Canonical ROI stamps"

## Context

Each of the four detection stages persists its detector's actual boundary as the final
`ROI.mask`: Stage 1 keeps Cellpose's instance-segmentation contour, Stage 2 rasterizes
Suite2p's sparse pixel list, Stage 3 already builds a fixed disk (see below), and Stage 4
keeps a regionprops-derived connected component from bandpass-filtered correlation
contrast. That mask is load-bearing well past detection: Gates 1/2/4 compute
area/solidity/eccentricity from it, the subtraction engine (`subtraction.py`) builds its
per-ROI intensity-weighted spatial profile from its exact pixels, trace extraction
(`traces.py`) averages fluorescence over it and dilates it into a neuropil annulus, and the
cross-session registry's ROICaT matcher (`roicat_adapter.py`) crops it into a per-ROI image
patch that feeds the embedding used for cross-session cell identity.

That last dependency is the problem. Real segmentation boundaries are not stable for the
same physical cell across imaging sessions — different noise realizations produce
different Cellpose/Suite2p outputs even when nothing about the underlying tissue changed.
Session-to-session shape variance therefore leaks into the ROICaT embeddings used to decide
whether a cell today is the same cell as one from a prior session — a plausible source of
cross-session matching noise that a prior (unversioned, pre-dates this repo's ADR history)
iteration of this project ran into directly enough to motivate this change.

**Stage 3 is already a partial precedent.** `stage3.py::run_stage3` builds a disk mask
(`radius=cfg.spatial_pool_radius`) for every candidate *before* Gate 3 runs, so Gate 3's
solidity/eccentricity check has always been evaluated against a disk, not real geometry —
Stage 3 already relies on waveform R² / rise-decay-ratio as its real discriminator, not
shape. Nothing in this pipeline broke when one of its four stages stopped using
shape-based discrimination; that's the existing evidence this ADR generalizes from.

## Decision

Every `accept`/`flag` ROI, from all four stages, is canonicalized post-gate: its
detector-native mask is replaced with a fixed-radius disk (`cfg.roi_stamp_radius`, default
8 px) centered on the ROI's own centroid (`roigbiv/pipeline/roi_stamp.py::canonicalize`).
Concretely, for each stage, one call is inserted between that stage's gate decision and its
source-subtraction call (`run.py`):

```
detect (real, irregular boundary) → gate (evaluates real boundary) → canonicalize (stamp) → subtract
```

Gates 1, 2, and 4 are **unchanged** — they still receive and evaluate each stage's raw
detector output, so their morphology thresholds (`min_area`/`max_area`/`min_solidity`/
`max_eccentricity`/etc.) retain full discriminative power. `area`/`solidity`/
`eccentricity` on the `ROI` object remain that gate-time record and are not recomputed from
the stamp — they diverge from the persisted mask by design (see `types.py::ROI` docstring).
Reject-outcome ROIs keep their real mask; they're never subtracted, so canonicalizing them
would only erase useful audit information for no benefit.

Stage 3's existing pre-gate disk (keyed on `spatial_pool_radius`, its own trace-extraction
pooling radius — a distinct, still-meaningful parameter) is re-centered at
`roi_stamp_radius` post-gate exactly like the other three stages, rather than repurposing
`spatial_pool_radius` itself — `spatial_pool_radius` has exactly one consumer in the
codebase (that one disk-construction call), and overloading it with a second meaning would
make it impossible to tune trace-extraction pooling and output-stamp size independently.

**Crowding guard.** Fixed-radius disks can overlap in ways real (Cellpose-instance,
non-overlapping-by-construction) segmentation wouldn't, for cells whose true somata are
close together. `roi_stamp.py::resolve_crowding` runs after each stage's canonicalization,
over the cumulative accepted/flagged ROI pool: two centroids closer than one stamp radius
apart demote the weaker ROI (lower confidence tier, tie-broken by earlier stage then lower
label id) from `accept` to `flag` — the same accept-safe convention Gate 1's own
merge-peak check already uses (`gate1.py`, spec §6). The demoted ROI still enters
subtraction; the flag only surfaces it for HITL review. Numerical safety of the
simultaneous least-squares solve under overlapping profiles is the existing ridge
regularization's job (`subtract_ridge_lambda_scale`), not this guard's.

**Auto-scaling.** `roi_stamp_radius` is added to `optics.py::SCALE_DERIVED_FIELDS` and
`derive_scale_params()` with the same formula as `spatial_pool_radius`
(`max(4, round(soma_radius))`), so GRIN/PRISM/generic profiles each get a stamp sized to
their own measured soma scale rather than one hardcoded constant across the whole lab's
lens types.

**HITL corrections are exempt.** User-drawn correction polygons (`corrections.py`) are not
snapped to the canonical stamp — a human has already exercised shape judgment at that
point, and there's no gate/subtraction dependency downstream of a correction that would
need shape uniformity.

**Subtraction, traces, and the registry needed no code changes.** All three already consume
`roi.mask` generically; once the mask is canonical, they're canonical automatically. This
was verified by running the full pipeline/registry test suite unmodified after the change
(471 pipeline tests + 51 registry tests passed; the only two pre-existing failures —
`test_gpuguard.py::test_config_defaults` and
`test_workspace_isolation.py::test_explicit_cfg_takes_precedence_over_env` — reproduce
identically on `main` before this change and are unrelated to it).

## Consequences

**Benefits**

- Cross-session ROICaT footprint embeddings stop carrying session-to-session segmentation
  shape noise as a confound — the only thing that can vary between two sessions'
  embeddings for the same cell is the underlying image content inside a stamp of fixed
  shape and size, not the stamp's own boundary.
- One canonicalization function, four call sites, zero changes to the four gate files or
  to subtraction/traces/registry — a small, reviewable, and reversible change relative to
  its stated goal.
- Gate 1's own merge-peak precedent (`accept → flag`, never silent reject) generalizes
  cleanly to the new crowding guard rather than inventing a new demotion convention.

**Costs / accepted tradeoffs**

- **Subtraction/residual fidelity.** A canonical disk doesn't match a real soma's true
  extent. Over-subtraction (disk includes non-cell background/neuropil pixels) or
  under-subtraction (real signal extends past the disk) both leak into the next stage's
  residual. The subtraction engine's intensity-weighted profiles (`w[mask] = vals/peak`,
  not a flat indicator) partially self-correct but don't eliminate this.
- **Trace/neuropil precision.** Fixed-radius averaging is a real change from true-footprint
  averaging; some ROIs will get noisier or more neuropil-contaminated traces than today's
  real-mask extraction produces.
- **The core hypothesis is unvalidated.** Whether session-to-session shape variance is
  actually degrading cross-session matches in this lab's data was never measured before
  this ADR — no ablation or before/after accuracy comparison was run. This decision
  proceeds on the hypothesis alone (informed by the Stage-3 precedent and the mechanistic
  path from mask shape → ROICaT ROI-image crop → embedding). Recommended minimum
  follow-up: watch `pipeline_log.json:review_queue_summary.total` and the registry's
  `auto_match`/`review`/`new_fov` distribution on a real repeated-FOV dataset after this
  ships, even without a formal study.
- **Per-stage diagnostic artifacts now show canonical shape, not raw detection shape.**
  `stageN_masks.tif` and `stageN_report.json` are written *after* canonicalization (and,
  for Stage 2–4, after the cross-stage crowding guard), so they reflect the same shape
  subtraction/traces/registry consume — not what the detector originally proposed. The
  gate-time `area`/`solidity`/`eccentricity` fields remain the real-geometry record for
  anyone auditing why a gate decision was made.
- **Future `candidate_union` migration** (ADR-0001/0002): this canonicalization step lives
  in the `cascade_legacy` code path (`run.py`'s stage-by-stage gate→subtract sequence,
  still the only implemented and default mode per ADR-0002). If/when `candidate_union`
  ever becomes default, this logic needs re-homing into whatever that architecture's
  candidate/deconfliction pipeline looks like — out of scope here.

## References

- `roigbiv/pipeline/roi_stamp.py` — `disk_mask`, `canonicalize`, `resolve_crowding`.
- `roigbiv/pipeline/run.py` — the four canonicalize/resolve_crowding call sites, one per
  stage.
- `roigbiv/pipeline/types.py::PipelineConfig.roi_stamp_radius`, `ROI` docstring.
- `roigbiv/pipeline/optics.py::SCALE_DERIVED_FIELDS`, `derive_scale_params` — auto-scaling.
- `roigbiv/pipeline/gate1.py` — the merge-peak `accept → flag` convention this ADR's
  crowding guard reuses.
- `docs/design/OVERVIEW.md` §9 "Canonical ROI stamps" and §13 parameter reference.
- [ADR-0001](0001-non-destructive-candidate-union.md),
  [ADR-0002](0002-cascade-default-deprecation-criteria.md) — why `cascade_legacy` remains
  the correct target for this change today.
