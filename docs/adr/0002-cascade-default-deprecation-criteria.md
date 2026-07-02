# ADR-0002 — Criteria and migration path for retiring `cascade_legacy` as the default pipeline mode

- **Status:** Accepted
- **Date:** 2026-07-02
- **Deciders:** Josh Boquiren
- **Supersedes / relates to:** [ADR-0001](0001-non-destructive-candidate-union.md) §Decision 2
  (inter-stage subtraction deprecated as default control flow); operationalizes ADR-0001's
  Consequences prerequisite ("the pivot is only measurable with a benchmark harness... that
  harness (Milestone A) is a prerequisite, not optional").

## Context

ADR-0001 decided that ROI G. Biv is pivoting from the destructive sequential-subtractive
**cascade** toward a non-destructive **candidate union** architecture, but explicitly left the
timing of that pivot open, gated on a benchmark harness that did not yet exist. That gate is
now closer: `roigbiv/pipeline/types.py:25-26` defines

```python
PIPELINE_MODES = ("cascade_legacy", "candidate_union", "candidate_union_with_residual_refinement", "benchmark_only")
DEFAULT_PIPELINE_MODE = "cascade_legacy"
```

and `PipelineConfig.pipeline_mode` (`types.py:210-223`) plus the `--pipeline-mode` CLI flag
(`roigbiv/pipeline/run.py:1790-1799`) are wired end to end. But this is **inert plumbing only**
— no stage reads `cfg.pipeline_mode` yet, and `roigbiv/pipeline/resume.py:145-149` explicitly
excludes it from the resume fingerprint "until a stage reads it and its value can actually
change pipeline output." None of the candidate-union architecture itself (Milestones B–G) is
implemented; only the mode enum exists.

Roadmap item **H5** (this ADR) asks: decide *when* `candidate_union` becomes default. Its
companion, roadmap item **D10** (issue #64, "benchmark cascade vs candidate-union
architecture"), will *run* the actual comparison — cascade_legacy vs. candidate_union vs.
candidate_union_with_residual_refinement — and record results in
`docs/experiments/cascade_vs_candidate_union.md`. This ADR exists so #64 has a fixed,
unambiguous target to evaluate against, decided now rather than invented after the results are
already in hand.

## Decision

**This ADR does not change any runtime default.** It fixes a benchmark-checkable criteria set
and a migration sequence; the default flip itself requires a separate future PR per that
sequence (see Migration path below).

### Criteria

All criteria are relative to a `cascade_legacy` baseline run on the same benchmark manifest
(`roigbiv/benchmark/schema.py::BenchmarkManifest`) — none is a fabricated absolute threshold,
per the pattern already used in issue #92's own wording (F1 ≥, counts ≤, rate ≤):

| Criterion | Comparator | Metric source |
|---|---|---|
| Detection F1 | candidate ≥ cascade | `DetectionMetrics.f1` (`roigbiv/benchmark/metrics.py`) |
| Split count | candidate ≤ cascade | `TrackingMetrics.split_count` |
| Merge count | candidate ≤ cascade | `TrackingMetrics.merge_count` |
| False-transient rate | candidate ≤ cascade | **blocked** — no field exists yet; see "False-transient gap" below |
| Review burden | candidate ≤ cascade | `pipeline_log.json:review_queue_summary.total`, already written by every run today (`roigbiv/pipeline/outputs.py:139`) |
| Runtime | candidate ≤ cascade × 1.20 | `RuntimeMetrics.runtime_seconds`; a 20% regression budget, a stated policy default — adjustable, not derived from any existing measurement |

**Review burden** is anchored to `review_queue_summary.total` rather than
`HitlMetrics.total_corrections`: the queue-summary count is emitted automatically by every
pipeline run, while `total_corrections` only exists after a human actually performs HITL
review, and no HITL sessions against candidate-union output exist yet.

**Runtime** gets an explicit tolerance band rather than a strict `≤`, because a non-destructive
union that proposes more candidates than a subtractive cascade discards could legitimately cost
somewhat more compute even while winning every other criterion. 20% is a deliberately chosen
policy number for issue #64 to apply, not a value backed by any candidate-union runtime data —
none exists yet.

The criteria are captured below in a structured form so the future benchmark comparison report
(roadmap item A8, issue #32) can consume them mechanically rather than re-deriving them from
prose:

```yaml
# Go/no-go criteria for flipping DEFAULT_PIPELINE_MODE away from "cascade_legacy".
# All comparisons are candidate-union vs. a cascade_legacy baseline run on the
# same benchmark manifest — no criterion here is an absolute threshold.
criteria:
  - id: detection_f1
    metric_field: DetectionMetrics.f1
    comparator: ">="
    pass_condition: "candidate.f1 >= cascade.f1"
    status: ready   # blocked on runner (#28) + matcher (#30), not on this ADR

  - id: split_count
    metric_field: TrackingMetrics.split_count
    comparator: "<="
    pass_condition: "candidate.split_count <= cascade.split_count"
    status: ready

  - id: merge_count
    metric_field: TrackingMetrics.merge_count
    comparator: "<="
    pass_condition: "candidate.merge_count <= cascade.merge_count"
    status: ready

  - id: false_transient_rate
    metric_field: null   # does not exist yet — see "False-transient gap"
    comparator: "<="
    pass_condition: "candidate.false_transient_rate <= cascade.false_transient_rate"
    status: blocked
    blocking_note: >
      No field in roigbiv/benchmark/metrics.py computes this today. Must be added
      (event-level false-positive count against synthetic-injection / manual
      transient ground truth) before this criterion is automatically checkable.

  - id: review_burden
    metric_field: "pipeline_log.json:review_queue_summary.total"
    comparator: "<="
    pass_condition: "candidate.review_burden <= cascade.review_burden"
    status: ready

  - id: runtime
    metric_field: RuntimeMetrics.runtime_seconds
    comparator: "<= baseline * (1 + tolerance)"
    tolerance: 0.20   # policy default, adjustable — not derived from data
    pass_condition: "candidate.runtime_seconds <= cascade.runtime_seconds * 1.20"
    status: ready

overall_pass_condition: >
  All criteria with status: ready must pass. false_transient_rate is required by
  issue #92 but currently blocked; the go/no-go review (see Migration path) cannot
  record a full pass until its metric_field is implemented and evaluated.
```

### False-transient-rate gap

A **false transient** is a calcium-transient-shaped event that a detector's matched-filter /
waveform validation (Stage 3's FFT matched filter + Gate 3 R² check in `cascade_legacy`, or
whichever stage proposes transient-typed candidates under `candidate_union`) accepts or flags,
but which does not correspond to any ground-truth transient in the benchmark FOV's
synthetic-injection log or manual annotation. Rate is computed as false transient events
divided by total detected transient events, per FOV, then aggregated across the benchmark
manifest.

`roigbiv/benchmark/metrics.py` has no field for this today: `DetectionMetrics` covers
spatial precision/recall/F1/IoU/FP/FN; `TrackingMetrics` covers only `split_count` /
`merge_count`; no dataclass in that module has an event-level false-positive field. This is a
named prerequisite gap — closing it (a new field on `DetectionMetrics`, or a new dataclass)
is required before issue #64/#32 can satisfy issue #92's acceptance criterion "the benchmark
report can evaluate criteria automatically." It is intentionally left `status: blocked` above
rather than silently dropped from the criteria set.

## Migration path

None of the following happens in this ADR — it is the sequence future work must follow:

1. Milestones B–G land actual `candidate_union` support (candidate schema, joint
   deconfliction/validation, matcher, runner, report generator — issues #28, #30, #32,
   #42–#68, etc.).
2. Issue #64 runs `cascade_legacy` vs. `candidate_union` vs.
   `candidate_union_with_residual_refinement` on the benchmark manifest and writes
   `docs/experiments/cascade_vs_candidate_union.md`.
3. A go/no-go review evaluates #64's results against the criteria block above; the outcome is
   recorded either in that experiments doc or a short addendum to this ADR — not pre-decided
   here.
4. **If the review passes** — a separate, dedicated PR flips `DEFAULT_PIPELINE_MODE` in
   `roigbiv/pipeline/types.py:26`, adds a `docs/CHANGELOG.md` entry, and ships as a **minor**
   version bump (the package is pre-1.0, currently `0.1.10`; a minor bump is appropriate because
   a backward-compatible escape hatch — `--pipeline-mode cascade_legacy` — remains available, so
   no caller is forced to break).
5. **No forced removal timeline for `cascade_legacy`.** It remains selectable indefinitely via
   `--pipeline-mode cascade_legacy`: existing `inference/pipeline/{stem}/` outputs, and any
   HITL corrections tied to them, stay reproducible only if the mode that produced them remains
   runnable. Removing `cascade_legacy` entirely, if ever proposed, requires its own future ADR
   that supersedes this one.
6. **Runs already self-describe their mode.** `PipelineConfig.summary_for_log()`
   (`types.py:558-568`) serializes every config field via `self.__dict__`, so `pipeline_mode`
   is already captured in each run's `pipeline_log.json` under `"config"`
   (`roigbiv/pipeline/outputs.py:129`) — no action needed to make past or future runs
   self-describing.
7. **Resume-fingerprint follow-up (not a change this ADR makes).** `resume.py:145-149`
   excludes `pipeline_mode` from the resume fingerprint pending a stage actually reading it;
   whichever Milestone-C/D issue first wires that should revisit the exclusion.

## Consequences

**Benefits**

- A fixed, criteria-first target exists before the candidate-union comparison (#64) is ever
  run, preventing post-hoc rationalization of whichever result comes back.
- The criteria are expressed in a structured form a future report generator can consume
  mechanically, satisfying issue #92's acceptance criterion once the blocking metric gap
  (false-transient rate) is closed.
- The migration path makes explicit that `cascade_legacy` never disappears out from under
  existing runs — reproducibility of prior results is preserved by construction.

**Costs / obligations**

- The false-transient-rate metric must be implemented (new field/dataclass in
  `roigbiv/benchmark/metrics.py`) before this ADR's criteria set is fully checkable — tracked
  as a dependency of issues #64/#32, not of this ADR.
- The 20% runtime tolerance and the review-burden anchor are policy choices made without any
  candidate-union data to calibrate against; they may need revision once issue #64 actually
  runs and produces real numbers.

## References

- [ADR-0001: non-destructive candidate union](0001-non-destructive-candidate-union.md) — the
  decision this ADR operationalizes.
- Companion benchmark-run issue: GitHub issue #64 (roadmap D10), "benchmark cascade vs
  candidate-union architecture" — states the same five criteria from the benchmark-execution
  side and the "Go/No-Go Gate 2: candidate union becomes default only if it beats the cascade
  on a criterion" framing.
- `roigbiv/pipeline/types.py:25-26,210-223` — `PIPELINE_MODES` / `DEFAULT_PIPELINE_MODE` /
  `pipeline_mode` field.
- `roigbiv/pipeline/resume.py:145-149` — resume-fingerprint exclusion for `pipeline_mode`.
- `roigbiv/benchmark/metrics.py` — `DetectionMetrics`, `TrackingMetrics`, `RuntimeMetrics`,
  `HitlMetrics` data models; source of the false-transient-rate gap.
- `roigbiv/pipeline/outputs.py:129,139` — `pipeline_log.json`'s `config` and
  `review_queue_summary` fields.
- `docs/design/OVERVIEW.md` §7 (Stage 3 template sweep, including the Gate 3 waveform
  validation subsection) — the transient-detection concepts the false-transient-rate
  definition is tied to.
