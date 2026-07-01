# ADR-0001 — Non-destructive candidate union replaces the destructive subtractive cascade

- **Status:** Accepted
- **Date:** 2026-07-01
- **Deciders:** Josh Boquiren
- **Supersedes / relates to:** the sequential-subtractive architecture documented in
  [`docs/design/OVERVIEW.md`](../design/OVERVIEW.md) §2

> This is the first Architecture Decision Record in the repo and establishes the
> `docs/adr/NNNN-slug.md` convention: one decision per file, zero-padded sequential
> numbering, `Status` ∈ {Proposed, Accepted, Superseded, Deprecated}. Later ADRs that
> reverse a decision should link back rather than edit the original.

## Context

ROIGBIV currently runs a **sequential subtractive cascade**. A shared Foundation prepares
the movie (motion correction → truncated-SVD low-rank/sparse background split → summary
images), then four detection stages run in order, each operating on the **residual** left
after the previous stages subtract the sources they found:

```
Stage 1 Cellpose → Gate 1 → subtract
      → Stage 2 Suite2p → Gate 2 → subtract
      → Stage 3 template sweep → Gate 3 → subtract
      → Stage 4 tonic-neuron search → Gate 4
```

The subtraction is already *virtual* on disk — the residual is reconstructed on demand by
`ResidualView` (`roigbiv/pipeline/residual.py`) rather than materialized, and the codebase
describes that chain as "virtual and non-destructive" (`roigbiv/pipeline/resume.py:22`). But
it remains **destructive in the detection-ordering sense**: each stage sees a movie from
which earlier candidates have already been removed, so an upstream error changes what every
downstream stage can find.

The pipeline's own design docs flagged this risk. The (now-deleted, retained in git history
at `c5b8664^`) `docs/roi-pipeline-specification.md` §1.3 stated:

> "Source subtraction is imperfect. Between every detection stage, a validation gate checks
> new candidates for artifacts, redundancy, and biological plausibility before allowing
> subtraction to proceed. The gates prevent error propagation through the pipeline."

Inter-stage gates mitigate but do not eliminate the failure mode: a bad Stage-1 mask can
subtract real signal a later stage needed, and a missed Stage-1 cell alters the residual
every later stage sees. Making the cascade itself the core scientific claim is therefore
architecturally fragile — a single early decision can silently corrupt a whole run.

A strategic assessment of the project (2026) recommended shifting ROIGBIV's center of gravity
**from "sequential subtractive detector" to "auditable high-recall ROI discovery + calibrated
longitudinal identity."** The proposed structure:

> candidate generation → non-destructive candidate union → joint deconfliction / validation
> → optional (reversible) subtraction for residual discovery → HITL + calibrated registry

This preserves the clever pieces (the fine-tuned Cellpose stage, Suite2p reuse, matched-filter
transient recovery, the slow/tonic candidate search, the ROICaT registry) while removing the
single largest architectural risk: irreversible upstream-to-downstream error propagation.

## Decision

1. **Stages 1–4 become proposal (candidate) generators**, not links in a destructive chain.
   Each stage proposes candidate ROIs with full provenance (source method, mask geometry,
   temporal support, stage-specific scores, uncertainty); no stage's output silently removes
   evidence from another stage's input.

2. **Inter-stage subtraction is deprecated as the default control flow.** Candidates flow into
   a **non-destructive candidate union** and are reconciled by a **joint deconfliction /
   validation** step (merge near-duplicates, split merged blobs, reject unsupported candidates,
   route ambiguous ones to HITL) instead of each stage subtracting before the next runs.

3. **Reversible residual discovery may remain as a bounded, optional refinement.** The already
   virtual, non-destructive `ResidualView` makes residual-based discovery cheap to offer as an
   *opt-in* pass for hard cases — provided every subtraction layer records which candidates
   caused it and can be replayed with or without that layer. It is a refinement, not the
   primary architecture.

4. **L+S+T (low-rank + sparse + tonic) decomposition is out of scope** unless future validation
   reverses this decision. A constant-bright soma is both spatially localized and temporally
   low-rank, so without strong spatial/morphological priors it is not separably identifiable
   from background; the current decomposition stays **L+S**. If tonic loss proves real after
   denoising + anatomical detection, the publishable path is a ground-truth benchmark
   demonstrating the loss — not a new decomposition first.

## Consequences

**Benefits**

- No irreversible upstream-to-downstream error propagation; a wrong early mask no longer
  removes evidence a later detector needed.
- Auditable provenance: every candidate is traceable to its generator(s) and its subtraction
  state, so runs can be replayed and compared.
- Honest recall: the union is a high-recall proposal engine; correctness is decided once, in a
  joint validation step, rather than implicitly by detection order.

**Costs / obligations**

- Correctness now lives in the **deconfliction + joint-validation** step, which must be built
  and calibrated; this is new surface area.
- The pivot is only measurable with a **benchmark harness** (frozen representative FOV set +
  detection / trace / HITL / tracking metrics). That harness (Milestone A) is a prerequisite,
  not optional.
- Provenance/candidate-table schema and the optional reversible-subtraction bookkeeping are
  additional state to design and maintain.

## Do not build yet (research-stage only)

The following are explicitly **not** approved for implementation under this decision. They may
be prototyped as research spikes behind the benchmark harness, but none ships as a default path
until evidence justifies it:

- **L+S+T tonic decomposition** — not separably identifiable without strong priors (see
  Decision 4); revisit only if a ground-truth benchmark proves systematic tonic-cell loss.
- **DNANet** (infrared small-target detection net) — cross-domain candidate generator; unproven
  on low-SNR two-photon summaries and adds a heavy dependency.
- **SAM2 video tracking** — promising for temporal object tracking but unvalidated for
  cross-session cell identity here; the sanctioned tracking path is the ROICaT registry.
- **Tensor decomposition** — same identifiability concern as L+S+T; SVD / low-rank-sparse / RPCA
  remain the sanctioned decompositions.

## References

- Current-architecture authority: [`docs/design/OVERVIEW.md`](../design/OVERVIEW.md) (§2
  virtual `ResidualView`; §8 Stage-4 tonic-neuron search; §11 optional/aspirational subsystems).
- Non-destructive precedent in code: `roigbiv/pipeline/residual.py`, `roigbiv/pipeline/resume.py:22`.
- Retired design docs (git history at `c5b8664^`, deleted by `c5b8664 chore(cleaning up docs)`):
  `docs/roi-pipeline-specification.md` §1.3 (subtraction-imperfection admission),
  `docs/publication/algorithms_v2.md` (as-built stage/algorithm reference).
- Version context: released `0.1.0` (2026-03-25); this ADR lands under `Unreleased`.
