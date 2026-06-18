"""Phase 5b — tonic accept tier (post-classification, OFF by default).

A narrow, auditable promotion step that runs AFTER `classify_all_rois`. It lets
*anatomically-detected* tonic somata skip human review when their baseline sits
convincingly above the surrounding neuropil:

    promote to gate_outcome="accept" iff
        cfg.tonic_accept_tier is True
        AND activity_type == "tonic"
        AND source_stage in {1, 2}                  # anatomical detectors only
        AND gate_outcome != "reject"                # already in merged_masks
        AND neuropil_baseline_elevation >= cfg.tonic_accept_min_elevation

Design constraints (engagement invariants):
  * Stage-4 tonics (source_stage == 4) are NEVER touched — Gate 4's
    requires_review contract is load-bearing and stays exactly as-is.
  * This changes NO mask in merged_masks.tif (those carry every non-rejected
    ROI regardless of gate_outcome). The only effect is review-queue
    membership: a promoted ROI no longer consumes human review.
  * Strictly additive provenance: the original outcome/confidence is recorded
    in gate_reasons so the promotion is reversible/auditable.
  * OFF by default (no_default_flip). Flipping the flag needs the gate-aware
    A/B + explicit approval.
"""
from __future__ import annotations

from roigbiv.pipeline.types import ROI, PipelineConfig

# Outcomes/confidence that mean "this ROI would otherwise consume review".
_REVIEW_OUTCOMES = frozenset({"flag"})
_REVIEW_CONFIDENCE = frozenset({"requires_review", "moderate", "low"})


def _is_promotion_candidate(roi: ROI, cfg: PipelineConfig) -> bool:
    if roi.activity_type != "tonic":
        return False
    if int(roi.source_stage) not in (1, 2):
        return False
    if roi.gate_outcome == "reject":
        return False
    elev = float(roi.features.get("neuropil_baseline_elevation", 0.0))
    return elev >= float(cfg.tonic_accept_min_elevation)


def apply_tonic_accept_tier(rois: list[ROI], cfg: PipelineConfig) -> int:
    """Promote qualifying anatomical tonic ROIs to ``accept`` in place.

    Returns the number of ROIs whose review routing actually changed (i.e. they
    were headed to review and are now auto-accepted). No-op (returns 0) when
    ``cfg.tonic_accept_tier`` is False.
    """
    if not getattr(cfg, "tonic_accept_tier", False):
        return 0

    promoted = 0
    for roi in rois:
        if not _is_promotion_candidate(roi, cfg):
            continue
        was_in_review = (
            roi.gate_outcome in _REVIEW_OUTCOMES
            or roi.confidence in _REVIEW_CONFIDENCE
        )
        if not was_in_review:
            continue  # already auto-accepted with high confidence; leave it
        elev = float(roi.features.get("neuropil_baseline_elevation", 0.0))
        roi.gate_reasons.append(
            f"tonic_accept_tier(elev={elev:.3f},was={roi.gate_outcome}/"
            f"{roi.confidence},thr={float(cfg.tonic_accept_min_elevation):.3f})"
        )
        roi.gate_outcome = "accept"
        roi.confidence = "high"
        promoted += 1

    if promoted:
        print(f"  Tonic accept tier: promoted {promoted} anatomical tonic ROI(s) "
              f"to accept (elev ≥ {float(cfg.tonic_accept_min_elevation):.3f})",
              flush=True)
    return promoted
