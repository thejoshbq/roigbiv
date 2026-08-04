"""
ROI G. Biv pipeline — canonical ROI stamps.

Post-gate step: replace each accepted/flagged ROI's detector-native boundary
with a fixed-radius disk centered on its own centroid. Gates (gate1/2/4) still
evaluate real detector geometry — regionprops on the raw candidate mask —
before this runs; canonicalize() only changes what subtraction, trace
extraction, HITL, the viewer, and the registry consume afterward, not what the
accept/flag/reject decision was based on. See docs/adr/0003-centroid-canonical-
roi-stamps.md for the rationale (cross-session tracking stability).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass

from roigbiv.pipeline.types import ROI

# Confidence rank, lower = weaker. Used only to break resolve_crowding ties —
# never to change what gate1-4 already decided about accept/flag/reject.
_CONFIDENCE_RANK = {"high": 2, "moderate": 1, "requires_review": 0}


def disk_mask(cy: float, cx: float, radius: int, H: int, W: int) -> np.ndarray:
    """Filled circular disk, clipped to image bounds."""
    ys, xs = np.ogrid[:H, :W]
    return ((ys - cy) ** 2 + (xs - cx) ** 2) <= radius ** 2


def canonicalize(roi: ROI, radius: int, shape: tuple[int, int]) -> None:
    """Replace ``roi.mask`` in place with a fixed-radius disk at its own centroid.

    ``area`` / ``solidity`` / ``eccentricity`` are left untouched — they remain
    the gate-time record of the real detected geometry, not the persisted
    stamp (see the ``ROI`` dataclass docstring in types.py). No-op on an
    already-empty mask.
    """
    if not roi.mask.any():
        return
    cy, cx = center_of_mass(roi.mask)
    roi.mask = disk_mask(cy, cx, radius, *shape)


def resolve_crowding(rois: list[ROI], radius: int) -> None:
    """Demote the weaker of any two heavily-overlapping canonical stamps to "flag".

    Two centroids closer than ``radius`` apart imply >50% disk overlap — a
    crowding condition real (non-uniform, non-overlapping-by-construction)
    segmentation wouldn't produce. Mirrors gate1's own merge-peak convention
    (accept -> flag, never silently reject or drop): the ROI still enters
    subtraction — the existing ridge regularization there guards the solve
    numerically — flag only surfaces the pair for HITL review.

    Operates on the full accepted/flagged ``rois`` accumulated so far (i.e.
    call after each stage with the cumulative ``fov.rois``, not just that
    stage's own new candidates), since a later stage's stamp can crowd an
    earlier stage's already-persisted one.
    """
    active = [r for r in rois if r.gate_outcome in ("accept", "flag") and r.mask.any()]
    if len(active) < 2:
        return
    centroids = [center_of_mass(r.mask) for r in active]

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            cyi, cxi = centroids[i]
            cyj, cxj = centroids[j]
            dist = float(np.hypot(cyi - cyj, cxi - cxj))
            if dist >= radius:
                continue

            a, b = active[i], active[j]
            rank_a = _CONFIDENCE_RANK.get(a.confidence, 0)
            rank_b = _CONFIDENCE_RANK.get(b.confidence, 0)
            if rank_a == rank_b:
                # Deterministic tiebreak: earlier stage, then lower label_id, wins
                # (earlier-stage ROIs are already subtracted from the residual
                # that later stages detect on, so they're the more established call).
                weaker = a if (a.source_stage, a.label_id) > (b.source_stage, b.label_id) else b
            else:
                weaker = a if rank_a < rank_b else b

            if weaker.gate_outcome == "accept":
                weaker.gate_outcome = "flag"
                weaker.confidence = "moderate"
                weaker.gate_reasons.append(
                    f"crowded_neighbor:dist={dist:.1f}px<radius={radius}"
                )
