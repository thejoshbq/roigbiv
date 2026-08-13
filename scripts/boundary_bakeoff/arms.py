"""The three boundary sources under comparison.

Each arm turns one FOV's anatomical image (plus, for the seeded arm, a set of
confirmed centroids) into a ``(H, W)`` uint16 label image. Nothing here
reimplements detection: every arm calls the same production code path the
pipeline uses, so a bake-off result is a statement about the pipeline rather
than about this script.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Arm:
    """One arm's output for one FOV."""

    name: str
    labels: np.ndarray
    n_disk_fallback: int = 0
    n_orphan_basin_px: int = 0
    # label -> ORIGIN_*, for the seeded arm only. Without it the arm's mean IoU
    # silently averages real boundaries together with the disks it fell back to,
    # and "did the seeded boundary help" becomes unanswerable from the report.
    origins: dict = field(default_factory=dict)


def _detect(morph: np.ndarray, cfg):
    """Cellpose inference on the anatomical image, with the flow field kept.

    Mirrors ``centroids.py``'s pinned substrate — single channel, undenoised,
    globally normalized. Each of those was measured as load-bearing on the
    reference prism FOV; scoring boundaries under Stage 1's dual-channel
    convention instead would measure a configuration nobody runs here.
    """
    from roigbiv.pipeline.stage1 import run_cellpose_flows

    return run_cellpose_flows(morph, np.zeros_like(morph), cfg)


def free_cellpose(flows) -> Arm:
    """Cellpose's own instance segmentation — today's Stage 1 detector."""
    return Arm(name="free_cellpose",
               labels=np.asarray(flows.label_image, dtype=np.uint16))


def disk_stamps(seeds: dict[int, tuple[float, float]], shape, radius: int) -> Arm:
    """Fixed-radius disks — what the tracking workflow writes today (ADR-0003)."""
    from roigbiv.pipeline.centroid_masks import stamp_labeled_centroids

    return Arm(name="disk_stamps",
               labels=stamp_labeled_centroids(seeds, shape, radius).labels)


def seeded(
    seeds: dict[int, tuple[float, float]], shape, flows, cfg, *,
    capture_px: float, fallback_radius: int,
) -> Arm:
    """Flow-field extent partitioned by confirmed centroids."""
    from roigbiv.pipeline.seeded_masks import converge_pixels, seeded_labels

    inds, converged = converge_pixels(
        flows.dP, flows.cellprob,
        cellprob_threshold=float(getattr(cfg, "cellprob_threshold", -2.0)),
        niter=flows.niter, dp_scale=flows.dp_scale,
    )
    result = seeded_labels(
        seeds, shape, inds=inds, converged=converged, cellprob=flows.cellprob,
        capture_px=capture_px, fallback_radius=fallback_radius,
        min_area=int(getattr(cfg, "boundary_min_area", 0) or 0),
        max_area=getattr(cfg, "boundary_max_area", None),
    )
    return Arm(name="seeded", labels=result.labels,
               n_disk_fallback=result.n_disk_fallback,
               n_orphan_basin_px=result.n_orphan_basin_px,
               origins=dict(result.origins))
