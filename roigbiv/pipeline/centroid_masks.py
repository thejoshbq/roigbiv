"""Turn a FOV's ``centroids.json`` into the label image the registry consumes.

The cross-session registry reads a session as ``merged_masks.tif`` (a uint16
label image) plus ``summary/mean_M.tif``
(:func:`roigbiv.registry.roicat_adapter.load_session_input`). Centroid
discovery writes neither, so a centroids-only workspace is invisible to it.

This module bridges the two by stamping a fixed-radius disk at every centroid.
That is not a shortcut around real segmentation — it is what ADR-0003 already
requires of *every* ROI source: detector-native boundaries are replaced by
canonical disks before the registry ever sees them, specifically so that
session-to-session segmentation shape noise stops leaking into the ROICaT
embeddings that decide cross-session cell identity. Centroids simply arrive in
canonical form already; no boundary is invented here that a later stage would
have thrown away anyway.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.pipeline.roi_stamp import disk_mask

# Floor matches optics.py's derivation — below this a stamp is too small to
# carry any footprint overlap at all.
_MIN_STAMP_RADIUS = 4

# ROICaT crops every ROI to a fixed 36x36 window before embedding
# (``Data_roicat.transform_spatialFootprints_to_ROIImages``). That size is a
# constant — measured, it does *not* scale with ``um_per_pixel``, and neither
# does the footprint drawn inside it. So a stamp bigger than the window fills
# it edge to edge and every ROI image becomes the same solid square, leaving
# ROInet and SWT with nothing to tell ROIs apart.
#
# Measured on the reference prism FOV (44 ROIs over 3 sessions), disk fill of
# the crop vs. ROIs clustered:
#
#     radius 12   35% fill   cosine 0.971   10/44
#     radius 20   90% fill   cosine 0.992   30/44   <- cap
#     radius 22   96% fill   cosine 0.996   29/44
#     radius 24   99% fill   cosine 0.998    0/44   <- cliff
#     radius 30  100% fill   cosine 1.000    0/44
#
# The cliff is sharp, so the cap is set one step below the knee rather than at
# the last value that happens to work. End-to-end this is the difference
# between all three sessions auto-matching (cap 20) and the third landing in
# the review band (caps 18 and 22).
_ROICAT_ROI_IMAGE_PX = 36
_MAX_STAMP_RADIUS = 20


def calibrated_stamp_radius(output_dir: Path, cfg) -> int:
    """Radius implied by this FOV's measurements, before the matcher's cap.

    ``PipelineConfig.roi_stamp_radius`` defaults to 8 px and is only rescaled
    by ``derive_scale_params`` under ``--auto-scale``, so a FOV whose somata
    are much larger gets disks too small to overlap across sessions. Measured
    on the reference prism FOV, centroids move a median 18-53 px between
    sessions, so radius-8 disks never intersect and the spatial-footprint
    similarity is zero.

    Calibration wins when present because it is a *measurement* of this FOV;
    the config default is a guess that happens to suit a 512-px GRIN FOV.
    """
    from roigbiv.pipeline.calibration import load_calibration

    calibration = load_calibration(Path(output_dir))
    if calibration is not None and calibration.diameter_px > 0:
        return max(_MIN_STAMP_RADIUS, int(round(calibration.diameter_px / 2.0)))
    return int(getattr(cfg, "roi_stamp_radius", 8))


def resolve_stamp_radius(output_dir: Path, cfg) -> int:
    """Canonical disk radius to stamp, clamped to what the matcher can read.

    Anatomy sets the lower bound and ROICaT's fixed ROI crop sets the upper
    one. When a real soma is larger than the crop the stamp cannot represent
    it — that is a limitation of the matcher's input format, not of the FOV,
    and stamping the anatomical size anyway silently destroys the embeddings.
    """
    return min(calibrated_stamp_radius(output_dir, cfg), _MAX_STAMP_RADIUS)


@dataclass
class StampedMasks:
    """Outcome of stamping one FOV's centroids into a label image."""

    labels: np.ndarray          # uint16, 0 = background, 1..N = ROIs
    n_centroids: int
    n_overlapping_pairs: int    # centroid pairs closer than 2*radius
    radius_px: int
    # Radius anatomy asked for, when the matcher's ROI crop forced a smaller
    # one. None when nothing was clamped.
    radius_capped_from: Optional[int] = None

    @property
    def n_labels(self) -> int:
        """Labels actually present after overlap resolution.

        Lower than ``n_centroids`` when one stamp is completely buried by later
        ones — worth surfacing, since a swallowed label is an ROI the registry
        will never see.
        """
        return int(np.count_nonzero(np.unique(self.labels)))


def load_centroid_points(centroids_json: Path) -> list[tuple[float, float]]:
    """The ``(y, x)`` centroids from a ``centroids.json``, in file order."""
    payload = json.loads(Path(centroids_json).read_text())
    return [(float(c["y"]), float(c["x"])) for c in payload.get("centroids", [])]


def stamp_centroids(
    points: list[tuple[float, float]],
    shape: tuple[int, int],
    radius: int,
) -> StampedMasks:
    """Stamp a canonical disk of *radius* at each ``(y, x)`` in *points*.

    Labels are ``1..N`` in *points* order. Where disks overlap the later label
    wins — deterministic, and the same first-come-last-served convention a
    label image forces on any overlapping-footprint source.
    """
    height, width = int(shape[0]), int(shape[1])
    labels = np.zeros((height, width), dtype=np.uint16)
    for i, (cy, cx) in enumerate(points, start=1):
        labels[disk_mask(cy, cx, radius, height, width)] = i

    return StampedMasks(
        labels=labels,
        n_centroids=len(points),
        n_overlapping_pairs=_count_overlapping_pairs(points, radius),
        radius_px=int(radius),
    )


def _count_overlapping_pairs(points, radius: int) -> int:
    """Centroid pairs whose stamps intersect.

    Reported rather than resolved: ADR-0003's crowding guard demotes crowded
    ROIs at gate time, which centroid discovery has no equivalent of. Surfacing
    the count keeps stamp-radius mis-sizing visible instead of silent.
    """
    if len(points) < 2:
        return 0
    arr = np.asarray(points, dtype=np.float64)
    dy = arr[:, 0][:, None] - arr[:, 0][None, :]
    dx = arr[:, 1][:, None] - arr[:, 1][None, :]
    dist = np.hypot(dy, dx)
    return int(np.count_nonzero(np.triu(dist < 2 * radius, k=1)))


def write_merged_masks(
    output_dir: Path,
    cfg,
    *,
    shape: Optional[tuple[int, int]] = None,
) -> Optional[StampedMasks]:
    """Write ``{output_dir}/merged_masks.tif`` from this FOV's centroids.

    Returns ``None`` when the FOV has no ``centroids.json`` — the caller decides
    whether that is a skip or an error. *shape* defaults to that of
    ``summary/mean_M.tif``, which centroid discovery guarantees exists.

    A ``merged_masks.tif`` from a full cascade run is never overwritten: real
    per-stage detections outrank centroid stamps. ``pipeline_log.json`` is the
    marker for that — ``outputs.py`` writes it alongside the cascade's masks, so
    its absence means any masks here are ours to refresh.
    """
    import tifffile

    output_dir = Path(output_dir)
    centroids_json = output_dir / "centroids.json"
    if not centroids_json.exists():
        return None

    masks_path = output_dir / "merged_masks.tif"
    cascade_ran = (output_dir / "pipeline_log.json").exists()
    mean_path = output_dir / "summary" / "mean_M.tif"
    if shape is None:
        if not mean_path.exists():
            raise FileNotFoundError(
                f"no anatomical image at {mean_path} to size the label image — "
                "re-run centroid discovery to persist one")
        shape = tuple(np.asarray(tifffile.imread(mean_path)).shape[:2])

    points = load_centroid_points(centroids_json)
    wanted = calibrated_stamp_radius(output_dir, cfg)
    radius = min(wanted, _MAX_STAMP_RADIUS)
    stamped = stamp_centroids(points, shape, radius)
    if wanted > radius:
        stamped.radius_capped_from = wanted

    if not (cascade_ran and masks_path.exists()):
        tifffile.imwrite(str(masks_path), stamped.labels)
    return stamped
