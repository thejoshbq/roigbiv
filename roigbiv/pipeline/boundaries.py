"""Write a FOV's seeded cell boundaries from its confirmed centroids.

The counterpart to :mod:`roigbiv.pipeline.centroid_masks`, and deliberately not
a replacement for it. The two write different geometry for different consumers:

``merged_masks.tif``  fixed-radius disks, for the cross-session registry.
                      ADR-0003: real segmentation boundaries are not stable for
                      the same cell across sessions, and ROICaT crops every ROI
                      to a fixed 36x36 window, so a disk is both the more stable
                      and the only representable input there.
``boundaries.tif``    real seeded boundaries, for humans and for anything that
                      needs a cell's actual extent. Never reaches ROICaT.

Both are stamped from the *same* effective centroids — ``centroids.json``
replayed through ``corrections/centroids.jsonl`` — and carry the *same* label
ids, so ``CellObservation.local_label_id`` addresses a cell in either image and
the registry needs no knowledge that this file exists.

``compute_boundaries`` layers one more thing on top of ``seeded_labels``'s
output: any hand-drawn overrides from ``corrections/boundaries.jsonl`` (see
``roigbiv.pipeline.boundary_edits``). A label with an active manual override
keeps its hand-drawn shape regardless of what this module's own computation
says for it; every other label reflects the computation as usual.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.pipeline.seeded_masks import SeededMasks, seeded_labels

# Bumped when boundaries.json changes shape.
_SCHEMA = 1


def _saved_settings(output_dir: Path) -> dict:
    """Boundary settings a human explicitly pinned for this FOV, if any.

    ``boundaries.json`` doubles as the per-FOV settings store. Keeping them here
    rather than in ``calibration.json`` means the boundary page never has to
    invent a ``diameter_px`` for a FOV nobody calibrated — which the centroids
    page would then read back and report as a measurement.

    Only the ``settings`` block counts, and it is written only when a caller
    passed an explicit override. The top-level ``capture_px`` records what the
    image on disk was drawn with, which is not the same claim: reading *that*
    back would let the first automatic redraw pin whatever the calibration
    happened to be, and a later re-calibration would quietly stop reaching the
    boundaries.
    """
    path = Path(output_dir) / "boundaries.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    settings = data.get("settings") if isinstance(data, dict) else None
    return settings if isinstance(settings, dict) else {}


def resolve_capture_px(output_dir: Path, cfg) -> float:
    """How far a flow trajectory may land from a seed and still be that cell.

    ``cfg`` (a CLI flag or config file) beats the value this FOV was last drawn
    with, which beats the calibrated soma radius — a measurement, not a guess.
    Deliberately *uncapped*, unlike
    :func:`~roigbiv.pipeline.centroid_masks.resolve_stamp_radius`: that cap
    exists because ROICaT's 36x36 ROI crop cannot represent a bigger disk, and
    boundaries never go to ROICaT.
    """
    from roigbiv.pipeline.centroid_masks import calibrated_stamp_radius

    explicit = getattr(cfg, "boundary_capture_px", None)
    if explicit is not None:
        return float(explicit)
    saved = _saved_settings(Path(output_dir)).get("capture_px")
    if saved is not None:
        try:
            return float(saved)
        except (TypeError, ValueError):
            pass
    return float(calibrated_stamp_radius(Path(output_dir), cfg))


def resolve_min_area(output_dir: Path, cfg) -> int:
    """Smallest boundary this FOV keeps, same precedence as ``capture_px``."""
    explicit = getattr(cfg, "boundary_min_area", None)
    if explicit:
        return int(explicit)
    saved = _saved_settings(Path(output_dir)).get("min_area")
    try:
        return int(saved) if saved is not None else 0
    except (TypeError, ValueError):
        return 0


def compute_boundaries(
    output_dir: Path,
    cfg,
    *,
    shape: Optional[tuple[int, int]] = None,
    capture_px: Optional[float] = None,
    min_area: Optional[int] = None,
) -> Optional[SeededMasks]:
    """Seeded boundaries for this FOV, or ``None`` when they can't be drawn.

    ``None`` means a missing ``centroids.json`` or a missing/stale flow cache —
    both normal states (centroid discovery never ran, ran with
    ``centroid_persist_flows=False``, or ran under the cpsam backend, which
    returns no flow field). The caller keeps the disk stamps in that case.

    ``capture_px`` / ``min_area`` override the resolved values for this call
    only — the boundary page passes its live slider positions here, which is
    also what makes a Save write the settings the user was actually looking at.
    """
    from roigbiv.pipeline.centroid_masks import (
        load_effective_centroids,
        resolve_stamp_radius,
    )
    from roigbiv.pipeline.centroids import load_flow_cache
    from roigbiv.pipeline.seeded_masks import converge_pixels

    output_dir = Path(output_dir)
    centroids_json = output_dir / "centroids.json"
    if not centroids_json.exists():
        return None

    try:
        params = json.loads(centroids_json.read_text()).get("params")
    except json.JSONDecodeError:
        return None

    cached = load_flow_cache(output_dir, params)
    if cached is None:
        return None

    cellprob = np.asarray(cached["cellprob"], dtype=np.float32)
    if shape is None:
        shape = tuple(int(v) for v in cellprob.shape[:2])

    labeled, warnings = load_effective_centroids(output_dir)

    # The threshold that selected the dynamics pixels at inference time. Reading
    # it back off the recorded params rather than cfg keeps the field and the
    # threshold applied to it from drifting apart across a config change.
    cellprob_threshold = float(
        (params or {}).get("cellprob_threshold",
                           getattr(cfg, "cellprob_threshold", -2.0)))

    inds, converged = converge_pixels(
        np.asarray(cached["dP"], dtype=np.float32), cellprob,
        cellprob_threshold=cellprob_threshold,
        niter=cached["niter"], dp_scale=cached["dp_scale"],
    )

    result = seeded_labels(
        labeled, shape,
        inds=inds, converged=converged, cellprob=cellprob,
        capture_px=(float(capture_px) if capture_px is not None
                    else resolve_capture_px(output_dir, cfg)),
        fallback_radius=resolve_stamp_radius(output_dir, cfg),
        min_area=(int(min_area) if min_area is not None
                  else resolve_min_area(output_dir, cfg)),
        max_area=getattr(cfg, "boundary_max_area", None),
    )
    result.warnings = list(warnings) + result.warnings

    from roigbiv.pipeline.boundary_edits import layer_boundary_ops

    return layer_boundary_ops(result, output_dir)


def write_boundaries(
    output_dir: Path,
    cfg,
    *,
    shape: Optional[tuple[int, int]] = None,
    capture_px: Optional[float] = None,
    min_area: Optional[int] = None,
) -> Optional[SeededMasks]:
    """Write ``{output_dir}/boundaries.tif`` and ``boundaries.json``.

    Returns ``None`` when :func:`compute_boundaries` cannot draw them.

    An explicit ``capture_px`` / ``min_area`` is pinned into the ``settings``
    block, which :func:`resolve_capture_px` reads back — so a redraw triggered
    by a centroid edit or a later ``run_tracking`` reuses what the user tuned
    rather than reverting to the calibrated radius. Passing neither carries any
    existing pin forward untouched and pins nothing new.

    A ``boundaries.tif`` is never written over a full-cascade run: the four
    stages produce their own per-ROI geometry, and centroid-seeded boundaries
    would be describing a different set of ROIs entirely. ``pipeline_log.json``
    is the marker for that, matching ``centroid_masks.write_merged_masks``. In
    that case the returned ``SeededMasks.written`` is ``False`` — the caller can
    still inspect what would have been drawn.
    """
    import tifffile

    output_dir = Path(output_dir)
    result = compute_boundaries(output_dir, cfg, shape=shape,
                                capture_px=capture_px, min_area=min_area)
    if result is None:
        return None

    masks_path = output_dir / "boundaries.tif"
    cascade_ran = (output_dir / "pipeline_log.json").exists()
    result.written = not (cascade_ran and masks_path.exists())
    if not result.written:
        return result

    settings = _saved_settings(output_dir)
    if capture_px is not None:
        settings["capture_px"] = float(capture_px)
    if min_area is not None:
        settings["min_area"] = int(min_area)

    tifffile.imwrite(str(masks_path), result.labels)
    (output_dir / "boundaries.json").write_text(json.dumps({
        "schema": _SCHEMA,
        "settings": settings,
        "n_seeds": result.n_seeds,
        "n_disk_fallback": result.n_disk_fallback,
        "n_orphan_basin_px": result.n_orphan_basin_px,
        "capture_px": result.capture_px,
        "min_area": result.min_area,
        "fallback_radius": result.fallback_radius,
        "warnings": result.warnings,
        "labels": [
            {
                "label": int(label),
                "area": int(result.areas.get(label, 0)),
                "equiv_diameter_px": round(
                    float(2 * np.sqrt(max(result.areas.get(label, 0), 0) / np.pi)), 2),
                "origin": result.origins.get(label),
            }
            for label in sorted(result.origins)
        ],
    }, indent=2))
    return result


def load_boundary_labels(output_dir: Path) -> Optional[np.ndarray]:
    """This FOV's ``boundaries.tif`` as a label image, or ``None`` if absent."""
    import tifffile

    path = Path(output_dir) / "boundaries.tif"
    if not path.exists():
        return None
    try:
        return np.asarray(tifffile.imread(str(path)))
    except (OSError, ValueError):
        return None
