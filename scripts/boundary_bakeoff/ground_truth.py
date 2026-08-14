"""Ground-truth *label images* from ImageJ RoiSets.

``scripts/centroid_bakeoff/ground_truth.py`` already parses these RoiSets and
rasterizes each polygon — it just reduces every ROI to a centroid afterwards,
because that is all a centroid bake-off needs. Boundary scoring needs the
rasterization it throws away, so this module reuses the same discovery and the
same ``skimage.draw.polygon`` call and keeps the labeled mask instead.

Same rasterization, deliberately: a GT boundary defined differently than the
centroids the pipeline was tuned against would make the two bake-offs
incomparable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import roifile
from skimage.draw import polygon as _draw_polygon

log = logging.getLogger("boundary_bakeoff.ground_truth")


def imagej_roiset_to_labels(
    roi_zip_path: Path, shape: tuple[int, int],
) -> tuple[np.ndarray, list[str]]:
    """Rasterize an ImageJ RoiSet into a ``(H, W)`` uint16 label image.

    Labels are ``1..N`` in RoiSet order. Where two hand-drawn ROIs overlap the
    later one wins — the same last-writer convention any label image forces.

    POINT-type ROIs are skipped rather than given a synthetic radius: a point
    carries no boundary, and inventing one would score this bake-off against a
    shape nobody drew. They are counted in the log so a POINT-only RoiSet reads
    as "no usable ground truth" instead of "zero cells".
    """
    rois = roifile.roiread(str(roi_zip_path))
    if not isinstance(rois, list):
        rois = [rois]

    H, W = int(shape[0]), int(shape[1])
    labels = np.zeros((H, W), dtype=np.uint16)
    names: list[str] = []
    n_points = 0
    n_degenerate = 0

    for idx, roi in enumerate(rois):
        if roi.roitype == roifile.ROI_TYPE.POINT:
            n_points += 1
            continue
        coords = roi.coordinates()
        if coords is None or len(coords) < 3:
            n_degenerate += 1
            continue
        rr, cc = _draw_polygon(coords[:, 1], coords[:, 0], shape=(H, W))
        if rr.size == 0:
            n_degenerate += 1
            continue
        names.append(roi.name or f"roi{idx}")
        labels[rr, cc] = len(names)

    if n_points:
        log.warning("%s: skipped %d POINT ROI(s) — no boundary to score",
                    roi_zip_path.name, n_points)
    if n_degenerate:
        log.warning("%s: skipped %d degenerate ROI(s)",
                    roi_zip_path.name, n_degenerate)
    return labels, names


def label_centroids(labels: np.ndarray) -> dict[int, tuple[float, float]]:
    """``{label: (y, x)}`` centre of mass, the convention ADR-0003 uses."""
    from scipy.ndimage import center_of_mass

    ids = [int(i) for i in np.unique(labels) if i != 0]
    if not ids:
        return {}
    coms = center_of_mass(labels > 0, labels, ids)
    return {i: (float(c[0]), float(c[1])) for i, c in zip(ids, coms)}


def discover_pairs(
    search_roots: list[Path], exclude_patterns: Optional[list[str]] = None,
):
    """``(mc_tif, roi_zip, stem)`` triples — delegates to the centroid bake-off."""
    from scripts.centroid_bakeoff.ground_truth import discover_real_pairs

    return discover_real_pairs(search_roots, exclude_patterns)
