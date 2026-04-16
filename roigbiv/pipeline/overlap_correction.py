"""
ROI G. Biv pipeline — Overlapping Cell Trace Correction (spec §13.4, Blindspot 5).

Sequential subtraction can leave cross-contamination in the traces of
spatially overlapping ROIs. This module finds overlap groups via connected
components of the IoU graph and re-estimates their traces simultaneously
from the ORIGINAL registered movie using the shared lstsq solver from
subtraction.py.

Non-overlapping ROIs are left untouched; their neuropil-corrected traces
already reflect demixing via the neuropil subtraction step in traces.py.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from roigbiv.pipeline.types import FOVData, ROI, PipelineConfig
from roigbiv.pipeline.subtraction import (
    estimate_spatial_profiles,
    _build_union_design,
    solve_traces_from_chunks,
)


def _pair_overlaps(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, int, int]:
    """Return (iou, overlap_pixels, min_area)."""
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0:
        return 0.0, 0, min(int(mask_a.sum()), int(mask_b.sum()))
    union = int(np.logical_or(mask_a, mask_b).sum())
    iou = inter / union if union > 0 else 0.0
    min_area = min(int(mask_a.sum()), int(mask_b.sum()))
    return iou, inter, min_area


def find_overlap_groups(
    rois: list[ROI],
    iou_threshold: float = 0.1,
    overlap_fraction: float = 0.1,
) -> list[list[int]]:
    """Build connected components of the ROI overlap graph.

    Edge criteria (either triggers an edge):
      - IoU > iou_threshold
      - intersection_pixels > overlap_fraction × min(area_a, area_b)

    Returns
    -------
    List of groups, each a list of ROI indices (>= 2 members). Solo ROIs
    are omitted — they need no correction.
    """
    n = len(rois)
    if n < 2:
        return []

    # Bounding-box prefilter avoids n² mask intersections on large FOVs.
    bboxes = []
    for r in rois:
        ys, xs = np.where(r.mask)
        if ys.size == 0:
            bboxes.append((0, 0, 0, 0))
        else:
            bboxes.append((ys.min(), ys.max(), xs.min(), xs.max()))

    rows, cols = [], []
    for i in range(n):
        y0i, y1i, x0i, x1i = bboxes[i]
        for j in range(i + 1, n):
            y0j, y1j, x0j, x1j = bboxes[j]
            if y1i < y0j or y1j < y0i or x1i < x0j or x1j < x0i:
                continue  # bounding boxes disjoint → no pixel overlap
            iou, inter, min_area = _pair_overlaps(rois[i].mask, rois[j].mask)
            if inter == 0:
                continue
            if iou > iou_threshold or inter > overlap_fraction * max(min_area, 1):
                rows.append(i); cols.append(j)
                rows.append(j); cols.append(i)

    if not rows:
        return []

    data = np.ones(len(rows), dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)

    groups: list[list[int]] = []
    for c in range(n_comp):
        members = [int(i) for i in np.where(labels == c)[0]]
        if len(members) >= 2:
            groups.append(members)
    return groups


def correct_overlapping_traces(
    fov: FOVData,
    rois: list[ROI],
    groups: list[list[int]],
    F_corrected: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    """Re-estimate traces for every ROI in every overlap group.

    For each group, builds a design matrix W over the union of group members'
    pixels (weighted by spatial profiles derived from fov.std_S — same choice
    as subtraction.py for consistency with Blindspot 5's rationale), then
    solves W c(t) ≈ M(union_pixels, t) at every timepoint via the shared
    GPU-chunked normal-equations solver.

    F_corrected is updated IN PLACE for the members of each group. ROIs not
    in any group are unaffected.

    Parameters
    ----------
    fov         : FOVData (must have data_bin_path, shape, std_S populated)
    rois        : full sorted-by-label-id ROI list (matches F_corrected rows)
    groups      : output of find_overlap_groups
    F_corrected : (N, T) float32 — current neuropil-corrected traces (modified in place)
    cfg         : PipelineConfig

    Returns
    -------
    F_corrected (same array, modified in place) for convenience.
    """
    if not groups:
        return F_corrected

    T, H, W = fov.shape
    profile_source = fov.std_S  # same choice as subtraction.py

    for group in groups:
        group_rois = [rois[i] for i in group]
        profiles = estimate_spatial_profiles(profile_source, group_rois)
        design, union_flat_idx, _ = _build_union_design(profiles)
        P = design.shape[0]
        if P == 0 or design.shape[1] == 0:
            continue

        # Chunked iterator over data.bin (int16 → float32 per chunk).
        chunk = int(cfg.subtract_chunk_frames)
        mm = np.memmap(str(fov.data_bin_path), dtype=np.int16, mode="r",
                       shape=(T, H, W))

        def _iter(mm=mm, union_flat_idx=union_flat_idx, H=H, W=W, T=T, chunk=chunk):
            for t0 in range(0, T, chunk):
                t1 = min(t0 + chunk, T)
                cs = t1 - t0
                S_chunk = np.asarray(mm[t0:t1], dtype=np.float32).reshape(cs, H * W)[:, union_flat_idx]
                yield t0, t1, S_chunk

        traces_group = solve_traces_from_chunks(design, T, _iter(), cfg)
        del mm

        # Overwrite rows in F_corrected for members of this group
        for local_i, roi_idx in enumerate(group):
            F_corrected[roi_idx] = traces_group[local_i]
            rois[roi_idx].trace_corrected = traces_group[local_i].astype(np.float32, copy=True)

    return F_corrected
