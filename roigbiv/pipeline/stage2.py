"""
ROI G. Biv pipeline — Stage 2: Temporal Detection via Suite2p (spec §7).

Stage 2 catches neurons that Cellpose missed because they lack distinctive
spatial morphology in the mean image, but whose activity profile drives
Suite2p's SVD-based detector. Canonical cases: burst-firers, task-locked
neurons, cells obscured by a brighter neighbor in the projection.

Design (spec §7, plan D3-D5):
  1. Reuse the stat.npy / iscell.npy that Foundation already produced
     (Suite2p runs its full pipeline during motion correction). We do NOT
     re-run Suite2p here — that would double the Foundation cost.
  2. Filter by iscell probability (cfg.iscell_threshold).
  3. Convert Suite2p's sparse stat entries → dense binary masks via
     roigbiv.merge.stat_to_mask (existing utility).
  4. IoU filter against Stage 1 accept|flag ROIs: retain only masks with
     max-IoU ≤ cfg.gate2_iou_threshold (default 0.3) — these are NEW
     detections, not rediscoveries.
  5. Extract candidate traces from S₁ (NOT from the raw movie) for use by
     Gate 2's anti-correlation check and the subsequent subtraction step.

Returns a list of ROI objects with source_stage=2 and iscell_prob populated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.pipeline.types import FOVData, PipelineConfig, ROI


# ─────────────────────────────────────────────────────────────────────────
# Trace extraction from a residual memmap (shared with Stage 3)
# ─────────────────────────────────────────────────────────────────────────

def extract_traces_from_residual(
    residual_path: Path,
    shape: tuple,
    masks: list[np.ndarray],
    chunk: int = 500,
) -> np.ndarray:
    """For each mask, return the mean trace across mask pixels over time.

    Streams the (T, H, W) memmap in temporal chunks. Per chunk, reads all
    pixels into RAM (~500 MB for a 500-frame slice of 512×512 float32) and
    computes `traces[:, t0:t1] = (masks @ chunk.reshape(cs, H*W).T) / mask_sizes`.

    Parameters
    ----------
    residual_path : Path to the (T, H, W) float32 memmap
    shape         : (T, H, W)
    masks         : list of N (H, W) bool arrays
    chunk         : frames per temporal chunk

    Returns
    -------
    traces : (N, T) float32 — mean fluorescence per mask per frame
    """
    T, H, W = shape
    N = len(masks)
    if N == 0:
        return np.zeros((0, T), dtype=np.float32)

    # Pre-flatten masks into a (N, H*W) dense matrix. For typical N ~ 30-200
    # and H*W = 262144, this is 60-400 MB — fits. Sparse would help but adds
    # complexity for little RAM savings at these sizes.
    M = np.zeros((N, H * W), dtype=np.float32)
    mask_sizes = np.zeros(N, dtype=np.float32)
    for i, m in enumerate(masks):
        flat = m.ravel().astype(np.float32)
        M[i] = flat
        mask_sizes[i] = flat.sum()
    mask_sizes[mask_sizes == 0] = 1.0  # guard division

    traces = np.empty((N, T), dtype=np.float32)
    mm = np.memmap(str(residual_path), dtype=np.float32, mode="r", shape=(T, H, W))
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        cs = t1 - t0
        flat_chunk = np.asarray(mm[t0:t1], dtype=np.float32).reshape(cs, H * W)
        # (N, H*W) @ (H*W, cs) → (N, cs), then divide by per-mask pixel counts
        traces[:, t0:t1] = (M @ flat_chunk.T) / mask_sizes[:, None]
    del mm
    return traces


# ─────────────────────────────────────────────────────────────────────────
# Suite2p output loading
# ─────────────────────────────────────────────────────────────────────────

def _load_suite2p_outputs(
    data_bin_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load stat.npy and iscell.npy from the Suite2p plane directory.

    data_bin_path is {output_dir}/{stem}/suite2p/plane0/data.bin — sibling
    files stat.npy and iscell.npy are produced by Suite2p's detection pass.
    """
    plane_dir = data_bin_path.parent
    stat_path = plane_dir / "stat.npy"
    iscell_path = plane_dir / "iscell.npy"
    if not stat_path.exists():
        raise RuntimeError(
            f"Stage 2 requires Suite2p stat.npy at {stat_path}. "
            f"Foundation should have produced it (run_suite2p_fov runs full detection)."
        )
    if not iscell_path.exists():
        raise RuntimeError(
            f"Stage 2 requires Suite2p iscell.npy at {iscell_path}. "
            f"Foundation should have produced it."
        )
    stat = np.load(str(stat_path), allow_pickle=True)
    iscell = np.load(str(iscell_path))
    return stat, iscell


def _stat_entries_to_masks(
    stat: np.ndarray,
    iscell: np.ndarray,
    Ly: int,
    Lx: int,
    iscell_threshold: float,
) -> list[tuple[int, np.ndarray, float]]:
    """Convert Suite2p stat entries to binary (Ly, Lx) masks.

    Applies iscell_threshold on iscell[:, 1] (classifier probability).
    Returns list of (stat_index, mask, iscell_prob) tuples — stat_index
    preserves the mapping back to the original Suite2p ordering.
    """
    out = []
    for i, s in enumerate(stat):
        prob = float(iscell[i, 1]) if i < len(iscell) else 0.0
        if prob < iscell_threshold:
            continue
        ypix = np.asarray(s["ypix"], dtype=np.int64)
        xpix = np.asarray(s["xpix"], dtype=np.int64)
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        if not valid.any():
            continue
        mask = np.zeros((Ly, Lx), dtype=bool)
        mask[ypix[valid], xpix[valid]] = True
        out.append((i, mask, prob))
    return out


# ─────────────────────────────────────────────────────────────────────────
# IoU filter against Stage 1
# ─────────────────────────────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two dense bool masks."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _filter_against_stage1(
    candidates: list[tuple[int, np.ndarray, float]],
    stage1_masks: list[np.ndarray],
    iou_threshold: float,
) -> list[tuple[int, np.ndarray, float]]:
    """Retain candidates whose max IoU against any Stage 1 mask ≤ threshold."""
    kept = []
    for stat_idx, mask, prob in candidates:
        # Short-circuit: if any Stage 1 mask overlaps at all, compute IoU.
        # Otherwise IoU = 0 and we keep the candidate.
        max_iou = 0.0
        for s1 in stage1_masks:
            # Cheap bbox overlap check before computing full IoU
            if not (mask & s1).any():
                continue
            iou_val = _iou(mask, s1)
            if iou_val > max_iou:
                max_iou = iou_val
            if max_iou > iou_threshold:
                break
        if max_iou <= iou_threshold:
            kept.append((stat_idx, mask, prob))
    return kept


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────

def run_stage2(
    fov: FOVData,
    cfg: PipelineConfig,
    starting_label_id: int = 1,
) -> list[ROI]:
    """Full Stage 2: load Suite2p outputs → filter → extract traces → package ROIs.

    Parameters
    ----------
    fov               : FOVData with data_bin_path, residual_S1_path, shape, rois
                         (Stage 1 ROIs expected to be populated on fov.rois)
    cfg               : PipelineConfig
    starting_label_id : int — first label to assign to Stage 2 ROIs

    Returns
    -------
    list of ROI objects with source_stage=2, .trace populated, .iscell_prob set.
    Gate 2 is applied by the caller afterward.
    """
    import time

    T, Ly, Lx = fov.shape

    # 1. Load Suite2p outputs (already produced by Foundation's run_suite2p_fov)
    t0 = time.time()
    stat, iscell = _load_suite2p_outputs(fov.data_bin_path)
    print(f"  loaded stat.npy ({len(stat)} ROIs) + iscell.npy in {time.time()-t0:.2f}s",
          flush=True)

    # 2. iscell threshold + convert to masks
    candidates_raw = _stat_entries_to_masks(stat, iscell, Ly, Lx, cfg.iscell_threshold)
    print(f"  {len(candidates_raw)} candidates after iscell_threshold={cfg.iscell_threshold}",
          flush=True)

    # 3. IoU filter against Stage 1 accept|flag
    stage1_masks = [r.mask for r in fov.rois
                    if r.source_stage == 1 and r.gate_outcome in ("accept", "flag")]
    t0 = time.time()
    kept = _filter_against_stage1(candidates_raw, stage1_masks, cfg.gate2_iou_threshold)
    print(f"  {len(kept)} novel detections after IoU filter "
          f"(threshold={cfg.gate2_iou_threshold}) in {time.time()-t0:.2f}s",
          flush=True)

    if not kept:
        return []

    # 4. Extract traces from S₁
    t0 = time.time()
    masks_only = [m for _, m, _ in kept]
    # Prefer residual_S1; fall back to residual_S if Stage 1 was skipped (no ROIs subtracted)
    trace_src = fov.residual_S1_path if fov.residual_S1_path is not None else fov.residual_S_path
    traces = extract_traces_from_residual(trace_src, fov.shape, masks_only,
                                          chunk=cfg.reconstruct_chunk)
    print(f"  extracted {len(kept)} traces from {trace_src.name} in {time.time()-t0:.2f}s",
          flush=True)

    # 5. Package as ROI objects — Gate 2 fills in spatial features later
    rois: list[ROI] = []
    next_label = starting_label_id
    for (stat_idx, mask, prob), trace in zip(kept, traces):
        area = int(mask.sum())
        if area == 0:
            continue
        roi = ROI(
            mask=mask,
            label_id=next_label,
            source_stage=2,
            confidence="moderate",           # provisional, overwritten by Gate 2
            gate_outcome="accept",           # provisional
            area=area,
            solidity=0.0,                    # computed in Gate 2
            eccentricity=0.0,                # computed in Gate 2
            nuclear_shadow_score=0.0,        # not a Stage 2 feature
            soma_surround_contrast=0.0,      # not a Stage 2 feature
            iscell_prob=prob,
            trace=trace,
            features={"suite2p_stat_index": int(stat_idx)},
        )
        rois.append(roi)
        next_label += 1

    return rois
