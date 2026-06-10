"""
ROI G. Biv pipeline — Stage 3: Template Sweep (spec §9).

Detects sparse-firing neurons (1–15 transients across a session) whose
individual events are visible but that don't rank in Suite2p's SVD because
they fire too rarely. This is the first fully in-house detection stage.

Algorithm:
  1. Build L2-normalized template bank keyed on fs and tau.
  2. For each spatial chunk of the (T, H, W) residual (S₂):
       a. Pull pixel traces into GPU memory (n_pix, T).
       b. Compute per-pixel MAD-based σ for noise normalization.
       c. FFT-based cross-correlation against each template, take max across
          templates → score_max (n_pix, T).
       d. Threshold at cfg.template_threshold σ; collect suprathreshold events.
  3. Spatially cluster events (fcluster, single-linkage, 12 px threshold).
  4. For each cluster: count temporally-independent events (≥ 2 s apart),
     build a disk mask, extract the cluster's trace from S₂.
  5. Return as candidate ROI objects (gate_outcome=provisional, Gate 3 fills in).

Memory strategy:
  - The CPU staging buffer `chunk_pixels` of shape (rows_per_chunk*W, T) float32
    is pre-allocated ONCE outside the chunk loop and reused via np.copyto.
    The previous per-iteration `np.ascontiguousarray(... .reshape(...))` form
    leaked anonymous heap (a strong reference somewhere across the
    torch.from_numpy().to(device) hop prevented refcount-GC), accumulating
    `chunk_pixels.nbytes` per iteration — for a 27k-frame 1024² FOV that
    grew to 114 GB and tripped the kernel OOM-killer.
  - `rows_per_chunk` auto-scales via cfg.stage3_chunk_budget_bytes so the
    per-chunk float32 working set never exceeds the budget (default 1 GB).
  - GPU buffers inside _process_chunk are managed by PyTorch's CUDA caching
    allocator — same-shape allocations are reused across iterations.

Spec deviation (documented in pipeline log):
  σ is computed as per-pixel GLOBAL MAD rather than the sliding-window
  σ_local(p, t) called for in spec §9.2. The approximation is defensible —
  MAD-per-pixel still normalizes away per-pixel brightness scaling, and the
  sliding-window variant requires materializing (n_pix, T) of MAD values
  which costs another ~300 MB per chunk.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.pipeline.stage2 import extract_traces_from_residual
from roigbiv.pipeline.device import cuda_compute_capable
from roigbiv.pipeline.types import FOVData, PipelineConfig, ROI


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _pad_templates(template_bank: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Zero-pad templates to common length, preserving L2 normalization."""
    L_max = max(len(wf) for _, wf in template_bank)
    K = len(template_bank)
    out = np.zeros((K, L_max), dtype=np.float32)
    for k, (_, wf) in enumerate(template_bank):
        out[k, : len(wf)] = wf
    return out


def _disk_mask(cy: float, cx: float, radius: int, H: int, W: int) -> np.ndarray:
    """Filled circular disk, clipped to image bounds."""
    ys, xs = np.ogrid[:H, :W]
    return ((ys - cy) ** 2 + (xs - cx) ** 2) <= radius ** 2


def _resolve_rows_per_chunk(cfg: PipelineConfig, T: int, W: int, n_fft: int) -> int:
    """Cap rows_per_chunk so the peak per-chunk working set fits the budget.

    Accounts for all dominant allocations in _process_chunk (per pixel row):
      chunk_pixels (CPU staging) : W * T * 4          float32
      x_freq (rfft output)       : W * (n_fft//2+1)*8  complex64
      score_max                  : W * T * 4          float32
      xcorr_k (irfft, full n_fft): W * n_fft * 4     float32  ← dominates for large n_fft

    Always returns ≥ 1.
    """
    requested = int(getattr(cfg, "stage3_pixel_chunk_rows", 8))
    budget = int(getattr(cfg, "stage3_chunk_budget_bytes", 1_073_741_824))
    n_rfft = n_fft // 2 + 1
    per_row_bytes = max(
        1,
        W * T * 4          # chunk_pixels
        + W * n_rfft * 8   # x_freq (complex64)
        + W * T * 4        # score_max
        + W * n_fft * 4    # xcorr_k irfft output (pre-slice)
    )
    max_rows_by_budget = max(1, budget // per_row_bytes)
    return max(1, min(requested, max_rows_by_budget))


# ─────────────────────────────────────────────────────────────────────────
# Per-chunk FFT matched filter
# ─────────────────────────────────────────────────────────────────────────

def _process_chunk(
    chunk_pixels: np.ndarray,          # (n_pix, T) float32
    template_freqs,                    # torch complex tensor (K, n_rfft)
    n_fft: int,
    T: int,
    threshold: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFT cross-correlate all pixels against all templates on GPU.

    Returns parallel flat arrays of suprathreshold events:
        pixel_indices (M,) int64  — row into chunk_pixels
        times         (M,) int64
        scores        (M,) float32
        template_idx  (M,) int8
    """
    import torch

    n_pix = chunk_pixels.shape[0]
    K = template_freqs.shape[0]

    # All GPU tensors are released in the ``finally`` even if an op raises
    # mid-loop (e.g. CUDA OOM). Without this, a leaked chunk's tensors persist
    # in the caching allocator and accumulate across chunks — the failure class
    # behind the historical 114 GB blow-up on long FOVs.
    x = x_freq = score_max = template_idx_max = mask = event_coords = None
    try:
        x = torch.from_numpy(chunk_pixels).to(device)  # (n_pix, T)

        # Per-pixel MAD → σ (Gaussian-equivalent)
        med = torch.median(x, dim=1).values                                # (n_pix,)
        mad = torch.median(torch.abs(x - med[:, None]), dim=1).values      # (n_pix,)
        sigma = torch.clamp(mad / 0.6745, min=1e-6)                        # (n_pix,)

        # FFT of traces once
        x_freq = torch.fft.rfft(x, n=n_fft, dim=1)                         # (n_pix, n_rfft)

        # Running max across templates — avoids materializing (n_pix, K, T)
        score_max = torch.full((n_pix, T), -float("inf"), device=device)
        template_idx_max = torch.zeros((n_pix, T), dtype=torch.int8, device=device)

        for k in range(K):
            # Cross-correlation via FFT: ifft(fft(trace) * conj(fft(template)))
            xcorr_k = torch.fft.irfft(x_freq * torch.conj(template_freqs[k : k + 1]),
                                      n=n_fft, dim=1)[:, :T]               # (n_pix, T)
            xcorr_k = xcorr_k / sigma[:, None]
            update = xcorr_k > score_max
            score_max = torch.where(update, xcorr_k, score_max)
            template_idx_max = torch.where(
                update,
                torch.full_like(template_idx_max, k),
                template_idx_max,
            )

        # Threshold. If the event count per chunk is catastrophic (typical of
        # structured residuals at low σ), adaptively raise the threshold until
        # the chunk yields ≤ max_per_chunk events — effectively turning the
        # threshold into a "top-K by score" selector.
        max_per_chunk = 200_000
        effective_threshold = threshold
        mask = score_max > effective_threshold
        n_events = int(mask.sum().item())
        bumps = 0
        while n_events > max_per_chunk and bumps < 8:
            effective_threshold += 1.0
            mask = score_max > effective_threshold
            n_events = int(mask.sum().item())
            bumps += 1

        event_coords = mask.nonzero(as_tuple=False)
        pixel_idx = event_coords[:, 0].cpu().numpy()
        times = event_coords[:, 1].cpu().numpy()
        scores = score_max[event_coords[:, 0], event_coords[:, 1]].cpu().numpy()
        tmpl_idx = template_idx_max[event_coords[:, 0], event_coords[:, 1]].cpu().numpy()
    finally:
        del x, x_freq, score_max, template_idx_max, mask, event_coords
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pixel_idx.astype(np.int64), times.astype(np.int64), scores.astype(np.float32), tmpl_idx.astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────
# Event clustering (§9.4)
# ─────────────────────────────────────────────────────────────────────────

def _cluster_events_spatial(
    ys: np.ndarray, xs: np.ndarray,
    distance_threshold: float,
) -> np.ndarray:
    """Hierarchical single-linkage clustering of 2-D event centroids.

    Returns cluster_ids (n_events,) int64 — 1-indexed, one id per input event.
    Events closer than `distance_threshold` chain into the same cluster.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = len(ys)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.array([1], dtype=np.int64)

    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    # pdist is O(n²) memory and time. At n=20k that's 400M distances ≈ 1.6 GB —
    # borderline. Above 20k we use a vectorized grid-snap clustering: assign
    # every point to a coarse grid cell of size `distance_threshold`, then
    # merge adjacent cells. This is O(n) and produces compact clusters.
    if n > 20_000:
        return _grid_cluster(pts, distance_threshold)

    Z = linkage(pdist(pts), method="single")
    return fcluster(Z, t=distance_threshold, criterion="distance").astype(np.int64)


def _grid_cluster(pts: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Vectorized grid-snap clustering — O(n) in n_events.

    Snap each event to a grid with cell size == distance_threshold. Events
    in the same cell are in the same cluster. This is an approximation to
    single-linkage at `distance_threshold`: adjacent grid cells with chain
    connections aren't merged, which may split some chains. For Stage 3's
    event accumulation the approximation is acceptable because real cells
    produce >>1 event in a cell, and the fcluster-greedy switchover only
    happens when the event count is already pathological.
    """
    cell = max(float(distance_threshold), 1.0)
    keys = np.floor(pts / cell).astype(np.int64)
    # Combine (y_cell, x_cell) into a single int for unique-based ID assignment
    combined = keys[:, 0] * 10_000 + keys[:, 1]
    _, inverse = np.unique(combined, return_inverse=True)
    return (inverse + 1).astype(np.int64)


def _count_temporally_independent(
    times: np.ndarray,
    scores: np.ndarray,
    min_separation_frames: int,
) -> tuple[int, list[int]]:
    """Greedy-pick events separated by at least min_separation_frames.

    Sort by score (descending), pick highest-score events that are at least
    `min_separation_frames` away from already-picked ones.

    Returns (count, picked_indices_into_input).
    """
    n = len(times)
    if n == 0:
        return 0, []
    order = np.argsort(-scores)  # descending
    picked: list[int] = []
    picked_times: list[int] = []
    for idx in order:
        t = int(times[idx])
        if all(abs(t - pt) >= min_separation_frames for pt in picked_times):
            picked.append(int(idx))
            picked_times.append(t)
    return len(picked), picked


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────

def run_stage3(
    residual_view,
    fov: FOVData,
    template_bank: list[tuple[str, np.ndarray]],
    cfg: PipelineConfig,
    starting_label_id: int = 1,
) -> list[ROI]:
    """Full Stage 3: FFT matched filter → event clustering → ROI candidates.

    Parameters
    ----------
    residual_view : ResidualView reconstructing the (T, H, W) residual (typically S₂)
    fov           : FOVData (uses .shape, .mean_M)
    template_bank : output of stage3_templates.build_template_bank
    cfg           : PipelineConfig
    starting_label_id : int — first label to assign (usually next after Stage 2)

    Returns
    -------
    list of ROI objects with source_stage=3, event_count, and trace populated.
    Gate 3 is applied by the caller afterward.
    """
    import torch

    T, H, W = fov.shape
    device = "cpu" if cfg.force_cpu else ("cuda" if cuda_compute_capable() else "cpu")

    # Pad templates, move to GPU, precompute FFTs
    templates_padded = _pad_templates(template_bank)           # (K, L_max)
    K, L_max = templates_padded.shape
    n_fft = int(2 ** np.ceil(np.log2(max(T + L_max - 1, 2))))
    print(f"  template bank: {K} templates, L_max={L_max}, n_fft={n_fft}", flush=True)

    tmpl_tensor = torch.from_numpy(templates_padded).to(device)
    template_freqs = torch.fft.rfft(tmpl_tensor, n=n_fft, dim=1)  # (K, n_rfft)

    # Collect events across spatial chunks
    all_y: list[np.ndarray] = []
    all_x: list[np.ndarray] = []
    all_t: list[np.ndarray] = []
    all_score: list[np.ndarray] = []
    all_tmpl: list[np.ndarray] = []

    rows_per_chunk = _resolve_rows_per_chunk(cfg, T=T, W=W, n_fft=n_fft)
    if rows_per_chunk != int(cfg.stage3_pixel_chunk_rows):
        print(
            f"  rows_per_chunk auto-scaled "
            f"{cfg.stage3_pixel_chunk_rows} → {rows_per_chunk} "
            f"(budget {cfg.stage3_chunk_budget_bytes/1e9:.2f} GB, T={T}, W={W})",
            flush=True,
        )

    # Pre-allocate the CPU staging buffer ONCE; reused across all chunks. The
    # last chunk may be smaller — it writes into chunk_pixels[:n_pix].
    chunk_pixels = np.empty((rows_per_chunk * W, T), dtype=np.float32)

    t_fft_total = 0.0
    for y0 in range(0, H, rows_per_chunk):
        y1 = min(y0 + rows_per_chunk, H)
        chunk_rows = y1 - y0
        n_pix = chunk_rows * W
        t0 = time.time()
        # Fill the pre-allocated buffer in place. The memmap-slice transpose
        # is a non-contiguous view (zero allocation); np.copyto handles the
        # strided read into our contiguous destination buffer without
        # materializing an intermediate. This is what kills the per-iteration
        # anonymous-heap accumulation that previously caused the OOM.
        src_view = residual_view.read_rows(y0, y1).transpose(1, 2, 0)  # (chunk_rows, W, T)
        np.copyto(chunk_pixels[:n_pix].reshape(chunk_rows, W, T), src_view)

        pixel_idx, times, scores, tmpl_idx = _process_chunk(
            chunk_pixels[:n_pix], template_freqs, n_fft, T,
            threshold=cfg.template_threshold,
            device=device,
        )
        t_fft_total += time.time() - t0

        if pixel_idx.size > 0:
            # Map pixel_idx (row within chunk) → (y, x)
            y_local = pixel_idx // W
            x_local = pixel_idx % W
            all_y.append(y_local + y0)
            all_x.append(x_local)
            all_t.append(times)
            all_score.append(scores)
            all_tmpl.append(tmpl_idx)

    del tmpl_tensor, template_freqs, chunk_pixels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_y:
        print(f"  Stage 3: 0 events detected in {t_fft_total:.1f}s", flush=True)
        return []

    ys = np.concatenate(all_y)
    xs = np.concatenate(all_x)
    ts = np.concatenate(all_t)
    scs = np.concatenate(all_score)
    tis = np.concatenate(all_tmpl)
    n_events = len(ys)
    print(f"  Stage 3: {n_events} suprathreshold events in {t_fft_total:.1f}s", flush=True)

    # Global cap: if still too many events, keep top-K by score. This path is
    # unusual — real calcium data with a 6σ threshold typically yields < 1e6
    # events. Hitting the cap implies structured (non-Gaussian) residual.
    if n_events > cfg.stage3_max_events:
        top_k_idx = np.argsort(-scs)[: cfg.stage3_max_events]
        ys = ys[top_k_idx]; xs = xs[top_k_idx]; ts = ts[top_k_idx]
        scs = scs[top_k_idx]; tis = tis[top_k_idx]
        n_events = len(ys)
        print(f"  [cap] retained top {n_events} events by score "
              f"(above stage3_max_events={cfg.stage3_max_events})", flush=True)

    # ── Spatial clustering ────────────────────────────────────────────────
    t0 = time.time()
    cluster_ids = _cluster_events_spatial(
        ys.astype(np.float64), xs.astype(np.float64),
        distance_threshold=cfg.cluster_distance,
    )
    n_clusters = int(cluster_ids.max()) if cluster_ids.size else 0
    print(f"  {n_clusters} spatial clusters (distance≤{cfg.cluster_distance}px) "
          f"in {time.time()-t0:.2f}s", flush=True)

    # ── Per-cluster candidate construction ────────────────────────────────
    min_sep_frames = int(cfg.min_event_separation * cfg.fs)
    candidate_info: list[dict] = []
    for cid in range(1, n_clusters + 1):
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            continue
        cy = float(ys[idx].mean())
        cx = float(xs[idx].mean())
        # Temporally independent event count (per spec §9.4, §10)
        count_indep, indep_picks = _count_temporally_independent(
            ts[idx], scs[idx], min_separation_frames=min_sep_frames,
        )
        if count_indep < 1:
            continue
        candidate_info.append({
            "centroid_y": cy,
            "centroid_x": cx,
            "event_count": int(count_indep),
            "mean_score": float(scs[idx].mean()),
            "n_raw_events": int(idx.size),
            # Retain the full event list for Gate 3's waveform check
            "events": [
                {"y": int(ys[i]), "x": int(xs[i]), "frame": int(ts[i]),
                 "score": float(scs[i]), "template_idx": int(tis[i])}
                for i in idx.tolist()
            ],
            "picked_events": [
                {"y": int(ys[idx[p]]), "x": int(xs[idx[p]]), "frame": int(ts[idx[p]]),
                 "score": float(scs[idx[p]]), "template_idx": int(tis[idx[p]])}
                for p in indep_picks
            ],
        })
    print(f"  {len(candidate_info)} candidates after temporal-independence filter "
          f"(≥{cfg.min_event_separation}s separation)", flush=True)
    if not candidate_info:
        return []

    # ── Extract traces from residual using disk masks ─────────────────────
    t0 = time.time()
    disk_masks = [
        _disk_mask(c["centroid_y"], c["centroid_x"],
                   radius=cfg.spatial_pool_radius, H=H, W=W)
        for c in candidate_info
    ]
    traces = extract_traces_from_residual(
        residual_view, disk_masks, chunk=cfg.reconstruct_chunk,
    )
    print(f"  extracted {len(disk_masks)} candidate traces "
          f"in {time.time()-t0:.2f}s", flush=True)

    # ── Package as ROI objects ────────────────────────────────────────────
    rois: list[ROI] = []
    next_label = starting_label_id
    for info, mask, trace in zip(candidate_info, disk_masks, traces):
        area = int(mask.sum())
        if area == 0:
            continue
        roi = ROI(
            mask=mask,
            label_id=next_label,
            source_stage=3,
            confidence="moderate",      # provisional, Gate 3 overwrites
            gate_outcome="accept",      # provisional
            area=area,
            solidity=0.0,               # computed in Gate 3
            eccentricity=0.0,           # computed in Gate 3
            nuclear_shadow_score=0.0,
            soma_surround_contrast=0.0,
            event_count=info["event_count"],
            trace=trace,
            features={
                "centroid_y": info["centroid_y"],
                "centroid_x": info["centroid_x"],
                "mean_event_score": info["mean_score"],
                "n_raw_events": info["n_raw_events"],
                "events": info["events"],
                "picked_events": info["picked_events"],
            },
        )
        rois.append(roi)
        next_label += 1

    return rois
