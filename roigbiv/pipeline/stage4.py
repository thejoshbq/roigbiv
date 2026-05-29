"""
ROI G. Biv pipeline — Stage 4: Tonic Neuron Search (spec §11).

Detects tonic neurons — sustained-firing cells (2-5+ Hz) whose transients pile
up into quasi-constant fluorescence under τ≈1s GCaMP6s. These cells have low
temporal variance (Suite2p misses them), no discrete transients (Stage 3
misses them), and are partially absorbed into the low-rank L component during
Foundation (Blindspot 4).

Detection signal: spatially confined micro-fluctuations in the calcium-relevant
frequency band. After bandpass filtering isolates that band on S₃, the
inner-vs-outer correlation contrast (6px disk vs 6-15px annulus) discriminates
somata from neuropil — somata correlate locally but decorrelate at distance;
neuropil correlates broadly at similar levels in both zones.

Algorithm:
  1. Per-pixel linear detrend (removes residual drift / photobleaching —
     Blindspot 10) — done ONCE, reused across all windows.
  2. For each of three bandpass windows (fast 0.5-2 Hz, medium 0.1-1 Hz,
     slow 0.05-0.5 Hz; Blindspot 12): zero-phase Butterworth (sosfiltfilt),
     processed spatially-chunked (time axis must be intact for zero-phase).
  3. Temporal compression via binned averaging (shared time bins across
     pixels preserve pairwise correlation structure exactly).
  4. Correlation contrast map via spatial convolution (NOT all-pairs —
     that would be 256GB of matrix for a 512×512 FOV).
  5. Connected-component clustering + morphological filtering.
  6. Cross-window IoU merge: duplicates detected in multiple windows
     retained as higher-confidence (recorded via n_windows_detected).

Memory discipline: process ONE bandpass window at a time. The detrended
movie and the currently-filtered movie live as memmaps in a temp dir; all
ephemeral data discarded before the next window starts. Peak RAM < 1GB.
"""
from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import convolve as ndi_convolve, label as ndi_label
from skimage.measure import regionprops
from threadpoolctl import threadpool_limits

from roigbiv.pipeline.stage2 import extract_traces_from_residual
from roigbiv.pipeline.types import ROI, FOVData, PipelineConfig


# ─────────────────────────────────────────────────────────────────────────
# Stage 1: per-pixel linear detrend (shared across bandpass windows)
# ─────────────────────────────────────────────────────────────────────────

def detrend_to_memmap(
    in_path: Path,
    out_path: Path,
    shape: tuple,
    chunk_rows: int = 16,
) -> None:
    """Per-pixel linear detrend on a (T, H, W) memmap.

    Vectorized OLS: subtract slope*t + intercept from each pixel's timecourse.
    Streams spatial chunks so RAM stays bounded regardless of T.

    Writes a new float32 memmap at `out_path`. Does NOT modify input.
    """
    T, H, W = shape
    t_c = np.arange(T, dtype=np.float32) - (T - 1) / 2.0
    denom = float((t_c ** 2).sum())

    in_mm = np.memmap(str(in_path), dtype=np.float32, mode="r", shape=shape)
    out_mm = np.memmap(str(out_path), dtype=np.float32, mode="w+", shape=shape)

    for y0 in range(0, H, chunk_rows):
        y1 = min(y0 + chunk_rows, H)
        chunk = np.asarray(in_mm[:, y0:y1, :], dtype=np.float32)     # (T, h, W)
        intercept = chunk.mean(axis=0)                                # (h, W)
        slope = np.tensordot(t_c, chunk, axes=(0, 0)) / denom         # (h, W)
        out_mm[:, y0:y1, :] = (
            chunk - (t_c[:, None, None] * slope[None, :, :]
                     + intercept[None, :, :])
        ).astype(np.float32)

    out_mm.flush()
    del out_mm, in_mm


# ─────────────────────────────────────────────────────────────────────────
# Stage 2: zero-phase Butterworth bandpass filter (per bandpass window)
# ─────────────────────────────────────────────────────────────────────────

def bandpass_to_memmap(
    in_path: Path,
    out_path: Path,
    shape: tuple,
    fs: float,
    low: float,
    high: float,
    order: int = 4,
    chunk_rows: int = 16,
) -> None:
    """Zero-phase bandpass filter via scipy.signal.sosfiltfilt.

    Must chunk in SPACE, not time — sosfiltfilt runs forward+backward along
    the time axis and needs the full timecourse per pixel.
    """
    T, H, W = shape
    nyq = 0.5 * fs
    if not (0 < low < high < nyq):
        raise ValueError(f"Invalid bandpass: [{low}, {high}] Hz with fs={fs}")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")

    in_mm = np.memmap(str(in_path), dtype=np.float32, mode="r", shape=shape)
    out_mm = np.memmap(str(out_path), dtype=np.float32, mode="w+", shape=shape)

    for y0 in range(0, H, chunk_rows):
        y1 = min(y0 + chunk_rows, H)
        chunk = np.asarray(in_mm[:, y0:y1, :], dtype=np.float32)     # (T, h, W)
        chunk_flat = chunk.reshape(T, -1)                            # (T, h*W)
        filtered = sosfiltfilt(sos, chunk_flat, axis=0).astype(np.float32)
        out_mm[:, y0:y1, :] = filtered.reshape(T, y1 - y0, W)

    out_mm.flush()
    del out_mm, in_mm


# ─────────────────────────────────────────────────────────────────────────
# Stage 3: temporal compression for tractable correlation computation
# ─────────────────────────────────────────────────────────────────────────

def compress_temporal(
    filtered_path: Path,
    shape: tuple,
    n_components: int,
) -> np.ndarray:
    """Temporally compress a (T, H, W) memmap to a (N_pixels, D) matrix.

    Uses binned temporal averaging: divide T into D bins of size T//D, mean
    each bin. All pixels share the same bin edges so pairwise correlations
    are preserved exactly (it's an orthogonal projection onto the constant-
    per-bin basis).

    Returns
    -------
    compressed : (H*W, D) float32 — each row is a pixel's D-dim summary.
    """
    T, H, W = shape
    D = int(min(n_components, T))
    if D <= 0:
        raise ValueError(f"n_components must be positive (got {n_components})")
    bin_size = max(1, T // D)
    n_bins = T // bin_size

    mm = np.memmap(str(filtered_path), dtype=np.float32, mode="r", shape=shape)
    compressed = np.zeros((H * W, n_bins), dtype=np.float32)
    for b in range(n_bins):
        t0 = b * bin_size
        t1 = t0 + bin_size
        bin_mean = np.asarray(mm[t0:t1], dtype=np.float32).mean(axis=0)   # (H, W)
        compressed[:, b] = bin_mean.ravel()
    del mm
    return compressed


# ─────────────────────────────────────────────────────────────────────────
# Stage 4: correlation contrast via spatial convolution
# ─────────────────────────────────────────────────────────────────────────

def uniform_disk_kernel(radius: int, exclude_center: bool = False) -> tuple[np.ndarray, int]:
    """Return (normalized disk kernel, pixel count).

    When exclude_center=True, the central pixel is removed from the disk
    (useful for inner-radius kernels so the self-correlation of z-scored
    data doesn't bias the mean upward).
    """
    if radius < 1:
        raise ValueError(f"radius must be >= 1 (got {radius})")
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (y * y + x * x) <= radius * radius
    if exclude_center:
        disk[radius, radius] = False
    n = int(disk.sum())
    if n == 0:
        raise ValueError(f"Empty disk kernel (radius={radius}, exclude_center={exclude_center})")
    kernel = disk.astype(np.float32) / n
    return kernel, n


def compute_correlation_contrast(
    compressed: np.ndarray,
    image_shape: tuple,
    inner_radius: int,
    outer_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel correlation contrast via spatial convolution.

    For each pixel p:
      inner_corr(p) = mean Pearson correlation with neighbors within
                      `inner_radius` (excluding self)
      outer_corr(p) = mean Pearson correlation with neighbors in the
                      annulus from inner_radius+1 to outer_radius
      contrast(p)   = inner_corr(p) - outer_corr(p)

    Identity used: for per-pixel z-scored vectors,
        corr(x, y) = (1/D) * sum_d z_x[d] * z_y[d]
    and the mean over a neighborhood of q can be computed by
    spatially-convolving z[:,:,d] with the uniform disk kernel. This
    reduces an otherwise O(N²) pairwise-correlation computation to
    O(D * H * W * kernel_size) per radius.

    Returns
    -------
    contrast_map   : (H, W) float32
    inner_corr_map : (H, W) float32 — useful as a per-ROI quality feature
    """
    H, W = image_shape
    N, D = compressed.shape
    if N != H * W:
        raise ValueError(f"compressed has {N} rows; expected {H * W}")
    if outer_radius <= inner_radius:
        raise ValueError(f"outer_radius ({outer_radius}) must exceed inner_radius ({inner_radius})")

    # Per-pixel z-score along the compressed (D) axis
    mu = compressed.mean(axis=1, keepdims=True)
    std = compressed.std(axis=1, keepdims=True)
    # Pixels with zero variance produce all-zero z — safe (they contribute 0)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    z = ((compressed - mu) / std).astype(np.float32)          # (N, D)

    # Kernels: inner disk (self-excluded), inner disk (full), outer disk (full)
    inner_nc_k, n_inner_nc = uniform_disk_kernel(inner_radius, exclude_center=True)
    inner_full_k, n_inner_full = uniform_disk_kernel(inner_radius, exclude_center=False)
    outer_full_k, n_outer_full = uniform_disk_kernel(outer_radius, exclude_center=False)
    n_annulus = n_outer_full - n_inner_full
    if n_annulus <= 0:
        raise ValueError(f"Annulus empty: n_outer={n_outer_full} <= n_inner={n_inner_full}")

    inner_corr = np.zeros((H, W), dtype=np.float32)
    outer_corr = np.zeros((H, W), dtype=np.float32)

    for d in range(D):
        zd = z[:, d].reshape(H, W)
        # Inner (self-excluded) mean of z_d at each pixel
        inner_sm = ndi_convolve(zd, inner_nc_k, mode="reflect")
        # Outer disk & inner disk (full) — used to derive the annulus mean
        inner_full_sm = ndi_convolve(zd, inner_full_k, mode="reflect")
        outer_full_sm = ndi_convolve(zd, outer_full_k, mode="reflect")
        # Annulus = (outer_disk * n_outer - inner_disk * n_inner) / n_annulus
        annulus_sm = (outer_full_sm * n_outer_full - inner_full_sm * n_inner_full) / n_annulus

        inner_corr += zd * inner_sm
        outer_corr += zd * annulus_sm

    inner_corr /= D
    outer_corr /= D
    contrast = (inner_corr - outer_corr).astype(np.float32)
    return contrast, inner_corr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Stage 5: cluster correlation-contrast hotspots into candidate masks
# ─────────────────────────────────────────────────────────────────────────

def cluster_contrast_map(
    contrast_map: np.ndarray,
    inner_corr_map: np.ndarray,
    window_name: str,
    cfg: PipelineConfig,
) -> list[dict]:
    """Threshold + connected-component labeling + morphological filtering.

    Returns a list of candidate dicts. Each dict has keys:
      mask (H,W bool), centroid_y, centroid_x, area, solidity,
      eccentricity, corr_contrast, mean_intra_corr, bandpass_window.
    """
    binary = contrast_map > cfg.corr_contrast_threshold
    labels, n_labels = ndi_label(binary)
    if n_labels == 0:
        return []

    candidates: list[dict] = []
    for region in regionprops(labels):
        area = int(region.area)
        if area < cfg.stage4_min_area or area > cfg.stage4_max_area:
            continue
        solidity = float(region.solidity) if region.solidity is not None else 0.0
        eccentricity = float(region.eccentricity) if region.eccentricity is not None else 1.0
        if solidity < cfg.stage4_min_solidity:
            continue
        if eccentricity > cfg.stage4_max_eccentricity:
            continue
        mask = (labels == region.label)
        candidates.append({
            "mask": mask,
            "centroid_y": float(region.centroid[0]),
            "centroid_x": float(region.centroid[1]),
            "area": area,
            "solidity": solidity,
            "eccentricity": eccentricity,
            "corr_contrast": float(contrast_map[mask].mean()),
            "mean_intra_corr": float(inner_corr_map[mask].mean()),
            "bandpass_window": window_name,
        })
    return candidates


# ─────────────────────────────────────────────────────────────────────────
# Stage 6: cross-window IoU merge
# ─────────────────────────────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def merge_across_windows(
    candidates_per_window: list[list[dict]],
    iou_threshold: float,
) -> list[dict]:
    """Greedy IoU merge across bandpass windows.

    Pool all candidates, sort by corr_contrast descending, then greedily
    pick highest-score candidates and discard subsequent candidates whose
    IoU with an already-picked candidate exceeds the threshold. The
    additional window is recorded on the winner (n_windows_detected,
    bandpass_windows_detected).
    """
    pooled: list[dict] = []
    for window_cands in candidates_per_window:
        pooled.extend(window_cands)
    if not pooled:
        return []

    pooled.sort(key=lambda c: c["corr_contrast"], reverse=True)

    merged: list[dict] = []
    for cand in pooled:
        merged_into = None
        for m in merged:
            if _iou(cand["mask"], m["mask"]) > iou_threshold:
                merged_into = m
                break
        if merged_into is None:
            winner = dict(cand)
            winner["n_windows_detected"] = 1
            winner["bandpass_windows_detected"] = [cand["bandpass_window"]]
            merged.append(winner)
        else:
            win = cand["bandpass_window"]
            if win not in merged_into["bandpass_windows_detected"]:
                merged_into["bandpass_windows_detected"].append(win)
                merged_into["n_windows_detected"] += 1
    return merged


# ─────────────────────────────────────────────────────────────────────────
# Per-window worker (used by the orchestrator, optionally in a thread pool)
# ─────────────────────────────────────────────────────────────────────────

def _run_one_window(
    window_name: str,
    low: float,
    high: float,
    detrended_path: Path,
    shape: tuple,
    tmp: Path,
    cfg: PipelineConfig,
) -> tuple[str, Optional[np.ndarray], list[dict]]:
    """Run one bandpass window end-to-end: filter → compress → contrast → cluster.

    Returns (window_name, contrast_map_or_None, candidates). contrast_map is
    None when the recording is too short to stably filter this band; in that
    case candidates is an empty list and the caller records a skip.

    Thread-safe: reads the shared detrended memmap, writes its own per-window
    filtered memmap under `tmp`, and mutates no shared state. All prints go
    through the process stdout (captured by the Streamlit _QueueStreamer).
    """
    T, _, _ = shape
    H, W = shape[1], shape[2]

    # Minimum recording length for filter stability: 5 cycles of low freq
    min_duration_s = 5.0 / low
    if T / cfg.fs < min_duration_s:
        msg = (f"Stage 4 [{window_name}]: skipping ({low}-{high} Hz) — "
               f"recording too short "
               f"({T / cfg.fs:.1f}s < {min_duration_s:.1f}s required)")
        print(f"  WARNING: {msg}", flush=True)
        return window_name, None, []

    filtered_path = tmp / f"filtered_{window_name}.dat"
    print(f"Stage 4 [{window_name}]: bandpass [{low}, {high}] Hz", flush=True)
    bandpass_to_memmap(detrended_path, filtered_path, shape,
                       cfg.fs, low, high,
                       order=cfg.bandpass_order,
                       chunk_rows=cfg.stage4_pixel_chunk_rows)

    compressed = compress_temporal(filtered_path, shape,
                                   cfg.n_svd_components_stage4)

    contrast_map, inner_corr_map = compute_correlation_contrast(
        compressed, (H, W),
        cfg.corr_neighbor_radius_inner,
        cfg.corr_neighbor_radius_outer,
    )
    del compressed

    window_cands = cluster_contrast_map(contrast_map, inner_corr_map,
                                        window_name, cfg)
    print(f"Stage 4 [{window_name}]: {len(window_cands)} clusters "
          f"(thr={cfg.corr_contrast_threshold:.2f})", flush=True)

    # Explicit deletion to help the memmap be released
    try:
        filtered_path.unlink()
    except FileNotFoundError:
        pass

    return window_name, contrast_map.copy(), window_cands


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────

def run_stage4(
    residual_S3_path: Path,
    fov: FOVData,
    cfg: PipelineConfig,
    starting_label_id: int = 1,
) -> list[ROI]:
    """Run Stage 4: multi-scale bandpass → correlation contrast → cluster.

    Memory discipline:
      - S₃ and the detrended movie are memory-mapped (float32 on disk).
      - Each filtered movie is written to a tempfile memmap, consumed for
        clustering, then discarded before the next window starts.
      - The compressed matrix (N_pixels × D ≈ 300MB) is the only large
        in-RAM object at any time.
      - tempfile.TemporaryDirectory guarantees cleanup on exit or error.

    Populates fov.corr_contrast_maps with per-window (H, W) maps for the
    napari viewer and HITL review.

    Returns candidate ROIs (pre-Gate 4). Each has source_stage=4, the raw
    trace extracted from S₃, and provisional gate_outcome="flag" that Gate 4
    will overwrite.
    """
    T, H, W = fov.shape

    # Open S₃ read-only to verify existence
    if not Path(residual_S3_path).exists():
        raise FileNotFoundError(f"Stage 4 needs residual_S3 at {residual_S3_path}")

    fov.corr_contrast_maps = {}
    candidates_per_window: list[list[dict]] = []
    windows_processed: list[str] = []

    with tempfile.TemporaryDirectory(prefix="roigbiv_stage4_") as tmpdir:
        tmp = Path(tmpdir)
        detrended_path = tmp / "S3_detrended.dat"

        print(f"Stage 4: detrending S₃ (T={T}, H={H}, W={W})", flush=True)
        detrend_to_memmap(residual_S3_path, detrended_path,
                          fov.shape, chunk_rows=cfg.stage4_pixel_chunk_rows)

        n_windows = len(cfg.bandpass_windows)
        n_workers = max(1, min(int(cfg.stage4_n_workers), n_windows))

        if n_workers == 1:
            # Serial path — preserves original behavior when the pool is disabled.
            results = [
                _run_one_window(wn, lo, hi, detrended_path, fov.shape, tmp, cfg)
                for wn, (lo, hi) in cfg.bandpass_windows
            ]
        else:
            # Threads share the detrended memmap. sosfiltfilt + ndi_convolve
            # release the GIL, so parallelism is real. Cap per-worker BLAS
            # threads to prevent oversubscription on the machine's core count.
            blas_limit = max(1, (os.cpu_count() or 4) // n_workers)
            print(f"Stage 4: processing {n_windows} bandpass windows "
                  f"with {n_workers} worker thread(s) "
                  f"(BLAS threads/worker={blas_limit})", flush=True)
            with threadpool_limits(limits=blas_limit, user_api="blas"):
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [
                        ex.submit(_run_one_window, wn, lo, hi,
                                  detrended_path, fov.shape, tmp, cfg)
                        for wn, (lo, hi) in cfg.bandpass_windows
                    ]
                    # Collect in submit order so merge sees deterministic input.
                    results = [f.result() for f in futures]

        # Main-thread mutations only — no dict races on fov.corr_contrast_maps.
        for window_name, contrast_map, window_cands in results:
            if contrast_map is None:
                candidates_per_window.append([])
                continue
            fov.corr_contrast_maps[window_name] = contrast_map
            candidates_per_window.append(window_cands)
            windows_processed.append(window_name)

        merged = merge_across_windows(candidates_per_window,
                                      cfg.stage4_iou_merge_threshold)
        print(f"Stage 4: {len(merged)} merged candidates "
              f"across {len(windows_processed)} windows", flush=True)

        # Extract traces from S₃ for surviving candidates
        if merged:
            masks = [c["mask"] for c in merged]
            traces = extract_traces_from_residual(residual_S3_path, fov.shape, masks,
                                                  chunk=cfg.reconstruct_chunk)
        else:
            traces = np.zeros((0, T), dtype=np.float32)

    # ── Package into ROI objects ─────────────────────────────────────────
    rois: list[ROI] = []
    next_label = int(starting_label_id)
    for cand, trace in zip(merged, traces):
        roi = ROI(
            mask=cand["mask"].astype(bool),
            label_id=next_label,
            source_stage=4,
            confidence="requires_review",   # locked in for Stage 4 per spec §12
            gate_outcome="flag",            # provisional — Gate 4 may overwrite with "reject"
            area=int(cand["area"]),
            solidity=float(cand["solidity"]),
            eccentricity=float(cand["eccentricity"]),
            nuclear_shadow_score=0.0,
            soma_surround_contrast=0.0,
            corr_contrast=float(cand["corr_contrast"]),
            trace=trace.astype(np.float32),
            features={
                "centroid_y": float(cand["centroid_y"]),
                "centroid_x": float(cand["centroid_x"]),
                "mean_intra_corr": float(cand["mean_intra_corr"]),
                "n_windows_detected": int(cand["n_windows_detected"]),
                "bandpass_windows_detected": list(cand["bandpass_windows_detected"]),
            },
        )
        rois.append(roi)
        next_label += 1

    return rois
