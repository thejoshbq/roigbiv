"""
ROI G. Biv --- Step 9: Trace extraction from registered movie.

For every merged ROI, extracts:
  F(t)     -- raw fluorescence (mean over ROI pixels per frame)
  Fneu(t)  -- neuropil fluorescence (annular surround)
  Fcorr(t) -- neuropil-corrected: F - alpha * Fneu
  dF/F(t)  -- (Fcorr - F0) / F0, rolling 10th percentile baseline
  spks(t)  -- spike deconvolution via OASIS (optional)

All reads from data.bin via numpy memmap (streaming, never loads full movie).
"""

import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

log = logging.getLogger("traces")


# ---------------------------------------------------------------------------
# Neuropil masks
# ---------------------------------------------------------------------------

def build_neuropil_masks(
    roi_mask: np.ndarray,
    inner_radius: int = 2,
    min_pixels: int = 350,
    max_expansion: int = 15,
) -> np.ndarray:
    """Build neuropil annular surround masks for all ROIs.

    For each ROI, the neuropil region is an annulus starting inner_radius
    pixels outside the ROI boundary, expanded until at least min_pixels
    are included. Pixels belonging to any ROI are excluded.

    Parameters
    ----------
    roi_mask       : (Ly, Lx) uint16 labeled mask
    inner_radius   : pixels between ROI boundary and neuropil start
    min_pixels     : minimum neuropil pixel count
    max_expansion  : maximum expansion radius beyond inner_radius

    Returns
    -------
    neuropil_mask : (Ly, Lx) uint16 — label i's neuropil has value i
    """
    Ly, Lx = roi_mask.shape
    labels = np.unique(roi_mask[roi_mask > 0])
    neuropil_mask = np.zeros((Ly, Lx), dtype=np.uint16)
    any_roi = roi_mask > 0  # all ROI pixels (to exclude from neuropil)
    struct = generate_binary_structure(2, 1)  # 4-connected

    for lbl in labels:
        roi_pixels = roi_mask == lbl

        # Expand to create inner boundary
        expanded = roi_pixels.copy()
        for _ in range(inner_radius):
            expanded = binary_dilation(expanded, struct)

        # Grow outward until we reach min_pixels or max_expansion
        neuropil = np.zeros((Ly, Lx), dtype=bool)
        outer = expanded.copy()
        for _ in range(max_expansion):
            outer = binary_dilation(outer, struct)
            candidate = outer & ~expanded & ~any_roi
            neuropil = candidate
            if neuropil.sum() >= min_pixels:
                break

        if neuropil.sum() == 0:
            # Fallback: allow overlap with non-self ROI pixels
            neuropil = outer & ~expanded & ~roi_pixels
            if neuropil.sum() == 0:
                neuropil = outer & ~roi_pixels

        if neuropil.sum() < min_pixels:
            log.debug("ROI %d: only %d neuropil pixels (wanted %d)",
                      lbl, neuropil.sum(), min_pixels)

        neuropil_mask[neuropil] = lbl

    return neuropil_mask


# ---------------------------------------------------------------------------
# Raw trace extraction from data.bin
# ---------------------------------------------------------------------------

def extract_raw_traces(
    bin_path: Path,
    roi_mask: np.ndarray,
    neuropil_mask: np.ndarray,
    Ly: int,
    Lx: int,
    nframes: int,
    chunk_frames: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream data.bin and compute F(t) and Fneu(t) for all ROIs.

    Parameters
    ----------
    bin_path       : path to data.bin (int16, T x Ly x Lx)
    roi_mask       : (Ly, Lx) uint16 merged ROI mask
    neuropil_mask  : (Ly, Lx) uint16 neuropil mask
    Ly, Lx         : spatial dimensions
    nframes        : total number of frames
    chunk_frames   : frames per read chunk

    Returns
    -------
    F    : (n_rois, T) float32 raw fluorescence
    Fneu : (n_rois, T) float32 neuropil fluorescence
    """
    labels = np.unique(roi_mask[roi_mask > 0])
    n_rois = len(labels)
    label_to_idx = {int(l): i for i, l in enumerate(labels)}

    # Precompute flat pixel indices for each ROI and neuropil
    flat_roi = roi_mask.ravel()
    flat_neu = neuropil_mask.ravel()

    roi_indices = {}
    neu_indices = {}
    for lbl in labels:
        lbl_int = int(lbl)
        roi_indices[lbl_int] = np.where(flat_roi == lbl)[0]
        neu_pix = np.where(flat_neu == lbl)[0]
        neu_indices[lbl_int] = neu_pix if len(neu_pix) > 0 else None

    F = np.zeros((n_rois, nframes), dtype=np.float32)
    Fneu = np.zeros((n_rois, nframes), dtype=np.float32)

    movie = np.memmap(str(bin_path), dtype=np.int16, mode="r",
                      shape=(nframes, Ly, Lx))

    for t0 in range(0, nframes, chunk_frames):
        t1 = min(t0 + chunk_frames, nframes)
        chunk = movie[t0:t1].reshape(t1 - t0, -1).astype(np.float32)

        for lbl_int, idx in label_to_idx.items():
            roi_pix = roi_indices[lbl_int]
            F[idx, t0:t1] = chunk[:, roi_pix].mean(axis=1)

            neu_pix = neu_indices[lbl_int]
            if neu_pix is not None:
                Fneu[idx, t0:t1] = chunk[:, neu_pix].mean(axis=1)

    return F, Fneu


# ---------------------------------------------------------------------------
# Neuropil contamination coefficient
# ---------------------------------------------------------------------------

def estimate_alpha(
    F: np.ndarray,
    Fneu: np.ndarray,
    n_iter: int = 10,
) -> np.ndarray:
    """Estimate neuropil contamination coefficient alpha per ROI.

    Uses iterative bisection to find alpha in [0, 1] that minimizes
    the 10th percentile of (F - alpha * Fneu).

    Parameters
    ----------
    F      : (n_rois, T) raw fluorescence
    Fneu   : (n_rois, T) neuropil fluorescence
    n_iter : bisection iterations

    Returns
    -------
    alpha : (n_rois,) float32
    """
    n_rois = F.shape[0]
    alpha = np.full(n_rois, 0.7, dtype=np.float32)

    for i in range(n_rois):
        lo, hi = 0.0, 1.0
        f_i = F[i]
        fneu_i = Fneu[i]
        if fneu_i.std() < 1e-6:
            alpha[i] = 0.7
            continue

        for _ in range(n_iter):
            mid = (lo + hi) / 2.0
            corrected = f_i - mid * fneu_i
            baseline = np.percentile(corrected, 10)
            if baseline > 0:
                lo = mid
            else:
                hi = mid
        alpha[i] = (lo + hi) / 2.0

    return alpha


# ---------------------------------------------------------------------------
# dF/F with rolling baseline
# ---------------------------------------------------------------------------

def compute_dff(
    Fcorr: np.ndarray,
    window_seconds: float = 60.0,
    fs: float = 30.0,
    percentile: float = 10.0,
    is_tonic: np.ndarray | None = None,
    tonic_multiplier: float = 2.0,
) -> np.ndarray:
    """Compute dF/F = (Fcorr - F0) / F0 with rolling percentile baseline.

    Parameters
    ----------
    Fcorr             : (n_rois, T) corrected fluorescence
    window_seconds    : baseline window in seconds
    fs                : frame rate in Hz
    percentile        : percentile for baseline (default 10th)
    is_tonic          : (n_rois,) bool — tonic ROIs get wider window
    tonic_multiplier  : window multiplier for tonic ROIs

    Returns
    -------
    dFF : (n_rois, T) float32
    """
    n_rois, T = Fcorr.shape
    dFF = np.zeros_like(Fcorr)

    win_frames = int(window_seconds * fs)
    if win_frames < 1:
        win_frames = 1

    for i in range(n_rois):
        w = win_frames
        if is_tonic is not None and is_tonic[i]:
            w = int(w * tonic_multiplier)

        trace = Fcorr[i]
        half_w = w // 2

        # Compute rolling percentile baseline
        F0 = np.zeros(T, dtype=np.float32)
        for t in range(T):
            t_lo = max(0, t - half_w)
            t_hi = min(T, t + half_w + 1)
            F0[t] = np.percentile(trace[t_lo:t_hi], percentile)

        # Avoid division by zero
        F0_safe = np.maximum(F0, 1e-6)
        dFF[i] = (trace - F0) / F0_safe

    return dFF


# ---------------------------------------------------------------------------
# Spike deconvolution
# ---------------------------------------------------------------------------

def deconvolve(
    dFF: np.ndarray,
    tau: float = 1.0,
    fs: float = 30.0,
) -> np.ndarray:
    """Spike deconvolution via OASIS or Suite2p fallback.

    Parameters
    ----------
    dFF : (n_rois, T) dF/F traces
    tau : GCaMP decay time constant in seconds
    fs  : frame rate in Hz

    Returns
    -------
    spks : (n_rois, T) float32 deconvolved spike rates
    """
    # Try Suite2p's built-in deconvolution
    try:
        from suite2p.extraction.dcnv import oasis
        g = np.exp(-1.0 / (tau * fs))
        n_rois, T = dFF.shape
        spks = np.zeros_like(dFF)
        for i in range(n_rois):
            spks[i] = oasis(dFF[i], g=g, lam=0)
        return spks
    except (ImportError, Exception) as exc:
        log.warning("Suite2p OASIS unavailable (%s), using simple thresholding", exc)

    # Simple threshold fallback: rectified dF/F
    spks = np.maximum(dFF, 0)
    return spks


# ---------------------------------------------------------------------------
# Per-FOV pipeline
# ---------------------------------------------------------------------------

def extract_traces_fov(
    stem: str,
    merged_mask_dir: Path,
    s2p_dir: Path,
    out_dir: Path,
    merge_records: list[dict] | None = None,
    fs: float = 30.0,
    tau: float = 1.0,
    inner_radius: int = 2,
    min_neuropil_pixels: int = 350,
    baseline_window: float = 60.0,
    baseline_percentile: float = 10.0,
    tonic_multiplier: float = 2.0,
    do_deconvolve: bool = True,
    chunk_frames: int = 200,
) -> dict:
    """Full trace extraction pipeline for one FOV.

    Reads merged mask from merged_mask_dir/{stem}_merged_masks.tif,
    reads data.bin from s2p_dir/{stem}/suite2p/plane0/data.bin.

    Writes to out_dir/:
      {stem}_F.npy, {stem}_Fneu.npy, {stem}_dFF.npy,
      {stem}_spks.npy, {stem}_alpha.npy

    Resumability: skips if {stem}_F.npy already exists.

    Returns dict with stem, n_rois, n_frames, status.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    f_path = out_dir / f"{stem}_F.npy"
    if f_path.exists():
        log.info("  %s: skipped (traces exist)", stem)
        return {"stem": stem, "status": "skipped"}

    # Load merged mask
    mask_path = merged_mask_dir / f"{stem}_merged_masks.tif"
    if not mask_path.exists():
        log.warning("  %s: merged mask not found", stem)
        return {"stem": stem, "status": "no_mask"}

    import tifffile
    roi_mask = tifffile.imread(str(mask_path))
    if roi_mask.max() == 0:
        log.warning("  %s: empty merged mask", stem)
        return {"stem": stem, "status": "empty_mask"}

    # Load ops for dimensions
    plane_dir = s2p_dir / stem / "suite2p" / "plane0"
    ops_path = plane_dir / "ops.npy"
    bin_path = plane_dir / "data.bin"

    if not ops_path.exists():
        log.warning("  %s: ops.npy not found", stem)
        return {"stem": stem, "status": "no_ops"}
    if not bin_path.exists():
        log.warning("  %s: data.bin not found", stem)
        return {"stem": stem, "status": "no_databin"}

    ops = np.load(str(ops_path), allow_pickle=True).item()
    Ly, Lx = ops["Ly"], ops["Lx"]
    nframes = ops.get("nframes", None)
    if nframes is None:
        # Infer from file size
        file_size = bin_path.stat().st_size
        nframes = file_size // (2 * Ly * Lx)  # int16 = 2 bytes

    n_rois = int(roi_mask.max())
    log.info("  %s: %d ROIs, %d frames (%dx%d)", stem, n_rois, nframes, Ly, Lx)

    # Build neuropil masks
    neuropil_mask = build_neuropil_masks(
        roi_mask, inner_radius=inner_radius,
        min_pixels=min_neuropil_pixels,
    )

    # Extract raw traces
    F, Fneu = extract_raw_traces(
        bin_path, roi_mask, neuropil_mask,
        Ly, Lx, nframes, chunk_frames=chunk_frames,
    )

    # Neuropil correction
    alpha = estimate_alpha(F, Fneu)
    Fcorr = F - alpha[:, None] * Fneu

    # Identify tonic ROIs from merge records
    is_tonic = np.zeros(n_rois, dtype=bool)
    if merge_records is not None:
        for rec in merge_records:
            idx = rec.get("roi_id", 0) - 1
            if 0 <= idx < n_rois:
                branches = rec.get("source_branches", "")
                if "C" in branches:
                    is_tonic[idx] = True

    # dF/F
    dFF = compute_dff(
        Fcorr, window_seconds=baseline_window, fs=fs,
        percentile=baseline_percentile, is_tonic=is_tonic,
        tonic_multiplier=tonic_multiplier,
    )

    # Spike deconvolution
    spks = np.zeros_like(dFF)
    if do_deconvolve:
        spks = deconvolve(dFF, tau=tau, fs=fs)

    # Save
    np.save(str(out_dir / f"{stem}_F.npy"), F)
    np.save(str(out_dir / f"{stem}_Fneu.npy"), Fneu)
    np.save(str(out_dir / f"{stem}_dFF.npy"), dFF)
    np.save(str(out_dir / f"{stem}_spks.npy"), spks)
    np.save(str(out_dir / f"{stem}_alpha.npy"), alpha)

    log.info("    -> %d ROIs, %d frames, mean alpha=%.3f",
             n_rois, nframes, alpha.mean())

    return {
        "stem": stem,
        "n_rois": n_rois,
        "n_frames": nframes,
        "alpha_mean": float(alpha.mean()),
        "status": "done",
    }
