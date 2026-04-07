"""
ROI G. Biv --- Branch C: Tonic neuron detection.

Detects tonically active neurons that Suite2p's SVD-based detection misses
because their near-constant firing produces low temporal variance.

Algorithm (Steps 6--7 of the pipeline):
  1. Stream data.bin and compute truncated SVD (temporal compression)
  2. Bandpass-filter the temporal components to isolate calcium fluctuations
  3. Compute a per-pixel local correlation score in SVD space
  4. Threshold + connected-component labeling + size filter -> uint16 masks

All heavy computation stays in SVD space (k ~ 500 components), avoiding
full-movie reconstruction. Peak RAM ~ 2 GB for 512x512 FOVs.

Provides
--------
svd_from_binary()      -- streaming randomized SVD from Suite2p data.bin
bandpass_temporal()     -- zero-phase Butterworth on temporal components
local_corr_svd()       -- local correlation map via spatial convolution
detect_tonic_rois()    -- threshold + label + size filter -> masks
run_tonic_detection()  -- end-to-end per-FOV pipeline
"""

import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import label, uniform_filter
from skimage.measure import regionprops


# ---------------------------------------------------------------------------
# SVD from data.bin (streaming, memory-bounded)
# ---------------------------------------------------------------------------

def svd_from_binary(
    bin_path: Path,
    Ly: int,
    Lx: int,
    n_components: int = 500,
    chunk_frames: int = 500,
    n_oversamples: int = 50,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute truncated SVD from Suite2p data.bin via streaming randomized SVD.

    Uses two passes over the memmap:
      Pass 1: Y = (A - mean) @ Omega  (random projection)
      Pass 2: B = Q^T @ (A - mean)    (project onto range basis)

    Then: U_hat, S, V = svd(B) ; U = Q @ U_hat

    Parameters
    ----------
    bin_path       : path to data.bin (int16 memmap, shape T x Ly x Lx)
    Ly, Lx         : spatial dimensions
    n_components   : number of SVD components to retain
    chunk_frames   : frames per read (controls peak RAM)
    n_oversamples  : oversampling for randomized SVD accuracy
    seed           : RNG seed for reproducibility

    Returns
    -------
    U : (T, k) temporal components
    S : (k,)   singular values
    V : (k, N) spatial components (N = Ly * Lx)
    T : int    number of frames
    """
    bin_path = Path(bin_path)
    N = Ly * Lx
    k = n_components + n_oversamples

    # Open memmap
    data = np.memmap(str(bin_path), dtype=np.int16, mode="r")
    T = data.shape[0] // N
    data = data.reshape(T, N)

    # --- Pass 0: compute temporal mean ---
    mean = np.zeros(N, dtype=np.float64)
    for t0 in range(0, T, chunk_frames):
        t1 = min(t0 + chunk_frames, T)
        mean += data[t0:t1].astype(np.float64).sum(axis=0)
    mean /= T
    mean = mean.astype(np.float32)

    # --- Pass 1: random projection Y = (A - mean) @ Omega ---
    rng = np.random.default_rng(seed)
    Omega = rng.standard_normal((N, k)).astype(np.float32)
    Y = np.zeros((T, k), dtype=np.float32)

    for t0 in range(0, T, chunk_frames):
        t1 = min(t0 + chunk_frames, T)
        chunk = data[t0:t1].astype(np.float32) - mean[np.newaxis, :]
        Y[t0:t1] = chunk @ Omega

    # QR factorization of Y
    Q, _ = np.linalg.qr(Y, mode="reduced")  # (T, k)

    # --- Pass 2: B = Q^T @ (A - mean) ---
    B = np.zeros((k, N), dtype=np.float32)

    for t0 in range(0, T, chunk_frames):
        t1 = min(t0 + chunk_frames, T)
        chunk = data[t0:t1].astype(np.float32) - mean[np.newaxis, :]
        B += Q[t0:t1].T @ chunk

    # SVD of the small matrix B
    U_hat, S, V = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat  # (T, k)

    # Truncate to n_components
    U = U[:, :n_components]
    S = S[:n_components]
    V = V[:n_components, :]

    del data  # release memmap
    return U, S, V, T


# ---------------------------------------------------------------------------
# Bandpass filtering
# ---------------------------------------------------------------------------

def bandpass_temporal(
    U: np.ndarray,
    fs: float,
    flo: float,
    fhi: float,
    order: int = 3,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass to each temporal component.

    Parameters
    ----------
    U    : (T, k) temporal SVD components
    fs   : sampling rate in Hz
    flo  : low cutoff frequency in Hz
    fhi  : high cutoff frequency in Hz
    order: Butterworth filter order

    Returns
    -------
    U_bp : (T, k) bandpass-filtered temporal components
    """
    nyq = fs / 2.0
    lo = flo / nyq
    hi = fhi / nyq

    # Clamp to valid Butterworth range
    lo = max(lo, 1e-6)
    hi = min(hi, 1.0 - 1e-6)

    sos = butter(order, [lo, hi], btype="band", output="sos")
    U_bp = sosfiltfilt(sos, U, axis=0).astype(np.float32)
    return U_bp


# ---------------------------------------------------------------------------
# Local correlation map (computed entirely in SVD space)
# ---------------------------------------------------------------------------

def local_corr_svd(
    U_bp: np.ndarray,
    S: np.ndarray,
    V: np.ndarray,
    Ly: int,
    Lx: int,
    radius: int = 8,
) -> np.ndarray:
    """Compute per-pixel correlation with local neighborhood mean in SVD space.

    For each pixel p, computes Pearson correlation between p's bandpass-filtered
    timecourse and the mean timecourse of all pixels within *radius* of p.
    This identifies soma-sized islands of correlated activity.

    The computation avoids reconstructing the full movie by working with the
    SVD representation:
      signal(t, y, x) = U_bp(t, :) @ diag(S) @ V(:, y*Lx+x)

    Local mean is computed by convolving each spatial component with a disk
    kernel (approximated by uniform_filter with size=2*radius+1).

    Parameters
    ----------
    U_bp   : (T, k) bandpass-filtered temporal components
    S      : (k,) singular values
    V      : (k, N) spatial components
    Ly, Lx : spatial dimensions
    radius : neighborhood radius in pixels

    Returns
    -------
    corr_map : (Ly, Lx) float32 local correlation score per pixel
    """
    k = len(S)
    N = Ly * Lx
    diam = 2 * radius + 1

    # Weighted spatial components: A[i] = S[i] * V[i] reshaped to (Ly, Lx)
    A = (S[:, np.newaxis] * V).reshape(k, Ly, Lx)  # (k, Ly, Lx)

    # Local mean spatial components: disk convolution per component
    A_bar = np.empty_like(A)
    for i in range(k):
        A_bar[i] = uniform_filter(A[i], size=diam, mode="constant")

    # Temporal covariance matrix of filtered components: C = (1/T) * U_bp^T @ U_bp
    T = U_bp.shape[0]
    C = (U_bp.T @ U_bp) / T  # (k, k)

    # Temporal mean of each component: mu = (1/T) * U_bp^T @ 1
    mu = U_bp.mean(axis=0)  # (k,)

    # Reshape to 2D for vectorized computation
    A_2d = A.reshape(k, N)       # (k, N)
    Ab_2d = A_bar.reshape(k, N)  # (k, N)

    # Cross terms: cov(f, f_bar) = A^T C A_bar - (A^T mu)(A_bar^T mu)
    CA_bar = C @ Ab_2d          # (k, N)
    cross = (A_2d * CA_bar).sum(axis=0)  # (N,)
    mean_f = A_2d.T @ mu        # (N,)
    mean_fb = Ab_2d.T @ mu      # (N,)
    cov = cross - mean_f * mean_fb

    # Variance of f
    CA = C @ A_2d
    var_f = (A_2d * CA).sum(axis=0) - mean_f ** 2

    # Variance of f_bar
    CAb = C @ Ab_2d
    var_fb = (Ab_2d * CAb).sum(axis=0) - mean_fb ** 2

    # Pearson correlation
    denom = np.sqrt(np.maximum(var_f * var_fb, 1e-12))
    corr = cov / denom

    return corr.reshape(Ly, Lx).astype(np.float32)


# ---------------------------------------------------------------------------
# ROI extraction from correlation map
# ---------------------------------------------------------------------------

def detect_tonic_rois(
    corr_map: np.ndarray,
    corr_threshold: float = 0.25,   # was 0.15
    min_size: int = 80,
    max_size: int = 350,             # was 300
    min_solidity: float = 0.6,
    max_eccentricity: float = 0.85,
) -> np.ndarray:
    """Threshold correlation map and extract soma-sized ROIs.

    Parameters
    ----------
    corr_map         : (Ly, Lx) local correlation scores
    corr_threshold   : minimum correlation to include a pixel
    min_size         : minimum ROI area in pixels
    max_size         : maximum ROI area in pixels
    min_solidity     : minimum solidity (area / convex-hull area); rejects
                       spindly/concave neuropil fragments
    max_eccentricity : maximum eccentricity of equivalent ellipse; rejects
                       elongated axon/process fragments

    Returns
    -------
    masks : (Ly, Lx) uint16 labeled ROI mask (0 = background)
    """
    binary = corr_map >= corr_threshold
    labeled, _ = label(binary)

    output = np.zeros_like(labeled, dtype=np.uint16)
    roi_id = 0
    for props in regionprops(labeled):
        if not (min_size <= props.area <= max_size):
            continue
        if props.solidity < min_solidity:          # reject spindly/concave
            continue
        if props.eccentricity > max_eccentricity:  # reject elongated fragments
            continue
        roi_id += 1
        output[labeled == props.label] = roi_id

    return output


# ---------------------------------------------------------------------------
# End-to-end per-FOV pipeline
# ---------------------------------------------------------------------------

def run_tonic_detection(
    bin_path: Path,
    ops_path: Path,
    fs: float,
    band: str = "neuronal",
    n_components: int = 500,
    chunk_frames: int = 500,
    soma_radius: int = 8,
    corr_threshold: float = 0.25,   # was 0.15
    min_size: int = 80,
    max_size: int = 350,             # was 300
    filter_order: int = 3,
    band_lo: float | None = None,
    band_hi: float | None = None,
    min_solidity: float = 0.6,
    max_eccentricity: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run full Branch C pipeline on a single FOV.

    Parameters
    ----------
    bin_path         : path to data.bin
    ops_path         : path to ops.npy (for Ly, Lx)
    fs               : acquisition frame rate in Hz
    band             : "neuronal" (0.05-2.0 Hz) or "astrocyte" (0.01-0.3 Hz)
    n_components     : SVD components to retain
    chunk_frames     : frames per chunk for streaming SVD
    soma_radius      : local correlation neighborhood radius
    corr_threshold   : minimum correlation for ROI inclusion
    min_size         : minimum ROI area in pixels
    max_size         : maximum ROI area in pixels
    filter_order     : Butterworth filter order
    band_lo          : override low cutoff (Hz)
    band_hi          : override high cutoff (Hz)
    min_solidity     : minimum cluster solidity; rejects neuropil fragments
    max_eccentricity : maximum eccentricity; rejects elongated fragments

    Returns
    -------
    masks    : (Ly, Lx) uint16 labeled ROI mask
    corr_map : (Ly, Lx) float32 local correlation map
    info     : dict with n_rois, n_frames, n_components, band, flo, fhi
    """
    ops = np.load(str(ops_path), allow_pickle=True).item()
    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])

    # Resolve frequency band
    if band_lo is not None and band_hi is not None:
        flo, fhi = band_lo, band_hi
    elif band == "astrocyte":
        flo, fhi = 0.01, 0.3
    else:
        flo, fhi = 0.05, 2.0

    # Step 1: Streaming SVD
    U, S, V, n_frames = svd_from_binary(
        bin_path, Ly, Lx,
        n_components=n_components,
        chunk_frames=chunk_frames,
    )

    # Step 2: Bandpass filter temporal components
    U_bp = bandpass_temporal(U, fs, flo, fhi, order=filter_order)

    # Step 3: Local correlation map
    corr_map = local_corr_svd(U_bp, S, V, Ly, Lx, radius=soma_radius)

    # Step 4: Threshold + cluster + size + morphological filter
    masks = detect_tonic_rois(
        corr_map, corr_threshold, min_size, max_size,
        min_solidity=min_solidity, max_eccentricity=max_eccentricity,
    )

    info = {
        "n_rois": int(masks.max()),
        "n_frames": n_frames,
        "n_components": n_components,
        "band": band,
        "flo": flo,
        "fhi": fhi,
    }

    return masks, corr_map, info
