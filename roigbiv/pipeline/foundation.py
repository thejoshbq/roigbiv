"""
ROI G. Biv pipeline — Foundation module.

Wraps Suite2p for motion correction + writes data.bin, then computes:
  - Binned truncated SVD via torch.svd_lowrank (GPU)
  - L+S background separation, streamed per temporal chunk to disk memmap
  - Summary images (mean, max, std, Vcorr) computed on residual S
  - Difference-of-Gaussians nuclear shadow map on denoised mean(S)

Memory strategy (spec §3, Plan agent D2):
  - data.bin is opened as int16 np.memmap (zero RAM cost)
  - Movie is temporally binned to ~5000 frames before SVD (mirrors Suite2p)
  - SVD factors held in RAM (trivial: N_pix × n_svd × 4B ≈ 200 MB at 512² × 200)
  - V_bin is interpolated to full T via nearest-repeat
  - L and S are reconstructed per 500-frame chunk and S is written to a
    disk-backed np.memmap; only one chunk lives in RAM at a time
  - Summary images accumulate via running stats per chunk

See spec §3 for algorithmic detail and §18.1 for parameter defaults.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from roigbiv.pipeline.types import FOVData, PipelineConfig


# ─────────────────────────────────────────────────────────────────────────
# Motion correction (Suite2p wrapper)
# ─────────────────────────────────────────────────────────────────────────

def run_motion_correction(
    tif_path: Path,
    cfg: PipelineConfig,
    output_dir: Path,
) -> tuple[dict, Path, np.ndarray, np.ndarray]:
    """Run Suite2p registration (or skip for *_mc.tif) via the existing wrapper.

    Delegates to roigbiv.suite2p.run_suite2p_fov which:
      - Stages the TIF to a local directory
      - Invokes suite2p.run_s2p.run_s2p() with our ops
      - Writes {output_dir}/{stem}/suite2p/plane0/{ops.npy, data.bin, ...}
      - Retains data.bin (we need it as the input to L+S separation)

    Returns
    -------
    ops             : dict — Suite2p ops dict (contains Ly, Lx, meanImg, xoff, yoff, ...)
    data_bin_path   : Path — {output_dir}/{stem}/suite2p/plane0/data.bin
    motion_x        : (T,) float32 — per-frame rigid x displacement
    motion_y        : (T,) float32 — per-frame rigid y displacement
    """
    from roigbiv.suite2p import run_suite2p_fov

    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    stem = tif_path.stem.replace("_mc", "")
    s2p_root = output_dir / stem  # run_suite2p_fov lands outputs at output_dir/{stem}/suite2p/...

    # run_suite2p_fov wants output_dir as its root; it creates {output_dir}/{stem}/suite2p/...
    # For our nested layout (inference/pipeline/{stem}/suite2p/...), we pass output_dir
    # without the stem — the wrapper appends {stem} itself.
    processed = run_suite2p_fov(
        tif_path,
        output_dir,
        fs=cfg.fs,
        anatomical_only=0,
        tau=cfg.tau,
        do_registration=cfg.do_registration,
        cfg=None,  # don't load pipeline.yaml; we hardcode defaults per plan rule 4
    )

    ops_path = s2p_root / "suite2p" / "plane0" / "ops.npy"
    data_bin_path = s2p_root / "suite2p" / "plane0" / "data.bin"

    if not ops_path.exists():
        raise RuntimeError(f"Suite2p did not produce ops.npy at {ops_path}")
    if not data_bin_path.exists():
        raise RuntimeError(
            f"Suite2p did not produce data.bin at {data_bin_path}. "
            f"Check save_path0 / tiff_list wiring in roigbiv.suite2p."
        )

    ops = np.load(ops_path, allow_pickle=True).item()

    # Motion traces — may be absent when do_registration=False
    motion_x = np.asarray(ops.get("xoff", np.zeros(ops.get("nframes", 0))), dtype=np.float32)
    motion_y = np.asarray(ops.get("yoff", np.zeros(ops.get("nframes", 0))), dtype=np.float32)

    return ops, data_bin_path, motion_x, motion_y


# ─────────────────────────────────────────────────────────────────────────
# Binned SVD + L+S separation
# ─────────────────────────────────────────────────────────────────────────

def _open_data_bin(data_bin_path: Path, Ly: int, Lx: int) -> np.memmap:
    """Open Suite2p's data.bin as an int16 memmap of shape (T, Ly, Lx)."""
    path = Path(data_bin_path)
    nbytes = path.stat().st_size
    bytes_per_frame = Ly * Lx * 2  # int16
    if nbytes % bytes_per_frame != 0:
        raise RuntimeError(
            f"data.bin size {nbytes} is not a multiple of Ly*Lx*2={bytes_per_frame}. "
            f"Check Ly={Ly}, Lx={Lx} from ops.npy."
        )
    T = nbytes // bytes_per_frame
    return np.memmap(str(path), dtype=np.int16, mode="r", shape=(T, Ly, Lx))


def _compute_binned_movie(
    movie: np.memmap,
    target_T_bin: int,
) -> tuple[np.ndarray, int]:
    """Temporally bin a (T, Ly, Lx) memmap to (T_bin, Ly*Lx) float32.

    Bin size = ceil(T / target_T_bin); last bin may be shorter (handled by
    mean-with-correct-denominator).

    Returns (M_bin, bin_size) where M_bin.shape == (T_bin_actual, N_pix).
    Reads in chunks of bin_size frames at a time to bound RAM.
    """
    T, Ly, Lx = movie.shape
    N_pix = Ly * Lx
    bin_size = max(1, int(np.ceil(T / target_T_bin)))
    T_bin = int(np.ceil(T / bin_size))

    M_bin = np.empty((T_bin, N_pix), dtype=np.float32)
    for b in range(T_bin):
        t0 = b * bin_size
        t1 = min(t0 + bin_size, T)
        # Cast int16 -> float32 chunk, reshape (chunk, Ly, Lx) -> (chunk, N_pix), mean over chunk
        chunk = np.asarray(movie[t0:t1], dtype=np.float32).reshape(t1 - t0, N_pix)
        np.mean(chunk, axis=0, out=M_bin[b])

    return M_bin, bin_size


def _binned_svd_gpu(
    M_bin: np.ndarray,
    n_svd: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute top-n_svd truncated SVD of M_bin (T_bin, N_pix) on GPU.

    Returns (U, S, V) where M_bin ≈ V @ diag(S) @ U.T under the spatial
    decomposition convention used downstream:
      - U (N_pix, n_svd)  — spatial components
      - S (n_svd,)        — singular values
      - V (T_bin, n_svd)  — temporal components

    We transpose the raw torch output because we factor M_bin^T (pixels × time)
    so that the "U" matrix indexes pixels directly — convenient for reconstructing
    L at arbitrary spatial subsets.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Move M_bin^T (N_pix, T_bin) to GPU; if it doesn't fit, fall back to CPU
    try:
        A = torch.from_numpy(M_bin.T).to(device)  # shape (N_pix, T_bin)
        U_t, S_t, V_t = torch.svd_lowrank(A, q=int(n_svd), niter=2)
        U = U_t.detach().cpu().numpy().astype(np.float32)       # (N_pix, n_svd)
        S = S_t.detach().cpu().numpy().astype(np.float32)       # (n_svd,)
        V = V_t.detach().cpu().numpy().astype(np.float32)       # (T_bin, n_svd)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        # GPU OOM or unavailable — fall back to CPU
        A = torch.from_numpy(M_bin.T)
        U_t, S_t, V_t = torch.svd_lowrank(A, q=int(n_svd), niter=2)
        U = U_t.numpy().astype(np.float32)
        S = S_t.numpy().astype(np.float32)
        V = V_t.numpy().astype(np.float32)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return U, S, V


def _upsample_V(V_bin: np.ndarray, bin_size: int, T_full: int) -> np.ndarray:
    """Upsample temporal components from T_bin to T_full by repeat.

    V_bin : (T_bin, n_svd)
    returns V_full : (T_full, n_svd) float32

    Repeat each binned timepoint `bin_size` times then truncate to T_full.
    This is a valid approximation for background components (which are slow
    by construction — spec §3.3) because the binning already captured the
    dominant low-frequency structure. High-frequency components that *shouldn't*
    be in the background get naturally suppressed by this step.
    """
    T_bin, k = V_bin.shape
    V_full = np.repeat(V_bin, bin_size, axis=0)[:T_full]
    if V_full.shape[0] < T_full:
        # edge case: T_full not divisible by bin_size and last bin was partial
        pad = np.tile(V_bin[-1:], (T_full - V_full.shape[0], 1))
        V_full = np.concatenate([V_full, pad], axis=0)
    return V_full.astype(np.float32, copy=False)


def compute_background_separation(
    data_bin_path: Path,
    ops: dict,
    cfg: PipelineConfig,
    output_dir: Path,
) -> tuple[Path, Path, int, np.ndarray]:
    """L+S background separation via binned truncated SVD.

    Algorithm (spec §3.3):
      1. Open data.bin as (T, Ly, Lx) int16 memmap.
      2. Temporally bin to ~5000 frames → (T_bin, N_pix) float32.
      3. Compute top-n_svd SVD on the binned movie.
      4. Interpolate V_bin → V_full (T, n_svd).
      5. Stream per-chunk:
           L_chunk = U_k @ diag(S_k) @ V_full[t0:t1].T      (using k = k_background)
           S_chunk = M_chunk - L_chunk
         Write S_chunk to disk memmap at residual_S.dat.
         Accumulate mean(L) during the same pass.
      6. Persist all n_svd SVD factors to svd_factors.npz (for future Stage 2/4 reuse).

    Returns
    -------
    residual_S_path  : Path to (T, Ly, Lx) float32 memmap
    svd_factors_path : Path to .npz with U, S, V_bin (full n_svd components)
    k_used           : int (= cfg.k_background)
    mean_L           : (Ly, Lx) float32 — accumulated mean of L during reconstruction
    """
    Ly = int(ops["Ly"])
    Lx = int(ops["Lx"])
    N_pix = Ly * Lx

    movie = _open_data_bin(data_bin_path, Ly, Lx)
    T = movie.shape[0]

    # 1. Bin movie
    t0 = time.time()
    M_bin, bin_size = _compute_binned_movie(movie, cfg.svd_bin_frames)
    T_bin = M_bin.shape[0]
    print(f"  binned movie ({T}→{T_bin} frames, bin_size={bin_size}) "
          f"in {time.time()-t0:.1f}s", flush=True)

    # 2. SVD on binned
    t0 = time.time()
    n_svd = min(cfg.n_svd, T_bin - 1, N_pix - 1)  # svd rank upper bounds
    U, S, V_bin = _binned_svd_gpu(M_bin, n_svd)
    print(f"  SVD top-{n_svd} on binned movie in {time.time()-t0:.1f}s", flush=True)
    del M_bin  # free ~5 GB

    # 3. Persist SVD factors
    svd_factors_path = output_dir / "svd_factors.npz"
    np.savez(str(svd_factors_path),
             U=U, S=S, V_bin=V_bin, bin_size=np.int32(bin_size), T=np.int32(T))

    # 4. Upsample V for reconstruction
    V_full = _upsample_V(V_bin, bin_size, T)

    # 5. Stream L + S per chunk, accumulate mean(L)
    k = min(int(cfg.k_background), n_svd)
    U_k = U[:, :k]                            # (N_pix, k)
    S_k = S[:k]                               # (k,)
    V_k_full = V_full[:, :k]                  # (T, k)

    US = U_k * S_k[np.newaxis, :]             # (N_pix, k) — precompute U @ diag(S)

    residual_S_path = output_dir / "residual_S.dat"
    S_mm = np.memmap(str(residual_S_path), dtype=np.float32, mode="w+",
                     shape=(T, Ly, Lx))

    mean_L = np.zeros((Ly, Lx), dtype=np.float64)  # accumulator
    chunk = int(cfg.reconstruct_chunk)
    t0 = time.time()
    for t_start in range(0, T, chunk):
        t_end = min(t_start + chunk, T)
        cs = t_end - t_start
        # L_chunk (N_pix, cs) = US @ V_k_full[t_start:t_end].T
        L_flat = US @ V_k_full[t_start:t_end].T   # (N_pix, cs)
        L_chunk = L_flat.T.reshape(cs, Ly, Lx)    # (cs, Ly, Lx)
        # M_chunk (cs, Ly, Lx) float32
        M_chunk = np.asarray(movie[t_start:t_end], dtype=np.float32)
        # S_chunk = M - L
        S_mm[t_start:t_end] = M_chunk - L_chunk
        mean_L += L_chunk.sum(axis=0, dtype=np.float64)
        del L_flat, L_chunk, M_chunk
    S_mm.flush()
    del S_mm
    mean_L = (mean_L / T).astype(np.float32)
    print(f"  L+S reconstruction streamed to {residual_S_path.name} "
          f"({T*Ly*Lx*4/1e9:.1f} GB) in {time.time()-t0:.1f}s", flush=True)

    # Write memmap metadata for future readers
    meta = {"shape": [int(T), int(Ly), int(Lx)], "dtype": "float32"}
    (output_dir / "residual_S.meta.json").write_text(json.dumps(meta, indent=2))

    return residual_S_path, svd_factors_path, k, mean_L


# ─────────────────────────────────────────────────────────────────────────
# Summary images on S
# ─────────────────────────────────────────────────────────────────────────

def _iter_S_chunks(residual_S_path: Path, shape: tuple, chunk: int = 500):
    """Generator yielding (t0, t1, S_chunk) — S_chunk is (cs, Ly, Lx) float32."""
    T, Ly, Lx = shape
    S_mm = np.memmap(str(residual_S_path), dtype=np.float32, mode="r", shape=(T, Ly, Lx))
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        yield t0, t1, np.asarray(S_mm[t0:t1])  # copy chunk out so memmap can advance
    del S_mm


def generate_summary_images(
    residual_S_path: Path,
    shape: tuple,
    chunk: int = 500,
) -> dict:
    """Compute mean, max, std, and 8-neighbor Vcorr projections of residual S.

    All accumulators run in a single pass through S_mm. Memory per projection
    is ~1 MB (H,W float64/32). Vcorr needs 5 accumulators × 8 neighbors ≈ 40 MB.

    Returns dict with keys 'mean', 'max', 'std', 'vcorr', each a (Ly, Lx) float32.
    """
    T, Ly, Lx = shape

    # First pass: running sum, sum-of-squares, max
    sum_ = np.zeros((Ly, Lx), dtype=np.float64)
    sumsq = np.zeros((Ly, Lx), dtype=np.float64)
    max_ = np.full((Ly, Lx), -np.inf, dtype=np.float32)

    # 8-neighbor Vcorr: 4 unique direction pairs (each pixel's correlation with
    # each of 8 neighbors is symmetric, so we accumulate directed pairs and
    # each pixel ends up averaged over its valid 8 neighbors).
    # Offsets: (dy, dx) for the 8 neighbors
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               ( 0, -1),          ( 0, 1),
               ( 1, -1), ( 1, 0), ( 1, 1)]
    # Per-offset accumulators: sum_x, sum_y, sum_xx, sum_yy, sum_xy, count
    # where x is the shifted source and y is the reference.
    # Maintaining these in float64 avoids catastrophic cancellation.
    acc = {
        (dy, dx): {
            "sx": np.zeros((Ly, Lx), dtype=np.float64),
            "sy": np.zeros((Ly, Lx), dtype=np.float64),
            "sxx": np.zeros((Ly, Lx), dtype=np.float64),
            "syy": np.zeros((Ly, Lx), dtype=np.float64),
            "sxy": np.zeros((Ly, Lx), dtype=np.float64),
            "n":  0,  # scalar — all pixels get same count since we process full chunks
        } for (dy, dx) in offsets
    }

    t_total = 0
    for t0, t1, chunk_arr in _iter_S_chunks(residual_S_path, shape, chunk):
        cs = t1 - t0
        t_total += cs

        # Single float64 cast per chunk — shared by sumsq and all 8 Vcorr offsets
        # below. Slicing chunk64 below yields views (no additional copy).
        chunk64 = chunk_arr.astype(np.float64)

        # mean / std / max accumulators (straightforward over full FOV).
        # chunk_arr may be a memmap view — go through chunk64 (in-RAM) so
        # `.max()` doesn't re-scan the residual from disk on every chunk.
        sum_ += chunk64.sum(axis=0)
        sumsq += (chunk64 ** 2).sum(axis=0)
        np.maximum(max_, chunk64.max(axis=0).astype(np.float32), out=max_)

        # Vcorr accumulators — for each offset, accumulate stats of pixel and its shifted neighbor
        # over the valid (non-boundary) region. We use slicing to avoid boundary issues.
        for (dy, dx) in offsets:
            # "center" slice: the pixel we're computing vcorr for
            cy0 = max(0, dy); cy1 = Ly + min(0, dy)
            cx0 = max(0, dx); cx1 = Lx + min(0, dx)
            # "shifted" slice: the neighbor (source of x)
            sy0 = max(0, -dy); sy1 = Ly + min(0, -dy)
            sx0 = max(0, -dx); sx1 = Lx + min(0, -dx)

            # y (reference) and x (neighbor) — views into the pre-cast chunk64
            y_chunk = chunk64[:, cy0:cy1, cx0:cx1]  # (cs, H', W')
            x_chunk = chunk64[:, sy0:sy1, sx0:sx1]

            a = acc[(dy, dx)]
            a["sx"][cy0:cy1, cx0:cx1] += x_chunk.sum(axis=0)
            a["sy"][cy0:cy1, cx0:cx1] += y_chunk.sum(axis=0)
            a["sxx"][cy0:cy1, cx0:cx1] += (x_chunk ** 2).sum(axis=0)
            a["syy"][cy0:cy1, cx0:cx1] += (y_chunk ** 2).sum(axis=0)
            a["sxy"][cy0:cy1, cx0:cx1] += (x_chunk * y_chunk).sum(axis=0)
            a["n"] += cs

    mean = (sum_ / t_total).astype(np.float32)
    var = (sumsq / t_total) - (sum_ / t_total) ** 2
    var = np.maximum(var, 0.0)  # guard numerical negatives
    std = np.sqrt(var).astype(np.float32)

    # Vcorr: average of 8 neighbor correlations per pixel, only counting neighbors
    # that exist (boundary pixels get averaged over fewer neighbors).
    vcorr = np.zeros((Ly, Lx), dtype=np.float64)
    count = np.zeros((Ly, Lx), dtype=np.int32)
    eps = 1e-12
    for (dy, dx), a in acc.items():
        n = a["n"]
        if n == 0:
            continue
        # Validity mask: where this offset had an in-bounds neighbor
        valid = np.zeros((Ly, Lx), dtype=bool)
        cy0 = max(0, dy); cy1 = Ly + min(0, dy)
        cx0 = max(0, dx); cx1 = Lx + min(0, dx)
        valid[cy0:cy1, cx0:cx1] = True

        num = n * a["sxy"] - a["sx"] * a["sy"]
        den = np.sqrt(np.maximum(n * a["sxx"] - a["sx"] ** 2, 0.0) *
                       np.maximum(n * a["syy"] - a["sy"] ** 2, 0.0))
        r = np.where(valid & (den > eps), num / (den + eps), 0.0)
        vcorr += r
        count += valid.astype(np.int32)

    vcorr = (vcorr / np.maximum(count, 1)).astype(np.float32)

    return {"mean": mean, "max": max_.astype(np.float32),
            "std": std, "vcorr": vcorr}


# ─────────────────────────────────────────────────────────────────────────
# Nuclear shadow (DoG) map
# ─────────────────────────────────────────────────────────────────────────

def compute_nuclear_shadow_map(
    mean_S: np.ndarray,
    sigma_inner: float = 2.0,
    sigma_outer: float = 6.0,
) -> np.ndarray:
    """Difference-of-Gaussians nuclear shadow score (spec §3.4, §4).

    Convention: DoG = G(σ_outer) - G(σ_inner) so that a pixel at the *dark
    nucleus* center of a cell with cytoplasmic GCaMP gives a POSITIVE score
    — the narrow Gaussian picks up the dark nucleus (low value), the wide
    Gaussian averages over soma+surround (higher value), so G(wide)-G(narrow)
    is positive at the nucleus.

    This matches the spec semantic: "strong positive response indicates likely
    soma" when evaluated at the ROI centroid (which sits on the nucleus for
    cells with visible GCaMP-excluded nuclei).
    """
    from scipy.ndimage import gaussian_filter
    g_outer = gaussian_filter(mean_S.astype(np.float32), sigma=sigma_outer)
    g_inner = gaussian_filter(mean_S.astype(np.float32), sigma=sigma_inner)
    return (g_outer - g_inner).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────

def run_foundation(
    tif_path: Path,
    cfg: PipelineConfig,
    output_dir: Path,
) -> FOVData:
    """Run Foundation: motion correction + L+S + summary images + DoG.

    Writes to {output_dir}:
      suite2p/plane0/{ops.npy, data.bin, ...}
      svd_factors.npz, residual_S.dat (+ .meta.json), motion_trace.npz
      summary/{mean_S,max_S,std_S,vcorr_S,mean_L,dog_map}.tif

    Returns a populated FOVData with summary images in RAM and paths to
    heavy arrays on disk.
    """
    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary").mkdir(exist_ok=True)

    print("Foundation: motion correction via Suite2p wrapper", flush=True)
    ops, data_bin_path, motion_x, motion_y = run_motion_correction(tif_path, cfg, output_dir)
    Ly = int(ops["Ly"]); Lx = int(ops["Lx"])
    # Determine T from data.bin size (more reliable than ops fields across Suite2p versions)
    T = Path(data_bin_path).stat().st_size // (Ly * Lx * 2)
    print(f"  ops: T={T}, Ly={Ly}, Lx={Lx}  (fs={cfg.fs} tau={cfg.tau} "
          f"registration={'ON' if cfg.do_registration else 'OFF'})", flush=True)

    # Persist motion traces (spec §3.1 Blindspot 9 — for future Gate 4)
    np.savez(str(output_dir / "motion_trace.npz"),
             xoff=motion_x, yoff=motion_y, fs=np.float32(cfg.fs))

    print(f"Foundation: L+S background separation (k_background={cfg.k_background}, "
          f"n_svd={cfg.n_svd})", flush=True)
    residual_S_path, svd_factors_path, k_used, mean_L = compute_background_separation(
        data_bin_path, ops, cfg, output_dir,
    )

    print("Foundation: summary images (mean, max, std, vcorr) on S", flush=True)
    t0 = time.time()
    # Cap at 128 frames/chunk regardless of reconstruct_chunk. Each chunk
    # allocates `chunk64` (cs·Ly·Lx·8 B) plus transient `(chunk64 ** 2)` /
    # per-offset temporaries of the same size; with cs=500 on a 505×493 FOV
    # that's ~2 GB peak, which swaps on RAM-constrained hosts and stalls for
    # >10 min. cs=128 caps peak at ~500 MB — comfortable on 16 GB systems
    # and still a single sequential pass through the residual memmap.
    summary_chunk = min(128, int(cfg.reconstruct_chunk))
    summaries = generate_summary_images(residual_S_path, (T, Ly, Lx),
                                        chunk=summary_chunk)
    mean_S = summaries["mean"]
    max_S = summaries["max"]
    std_S = summaries["std"]
    vcorr_S = summaries["vcorr"]
    print(f"  computed in {time.time()-t0:.1f}s", flush=True)

    # Raw movie mean (morphological channel for Cellpose).
    # With top-k SVD-based L, mean_S ≈ 0 because the first few components
    # absorb per-pixel brightness. mean_M preserves the raw morphological contrast
    # that Cellpose's training regime expects (spec §4 "morphological contrast channel").
    mean_M = np.asarray(ops.get("meanImg"), dtype=np.float32)
    if mean_M is None or mean_M.shape != (Ly, Lx):
        # Fallback: reconstruct from data.bin (should rarely be needed)
        movie = _open_data_bin(data_bin_path, Ly, Lx)
        mean_M = np.zeros((Ly, Lx), dtype=np.float64)
        for t0_ in range(0, T, cfg.reconstruct_chunk):
            t1_ = min(t0_ + cfg.reconstruct_chunk, T)
            mean_M += np.asarray(movie[t0_:t1_], dtype=np.float64).sum(axis=0)
        mean_M = (mean_M / T).astype(np.float32)
        del movie

    print("Foundation: nuclear shadow (DoG) map on mean_M", flush=True)
    # DoG on mean_M (raw brightness), since mean_S is near-zero under SVD L+S.
    # The nuclear-shadow pattern is visible in the raw morphological image.
    dog_map = compute_nuclear_shadow_map(mean_M)

    # Save all summary images as .tif
    summary_dir = output_dir / "summary"
    for name, arr in [("mean_M", mean_M), ("mean_S", mean_S),
                      ("max_S", max_S), ("std_S", std_S),
                      ("vcorr_S", vcorr_S), ("mean_L", mean_L), ("dog_map", dog_map)]:
        tifffile.imwrite(str(summary_dir / f"{name}.tif"), arr.astype(np.float32))

    # Lightweight ops snapshot (drop heavy arrays)
    ops_snapshot = {k: v for k, v in ops.items()
                    if isinstance(v, (int, float, str, bool, list, tuple))}

    return FOVData(
        raw_path=tif_path,
        output_dir=output_dir,
        data_bin_path=data_bin_path,
        shape=(T, Ly, Lx),
        residual_S_path=residual_S_path,
        mean_M=mean_M,
        mean_S=mean_S,
        max_S=max_S,
        std_S=std_S,
        vcorr_S=vcorr_S,
        dog_map=dog_map,
        mean_L=mean_L,
        svd_factors_path=svd_factors_path,
        motion_x=motion_x,
        motion_y=motion_y,
        k_background=k_used,
        rois=[],
        stage_counts={},
        ops=ops_snapshot,
    )
