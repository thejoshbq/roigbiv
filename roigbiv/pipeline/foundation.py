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

from roigbiv.pipeline import fmt
from roigbiv.pipeline.device import cuda_compute_capable
from roigbiv.pipeline.types import FOVData, PipelineConfig, BranchView


# ─────────────────────────────────────────────────────────────────────────
# Motion correction (Suite2p wrapper)
# ─────────────────────────────────────────────────────────────────────────

def _mc_stage_label(do_registration: bool) -> str:
    """Foundation stage header text for the actual motion-correction mode.

    ``do_registration`` is the correct single proxy across all three backends:
    when False the input was detected as pre-corrected and Suite2p runs
    detection-only, so the header must not imply registration occurred.
    """
    if do_registration:
        return "Motion correction"
    return "Motion correction (detection-only · pre-corrected input)"


def run_motion_correction(
    tif_path: Path,
    cfg: PipelineConfig,
    output_dir: Path,
    gpu_lock=None,
) -> tuple[dict, Path, np.ndarray, np.ndarray]:
    """Motion-correct one FOV via the configured backend, write ``{stem}_mc.tif``.

    Dispatches on ``cfg.motion_correction_backend``:
      - ``"phasecorr"`` (default) — Suite2p does both registration and detection
        in one pass; a ``{stem}_mc.tif`` is exported afterward. Robust on dim,
        shot-noise-dominated frames (it smooths + builds an iterative reference),
        so it matches the legacy SIMA quality where ``rowwise-pcc`` regresses.
      - ``"rowwise-pcc"`` (opt-in) — GPU row-wise non-rigid phase correlation
        (:func:`roigbiv.pipeline.registration.run_rowwise_pcc_register`) writes a
        corrected ``{stem}_mc.tif``; Suite2p then runs **detection-only**
        (``do_registration=False``) on it to produce ``data.bin``/``stat.npy``.
        Fast on high-SNR data but injects noise-driven per-row warps on low-SNR
        FOVs (hazy, horizontally-banded mean) — see the MC bench audit.
      - ``"legacy"`` (opt-in) — genuine SIMA ``HiddenMarkov2D(granularity='row')``
        run in the ``sima-legacy`` py3.8 sidecar conda env via subprocess
        (:func:`roigbiv.pipeline.legacy_mc.run_sima_legacy_register`). Writes a
        corrected ``{stem}_mc.tif``; Suite2p then runs **detection-only** on it,
        exactly like ``rowwise-pcc``. CPU-only and slow (tens of minutes to hours
        per FOV); faithful reproduction of the legacy notebook's correction.

    Either way Suite2p produces the int16 ``data.bin`` + ``stat.npy``/``ops.npy``
    that Foundation's L+S separation and Stage 2 consume; only *which* algorithm
    corrected the motion differs. Returns the uniform
    ``(ops, data_bin_path, motion_x, motion_y)`` tuple, with motion traces coming
    from whichever backend did the correction.

    Throughout, an :class:`~roigbiv.pipeline.mc_preview.MCPreviewWriter` streams
    raw/corrected frame pairs to ``{output_dir}/mc_preview/`` for the UI's live
    view. It is diagnostic-only and cannot change the registered output.
    """
    from roigbiv.pipeline.mc_preview import writer_for
    from roigbiv.pipeline.registration import (
        run_rowwise_pcc_register, _write_mc_tif)
    from roigbiv.suite2p import run_suite2p_fov

    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    stem = tif_path.stem.replace("_mc", "")
    backend = getattr(cfg, "motion_correction_backend", "phasecorr")

    if backend not in ("rowwise-pcc", "phasecorr", "legacy"):
        raise ValueError(
            f"Unknown motion_correction_backend {backend!r}; "
            f"expected 'rowwise-pcc', 'phasecorr', or 'legacy'."
        )

    with writer_for(cfg, output_dir, stem=stem, backend=backend) as preview:
        return _run_motion_correction(
            tif_path, cfg, output_dir, stem, backend, gpu_lock, preview,
            run_rowwise_pcc_register, run_suite2p_fov, _write_mc_tif)


def _run_motion_correction(
    tif_path: Path,
    cfg: PipelineConfig,
    output_dir: Path,
    stem: str,
    backend: str,
    gpu_lock,
    preview,
    run_rowwise_pcc_register,
    run_suite2p_fov,
    _write_mc_tif,
) -> tuple[dict, Path, np.ndarray, np.ndarray]:
    """Backend dispatch body of :func:`run_motion_correction` (preview-scoped)."""
    rowwise_motion: tuple | None = None

    if backend == "rowwise-pcc":
        # 1. Pre-register on the GPU → corrected {stem}_mc.tif in the output dir.
        mc_tif_path, motion_x, motion_y = run_rowwise_pcc_register(
            tif_path,
            output_dir,
            fs=cfg.fs,
            do_registration=cfg.do_registration,
            max_displacement=getattr(cfg, "mc_max_displacement", 50),
            strip_height=getattr(cfg, "mc_strip_height", 32),
            n_template_iters=getattr(cfg, "mc_n_template_iters", 2),
            subpixel_upsample=getattr(cfg, "mc_subpixel_upsample", 10),
            smooth_sigma_rows=getattr(cfg, "mc_smooth_sigma_rows", 6.0),
            smooth_sigma_time=getattr(cfg, "mc_smooth_sigma_time", 1.0),
            prefilter=getattr(cfg, "mc_prefilter", False),
            prefilter_sigma_low=getattr(cfg, "mc_prefilter_sigma_low", 1.0),
            prefilter_sigma_high=getattr(cfg, "mc_prefilter_sigma_high", 8.0),
            strip_confidence_weight=getattr(cfg, "mc_strip_confidence_weight", True),
            frame_batch=getattr(cfg, "mc_frame_batch", 256),
            force_cpu=cfg.force_cpu,
            gpu_lock=gpu_lock,
            preview=preview,
        )
        rowwise_motion = (motion_x, motion_y)
        # 2. Suite2p detection-only on the already-corrected movie.
        s2p_input = mc_tif_path
        s2p_do_registration = False
    elif backend == "legacy":
        # Genuine SIMA HiddenMarkov2D in the sima-legacy sidecar env (subprocess).
        # No live preview: the correction happens in a py3.8 child process with
        # no in-process hook, so say so rather than showing an empty card.
        preview.set_phase(
            "unsupported",
            note="legacy (SIMA) runs in a subprocess sidecar; no live preview")
        from roigbiv.pipeline.legacy_mc import run_sima_legacy_register
        mc_tif_path, motion_x, motion_y = run_sima_legacy_register(
            tif_path,
            output_dir,
            fs=cfg.fs,
            do_registration=cfg.do_registration,
            max_displacement=getattr(cfg, "mc_max_displacement", 50),
            granularity=getattr(cfg, "mc_granularity", "row"),
            sima_env=getattr(cfg, "mc_sima_env", "sima-legacy"),
            gpu_lock=gpu_lock,  # accepted and ignored (SIMA is CPU-only)
        )
        rowwise_motion = (motion_x, motion_y)
        # Suite2p detection-only on the SIMA-corrected movie.
        s2p_input = mc_tif_path
        s2p_do_registration = False
    else:
        s2p_input = tif_path
        s2p_do_registration = cfg.do_registration

    s2p_root = output_dir / stem  # run_suite2p_fov lands outputs at output_dir/{stem}/suite2p/...

    # Forward the phasecorr registration knobs from PipelineConfig into the
    # Suite2p ops dict. Only *registration* keys are passed, so detection params
    # stay at their _build_ops defaults (unchanged from the old cfg=None path).
    # Defaults of these mc_s2p_* fields equal Suite2p's own, so a stock run is
    # byte-identical; they diverge only when explicitly tuned. (For rowwise-pcc/
    # legacy this is do_registration=False, so the keys are inert.)
    s2p_reg_cfg = {"suite2p": {
        "block_size":            getattr(cfg, "mc_s2p_block_size", [64, 64]),
        "smooth_sigma":          getattr(cfg, "mc_s2p_smooth_sigma", 1.15),
        "smooth_sigma_time":     getattr(cfg, "mc_s2p_smooth_sigma_time", 0.0),
        "maxregshift":           getattr(cfg, "mc_s2p_maxregshift", 0.1),
        "nonrigid":              getattr(cfg, "mc_s2p_nonrigid", True),
        "maxregshiftNR":         getattr(cfg, "mc_s2p_maxregshift_nr", 5),
        "nimg_init":             getattr(cfg, "mc_s2p_nimg_init", 300),
        "two_step_registration": getattr(cfg, "mc_s2p_two_step_registration", False),
        "1Preg":                 getattr(cfg, "mc_s2p_one_photon_reg", True),
        "spatial_hp_reg":        getattr(cfg, "mc_s2p_spatial_hp_reg", 42),
        "pre_smooth":            getattr(cfg, "mc_s2p_pre_smooth", 0.0),
        "spatial_taper":         getattr(cfg, "mc_s2p_spatial_taper", 40.0),
    }}

    # run_suite2p_fov wants output_dir as its root; it creates {output_dir}/{stem}/suite2p/...
    # The preview is only handed over when Suite2p is the thing doing the
    # correcting: under rowwise-pcc/legacy this call is detection-only and the
    # preview has already been filled (or marked unsupported) by that backend.
    run_suite2p_fov(
        s2p_input,
        output_dir,
        fs=cfg.fs,
        anatomical_only=0,
        tau=cfg.tau,
        do_registration=s2p_do_registration,
        cfg=s2p_reg_cfg,
        preview=preview if backend == "phasecorr" else None,
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

    if rowwise_motion is not None:
        # rowwise-pcc did the correction; use its traces (Suite2p's are ~0 here).
        motion_x, motion_y = rowwise_motion
    else:
        motion_x = np.asarray(ops.get("xoff", np.zeros(ops.get("nframes", 0))), dtype=np.float32)
        motion_y = np.asarray(ops.get("yoff", np.zeros(ops.get("nframes", 0))), dtype=np.float32)
        # phasecorr: export {stem}_mc.tif from the Suite2p-registered data.bin.
        try:
            _write_mc_tif(data_bin_path, output_dir / f"{stem}_mc.tif",
                          int(ops["Ly"]), int(ops["Lx"]))
        except Exception as exc:  # noqa: BLE001 — export is a convenience artifact
            print(f"  WARN: could not export {stem}_mc.tif: {exc}", flush=True)

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
    force_cpu: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute top-n_svd truncated SVD of M_bin (T_bin, N_pix) on GPU or CPU.

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

    device = "cpu" if force_cpu else ("cuda" if cuda_compute_capable() else "cpu")
    # torch.svd_lowrank is a randomized algorithm; without seeding the top-k
    # subspace it returns drifts run-to-run (mean principal-angle cosine ≈0.65
    # on real movies), which propagates into S → vcorr_S → Cellpose channel 2
    # and shifts borderline detections.
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    # Move M_bin^T (N_pix, T_bin) to GPU; if it doesn't fit, fall back to CPU
    try:
        A = torch.from_numpy(M_bin.T).to(device)  # shape (N_pix, T_bin)
        U_t, S_t, V_t = torch.svd_lowrank(A, q=int(n_svd), niter=2)
        U = U_t.detach().cpu().numpy().astype(np.float32)       # (N_pix, n_svd)
        S = S_t.detach().cpu().numpy().astype(np.float32)       # (n_svd,)
        V = V_t.detach().cpu().numpy().astype(np.float32)       # (T_bin, n_svd)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        # GPU OOM or unavailable — fall back to CPU. Re-seed since the failed
        # GPU call already consumed RNG state.
        torch.manual_seed(0)
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
):
    """L+S background separation via binned truncated SVD.

    Algorithm (spec §3.3):
      1. Open data.bin as (T, Ly, Lx) int16 memmap.
      2. Temporally bin to ~5000 frames → (T_bin, N_pix) float32.
      3. Compute top-n_svd SVD on the binned movie.
      4. Interpolate V_bin → V_full (T, n_svd).
      5. Persist all n_svd SVD factors to svd_factors.npz.

    The residual S = M − L is **not** materialized to disk. Instead a
    :class:`~roigbiv.pipeline.residual.ResidualView` is returned that
    reconstructs any chunk on demand from ``data.bin`` + the SVD factors (the
    same arithmetic the old streaming write used, ``S_chunk = M − L``). This
    eliminates the ~10-19 GB ``residual_S.dat`` write that silently crashed the
    process (SIGBUS) on a full disk. ``mean_L`` is computed in closed form:
    ``mean_t L = US_k @ mean_t(V_k_full)``.

    Returns
    -------
    residual_view    : ResidualView reconstructing S = M − L on demand
    svd_factors_path : Path to .npz with U, S, V_bin (full n_svd components)
    k_used           : int (= cfg.k_background)
    mean_L           : (Ly, Lx) float32 — mean of L over time
    """
    from roigbiv.pipeline.residual import ResidualView

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
    U, S, V_bin = _binned_svd_gpu(M_bin, n_svd, force_cpu=cfg.force_cpu)
    print(f"  SVD top-{n_svd} on binned movie in {time.time()-t0:.1f}s", flush=True)
    del M_bin  # free ~5 GB

    # 3. Persist SVD factors — the irreplaceable substrate for on-demand
    #    residual reconstruction (data.bin is the other half). Guard the write:
    #    np.savez surfaces ENOSPC as a catchable OSError (not a memmap SIGBUS),
    #    but pre-checking keeps the failure mode uniform with the rest of the
    #    pipeline and fails before the interpolation/summary work below.
    from roigbiv.pipeline.diskguard import ensure_free_space
    svd_factors_path = output_dir / "svd_factors.npz"
    svd_nbytes = int(U.nbytes + S.nbytes + V_bin.nbytes) + 4096  # +npz/zip overhead
    ensure_free_space(svd_factors_path, svd_nbytes, label="svd_factors.npz")
    np.savez(str(svd_factors_path),
             U=U, S=S, V_bin=V_bin, bin_size=np.int32(bin_size), T=np.int32(T))

    # 4. Build the lazy residual view and the closed-form mean(L).
    k = min(int(cfg.k_background), n_svd)
    V_full = _upsample_V(V_bin, bin_size, T)
    US_k = (U[:, :k] * S[:k][np.newaxis, :]).astype(np.float32)
    V_k_full = V_full[:, :k]
    # mean_t L[t] = US_k @ mean_t(V_k_full) — exact, no streaming pass needed.
    mean_L = (US_k @ V_k_full.mean(axis=0)).reshape(Ly, Lx).astype(np.float32)

    residual_view = ResidualView.from_factors(
        data_bin_path, U, S, V_bin, bin_size, (T, Ly, Lx), k,
    )

    # Virtual-residual sidecar so resume can detect a complete foundation
    # without a dense .dat. ``kind: virtual`` signals "reconstruct, don't read".
    meta = {"shape": [int(T), int(Ly), int(Lx)], "dtype": "float32",
            "kind": "virtual", "svd_factors": "svd_factors.npz"}
    (output_dir / "residual_S.meta.json").write_text(json.dumps(meta, indent=2))

    return residual_view, svd_factors_path, k, mean_L


# ─────────────────────────────────────────────────────────────────────────
# Summary images on S
# ─────────────────────────────────────────────────────────────────────────

def _iter_S_chunks(residual_view, chunk: int = 500):
    """Generator yielding (t0, t1, S_chunk) — S_chunk is (cs, Ly, Lx) float32."""
    yield from residual_view.iter_chunks(chunk)


# Undirected edge families whose symmetric contributions reproduce the directed
# 4-/8-neighbor averages while computing each pixel-pair correlation once.
_VCORR_EDGES_8 = [(0, 1), (1, 0), (1, 1), (1, -1)]
_VCORR_EDGES_4 = [(0, 1), (1, 0)]


def _accumulate_summaries(chunk_iter, Ly: int, Lx: int, *, neighbors: int = 8) -> dict:
    """Single-pass mean/max/std/Vcorr accumulator over a chunk iterator.

    ``chunk_iter`` yields ``(t0, t1, chunk_arr)`` where ``chunk_arr`` is a
    ``(cs, Ly, Lx)`` float array (cs = frames in this chunk; may be < t1-t0 when
    the source iterator decimates). The arithmetic is identical whether the
    chunks come from a reconstructed residual view (production) or a raw
    ``data.bin`` reader (scout) — only the source differs.

    Returns dict with keys 'mean', 'max', 'std', 'vcorr', each (Ly, Lx) float32.
    """
    if neighbors == 4:
        edges = _VCORR_EDGES_4
    elif neighbors == 8:
        edges = _VCORR_EDGES_8
    else:
        raise ValueError(f"neighbors must be 4 or 8 (got {neighbors})")

    # First pass: running sum, sum-of-squares, max
    sum_ = np.zeros((Ly, Lx), dtype=np.float64)
    sumsq = np.zeros((Ly, Lx), dtype=np.float64)
    max_ = np.full((Ly, Lx), -np.inf, dtype=np.float32)

    # Per-edge cross-products. The other Pearson moments are shifted views
    # into the global sum_/sumsq maps below. Each undirected edge contributes
    # to both endpoint pixels during finalization, reproducing the directed
    # neighbor average while computing each pair once.
    sxy_by_edge = {
        (dy, dx): np.zeros((Ly, Lx), dtype=np.float64)
        for (dy, dx) in edges
    }

    t_total = 0
    for t0, t1, chunk_arr in chunk_iter:
        cs = int(chunk_arr.shape[0])
        if cs == 0:
            continue
        t_total += cs

        # Single float64 cast per chunk — shared by sumsq and all Vcorr offsets
        # below. Slicing chunk64 below yields views (no additional copy).
        chunk64 = chunk_arr.astype(np.float64)

        # mean / std / max accumulators (straightforward over full FOV).
        # chunk_arr may be a memmap view — go through chunk64 (in-RAM) so
        # `.max()` doesn't re-scan the residual from disk on every chunk.
        sum_ += chunk64.sum(axis=0)
        sumsq += (chunk64 ** 2).sum(axis=0)
        np.maximum(max_, chunk64.max(axis=0).astype(np.float32), out=max_)

        # Vcorr accumulators — for each undirected edge, accumulate the
        # cross-product over valid (non-boundary) endpoint pairs.
        for (dy, dx) in edges:
            py0 = max(0, -dy); py1 = Ly - max(0, dy)
            px0 = max(0, -dx); px1 = Lx - max(0, dx)
            qy0 = py0 + dy; qy1 = py1 + dy
            qx0 = px0 + dx; qx1 = px1 + dx

            p_chunk = chunk64[:, py0:py1, px0:px1]
            q_chunk = chunk64[:, qy0:qy1, qx0:qx1]

            sxy_by_edge[(dy, dx)][py0:py1, px0:px1] += (
                p_chunk * q_chunk
            ).sum(axis=0)

    mean = (sum_ / t_total).astype(np.float32)
    var = (sumsq / t_total) - (sum_ / t_total) ** 2
    var = np.maximum(var, 0.0)  # guard numerical negatives
    std = np.sqrt(var).astype(np.float32)

    # Vcorr: average of 8 neighbor correlations per pixel, only counting neighbors
    # that exist (boundary pixels get averaged over fewer neighbors).
    vcorr = np.zeros((Ly, Lx), dtype=np.float64)
    count = np.zeros((Ly, Lx), dtype=np.int32)
    eps = 1e-12
    for (dy, dx), sxy in sxy_by_edge.items():
        py0 = max(0, -dy); py1 = Ly - max(0, dy)
        px0 = max(0, -dx); px1 = Lx - max(0, dx)
        qy0 = py0 + dy; qy1 = py1 + dy
        qx0 = px0 + dx; qx1 = px1 + dx

        sp = sum_[py0:py1, px0:px1]
        sq = sum_[qy0:qy1, qx0:qx1]
        spp = sumsq[py0:py1, px0:px1]
        sqq = sumsq[qy0:qy1, qx0:qx1]
        sxy_region = sxy[py0:py1, px0:px1]

        num = t_total * sxy_region - sp * sq
        den = np.sqrt(np.maximum(t_total * spp - sp ** 2, 0.0) *
                       np.maximum(t_total * sqq - sq ** 2, 0.0))
        r = np.where(den > eps, num / (den + eps), 0.0)
        vcorr[py0:py1, px0:px1] += r
        vcorr[qy0:qy1, qx0:qx1] += r
        count[py0:py1, px0:px1] += 1
        count[qy0:qy1, qx0:qx1] += 1

    vcorr = (vcorr / np.maximum(count, 1)).astype(np.float32)

    return {"mean": mean, "max": max_.astype(np.float32),
            "std": std, "vcorr": vcorr}


def generate_summary_images(
    residual_view,
    chunk: int = 500,
) -> dict:
    """Compute mean, max, std, and 8-neighbor Vcorr projections of residual S.

    All accumulators run in a single pass through the residual view (each chunk
    reconstructed on demand). Memory per projection is ~1 MB (H,W float64/32).
    Vcorr needs 5 accumulators × 8 neighbors ≈ 40 MB.

    Returns dict with keys 'mean', 'max', 'std', 'vcorr', each a (Ly, Lx) float32.
    """
    _, Ly, Lx = residual_view.shape
    return _accumulate_summaries(
        _iter_S_chunks(residual_view, chunk), Ly, Lx, neighbors=8,
    )


def vcorr_on_movie(
    data_bin_path: Path,
    Ly: int,
    Lx: int,
    T: int,
    *,
    stride: int = 1,
    neighbors: int = 8,
    chunk: int = 128,
) -> dict:
    """Scout-mode summaries: mean/Vcorr on the *registered movie* (no residual).

    Streams ``data.bin`` directly — no SVD, no L+S, no per-chunk low-rank
    reconstruction matmul. Channel 2 for Cellpose becomes a correlation map on
    the raw registered movie rather than the background-subtracted residual,
    which is adequate for FOV-clarity triage and model A/B testing but is *not*
    a substitute for a production run (cells over bright background are less
    crisp).

    Parameters
    ----------
    stride    : frame decimation (1 = every frame). Vcorr is stable under mild
                decimation; >1 trades a little contrast for speed.
    neighbors : 8 (full) or 4 (von Neumann, ~half the arithmetic).
    chunk     : frames read per ``data.bin`` slice before decimation.

    Returns dict with keys 'mean', 'max', 'std', 'vcorr', each (Ly, Lx) float32.
    The 'mean' falls out of the same pass, so it doubles as ``mean_M``.
    """
    movie = _open_data_bin(data_bin_path, Ly, Lx)
    stride = max(1, int(stride))

    def _iter():
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            sub = movie[t0:t1]
            if stride > 1:
                sub = sub[::stride]
            yield t0, t1, np.asarray(sub, dtype=np.float32)

    return _accumulate_summaries(_iter(), Ly, Lx, neighbors=neighbors)


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


def _build_branches_manifest(branches: list, summary_dir: Path) -> list:
    """JSON-safe manifest records for ``branches.json``, referencing on-disk summary TIFFs.

    Filters out any ``summary_images`` entries that are ``None`` — a BranchView's
    array set is only a subset of the standard names for some sources (e.g. a
    future denoised branch may not produce all seven).
    """
    manifest = []
    for branch in branches:
        summary_image_paths = {
            name: str(summary_dir / f"{name}.tif")
            for name, arr in branch.summary_images.items()
            if arr is not None
        }
        manifest.append({
            "branch_name": branch.branch_name,
            "is_denoised": branch.is_denoised,
            "provenance": branch.provenance,
            "summary_image_paths": summary_image_paths,
        })
    return manifest


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────

def _run_foundation_scout(
    tif_path: Path,
    cfg: PipelineConfig,
    output_dir: Path,
    ops: dict,
    data_bin_path: Path,
    motion_x: np.ndarray,
    motion_y: np.ndarray,
    T: int,
    Ly: int,
    Lx: int,
) -> FOVData:
    """Fast scout Foundation — Cellpose-only triage path (see ``vcorr_on_movie``).

    Skips SVD / L+S / residual reconstruction entirely. A single ``data.bin``
    pass yields ``mean_M`` (Cellpose channel 1) and ``vcorr`` on the registered
    movie (channel 2). No residual view, no ``svd_factors.npz``: a scout FOV is
    *not* resumable into a full run.
    """
    print(fmt.stage_header("F", "SCOUT: Vcorr on registered movie "
                           f"(stride={cfg.scout_vcorr_stride}, "
                           f"neighbors={cfg.scout_vcorr_neighbors}) — "
                           "no SVD/L+S"), flush=True)
    print(fmt.sub_phase(
        "scout mode: channel 2 is correlation on the raw registered movie, "
        "not the background-subtracted residual. Masks are for FOV/model "
        "triage only — re-run without scout for analysis-grade output."
    ), flush=True)
    t0 = time.time()
    summaries = vcorr_on_movie(
        data_bin_path, Ly, Lx, T,
        stride=cfg.scout_vcorr_stride,
        neighbors=cfg.scout_vcorr_neighbors,
    )
    mean_M = summaries["mean"]      # raw registered-movie mean = morphological channel
    vcorr_S = summaries["vcorr"]    # correlation on M; reuses the vcorr_S field/contract
    print(fmt.sub_phase("scout summary images", time.time() - t0), flush=True)

    print(fmt.sub_phase("DoG (nuclear shadow) map on mean_M"), flush=True)
    dog_map = compute_nuclear_shadow_map(mean_M)

    # Write only the channels scout produces.
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    for name, arr in [("mean_M", mean_M), ("vcorr_S", vcorr_S),
                      ("dog_map", dog_map)]:
        tifffile.imwrite(str(summary_dir / f"{name}.tif"), arr.astype(np.float32))

    ops_snapshot = {k: v for k, v in ops.items()
                    if isinstance(v, (int, float, str, bool, list, tuple))}

    # mean_S kept as a zeros stand-in so the (Ly, Lx) shape contract holds for
    # any downstream `.shape` reference; max_S/std_S/mean_L/residual_view absent.
    return FOVData(
        raw_path=tif_path,
        output_dir=output_dir,
        data_bin_path=data_bin_path,
        shape=(T, Ly, Lx),
        residual_view=None,
        mean_M=mean_M,
        mean_S=np.zeros_like(mean_M),
        max_S=None,
        std_S=None,
        vcorr_S=vcorr_S,
        dog_map=dog_map,
        mean_L=None,
        svd_factors_path=None,
        motion_x=motion_x,
        motion_y=motion_y,
        k_background=0,
        rois=[],
        stage_counts={},
        ops=ops_snapshot,
    )


def run_foundation(
    tif_path: Path,
    cfg: PipelineConfig,
    output_dir: Path,
    gpu_lock=None,
) -> FOVData:
    """Run Foundation: motion correction + L+S + summary images + DoG.

    Writes to {output_dir}:
      suite2p/plane0/{ops.npy, data.bin, ...}
      svd_factors.npz, residual_S.meta.json (virtual — no dense .dat), motion_trace.npz
      summary/{mean_S,max_S,std_S,vcorr_S,mean_L,dog_map}.tif

    Returns a populated FOVData with summary images in RAM and a lazy
    ResidualView (reconstructs S = M − L on demand from data.bin + SVD factors).

    Raises:
        ValueError: if cfg's denoising fields are misconfigured (unknown
            denoiser_backend, enable_denoised_branch=True with backend='none',
            or a real backend selected without denoiser_model_path).
    """
    tif_path = Path(tif_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary").mkdir(exist_ok=True)

    # Validate denoising configuration (issue #34's generic denoiser_backend
    # config surface — inert/documentary; issue #37's deepcad_denoise below is
    # the only backend currently wired to execution).
    backend = getattr(cfg, "denoiser_backend", "none")
    if backend not in ("deepcad_rt", "deepinterpolation", "pmd", "none"):
        raise ValueError(
            f"Unknown denoiser_backend {backend!r}; expected 'deepcad_rt', "
            f"'deepinterpolation', 'pmd', or 'none'."
        )

    enable_denoised = getattr(cfg, "enable_denoised_branch", False)
    if enable_denoised and backend == "none":
        raise ValueError(
            "enable_denoised_branch=True requires a denoiser_backend other "
            "than 'none'."
        )

    # Model path is required whenever a real backend is selected, independent of
    # enable_denoised_branch; this catches incomplete configs early.
    model_path = getattr(cfg, "denoiser_model_path", None)
    if backend != "none" and model_path is None:
        raise ValueError(
            f"denoiser_backend {backend!r} requires denoiser_model_path to "
            f"be set."
        )

    # DeepCAD-RT out-of-process denoising (opt-in). Runs on the RAW input
    # movie, before motion correction. Skipped in scout_mode: scout is a fast
    # FOV/model triage path, and running an expensive out-of-process denoise
    # just to discard the result (scout's FOVData does not carry it) would be
    # wasted work — re-run without --scout for the full denoised path.
    denoised_path = None
    if getattr(cfg, "deepcad_denoise", False) and not getattr(cfg, "scout_mode", False):
        from roigbiv.pipeline.deepcad import run_deepcad_denoise
        denoised_path = run_deepcad_denoise(tif_path, output_dir, cfg, gpu_lock=gpu_lock)

    # Header reflects the actual mode: a pre-corrected input runs Suite2p in
    # detection-only mode (cfg.do_registration is False across all three backends),
    # so don't imply registration happened when it didn't.
    print(fmt.stage_header("F", _mc_stage_label(cfg.do_registration)), flush=True)
    ops, data_bin_path, motion_x, motion_y = run_motion_correction(
        tif_path, cfg, output_dir, gpu_lock=gpu_lock)
    Ly = int(ops["Ly"]); Lx = int(ops["Lx"])
    # Determine T from data.bin size (more reliable than ops fields across Suite2p versions)
    T = Path(data_bin_path).stat().st_size // (Ly * Lx * 2)
    print(fmt.sub_phase(
        f"ops: T={T}, Ly={Ly}, Lx={Lx}  (fs={cfg.fs} tau={cfg.tau} "
        f"registration={'ON' if cfg.do_registration else 'OFF'})"
    ), flush=True)

    # Persist motion traces (spec §3.1 Blindspot 9 — for future Gate 4)
    np.savez(str(output_dir / "motion_trace.npz"),
             xoff=motion_x, yoff=motion_y, fs=np.float32(cfg.fs))

    if getattr(cfg, "scout_mode", False):
        return _run_foundation_scout(
            tif_path, cfg, output_dir, ops, data_bin_path,
            motion_x, motion_y, T, Ly, Lx,
        )

    print(fmt.stage_header("F", f"L+S background separation (k={cfg.k_background}, n_svd={cfg.n_svd})"),
          flush=True)
    residual_view, svd_factors_path, k_used, mean_L = compute_background_separation(
        data_bin_path, ops, cfg, output_dir,
    )

    print(fmt.stage_header("F", "Summary images (mean, max, std, vcorr) on S"), flush=True)
    t0 = time.time()
    # Cap at 128 frames/chunk regardless of reconstruct_chunk. Each chunk
    # allocates `chunk64` (cs·Ly·Lx·8 B) plus transient `(chunk64 ** 2)` /
    # per-offset temporaries of the same size; with cs=500 on a 505×493 FOV
    # that's ~2 GB peak, which swaps on RAM-constrained hosts and stalls for
    # >10 min. cs=128 caps peak at ~500 MB — comfortable on 16 GB systems
    # and still a single reconstruction pass through the residual view.
    summary_chunk = min(128, int(cfg.reconstruct_chunk))
    summaries = generate_summary_images(residual_view, chunk=summary_chunk)
    mean_S = summaries["mean"]
    max_S = summaries["max"]
    std_S = summaries["std"]
    vcorr_S = summaries["vcorr"]
    print(fmt.sub_phase("summary images", time.time() - t0), flush=True)

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

    print(fmt.sub_phase("DoG (nuclear shadow) map on mean_M"), flush=True)
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

    summary_images = {
        "mean_M": mean_M, "mean_S": mean_S, "max_S": max_S, "std_S": std_S,
        "vcorr_S": vcorr_S, "mean_L": mean_L, "dog_map": dog_map,
    }
    branch_provenance = {
        "motion_correction_backend": cfg.motion_correction_backend,
        "k_used": int(k_used),
        "fs": cfg.fs,
        "tau": cfg.tau,
        "n_svd": cfg.n_svd,
    }

    branch_view = BranchView(
        branch_name="raw",
        movie_view=data_bin_path,
        summary_images=summary_images,
        provenance=branch_provenance,
        is_denoised=False,
    )

    branches_manifest = _build_branches_manifest([branch_view], summary_dir)
    (output_dir / "branches.json").write_text(json.dumps(branches_manifest, indent=2))

    return FOVData(
        raw_path=tif_path,
        output_dir=output_dir,
        data_bin_path=data_bin_path,
        shape=(T, Ly, Lx),
        residual_view=residual_view,
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
        branches=[branch_view],
        denoised_path=denoised_path,
    )
