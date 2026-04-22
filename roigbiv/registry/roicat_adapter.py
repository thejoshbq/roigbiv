"""Thin adapter between roigbiv and ROICaT's tracking pipeline.

This is the *sole* import site for ``roicat`` inside the roigbiv registry.
Everything ROICaT-specific (Aligner, Blurrer, ROInet, SWT, ROI_graph,
Clusterer) is encapsulated here and returned as plain numpy arrays + plain
dataclasses. Callers in ``roigbiv.registry`` must import nothing from
``roicat`` directly — go through this module.

Public API
----------
:class:`SessionInput`       — one session's inputs (mean projection + label mask).
:class:`ClusterResult`      — ROICaT's clustering output, normalised to roigbiv shapes.
:class:`AdapterConfig`      — tunable knobs (um_per_pixel, device, alignment method, ...).
:func:`load_session_input`  — read merged_masks.tif + summary/mean_M.tif from a
                              roigbiv pipeline output directory.
:func:`footprints_from_merged_masks` — convert a uint16 label image to a sparse
                              per-ROI footprint matrix.
:func:`cluster_sessions`    — run ROICaT's align → blur → embed → SWT → similarity
                              → cluster sequence on a list of SessionInputs.

Alignment method default
------------------------
``RoMa`` is the default — validated on the T1 three-session dataset as the
only method that produces correct cross-session FOV merges (PhaseCorrelation
mis-merged at posterior 0.33 for a true same-FOV pair). First use triggers
~1.5 GB of weight downloads (`roma_outdoor.pth` 425 MB + `dinov2_vitl14` 1.1 GB)
cached under ``~/.cache/torch/hub/checkpoints/``. CPU inference is impractically
slow (minutes per session pair); the adapter auto-detects CUDA when available
and falls back to CPU only if not. ``PhaseCorrelation`` remains selectable via
:class:`AdapterConfig.alignment_method` or ``ROIGBIV_ROICAT_ALIGNMENT``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.ndimage import center_of_mass

log = logging.getLogger(__name__)


def _auto_device() -> str:
    """Pick a torch device string: ``cuda`` when available, else ``cpu``.

    Kept as a small free function (rather than inlined) so the adapter config
    dataclass can use it via ``field(default_factory=_auto_device)`` without
    importing torch at module-import time.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── RoMa native-fallback shim ──────────────────────────────────────────────
# ROICaT's Aligner doesn't expose `use_custom_corr` on its RoMa wrapper, so
# when the compiled `local_corr` CUDA extension isn't installed, the RoMa
# path crashes with ModuleNotFoundError partway through inference. Patch
# romatch.roma_{outdoor,indoor} + tiny_roma_v1_outdoor to force
# use_custom_corr=False — the PyTorch-native correlation path is ~2–3× slower
# but numerically equivalent for inference.
_roma_patched: bool = False


def _maybe_patch_romatch_for_native_fallback() -> None:
    global _roma_patched
    if _roma_patched:
        return
    try:
        import local_corr  # noqa: F401
        _roma_patched = True  # kernel present; no patch needed.
        return
    except ImportError:
        pass

    try:
        import romatch
        from romatch.models.model_zoo import roma_models as _rmz
    except ImportError:
        return  # romatch not installed; nothing to patch.

    log.warning(
        "RoMa: 'local_corr' CUDA extension not found — patching "
        "romatch.roma_{outdoor,indoor,tiny_v1_outdoor} to use the native "
        "PyTorch correlation fallback (~2-3x slower, numerically equivalent)."
    )

    for fn_name in ("roma_outdoor", "roma_indoor", "tiny_roma_v1_outdoor"):
        orig = getattr(romatch, fn_name, None) or getattr(_rmz, fn_name, None)
        if orig is None:
            continue

        def _wrap(orig_fn):
            def _patched(*args, **kwargs):
                kwargs["use_custom_corr"] = False
                return orig_fn(*args, **kwargs)
            _patched.__wrapped__ = orig_fn
            _patched.__name__ = getattr(orig_fn, "__name__", "patched")
            return _patched

        patched = _wrap(orig)
        if hasattr(romatch, fn_name):
            setattr(romatch, fn_name, patched)
        if hasattr(_rmz, fn_name):
            setattr(_rmz, fn_name, patched)

    _roma_patched = True


# ── Public data classes ────────────────────────────────────────────────────


@dataclass
class SessionInput:
    """Inputs for one session of the ROICaT tracking pipeline.

    Parameters
    ----------
    session_key : str
        Stable identifier for this session (usually the output directory
        basename). Used to key back into roigbiv registry state.
    mean_m : np.ndarray
        ``(H, W)`` float32 mean projection of the motion-corrected movie.
    merged_masks : np.ndarray
        ``(H, W)`` uint16 label image. Pixel value = ROI ``label_id``;
        0 = background. Matches the pipeline's ``merged_masks.tif`` contract.
    """

    session_key: str
    mean_m: np.ndarray
    merged_masks: np.ndarray


@dataclass
class AdapterConfig:
    """Tunable knobs for :func:`cluster_sessions`.

    Defaults are chosen for small session counts (N ≤ 6) on CPU without
    requiring any weight downloads beyond the 75 MB ROInet bundle.
    """

    um_per_pixel: float = 1.0
    device: str = field(default_factory=_auto_device)
    all_to_all: bool = False
    nonrigid: bool = False
    alignment_method: str = "RoMa"
    sequential_hungarian_thresh_cost: float = 0.6
    # Weight of the ROI-footprint density image in Aligner.augment_FOV_images.
    # Left at ROICaT's default. Exposed as a knob (ROIGBIV_ROICAT_ROI_MIXING)
    # for experimentation — 0.9 was tested on the T1 three-session dataset
    # under PhaseCorrelation and regressed inlier rate (mean projection has
    # more distinctive spatial content than uniform footprint density for
    # Fourier-domain methods). Not re-tested under RoMa; default preserved.
    roi_mixing_factor: float = 0.5
    # d_cutoff for Clusterer.make_pruned_similarity_graphs. None = let ROICaT
    # infer from the same/diff distribution crossover (the usual path on
    # real-sized FOVs with hundreds of ROIs). Pass a float to bypass the
    # inference — useful for very small inputs where crossover detection fails.
    d_cutoff: Optional[float] = None
    roinet_cache_dir: Optional[Path] = None
    roinet_batch_size: int = 8
    verbose: bool = False


@dataclass
class ClusterResult:
    """Normalised output of :func:`cluster_sessions`.

    Attributes
    ----------
    labels : np.ndarray
        ``(n_rois_total,)`` int32. ROICaT's cluster assignment, concatenated
        across sessions in input order. ``-1`` = unclustered.
    session_bool : np.ndarray
        ``(n_rois_total, n_sessions)`` bool. ``session_bool[i, j]`` is True
        iff ROI ``i`` came from session ``j``.
    per_session_label_ids : List[np.ndarray]
        Per-session array of ``label_id`` values (pulled from each session's
        ``merged_masks``). Index ``k`` within the per-session array aligns
        with the corresponding slice of :attr:`labels`.
    per_session_roi_count : List[int]
        ROI count per session, in input order.
    quality_metrics : dict
        Output of ``Clusterer.compute_quality_metrics()``.
    alignment_method : str
        Which :class:`roicat.tracking.alignment.Aligner` method was used.
    alignment_inlier_rate : float
        In ``[0, 1]``. For deep-learning alignment methods this is the
        ROICaT-reported RANSAC inlier rate when available; for the
        PhaseCorrelation / ECC defaults we use post-warp Pearson correlation
        against the template, clipped to ``[0, 1]``.
    fov_height : int
    fov_width : int
        The (possibly padded) working frame size. All sessions are padded to
        this shape before alignment.
    """

    labels: np.ndarray
    session_bool: np.ndarray
    per_session_label_ids: List[np.ndarray]
    per_session_roi_count: List[int]
    quality_metrics: dict
    alignment_method: str
    alignment_inlier_rate: float
    fov_height: int
    fov_width: int


# ── Public helpers ─────────────────────────────────────────────────────────


def load_session_input(
    output_dir: Path, *, session_key: Optional[str] = None
) -> SessionInput:
    """Read a roigbiv pipeline output directory into a :class:`SessionInput`.

    Expects ``{output_dir}/merged_masks.tif`` and ``{output_dir}/summary/mean_M.tif``.
    """
    output_dir = Path(output_dir)
    merged_masks_path = output_dir / "merged_masks.tif"
    mean_m_path = output_dir / "summary" / "mean_M.tif"
    if not merged_masks_path.exists():
        raise FileNotFoundError(f"missing merged_masks.tif in {output_dir}")
    if not mean_m_path.exists():
        raise FileNotFoundError(f"missing summary/mean_M.tif in {output_dir}")

    import tifffile  # local import — keeps module import cheap

    merged_masks = np.asarray(tifffile.imread(str(merged_masks_path)))
    if merged_masks.ndim != 2:
        raise ValueError(
            f"merged_masks.tif must be 2-D, got shape {merged_masks.shape}"
        )
    if merged_masks.dtype != np.uint16:
        merged_masks = merged_masks.astype(np.uint16)

    mean_m = np.asarray(tifffile.imread(str(mean_m_path)), dtype=np.float32)
    if mean_m.ndim != 2:
        raise ValueError(f"mean_M.tif must be 2-D, got shape {mean_m.shape}")

    return SessionInput(
        session_key=session_key or output_dir.name,
        mean_m=mean_m,
        merged_masks=merged_masks,
    )


def footprints_from_merged_masks(
    merged_masks: np.ndarray,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Convert a uint16 label image to ROICaT's per-session footprint format.

    Returns
    -------
    footprints : scipy.sparse.csr_matrix
        ``(n_rois, H * W)`` float32 binary (0.0 / 1.0). Row ``k`` is the
        flattened mask of the ``k``-th ROI, with ROIs ordered by ascending
        ``label_id``. Flattening uses C order (row-major) to match ROICaT's
        convention. Float32 rather than uint8 because ROICaT internally
        reshapes the matrix to ``(n_rois, H, W)`` and calls ``.sum(axis=W)``,
        whose intermediate result would overflow an 8-bit fill value.
    label_ids : np.ndarray
        ``(n_rois,)`` int64. The original ``label_id`` values in the same
        order as the rows of ``footprints``.
    """
    merged_masks = np.asarray(merged_masks)
    if merged_masks.ndim != 2:
        raise ValueError(f"merged_masks must be 2-D, got {merged_masks.shape}")
    label_ids = np.unique(merged_masks)
    label_ids = label_ids[label_ids != 0].astype(np.int64)
    if label_ids.size == 0:
        H, W = merged_masks.shape
        return sparse.csr_matrix((0, H * W), dtype=np.float32), label_ids

    H, W = merged_masks.shape
    flat = merged_masks.reshape(-1)
    # Build CSR directly: for each ROI, collect pixel indices where flat == label.
    rows: list[int] = []
    cols: list[int] = []
    for row_idx, lbl in enumerate(label_ids.tolist()):
        pixel_idx = np.flatnonzero(flat == lbl)
        rows.append(pixel_idx.size)
        cols.append(pixel_idx)
    indptr = np.empty(len(label_ids) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(rows, out=indptr[1:])
    indices = np.concatenate(cols).astype(np.int64) if cols else np.zeros(0, np.int64)
    data = np.ones(indices.size, dtype=np.float32)
    footprints = sparse.csr_matrix((data, indices, indptr), shape=(len(label_ids), H * W))
    return footprints, label_ids


def cluster_sessions(
    sessions: List[SessionInput], config: Optional[AdapterConfig] = None
) -> ClusterResult:
    """Run ROICaT's tracking pipeline on a list of sessions.

    For ``len(sessions) == 1`` the clustering step is skipped and a degenerate
    :class:`ClusterResult` is returned where every ROI is its own singleton
    cluster. This is the "new FOV, no candidates" case — callers still receive
    a uniform shape.

    For ``len(sessions) >= 2`` the pipeline is:
    Aligner (geometric) → ROI_Blurrer → ROInet → SWT → ROI_graph → Clusterer
    (``fit_sequentialHungarian`` per ROICaT's guidance for ≤ 8 sessions).
    """
    cfg = config or AdapterConfig()
    if not sessions:
        raise ValueError("cluster_sessions requires at least one SessionInput")

    # Normalise all sessions to a common (H, W) by zero-padding the smaller ones.
    target_h = max(s.mean_m.shape[0] for s in sessions)
    target_w = max(s.mean_m.shape[1] for s in sessions)

    per_session_footprints: List[sparse.csr_matrix] = []
    per_session_label_ids: List[np.ndarray] = []
    per_session_fov_images: List[np.ndarray] = []
    per_session_roi_count: List[int] = []

    for s in sessions:
        mean_m = _pad_to(s.mean_m.astype(np.float32), target_h, target_w)
        merged = _pad_to(s.merged_masks, target_h, target_w).astype(np.uint16)
        fps, lbls = footprints_from_merged_masks(merged)
        per_session_footprints.append(fps)
        per_session_label_ids.append(lbls)
        per_session_fov_images.append(mean_m)
        per_session_roi_count.append(int(fps.shape[0]))

    total_rois = sum(per_session_roi_count)
    n_sessions = len(sessions)
    session_bool = _make_session_bool(per_session_roi_count)

    # Degenerate single-session case: no alignment / clustering.
    if n_sessions == 1 or total_rois == 0:
        labels = np.arange(total_rois, dtype=np.int32)
        return ClusterResult(
            labels=labels,
            session_bool=session_bool,
            per_session_label_ids=per_session_label_ids,
            per_session_roi_count=per_session_roi_count,
            quality_metrics={},
            alignment_method="none",
            alignment_inlier_rate=1.0,
            fov_height=target_h,
            fov_width=target_w,
        )

    # Lazy import — ROICaT + torch are only needed when we actually cluster.
    from roicat import ROInet  # noqa: F401
    from roicat.tracking import alignment, blurring, clustering, similarity_graph
    from roicat.tracking import scatteringWaveletTransformer as swt_mod
    from roicat.data_importing import Data_roicat

    verbose = bool(cfg.verbose)

    data = Data_roicat(verbose=verbose)
    data.set_FOVHeightWidth(FOV_height=target_h, FOV_width=target_w)
    data.set_FOV_images(FOV_images=per_session_fov_images)
    data.set_spatialFootprints(
        spatialFootprints=per_session_footprints,
        um_per_pixel=cfg.um_per_pixel,
    )
    data.transform_spatialFootprints_to_ROIImages()

    # Alignment. If RoMa is selected, ensure the native-torch correlation
    # fallback is in place (no-op when the CUDA extension is installed).
    if "RoMa" in cfg.alignment_method or "roma" in cfg.alignment_method:
        _maybe_patch_romatch_for_native_fallback()
    aligner = alignment.Aligner(
        use_match_search=True,
        all_to_all=cfg.all_to_all,
        um_per_pixel=cfg.um_per_pixel,
        device=cfg.device,
        verbose=verbose,
    )
    augmented = aligner.augment_FOV_images(
        FOV_images=data.FOV_images,
        spatialFootprints=data.spatialFootprints,
        roi_FOV_mixing_factor=float(cfg.roi_mixing_factor),
    )
    aligner.fit_geometric(
        template=0.5,
        ims_moving=augmented,
        template_method="image",
        method=cfg.alignment_method,
        constraint="affine",
        verbose=verbose,
    )
    aligner.transform_images_geometric(augmented)
    if cfg.nonrigid:
        aligner.fit_nonrigid(
            template=0.5,
            ims_moving=aligner.ims_registered_geo,
            remappingIdx_init=aligner.remappingIdx_geo,
            template_method="image",
            method=cfg.alignment_method,
        )
        remapping = aligner.remappingIdx_nonrigid
    else:
        remapping = aligner.remappingIdx_geo
    aligner.transform_ROIs(ROIs=data.spatialFootprints, remappingIdx=remapping)

    # Post-alignment quality: mean Pearson correlation of aligned images against template.
    alignment_inlier_rate = _pearson_to_template(
        getattr(aligner, "ims_registered_geo", augmented),
        template_idx=_resolve_template_index(n_sessions),
    )

    # Blur.
    blurrer = blurring.ROI_Blurrer(
        frame_shape=(target_h, target_w),
        kernel_halfWidth=2,
        plot_kernel=False,
        verbose=verbose,
    )
    blurrer.blur_ROIs(spatialFootprints=list(aligner.ROIs_aligned))

    # ROInet embedding.
    roinet_cache = cfg.roinet_cache_dir or _default_roinet_cache()
    roinet_cache.mkdir(parents=True, exist_ok=True)
    roinet = ROInet.ROInet_embedder(
        dir_networkFiles=str(roinet_cache),
        device=cfg.device,
        download_method="check_local_first",
        download_url="https://osf.io/x3fd2/download",
        download_hash="7a5fb8ad94b110037785a46b9463ea94",
        forward_pass_version="latent",
        verbose=verbose,
    )
    roinet.generate_dataloader(
        ROI_images=data.ROI_images,
        um_per_pixel=data.um_per_pixel,
        pref_plot=False,
        batchSize_dataloader=cfg.roinet_batch_size,
        pinMemory_dataloader=False,
        numWorkers_dataloader=0,
        persistentWorkers_dataloader=False,
        prefetchFactor_dataloader=None,
    )
    roinet.generate_latents()

    # Scattering wavelet transform.
    swt_ob = swt_mod.SWT(
        image_shape=data.ROI_images[0].shape[1:3],
        device=cfg.device,
        verbose=verbose,
    )
    swt_ob.transform(ROI_images=roinet.ROI_images_rs, batch_size=100)

    # Similarity graph.
    sim = similarity_graph.ROI_graph(
        n_workers=1,
        frame_height=target_h,
        frame_width=target_w,
        block_height=min(128, target_h),
        block_width=min(128, target_w),
        algorithm_nearestNeigbors_spatialFootprints="brute",
        verbose=verbose,
    )
    sim.compute_similarity_blockwise(
        spatialFootprints=blurrer.ROIs_blurred,
        features_NN=roinet.latents,
        features_SWT=swt_ob.latents,
        ROI_session_bool=data.session_bool,
        spatialFootprint_maskPower=1.0,
    )
    # k_max / k_min define the sample range for the centroid-distance kNN graph
    # that ROICaT uses to fit its null distribution. Internally ROICaT caps them
    # at (n_rois - 1), so on very small inputs (unit tests, tiny real FOVs) they
    # can collapse to the same value and trip an assertion. Pre-clamp.
    k_max_target = n_sessions * 100
    k_min_target = n_sessions * 10
    k_max = max(2, min(k_max_target, total_rois - 1))
    k_min = max(1, min(k_min_target, k_max - 1))
    sim.make_normalized_similarities(
        centers_of_mass=data.centroids,
        features_NN=roinet.latents,
        features_SWT=swt_ob.latents,
        k_max=k_max,
        k_min=k_min,
        algo_NN="kd_tree",
        device=cfg.device,
        verbose=verbose,
    )

    # Clustering.
    clusterer = clustering.Clusterer(
        s_sf=sim.s_sf,
        s_NN_z=sim.s_NN_z,
        s_SWT_z=sim.s_SWT_z,
        s_sesh=sim.s_sesh,
        verbose=verbose,
    )
    manual_mixing = dict(
        power_SF=1.0,
        power_NN=0.5,
        power_SWT=0.5,
        p_norm=-1.0,
        sig_SF_kwargs=None,
        sig_NN_kwargs={"mu": 0.5, "b": 1.0},
        sig_SWT_kwargs={"mu": 0.5, "b": 1.0},
    )
    if cfg.d_cutoff is not None:
        # Workaround: ROICaT 1.5.5's Clusterer.make_pruned_similarity_graphs
        # reads ``self.d_cutoff`` unconditionally but only writes it from the
        # crossover-inference branch (when d_cutoff kwarg is None). Set it
        # directly so a user-provided cutoff is honoured.
        clusterer.d_cutoff = float(cfg.d_cutoff)
    clusterer.make_pruned_similarity_graphs(
        kwargs_makeConjunctiveDistanceMatrix=manual_mixing,
        stringency=1.0,
        convert_to_probability=False,
        d_cutoff=cfg.d_cutoff,
    )
    labels = clusterer.fit_sequentialHungarian(
        d_conj=clusterer.dConj_pruned,
        session_bool=data.session_bool,
        thresh_cost=cfg.sequential_hungarian_thresh_cost,
    )
    try:
        quality_metrics = clusterer.compute_quality_metrics()
    except Exception as exc:  # pragma: no cover - informational only
        log.warning("Clusterer.compute_quality_metrics failed: %s", exc)
        quality_metrics = {}

    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if labels.shape[0] != total_rois:
        raise RuntimeError(
            f"ROICaT label count {labels.shape[0]} != expected {total_rois}"
        )

    return ClusterResult(
        labels=labels,
        session_bool=session_bool,
        per_session_label_ids=per_session_label_ids,
        per_session_roi_count=per_session_roi_count,
        quality_metrics=dict(quality_metrics) if quality_metrics else {},
        alignment_method=cfg.alignment_method,
        alignment_inlier_rate=float(alignment_inlier_rate),
        fov_height=target_h,
        fov_width=target_w,
    )


# ── Internal helpers ───────────────────────────────────────────────────────


def _pad_to(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    H, W = image.shape
    if H == target_h and W == target_w:
        return image
    if H > target_h or W > target_w:
        raise ValueError(
            f"image shape {image.shape} exceeds target ({target_h}, {target_w})"
        )
    out = np.zeros((target_h, target_w), dtype=image.dtype)
    out[:H, :W] = image
    return out


def _make_session_bool(per_session_counts: List[int]) -> np.ndarray:
    total = int(sum(per_session_counts))
    n = len(per_session_counts)
    bool_mat = np.zeros((total, n), dtype=bool)
    offset = 0
    for j, count in enumerate(per_session_counts):
        bool_mat[offset : offset + count, j] = True
        offset += count
    return bool_mat


def _default_roinet_cache() -> Path:
    override = os.environ.get("ROIGBIV_ROINET_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "roigbiv" / "roinet"


def _resolve_template_index(n_sessions: int) -> int:
    # ROICaT's default template=0.5 means "middle session"; replicate here for
    # the Pearson-to-template proxy.
    return int(round(0.5 * (n_sessions - 1)))


def _pearson_to_template(
    ims_registered: List[np.ndarray], *, template_idx: int
) -> float:
    if len(ims_registered) < 2:
        return 1.0
    template = np.asarray(ims_registered[template_idx], dtype=np.float64)
    t_flat = template.ravel()
    t_centered = t_flat - t_flat.mean()
    denom_t = float(np.sqrt((t_centered * t_centered).sum()))
    if denom_t <= 0:
        return 0.0
    scores: list[float] = []
    for idx, im in enumerate(ims_registered):
        if idx == template_idx:
            continue
        a = np.asarray(im, dtype=np.float64).ravel()
        a = a - a.mean()
        denom_a = float(np.sqrt((a * a).sum()))
        if denom_a <= 0:
            continue
        r = float((a * t_centered).sum() / (denom_a * denom_t))
        scores.append(max(0.0, min(1.0, r)))
    if not scores:
        return 0.0
    return float(np.mean(scores))


# Kept as utility for callers inspecting per-ROI centroids without a full
# clustering pass. Not used by cluster_sessions itself (ROICaT computes
# centroids internally from the spatial footprints).
def centroids_from_merged_masks(merged_masks: np.ndarray) -> np.ndarray:
    """Return ``(n_rois, 2)`` int64 (y, x) centroids for ROIs in label order."""
    _, label_ids = footprints_from_merged_masks(merged_masks)
    if label_ids.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    coms = center_of_mass(merged_masks > 0, merged_masks, label_ids.tolist())
    return np.asarray(coms, dtype=np.int64)
