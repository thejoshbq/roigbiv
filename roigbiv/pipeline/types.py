"""
Shared data structures for the sequential subtractive pipeline.

Design note: Stage 2-4 ROI fields are nullable so a Stage 1-only ROI
serializes cleanly today and later stages can populate their own scores
without schema changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ROI:
    """A single detected region of interest.

    Populated incrementally: spatial features at gate time, trace fields
    during/after source subtraction, activity_type during classification.
    """
    mask: np.ndarray                        # (H, W) bool
    label_id: int                           # unique across all stages on this FOV
    source_stage: int                       # 1, 2, 3, or 4
    confidence: str                         # "high" | "moderate" | "requires_review"
    gate_outcome: str                       # "accept" | "flag" | "reject"

    # Spatial features (spec §13.1)
    area: int = 0
    solidity: float = 0.0
    eccentricity: float = 0.0
    nuclear_shadow_score: float = 0.0
    soma_surround_contrast: float = 0.0

    # Per-stage provenance scores (nullable for stages that don't populate)
    cellpose_prob: Optional[float] = None   # Stage 1
    iscell_prob: Optional[float] = None     # Stage 2, future
    event_count: Optional[int] = None       # Stage 3, future
    corr_contrast: Optional[float] = None   # Stage 4, future

    # Traces (populated by subtraction engine or later trace extraction phase)
    trace: Optional[np.ndarray] = None              # (T,) raw fluorescence
    trace_corrected: Optional[np.ndarray] = None    # (T,) neuropil-corrected
    activity_type: Optional[str] = None             # "phasic"|"sparse"|"tonic"|"silent"|"ambiguous"

    # Per-gate feature bucket for anything not promoted to a field
    features: dict = field(default_factory=dict)

    # Reasons the gate flagged/rejected this ROI (human-readable)
    gate_reasons: list = field(default_factory=list)

    def to_serializable(self) -> dict:
        """Return a JSON-safe dict (drops mask and traces, keeps metadata)."""
        return {
            "label_id": int(self.label_id),
            "source_stage": int(self.source_stage),
            "confidence": self.confidence,
            "gate_outcome": self.gate_outcome,
            "area": int(self.area),
            "solidity": float(self.solidity),
            "eccentricity": float(self.eccentricity),
            "nuclear_shadow_score": float(self.nuclear_shadow_score),
            "soma_surround_contrast": float(self.soma_surround_contrast),
            "cellpose_prob": None if self.cellpose_prob is None else float(self.cellpose_prob),
            "iscell_prob": None if self.iscell_prob is None else float(self.iscell_prob),
            "event_count": None if self.event_count is None else int(self.event_count),
            "corr_contrast": None if self.corr_contrast is None else float(self.corr_contrast),
            "activity_type": self.activity_type,
            "gate_reasons": list(self.gate_reasons),
            "features": _jsonable_features(self.features),
        }


def _jsonable_features(features: dict) -> dict:
    """Coerce feature dict values to JSON-safe types.

    Drops large numpy arrays (e.g., trace_bandpass stored for HITL/Napari use).
    Scalar numpy types are cast to Python natives.
    """
    out = {}
    for k, v in features.items():
        if isinstance(v, np.ndarray):
            # Skip bulky array features from JSON; they live on the ROI object.
            continue
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int, bool, np.bool_)):
            out[k] = int(v) if not isinstance(v, (bool, np.bool_)) else bool(v)
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = v
    return out


@dataclass
class FOVData:
    """Container for all per-FOV intermediate products.

    Heavy arrays (registered movie, residual S) live on disk as memmaps;
    summary images (H, W) are held in RAM since they're ~1 MB each.
    """
    raw_path: Path
    output_dir: Path
    data_bin_path: Path                     # int16 memmap of registered movie
    shape: tuple                            # (T, Ly, Lx)

    residual_S_path: Path                   # float32 (T, Ly, Lx) memmap on disk
    residual_S1_path: Optional[Path] = None # populated after Stage 1 subtraction
    residual_S2_path: Optional[Path] = None # populated after Stage 2 subtraction
    residual_S3_path: Optional[Path] = None # populated after Stage 3 subtraction (Stage 4 reads this)
    residual_S4_path: Optional[Path] = None # reserved — Phase 1E does not subtract after Stage 4

    # Summary images in RAM (H, W float32)
    mean_M: Optional[np.ndarray] = None      # raw registered movie mean (morphological channel)
    mean_S: Optional[np.ndarray] = None      # residual mean (near-zero for SVD-based L+S)
    max_S: Optional[np.ndarray] = None
    std_S: Optional[np.ndarray] = None
    vcorr_S: Optional[np.ndarray] = None
    dog_map: Optional[np.ndarray] = None
    mean_L: Optional[np.ndarray] = None

    # SVD factors and motion traces (persisted to disk, paths here)
    svd_factors_path: Optional[Path] = None
    motion_x: Optional[np.ndarray] = None   # (T,)
    motion_y: Optional[np.ndarray] = None   # (T,)

    k_background: int = 30
    rois: list = field(default_factory=list)        # list[ROI]
    stage_counts: dict = field(default_factory=dict)

    ops: Optional[dict] = None              # Suite2p ops dict (lightweight snapshot)

    # Stage 4 per-bandpass-window correlation contrast maps — populated by run_stage4,
    # consumed by the napari viewer and by the Stage 4 TIFF exports.
    corr_contrast_maps: dict = field(default_factory=dict)  # {"fast": (H,W) float32, "medium": ..., "slow": ...}

    # Trace matrices populated at the end of run_pipeline (rows aligned to
    # rois sorted by label_id). Consumed by traces_io.finalize_fov_bundle
    # after the optional registry step.
    F_raw: Optional[np.ndarray] = None
    F_neu: Optional[np.ndarray] = None
    F_corrected: Optional[np.ndarray] = None


@dataclass
class PipelineConfig:
    """All pipeline parameters. Defaults track spec §18.

    Only the user-facing CLI flags (fs, cellpose_model, tau, k_background,
    output_dir, no-viewer) are exposed; everything else is hardcoded here.
    """
    # ── Foundation ────────────────────────────────────────────────────────
    k_background: int = 30                  # spec §3.3 default
    n_svd: int = 200                        # keep for future Stage 2/4 reuse
    batch_size: int = 500                   # Suite2p registration batch
    nonrigid: bool = True
    do_registration: bool = False           # *_mc.tif inputs are pre-corrected
    fs: float = 30.0                        # user-required via CLI; effective Hz (after frame averaging)
    frame_averaging: int = 1                # temporal binning factor that produced fs (1 = un-averaged)
    tau: float = 1.0                        # GCaMP6s
    svd_bin_frames: int = 5000              # target binned frame count
    reconstruct_chunk: int = 500            # temporal chunk size for L+S streaming

    # ── Stage 1 (Cellpose) ────────────────────────────────────────────────
    cellpose_model: str = "models/deployed/current_model"
    diameter: int = 12
    cellprob_threshold: float = -2.0
    flow_threshold: float = 0.6
    channels: tuple = (1, 2)
    tile_norm_blocksize: int = 128
    use_denoise: bool = True                # Cellpose3 denoise_cyto3

    # ── Gate 1 (Morphology) ───────────────────────────────────────────────
    min_area: int = 80
    max_area: int = 600
    min_solidity: float = 0.55
    max_eccentricity: float = 0.90
    min_contrast: float = 0.10
    # Per-criterion absolute margins for marginal flagging
    flag_area_margin: int = 20
    flag_solidity_margin: float = 0.05
    flag_eccentricity_margin: float = 0.03
    flag_contrast_margin: float = 0.03
    # DoG rejection is conjunctive with contrast failure (spec §6)
    dog_strong_negative_percentile: float = 10.0   # score below this dog_map percentile = strong neg

    # ── Annulus for soma-surround contrast ────────────────────────────────
    annulus_inner_buffer: int = 2           # px dilation before ring
    annulus_outer_radius: int = 15          # px dilation for outer edge

    # ── Neuropil / Trace extraction (spec §13.2, §18.10) ──────────────────
    neuropil_coeff: float = 0.7
    neuropil_inner_buffer: int = 2          # px gap between ROI and annulus
    neuropil_outer_radius: int = 15         # px extent of annulus
    baseline_window_s: float = 60.0         # sliding F0 window
    baseline_percentile: int = 10
    tonic_baseline_window_s: float = 120.0  # wider for tonic neurons

    # ── Activity classification (spec §13.3) ──────────────────────────────
    phasic_min_transients: int = 5
    phasic_min_skew: float = 0.5
    sparse_min_transients: int = 1
    sparse_min_skew: float = 0.3
    tonic_bp_std_factor: float = 2.0        # bp_std > this × noise_floor

    # ── Subtraction engine ────────────────────────────────────────────────
    subtract_chunk_frames: int = 2000
    subtract_ridge_lambda_scale: float = 1e-6
    subtract_anticorr_threshold: float = -0.3
    subtract_anticorr_failure_fraction: float = 0.10   # trigger NNLS fallback
    subtract_nnls_fallback_max_rois: int = 30

    # ── Stage 2 (Suite2p) ─────────────────────────────────────────────────
    threshold_scaling: float = 1.0          # Suite2p detection sensitivity (unused when re-reading)
    iscell_threshold: float = 0.3           # cell-classifier cutoff on iscell[:,1]

    # ── Gate 2 (Temporal cross-validation) ────────────────────────────────
    gate2_iou_threshold: float = 0.3        # IoU above which candidate is a rediscovery
    gate2_max_correlation: float = 0.7      # |r| above which candidate is redundant/spillover
    gate2_anticorr_threshold: float = -0.5  # r at/below which candidate is subtraction artifact
    gate2_spatial_radius: int = 20          # px — neighborhood for correlation check
    gate2_min_area: int = 60                # relaxed vs Gate 1 (Suite2p footprints are noisier)
    gate2_max_area: int = 400
    gate2_min_solidity: float = 0.4         # relaxed vs Gate 1
    gate2_near_distance: int = 5            # px — centroid distance triggering near-duplicate check
    gate2_near_corr_threshold: float = 0.5  # |r| above which near-duplicate rejects
    gate2_flag_corr_threshold: float = 0.5  # |r| above which to FLAG rather than ACCEPT

    # ── Stage 3 (Template sweep) ──────────────────────────────────────────
    # Threshold at the high end of spec §18.6 (3.0-6.0σ) because in real
    # residual data the per-pixel noise distribution has a heavier right tail
    # than pure Gaussian (structured neuropil/background leakage). At 4σ we've
    # observed 150M+ false crossings on a single FOV; 6σ brings counts into
    # the 1e3–1e5 range where clustering is tractable.
    template_threshold: float = 6.0         # σ for per-pixel event detection
    spatial_pool_radius: int = 8            # px — soma-radius disk
    spatial_pool_threshold: float = 3.0     # σ for spatial coherence
    cluster_distance: int = 12              # px — fcluster threshold for event accumulation
    min_event_separation: float = 2.0       # seconds — temporal-independence cutoff
    stage3_pixel_chunk_rows: int = 8        # rows of the (T,H,W) memmap per chunk → 4096 px on 512×512
    stage3_sigma_window_frames: int = 500   # sliding MAD window for per-pixel noise
    stage3_max_events: int = 2_000_000      # hard cap — if exceeded, raise threshold adaptively

    # ── Gate 3 (Waveform validation) ──────────────────────────────────────
    gate3_min_waveform_r2: float = 0.6
    gate3_min_waveform_r2_single_event: float = 0.5  # relaxed for confidence=low candidates
    gate3_max_rise_decay_ratio: float = 0.5
    gate3_anticorr_threshold: float = -0.5
    gate3_min_solidity: float = 0.5
    gate3_waveform_window_tau_multiple: float = 5.0  # window = 5 * tau * fs

    # ── Stage 4 (Tonic Neuron Search) — spec §11, §18.8 ───────────────────
    bandpass_windows: list = field(default_factory=lambda: [
        ("fast",   (0.5, 2.0)),    # high-rate tonic (3-5 Hz firing)
        ("medium", (0.1, 1.0)),    # moderate-rate tonic (1-3 Hz)
        ("slow",   (0.05, 0.5)),   # low-rate tonic / slow modulation
    ])
    bandpass_order: int = 4
    n_svd_components_stage4: int = 300
    corr_neighbor_radius_inner: int = 6
    corr_neighbor_radius_outer: int = 15
    corr_contrast_threshold: float = 0.10
    stage4_min_area: int = 80
    stage4_max_area: int = 350
    stage4_min_solidity: float = 0.6
    stage4_max_eccentricity: float = 0.85
    stage4_iou_merge_threshold: float = 0.3
    stage4_pixel_chunk_rows: int = 16   # rows per spatial chunk for sosfiltfilt
    stage4_n_workers: int = 3   # parallel bandpass windows; 1 disables the pool

    # ── Batch execution (Phase B) ─────────────────────────────────────────
    batch_n_workers: int = 1    # 1 = sequential (current); 2 = parallel FOV pool (hard-capped at 2)

    # ── Gate 4 (Correlation Contrast Validation) — spec §12, §18.9 ────────
    gate4_min_corr_contrast: float = 0.10
    gate4_max_motion_corr: float = 0.3
    gate4_anticorr_threshold: float = -0.5
    gate4_min_mean_intensity_pct: int = 25      # percentile of mean_M (see Gate 4 docstring)
    gate4_spatial_radius: int = 20              # reuse Gate 2/3 convention

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: Optional[Path] = None       # None = auto: inference/pipeline/{stem}/
    no_viewer: bool = False

    def summary_for_log(self) -> dict:
        """JSON-serializable snapshot of all config values for pipeline_log.json."""
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                out[k] = str(v)
            elif isinstance(v, tuple):
                out[k] = list(v)
            else:
                out[k] = v
        return out
