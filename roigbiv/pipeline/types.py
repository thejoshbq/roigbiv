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


# Anchor the default Cellpose model path to the package root (not cwd) so
# pipeline runs from any working directory load the fine-tuned model. A
# cwd-relative default silently fell back to stock cyto3 when run from
# outside the repo, masking the regression as a detection-quality issue.
_DEFAULT_CELLPOSE_MODEL = str(
    Path(__file__).resolve().parents[2] / "models" / "deployed" / "current_model"
)


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
class BranchView:
    """A view of preprocessed movie data for a given branch (raw, denoised, etc.)."""
    branch_name: str
    movie_view: Path                                     # path to this branch's movie data (e.g. data.bin for raw)
    summary_images: dict = field(default_factory=dict)   # {name: Optional[np.ndarray]}
    provenance: dict = field(default_factory=dict)
    is_denoised: bool = False


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

    # Lazy virtual residual. A single live view accumulates one SourceLayer per
    # subtraction stage; it reconstructs S = M − L − Σsources on demand from
    # data.bin + svd_factors.npz (no dense .dat on disk). Replaces the former
    # residual_S{,1,2,3}_path memmap chain. See roigbiv/pipeline/residual.py.
    residual_view: object = None            # ResidualView (forward ref to avoid import cycle)
    residual_S_path: Optional[Path] = None  # deprecated — kept None; nothing materializes

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
    branches: list = field(default_factory=list)   # list[BranchView]
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

    # ── Scout mode (Cellpose-only triage) ─────────────────────────────────
    # Skip SVD/L+S/residual; compute Cellpose channel 2 as a correlation map on
    # the registered movie. Stops after Stage 1 + Gate 1. Fast FOV-clarity and
    # model A/B triage — NOT analysis-grade (no traces/QC/registry; not resumable).
    scout_mode: bool = False
    scout_vcorr_stride: int = 1             # frame decimation for scout Vcorr (1 = every frame)
    scout_vcorr_neighbors: int = 8          # 8 (full) or 4 (von Neumann) stencil

    # ── Foundation-only dry run ───────────────────────────────────────────
    # Stop immediately after Foundation (motion correction + SVD/L+S + summary
    # images), before Stage 1, so the motion-corrected FOV can be inspected
    # before committing to ROI detection. Writes a foundation_only.json sentinel.
    # Resumable: a later --resume run (without the flag) continues from Stage 1.
    foundation_only: bool = False

    # ── Motion correction backend ─────────────────────────────────────────
    # "phasecorr":   Suite2p rigid + non-rigid registration (default). Robust on
    #                dim, shot-noise-dominated frames where naive per-frame phase
    #                correlation fails; visually matches the legacy SIMA output.
    # "rowwise-pcc": GPU row-wise non-rigid phase correlation. The strip
    #                regularization below (taller strips + median/confidence +
    #                smoothing) suppresses the noise-driven per-row warps that
    #                regressed dim/low-SNR FOVs (~30× less spurious warp on a
    #                still frame); without it, it injects those warps. Still
    #                opt-in: validate parity on the real stack before trusting it
    #                over the phasecorr default.
    # "legacy":      genuine SIMA HiddenMarkov2D(granularity='row') run in the
    #                sima-legacy py3.8 sidecar conda env via subprocess
    #                (roigbiv/pipeline/legacy_mc.py). CPU-only and slow (tens of
    #                minutes to hours per FOV); a faithful reproduction of the
    #                legacy notebook's correction. Opt-in for exact legacy repro.
    # The mc_* knobs apply to rowwise-pcc except where noted (mc_max_displacement
    # is shared; mc_sima_env / mc_granularity apply to legacy).
    motion_correction_backend: str = "phasecorr"
    mc_max_displacement: int = 50           # px clamp; shared by rowwise-pcc + legacy
    mc_strip_height: int = 32               # horizontal strip height (rows); larger
                                            # = higher per-strip SNR on dim data
    mc_n_template_iters: int = 2            # template refinement iterations
    mc_subpixel_upsample: int = 10          # parabolic refinement precision knob
    mc_frame_batch: int = 256              # frames per GPU batch (auto-capped by VRAM)
    # Strip regularization that closes the rowwise-pcc quality gap (Option B).
    mc_smooth_sigma_rows: float = 6.0       # per-row displacement-field smoothing
    mc_smooth_sigma_time: float = 1.0       # temporal smoothing across the frame batch
    mc_strip_confidence_weight: bool = True  # median + confidence outlier rejection
    # DoG band-pass on shift-estimation inputs. Off by default: it helps only when
    # a structured background dominates; on white-noise-limited frames it degrades
    # the correlation peak. Toggle per-dataset via the bench harness.
    mc_prefilter: bool = False
    mc_prefilter_sigma_low: float = 1.0     # shot-noise suppression (small blur)
    mc_prefilter_sigma_high: float = 8.0    # background high-pass (large blur)
    # legacy (SIMA) backend knobs:
    mc_sima_env: str = "sima-legacy"        # conda env hosting SIMA 1.3.2
    mc_granularity: str = "row"             # SIMA HMM2D granularity ('row' | 'frame')

    # ── phasecorr (Suite2p) registration knobs ────────────────────────────
    # These feed the Suite2p ops dict for the *registration* pass of the
    # phasecorr backend (and are forwarded but inert for rowwise-pcc/legacy,
    # which register elsewhere and run Suite2p detection-only). Namespaced
    # mc_s2p_* to avoid colliding with the rowwise-pcc mc_* knobs above (e.g.
    # mc_smooth_sigma_time means something different there).
    #
    # TUNED DEFAULTS: block_size=[64,64] + one_photon_reg(1Preg)=True. Full-
    # session validation on the Logan Prism FOV (2271-frame mean vs a grid-
    # aligned legacy SIMA mean) showed the old [128,128]/no-1Preg default reached
    # only 58% of legacy cell-sharpness, while these reach ~103% (at/above legacy)
    # with no over-fit banding. [64,64] alone gets 91%; 1Preg supplies the rest
    # but is a 1-photon high-pass — if you process bright high-SNR 2P (non-Prism)
    # data, pass --no-mc-1preg (and/or --mc-block-size 128 128). All other knobs
    # keep Suite2p's own defaults. See scripts/sweep_suite2p_reg.py for the sweep.
    mc_s2p_block_size: list = field(default_factory=lambda: [64, 64])  # non-rigid block px (tuned)
    mc_s2p_smooth_sigma: float = 1.15       # spatial Gaussian blur of the reference
    mc_s2p_smooth_sigma_time: float = 0.0   # temporal smoothing for shift estimation
    mc_s2p_maxregshift: float = 0.1         # rigid shift clamp (fraction of frame)
    mc_s2p_nonrigid: bool = True            # enable piecewise (non-rigid) registration
    mc_s2p_maxregshift_nr: int = 5          # max non-rigid block shift (px) → ops maxregshiftNR
    mc_s2p_nimg_init: int = 300             # frames used to build the reference image
    mc_s2p_two_step_registration: bool = False  # rigid pass then non-rigid (needs raw movie)
    # 1-photon-style high-pass before registration (Suite2p 1Preg family). Raises
    # shift-estimation SNR on dim/low-contrast (GRIN/Prism) frames; ON by default
    # (load-bearing for reaching legacy parity). Pass --no-mc-1preg for bright 2P.
    mc_s2p_one_photon_reg: bool = True      # → ops "1Preg" (tuned)
    mc_s2p_spatial_hp_reg: int = 42         # spatial high-pass window (px)
    mc_s2p_pre_smooth: float = 0.0          # pre-high-pass Gaussian smoothing
    mc_s2p_spatial_taper: float = 40.0      # edge pixels tapered out of registration

    # ── Acquisition / lens profile (see pipeline/profiles.py) ─────────────
    # Records which profile bundle the CLI/UI resolver applied
    # (grin/prism/generic). "grin" = dataclass defaults (no-op). Serialized
    # into summary_for_log so the manifest records the resolved profile.
    profile: str = "grin"

    # ── Optics auto-adaptation (see pipeline/optics.py) ───────────────────
    # auto_scale: after foundation, measure the FOV's soma scale and DERIVE the
    # numeric gates (areas, separations, pool radii) from it, overriding the
    # profile's hardcoded numbers but never an explicit user flag. Gated to the
    # prism/generic profiles (or large frames) so the validated GRIN path stays
    # byte-identical unless explicitly opted in. explicit_fields lists the cfg
    # fields the user pinned (which derivation must not clobber); auto_adapt is
    # the provenance record (prior reasons, measured scale, fields overridden),
    # serialized into summary_for_log for an auditable, HITL-reviewable manifest.
    auto_scale: bool = True
    explicit_fields: tuple = ()
    auto_adapt: dict = field(default_factory=dict)
    # When auto-adaptation is uncertain (ambiguous frame size, or the measured
    # soma scale is unreliable/implausible), the run pauses after foundation and
    # writes needs_optics_confirmation.json for the user to confirm the optics,
    # then continues via --resume. assume_optics=True suppresses the pause and
    # proceeds on the best guess — for headless/batch runs that cannot prompt.
    assume_optics: bool = False

    # ── Stage 1 (Cellpose) ────────────────────────────────────────────────
    cellpose_model: str = _DEFAULT_CELLPOSE_MODEL
    diameter: int = 12
    # When True, Stage 1 runs a calibration Cellpose pass with diameter=None
    # on the downsampled mean_M and uses Cellpose's SizeModel estimate as the
    # effective diameter. Overrides `diameter` when the estimator succeeds.
    diameter_auto: bool = False
    cellprob_threshold: float = -2.0
    flow_threshold: float = 0.4
    channels: tuple = (1, 2)
    tile_norm_blocksize: int = 128
    use_denoise: bool = True                # Cellpose3 denoise_cyto3

    # ── Stage 1 backend (Phase M; OFF default — cellpose3 path unchanged) ──
    # "cellpose3"     : in-process cellpose 3.x (deployed CP3 checkpoint / cyto3
    #                   etc.) — the current, default behavior.
    # "cpsam_sidecar" : Cellpose-SAM (cellpose 4.x) run OUT-OF-PROCESS in the
    #                   `cp-sam` conda env. 4.x needs numpy 2.x and cannot share
    #                   this interpreter; the deployed CP3 checkpoint cannot load
    #                   under 4.x (CP3 != CP4). Stage-1 inputs/outputs are
    #                   identical either way, so gates / subtraction / provenance
    #                   / the residual engine are untouched. cpsam is channel-
    #                   invariant and noise-robust → the sidecar drops denoise
    #                   and ignores the channels=(1,2) role convention.
    stage1_backend: str = "cellpose3"
    # Path to the cp-sam env python. "" → auto-resolve: $ROIGBIV_CPSAM_PYTHON,
    # else the sibling `cp-sam` conda env of the running interpreter.
    cpsam_sidecar_python: str = ""

    # ── Stage 1 channel-2 content (Phase 4; OFF default — vcorr_S unchanged) ──
    # The morphological channel-1 is always mean_M (raw movie mean). This selects
    # what fills Cellpose's *second* input channel. CP3's deployed checkpoint is
    # architecturally 2-channel (conv1 in_channels=2), so enrichment happens by
    # swapping ch2 content, not by adding a 3rd channel.
    #   "vcorr_S"          : pixel-correlation map — legacy behavior.
    #   "max_S"            : residual peak-intensity (single-firer / sparse cue).
    #   "vcorr_max_fused"  : per-image min-max-normalized max(vcorr_S, max_S)
    #                        (union of "correlated" OR "has a bright peak") — DEFAULT.
    # Gate 1 always uses vcorr_S regardless — this changes the Stage-1 detector
    # input ONLY (one variable). Falls back to vcorr_S with a warning when max_S
    # is unavailable (e.g. scout-mode foundation).
    # Default flipped vcorr_S -> vcorr_max_fused after Phase-4 A/B (recall +0.017,
    # 0/13 FOV regressions, FP +2.4%); see docs/phase4_channel_ab_report.md.
    stage1_ch2_source: str = "vcorr_max_fused"

    # ── Gate 1 (Morphology) ───────────────────────────────────────────────
    min_area: int = 80
    max_area: int = 600
    min_solidity: float = 0.55
    max_eccentricity: float = 0.90
    min_contrast: float = 0.10
    # Merge detection: a large mask with >=2 intensity peaks is a 2-soma merge
    # admitted by a high max_area ceiling. Such masks are demoted accept->flag
    # (never silently accepted; splitting is a downstream/HITL concern). Inert
    # wherever masks never exceed gate1_merge_peak_min_area (e.g. GRIN profile,
    # max_area=600). Grounded by the Stage-1 recall OFAT (scripts/stage1_matrix).
    gate1_merge_peak_min_area: int = 4000       # only peak-check masks larger than this (px²)
    gate1_merge_peak_min_separation: int = 28   # peak_local_max min_distance (~1 soma radius @ d=56)
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

    # ── Tonic accept tier (Phase 5b, OFF by default — no_default_flip) ─────
    # When enabled, anatomically-detected (source_stage ∈ {1,2}) ROIs that
    # classify as tonic AND whose neuropil_baseline_elevation (5a feature) is
    # ≥ tonic_accept_min_elevation are promoted gate_outcome→"accept" so they
    # skip human review. Stage-4 tonics (source_stage==4) are NEVER touched —
    # their requires_review path (gate4) is unchanged. Threshold is provisional
    # until set from the held-out elevation sweep; flag stays OFF pending an
    # A/B + explicit approval.
    tonic_accept_tier: bool = False
    tonic_accept_min_elevation: float = 0.5

    # ── Subtraction engine ────────────────────────────────────────────────
    subtract_chunk_frames: int = 2000
    subtract_ridge_lambda_scale: float = 1e-6
    subtract_anticorr_threshold: float = -0.3
    subtract_anticorr_failure_fraction: float = 0.10   # trigger NNLS fallback
    subtract_nnls_fallback_max_rois: int = 30
    subtract_solver: str = "ridge"           # "ridge" | "robust"
    subtract_robust_kappa: float = 0.5       # one-sided Huber threshold (sigma units)
    subtract_robust_max_iter: int = 5        # IRLS iteration cap

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
    gate2_min_solidity: float = 0.4         # relaxed vs Gate 1 — Suite2p footprints are noisier than Cellpose
    gate2_max_eccentricity: float = 0.85   # rejects fiber/axon shapes (no Gate 1 equivalent)
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
    stage3_chunk_budget_bytes: int = 1_073_741_824   # 1 GB cap on the per-chunk float32 working set
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

    # ── PMD spatiotemporal denoise (Phase 2, OPTIONAL; OFF by default) ─────
    # Patch-wise penalized-matrix-decomposition denoiser (Buchanan et al.
    # lineage) applied to the residual that feeds Stages 3 and 4. When True it
    # materializes a denoised (T,H,W) float32 memmap and swaps fov.residual_view
    # for a dense-backed ResidualView at the single insertion point in run.py
    # (see docs/phase2_pmd_insertion_point.md). The L+S decomposition, Stage 2's
    # Suite2p reuse, and the ResidualView reconstruction contract are untouched.
    # Full view swap: the denoised residual also feeds trace extraction and the
    # Stage-3 subtraction std (decision D1-b). Implemented in torch (reuses the
    # cu130/sm_120 GPU stack) with CPU fallback. No default flip in this phase.
    use_pmd_denoise: bool = False
    pmd_patch_size: int = 32             # spatial patch edge (px)
    pmd_patch_overlap: int = 8           # patch overlap (px) for averaged blending
    pmd_max_rank: int = 30               # cap on components retained per patch
    pmd_rank_margin: float = 0.0         # extra margin (×) above the MP noise edge
    pmd_band_budget_bytes: int = 1_073_741_824   # ~1GB soft cap on per-band RAM (warns if exceeded)

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

    # ── Resume ────────────────────────────────────────────────────────────
    # When True, run_pipeline consults output_dir for prior-run artifacts
    # and skips stages that already completed. Refuses to resume if the
    # config or input differs from what wrote those artifacts. See
    # roigbiv/pipeline/resume.py for the full state machine.
    resume: bool = False

    # ── Per-stage opt-in flags ────────────────────────────────────────────
    # All stages default on so the cheapest invocation gives full coverage.
    # Stages 3 and 4 add ~10–25 min/FOV but yield real cells in some FOVs;
    # users who want the fast path drop --no-stage-3 / --no-stage-4.
    # Combined with --resume, flipping a flag from True → False (or vice
    # versa) on a prior workspace runs only the now-enabled stage(s); these
    # flags are excluded from the resume fingerprint
    # (resume.py:compute_cfg_fingerprint).
    enable_stage_2: bool = True
    enable_stage_3: bool = True
    enable_stage_4: bool = True
    force_cpu: bool = False

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
