"""Resume helpers for the roigbiv sequential pipeline.

Detects which stages of a prior run already completed in ``output_dir``,
reconstructs the in-memory :class:`FOVData` needed to skip past them, and
validates that on-disk artifacts are consistent with the supplied config.

Authoritative on-disk state is the JSON reports
(``stage{N}/stage{N}_report.json``) and the virtual-residual source sidecars
(``residual_S{N}.sources.npz`` + ``residual_S{N}.meta.json``). The advisory
manifest at ``{output_dir}/.roigbiv_manifest.json`` records the config + input
fingerprint and per-step completion timestamps; it is never the source of
truth for ROI content.

Constraints
-----------
- The ``mask`` and ``features`` numpy arrays on :class:`ROI` are not in the
  JSON report. ``mask`` is reconstructed by looking up the ROI's
  ``label_id`` in ``stage{N}_masks.tif`` (which only contains accept + flag
  ROIs — see ``run.py:259-264``). Rejected ROIs are dropped on resume; they
  are never consumed downstream of Stage 1 anyway (gates 2/3/4 filter to
  accept+flag at the call sites).
- The residual chain is **virtual and non-destructive**. The irreplaceable
  substrate is ``svd_factors.npz`` + Suite2p ``data.bin`` (from which S = M − L
  is reconstructed); each subtraction stage adds an independent, never-deleted
  ``residual_S{N}.sources.npz`` (mask pixels, weights, traces). A stage is
  complete when both its report AND its ``sources.npz`` exist. "report present,
  sources.npz missing" means the subtraction was interrupted and resume re-runs
  only that subtraction (the prior view is always rebuildable from the
  substrate + earlier sources files).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from roigbiv import __version__ as _ROIGBIV_VERSION
from roigbiv.pipeline.types import FOVData, PipelineConfig, ROI


MANIFEST_FILENAME = ".roigbiv_manifest.json"

# Step identifiers in execution order. Each appears in the manifest's
# ``stages`` dict once it completes. ``post_detection`` is not a manifest
# step — it is the resume target after Stage 4 detection finishes.
_STEP_ORDER: tuple[str, ...] = (
    "foundation",
    "stage1", "stage1_subtract",
    "stage2", "stage2_subtract",
    "stage3", "stage3_subtract",
    "stage4",
)


class ResumeError(RuntimeError):
    """Raised when ``--resume`` is requested but on-disk state is inconsistent."""


@dataclass
class ResumePlan:
    """Plan describing where ``run_pipeline`` should start.

    Attributes
    ----------
    start_stage
        One of ``"foundation"``, ``"stage1"``, ``"stage1_subtract"``,
        ``"stage2"``, ``"stage2_subtract"``, ``"stage3"``,
        ``"stage3_subtract"``, ``"stage4"``, ``"post_detection"``. The
        pipeline branches on this value at each stage block.
    fov
        Pre-populated FOVData when ``start_stage != "foundation"``. Carries
        summary images in RAM, paths to residuals on disk, and ROIs from
        all previously-completed stages spliced into ``rois``.
    prior_reports
        ``{stage_idx: report_dict}`` for completed stages, so callers can
        recover ``stage_counts`` and emit cascade warnings without
        re-reading JSON.
    cfg_fingerprint
        Fingerprint of the *new* config + input, written into the manifest
        as each step completes.
    enabled
        True iff the caller passed ``enable=True`` and a valid resume plan
        was constructed. False means "start fresh from Foundation".
    """
    start_stage: str = "foundation"
    fov: Optional[FOVData] = None
    prior_reports: dict = field(default_factory=dict)
    cfg_fingerprint: str = ""
    enabled: bool = False

    def should_run(self, step: str) -> bool:
        """Return True iff ``step`` is at or after ``start_stage``.

        Steps strictly before ``start_stage`` were already done on disk and
        are skipped on resume. The orchestrator branches each stage block
        on this predicate.
        """
        return _STAGE_RANKS[step] >= _STAGE_RANKS[self.start_stage]


_STAGE_RANKS: dict[str, int] = {
    "foundation":       0,
    "stage1":           1,
    "stage1_subtract":  2,
    "stage2":           3,
    "stage2_subtract":  4,
    "stage3":           5,
    "stage3_subtract":  6,
    "stage4":           7,
    "post_detection":   8,
}


# ─────────────────────────────────────────────────────────────────────────
# Fingerprint
# ─────────────────────────────────────────────────────────────────────────

# Config keys excluded from the resume fingerprint. Toggling a stage on/off
# between runs must not refuse resume — the whole point is to add a stage to
# a prior run's workspace. Per-stage *parameter* knobs (e.g.
# ``cellprob_threshold``) are still included and will correctly invalidate.
_FINGERPRINT_EXCLUDE: frozenset = frozenset({
    "enable_stage_2", "enable_stage_3", "enable_stage_4",
    "force_cpu",
    # Foundation-only is a stop-point, not a correctness knob: a dry run writes
    # the "foundation" manifest entry, then a later --resume run (with the flag
    # off) must continue from Stage 1 without a fingerprint mismatch.
    "foundation_only",
    # Optics auto-adaptation provenance: recorded for the manifest, not a
    # correctness knob, and mutated post-foundation. Hashing it would make every
    # auto-scaled run un-resumable (resume recomputes the pre-derivation cfg).
    "auto_adapt", "explicit_fields",
    # assume_optics is a pause-suppression knob (like foundation_only): a paused
    # run resumed with --assume-optics / a confirmed profile must not mismatch.
    "assume_optics",
    # resume is an execution-control flag: enabling it is how a prior run is
    # continued, not a change to the artifacts' semantics.
    "resume",
})


# Fields an optics-confirmation resume is allowed to change without forcing a
# foundation re-run: the resolved profile + every categorical/scale knob it
# implies. Anything outside this set (fs, tau, backend, k_background…) must still
# trip the fingerprint guard. Built from the scale-derived set plus the Stage-1
# categoricals + optics provenance/control fields.
def _confirm_resume_fields() -> frozenset:
    from roigbiv.pipeline.optics import SCALE_DERIVED_FIELDS
    return SCALE_DERIVED_FIELDS | {
        "profile", "channels", "cellpose_model", "use_denoise",
        "flow_threshold", "cellprob_threshold", "min_solidity",
        "max_eccentricity", "mc_strip_height", "diameter_auto",
        "auto_scale", "assume_optics", "auto_adapt", "explicit_fields",
        "resume",
    }


_CONFIRM_RESUME_FIELDS: frozenset = _confirm_resume_fields()


def _changed_cfg_fields(snapshot: dict, cfg: PipelineConfig) -> set:
    """Field names whose value differs between a manifest cfg_snapshot and *cfg*.

    Both sides are normalized through ``summary_for_log`` form (tuples → lists)
    so the comparison is apples-to-apples. Keys present on only one side count
    as changed.
    """
    old = _json_ready(snapshot)
    current = _json_ready(cfg.summary_for_log())
    keys = set(old) | set(current)
    return {k for k in keys if old.get(k) != current.get(k)}


def compute_cfg_fingerprint(cfg: PipelineConfig, tif_path: Path) -> str:
    """SHA-256 over ``cfg.summary_for_log()`` + input TIFF size/mtime.

    Stable across runs on the same inputs; changes when any config field or
    the input file's size/mtime changes. Used to refuse resume when state
    on disk was written under a different config. Per-stage on/off flags
    listed in ``_FINGERPRINT_EXCLUDE`` are intentionally dropped before
    hashing so users can flip a stage on with ``--resume``.
    """
    from roigbiv.pipeline.optics import SCALE_DERIVED_FIELDS, auto_scale_active
    exclude = set(_FINGERPRINT_EXCLUDE)
    # When auto-scale is active, the gate numerics are re-derived identically on
    # resume from the on-disk mean_M, so exclude them — EXCEPT any the user
    # pinned explicitly (those stay in the fingerprint so a changed flag is
    # still detected). User-pinned fields are recorded in cfg.explicit_fields.
    if auto_scale_active(cfg):
        exclude |= (SCALE_DERIVED_FIELDS - set(getattr(cfg, "explicit_fields", ()) or ()))
    summary = {k: _json_ready(v) for k, v in cfg.summary_for_log().items()
               if k not in exclude}
    cfg_json = json.dumps(summary, sort_keys=True, default=_json_default)
    stat = _stat_tif(Path(tif_path))
    payload = f"{cfg_json}|{stat['size']}|{stat['mtime_ns']}"
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stat_tif(tif_path: Path) -> dict:
    st = tif_path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _json_default(v):
    if isinstance(v, Path):
        return str(v)
    return str(v)


def _json_ready(v):
    """Recursively canonicalize values the same way JSON manifests store them."""
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, tuple):
        return [_json_ready(x) for x in v]
    if isinstance(v, list):
        return [_json_ready(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_ready(val) for k, val in v.items()}
    return v


# ─────────────────────────────────────────────────────────────────────────
# Manifest IO
# ─────────────────────────────────────────────────────────────────────────

def _manifest_path(output_dir: Path) -> Path:
    return Path(output_dir) / MANIFEST_FILENAME


def read_manifest(output_dir: Path) -> Optional[dict]:
    """Return the parsed manifest dict, or ``None`` if absent/corrupt."""
    p = _manifest_path(output_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def update_manifest(
    output_dir: Path,
    step: str,
    cfg: PipelineConfig,
    tif_path: Path,
    *,
    status: str = "completed",
) -> None:
    """Mark ``step`` complete (or skipped) in the manifest. Creates the file if needed.

    Idempotent: re-marking an already-complete step refreshes its
    timestamp without harm. The manifest carries the *current* config
    snapshot + fingerprint; callers must not call this with a config that
    differs from the one used to produce the on-disk artifacts.

    ``status="skipped"`` records that the stage was intentionally bypassed
    (via ``cfg.enable_stage_N=False``) so a later resume can distinguish
    "skipped on a prior run" from "never ran." A skipped step is still
    recorded — that's how the planner knows the step is "done" for the
    purposes of advancing past it.
    """
    if step not in _STEP_ORDER:
        raise ValueError(f"unknown manifest step: {step!r}")
    if status not in ("completed", "skipped"):
        raise ValueError(f"status must be 'completed' or 'skipped', got {status!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(output_dir) or {}
    if step == "foundation":
        manifest["stages"] = {}
    else:
        manifest.setdefault("stages", {})

    fingerprint = compute_cfg_fingerprint(cfg, tif_path)
    manifest["roigbiv_version"] = _ROIGBIV_VERSION
    manifest["input_tif"] = str(Path(tif_path).resolve())
    manifest["input_stat"] = _stat_tif(Path(tif_path))
    manifest["cfg_fingerprint"] = fingerprint
    manifest["cfg_snapshot"] = _json_ready(cfg.summary_for_log())
    manifest["stages"][step] = {
        "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "version": _ROIGBIV_VERSION,
        "status": status,
    }

    _manifest_path(output_dir).write_text(
        json.dumps(manifest, indent=2, default=_json_default)
    )


# ─────────────────────────────────────────────────────────────────────────
# Disk-state probes
# ─────────────────────────────────────────────────────────────────────────

def _summary_paths(output_dir: Path) -> dict:
    sd = Path(output_dir) / "summary"
    return {
        "mean_M":  sd / "mean_M.tif",
        "mean_S":  sd / "mean_S.tif",
        "max_S":   sd / "max_S.tif",
        "std_S":   sd / "std_S.tif",
        "vcorr_S": sd / "vcorr_S.tif",
        "mean_L":  sd / "mean_L.tif",
        "dog_map": sd / "dog_map.tif",
    }


def _suite2p_plane_dir(output_dir: Path, tif_path: Path | None = None) -> Path:
    output_dir = Path(output_dir)
    stem = (Path(tif_path).stem.replace("_mc", "")
            if tif_path is not None else output_dir.name)
    candidates = (
        output_dir / stem / "suite2p" / "plane0",
        output_dir / "suite2p" / "plane0",
    )
    for plane_dir in candidates:
        if (plane_dir / "data.bin").exists():
            return plane_dir
    return candidates[0]


def _foundation_paths(output_dir: Path, tif_path: Path | None = None) -> dict:
    output_dir = Path(output_dir)
    plane_dir = _suite2p_plane_dir(output_dir, tif_path)
    return {
        "data_bin":   plane_dir / "data.bin",
        "ops":        plane_dir / "ops.npy",
        "svd_factors": output_dir / "svd_factors.npz",
        "residual_meta": output_dir / "residual_S.meta.json",
        "motion_trace": output_dir / "motion_trace.npz",
    }


def _foundation_complete(output_dir: Path, tif_path: Path | None = None) -> bool:
    """All Foundation outputs (summary, data.bin, residual, motion) present."""
    for p in _foundation_paths(output_dir, tif_path).values():
        if not p.exists():
            return False
    for p in _summary_paths(output_dir).values():
        if not p.exists():
            return False
    return True


def _residual_path(output_dir: Path, stage_idx: int) -> tuple[Path, Path]:
    """Return ``(meta_path, sources_path)`` for stage ``stage_idx``'s virtual residual."""
    output_dir = Path(output_dir)
    return (
        output_dir / f"residual_S{stage_idx}.meta.json",
        output_dir / f"residual_S{stage_idx}.sources.npz",
    )


def warn_stale_dense_residuals(output_dir: Path) -> Optional[str]:
    """Flag legacy dense ``residual_S*.dat`` files from the pre-virtual pipeline.

    The current pipeline never materializes a dense residual — it reconstructs
    S = M − L − Σsources on demand (see ``residual.py``). A workspace carried
    over from an older run can still hold multi-gigabyte ``residual_S{,1,2,3}.dat``
    files that the new code path will never read, silently eating disk and
    inflating the very full-disk condition that caused the original SIGBUS.

    Returns a warning string (also printed) listing the stale files and their
    total size, or ``None`` if there are none. Non-destructive: it never
    deletes — it tells the user what to ``rm``.
    """
    output_dir = Path(output_dir)
    stale = sorted(p for p in output_dir.glob("residual_S*.dat") if p.is_file())
    if not stale:
        return None
    total = sum(p.stat().st_size for p in stale)
    names = ", ".join(p.name for p in stale)
    msg = (
        f"Stale dense residual file(s) found in {output_dir} ({names}; "
        f"{total / 1e9:.1f} GB total). The current pipeline reconstructs the "
        f"residual virtually and never reads these — they are safe to delete: "
        f"rm {output_dir}/residual_S*.dat"
    )
    print(f"WARN: {msg}", flush=True)
    return msg


def _verify_foundation_residual(
    meta_path: Path, svd_factors_path: Path, data_bin_path: Path,
) -> tuple[int, int, int]:
    """Validate the virtual foundation residual against its substrate.

    Returns ``(T, Ly, Lx)``. Raises :class:`ResumeError` if the meta is
    malformed or ``data.bin`` is inconsistent with the declared shape. The
    ``svd_factors.npz`` + ``data.bin`` pair is the irreplaceable substrate for
    on-demand reconstruction — both must exist.
    """
    meta = json.loads(meta_path.read_text())
    shape = tuple(int(s) for s in meta["shape"])
    if len(shape) != 3:
        raise ResumeError(f"resume: {meta_path.name} has unexpected shape {shape!r}")
    T, Ly, Lx = shape
    if not Path(svd_factors_path).exists():
        raise ResumeError(
            f"resume: {Path(svd_factors_path).name} missing — cannot reconstruct "
            f"the residual. Re-run without --resume."
        )
    expected = T * Ly * Lx * 2  # int16 data.bin
    actual = Path(data_bin_path).stat().st_size
    if actual != expected:
        raise ResumeError(
            f"resume: data.bin is {actual} bytes but meta declares "
            f"{expected} ({T}x{Ly}x{Lx} int16). Re-run without --resume."
        )
    return T, Ly, Lx


def _verify_sources(sources_path: Path, T: int) -> None:
    """Validate a stage's source sidecar (catches truncated/partial writes)."""
    z = np.load(str(sources_path))
    flat_idx = z["flat_idx"]
    W_design = z["W_design"]
    traces = z["traces"]
    if traces.ndim != 2 or (traces.shape[1] != T and traces.shape[0] != 0):
        raise ResumeError(
            f"resume: {Path(sources_path).name} traces shape {traces.shape} "
            f"inconsistent with T={T}. Re-run without --resume."
        )
    if W_design.shape[1] != flat_idx.size:
        raise ResumeError(
            f"resume: {Path(sources_path).name} W_design cols "
            f"{W_design.shape[1]} != |flat_idx| {flat_idx.size}. "
            f"Likely a truncated write. Re-run without --resume."
        )


# ─────────────────────────────────────────────────────────────────────────
# FOVData reconstruction
# ─────────────────────────────────────────────────────────────────────────

def _load_summary(output_dir: Path, key: str) -> np.ndarray:
    return np.asarray(
        tifffile.imread(str(_summary_paths(output_dir)[key])),
        dtype=np.float32,
    )


def _load_fov_after_foundation(
    output_dir: Path,
    tif_path: Path,
    cfg: PipelineConfig,
) -> FOVData:
    """Build an FOVData equivalent to ``run_foundation``'s return.

    Reads summary images from ``summary/*.tif``, motion traces from
    ``motion_trace.npz``, and shape from ``residual_S.meta.json``. Builds the
    lazy :class:`ResidualView` from ``svd_factors.npz`` + ``data.bin``. The
    Suite2p ops dict is left as ``None`` because (i) ``run_pipeline`` does
    not consume it after Foundation, and (ii) re-loading ``ops.npy`` pulls
    in the full Suite2p import path on every resume.
    """
    from roigbiv.pipeline.residual import ResidualView

    paths = _foundation_paths(output_dir, Path(tif_path))
    T, Ly, Lx = _verify_foundation_residual(
        paths["residual_meta"], paths["svd_factors"], paths["data_bin"],
    )

    motion = np.load(str(paths["motion_trace"]))
    motion_x = np.asarray(motion["xoff"], dtype=np.float32)
    motion_y = np.asarray(motion["yoff"], dtype=np.float32)

    residual_view = ResidualView.from_foundation(
        paths["data_bin"], paths["svd_factors"], (T, Ly, Lx),
        int(cfg.k_background),
    )

    return FOVData(
        raw_path=Path(tif_path),
        output_dir=Path(output_dir),
        data_bin_path=paths["data_bin"],
        shape=(T, Ly, Lx),
        residual_view=residual_view,
        mean_M=_load_summary(output_dir, "mean_M"),
        mean_S=_load_summary(output_dir, "mean_S"),
        max_S=_load_summary(output_dir, "max_S"),
        std_S=_load_summary(output_dir, "std_S"),
        vcorr_S=_load_summary(output_dir, "vcorr_S"),
        mean_L=_load_summary(output_dir, "mean_L"),
        dog_map=_load_summary(output_dir, "dog_map"),
        motion_x=motion_x,
        motion_y=motion_y,
        k_background=int(cfg.k_background),
        ops=None,
    )


def _load_rois_from_report(
    report_path: Path,
    masks_tif_path: Path,
) -> list[ROI]:
    """Reconstruct ROI objects from a stage's report + masks TIFF.

    The mask TIFF only contains accept + flag ROIs (run.py:259-264), so
    rejected ROIs from the report are dropped — they have no recoverable
    mask and are unused downstream of Stage 1 anyway.
    """
    report = json.loads(report_path.read_text())
    label_image = np.asarray(tifffile.imread(str(masks_tif_path)))

    rois: list[ROI] = []
    for entry in report.get("rois", []):
        if entry.get("gate_outcome") == "reject":
            continue
        label_id = int(entry["label_id"])
        mask = (label_image == label_id)
        if not mask.any():
            # Inconsistent: report says accept/flag but mask is empty.
            raise ResumeError(
                f"resume: ROI label_id={label_id} reported as "
                f"{entry.get('gate_outcome')!r} but absent from "
                f"{masks_tif_path.name}."
            )
        rois.append(ROI(
            mask=mask,
            label_id=label_id,
            source_stage=int(entry["source_stage"]),
            confidence=str(entry["confidence"]),
            gate_outcome=str(entry["gate_outcome"]),
            area=int(entry.get("area", int(mask.sum()))),
            solidity=float(entry.get("solidity", 0.0)),
            eccentricity=float(entry.get("eccentricity", 0.0)),
            nuclear_shadow_score=float(entry.get("nuclear_shadow_score", 0.0)),
            soma_surround_contrast=float(entry.get("soma_surround_contrast", 0.0)),
            cellpose_prob=_opt_float(entry.get("cellpose_prob")),
            iscell_prob=_opt_float(entry.get("iscell_prob")),
            event_count=_opt_int(entry.get("event_count")),
            corr_contrast=_opt_float(entry.get("corr_contrast")),
            activity_type=entry.get("activity_type"),
            gate_reasons=list(entry.get("gate_reasons", [])),
            features=dict(entry.get("features", {})),
        ))
    return rois


def _opt_float(v):
    return None if v is None else float(v)


def _opt_int(v):
    return None if v is None else int(v)


# ─────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ─────────────────────────────────────────────────────────────────────────

def plan_resume(
    output_dir: Path,
    tif_path: Path,
    cfg: PipelineConfig,
    *,
    enable: bool,
) -> ResumePlan:
    """Decide where the upcoming ``run_pipeline`` call should start.

    When ``enable=False``, returns a no-op plan (start at foundation). When
    ``enable=True``, walks ``output_dir`` for prior-run artifacts and
    refuses if the cached config fingerprint differs from the new one.

    Raises
    ------
    ResumeError
        If the on-disk state is internally inconsistent (e.g. the manifest
        claims a stage completed but its report or ``sources.npz`` is missing),
        or if the cached fingerprint mismatches.
    """
    output_dir = Path(output_dir)
    fingerprint = compute_cfg_fingerprint(cfg, Path(tif_path))

    if not enable:
        return ResumePlan(start_stage="foundation",
                          cfg_fingerprint=fingerprint,
                          enabled=False)

    manifest = read_manifest(output_dir)
    if manifest is None:
        return ResumePlan(start_stage="foundation",
                          cfg_fingerprint=fingerprint,
                          enabled=True)

    confirm_sentinel = output_dir / "needs_optics_confirmation.json"
    foundation_complete = _foundation_complete(output_dir, Path(tif_path))

    # Optics-confirmation resume: a prior run paused after foundation for the
    # user to confirm the optics, leaving needs_optics_confirmation.json. The
    # confirmed profile legitimately changes Stage-1 config (and thus the
    # fingerprint), but foundation is profile-independent for the default
    # backend — so resume from Stage 1 on the existing foundation rather than
    # refusing. One-shot: consume the sentinel so later plain resumes still
    # enforce the (now-updated) fingerprint.
    is_confirm_resume = False
    changed = set()
    if confirm_sentinel.exists() and foundation_complete:
        # Only bypass the fingerprint if the *only* changed fields are optics /
        # Stage-1 (the confirmed profile + its derived gates). An unrelated edit
        # (fs, tau, backend, k_background…) must still be caught — never absorbed
        # silently by the confirmation path.
        changed = _changed_cfg_fields(manifest.get("cfg_snapshot", {}) or {}, cfg)
        is_confirm_resume = bool(changed) and changed <= _CONFIRM_RESUME_FIELDS

    prior_fp = manifest.get("cfg_fingerprint", "")
    if prior_fp and prior_fp != fingerprint and not is_confirm_resume:
        diff = _describe_config_diff(manifest, cfg, Path(tif_path))
        raise ResumeError(
            f"resume: config or input changed since last run "
            f"({diff}); re-run without --resume to discard prior outputs."
        )
    if is_confirm_resume:
        try:
            confirm_sentinel.unlink()
        except OSError:
            pass

    manifest_stages = manifest.get("stages", {}) or {}

    if not _foundation_complete(output_dir, Path(tif_path)):
        if "foundation" in manifest_stages:
            raise ResumeError(
                "resume: manifest claims foundation completed but its "
                "artifacts (summary/, data.bin, svd_factors.npz, "
                "residual_S.meta.json, motion_trace) are not all present. "
                "Re-run without --resume."
            )
        return ResumePlan(start_stage="foundation",
                          cfg_fingerprint=fingerprint,
                          enabled=True)

    fov = _load_fov_after_foundation(output_dir, Path(tif_path), cfg)
    if is_confirm_resume:
        return ResumePlan(start_stage="stage1",
                          fov=fov,
                          prior_reports={},
                          cfg_fingerprint=fingerprint,
                          enabled=True)
    prior_reports: dict = {}
    start_stage = "stage1"

    for stage_idx in (1, 2, 3, 4):
        report_path = output_dir / f"stage{stage_idx}" / f"stage{stage_idx}_report.json"
        masks_path = output_dir / f"stage{stage_idx}" / f"stage{stage_idx}_masks.tif"

        if not report_path.exists():
            detect_status = (manifest_stages.get(f"stage{stage_idx}", {})
                             .get("status", None))
            if detect_status == "completed":
                raise ResumeError(
                    f"resume: manifest claims stage{stage_idx} completed but "
                    f"{report_path.name} is missing. Re-run without --resume."
                )
            # Either no manifest entry (never ran) or status="skipped"
            # (intentionally bypassed via cfg.enable_stage_N=False). Either
            # way, the next thing to do is enter stage_idx's block — run.py
            # will re-skip if the flag is still off, or run it if newly on.
            start_stage = f"stage{stage_idx}"
            break
        if not masks_path.exists():
            raise ResumeError(
                f"resume: {report_path.name} present but {masks_path.name} "
                f"is missing. Re-run without --resume."
            )

        report = json.loads(report_path.read_text())
        prior_reports[stage_idx] = report
        stage_rois = _load_rois_from_report(report_path, masks_path)
        # ``run_pipeline`` only carries non-rejected ROIs forward (run.py:349).
        # Stage 1 historically keeps rejects in fov.rois, but they have no
        # recoverable mask and are unused downstream — dropping them on
        # resume is safe and keeps fov.rois mask-complete.
        fov.rois.extend(stage_rois)
        fov.stage_counts[f"stage{stage_idx}"] = {
            "detected": int(report.get("detected", 0)),
            "accepted": int(report.get("accepted", 0)),
            "flagged": int(report.get("flagged", 0)),
            "rejected": int(report.get("rejected", 0)),
        }

        if stage_idx == 4:
            # No subtraction after Stage 4; full detection done.
            start_stage = "post_detection"
            break

        # Subtraction step for stages 1-3 (virtual: a sources sidecar).
        out_meta, sources_path = _residual_path(output_dir, stage_idx)
        subtract_step = f"stage{stage_idx}_subtract"

        if out_meta.exists() and sources_path.exists():
            # Completed subtraction: rebuild the layer onto the live view.
            from roigbiv.pipeline.residual import SourceLayer
            _verify_sources(sources_path, fov.shape[0])
            layer = SourceLayer.load(sources_path)
            if layer.traces.shape[0] > 0:
                fov.residual_view = fov.residual_view.with_source(
                    layer.flat_idx, layer.W_design, layer.traces,
                    stage_idx=layer.stage_idx,
                )
            # else: empty (no-ROI) layer — view unchanged, stage still complete.
            continue

        # Subtraction's output is missing. Check the manifest to decide
        # whether this is "skipped on purpose," "completed-but-deleted"
        # (which should refuse), or "genuinely interrupted" (replay).
        subtract_status = (manifest_stages.get(subtract_step, {})
                           .get("status", None))
        if subtract_status == "completed":
            raise ResumeError(
                f"resume: manifest claims {subtract_step} completed but "
                f"residual_S{stage_idx}.sources.npz is missing. "
                f"Re-run without --resume."
            )
        if subtract_status == "skipped":
            # Subtraction was intentionally bypassed (no downstream consumer in
            # the prior run). The live view already reflects the deepest applied
            # subtraction; nothing to append.
            continue

        # Genuinely interrupted subtraction. The substrate (svd_factors.npz +
        # data.bin) is guaranteed present (foundation completeness was checked
        # above) and earlier stages' sources files are never deleted, so the
        # input view is always rebuildable — replay just this subtraction.
        start_stage = subtract_step
        break

    fov.rois.sort(key=lambda r: int(r.label_id))
    return ResumePlan(
        start_stage=start_stage,
        fov=fov,
        prior_reports=prior_reports,
        cfg_fingerprint=fingerprint,
        enabled=True,
    )


def _describe_config_diff(
    manifest: dict,
    cfg: PipelineConfig,
    tif_path: Path,
) -> str:
    """Human-readable diff for the fingerprint-mismatch error message."""
    prior = _json_ready(manifest.get("cfg_snapshot", {}) or {})
    new = _json_ready(cfg.summary_for_log())
    changed_fields: list[str] = []
    for k in sorted(set(prior) | set(new)):
        pv, nv = prior.get(k, "<unset>"), new.get(k, "<unset>")
        if pv != nv:
            changed_fields.append(f"{k}: {pv!r} → {nv!r}")

    prior_stat = manifest.get("input_stat") or {}
    new_stat = _stat_tif(Path(tif_path))
    if prior_stat != new_stat:
        changed_fields.append(
            f"input_stat: size={prior_stat.get('size')} → "
            f"{new_stat['size']}, mtime_ns={prior_stat.get('mtime_ns')} → "
            f"{new_stat['mtime_ns']}"
        )

    if not changed_fields:
        return "fingerprint changed (no field-level diff available)"
    if len(changed_fields) > 4:
        return ", ".join(changed_fields[:4]) + f", +{len(changed_fields)-4} more"
    return ", ".join(changed_fields)
