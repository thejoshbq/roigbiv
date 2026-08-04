"""
ROI G. Biv pipeline — CLI entry point + orchestrator.

Wires Foundation → Stage 1 Cellpose → Stage 2 Suite2p → Stage 3 FFT sweep →
Stage 4 tonic search → outputs → optional Napari + overlay PNG + email.

The ``roigbiv-pipeline`` console script accepts either a single ``.tif``
file (classic single-FOV mode) or a directory of stacks (workspace mode:
in-input ``output/``, ``registry.db``, auto-migrate, auto-backfill). All
four stages run by default; pass ``--no-stage-N`` to skip. See
``docs/email-notifications.md`` for the optional email-on-done path.

Non-flag parameters are hardcoded in :class:`PipelineConfig` per spec §18.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.pipeline import fmt
from roigbiv.pipeline.optics import (
    OpticsPrior,
    auto_scale_active,
    classify_optics_prior,
)
from roigbiv.pipeline.profiles import (
    AUTO,
    STAGE1_CLI_DEFAULTS,
    get_profile,
    is_profile,
    list_profiles,
    merged_overrides,
)
from roigbiv.pipeline.roi_stamp import canonicalize, resolve_crowding
from roigbiv.pipeline.types import (
    DEFAULT_PIPELINE_MODE,
    FOVData,
    PIPELINE_MODES,
    PipelineConfig,
)


class OpticsConfirmationRequired(Exception):
    """Raised after foundation when auto-adaptation is too uncertain to proceed.

    The foundation outputs are already on disk (its manifest entry is written),
    so resolving the optics and re-running with ``--resume`` continues from
    Stage 1 — nothing recomputes. ``payload`` is the confirmation context
    (candidate profile, prior reasons, measured scale) also written to
    ``needs_optics_confirmation.json``.
    """

    def __init__(self, output_dir: "Path", payload: dict) -> None:
        self.output_dir = output_dir
        self.payload = payload
        super().__init__(
            f"optics confirmation required for {Path(output_dir).name}: "
            f"candidate profile {payload.get('candidate_profile')!r}"
        )


class PipelineAborted(BaseException):
    """Raised at a stage boundary when a cooperative stop was requested.

    Derives from ``BaseException`` (not ``Exception``) so the per-stage
    ``except Exception`` handlers inside :func:`run_pipeline` don't swallow
    it — it propagates to the workspace runner, which records the FOV as
    ``aborted`` and skips the registry write.
    """


def _check_abort(abort_event) -> None:
    """Raise :class:`PipelineAborted` if a stop was requested.

    No-op when ``abort_event`` is ``None`` (CLI, single-FOV, and batch-worker
    paths — the batch path stops via worker-process termination, not this
    in-process flag).
    """
    if abort_event is not None and abort_event.is_set():
        raise PipelineAborted()


_GATE2_MIN_SCALE = 3 / 4   # gate2_min_area / gate1_min_area at GRIN defaults (60/80)
_GATE2_MAX_SCALE = 2 / 3   # gate2_max_area / gate1_max_area at GRIN defaults (400/600)


def _build_gate2_overrides(args: "argparse.Namespace") -> dict:
    """Derive Gate 2 area bounds from Gate 1 when the user sets --min/max-area.

    Priority: explicit --gate2-* flag > proportional scale from Gate 1 > nothing
    (PipelineConfig defaults remain when neither is set).

    Suite2p footprints run ~0.75× the Cellpose footprint area at the same
    cell scale; the scale factors preserve that ratio from the GRIN defaults.

    NOTE: When auto_scale is active, _apply_auto_scale() runs post-foundation and
    overwrites gate2_min_area / gate2_max_area via derive_scale_params(). Intentional
    — measured soma diameter is more authoritative than the proportional CLI hint.
    """
    out: dict = {}
    g2_min = getattr(args, "gate2_min_area", None)
    g2_max = getattr(args, "gate2_max_area", None)
    g1_min = getattr(args, "min_area", None)
    g1_max = getattr(args, "max_area", None)
    if g2_min is None and g1_min is not None:
        g2_min = max(1, round(g1_min * _GATE2_MIN_SCALE))
    if g2_max is None and g1_max is not None:
        g2_max = max(1, round(g1_max * _GATE2_MAX_SCALE))
    if g2_min is not None:
        out["gate2_min_area"] = g2_min
    if g2_max is not None:
        out["gate2_max_area"] = g2_max
    return out


def _mem_snapshot(label: str) -> None:
    """Print a one-line process-memory snapshot, tagged with ``label``.

    Reads /proc/self/status (Linux only); silently no-ops elsewhere. Splits
    out anonymous-heap vs file-backed resident, because only anonymous-rss
    is non-reclaimable and drives the kernel OOM-killer. Useful for tracking
    down anon-rss leaks across stage boundaries.
    """
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            d = {}
            for line in f:
                for k in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSize"):
                    if line.startswith(k + ":"):
                        d[k] = int(line.split()[1]) / 1024  # MB
        print(
            f"  [mem] {label:<28s} "
            f"anon={d.get('RssAnon', 0):>7.0f} MB  "
            f"file={d.get('RssFile', 0):>7.0f} MB  "
            f"total={d.get('VmRSS', 0):>7.0f} MB  "
            f"vsz={d.get('VmSize', 0):>7.0f} MB",
            flush=True,
        )
    except (OSError, KeyError, ValueError):
        pass


def _summarize_cellpose_probs(probs: list[float]) -> str:
    """Compact one-line summary of Cellpose cellprob distribution.

    When Cellpose returns very few candidates (e.g. on optics the model
    wasn't trained on), this lets the user see at a glance whether the
    detector is finding *nothing* (low probs) vs. finding things that
    Gate 1 then rejects (high probs, see Gate 1 reasons).
    """
    if not probs:
        return "no candidates"
    arr = np.asarray(probs, dtype=np.float32)
    return (
        f"n={len(arr)} "
        f"prob min={arr.min():.2f} med={float(np.median(arr)):.2f} "
        f"max={arr.max():.2f}"
    )


def _gate_reject_histogram(rois) -> str:
    """Histogram (top 5) of rejection reasons across a ROI list.

    Reads each ROI's ``gate_reasons`` list (populated by gate evaluators)
    and counts the unique reason prefixes. When everything is rejected,
    this surfaces *why* — e.g. "area_low:80 solidity:31 contrast:5".
    """
    counter: Counter = Counter()
    for r in rois:
        if getattr(r, "gate_outcome", None) != "reject":
            continue
        for reason in getattr(r, "gate_reasons", []) or []:
            head = str(reason).split(":", 1)[0]
            counter[head] += 1
    if not counter:
        return "(no reasons recorded)"
    return " ".join(f"{k}:{v}" for k, v in counter.most_common(5))


def _gpu_section(gpu_lock):
    """Return `gpu_lock` as a context manager, or a no-op if None.

    Used by `run_pipeline` to serialize GPU-heavy phases (Cellpose, Suite2p,
    Stage 3 FFT, source subtraction) across FOV workers when a shared
    `multiprocessing.Manager().Lock()` is passed by the batch runner. When
    `gpu_lock is None` (single-FOV default), this is a zero-cost no-op.
    """
    return gpu_lock if gpu_lock is not None else nullcontext()


def _any_downstream_enabled(cfg: PipelineConfig, after_stage: int) -> bool:
    """True iff any stage strictly after ``after_stage`` is enabled.

    Used to decide whether to run a stage's subtraction step: subtraction is
    only useful if a later enabled stage will consume the residual it writes.
    Stage 1's subtraction is unconditional (handled by callers); this helper
    is for stages 2 and 3.
    """
    for k in range(after_stage + 1, 5):
        if getattr(cfg, f"enable_stage_{k}", False):
            return True
    return False


def _read_subtraction_pass_count(
    output_dir: Path, stage_idx: int, fallback: int,
) -> int:
    """Read ``subtraction_report_residual_S{N}.json`` and count passes.

    Used on resume to populate ``n_sub_pass`` when the subtraction step was
    skipped. Falls back to ``fallback`` (typically the ROI count, i.e.
    "assume all passed") when the report is absent or unparseable.
    """
    report_path = output_dir / f"subtraction_report_residual_S{stage_idx}.json"
    if not report_path.exists():
        return fallback
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback
    return sum(1 for v in report.values()
               if isinstance(v, dict) and v.get("pass"))


def _default_output_dir(tif_path: Path) -> Path:
    """Default: inference/pipeline/{stem}/ relative to the current working dir.

    If cwd contains an `inference/` directory, use it; otherwise, use
    {input_parent}/../inference/pipeline/{stem}/ (walks up to find a project root).
    """
    stem = tif_path.stem.replace("_mc", "")
    cwd = Path.cwd()
    if (cwd / "inference").exists() or cwd.name == "roigbiv":
        return cwd / "inference" / "pipeline" / stem
    # Walk up from the input looking for a project root (has 'inference/' or 'roigbiv/')
    for parent in tif_path.resolve().parents:
        if (parent / "inference").exists() or (parent / "roigbiv").exists():
            return parent / "inference" / "pipeline" / stem
    # Fallback: sibling of input
    return tif_path.resolve().parent / "inference" / "pipeline" / stem


def print_detection_summary(fov: FOVData) -> list[str]:
    """Print the final four-stage detection table with monotonicity check.

    Returns any warnings generated (e.g., monotonicity violations indicating
    cascade artifact propagation — Blindspot 2).
    """
    sc = fov.stage_counts
    stage_labels = {
        "stage1": "Stage 1 (Cellpose)",
        "stage2": "Stage 2 (Suite2p) ",
        "stage3": "Stage 3 (Template)",
        "stage4": "Stage 4 (Tonic)   ",
    }
    print(fmt.stage_header("Summary", "Detection Complete"), flush=True)
    detected_seq: list[tuple[str, int]] = []
    total_kept = 0
    for key in ("stage1", "stage2", "stage3", "stage4"):
        if key not in sc:
            continue
        s = sc[key]
        det = int(s.get("detected", 0))
        acc = int(s.get("accepted", 0))
        flg = int(s.get("flagged", 0))
        rej = int(s.get("rejected", 0))
        detected_seq.append((key, det))
        n = int(key[-1])
        print(fmt.gate_outcome(n, det, acc, flg, rej), flush=True)
        total_kept += acc + flg

    print(fmt.sub_phase(f"Total ROIs (accept+flag): {total_kept}"), flush=True)

    # Monotonicity check — detected counts should decrease across stages
    # since each stage subtracts the strongest sources before the next runs.
    warnings: list[str] = []
    violations: list[str] = []
    for i in range(1, len(detected_seq)):
        prev_key, prev_n = detected_seq[i - 1]
        curr_key, curr_n = detected_seq[i]
        if prev_n > 0 and curr_n > prev_n:
            violations.append(
                f"{curr_key} detected {curr_n} > {prev_key} detected {prev_n}"
            )
    if detected_seq:
        chain = " > ".join(f"{n}" for _, n in detected_seq)
        if violations:
            msg = ("Monotonicity WARNING: detection counts should decrease "
                   "across stages (Blindspot 2). Offenders: "
                   + "; ".join(violations)
                   + f"  (chain: {chain})")
            print(fmt.sub_phase(f"WARNING: {msg}"), flush=True)
            warnings.append(msg)
        else:
            print(fmt.sub_phase(f"Monotonicity check: {chain}  ✓"), flush=True)
    return warnings


def _write_scout_overlay(fov: FOVData, output_dir: Path, cfg: PipelineConfig) -> None:
    """Render the Stage 1 + Gate 1 masks over mean_M as a PNG (scout mode)."""
    from roigbiv import overlay as _overlay
    fov_stem = Path(fov.raw_path).stem.replace("_mc", "")
    try:
        png = _overlay.render_overlay(
            fov, output_dir, Path(cfg.cellpose_model).name, fov_stem,
        )
        print(fmt.sub_phase(f"scout overlay → {png}"), flush=True)
    except Exception as exc:  # noqa: BLE001 — overlay is best-effort in triage
        print(fmt.sub_phase(f"scout overlay skipped ({exc})"), flush=True)


def _write_stage4_outputs(
    fov: FOVData,
    stage4_rois: list,
    cfg: PipelineConfig,
    output_dir: Path,
) -> None:
    """Write Stage 4 per-FOV outputs: masks, contrast maps, corr scores, report."""
    import json
    import numpy as np
    import tifffile

    stage4_dir = output_dir / "stage4"
    stage4_dir.mkdir(exist_ok=True)

    # Mask image (uint16 label over accept+flag ROIs — for Stage 4 all survivors are "flag")
    Ly, Lx = fov.shape[1], fov.shape[2]
    stage4_mask_img = np.zeros((Ly, Lx), dtype=np.uint16)
    for r in stage4_rois:
        if r.gate_outcome in ("accept", "flag"):
            stage4_mask_img[r.mask] = r.label_id
    tifffile.imwrite(str(stage4_dir / "stage4_masks.tif"), stage4_mask_img)

    # Per-window correlation contrast maps
    for window_name, cmap in fov.corr_contrast_maps.items():
        tifffile.imwrite(
            str(stage4_dir / f"corr_contrast_{window_name}.tif"),
            cmap.astype(np.float32),
        )

    # Per-ROI corr_contrast scores (label_id, corr_contrast)
    cc_entries = np.array(
        [(r.label_id, float(r.corr_contrast or 0.0)) for r in stage4_rois],
        dtype=[("label_id", np.int32), ("corr_contrast", np.float32)],
    )
    np.save(str(stage4_dir / "stage4_corr_contrast.npy"), cc_entries)

    n_det = len(stage4_rois)
    n_flag = sum(1 for r in stage4_rois if r.gate_outcome == "flag")
    n_rej = sum(1 for r in stage4_rois if r.gate_outcome == "reject")
    (stage4_dir / "stage4_report.json").write_text(json.dumps({
        "detected": n_det,
        "accepted": 0,   # by design — Gate 4 has no accept tier
        "flagged": n_flag,
        "rejected": n_rej,
        "rois": [r.to_serializable() for r in stage4_rois],
    }, indent=2))


def run_pipeline(tif_path: Path, cfg: PipelineConfig, gpu_lock=None,
                 abort_event=None) -> FOVData:
    """Run the full four-stage pipeline on a single FOV.

    Writes all outputs to cfg.output_dir and (unless cfg.no_viewer) opens a
    napari viewer at the end.

    If `gpu_lock` is provided (a `multiprocessing.Manager().Lock()` from the
    batch runner), GPU-heavy phases are serialized across concurrent FOV
    workers so the RTX-4060 8 GiB VRAM isn't double-booked. CPU phases
    (Foundation summary images, Stage 4 bandpass, traces, QC) overlap freely.

    If `abort_event` (a `threading.Event`) is provided and set, the run stops
    cooperatively at the next stage boundary by raising :class:`PipelineAborted`.
    The last fully-completed stage's outputs + manifest entry are intact, so a
    subsequent ``--resume`` run continues from there. ``None`` (CLI / batch
    workers) disables the check.

    Returns a fully populated FOVData (through Stage 4).
    """
    from roigbiv.pipeline.foundation import run_foundation
    from roigbiv.pipeline.stage1 import run_cellpose_detection
    from roigbiv.pipeline.gate1 import evaluate_gate1
    from roigbiv.pipeline.subtraction import run_source_subtraction, compute_std_map
    from roigbiv.pipeline.stage2 import run_stage2
    from roigbiv.pipeline.gate2 import evaluate_gate2
    from roigbiv.pipeline.stage3 import run_stage3
    from roigbiv.pipeline.stage3_templates import build_template_bank
    from roigbiv.pipeline.gate3 import evaluate_gate3
    from roigbiv.pipeline.stage4 import run_stage4
    from roigbiv.pipeline.gate4 import evaluate_gate4
    from roigbiv.pipeline.pmd import pmd_denoise_view
    from roigbiv.pipeline.traces import extract_all_traces
    from roigbiv.pipeline.overlap_correction import (
        find_overlap_groups, correct_overlapping_traces,
    )
    from roigbiv.pipeline.qc_features import compute_all_features
    from roigbiv.pipeline.classify import classify_all_rois
    from roigbiv.pipeline.gate_tonic_elevation import apply_tonic_accept_tier
    from roigbiv.pipeline.dff import compute_all_dff
    from roigbiv.pipeline.deconvolution import deconvolve_traces
    from roigbiv.pipeline.hitl import build_review_queue, export_hitl_package
    from roigbiv.pipeline.outputs import save_pipeline_outputs, print_final_summary
    from roigbiv.pipeline.resume import plan_resume, update_manifest

    tif_path = Path(tif_path).resolve()
    if cfg.output_dir is None:
        cfg.output_dir = _default_output_dir(tif_path)

    # Enable motion correction for stacks that aren't already pre-corrected.
    # Detection is content-first (embedded TIFF Software tag on roigbiv-produced
    # movies) with a strict ``_mc`` filename fallback for external/legacy inputs —
    # see roigbiv.io.detect_motion_corrected. Callers need not know about
    # do_registration; the input itself is the canonical signal.
    from roigbiv.io import detect_motion_corrected
    _pre_corrected, _mc_signal = detect_motion_corrected(tif_path)
    if not _pre_corrected:
        cfg.do_registration = True
    else:
        print(f"  pre-corrected input detected via {_mc_signal}; "
              f"Suite2p registration disabled", flush=True)

    # Frame-size sanity warn: defaults are tuned for 512×512 GRIN. On larger
    # frames (prism, 1024×1024) the GRIN diameter starves Cellpose and the
    # Gate 1 area ceiling clips real cells. Catches silent misuse at startup.
    # The same peek captures (T, H, W) for the disk-budget preflight below.
    _peek_shape: tuple | None = None
    _peek_pages: int | None = None
    try:
        import tifffile as _tf
        with _tf.TiffFile(str(tif_path)) as _peek:
            _shape = _peek.pages[0].shape
            _peek_pages = len(_peek.pages)
        _peek_shape = _shape
        _max_dim = max(_shape[-2], _shape[-1])
        if _max_dim > 768 and cfg.diameter < 25 and not cfg.diameter_auto:
            print(
                f"WARN: frame size {_shape[-2]}×{_shape[-1]} with "
                f"diameter={cfg.diameter}. Defaults are tuned for 512×512 "
                f"GRIN imaging; on larger frames cells will be undersegmented "
                f"and rejected by Gate 1. Consider --diameter-auto, an "
                f"explicit --diameter (target ~50 for 1024×1024 prism), or "
                f"raise --max-area above the configured default.",
                flush=True,
            )
    except BaseException:  # noqa: BLE001
        pass

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_timings: dict = {}
    warnings: list[str] = []

    # ── Resume planning ───────────────────────────────────────────────────
    rp = plan_resume(output_dir, tif_path, cfg, enable=cfg.resume)
    if rp.enabled and rp.start_stage != "foundation":
        print(f"Resume: starting from {rp.start_stage} "
              f"(prior outputs in {output_dir.name})", flush=True)

    # Surface legacy dense residual_S*.dat files (pre-virtual pipeline). They
    # are never read now but eat disk — the same full-disk condition behind the
    # original SIGBUS. Non-destructive: warns and tells the user what to rm.
    from roigbiv.pipeline.resume import warn_stale_dense_residuals
    _stale = warn_stale_dense_residuals(output_dir)
    if _stale:
        warnings.append(_stale)

    # ── Disk-budget preflight ─────────────────────────────────────────────
    # Fail fast (or warn) on a near-full disk before Foundation spends ~20 min
    # writing data.bin only to SIGBUS/ENOSPC. Skipped when Foundation is being
    # resumed (data.bin already on disk) or when we couldn't peek the stack.
    if rp.should_run("foundation") and _peek_shape is not None and _peek_pages:
        try:
            from roigbiv.pipeline.diskguard import preflight_disk_budget
            _T = int(_peek_pages)
            _H, _W = int(_peek_shape[-2]), int(_peek_shape[-1])
            _px = _T * _H * _W
            _stage4_temp = (2 * _px * 4) if getattr(cfg, "enable_stage_4", True) else 0
            _warn = preflight_disk_budget(
                output_dir,
                data_bin_bytes=_px * 2,          # int16 Suite2p substrate
                stage4_temp_bytes=_stage4_temp,  # detrend + bandpass float32 temps
                label=f"pipeline run ({tif_path.name})",
            )
            if _warn:
                print(f"WARN: {_warn}", flush=True)
                warnings.append(_warn)
        except RuntimeError:
            raise
        except BaseException:  # noqa: BLE001 — preflight is best-effort, never fatal-by-bug
            pass

    # Locals downstream code reads regardless of whether the stage ran or
    # was skipped. Skipped-stage paths repopulate these from on-disk reports.
    n_accept = n_flag = n_reject = 0
    n2_det = n2_acc = n2_flag = n2_rej = 0
    n3_det = n3_acc = n3_flag = n3_rej = 0
    n4_det = n4_flag = n4_rej = 0
    rois_to_subtract: list = []
    n_sub_pass = n_sub_total = 0
    s2_subtract: list = []
    n2_sub_pass = 0
    s3_subtract: list = []
    n3_sub_pass = 0

    _mem_snapshot("pre-foundation")

    # ── Foundation ────────────────────────────────────────────────────────
    if rp.should_run("foundation"):
        t_start = time.time()
        fov = run_foundation(tif_path, cfg, output_dir, gpu_lock=gpu_lock)
        stage_timings["foundation_s"] = time.time() - t_start
        update_manifest(output_dir, "foundation", cfg, tif_path)
        print(fmt.sub_phase(
            f"Foundation complete. k_background={fov.k_background}  "
            f"(T={fov.shape[0]}, H={fov.shape[1]}, W={fov.shape[2]})"
        ), flush=True)
        _mem_snapshot("post-foundation")
    else:
        fov = rp.fov  # populated by plan_resume from disk
        print(fmt.sub_phase(
            f"Resume: foundation skipped "
            f"(T={fov.shape[0]}, H={fov.shape[1]}, W={fov.shape[2]})"
        ), flush=True)

    # Cooperative stop: foundation is on disk; halt before the optics gate /
    # Stage 1 so a --resume run continues from here.
    _check_abort(abort_event)

    # ── Optics auto-scale (post-foundation, pre-Stage-1) ──────────────────
    # Derive the numeric gates (areas/diameter/separations/pool radii) from the
    # FOV's own measured soma scale, overriding the profile's hardcoded numbers
    # but never an explicit user flag (cfg.explicit_fields). Gated to
    # prism/generic so the validated GRIN path stays byte-identical, and only
    # when Stage 1 will actually run. Re-derives identically on resume (it reads
    # the on-disk mean_M), which is why these fields are excluded from the resume
    # fingerprint. Runs before the foundation-only short-circuit so a dry run
    # still records the measured scale for the user to inspect.
    if rp.should_run("stage1") and auto_scale_active(cfg):
        _apply_auto_scale(cfg, fov)

    # ── Foundation-only short-circuit ─────────────────────────────────────
    # Stop before any ROI detection so the motion-corrected FOV + summary
    # images can be inspected first. A sentinel marks the dry run; the
    # "foundation" manifest entry is already written, so a later --resume run
    # (without --foundation-only) continues from Stage 1.
    if getattr(cfg, "foundation_only", False):
        (output_dir / "foundation_only.json").write_text(json.dumps({
            "mode": "foundation_only",
            "backend": getattr(cfg, "motion_correction_backend", "phasecorr"),
            "shape": list(fov.shape),
            "note": ("Foundation only: motion correction + SVD/L+S + summary "
                     "images. No ROI detection / traces / QC / registry. "
                     "Re-run with --resume (without --foundation-only) to "
                     "continue from Stage 1."),
        }, indent=2))
        print(fmt.sub_phase(
            "Foundation-only dry run: stopping before Stage 1. "
            f"Preview summary/*.tif + {tif_path.stem.replace('_mc', '')}_mc.tif."
        ), flush=True)
        if not cfg.no_viewer:
            print("\nLaunching Napari viewer (foundation-only)...", flush=True)
            from roigbiv.pipeline.napari_viewer import display_pipeline_results
            display_pipeline_results(fov, review_queue=[])
        return fov

    # ── Optics confirmation gate (pause-to-confirm) ───────────────────────
    # When auto-adaptation is too uncertain, stop here (foundation is on disk,
    # like a foundation-only dry run) and ask the user to confirm the optics.
    # --assume-optics proceeds on the best guess for headless/batch runs.
    if rp.should_run("stage1") and not getattr(cfg, "assume_optics", False):
        _confirm = _optics_confirmation_decision(cfg)
        if _confirm is not None:
            sentinel = _write_optics_confirmation_sentinel(output_dir, cfg, _confirm)
            print(fmt.sub_phase(
                f"Optics confirmation needed → wrote {sentinel.name}. "
                f"Candidate '{_confirm['candidate_profile']}' "
                f"({_confirm.get('confidence')}). Re-run with --resume "
                f"[--profile <choice>] to continue, or --assume-optics to proceed."
            ), flush=True)
            raise OpticsConfirmationRequired(output_dir, _confirm)

    # ── Stage 1 Cellpose detection ────────────────────────────────────────
    if rp.should_run("stage1"):
        print(fmt.stage_header(1, "Cellpose detection"), flush=True)
        _check_abort(abort_event)
        t_start = time.time()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Use mean_M (raw movie mean) as the morphological channel for Cellpose.
        # mean_S ≈ 0 under SVD-based L+S (top components absorb per-pixel mean),
        # so mean_M preserves the morphological contrast Cellpose's training expects.
        with _gpu_section(gpu_lock):
            candidates, probs, label_image, cellprob_map = run_cellpose_detection(
                fov.mean_M, fov.vcorr_S, cfg, max_S=fov.max_S,
            )
        stage_timings["stage1_detect_s"] = time.time() - t_start
        n_detected = len(candidates)
        print(fmt.sub_phase(
            f"Cellpose: {_summarize_cellpose_probs(probs)}"
        ), flush=True)

        # Save raw Stage 1 outputs (pre-gate)
        (output_dir / "stage1").mkdir(exist_ok=True)
        tifffile.imwrite(str(output_dir / "stage1" / "stage1_probs.tif"),
                         cellprob_map.astype(np.float32))

        # ── Gate 1 morphological validation ───────────────────────────────
        # Use mean_M for soma-surround contrast (raw brightness levels;
        # mean_S is near-zero and would make the contrast metric numerically
        # unstable).
        t_start = time.time()
        rois = evaluate_gate1(candidates, probs, fov.mean_M, fov.vcorr_S,
                              fov.dog_map, cfg, starting_label_id=1)
        stage_timings["gate1_s"] = time.time() - t_start

        # Canonicalize: replace each accept/flag ROI's raw Cellpose boundary
        # with a fixed-radius disk at its own centroid (roi_stamp.py). Gate 1
        # already ran on the real geometry above; only what subtraction/
        # traces/HITL/registry consume from here on changes. Reject-outcome
        # ROIs keep their real mask (audit trail only, never subtracted).
        for r in rois:
            if r.gate_outcome in ("accept", "flag"):
                canonicalize(r, cfg.roi_stamp_radius, fov.mean_M.shape)
        # Two originally-distinct (non-overlapping) Cellpose instances can end
        # up with crowded canonical stamps if their centroids are close — guard
        # before counting outcomes below, since crowding can demote accept->flag.
        resolve_crowding(rois, cfg.roi_stamp_radius)

        n_accept = sum(1 for r in rois if r.gate_outcome == "accept")
        n_flag = sum(1 for r in rois if r.gate_outcome == "flag")
        n_reject = sum(1 for r in rois if r.gate_outcome == "reject")
        print(fmt.gate_outcome(1, n_detected, n_accept, n_flag, n_reject), flush=True)
        if n_reject:
            print(fmt.sub_phase(
                f"Gate 1 reject reasons: {_gate_reject_histogram(rois)}"
            ), flush=True)
        _mem_snapshot("post-stage1+gate1")

        # Save Stage 1 mask image (accepted + flagged only — rejects not subtracted)
        stage1_mask_img = np.zeros(fov.mean_M.shape, dtype=np.uint16)
        for r in rois:
            if r.gate_outcome in ("accept", "flag"):
                stage1_mask_img[r.mask] = r.label_id
        tifffile.imwrite(str(output_dir / "stage1" / "stage1_masks.tif"),
                         stage1_mask_img)

        stage1_report = {
            "detected": n_detected,
            "accepted": n_accept,
            "flagged": n_flag,
            "rejected": n_reject,
            "rois": [r.to_serializable() for r in rois],
        }
        (output_dir / "stage1" / "stage1_report.json").write_text(
            json.dumps(stage1_report, indent=2)
        )
        fov.rois = rois
        fov.stage_counts["stage1"] = {
            "detected": n_detected, "accepted": n_accept,
            "flagged": n_flag, "rejected": n_reject,
        }
        update_manifest(output_dir, "stage1", cfg, tif_path)
    else:
        sc = fov.stage_counts.get("stage1", {})
        n_accept = int(sc.get("accepted", 0))
        n_flag = int(sc.get("flagged", 0))
        n_reject = int(sc.get("rejected", 0))
        print(fmt.sub_phase(
            f"Resume: stage1 skipped "
            f"({n_accept} accepted, {n_flag} flagged, {n_reject} rejected)"
        ), flush=True)

    _mem_snapshot("pre-stage1-subtract")

    # ── Scout short-circuit ───────────────────────────────────────────────
    # Cellpose-only triage: masks + overlay are written, everything residual-
    # dependent (subtraction, Stages 2-4, traces, QC, classify, HITL, registry)
    # is skipped. See foundation._run_foundation_scout.
    if cfg.scout_mode:
        _write_scout_overlay(fov, output_dir, cfg)
        # Sentinel so the UI / registry can tell a triage run from a full one.
        (output_dir / "scout.json").write_text(json.dumps({
            "mode": "scout",
            "vcorr_stride": cfg.scout_vcorr_stride,
            "vcorr_neighbors": cfg.scout_vcorr_neighbors,
            "note": ("Cellpose-only triage: channel 2 is correlation on the "
                     "registered movie, not the residual. No traces/QC/"
                     "registry; re-run without scout for analysis-grade output."),
            "n_rois": len(fov.rois),
        }, indent=2))
        print_final_summary(fov, review_queue=[], output_dir=output_dir)
        if not cfg.no_viewer:
            print("\nLaunching Napari viewer (scout)...", flush=True)
            from roigbiv.pipeline.napari_viewer import display_pipeline_results
            display_pipeline_results(fov, review_queue=[])
        return fov

    # ── Source subtraction (on accept + flag only) ────────────────────────
    if rp.should_run("stage1_subtract"):
        print(fmt.stage_header("1→S", "Source subtraction"), flush=True)
        _check_abort(abort_event)
        t_start = time.time()
        rois_to_subtract = [r for r in fov.rois
                            if r.source_stage == 1
                            and r.gate_outcome in ("accept", "flag")]
        # Spatial profile source: std_S (per-pixel rms activity) rather than
        # mean_S. Under truncated-SVD L+S, mean_S ≈ 0 everywhere so it can't
        # represent the neuron's spatial activity pattern. std_S = rms(S)
        # preserves it. See subtraction.estimate_spatial_profiles docstring.
        with _gpu_section(gpu_lock):
            view_s1, validation, traces = run_source_subtraction(
                fov.residual_view,
                fov.std_S,
                rois_to_subtract,
                output_dir,
                cfg,
                stage_idx=1,
                output_name="residual_S1",
            )
        stage_timings["subtraction_s"] = time.time() - t_start
        fov.residual_view = view_s1

        n_sub_pass = sum(1 for v in validation.values() if v.get("pass"))
        n_sub_total = len(validation)
        if rois_to_subtract:
            for roi, trace in zip(rois_to_subtract, traces):
                roi.trace = trace
        print(fmt.sub_phase(
            f"Subtraction complete. Validation: {n_sub_pass}/{n_sub_total} passed"
        ), flush=True)

        if n_sub_total > 0 and n_sub_pass < n_sub_total:
            warnings.append(
                f"Subtraction validation: {n_sub_total - n_sub_pass} ROIs "
                f"flagged for HITL review "
                f"(anticorr < {cfg.subtract_anticorr_threshold} or "
                f"std_ratio out of [0.3, 3.0])"
            )
        update_manifest(output_dir, "stage1_subtract", cfg, tif_path)
    else:
        # plan_resume already rebuilt fov.residual_view with the Stage 1 layer.
        rois_to_subtract = [r for r in fov.rois
                            if r.source_stage == 1
                            and r.gate_outcome in ("accept", "flag")]
        n_sub_total = len(rois_to_subtract)
        n_sub_pass = _read_subtraction_pass_count(output_dir, 1, fallback=n_sub_total)
        print(fmt.sub_phase(
            f"Resume: stage1 subtraction skipped "
            f"({n_sub_pass}/{n_sub_total} previously passed)"
        ), flush=True)

    next_label = max((r.label_id for r in fov.rois), default=0) + 1

    _mem_snapshot("pre-stage2")

    # ── Stage 2 Suite2p Temporal Detection ────────────────────────────────
    if rp.should_run("stage2") and cfg.enable_stage_2:
        print(fmt.stage_header(2, "Suite2p temporal detection"), flush=True)
        _check_abort(abort_event)
        t_start = time.time()
        with _gpu_section(gpu_lock):
            stage2_candidates = run_stage2(fov, cfg, starting_label_id=next_label)
        _mem_snapshot("stage2 post-run_stage2")
        stage1_for_gate2 = [r for r in fov.rois
                            if r.source_stage == 1
                            and r.gate_outcome in ("accept", "flag")]
        stage2_rois = evaluate_gate2(stage2_candidates, stage1_for_gate2, cfg)
        stage_timings["stage2_detect_s"] = time.time() - t_start

        # Canonicalize: gate 2 already validated real Suite2p footprint
        # geometry above; from here on, accept/flag ROIs persist as a
        # fixed-radius disk at their own centroid (roi_stamp.py).
        for r in stage2_rois:
            if r.gate_outcome in ("accept", "flag"):
                canonicalize(r, cfg.roi_stamp_radius, fov.shape[1:])
        # Cross-stage crowding guard, evaluated against the cumulative pool
        # (already-persisted Stage 1 stamps + these new Stage 2 stamps) before
        # counting/reporting outcomes below, since crowding can demote
        # accept->flag and the report should reflect the true final outcome.
        # fov.rois isn't extended yet, so pass the union explicitly; objects
        # are mutated in place either way.
        resolve_crowding(
            fov.rois + [r for r in stage2_rois if r.gate_outcome != "reject"],
            cfg.roi_stamp_radius,
        )

        n2_det = len(stage2_rois)
        n2_acc = sum(1 for r in stage2_rois if r.gate_outcome == "accept")
        n2_flag = sum(1 for r in stage2_rois if r.gate_outcome == "flag")
        n2_rej = sum(1 for r in stage2_rois if r.gate_outcome == "reject")
        print(fmt.gate_outcome(2, n2_det, n2_acc, n2_flag, n2_rej), flush=True)
        if n2_rej:
            print(fmt.sub_phase(
                f"Gate 2 reject reasons: {_gate_reject_histogram(stage2_rois)}"
            ), flush=True)
        _mem_snapshot("post-stage2+gate2")

        # Save Stage 2 outputs
        (output_dir / "stage2").mkdir(exist_ok=True)
        stage2_mask_img = np.zeros(fov.shape[1:], dtype=np.uint16)
        for r in stage2_rois:
            if r.gate_outcome in ("accept", "flag"):
                stage2_mask_img[r.mask] = r.label_id
        tifffile.imwrite(str(output_dir / "stage2" / "stage2_masks.tif"),
                         stage2_mask_img)
        (output_dir / "stage2" / "stage2_report.json").write_text(json.dumps({
            "detected": n2_det, "accepted": n2_acc, "flagged": n2_flag,
            "rejected": n2_rej,
            "rois": [r.to_serializable() for r in stage2_rois],
        }, indent=2))

        fov.rois.extend([r for r in stage2_rois if r.gate_outcome != "reject"])
        fov.stage_counts["stage2"] = {
            "detected": n2_det, "accepted": n2_acc,
            "flagged": n2_flag, "rejected": n2_rej,
        }
        update_manifest(output_dir, "stage2", cfg, tif_path)
    elif rp.should_run("stage2") and not cfg.enable_stage_2:
        print(fmt.sub_phase("Stage 2: skipped (disabled via cfg.enable_stage_2=False)"),
              flush=True)
        fov.stage_counts["stage2"] = {"detected": 0, "accepted": 0,
                                      "flagged": 0, "rejected": 0}
        update_manifest(output_dir, "stage2", cfg, tif_path, status="skipped")
    else:
        sc = fov.stage_counts.get("stage2", {})
        n2_det = int(sc.get("detected", 0))
        n2_acc = int(sc.get("accepted", 0))
        n2_flag = int(sc.get("flagged", 0))
        n2_rej = int(sc.get("rejected", 0))
        print(fmt.sub_phase(
            f"Resume: stage2 skipped "
            f"({n2_acc} accepted, {n2_flag} flagged, {n2_rej} rejected)"
        ), flush=True)

    # Subtract Stage 2 accept+flag from S₁ → S₂. Only runs when Stage 2
    # itself ran AND a downstream stage will read the residual.
    s2_should_subtract = (
        cfg.enable_stage_2
        and _any_downstream_enabled(cfg, after_stage=2)
        and rp.should_run("stage2_subtract")
    )
    if s2_should_subtract:
        print(fmt.stage_header("2→S", "Source subtraction"), flush=True)
        _check_abort(abort_event)
        t_start = time.time()
        s2_subtract = [r for r in fov.rois
                       if r.source_stage == 2
                       and r.gate_outcome in ("accept", "flag")]
        if s2_subtract:
            std_S1 = compute_std_map(fov.residual_view, chunk=cfg.reconstruct_chunk)
        else:
            std_S1 = fov.std_S  # unused when no ROIs, but pass something valid
        with _gpu_section(gpu_lock):
            view_s2, validation2, traces2 = run_source_subtraction(
                fov.residual_view, std_S1, s2_subtract,
                output_dir, cfg,
                stage_idx=2,
                output_name="residual_S2",
            )
        stage_timings["stage2_subtract_s"] = time.time() - t_start
        fov.residual_view = view_s2
        if s2_subtract:
            for roi, tr in zip(s2_subtract, traces2):
                roi.trace = tr
        n2_sub_pass = sum(1 for v in validation2.values() if v.get("pass"))
        update_manifest(output_dir, "stage2_subtract", cfg, tif_path)
    elif rp.should_run("stage2_subtract"):
        # No downstream consumer (or Stage 2 disabled). The live residual view
        # already represents the deepest subtraction applied; nothing to do.
        if not cfg.enable_stage_2:
            reason = "Stage 2 disabled"
        else:
            reason = "no downstream stage enabled"
        print(fmt.sub_phase(f"Stage 2 subtraction: skipped ({reason})"), flush=True)
        s2_subtract = []
        n2_sub_pass = 0
        update_manifest(output_dir, "stage2_subtract", cfg, tif_path,
                        status="skipped")
    else:
        s2_subtract = [r for r in fov.rois
                       if r.source_stage == 2
                       and r.gate_outcome in ("accept", "flag")]
        n2_sub_pass = _read_subtraction_pass_count(output_dir, 2,
                                                   fallback=len(s2_subtract))
        print(fmt.sub_phase(
            f"Resume: stage2 subtraction skipped "
            f"({n2_sub_pass}/{len(s2_subtract)} previously passed)"
        ), flush=True)

    # Cascade warning (Blindspot 2): Stage 2 accept > 1.5 × Stage 1 accept
    if n_accept > 0 and n2_acc > 1.5 * n_accept:
        warnings.append(
            f"Cascade: Stage 2 accepted {n2_acc} vs Stage 1 {n_accept}. "
            f"May indicate anti-correlation artifacts; review stage2_report.json."
        )

    _mem_snapshot("pre-stage3")

    # ── PMD spatiotemporal denoise (Phase 2, optional; OFF by default) ─────
    # Single insertion point (docs/phase2_pmd_insertion_point.md): swap the live
    # residual view for a dense-backed view over a PMD-denoised memmap. Stage 4
    # inherits it automatically — the Stage-3 source subtraction advances the
    # view via with_source, which carries the dense base forward. L+S, Stage 1,
    # and Stage 2 ran already and are unaffected.
    if cfg.use_pmd_denoise and rp.should_run("stage3"):
        print(fmt.stage_header("PMD", "Spatiotemporal denoise of residual (Stage 3/4 input)"),
              flush=True)
        _check_abort(abort_event)
        t_pmd = time.time()
        with _gpu_section(gpu_lock):
            fov.residual_view = pmd_denoise_view(fov.residual_view, output_dir, cfg)
        stage_timings["pmd_denoise_s"] = time.time() - t_pmd
        _mem_snapshot("post-pmd")

    # ── Stage 3 Template Sweep ────────────────────────────────────────────
    template_bank = build_template_bank(cfg.fs, cfg.tau)
    if rp.should_run("stage3") and cfg.enable_stage_3:
        # Stage 3 reads the live residual view (deepest subtraction applied:
        # S₂ if Stage 2 subtracted, else S₁, else S — the view tracks this).
        print(fmt.stage_header(3, "Template sweep on residual view"), flush=True)
        _check_abort(abort_event)
        next_label = max((r.label_id for r in fov.rois), default=0) + 1
        t_start = time.time()
        with _gpu_section(gpu_lock):
            stage3_candidates = run_stage3(
                fov.residual_view, fov, template_bank, cfg,
                starting_label_id=next_label,
            )
        _mem_snapshot("post-stage3 run")
        prior_for_gate3 = [r for r in fov.rois
                           if r.gate_outcome in ("accept", "flag")]
        stage3_rois = evaluate_gate3(
            stage3_candidates, prior_for_gate3, template_bank, cfg,
        )
        stage_timings["stage3_detect_s"] = time.time() - t_start

        # Canonicalize: Stage 3 candidates are already a disk keyed on
        # spatial_pool_radius (its own trace-extraction pooling radius, an
        # independent detection-time parameter); re-center at roi_stamp_radius
        # so the persisted stamp matches Stages 1/2/4 exactly. Gate 3 already
        # ran above (on the spatial_pool_radius disk); this only changes what
        # subtraction/traces/HITL/registry consume from here on.
        for r in stage3_rois:
            if r.gate_outcome in ("accept", "flag"):
                canonicalize(r, cfg.roi_stamp_radius, fov.shape[1:])
        # Cross-stage crowding guard, evaluated before counting/reporting so
        # the report reflects the true final outcome (see Stage 2 above).
        resolve_crowding(
            fov.rois + [r for r in stage3_rois if r.gate_outcome != "reject"],
            cfg.roi_stamp_radius,
        )

        n3_det = len(stage3_rois)
        n3_acc = sum(1 for r in stage3_rois if r.gate_outcome == "accept")
        n3_flag = sum(1 for r in stage3_rois if r.gate_outcome == "flag")
        n3_rej = sum(1 for r in stage3_rois if r.gate_outcome == "reject")
        print(fmt.gate_outcome(3, n3_det, n3_acc, n3_flag, n3_rej), flush=True)

        # Save Stage 3 outputs
        (output_dir / "stage3").mkdir(exist_ok=True)
        stage3_mask_img = np.zeros(fov.shape[1:], dtype=np.uint16)
        for r in stage3_rois:
            if r.gate_outcome in ("accept", "flag"):
                stage3_mask_img[r.mask] = r.label_id
        tifffile.imwrite(str(output_dir / "stage3" / "stage3_masks.tif"),
                         stage3_mask_img)
        events_blob = [
            {
                "label_id": r.label_id,
                "gate_outcome": r.gate_outcome,
                "event_count": r.event_count,
                "events": r.features.get("events", []),
                "picked_events": r.features.get("picked_events", []),
            }
            for r in stage3_rois
        ]
        np.save(str(output_dir / "stage3" / "stage3_events.npy"),
                np.array(events_blob, dtype=object), allow_pickle=True)
        (output_dir / "stage3" / "stage3_report.json").write_text(json.dumps({
            "detected": n3_det, "accepted": n3_acc, "flagged": n3_flag,
            "rejected": n3_rej,
            "rois": [r.to_serializable() for r in stage3_rois],
        }, indent=2))

        fov.rois.extend([r for r in stage3_rois if r.gate_outcome != "reject"])
        fov.stage_counts["stage3"] = {
            "detected": n3_det, "accepted": n3_acc,
            "flagged": n3_flag, "rejected": n3_rej,
        }
        update_manifest(output_dir, "stage3", cfg, tif_path)
    elif rp.should_run("stage3") and not cfg.enable_stage_3:
        print(fmt.sub_phase("Stage 3: skipped (disabled via cfg.enable_stage_3=False)"),
              flush=True)
        fov.stage_counts["stage3"] = {"detected": 0, "accepted": 0,
                                      "flagged": 0, "rejected": 0}
        update_manifest(output_dir, "stage3", cfg, tif_path, status="skipped")
    else:
        sc = fov.stage_counts.get("stage3", {})
        n3_det = int(sc.get("detected", 0))
        n3_acc = int(sc.get("accepted", 0))
        n3_flag = int(sc.get("flagged", 0))
        n3_rej = int(sc.get("rejected", 0))
        print(fmt.sub_phase(
            f"Resume: stage3 skipped "
            f"({n3_acc} accepted, {n3_flag} flagged, {n3_rej} rejected)"
        ), flush=True)

    # Subtract Stage 3 accept+flag from the latest residual → S₃. Only
    # runs when Stage 3 detected AND Stage 4 will consume the residual.
    s3_should_subtract = (
        cfg.enable_stage_3
        and _any_downstream_enabled(cfg, after_stage=3)
        and rp.should_run("stage3_subtract")
    )
    if s3_should_subtract:
        print(fmt.stage_header("3→S", "Source subtraction on residual view"),
              flush=True)
        _check_abort(abort_event)
        t_start = time.time()
        s3_subtract = [r for r in fov.rois
                       if r.source_stage == 3
                       and r.gate_outcome in ("accept", "flag")]
        if s3_subtract:
            std_in = compute_std_map(fov.residual_view, chunk=cfg.reconstruct_chunk)
        else:
            std_in = fov.std_S
        with _gpu_section(gpu_lock):
            view_s3, validation3, traces3 = run_source_subtraction(
                fov.residual_view, std_in, s3_subtract,
                output_dir, cfg,
                stage_idx=3,
                output_name="residual_S3",
            )
        stage_timings["stage3_subtract_s"] = time.time() - t_start
        fov.residual_view = view_s3
        if s3_subtract:
            for roi, tr in zip(s3_subtract, traces3):
                roi.trace = tr
        n3_sub_pass = sum(1 for v in validation3.values() if v.get("pass"))
        update_manifest(output_dir, "stage3_subtract", cfg, tif_path)
    elif rp.should_run("stage3_subtract"):
        if not cfg.enable_stage_3:
            reason = "Stage 3 disabled"
        else:
            reason = "no downstream stage enabled"
        print(fmt.sub_phase(f"Stage 3 subtraction: skipped ({reason})"), flush=True)
        s3_subtract = []
        n3_sub_pass = 0
        update_manifest(output_dir, "stage3_subtract", cfg, tif_path,
                        status="skipped")
    else:
        s3_subtract = [r for r in fov.rois
                       if r.source_stage == 3
                       and r.gate_outcome in ("accept", "flag")]
        n3_sub_pass = _read_subtraction_pass_count(output_dir, 3,
                                                   fallback=len(s3_subtract))
        print(fmt.sub_phase(
            f"Resume: stage3 subtraction skipped "
            f"({n3_sub_pass}/{len(s3_subtract)} previously passed)"
        ), flush=True)

    # Cascade warning: Stage 3 accept > 0.5 × Stage 2 accept
    if n2_acc > 0 and n3_acc > 0.5 * n2_acc:
        warnings.append(
            f"Cascade: Stage 3 accepted {n3_acc} vs Stage 2 {n2_acc}. "
            f"Expected Stage 3 << Stage 2 given sparse-firing targets."
        )

    # ── Stage 4: Tonic Neuron Search ─────────────────────────────────────
    if rp.should_run("stage4") and cfg.enable_stage_4:
        print(fmt.stage_header(4, "Tonic neuron search on residual view"),
              flush=True)
        _check_abort(abort_event)
        next_label = max((r.label_id for r in fov.rois), default=0) + 1
        t_start = time.time()
        stage4_candidates = run_stage4(fov.residual_view, fov, cfg,
                                       starting_label_id=next_label)
        prior_for_gate4 = [r for r in fov.rois
                           if r.gate_outcome in ("accept", "flag")]
        # Gate 4 uses mean_M (raw morphological mean) for the intensity-
        # percentile check. mean_S ≈ 0 under SVD-based L+S
        # (foundation.py:500) so it can't represent actual brightness; see
        # gate4.py module docstring.
        stage4_rois = evaluate_gate4(
            stage4_candidates, prior_for_gate4,
            fov.mean_M, fov.motion_x, fov.motion_y, cfg,
        )
        stage_timings["stage4_detect_s"] = time.time() - t_start

        # Canonicalize: Gate 4 already validated real connected-component
        # geometry above; Stage 4 is terminal (no subtraction follows), but
        # traces/HITL/registry still consume this mask, so persist the same
        # fixed-radius disk as the other three stages.
        for r in stage4_rois:
            if r.gate_outcome in ("accept", "flag"):
                canonicalize(r, cfg.roi_stamp_radius, fov.shape[1:])
        # Cross-stage crowding guard, evaluated before counting/reporting so
        # the report reflects the true final outcome (see Stage 2 above).
        resolve_crowding(
            fov.rois + [r for r in stage4_rois if r.gate_outcome != "reject"],
            cfg.roi_stamp_radius,
        )

        n4_det = len(stage4_rois)
        n4_flag = sum(1 for r in stage4_rois if r.gate_outcome == "flag")
        n4_rej = sum(1 for r in stage4_rois if r.gate_outcome == "reject")
        print(fmt.gate_outcome(4, n4_det, 0, n4_flag, n4_rej), flush=True)

        _write_stage4_outputs(fov, stage4_rois, cfg, output_dir)
        fov.rois.extend([r for r in stage4_rois if r.gate_outcome != "reject"])
        fov.stage_counts["stage4"] = {
            "detected": n4_det,
            "accepted": 0,         # Gate 4 has no accept tier
            "flagged": n4_flag,
            "rejected": n4_rej,
        }
        update_manifest(output_dir, "stage4", cfg, tif_path)
    elif rp.should_run("stage4") and not cfg.enable_stage_4:
        print(fmt.sub_phase("Stage 4: skipped (disabled via cfg.enable_stage_4=False)"),
              flush=True)
        fov.stage_counts["stage4"] = {"detected": 0, "accepted": 0,
                                      "flagged": 0, "rejected": 0}
        update_manifest(output_dir, "stage4", cfg, tif_path, status="skipped")
    else:
        sc = fov.stage_counts.get("stage4", {})
        n4_det = int(sc.get("detected", 0))
        n4_flag = int(sc.get("flagged", 0))
        n4_rej = int(sc.get("rejected", 0))
        print(fmt.sub_phase(
            f"Resume: stage4 skipped ({n4_flag} requires_review, {n4_rej} rejected)"
        ), flush=True)

    # Cascade warning: Stage 4 detected > Stage 3 detected is a red flag
    if n3_det > 0 and n4_det > n3_det:
        warnings.append(
            f"Cascade: Stage 4 detected {n4_det} vs Stage 3 {n3_det}. "
            f"Tonic candidates should be rarer than sparse-firing candidates; "
            f"review for subtraction artifact propagation (Blindspot 2)."
        )

    # ── Detection complete — print summary + monotonicity check ──────────
    summary_warnings = print_detection_summary(fov)
    warnings.extend(summary_warnings)

    # No residual cleanup needed — the residual chain is virtual (no dense
    # .dat files). The small *.sources.npz sidecars are retained so a later
    # --resume can rebuild the view; extract_all_traces reads data.bin.

    # ── Post-detection: traces, overlap correction, features, classify,
    #    dF/F, deconvolution, HITL, final outputs ─────────────────────────
    # Sort by label_id to lock in the ROI ordering contract: row K of every
    # trace array corresponds to the K-th entry here and to label=label_id in
    # merged_masks.tif.
    fov.rois.sort(key=lambda r: int(r.label_id))

    print(fmt.stage_header("Post", "Trace extraction + QC"), flush=True)
    _check_abort(abort_event)
    t_start = time.time()
    F_raw, F_neu, F_corrected = extract_all_traces(fov, fov.rois, cfg)
    stage_timings["traces_s"] = time.time() - t_start

    t_start = time.time()
    overlap_groups = find_overlap_groups(fov.rois)
    if overlap_groups:
        F_corrected = correct_overlapping_traces(
            fov, fov.rois, overlap_groups, F_corrected, cfg,
        )
        print(fmt.sub_phase(
            f"Overlap correction: re-estimated traces for "
            f"{sum(len(g) for g in overlap_groups)} ROIs "
            f"across {len(overlap_groups)} groups"
        ), flush=True)
    stage_timings["overlap_s"] = time.time() - t_start

    print(fmt.sub_phase("Computing unified QC features"), flush=True)
    t_start = time.time()
    compute_all_features(fov, fov.rois, cfg, template_bank)
    stage_timings["features_s"] = time.time() - t_start

    t_start = time.time()
    classify_all_rois(fov.rois, cfg)
    # Phase 5b: optional tonic accept tier (OFF by default). Post-classification
    # so it can read activity_type; promotes anatomical tonics only. No-op unless
    # cfg.tonic_accept_tier is True.
    apply_tonic_accept_tier(fov.rois, cfg)
    stage_timings["classify_s"] = time.time() - t_start

    t_start = time.time()
    dFF = compute_all_dff(fov.rois, cfg.fs, cfg)
    stage_timings["dff_s"] = time.time() - t_start

    t_start = time.time()
    spks = deconvolve_traces(dFF, fov.rois, cfg.tau, cfg.fs)
    stage_timings["deconv_s"] = time.time() - t_start

    t_start = time.time()
    review_queue = build_review_queue(fov.rois)
    export_hitl_package(fov, fov.rois, review_queue, output_dir)
    subtraction_summary = {
        "stage1": {"n_rois": len(rois_to_subtract),
                   "n_passed": n_sub_pass,
                   "n_failed": n_sub_total - n_sub_pass},
        "stage2": {"n_rois": len(s2_subtract),
                   "n_passed": n2_sub_pass,
                   "n_failed": len(s2_subtract) - n2_sub_pass},
        "stage3": {"n_rois": len(s3_subtract),
                   "n_passed": n3_sub_pass,
                   "n_failed": len(s3_subtract) - n3_sub_pass},
    }
    save_pipeline_outputs(
        fov, fov.rois, F_raw, F_neu, F_corrected, dFF, spks,
        review_queue, overlap_groups, output_dir, cfg,
        stage_timings, warnings, subtraction_summary,
    )
    # Stash trace matrices on the FOVData so downstream callers (registry +
    # traces_io.finalize_fov_bundle) can write the pynapse-facing bundle once
    # the registry decision is known.
    fov.F_raw = F_raw
    fov.F_neu = F_neu
    fov.F_corrected = F_corrected
    stage_timings["hitl_s"] = time.time() - t_start

    if warnings:
        for w in warnings:
            print(fmt.sub_phase(f"WARNING: {w}"), flush=True)

    print_final_summary(fov, review_queue, output_dir)

    # ── Napari viewer ─────────────────────────────────────────────────────
    if not cfg.no_viewer:
        print("\nLaunching Napari viewer...", flush=True)
        from roigbiv.pipeline.napari_viewer import display_pipeline_results
        display_pipeline_results(fov, review_queue=review_queue)

    return fov


_DEFAULT_MODEL = "models/deployed/current_model"
_CHECKPOINTS_GLOB = "models/checkpoints/models/run*_epoch_*"


def _parse_channels(spec: str) -> tuple:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--channels must be 'int,int' (got {spec!r})"
        )
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --channels value: {exc}")


def _parse_overlay_outcomes(spec: str) -> tuple[str, ...]:
    valid = {"accept", "flag", "reject"}
    parts = tuple(p.strip().lower() for p in spec.split(",") if p.strip())
    if not parts:
        raise argparse.ArgumentTypeError(
            "--overlay-outcomes cannot be empty"
        )
    bad = [p for p in parts if p not in valid]
    if bad:
        raise argparse.ArgumentTypeError(
            f"invalid --overlay-outcomes value(s): {bad}. "
            "Valid: accept, flag, reject"
        )
    seen: set[str] = set()
    return tuple(p for p in parts if not (p in seen or seen.add(p)))


def _resolve_model(spec: str) -> str:
    """Resolve the --model argument to a path or Cellpose builtin name.

    ``latest`` walks ``models/checkpoints/models/run*_epoch_*`` and picks the
    newest by mtime. Anything else is returned verbatim.
    """
    import sys as _sys

    project_root = Path(__file__).resolve().parents[2]
    if spec == "latest":
        candidates = sorted(
            project_root.glob(_CHECKPOINTS_GLOB),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            chosen = candidates[-1]
            print(f"Resolved --model latest → {chosen}", flush=True)
            return str(chosen)
        fallback = project_root / _DEFAULT_MODEL
        print(
            f"WARN: no checkpoints under {_CHECKPOINTS_GLOB}; "
            f"falling back to {fallback}",
            file=_sys.stderr,
        )
        return str(fallback)
    return spec


def _resolve_profile_name(
    name: str,
    input_path: Path,
    *,
    shape: "tuple | None" = None,
    metadata: "dict | None" = None,
) -> "tuple[str, OpticsPrior | None]":
    """Resolve ``--profile`` to a concrete profile name (+ the optics prior).

    A concrete ``name`` passes through unchanged (explicit ``--profile``/UI
    dropdown always wins) and returns ``(name, None)``. ``auto`` is resolved from
    the pre-foundation optics prior when a frame ``shape`` is available; with no
    peekable shape it falls back to the legacy ``"grin"`` for byte-identical
    behavior. Resolution happens here — where explicit-vs-default flags are still
    distinguishable — not inside ``get_profile`` (which rejects ``auto``). Raises
    on an unknown concrete name so a typo fails fast.
    """
    if name == AUTO:
        if shape is None:
            return "grin", None   # no peek available → legacy default
        prior = classify_optics_prior(shape, metadata)
        return prior.profile_name, prior
    if not is_profile(name):
        raise ValueError(
            f"unknown profile {name!r}; choose one of {list_profiles()}."
        )
    return name, None


def _peek_optics(tif_path: Path) -> "tuple[tuple | None, dict]":
    """Peek a TIF's (T,H,W) shape + optional pixel-size metadata. Total."""
    from roigbiv.io import read_tiff_optics_metadata, validate_tif
    shape = None
    try:
        _, shape = validate_tif(tif_path)
    except Exception:
        shape = None
    try:
        md = read_tiff_optics_metadata(tif_path)
    except Exception:
        md = {}
    return shape, md


def _classify_workspace_optics(tifs: "list[Path]") -> "OpticsPrior | None":
    """Aggregate a per-FOV optics prior across a directory of TIFs.

    A workspace is usually one imaging session → one lens, so the common case is
    a unanimous high-confidence class. Mixed frame sizes (rare) resolve to the
    safe ``generic`` profile at low confidence — Phase-2 per-FOV scale derivation
    then corrects the numeric gates per FOV regardless of the batch profile.
    Returns ``None`` if nothing could be peeked (→ caller keeps legacy default).
    """
    priors: list[OpticsPrior] = []
    for t in tifs:
        shape, md = _peek_optics(t)
        if shape is not None:
            priors.append(classify_optics_prior(shape, md))
    if not priors:
        return None
    names = {p.profile_name for p in priors}
    if len(names) == 1 and all(p.confidence == "high" for p in priors):
        return priors[0]  # unanimous, confident → representative prior
    if len(names) == 1:
        return priors[0]  # unanimous but low-confidence → carries the confirm flag
    # Mixed optics in one workspace → safest single-channel fallback + confirm.
    max_dim = max(p.max_dim for p in priors)
    return OpticsPrior(
        "generic", "low",
        [f"mixed frame sizes across {len(priors)} FOVs ({sorted(names)}) → "
         f"generic; confirm per-FOV optics"],
        max_dim, None,
    )


def build_auto_workspace_overrides(tifs: "list[Path]", base: dict) -> dict:
    """Build per-FOV overrides for an AUTO-profile workspace run (UI + CLI share).

    Classifies the optics prior across *tifs*, resolves the categorical profile,
    and merges it over *base* (defaults < profile), enabling auto-scale and the
    pause-to-confirm path. *base* carries the non-optics fields (fs, tau,
    backend, stage flags, resume…). Numeric gates are derived post-foundation, so
    nothing here pins them. Returns a flat dict for ``PipelineConfig(**...)``.
    """
    prior = _classify_workspace_optics(list(tifs))
    resolved = prior.profile_name if prior is not None else "grin"
    return merged_overrides(resolved, {
        **base,
        "auto_scale": True,
        "assume_optics": False,
        "explicit_fields": (),
        "auto_adapt": _seed_auto_adapt(prior),
        "no_viewer": True,
    }, [])


def _user_pinned_scale_fields(args: argparse.Namespace) -> tuple:
    """Scale-derived cfg fields the user pinned via CLI (derivation must not clobber).

    Only the scale-derived fields with a direct CLI flag are checkable here; the
    rest have no flag so can never be user-pinned. Recorded on
    ``cfg.explicit_fields`` and consumed by both ``_apply_auto_scale`` and the
    resume fingerprint.
    """
    pinned: list[str] = []
    for arg_name, cfg_key in (
        ("diameter", "diameter"),
        ("min_area", "min_area"),
        ("max_area", "max_area"),
        ("tile_norm_blocksize", "tile_norm_blocksize"),
    ):
        if getattr(args, arg_name, None) is not None:
            pinned.append(cfg_key)
    return tuple(pinned)


def _seed_auto_adapt(prior: "OpticsPrior | None") -> dict:
    """Provenance seed for cfg.auto_adapt from the pre-foundation prior (or {})."""
    if prior is None:
        return {}
    return {"prior": {
        "profile": prior.profile_name,
        "confidence": prior.confidence,
        "reasons": list(prior.reasons),
        "max_dim": prior.max_dim,
        "pixel_size_um": prior.pixel_size_um,
    }}


def _apply_auto_scale(cfg: PipelineConfig, fov: FOVData) -> None:
    """Post-foundation: measure the FOV's soma scale and derive its numeric gates.

    Mutates *cfg* in place when the measurement is trustworthy, honoring
    ``cfg.explicit_fields`` so a user-pinned flag is never overwritten, and
    records provenance in ``cfg.auto_adapt``. Total — a failed or implausible
    measurement leaves the profile fallback untouched and records low confidence
    (Phase 3 turns that into pause-to-confirm).
    """
    from roigbiv.pipeline.optics import (
        derive_scale_params,
        measure_soma_scale,
        scale_plausible,
    )

    scale = measure_soma_scale(fov.mean_M, getattr(fov, "dog_map", None))
    pinned = set(cfg.explicit_fields or ())

    if scale.ok and scale_plausible(scale, cfg.profile):
        applied: dict = {}
        for k, v in derive_scale_params(scale).items():
            if k in pinned:
                continue
            setattr(cfg, k, v)
            applied[k] = v
        # The derived diameter is authoritative and is what the Gate-1 area
        # bounds were derived from. Disable per-image diameter_auto (the generic
        # profile sets it) so Stage 1 doesn't re-estimate from a different image
        # and desync the Cellpose diameter from the area gates.
        if "diameter" in applied:
            cfg.diameter_auto = False
        cfg.auto_adapt = {
            **cfg.auto_adapt,
            "scale_ok": True,
            "soma_diameter_med": round(scale.diameter_med, 1),
            "n_somata": scale.n_somata,
            "applied": applied,
            "pinned": sorted(pinned),
        }
        print(fmt.sub_phase(
            f"Auto-scale: soma d≈{scale.diameter_med:.0f}px (n={scale.n_somata}) "
            f"→ diameter={cfg.diameter}, area=[{cfg.min_area},{cfg.max_area}]"
            + (f"  (pinned: {sorted(pinned)})" if pinned else "")
        ), flush=True)
    else:
        cfg.auto_adapt = {
            **cfg.auto_adapt,
            "scale_ok": False,
            "n_somata": scale.n_somata,
            "reason": "too few/implausible somata to size reliably; kept profile fallback",
        }
        print(fmt.sub_phase(
            f"Auto-scale: measurement not trustworthy (n={scale.n_somata}); "
            f"keeping '{cfg.profile}' defaults "
            f"[diameter={cfg.diameter}, area={cfg.min_area}..{cfg.max_area}]"
        ), flush=True)


def _optics_confirmation_decision(cfg: PipelineConfig) -> "dict | None":
    """Decide whether the run should pause for the user to confirm the optics.

    Pauses only in AUTO mode (a pre-foundation prior was recorded) when either
    the prior is low-confidence (ambiguous frame size / mixed workspace) or the
    post-foundation scale measurement was untrustworthy. Explicit ``--profile``
    runs (no prior) never pause — the user already asserted the optics. Returns
    the confirmation payload, or ``None`` to proceed.
    """
    aa = cfg.auto_adapt or {}
    prior = aa.get("prior")
    if prior is None:
        return None                                  # explicit profile → never pause
    low_conf = prior.get("confidence") == "low"
    scale_failed = auto_scale_active(cfg) and not aa.get("scale_ok", False)
    if not (low_conf or scale_failed):
        return None
    reasons = list(prior.get("reasons", []))
    if scale_failed:
        reasons.append(
            f"post-foundation soma scale unreliable (n={aa.get('n_somata', 0)})"
        )
    return {
        "candidate_profile": cfg.profile,
        "confidence": prior.get("confidence"),
        "scale_ok": bool(aa.get("scale_ok", False)),
        "soma_diameter_med": aa.get("soma_diameter_med"),
        "n_somata": aa.get("n_somata", 0),
        "reasons": reasons,
        "choices": list(list_profiles()),
    }


def _write_optics_confirmation_sentinel(
    output_dir: Path, cfg: PipelineConfig, decision: dict,
) -> Path:
    """Write needs_optics_confirmation.json (mirrors the foundation_only sentinel)."""
    path = output_dir / "needs_optics_confirmation.json"
    path.write_text(json.dumps({
        "mode": "needs_optics_confirmation",
        "candidate_profile": decision["candidate_profile"],
        "confidence": decision.get("confidence"),
        "soma_diameter_med": decision.get("soma_diameter_med"),
        "n_somata": decision.get("n_somata"),
        "reasons": decision.get("reasons", []),
        "choices": decision.get("choices", []),
        "note": ("Auto-adaptation was uncertain. Foundation is complete on "
                 "disk; confirm the optics and re-run with --resume (and "
                 "--profile <choice> to override) to continue from Stage 1, "
                 "or re-run with --assume-optics to proceed on the best guess."),
    }, indent=2))
    return path


def _build_explicit_stage1(args: argparse.Namespace, profile_bundle: dict) -> dict:
    """Stage-1 args the user set explicitly, with historical-CLI-default backfill.

    For each Stage-1 field an explicit flag (value is not ``None``) always wins.
    A field left at ``None`` (omitted) is backfilled from ``STAGE1_CLI_DEFAULTS``
    (or ``_DEFAULT_MODEL`` for the model) ONLY IF the resolved profile does not
    already set it — so the profile can own the field, and the GRIN path stays
    byte-identical to the pre-profile CLI defaults. The returned dict is applied
    *after* the profile bundle by ``merged_overrides`` (precedence
    ``defaults < profile < explicit``).
    """
    out: dict = {}
    # (argparse attr, PipelineConfig field) — model maps to cellpose_model.
    fields = (
        ("diameter", "diameter"),
        ("diameter_auto", "diameter_auto"),
        ("cellprob_threshold", "cellprob_threshold"),
        ("flow_threshold", "flow_threshold"),
        ("channels", "channels"),
        ("model", "cellpose_model"),
    )
    for arg_name, cfg_key in fields:
        val = getattr(args, arg_name, None)
        if val is not None:
            out[cfg_key] = val                      # explicit flag wins
        elif cfg_key not in profile_bundle:         # omitted & profile is silent
            out[cfg_key] = (
                _DEFAULT_MODEL if arg_name == "model"
                else STAGE1_CLI_DEFAULTS[arg_name]
            )
    return out


def _stage_flags(cfg: PipelineConfig) -> dict[int, bool]:
    return {2: cfg.enable_stage_2, 3: cfg.enable_stage_3, 4: cfg.enable_stage_4}


def _build_pipeline_summary(cfg: PipelineConfig, args: argparse.Namespace) -> dict:
    return {
        "model_name": Path(cfg.cellpose_model).name,
        "fs": cfg.fs,
        "tau": cfg.tau,
        "diameter": cfg.diameter,
        "cellprob_threshold": cfg.cellprob_threshold,
        "flow_threshold": cfg.flow_threshold,
        "stage_flags": _stage_flags(cfg),
    }


def main(argv: "list[str] | None" = None) -> int:
    """Entry point for the ``roigbiv-pipeline`` console script.

    Exit codes:
        0  pipeline succeeded; notifications succeeded (or were not requested)
        1  all FOVs failed
        2  bad input (missing path, no TIFs found)
        3  pipeline succeeded but SMTP delivery failed (overlays preserved)
        4  pipeline succeeded but Slack delivery failed (overlays preserved)

    When both email and Slack fail, 3 (email) takes precedence; pipeline (1)
    and bad-input (2) always dominate notifier codes.
    """
    import sys

    parser = argparse.ArgumentParser(
        prog="roigbiv-pipeline",
        description=(
            "ROI G. Biv sequential subtractive detection pipeline. Accepts "
            "a single TIF or a directory of TIFs. A directory input triggers "
            "workspace mode (in-input output/, registry.db, auto-migrate). "
            "All four stages run by default; pass --no-stage-N to skip."
        ),
        epilog=(
            "Email goes through Proton Mail Bridge on 127.0.0.1:1025 by\n"
            "default. One-time Bridge setup is documented in\n"
            "docs/email-notifications.md; copy the per-mailbox password\n"
            "into ROIGBIV_SMTP_PASSWORD.\n\n"
            "Slack posts a summary + overlay PNGs via a bot token in\n"
            "ROIGBIV_SLACK_TOKEN (one-time setup in docs/slack-notifications.md).\n\n"
            "Examples:\n"
            "  roigbiv-pipeline --input stack_mc.tif --fs 7.5\n"
            "  roigbiv-pipeline --input fov_dir/ --fs 7.5 --n-workers 2\n"
            "  roigbiv-pipeline --input fov_dir/ --fs 7.5 \\\n"
            "      --email-to me@example.com --smtp-user me@proton.me\n"
            "  roigbiv-pipeline --input fov_dir/ --fs 7.5 --slack-channel C0123ABCD\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path,
                        help=("Path to a *.tif stack OR a directory of "
                              "stacks. Directory ⇒ workspace mode."))
    parser.add_argument("--fs", required=True, type=float,
                        help=("Effective frame rate (Hz). For 4×-averaged "
                              "30 Hz acquisitions pass 7.5."))
    parser.add_argument("--frame-averaging", dest="frame_averaging",
                        type=int, default=1,
                        help=("Temporal binning factor that produced --fs "
                              "(default 1 = un-averaged). Recorded in "
                              "traces_meta.json for pynapse handoff."))
    parser.add_argument("--model", type=str, default=None,
                        help=("Cellpose model path, built-in name, or "
                              "'latest' (newest mtime in "
                              "models/checkpoints/models/). "
                              f"Default: {_DEFAULT_MODEL}"))
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Indicator decay constant (default: 1.0, GCaMP6s).")
    parser.add_argument("--profile", type=str, default=AUTO,
                        choices=list_profiles(),
                        help=("Acquisition/lens profile bundling Stage-1 + "
                              "Gate-1 defaults (see pipeline/profiles.py). "
                              "'grin' = the 512² baseline (no-op); 'prism' = "
                              "single-channel cyto3 for diffuse PRISM FOVs; "
                              "'generic' = conservative unknown-optics fallback. "
                              "Explicit flags override the profile. Default: "
                              f"{AUTO} (currently resolves to grin)."))
    # NOTE: these Stage-1 flags default to None so the profile resolver can
    # distinguish "omitted" (backfilled from STAGE1_CLI_DEFAULTS / the profile)
    # from "explicitly set" (wins over the profile). See _build_explicit_stage1.
    # Help text states the historical GRIN default that backfill restores.
    parser.add_argument("--diameter", type=int, default=None,
                        help="Cellpose diameter (GRIN default 12).")
    parser.add_argument("--diameter-auto", dest="diameter_auto",
                        action="store_true", default=None,
                        help=("Override --diameter with an image-based "
                              "estimate (DoG peaks + Otsu sizing on mean_M) "
                              "computed at the start of Stage 1. Use when "
                              "frame size or cell scale is unknown."))
    parser.add_argument("--cellprob-threshold", dest="cellprob_threshold",
                        type=float, default=None,
                        help="Cellpose cell-probability threshold (GRIN default -2.0).")
    parser.add_argument("--flow-threshold", dest="flow_threshold",
                        type=float, default=None,
                        help="Cellpose flow threshold (GRIN default 0.6).")
    parser.add_argument("--channels", type=_parse_channels, default=None,
                        help="Cellpose channels as 'cyto,nucleus' (GRIN default 1,2).")
    parser.add_argument("--min-area", dest="min_area", type=int, default=None,
                        help=("Gate 1 minimum ROI area in px² (default 80, "
                              "tuned for 512×512 GRIN). Raise for prism / "
                              "1024×1024 imaging."))
    parser.add_argument("--max-area", dest="max_area", type=int, default=None,
                        help=("Gate 1 maximum ROI area in px² (default 600, "
                              "tuned for 512×512 GRIN). Raise for prism / "
                              "1024×1024 imaging."))
    parser.add_argument("--tile-norm-blocksize", dest="tile_norm_blocksize",
                        type=int, default=None,
                        help=("Cellpose tile-norm blocksize in px (default "
                              "128). Roughly double for 1024×1024 inputs."))
    parser.add_argument("--k", type=int, default=30,
                        help="Background components for L+S separation (default 30).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=("Output directory for single-FOV runs (default: "
                              "inference/pipeline/{stem}/). Ignored in "
                              "workspace mode."))
    parser.add_argument("--no-viewer", action="store_true",
                        help=("Skip Napari display at the end of single-FOV "
                              "runs. Implied when --email-to is set or "
                              "input is a directory."))
    parser.add_argument("--registry", action="store_true",
                        help=("Register/match this FOV against the cross-"
                              "session registry (single-FOV mode only — "
                              "workspace mode always registers)."))
    parser.add_argument("--resume", action="store_true",
                        help=("Skip stages whose outputs already exist in "
                              "the output directory. Refuses if config or "
                              "input has changed."))
    parser.add_argument("--n-workers", dest="n_workers", type=int, default=1,
                        help=("Parallel FOV workers for directory inputs "
                              "(capped at 2). Ignored for single-TIF input."))
    parser.add_argument("--email-to", dest="email_to", type=str, default=None,
                        help=("Recipient email. Omit to skip email entirely. "
                              "On SMTP failure exit code is 3; overlays are "
                              "preserved on disk."))
    parser.add_argument("--smtp-host", dest="smtp_host", type=str,
                        default="127.0.0.1",
                        help=("SMTP host. Default targets a local Proton "
                              "Mail Bridge instance."))
    parser.add_argument("--smtp-port", dest="smtp_port", type=int, default=1025,
                        help="SMTP port (STARTTLS).")
    parser.add_argument("--smtp-user", dest="smtp_user", type=str, default=None,
                        help="SMTP username. Required when --email-to is set.")
    parser.add_argument("--smtp-password-env", dest="smtp_password_env",
                        type=str, default="ROIGBIV_SMTP_PASSWORD",
                        help=("Env-var name holding the SMTP password. "
                              "Never pass the password on the command line."))
    parser.add_argument("--no-email", dest="no_email", action="store_true",
                        help="Skip email even when --email-to is set.")
    parser.add_argument("--slack-channel", dest="slack_channel", type=str,
                        default=None,
                        help=("Slack channel ID (e.g. C0123ABCD) to post a run "
                              "summary + overlay PNGs to. Omit to skip Slack. "
                              "On send failure exit code is 4; overlays are "
                              "preserved on disk. See docs/slack-notifications.md."))
    parser.add_argument("--slack-token-env", dest="slack_token_env",
                        type=str, default="ROIGBIV_SLACK_TOKEN",
                        help=("Env-var name holding the Slack bot token "
                              "(xoxb-…). Never pass the token on the command "
                              "line."))
    parser.add_argument("--no-slack", dest="no_slack", action="store_true",
                        help="Skip Slack even when --slack-channel is set.")
    parser.add_argument("--overlay-outcomes", dest="overlay_outcomes",
                        type=_parse_overlay_outcomes,
                        default=("accept", "flag", "reject"),
                        help=("Comma-separated subset of accept,flag,reject "
                              "controlling which gate outcomes are drawn on "
                              "the overlay PNG. Default: all three (every "
                              "detected ROI). Example: --overlay-outcomes "
                              "accept,flag to hide rejects."))
    # Scout mode — Cellpose-only fast triage (skips SVD/L+S/residual; stops
    # after Stage 1 + Gate 1; no traces/QC/registry; not resumable).
    parser.add_argument("--scout", dest="scout_mode", action="store_true",
                        help=("SCOUT mode: fast FOV-clarity / model triage. "
                              "Computes Cellpose channel 2 as a correlation map "
                              "on the registered movie (no SVD/L+S/residual), "
                              "runs only Stage 1 + Gate 1, writes masks + "
                              "overlay. NOT analysis-grade (no traces, QC, or "
                              "registry; not resumable). Incompatible with "
                              "--no-stage-2/3/4 toggles and --resume."))
    parser.add_argument("--scout-stride", dest="scout_vcorr_stride", type=int,
                        default=1, help=("Frame decimation for scout Vcorr "
                                         "(1 = every frame; only with --scout)."))
    parser.add_argument("--scout-neighbors", dest="scout_vcorr_neighbors",
                        type=int, choices=(4, 8), default=8,
                        help="Vcorr stencil for --scout: 8 (full) or 4. Default 8.")
    # Foundation-only "dry run" — motion correction + SVD/L+S + summary images,
    # then stop before Stage 1 so the corrected FOV can be inspected first.
    parser.add_argument("--foundation-only", dest="foundation_only",
                        action="store_true", default=False,
                        help=("FOUNDATION-ONLY dry run: run stacking + motion "
                              "correction + SVD/L+S + summary images, then STOP "
                              "before Stage 1. Writes summary/*.tif + {stem}_mc.tif "
                              "for FOV preview (no ROI detection / traces / QC / "
                              "registry). A later --resume run (without this flag) "
                              "continues from Stage 1. Incompatible with --scout, "
                              "--resume, and --no-stage-N toggles."))
    parser.add_argument("--auto-scale", dest="auto_scale",
                        action=argparse.BooleanOptionalAction, default=True,
                        help=("Optics auto-scale: after foundation, measure the "
                              "FOV's soma scale and DERIVE the numeric gates "
                              "(areas/diameter/separations/pool radii) from it, "
                              "overriding the profile's hardcoded numbers but "
                              "never an explicit flag. Active on the prism/"
                              "generic profiles only (GRIN stays byte-identical). "
                              "Pass --no-auto-scale to keep the profile's fixed "
                              "gates."))
    parser.add_argument("--assume-optics", dest="assume_optics",
                        action="store_true", default=False,
                        help=("Suppress the optics pause-to-confirm: when "
                              "auto-adaptation is uncertain (ambiguous frame "
                              "size / unreliable soma scale), proceed on the "
                              "best guess instead of stopping for confirmation. "
                              "Use for headless/batch runs that cannot prompt."))
    # Per-stage flags. BooleanOptionalAction gives us --stage-N / --no-stage-N.
    parser.add_argument("--stage-2", dest="enable_stage_2",
                        action=argparse.BooleanOptionalAction, default=None,
                        help=("Run Stage 2 (Suite2p temporal detection). "
                              "Default: enabled. Pass --no-stage-2 to skip."))
    parser.add_argument("--stage-3", dest="enable_stage_3",
                        action=argparse.BooleanOptionalAction, default=None,
                        help=("Run Stage 3 (FFT template sweep for sparse-"
                              "firing cells). Default: enabled. Pass "
                              "--no-stage-3 to skip."))
    parser.add_argument("--stage-4", dest="enable_stage_4",
                        action=argparse.BooleanOptionalAction, default=None,
                        help=("Run Stage 4 (tonic neuron correlation-"
                              "contrast search). Default: enabled. Pass "
                              "--no-stage-4 to skip."))
    parser.add_argument("--subtract-solver", dest="subtract_solver",
                        choices=("ridge", "robust"), default=None,
                        help=("Trace-estimation solver for source subtraction. "
                              "'ridge': GPU-chunked normal equations (default). "
                              "'robust': one-sided Huber IRLS M-estimator "
                              "(experimental, config-gated)."))
    parser.add_argument("--subtract-robust-kappa", dest="subtract_robust_kappa",
                        type=float, default=None,
                        help=("Robustness threshold for --subtract-solver robust "
                              "(default: 0.5 sigma units)."))
    parser.add_argument("--subtract-robust-max-iter", dest="subtract_robust_max_iter",
                        type=int, default=None,
                        help=("Max IRLS iterations for --subtract-solver robust "
                              "(default: 5)."))
    parser.add_argument("--motion-correction", dest="motion_correction_backend",
                        choices=("rowwise-pcc", "phasecorr", "legacy"), default=None,
                        help=("Motion-correction backend. 'phasecorr': Suite2p "
                              "rigid+non-rigid registration (default; robust on "
                              "dim/low-SNR data). 'rowwise-pcc': GPU row-wise "
                              "phase correlation (opt-in; fast but degrades "
                              "low-SNR FOVs). 'legacy': genuine SIMA "
                              "HiddenMarkov2D in the sima-legacy conda env "
                              "(opt-in; CPU, slow — exact legacy-notebook repro; "
                              "build it with bash envs/build_sima_legacy.sh). "
                              "All write a {stem}_mc.tif."))
    parser.add_argument("--mc-sima-env", dest="mc_sima_env", default=None,
                        help=("legacy: conda env hosting SIMA 1.3.2 "
                              "(default 'sima-legacy')."))
    parser.add_argument("--mc-granularity", dest="mc_granularity", default=None,
                        help=("legacy: SIMA HMM2D granularity, 'row' (non-rigid, "
                              "default) or 'frame' (rigid)."))
    parser.add_argument("--pipeline-mode", dest="pipeline_mode",
                        choices=PIPELINE_MODES, default=DEFAULT_PIPELINE_MODE,
                        help=("Pipeline execution mode (ADR-0001 migration switch). "
                              "'cascade_legacy': sequential subtractive detection "
                              "(default; current behavior). 'candidate_union': "
                              "non-destructive candidate-union architecture. "
                              "'candidate_union_with_residual_refinement': candidate "
                              "union plus a residual-refinement pass. "
                              "'benchmark_only': reserved for the benchmark harness. "
                              "Currently inert plumbing only — no stage branches on "
                              f"this yet. Default: {DEFAULT_PIPELINE_MODE}."))
    parser.add_argument("--mc-max-displacement", dest="mc_max_displacement",
                        type=int, default=None,
                        help=("rowwise-pcc: max per-frame/per-strip shift in px "
                              "(default 50)."))
    parser.add_argument("--mc-strip-height", dest="mc_strip_height",
                        type=int, default=None,
                        help=("rowwise-pcc: horizontal strip height in rows for "
                              "the non-rigid step (default 32; try 48 on dim "
                              "1024² prism FOVs)."))
    parser.add_argument("--mc-template-iters", dest="mc_n_template_iters",
                        type=int, default=None,
                        help=("rowwise-pcc: template refinement iterations "
                              "(default 2)."))
    parser.add_argument("--mc-prefilter", dest="mc_prefilter",
                        action=argparse.BooleanOptionalAction, default=None,
                        help=("rowwise-pcc: DoG band-pass on shift-estimation "
                              "inputs that suppresses noise-driven per-row warps "
                              "and holds phasecorr quality parity. Default: on. "
                              "Pass --no-mc-prefilter to measure the unregularized "
                              "regression."))
    parser.add_argument("--mc-smooth-rows", dest="mc_smooth_sigma_rows",
                        type=float, default=None,
                        help=("rowwise-pcc: per-row displacement-field smoothing "
                              "sigma in rows (default 6.0). Higher = stiffer warp "
                              "field, more noise rejection."))
    # phasecorr (Suite2p) registration knobs — see scripts/sweep_suite2p_reg.py.
    parser.add_argument("--mc-block-size", dest="mc_s2p_block_size",
                        type=int, nargs=2, metavar=("H", "W"), default=None,
                        help=("phasecorr: non-rigid block size in px (tuned "
                              "default 64 64). Pass 128 128 for the old default; "
                              "[32,32] over-fits dim FOVs — do not go finer."))
    parser.add_argument("--mc-smooth-sigma-time", dest="mc_s2p_smooth_sigma_time",
                        type=float, default=None,
                        help=("phasecorr: temporal smoothing for shift estimation "
                              "(default 0). Raises SNR on dim frames but can blur "
                              "fast motion; bench showed it net-hurt on prism."))
    parser.add_argument("--mc-1preg", dest="mc_s2p_one_photon_reg",
                        action=argparse.BooleanOptionalAction, default=None,
                        help=("phasecorr: 1-photon-style spatial high-pass before "
                              "registration (Suite2p 1Preg). Improves shift "
                              "estimation on dim/low-contrast GRIN/prism frames. "
                              "Tuned default: ON (load-bearing for legacy parity); "
                              "pass --no-mc-1preg for bright high-SNR 2P data."))
    parser.add_argument("--mc-maxregshift-nr", dest="mc_s2p_maxregshift_nr",
                        type=int, default=None,
                        help=("phasecorr: max non-rigid block shift in px "
                              "(default 5). Raise for larger field-dependent "
                              "deformation."))
    parser.add_argument("--mc-two-step-reg", dest="mc_s2p_two_step_registration",
                        action=argparse.BooleanOptionalAction, default=None,
                        help=("phasecorr: rigid pass then non-rigid (default off). "
                              "Keeps the raw movie for the second pass."))
    parser.add_argument("--mc-preview", dest="mc_preview_enabled",
                        action=argparse.BooleanOptionalAction, default=None,
                        help=("Write the live motion-correction preview sidecar to "
                              "{output}/mc_preview/ (default on). Feeds the Dash "
                              "UI's live view and leaves a scrubbable timeline for "
                              "backend A/B. Pass --no-mc-preview to skip the writes."))
    # ── Denoising configuration ──────────────────────────────────
    parser.add_argument("--denoised-branch", dest="enable_denoised_branch",
                        action=argparse.BooleanOptionalAction, default=None,
                        help=("Enable denoised parallel branch in foundation "
                              "stage. Default: uses PipelineConfig default "
                              "(disabled). Pass --no-denoised-branch to force "
                              "off."))
    parser.add_argument("--denoiser-backend", dest="denoiser_backend",
                        choices=("deepcad_rt", "deepinterpolation", "pmd", "none"),
                        default=None,
                        help=("Denoiser algorithm backend. Default: uses "
                              "PipelineConfig default ('none')."))
    parser.add_argument("--denoiser-model-path", dest="denoiser_model_path",
                        type=Path, default=None,
                        help=("Path to denoiser model checkpoint, required "
                              "when --denoiser-backend is not 'none'."))
    parser.add_argument("--denoised-branch-cache", dest="denoised_branch_cache",
                        type=Path, default=None,
                        help=("Output directory for cached denoised movie "
                              "and intermediate products."))
    parser.add_argument("--validate-denoised-against-raw",
                        dest="validate_denoised_against_raw",
                        action=argparse.BooleanOptionalAction, default=None,
                        help=("Compute validation metrics comparing denoised "
                              "vs raw. Default: uses PipelineConfig default "
                              "(enabled). Pass --no-validate-denoised-against-raw "
                              "to skip."))
    parser.add_argument("--cpu", dest="force_cpu", action="store_true",
                        default=False,
                        help=("Force CPU-only execution for all Torch and "
                              "Cellpose operations. Overrides CUDA "
                              "auto-detection. Useful on non-CUDA machines "
                              "or when GPU memory is unavailable."))
    # Gate 2 area overrides — escape hatch only; the common case derives
    # Gate 2 bounds automatically from --min-area / --max-area (×0.75 / ×0.667).
    parser.add_argument("--gate2-min-area", dest="gate2_min_area", type=int,
                        default=None,
                        help=("Gate 2 minimum ROI area in px² (default: "
                              "0.75 × --min-area when --min-area is set, else "
                              "PipelineConfig default of 60). Override only "
                              "when Suite2p footprints differ significantly "
                              "from Cellpose footprints."))
    parser.add_argument("--gate2-max-area", dest="gate2_max_area", type=int,
                        default=None,
                        help=("Gate 2 maximum ROI area in px² (default: "
                              "0.667 × --max-area when --max-area is set, "
                              "else PipelineConfig default of 400)."))
    # Stage 3 memory budget — the primary knob; pixel-chunk-rows is derived.
    parser.add_argument("--stage3-chunk-budget-gb", dest="stage3_chunk_budget_gb",
                        type=float, default=None,
                        help=("Stage 3 per-chunk peak-memory budget in GB "
                              "(default 1.0). The spatial chunk-row count is "
                              "auto-derived from this budget, T, W, and "
                              "n_fft. Raise on high-RAM machines to speed up "
                              "Stage 3; lower to protect against OOM on "
                              "memory-constrained hosts."))
    parser.add_argument("--stage3-pixel-chunk-rows", dest="stage3_pixel_chunk_rows",
                        type=int, default=None,
                        help=("Stage 3 spatial chunk size in rows (default "
                              "auto from --stage3-chunk-budget-gb). Set "
                              "explicitly only to override the budget "
                              "formula."))
    parser.add_argument("--template-threshold", dest="template_threshold",
                        type=float, default=None,
                        help=("Stage 3 event-detection threshold in σ (default "
                              "6.0). On structured, non-Gaussian residuals "
                              "(e.g. 1024² prism FOVs where GRIN-tuned "
                              "templates match everywhere) 6σ saturates; raise "
                              "(8–10) to suppress spurious events. The event "
                              "accumulator is hard-capped regardless, so a low "
                              "threshold degrades gracefully instead of OOM-ing."))
    parser.add_argument("--stage3-max-events", dest="stage3_max_events",
                        type=int, default=None,
                        help=("Hard cap on retained Stage 3 events (default "
                              "2,000,000). The accumulator prunes to the "
                              "top-N by score, bounding peak host RAM at ~2× "
                              "this count × ~21 B/event regardless of how many "
                              "suprathreshold events the threshold emits."))

    args = parser.parse_args(argv)

    if args.email_to and not args.no_email and not args.smtp_user:
        parser.error("--smtp-user is required when --email-to is set")

    if args.scout_mode:
        if any(v is False for v in (args.enable_stage_2, args.enable_stage_3,
                                    args.enable_stage_4)) or any(
                v is True for v in (args.enable_stage_2, args.enable_stage_3,
                                    args.enable_stage_4)):
            parser.error("--scout already runs only Stage 1; do not combine it "
                         "with --stage-2/3/4 toggles.")
        if args.resume:
            parser.error("--scout runs are not resumable; drop --resume.")
        if args.scout_vcorr_stride < 1:
            parser.error("--scout-stride must be >= 1.")

    if args.foundation_only:
        if args.scout_mode:
            parser.error("--foundation-only and --scout are mutually exclusive "
                         "(both stop the pipeline early at different points).")
        if args.resume:
            parser.error("--foundation-only is itself the dry run; drop --resume "
                         "(re-run later with --resume and without --foundation-only "
                         "to continue from Stage 1).")
        if any(v is not None for v in (args.enable_stage_2, args.enable_stage_3,
                                       args.enable_stage_4)):
            parser.error("--foundation-only already skips all stages; do not "
                         "combine it with --stage-N / --no-stage-N toggles.")

    try:
        input_path = args.input.resolve(strict=True)
    except FileNotFoundError:
        print(f"ERROR: --input path does not exist: {args.input}",
              file=sys.stderr)
        return 2

    # Only resolve an explicit --model; None (omitted) is filled later by the
    # profile resolver / _build_explicit_stage1 backfill.
    if args.model is not None:
        args.model = _resolve_model(args.model)

    # Stage-enable overrides only forwarded when explicitly set on the CLI
    # (BooleanOptionalAction default=None ⇒ "untouched"); otherwise fall
    # through to PipelineConfig's defaults.
    stage_overrides = {
        f"enable_stage_{n}": getattr(args, f"enable_stage_{n}")
        for n in (2, 3, 4)
        if getattr(args, f"enable_stage_{n}") is not None
    }

    if input_path.is_dir():
        return _run_workspace(args, input_path, stage_overrides)
    if input_path.is_file():
        return _run_single(args, input_path, stage_overrides)
    print(f"ERROR: --input is neither a file nor a directory: {input_path}",
          file=sys.stderr)
    return 2


def _run_single(
    args: argparse.Namespace,
    tif_path: Path,
    stage_overrides: dict,
) -> int:
    """Classic single-FOV path. Returns the process exit code."""
    import sys
    import time

    from roigbiv import overlay as _overlay
    from roigbiv.pipeline._email import (
        EmailFOVResult,
        EmailParams,
        fmt_duration,
        send_email,
    )
    from roigbiv.pipeline._slack import SlackParams, send_slack

    # Email or Slack path implies headless: never open Napari.
    email_active = bool(args.email_to and not args.no_email)
    slack_active = bool(args.slack_channel and not args.no_slack)
    no_viewer = args.no_viewer or email_active or slack_active

    if args.force_cpu:
        import os
        os.environ.setdefault("ROIGBIV_ROICAT_DEVICE", "cpu")
        if not os.environ.get("ROIGBIV_ROICAT_ALIGNMENT"):
            os.environ["ROIGBIV_ROICAT_ALIGNMENT"] = "PhaseCorrelation"
            print(
                "WARN: --cpu selected; registry alignment downgraded to "
                "PhaseCorrelation (RoMa is impractical on CPU). "
                "Set ROIGBIV_ROICAT_ALIGNMENT to override.",
                file=sys.stderr,
            )

    solver_overrides = {
        k: v
        for k, v in {
            "subtract_solver": args.subtract_solver,
            "subtract_robust_kappa": args.subtract_robust_kappa,
            "subtract_robust_max_iter": args.subtract_robust_max_iter,
        }.items()
        if v is not None
    }
    mc_overrides = {
        k: v
        for k, v in {
            "motion_correction_backend": args.motion_correction_backend,
            "mc_max_displacement": args.mc_max_displacement,
            "mc_strip_height": args.mc_strip_height,
            "mc_n_template_iters": args.mc_n_template_iters,
            "mc_prefilter": args.mc_prefilter,
            "mc_smooth_sigma_rows": args.mc_smooth_sigma_rows,
            "mc_sima_env": args.mc_sima_env,
            "mc_granularity": args.mc_granularity,
            "mc_s2p_block_size": args.mc_s2p_block_size,
            "mc_s2p_smooth_sigma_time": args.mc_s2p_smooth_sigma_time,
            "mc_s2p_one_photon_reg": args.mc_s2p_one_photon_reg,
            "mc_s2p_maxregshift_nr": args.mc_s2p_maxregshift_nr,
            "mc_s2p_two_step_registration": args.mc_s2p_two_step_registration,
            "mc_preview_enabled": args.mc_preview_enabled,
        }.items()
        if v is not None
    }
    stage1_overrides = {
        k: v
        for k, v in {
            "min_area": args.min_area,
            "max_area": args.max_area,
            "tile_norm_blocksize": args.tile_norm_blocksize,
        }.items()
        if v is not None
    }
    gate2_overrides = _build_gate2_overrides(args)
    stage3_overrides = {
        k: v
        for k, v in {
            "stage3_chunk_budget_bytes": (
                int(args.stage3_chunk_budget_gb * 1e9)
                if args.stage3_chunk_budget_gb is not None else None
            ),
            "stage3_pixel_chunk_rows": args.stage3_pixel_chunk_rows,
            "template_threshold": args.template_threshold,
            "stage3_max_events": args.stage3_max_events,
        }.items()
        if v is not None
    }
    denoising_overrides = {
        k: v
        for k, v in {
            "enable_denoised_branch": args.enable_denoised_branch,
            "denoiser_backend": args.denoiser_backend,
            "denoiser_model_path": args.denoiser_model_path,
            "denoised_branch_cache": args.denoised_branch_cache,
            "validate_denoised_against_raw": args.validate_denoised_against_raw,
        }.items()
        if v is not None
    }
    # Resolve the lens profile, then merge: defaults < profile < explicit flags.
    # _build_explicit_stage1 owns the channels/diameter/cellprob/flow/model
    # backfill; stage1_overrides (min/max/tile) and the other *_overrides are
    # already None-filtered explicit dicts and likewise win over the profile.
    _opt_shape, _opt_md = _peek_optics(tif_path)
    profile_name, optics_prior = _resolve_profile_name(
        args.profile, tif_path, shape=_opt_shape, metadata=_opt_md)
    if optics_prior is not None:
        print(fmt.sub_phase(
            f"Optics auto-profile → {profile_name} ({optics_prior.confidence}): "
            f"{'; '.join(optics_prior.reasons)}"
        ), flush=True)
    explicit_stage1 = _build_explicit_stage1(args, get_profile(profile_name))
    base = {
        "fs": args.fs,
        "frame_averaging": args.frame_averaging,
        "tau": args.tau,
        "k_background": args.k,
        "output_dir": args.output_dir,
        "no_viewer": no_viewer,
        "resume": args.resume,
        "force_cpu": args.force_cpu,
        "scout_mode": args.scout_mode,
        "scout_vcorr_stride": args.scout_vcorr_stride,
        "scout_vcorr_neighbors": args.scout_vcorr_neighbors,
        "foundation_only": args.foundation_only,
        "pipeline_mode": args.pipeline_mode,
        # Scale derivation pairs with auto-resolution: an explicit --profile keeps
        # its tuned gates unless the user opted in. optics_prior is non-None only
        # when the profile was auto-resolved. --no-auto-scale disables it.
        "auto_scale": bool(args.auto_scale) and optics_prior is not None,
        "assume_optics": args.assume_optics,
        "explicit_fields": _user_pinned_scale_fields(args),
        "auto_adapt": _seed_auto_adapt(optics_prior),
    }
    cfg = PipelineConfig(**merged_overrides(
        profile_name, base,
        [explicit_stage1, stage_overrides, solver_overrides, mc_overrides,
         stage1_overrides, gate2_overrides, stage3_overrides,
         denoising_overrides],
    ))

    fov_stem = tif_path.stem.replace("_mc", "")
    print(fmt.fov_banner(tif_path.name, 1, 1), flush=True)
    print(fmt.sub_phase(f"Pipeline mode: {cfg.pipeline_mode}"), flush=True)
    t0 = time.perf_counter()
    try:
        fov = run_pipeline(tif_path, cfg)
    except OpticsConfirmationRequired as need:
        p = need.payload
        print(
            "\n".join([
                "",
                "⏸  Optics confirmation needed — pipeline paused after foundation.",
                f"   Candidate profile: {p['candidate_profile']} "
                f"({p.get('confidence')})",
                *(f"   • {r}" for r in p.get("reasons", [])),
                f"   Sentinel: {need.output_dir / 'needs_optics_confirmation.json'}",
                "   Continue with:",
                f"     roigbiv-pipeline --input {tif_path} --fs {cfg.fs} "
                f"--resume --profile <{'/'.join(p.get('choices', []))}>",
                "   …or proceed on the best guess: add --assume-optics.",
            ]),
            file=sys.stderr, flush=True,
        )
        return 5
    except BaseException as exc:  # noqa: BLE001
        import traceback as _tb
        _tb.print_exc()
        duration = time.perf_counter() - t0
        if email_active or slack_active:
            failure_result = EmailFOVResult(
                tif=tif_path,
                output_dir=args.output_dir or tif_path.parent,
                duration_s=duration,
                error=f"{type(exc).__name__}: {exc}",
                roi_counts={"accept": 0, "flag": 0, "reject": 0},
            )
            summary = _build_pipeline_summary(cfg, args)
            if email_active:
                params = EmailParams(
                    email_to=args.email_to, smtp_host=args.smtp_host,
                    smtp_port=args.smtp_port, smtp_user=args.smtp_user,
                    smtp_password_env=args.smtp_password_env,
                )
                send_email([failure_result], params, summary)
            if slack_active:
                slack_params = SlackParams(
                    channel=args.slack_channel,
                    token_env=args.slack_token_env,
                )
                send_slack([failure_result], slack_params, summary)
        # Pipeline-crash exit (1) dominates any notifier-delivery code.
        return 1
    duration = time.perf_counter() - t0

    report = None
    if args.registry:
        from roigbiv.pipeline.optics import resolved_config_payload
        report = _register_fov_after_pipeline(
            tif_path, fov, resolved_config=resolved_config_payload(cfg))
    _write_traces_bundle(fov, cfg, registry_report=report)

    png_path = None
    try:
        png_path = _overlay.render_overlay(
            fov=fov,
            output_dir=fov.output_dir,
            model_name=Path(cfg.cellpose_model).name,
            fov_stem=fov_stem,
            outcomes=args.overlay_outcomes,
        )
        print(f"Overlay written: {png_path}", flush=True)
    except BaseException as exc:  # noqa: BLE001
        print(f"WARN: overlay render failed for {fov_stem}: {exc}",
              file=sys.stderr)

    counts = {"accept": 0, "flag": 0, "reject": 0}
    for roi in fov.rois:
        counts[roi.gate_outcome] = counts.get(roi.gate_outcome, 0) + 1

    print(fmt.pipeline_complete(fov_stem, duration), flush=True)
    png_name = png_path.name if png_path else "<no overlay>"
    print(fmt.sub_phase(
        f"{tif_path.name}: accept={counts['accept']} flag={counts['flag']} "
        f"reject={counts['reject']}  [{fmt_duration(duration)}]  → {png_name}"
    ), flush=True)

    email_failed = False
    slack_failed = False
    results = [EmailFOVResult(
        tif=tif_path, output_dir=fov.output_dir,
        duration_s=duration, png_path=png_path, roi_counts=counts,
    )]
    summary = _build_pipeline_summary(cfg, args)
    if email_active:
        params = EmailParams(
            email_to=args.email_to, smtp_host=args.smtp_host,
            smtp_port=args.smtp_port, smtp_user=args.smtp_user,
            smtp_password_env=args.smtp_password_env,
        )
        if not send_email(results, params, summary):
            print("Email FAILED — overlay remains on disk.",
                  file=sys.stderr, flush=True)
            email_failed = True
    elif args.no_email:
        print("--no-email set; skipping email dispatch.", flush=True)
    if slack_active:
        slack_params = SlackParams(
            channel=args.slack_channel, token_env=args.slack_token_env,
        )
        if not send_slack(results, slack_params, summary):
            print("Slack FAILED — overlay remains on disk.",
                  file=sys.stderr, flush=True)
            slack_failed = True
    elif args.no_slack:
        print("--no-slack set; skipping Slack dispatch.", flush=True)
    # Email (3) takes precedence over Slack (4) when both fail.
    return 3 if email_failed else (4 if slack_failed else 0)


def _run_workspace(
    args: argparse.Namespace,
    input_path: Path,
    stage_overrides: dict,
) -> int:
    """Workspace path: in-input output/ + registry.db + auto-migrate."""
    import sys

    from roigbiv import overlay as _overlay
    from roigbiv.pipeline._email import (
        EmailFOVResult,
        EmailParams,
        fmt_duration,
        send_email,
    )
    from roigbiv.pipeline._slack import SlackParams, send_slack
    from roigbiv.pipeline.workspace import resolve_workspace, run_with_workspace

    try:
        workspace = resolve_workspace(input_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.force_cpu:
        import os
        os.environ.setdefault("ROIGBIV_ROICAT_DEVICE", "cpu")
        if not os.environ.get("ROIGBIV_ROICAT_ALIGNMENT"):
            os.environ["ROIGBIV_ROICAT_ALIGNMENT"] = "PhaseCorrelation"
            print(
                "WARN: --cpu selected; registry alignment downgraded to "
                "PhaseCorrelation (RoMa is impractical on CPU). "
                "Set ROIGBIV_ROICAT_ALIGNMENT to override.",
                file=sys.stderr,
            )

    ws_solver_overrides = {
        k: v
        for k, v in {
            "subtract_solver": args.subtract_solver,
            "subtract_robust_kappa": args.subtract_robust_kappa,
            "subtract_robust_max_iter": args.subtract_robust_max_iter,
        }.items()
        if v is not None
    }
    ws_mc_overrides = {
        k: v
        for k, v in {
            "motion_correction_backend": args.motion_correction_backend,
            "mc_max_displacement": args.mc_max_displacement,
            "mc_strip_height": args.mc_strip_height,
            "mc_n_template_iters": args.mc_n_template_iters,
            "mc_prefilter": args.mc_prefilter,
            "mc_smooth_sigma_rows": args.mc_smooth_sigma_rows,
            "mc_sima_env": args.mc_sima_env,
            "mc_granularity": args.mc_granularity,
            "mc_s2p_block_size": args.mc_s2p_block_size,
            "mc_s2p_smooth_sigma_time": args.mc_s2p_smooth_sigma_time,
            "mc_s2p_one_photon_reg": args.mc_s2p_one_photon_reg,
            "mc_s2p_maxregshift_nr": args.mc_s2p_maxregshift_nr,
            "mc_s2p_two_step_registration": args.mc_s2p_two_step_registration,
            "mc_preview_enabled": args.mc_preview_enabled,
        }.items()
        if v is not None
    }
    ws_stage1_overrides = {
        k: v
        for k, v in {
            "min_area": args.min_area,
            "max_area": args.max_area,
            "tile_norm_blocksize": args.tile_norm_blocksize,
        }.items()
        if v is not None
    }
    ws_gate2_overrides = _build_gate2_overrides(args)
    ws_stage3_overrides = {
        k: v
        for k, v in {
            "stage3_chunk_budget_bytes": (
                int(args.stage3_chunk_budget_gb * 1e9)
                if args.stage3_chunk_budget_gb is not None else None
            ),
            "stage3_pixel_chunk_rows": args.stage3_pixel_chunk_rows,
            "template_threshold": args.template_threshold,
            "stage3_max_events": args.stage3_max_events,
        }.items()
        if v is not None
    }
    ws_denoising_overrides = {
        k: v
        for k, v in {
            "enable_denoised_branch": args.enable_denoised_branch,
            "denoiser_backend": args.denoiser_backend,
            "denoiser_model_path": args.denoiser_model_path,
            "denoised_branch_cache": args.denoised_branch_cache,
            "validate_denoised_against_raw": args.validate_denoised_against_raw,
        }.items()
        if v is not None
    }
    # Resolve the lens profile and merge (defaults < profile < explicit flags).
    # The merged dict (incl. the resolved "profile" label + backfilled Stage-1
    # fields) becomes the per-FOV cfg overrides consumed by run_with_workspace.
    if args.profile == AUTO:
        optics_prior = _classify_workspace_optics(list(workspace.tifs))
        profile_name = optics_prior.profile_name if optics_prior is not None else "grin"
    else:
        profile_name, optics_prior = _resolve_profile_name(args.profile, input_path)
    if optics_prior is not None:
        print(
            f"Optics auto-profile → {profile_name} ({optics_prior.confidence}): "
            f"{'; '.join(optics_prior.reasons)}",
            flush=True,
        )
    explicit_stage1 = _build_explicit_stage1(args, get_profile(profile_name))
    base = {
        "fs": args.fs,
        "frame_averaging": args.frame_averaging,
        "tau": args.tau,
        "k_background": args.k,
        "no_viewer": True,    # workspace runs are headless
        "resume": args.resume,
        "force_cpu": args.force_cpu,
        "foundation_only": args.foundation_only,
        "pipeline_mode": args.pipeline_mode,
        # Scale derivation pairs with auto-resolution: an explicit --profile keeps
        # its tuned gates unless the user opted in. optics_prior is non-None only
        # when the profile was auto-resolved. --no-auto-scale disables it.
        "auto_scale": bool(args.auto_scale) and optics_prior is not None,
        "assume_optics": args.assume_optics,
        "explicit_fields": _user_pinned_scale_fields(args),
        "auto_adapt": _seed_auto_adapt(optics_prior),
    }
    overrides = merged_overrides(
        profile_name, base,
        [explicit_stage1, stage_overrides, ws_solver_overrides, ws_mc_overrides,
         ws_stage1_overrides, ws_gate2_overrides, ws_stage3_overrides,
         ws_denoising_overrides],
    )

    print(fmt.sub_phase(f"Pipeline mode: {overrides['pipeline_mode']}"), flush=True)

    ws_results = run_with_workspace(
        workspace, overrides,
        log_cb=lambda m: print(m, flush=True),
        resume=args.resume,
        n_workers=args.n_workers,
    )

    model_name = Path(overrides["cellpose_model"]).name
    for wr in ws_results:
        if wr.error is not None or wr.fov is None:
            continue
        fov_stem = wr.tif.stem.replace("_mc", "")
        try:
            wr.png_path = _overlay.render_overlay(
                fov=wr.fov,
                output_dir=wr.output_dir,
                model_name=model_name,
                fov_stem=fov_stem,
                outcomes=args.overlay_outcomes,
            )
        except BaseException as exc:  # noqa: BLE001
            print(f"WARN: overlay render failed for {fov_stem}: {exc}",
                  file=sys.stderr)

    successes = [r for r in ws_results if r.error is None]
    failures = [r for r in ws_results if r.error is not None]
    n_tifs = len(workspace.tifs)
    print(fmt.pipeline_complete(f"{n_tifs} FOV(s)"), flush=True)
    for r in ws_results:
        if r.error is not None:
            print(fmt.sub_phase(f"{r.tif.name}: FAILED — {r.error}"), flush=True)
        else:
            c = r.roi_counts
            png = r.png_path.name if r.png_path else "<no overlay>"
            print(fmt.sub_phase(
                f"{r.tif.name}: accept={c.get('accept', 0)} "
                f"flag={c.get('flag', 0)} reject={c.get('reject', 0)}  "
                f"[{fmt_duration(r.duration_s)}]  → {png}"
            ), flush=True)
    print(fmt.sub_phase(f"{len(successes)} succeeded, {len(failures)} failed."),
          flush=True)

    if not successes:
        return 1

    email_active = bool(args.email_to and not args.no_email)
    slack_active = bool(args.slack_channel and not args.no_slack)
    if not (email_active or slack_active):
        return 0

    # Synthesize a representative cfg for the body summary (workspace
    # configs are per-FOV; the run-level params we want to echo are the
    # profile-merged overrides, which are uniform across FOVs).
    cfg_for_summary = PipelineConfig(**overrides)
    summary = _build_pipeline_summary(cfg_for_summary, args)
    email_results = [
        EmailFOVResult(
            tif=r.tif, output_dir=r.output_dir,
            duration_s=r.duration_s, error=r.error,
            png_path=r.png_path,
            roi_counts=r.roi_counts or {"accept": 0, "flag": 0, "reject": 0},
        )
        for r in ws_results
    ]

    email_failed = False
    slack_failed = False
    if email_active:
        params = EmailParams(
            email_to=args.email_to, smtp_host=args.smtp_host,
            smtp_port=args.smtp_port, smtp_user=args.smtp_user,
            smtp_password_env=args.smtp_password_env,
        )
        if not send_email(email_results, params, summary):
            print("Email FAILED — overlays remain on disk.",
                  file=sys.stderr, flush=True)
            email_failed = True
    if slack_active:
        slack_params = SlackParams(
            channel=args.slack_channel, token_env=args.slack_token_env,
        )
        if not send_slack(email_results, slack_params, summary):
            print("Slack FAILED — overlays remain on disk.",
                  file=sys.stderr, flush=True)
            slack_failed = True

    # Email (3) takes precedence over Slack (4) when both fail.
    return 3 if email_failed else (4 if slack_failed else 0)


def _register_fov_after_pipeline(
    tif_path: Path, fov: FOVData, *, resolved_config: "dict | None" = None,
) -> "dict | None":
    """Call the registry with the live FOVData after `run_pipeline` returns.

    Returns the registry report dict, or ``None`` if the registry call was
    skipped (no mean_M or no non-rejected ROIs).
    """
    from roigbiv.registry import (
        RegistryConfig,
        build_adapter_config,
        build_blob_store,
        build_store,
        load_calibration,
        register_or_match,
    )
    from roigbiv.registry.roicat_adapter import SessionInput

    if fov.mean_M is None:
        print("WARN: fov.mean_M is None; skipping registry.", flush=True)
        return None

    merged_masks = _build_merged_masks(fov)
    if merged_masks is None or int((merged_masks > 0).any()) == 0:
        print("WARN: no non-rejected ROIs on this FOV; skipping registry.", flush=True)
        return None

    stem = Path(tif_path).stem.replace("_mc", "")
    cfg = RegistryConfig.from_env()
    store = build_store(cfg)
    blob_store = build_blob_store(cfg)
    adapter_cfg = build_adapter_config(cfg)
    calibration = load_calibration(cfg)

    query = SessionInput(
        session_key=stem,
        mean_m=np.asarray(fov.mean_M, dtype=np.float32),
        merged_masks=np.asarray(merged_masks, dtype=np.uint16),
    )

    report = register_or_match(
        fov_stem=stem,
        query=query,
        output_dir=fov.output_dir,
        store=store,
        blob_store=blob_store,
        adapter_config=adapter_cfg,
        calibration=calibration,
        accept_threshold=cfg.fov_accept_threshold,
        review_threshold=cfg.fov_review_threshold,
        resolved_config=resolved_config,
    )
    decision = report.get("decision")
    posterior = report.get("fov_posterior") or report.get("fov_sim") or 1.0
    if decision == "new_fov":
        print(f"Registry: minted new fov_id={report['fov_id']} "
              f"({report['n_new_cells']} cells)", flush=True)
    elif decision in ("auto_match", "hash_match"):
        print(f"Registry: {decision} fov_id={report['fov_id']} "
              f"posterior={posterior:.3f} "
              f"matched={report.get('n_matched', 0)} "
              f"new={report.get('n_new', 0)} "
              f"missing={report.get('n_missing', 0)}", flush=True)
    elif decision == "review":
        print(f"Registry: review band (posterior={posterior:.3f}); "
              "no session written — resolve in Streamlit.", flush=True)
    return report


def _write_traces_bundle(
    fov: FOVData,
    cfg: PipelineConfig,
    *,
    registry_report: "dict | None" = None,
) -> None:
    """Write the canonical ``traces/`` bundle for pynapse handoff.

    Called unconditionally after ``run_pipeline`` (and after any registry
    call) so both classic and workspace modes produce ``traces/`` by default.
    No-op if the pipeline produced no traces (empty FOV / extraction skipped).
    """
    from roigbiv.pipeline.traces_io import finalize_fov_bundle

    if fov.F_raw is None or fov.F_neu is None or fov.F_corrected is None:
        print("WARN: no trace matrices on FOVData; skipping traces bundle.",
              flush=True)
        return
    rois_sorted = sorted(fov.rois, key=lambda r: int(r.label_id))
    finalize_fov_bundle(
        rois_sorted,
        fov.F_raw,
        fov.F_neu,
        fov.F_corrected,
        fov.output_dir,
        cfg,
        registry_report=registry_report,
        data_bin_path=fov.data_bin_path,
        fov_shape=tuple(fov.shape),
    )


def _build_merged_masks(fov: FOVData) -> "np.ndarray | None":
    """Reassemble the per-cell label image from the accepted ROIs.

    The registry wants a single uint16 label image where each cell's pixels
    carry its ``local_label_id``. Stitches them back together in the same
    coordinate frame as ``fov.mean_M``.
    """
    if fov.mean_M is None or not fov.rois:
        return None
    Ly, Lx = fov.mean_M.shape
    label_image = np.zeros((Ly, Lx), dtype=np.uint16)
    for roi in fov.rois:
        if getattr(roi, "gate_outcome", None) == "reject":
            continue
        mask = roi.mask
        if mask is None or not mask.any():
            continue
        label_image[mask] = int(roi.label_id)
    return label_image


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
