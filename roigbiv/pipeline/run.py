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
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.pipeline import fmt
from roigbiv.pipeline.types import FOVData, PipelineConfig


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


def _stage_input_residual(fov: FOVData, stage_idx: int) -> Path:
    """Return the latest residual path to feed into Stage ``stage_idx``.

    When intermediate stages are disabled (``cfg.enable_stage_N=False``),
    their subtraction step does not run, so ``fov.residual_S{N}_path`` stays
    ``None``. This walks the chain backward and returns the deepest residual
    that actually exists on disk. Fallback order for Stage 3 is S2 → S1 → S
    (Foundation); for Stage 4 it is S3 → S2 → S1 → S.
    """
    for i in range(stage_idx - 1, 0, -1):
        path = getattr(fov, f"residual_S{i}_path", None)
        if path and Path(path).exists():
            return Path(path)
    if fov.residual_S_path and Path(fov.residual_S_path).exists():
        return Path(fov.residual_S_path)
    raise RuntimeError(
        f"_stage_input_residual: no residual on disk for stage {stage_idx}"
    )


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


def run_pipeline(tif_path: Path, cfg: PipelineConfig, gpu_lock=None) -> FOVData:
    """Run the full four-stage pipeline on a single FOV.

    Writes all outputs to cfg.output_dir and (unless cfg.no_viewer) opens a
    napari viewer at the end.

    If `gpu_lock` is provided (a `multiprocessing.Manager().Lock()` from the
    batch runner), GPU-heavy phases are serialized across concurrent FOV
    workers so the RTX-4060 8 GiB VRAM isn't double-booked. CPU phases
    (Foundation summary images, Stage 4 bandpass, traces, QC) overlap freely.

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
    from roigbiv.pipeline.traces import extract_all_traces
    from roigbiv.pipeline.overlap_correction import (
        find_overlap_groups, correct_overlapping_traces,
    )
    from roigbiv.pipeline.qc_features import compute_all_features
    from roigbiv.pipeline.classify import classify_all_rois
    from roigbiv.pipeline.dff import compute_all_dff
    from roigbiv.pipeline.deconvolution import deconvolve_traces
    from roigbiv.pipeline.hitl import build_review_queue, export_hitl_package
    from roigbiv.pipeline.outputs import save_pipeline_outputs, print_final_summary
    from roigbiv.pipeline.resume import plan_resume, update_manifest

    tif_path = Path(tif_path).resolve()
    if cfg.output_dir is None:
        cfg.output_dir = _default_output_dir(tif_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_timings: dict = {}
    warnings: list[str] = []

    # ── Resume planning ───────────────────────────────────────────────────
    rp = plan_resume(output_dir, tif_path, cfg, enable=cfg.resume)
    if rp.enabled and rp.start_stage != "foundation":
        print(f"Resume: starting from {rp.start_stage} "
              f"(prior outputs in {output_dir.name})", flush=True)

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

    # ── Foundation ────────────────────────────────────────────────────────
    if rp.should_run("foundation"):
        t_start = time.time()
        fov = run_foundation(tif_path, cfg, output_dir)
        stage_timings["foundation_s"] = time.time() - t_start
        update_manifest(output_dir, "foundation", cfg, tif_path)
        print(fmt.sub_phase(
            f"Foundation complete. k_background={fov.k_background}  "
            f"(T={fov.shape[0]}, H={fov.shape[1]}, W={fov.shape[2]})"
        ), flush=True)
    else:
        fov = rp.fov  # populated by plan_resume from disk
        print(fmt.sub_phase(
            f"Resume: foundation skipped "
            f"(T={fov.shape[0]}, H={fov.shape[1]}, W={fov.shape[2]})"
        ), flush=True)

    # ── Stage 1 Cellpose detection ────────────────────────────────────────
    if rp.should_run("stage1"):
        print(fmt.stage_header(1, "Cellpose detection"), flush=True)
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
                fov.mean_M, fov.vcorr_S, cfg,
            )
        stage_timings["stage1_detect_s"] = time.time() - t_start
        n_detected = len(candidates)

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

        n_accept = sum(1 for r in rois if r.gate_outcome == "accept")
        n_flag = sum(1 for r in rois if r.gate_outcome == "flag")
        n_reject = sum(1 for r in rois if r.gate_outcome == "reject")
        print(fmt.gate_outcome(1, n_detected, n_accept, n_flag, n_reject), flush=True)

        # Save Stage 1 mask image (accepted + flagged only — rejects not subtracted)
        stage1_mask_img = np.zeros(fov.mean_S.shape, dtype=np.uint16)
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

    # ── Source subtraction (on accept + flag only) ────────────────────────
    if rp.should_run("stage1_subtract"):
        print(fmt.stage_header("1→S", "Source subtraction"), flush=True)
        t_start = time.time()
        rois_to_subtract = [r for r in fov.rois
                            if r.source_stage == 1
                            and r.gate_outcome in ("accept", "flag")]
        # Spatial profile source: std_S (per-pixel rms activity) rather than
        # mean_S. Under truncated-SVD L+S, mean_S ≈ 0 everywhere so it can't
        # represent the neuron's spatial activity pattern. std_S = rms(S)
        # preserves it. See subtraction.estimate_spatial_profiles docstring.
        with _gpu_section(gpu_lock):
            residual_S1_path, validation, traces = run_source_subtraction(
                fov.residual_S_path,
                fov.shape,
                fov.std_S,
                rois_to_subtract,
                output_dir,
                cfg,
                delete_input=True,
            )
        stage_timings["subtraction_s"] = time.time() - t_start
        fov.residual_S1_path = residual_S1_path

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
        # plan_resume already attached fov.residual_S1_path and validated it.
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

    # ── Stage 2 Suite2p Temporal Detection ────────────────────────────────
    if rp.should_run("stage2") and cfg.enable_stage_2:
        print(fmt.stage_header(2, "Suite2p temporal detection"), flush=True)
        t_start = time.time()
        with _gpu_section(gpu_lock):
            stage2_candidates = run_stage2(fov, cfg, starting_label_id=next_label)
        stage1_for_gate2 = [r for r in fov.rois
                            if r.source_stage == 1
                            and r.gate_outcome in ("accept", "flag")]
        stage2_rois = evaluate_gate2(stage2_candidates, stage1_for_gate2, cfg)
        stage_timings["stage2_detect_s"] = time.time() - t_start

        n2_det = len(stage2_rois)
        n2_acc = sum(1 for r in stage2_rois if r.gate_outcome == "accept")
        n2_flag = sum(1 for r in stage2_rois if r.gate_outcome == "flag")
        n2_rej = sum(1 for r in stage2_rois if r.gate_outcome == "reject")
        print(fmt.gate_outcome(2, n2_det, n2_acc, n2_flag, n2_rej), flush=True)

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
        t_start = time.time()
        s2_subtract = [r for r in fov.rois
                       if r.source_stage == 2
                       and r.gate_outcome in ("accept", "flag")]
        if s2_subtract:
            std_S1 = compute_std_map(fov.residual_S1_path, fov.shape,
                                     chunk=cfg.reconstruct_chunk)
        else:
            std_S1 = fov.std_S  # unused when no ROIs, but pass something valid
        with _gpu_section(gpu_lock):
            residual_S2_path, validation2, traces2 = run_source_subtraction(
                fov.residual_S1_path, fov.shape, std_S1, s2_subtract,
                output_dir, cfg,
                output_name="residual_S2",
                delete_input=True,
            )
        stage_timings["stage2_subtract_s"] = time.time() - t_start
        fov.residual_S2_path = residual_S2_path
        if s2_subtract:
            for roi, tr in zip(s2_subtract, traces2):
                roi.trace = tr
        n2_sub_pass = sum(1 for v in validation2.values() if v.get("pass"))
        update_manifest(output_dir, "stage2_subtract", cfg, tif_path)
    elif rp.should_run("stage2_subtract"):
        # No downstream consumer (or Stage 2 disabled). Stage 3+ readers will
        # fall back to residual_S1 via _stage_input_residual; nothing to do.
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

    # ── Stage 3 Template Sweep ────────────────────────────────────────────
    template_bank = build_template_bank(cfg.fs, cfg.tau)
    if rp.should_run("stage3") and cfg.enable_stage_3:
        # Stage 3 reads the latest residual on disk: S₂ if Stage 2 subtracted,
        # else S₁. _stage_input_residual handles the fallback.
        s3_input_residual = _stage_input_residual(fov, 3)
        print(fmt.stage_header(3, f"Template sweep on {s3_input_residual.name}"),
              flush=True)
        next_label = max((r.label_id for r in fov.rois), default=0) + 1
        t_start = time.time()
        with _gpu_section(gpu_lock):
            stage3_candidates = run_stage3(
                s3_input_residual, fov, template_bank, cfg,
                starting_label_id=next_label,
            )
        prior_for_gate3 = [r for r in fov.rois
                           if r.gate_outcome in ("accept", "flag")]
        stage3_rois = evaluate_gate3(
            stage3_candidates, prior_for_gate3, s3_input_residual,
            fov.shape, template_bank, cfg,
        )
        stage_timings["stage3_detect_s"] = time.time() - t_start

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
        s3_input_residual = _stage_input_residual(fov, 3)
        print(fmt.stage_header("3→S", f"Source subtraction on {s3_input_residual.name}"),
              flush=True)
        t_start = time.time()
        s3_subtract = [r for r in fov.rois
                       if r.source_stage == 3
                       and r.gate_outcome in ("accept", "flag")]
        if s3_subtract:
            std_in = compute_std_map(s3_input_residual, fov.shape,
                                     chunk=cfg.reconstruct_chunk)
        else:
            std_in = fov.std_S
        with _gpu_section(gpu_lock):
            residual_S3_path, validation3, traces3 = run_source_subtraction(
                s3_input_residual, fov.shape, std_in, s3_subtract,
                output_dir, cfg,
                output_name="residual_S3",
                delete_input=True,
            )
        stage_timings["stage3_subtract_s"] = time.time() - t_start
        fov.residual_S3_path = residual_S3_path
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
        s4_input_residual = _stage_input_residual(fov, 4)
        print(fmt.stage_header(4, f"Tonic neuron search on {s4_input_residual.name}"),
              flush=True)
        next_label = max((r.label_id for r in fov.rois), default=0) + 1
        t_start = time.time()
        stage4_candidates = run_stage4(s4_input_residual, fov, cfg,
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

    # Drop the final residual — extract_all_traces reads data.bin (not any
    # residual), so nothing downstream needs it. The deepest residual on disk
    # depends on which stages ran (Stage N's subtraction deletes its input,
    # so only one residual remains).
    for residual_attr in ("residual_S3_path", "residual_S2_path",
                          "residual_S1_path"):
        p = getattr(fov, residual_attr, None)
        if p and Path(p).exists():
            Path(p).unlink()
            meta = Path(p).with_suffix(".meta.json")
            if meta.exists():
                meta.unlink()
            break

    # ── Post-detection: traces, overlap correction, features, classify,
    #    dF/F, deconvolution, HITL, final outputs ─────────────────────────
    # Sort by label_id to lock in the ROI ordering contract: row K of every
    # trace array corresponds to the K-th entry here and to label=label_id in
    # merged_masks.tif.
    fov.rois.sort(key=lambda r: int(r.label_id))

    print(fmt.stage_header("Post", "Trace extraction + QC"), flush=True)
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


def _stage_flags(cfg: PipelineConfig) -> dict[int, bool]:
    return {2: cfg.enable_stage_2, 3: cfg.enable_stage_3, 4: cfg.enable_stage_4}


def _build_pipeline_summary(cfg: PipelineConfig, args: argparse.Namespace) -> dict:
    return {
        "model_name": Path(cfg.cellpose_model).name,
        "fs": cfg.fs,
        "tau": cfg.tau,
        "diameter": args.diameter,
        "cellprob_threshold": args.cellprob_threshold,
        "flow_threshold": args.flow_threshold,
        "stage_flags": _stage_flags(cfg),
    }


def main(argv: "list[str] | None" = None) -> int:
    """Entry point for the ``roigbiv-pipeline`` console script.

    Exit codes:
        0  pipeline succeeded; email succeeded (or was not requested)
        1  all FOVs failed
        2  bad input (missing path, no TIFs found)
        3  pipeline succeeded but SMTP delivery failed (overlays preserved)
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
            "Examples:\n"
            "  roigbiv-pipeline --input stack_mc.tif --fs 7.5\n"
            "  roigbiv-pipeline --input fov_dir/ --fs 7.5 --n-workers 2\n"
            "  roigbiv-pipeline --input fov_dir/ --fs 7.5 \\\n"
            "      --email-to me@example.com --smtp-user me@proton.me\n"
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
    parser.add_argument("--model", type=str, default=_DEFAULT_MODEL,
                        help=("Cellpose model path, built-in name, or "
                              "'latest' (newest mtime in "
                              "models/checkpoints/models/). "
                              f"Default: {_DEFAULT_MODEL}"))
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Indicator decay constant (default: 1.0, GCaMP6s).")
    parser.add_argument("--diameter", type=int, default=12,
                        help="Cellpose diameter (default 12).")
    parser.add_argument("--cellprob-threshold", dest="cellprob_threshold",
                        type=float, default=-2.0,
                        help="Cellpose cell-probability threshold (default -2.0).")
    parser.add_argument("--flow-threshold", dest="flow_threshold",
                        type=float, default=0.6,
                        help="Cellpose flow threshold (default 0.6).")
    parser.add_argument("--channels", type=_parse_channels, default=(1, 2),
                        help="Cellpose channels as 'cyto,nucleus' (default 1,2).")
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
    parser.add_argument("--overlay-outcomes", dest="overlay_outcomes",
                        type=_parse_overlay_outcomes,
                        default=("accept", "flag", "reject"),
                        help=("Comma-separated subset of accept,flag,reject "
                              "controlling which gate outcomes are drawn on "
                              "the overlay PNG. Default: all three (every "
                              "detected ROI). Example: --overlay-outcomes "
                              "accept,flag to hide rejects."))
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
    parser.add_argument("--cpu", dest="force_cpu", action="store_true",
                        default=False,
                        help=("Force CPU-only execution for all Torch and "
                              "Cellpose operations. Overrides CUDA "
                              "auto-detection. Useful on non-CUDA machines "
                              "or when GPU memory is unavailable."))

    args = parser.parse_args(argv)

    if args.email_to and not args.no_email and not args.smtp_user:
        parser.error("--smtp-user is required when --email-to is set")

    try:
        input_path = args.input.resolve(strict=True)
    except FileNotFoundError:
        print(f"ERROR: --input path does not exist: {args.input}",
              file=sys.stderr)
        return 2

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

    # Email path implies headless: never open Napari.
    no_viewer = args.no_viewer or bool(args.email_to and not args.no_email)

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

    cfg = PipelineConfig(
        fs=args.fs,
        frame_averaging=args.frame_averaging,
        tau=args.tau,
        k_background=args.k,
        cellpose_model=args.model,
        diameter=args.diameter,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        channels=args.channels,
        output_dir=args.output_dir,
        no_viewer=no_viewer,
        resume=args.resume,
        force_cpu=args.force_cpu,
        **stage_overrides,
    )

    fov_stem = tif_path.stem.replace("_mc", "")
    print(fmt.fov_banner(tif_path.name, 1, 1), flush=True)
    t0 = time.perf_counter()
    try:
        fov = run_pipeline(tif_path, cfg)
    except BaseException as exc:  # noqa: BLE001
        import traceback as _tb
        _tb.print_exc()
        duration = time.perf_counter() - t0
        if args.email_to and not args.no_email:
            failure_result = EmailFOVResult(
                tif=tif_path,
                output_dir=args.output_dir or tif_path.parent,
                duration_s=duration,
                error=f"{type(exc).__name__}: {exc}",
                roi_counts={"accept": 0, "flag": 0, "reject": 0},
            )
            params = EmailParams(
                email_to=args.email_to, smtp_host=args.smtp_host,
                smtp_port=args.smtp_port, smtp_user=args.smtp_user,
                smtp_password_env=args.smtp_password_env,
            )
            send_email([failure_result], params, _build_pipeline_summary(cfg, args))
        return 1
    duration = time.perf_counter() - t0

    report = None
    if args.registry:
        report = _register_fov_after_pipeline(tif_path, fov)
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

    if args.email_to and not args.no_email:
        params = EmailParams(
            email_to=args.email_to, smtp_host=args.smtp_host,
            smtp_port=args.smtp_port, smtp_user=args.smtp_user,
            smtp_password_env=args.smtp_password_env,
        )
        results = [EmailFOVResult(
            tif=tif_path, output_dir=fov.output_dir,
            duration_s=duration, png_path=png_path, roi_counts=counts,
        )]
        if not send_email(results, params, _build_pipeline_summary(cfg, args)):
            print("Email FAILED — overlay remains on disk.",
                  file=sys.stderr, flush=True)
            return 3
    elif args.no_email:
        print("--no-email set; skipping email dispatch.", flush=True)
    return 0


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

    overrides = {
        "fs": args.fs,
        "frame_averaging": args.frame_averaging,
        "tau": args.tau,
        "k_background": args.k,
        "cellpose_model": args.model,
        "diameter": args.diameter,
        "cellprob_threshold": args.cellprob_threshold,
        "flow_threshold": args.flow_threshold,
        "channels": args.channels,
        "no_viewer": True,    # workspace runs are headless
        "resume": args.resume,
        "force_cpu": args.force_cpu,
        **stage_overrides,
    }

    ws_results = run_with_workspace(
        workspace, overrides,
        log_cb=lambda m: print(m, flush=True),
        resume=args.resume,
        n_workers=args.n_workers,
    )

    model_name = Path(args.model).name
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

    if args.email_to and not args.no_email:
        params = EmailParams(
            email_to=args.email_to, smtp_host=args.smtp_host,
            smtp_port=args.smtp_port, smtp_user=args.smtp_user,
            smtp_password_env=args.smtp_password_env,
        )
        # Synthesize a representative cfg for the body summary (workspace
        # configs are per-FOV; the run-level params we want to echo are the
        # CLI-set overrides, which are uniform across FOVs).
        cfg_for_summary = PipelineConfig(
            fs=args.fs, frame_averaging=args.frame_averaging, tau=args.tau,
            k_background=args.k, cellpose_model=args.model,
            diameter=args.diameter, cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold, channels=args.channels,
            **stage_overrides,
        )
        email_results = [
            EmailFOVResult(
                tif=r.tif, output_dir=r.output_dir,
                duration_s=r.duration_s, error=r.error,
                png_path=r.png_path,
                roi_counts=r.roi_counts or {"accept": 0, "flag": 0, "reject": 0},
            )
            for r in ws_results
        ]
        if not send_email(email_results, params,
                          _build_pipeline_summary(cfg_for_summary, args)):
            print("Email FAILED — overlays remain on disk.",
                  file=sys.stderr, flush=True)
            return 3

    return 0


def _register_fov_after_pipeline(tif_path: Path, fov: FOVData) -> "dict | None":
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
