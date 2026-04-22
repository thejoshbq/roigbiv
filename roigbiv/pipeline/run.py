"""
ROI G. Biv pipeline — CLI entry point + orchestrator.

Wires Foundation → Stage 1 Cellpose → Gate 1 → Source Subtraction → outputs → Napari.

CLI:
  roigbiv-pipeline \\
    --input PATH         (required)  path to *_mc.tif or raw *.tif
    --fs FLOAT           (required)  acquisition frame rate (Hz)
    --model PATH         default: models/deployed/current_model
    --tau FLOAT          default: 1.0
    --k INT              default: 30
    --output-dir PATH    default: inference/pipeline/{stem}/
    --no-viewer          flag

All non-exposed parameters are hardcoded in PipelineConfig per spec §18.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.pipeline.types import FOVData, PipelineConfig


def _gpu_section(gpu_lock):
    """Return `gpu_lock` as a context manager, or a no-op if None.

    Used by `run_pipeline` to serialize GPU-heavy phases (Cellpose, Suite2p,
    Stage 3 FFT, source subtraction) across FOV workers when a shared
    `multiprocessing.Manager().Lock()` is passed by the batch runner. When
    `gpu_lock is None` (single-FOV default), this is a zero-cost no-op.
    """
    return gpu_lock if gpu_lock is not None else nullcontext()


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
    print("\n=== Detection Complete ===", flush=True)
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
        if key == "stage4":
            print(f"{stage_labels[key]}: {flg} requires_review "
                  f"(detected {det}, rejected {rej})", flush=True)
        else:
            print(f"{stage_labels[key]}: {acc} accepted, "
                  f"{flg} flagged, {rej} rejected (detected {det})", flush=True)
        total_kept += acc + flg

    print(f"Total ROIs (accept+flag): {total_kept}", flush=True)

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
            print(f"  {msg}", flush=True)
            warnings.append(msg)
        else:
            print(f"Monotonicity check: {chain}  ✓", flush=True)
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

    tif_path = Path(tif_path).resolve()
    if cfg.output_dir is None:
        cfg.output_dir = _default_output_dir(tif_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_timings = {}
    warnings = []

    # ── Foundation ────────────────────────────────────────────────────────
    t_start = time.time()
    fov = run_foundation(tif_path, cfg, output_dir)
    stage_timings["foundation_s"] = time.time() - t_start
    print(f"Foundation complete. k_background={fov.k_background}  "
          f"(T={fov.shape[0]}, H={fov.shape[1]}, W={fov.shape[2]})", flush=True)

    # ── Stage 1 Cellpose detection ────────────────────────────────────────
    print("\nStage 1: Cellpose detection (dual-channel mean_S + vcorr_S)", flush=True)
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

    # ── Gate 1 morphological validation ───────────────────────────────────
    # Use mean_M for soma-surround contrast (raw brightness levels; mean_S is
    # near-zero and would make the contrast metric numerically unstable).
    t_start = time.time()
    rois = evaluate_gate1(candidates, probs, fov.mean_M, fov.vcorr_S, fov.dog_map,
                          cfg, starting_label_id=1)
    stage_timings["gate1_s"] = time.time() - t_start

    n_accept = sum(1 for r in rois if r.gate_outcome == "accept")
    n_flag = sum(1 for r in rois if r.gate_outcome == "flag")
    n_reject = sum(1 for r in rois if r.gate_outcome == "reject")
    print(f"Stage 1: {n_detected} detected → "
          f"{n_accept} accepted, {n_flag} flagged, {n_reject} rejected", flush=True)

    # Save Stage 1 mask image (accepted + flagged only — rejects not subtracted)
    stage1_mask_img = np.zeros(fov.mean_S.shape, dtype=np.uint16)
    for r in rois:
        if r.gate_outcome in ("accept", "flag"):
            stage1_mask_img[r.mask] = r.label_id
    tifffile.imwrite(str(output_dir / "stage1" / "stage1_masks.tif"), stage1_mask_img)

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

    # ── Source subtraction (on accept + flag only) ────────────────────────
    print("\nSource subtraction: accept+flag ROIs only", flush=True)
    t_start = time.time()
    rois_to_subtract = [r for r in rois if r.gate_outcome in ("accept", "flag")]
    # Spatial profile source: std_S (per-pixel rms activity) rather than mean_S.
    # Under truncated-SVD L+S, mean_S ≈ 0 everywhere so it can't represent
    # the neuron's spatial activity pattern. std_S = rms(S) preserves it.
    # See subtraction.estimate_spatial_profiles docstring for rationale.
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

    # Populate traces on ROI objects and gather counts
    n_sub_pass = sum(1 for v in validation.values() if v.get("pass"))
    n_sub_total = len(validation)
    if rois_to_subtract:
        for roi, trace in zip(rois_to_subtract, traces):
            roi.trace = trace
    print(f"Source subtraction complete. Validation: "
          f"{n_sub_pass}/{n_sub_total} passed", flush=True)

    if n_sub_total > 0 and n_sub_pass < n_sub_total:
        warnings.append(
            f"Subtraction validation: {n_sub_total - n_sub_pass} ROIs flagged for HITL review "
            f"(anticorr < {cfg.subtract_anticorr_threshold} or std_ratio out of [0.3, 3.0])"
        )

    next_label = max((r.label_id for r in fov.rois), default=0) + 1

    # ── Stage 2 Suite2p Temporal Detection ────────────────────────────────
    print("\nStage 2: Suite2p temporal detection", flush=True)
    t_start = time.time()
    with _gpu_section(gpu_lock):
        stage2_candidates = run_stage2(fov, cfg, starting_label_id=next_label)
    stage1_for_gate2 = [r for r in fov.rois
                        if r.source_stage == 1 and r.gate_outcome in ("accept", "flag")]
    stage2_rois = evaluate_gate2(stage2_candidates, stage1_for_gate2, cfg)
    stage_timings["stage2_detect_s"] = time.time() - t_start

    n2_det = len(stage2_rois)
    n2_acc = sum(1 for r in stage2_rois if r.gate_outcome == "accept")
    n2_flag = sum(1 for r in stage2_rois if r.gate_outcome == "flag")
    n2_rej = sum(1 for r in stage2_rois if r.gate_outcome == "reject")
    print(f"Stage 2: {n2_det} detected → {n2_acc} accepted, {n2_flag} flagged, "
          f"{n2_rej} rejected", flush=True)

    # Save Stage 2 outputs
    (output_dir / "stage2").mkdir(exist_ok=True)
    stage2_mask_img = np.zeros(fov.shape[1:], dtype=np.uint16)
    for r in stage2_rois:
        if r.gate_outcome in ("accept", "flag"):
            stage2_mask_img[r.mask] = r.label_id
    tifffile.imwrite(str(output_dir / "stage2" / "stage2_masks.tif"), stage2_mask_img)
    (output_dir / "stage2" / "stage2_report.json").write_text(json.dumps({
        "detected": n2_det, "accepted": n2_acc, "flagged": n2_flag, "rejected": n2_rej,
        "rois": [r.to_serializable() for r in stage2_rois],
    }, indent=2))

    fov.rois.extend([r for r in stage2_rois if r.gate_outcome != "reject"])
    fov.stage_counts["stage2"] = {
        "detected": n2_det, "accepted": n2_acc, "flagged": n2_flag, "rejected": n2_rej,
    }

    # Subtract Stage 2 accept+flag from S₁ → S₂
    print("\nSource subtraction (Stage 2 → S₂): accept+flag ROIs only", flush=True)
    t_start = time.time()
    s2_subtract = [r for r in stage2_rois if r.gate_outcome in ("accept", "flag")]
    if s2_subtract:
        std_S1 = compute_std_map(fov.residual_S1_path, fov.shape,
                                 chunk=cfg.reconstruct_chunk)
    else:
        std_S1 = fov.std_S  # unused when no ROIs, but pass something valid
    with _gpu_section(gpu_lock):
        residual_S2_path, validation2, traces2 = run_source_subtraction(
            fov.residual_S1_path, fov.shape, std_S1, s2_subtract, output_dir, cfg,
            output_name="residual_S2",
            delete_input=True,
        )
    stage_timings["stage2_subtract_s"] = time.time() - t_start
    fov.residual_S2_path = residual_S2_path
    if s2_subtract:
        for roi, tr in zip(s2_subtract, traces2):
            roi.trace = tr
    n2_sub_pass = sum(1 for v in validation2.values() if v.get("pass"))

    # Cascade warning (Blindspot 2): Stage 2 accept > 1.5 × Stage 1 accept
    if n_accept > 0 and n2_acc > 1.5 * n_accept:
        warnings.append(
            f"Cascade: Stage 2 accepted {n2_acc} vs Stage 1 {n_accept}. "
            f"May indicate anti-correlation artifacts; review stage2_report.json."
        )

    # ── Stage 3 Template Sweep ────────────────────────────────────────────
    print("\nStage 3: template sweep on S₂", flush=True)
    next_label = max((r.label_id for r in fov.rois), default=0) + 1
    t_start = time.time()
    template_bank = build_template_bank(cfg.fs, cfg.tau)
    with _gpu_section(gpu_lock):
        stage3_candidates = run_stage3(
            residual_S2_path, fov, template_bank, cfg, starting_label_id=next_label,
        )
    prior_for_gate3 = [r for r in fov.rois
                       if r.gate_outcome in ("accept", "flag")]
    stage3_rois = evaluate_gate3(
        stage3_candidates, prior_for_gate3, residual_S2_path, fov.shape,
        template_bank, cfg,
    )
    stage_timings["stage3_detect_s"] = time.time() - t_start

    n3_det = len(stage3_rois)
    n3_acc = sum(1 for r in stage3_rois if r.gate_outcome == "accept")
    n3_flag = sum(1 for r in stage3_rois if r.gate_outcome == "flag")
    n3_rej = sum(1 for r in stage3_rois if r.gate_outcome == "reject")
    print(f"Stage 3: {n3_det} candidates → {n3_acc} accepted, {n3_flag} flagged, "
          f"{n3_rej} rejected", flush=True)

    # Save Stage 3 outputs
    (output_dir / "stage3").mkdir(exist_ok=True)
    stage3_mask_img = np.zeros(fov.shape[1:], dtype=np.uint16)
    for r in stage3_rois:
        if r.gate_outcome in ("accept", "flag"):
            stage3_mask_img[r.mask] = r.label_id
    tifffile.imwrite(str(output_dir / "stage3" / "stage3_masks.tif"), stage3_mask_img)
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
        "detected": n3_det, "accepted": n3_acc, "flagged": n3_flag, "rejected": n3_rej,
        "rois": [r.to_serializable() for r in stage3_rois],
    }, indent=2))

    fov.rois.extend([r for r in stage3_rois if r.gate_outcome != "reject"])
    fov.stage_counts["stage3"] = {
        "detected": n3_det, "accepted": n3_acc, "flagged": n3_flag, "rejected": n3_rej,
    }

    # Subtract Stage 3 accept+flag from S₂ → S₃ (handoff for Phase 1E)
    print("\nSource subtraction (Stage 3 → S₃): accept+flag ROIs only", flush=True)
    t_start = time.time()
    s3_subtract = [r for r in stage3_rois if r.gate_outcome in ("accept", "flag")]
    if s3_subtract:
        std_S2 = compute_std_map(residual_S2_path, fov.shape,
                                 chunk=cfg.reconstruct_chunk)
    else:
        std_S2 = fov.std_S
    with _gpu_section(gpu_lock):
        residual_S3_path, validation3, traces3 = run_source_subtraction(
            residual_S2_path, fov.shape, std_S2, s3_subtract, output_dir, cfg,
            output_name="residual_S3",
            delete_input=True,
        )
    stage_timings["stage3_subtract_s"] = time.time() - t_start
    fov.residual_S3_path = residual_S3_path
    if s3_subtract:
        for roi, tr in zip(s3_subtract, traces3):
            roi.trace = tr
    n3_sub_pass = sum(1 for v in validation3.values() if v.get("pass"))

    # Cascade warning: Stage 3 accept > 0.5 × Stage 2 accept
    if n2_acc > 0 and n3_acc > 0.5 * n2_acc:
        warnings.append(
            f"Cascade: Stage 3 accepted {n3_acc} vs Stage 2 {n2_acc}. "
            f"Expected Stage 3 << Stage 2 given sparse-firing targets."
        )

    # ── Stage 4: Tonic Neuron Search ─────────────────────────────────────
    print("\nStage 4: multi-scale bandpass + correlation contrast on S₃", flush=True)
    next_label = max((r.label_id for r in fov.rois), default=0) + 1
    t_start = time.time()
    stage4_candidates = run_stage4(fov.residual_S3_path, fov, cfg,
                                   starting_label_id=next_label)
    prior_for_gate4 = [r for r in fov.rois if r.gate_outcome in ("accept", "flag")]
    # Gate 4 uses mean_M (raw morphological mean) for the intensity-percentile
    # check. mean_S ≈ 0 under SVD-based L+S (foundation.py:500) so it can't
    # represent actual brightness; see gate4.py module docstring.
    stage4_rois = evaluate_gate4(
        stage4_candidates, prior_for_gate4,
        fov.mean_M, fov.motion_x, fov.motion_y, cfg,
    )
    stage_timings["stage4_detect_s"] = time.time() - t_start

    n4_det = len(stage4_rois)
    n4_flag = sum(1 for r in stage4_rois if r.gate_outcome == "flag")
    n4_rej = sum(1 for r in stage4_rois if r.gate_outcome == "reject")
    print(f"Stage 4: {n4_det} candidates → {n4_flag} requires_review, "
          f"{n4_rej} rejected (no accept tier by design)", flush=True)

    _write_stage4_outputs(fov, stage4_rois, cfg, output_dir)
    fov.rois.extend([r for r in stage4_rois if r.gate_outcome != "reject"])
    fov.stage_counts["stage4"] = {
        "detected": n4_det,
        "accepted": 0,         # Gate 4 has no accept tier
        "flagged": n4_flag,
        "rejected": n4_rej,
    }

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

    # Drop the final residual — S₃ is no longer read by anything downstream
    # (extract_all_traces reads data.bin, not residuals). Keeps permanent
    # footprint at ~1 movie's worth of bytes instead of ~4.
    if fov.residual_S3_path and Path(fov.residual_S3_path).exists():
        Path(fov.residual_S3_path).unlink()

    # ── Post-detection: traces, overlap correction, features, classify,
    #    dF/F, deconvolution, HITL, final outputs ─────────────────────────
    # Sort by label_id to lock in the ROI ordering contract: row K of every
    # trace array corresponds to the K-th entry here and to label=label_id in
    # merged_masks.tif.
    fov.rois.sort(key=lambda r: int(r.label_id))

    print("\nTrace extraction from registered movie (data.bin)", flush=True)
    t_start = time.time()
    F_raw, F_neu, F_corrected = extract_all_traces(fov, fov.rois, cfg)
    stage_timings["traces_s"] = time.time() - t_start

    t_start = time.time()
    overlap_groups = find_overlap_groups(fov.rois)
    if overlap_groups:
        F_corrected = correct_overlapping_traces(
            fov, fov.rois, overlap_groups, F_corrected, cfg,
        )
        print(f"Overlap correction: re-estimated traces for "
              f"{sum(len(g) for g in overlap_groups)} ROIs "
              f"across {len(overlap_groups)} groups", flush=True)
    stage_timings["overlap_s"] = time.time() - t_start

    print("Computing unified QC features", flush=True)
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
    stage_timings["hitl_s"] = time.time() - t_start

    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}", flush=True)

    print_final_summary(fov, review_queue, output_dir)

    # ── Napari viewer ─────────────────────────────────────────────────────
    if not cfg.no_viewer:
        print("\nLaunching Napari viewer...", flush=True)
        from roigbiv.pipeline.napari_viewer import display_pipeline_results
        display_pipeline_results(fov, review_queue=review_queue)

    return fov


def main():
    parser = argparse.ArgumentParser(
        prog="roigbiv-pipeline",
        description=(
            "ROI G. Biv sequential subtractive detection pipeline "
            "(Foundation + Stage 1 + Gate 1 + Source Subtraction)."
        ),
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to raw *.tif or motion-corrected *_mc.tif stack")
    parser.add_argument("--fs", required=True, type=float,
                        help="Acquisition frame rate (Hz)")
    parser.add_argument("--model", type=str, default="models/deployed/current_model",
                        help="Cellpose model path or built-in name (default: deployed fine-tune)")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Indicator decay constant (default: 1.0, GCaMP6s)")
    parser.add_argument("--k", type=int, default=30,
                        help="Background components for L+S separation (default: 30)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: inference/pipeline/{stem}/)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Skip Napari display at the end")
    parser.add_argument("--registry", action="store_true",
                        help=("Register/match this FOV against the cross-session "
                              "registry (SQLite under inference/registry.db by "
                              "default; override with ROIGBIV_REGISTRY_DSN)."))
    parser.add_argument("--workspace", action="store_true",
                        help=("Run in workspace mode: outputs go under "
                              "<input_root>/output/{stem}/, the registry lives "
                              "in <input_root>/registry.db, and alembic "
                              "upgrade + backfill are performed automatically. "
                              "Overrides --output-dir and --registry."))
    args = parser.parse_args()

    if args.workspace:
        from roigbiv.pipeline.workspace import resolve_workspace, run_with_workspace

        workspace = resolve_workspace(args.input)
        overrides = {
            "fs": args.fs,
            "tau": args.tau,
            "k_background": args.k,
            "cellpose_model": args.model,
            "no_viewer": True,   # workspace flow is non-interactive
        }
        results = run_with_workspace(
            workspace, overrides, log_cb=lambda m: print(m, flush=True),
        )
        failures = [r for r in results if r.error]
        if failures:
            raise SystemExit(1)
        return

    cfg = PipelineConfig(
        fs=args.fs,
        tau=args.tau,
        k_background=args.k,
        cellpose_model=args.model,
        output_dir=args.output_dir,
        no_viewer=args.no_viewer,
    )

    fov = run_pipeline(args.input, cfg)

    if args.registry:
        _register_fov_after_pipeline(args.input, fov)


def _register_fov_after_pipeline(tif_path: Path, fov: FOVData) -> None:
    """Call the registry with the live FOVData after `run_pipeline` returns."""
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
        return

    merged_masks = _build_merged_masks(fov)
    if merged_masks is None or int((merged_masks > 0).any()) == 0:
        print("WARN: no non-rejected ROIs on this FOV; skipping registry.", flush=True)
        return

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
    main()
