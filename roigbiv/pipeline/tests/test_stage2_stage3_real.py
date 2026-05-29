"""
Scoped end-to-end test of Stages 2 + 3 on an existing Phase 1B output directory.

Approach:
  1. Reconstruct FOVData from the previous session's pipeline outputs
     (residual_S1.dat + summary TIFs + roi_metadata.json + stage1_masks.tif).
  2. Re-extract Stage 1 traces from residual_S.dat using each mask (proxy for
     the in-memory traces that weren't persisted by to_serializable).
  3. Run Stage 2 → Gate 2 → subtract → residual_S2.dat.
  4. Run Stage 3 → Gate 3 → subtract → residual_S3.dat.
  5. Report counts, timing, and validation stats.

Run:
    conda run -n roigbiv python -m roigbiv.pipeline.tests.test_stage2_stage3_real
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.pipeline.gate2 import evaluate_gate2
from roigbiv.pipeline.gate3 import evaluate_gate3
from roigbiv.pipeline.stage2 import run_stage2, extract_traces_from_residual
from roigbiv.pipeline.stage3 import run_stage3
from roigbiv.pipeline.stage3_templates import build_template_bank
from roigbiv.pipeline.subtraction import compute_std_map, run_source_subtraction
from roigbiv.pipeline.types import FOVData, PipelineConfig, ROI


OUTPUT_DIR = Path("inference/pipeline/T1_230201_PrL-NAc-G6-6F_HI-D1_FOV1_PRE-000")


def _load_summary(name: str) -> np.ndarray:
    p = OUTPUT_DIR / "summary" / f"{name}.tif"
    return tifffile.imread(str(p))


def _load_shape() -> tuple:
    meta = json.loads((OUTPUT_DIR / "residual_S1.meta.json").read_text())
    return tuple(meta["shape"])


def _reconstruct_stage1_rois(
    roi_meta_path: Path,
    stage1_masks_path: Path,
    residual_S_path: Path,
    shape: tuple,
    chunk: int = 500,
) -> list[ROI]:
    """Rebuild ROI objects from on-disk metadata + labeled mask image.

    Re-extract traces from residual_S.dat (pre-Stage-1 subtraction) since
    traces were stripped by to_serializable during Phase 1B.
    """
    roi_meta = json.loads(roi_meta_path.read_text())
    labels = tifffile.imread(str(stage1_masks_path))
    rois: list[ROI] = []
    masks_for_trace = []
    masks_idx = []
    for i, entry in enumerate(roi_meta):
        lid = int(entry["label_id"])
        mask = labels == lid
        if not mask.any():
            # Rejected ROIs aren't in the mask image; use a dummy mask
            continue
        roi = ROI(
            mask=mask,
            label_id=lid,
            source_stage=int(entry["source_stage"]),
            confidence=entry["confidence"],
            gate_outcome=entry["gate_outcome"],
            area=int(entry["area"]),
            solidity=float(entry["solidity"]),
            eccentricity=float(entry["eccentricity"]),
            nuclear_shadow_score=float(entry["nuclear_shadow_score"]),
            soma_surround_contrast=float(entry["soma_surround_contrast"]),
            cellpose_prob=entry.get("cellpose_prob"),
            features=entry.get("features", {}),
            gate_reasons=entry.get("gate_reasons", []),
        )
        rois.append(roi)
        masks_for_trace.append(mask)
        masks_idx.append(len(rois) - 1)

    if masks_for_trace:
        print(f"  re-extracting {len(masks_for_trace)} Stage 1 traces "
              f"from {residual_S_path.name} ...", flush=True)
        t0 = time.time()
        traces = extract_traces_from_residual(
            residual_S_path, shape, masks_for_trace, chunk=chunk,
        )
        print(f"    done in {time.time()-t0:.1f}s", flush=True)
        for idx, tr in zip(masks_idx, traces):
            rois[idx].trace = tr
    return rois


def main():
    print(f"[harness] Loading Phase 1B state from {OUTPUT_DIR}", flush=True)
    shape = _load_shape()
    T, H, W = shape
    print(f"  shape = {shape}", flush=True)

    mean_M = _load_summary("mean_M")
    mean_S = _load_summary("mean_S")
    max_S = _load_summary("max_S")
    std_S = _load_summary("std_S")
    vcorr_S = _load_summary("vcorr_S")
    dog_map = _load_summary("dog_map")
    mean_L = _load_summary("mean_L")

    # Data bin lives at {OUTPUT_DIR}/{stem}/suite2p/plane0/data.bin
    stem = OUTPUT_DIR.name
    data_bin_path = OUTPUT_DIR / stem / "suite2p" / "plane0" / "data.bin"
    assert data_bin_path.exists(), f"missing data.bin at {data_bin_path}"

    residual_S_path = OUTPUT_DIR / "residual_S.dat"
    residual_S1_path = OUTPUT_DIR / "residual_S1.dat"

    # Reconstruct Stage 1 ROIs with freshly-extracted traces
    rois = _reconstruct_stage1_rois(
        OUTPUT_DIR / "roi_metadata.json",
        OUTPUT_DIR / "stage1" / "stage1_masks.tif",
        residual_S_path,
        shape,
    )
    n_accept = sum(1 for r in rois if r.gate_outcome == "accept")
    n_flag = sum(1 for r in rois if r.gate_outcome == "flag")
    print(f"  loaded {len(rois)} Stage 1 ROIs "
          f"({n_accept} accepted + {n_flag} flagged with traces)", flush=True)

    fov = FOVData(
        raw_path=Path("data/raw/T1_230201_PrL-NAc-G6-6F_HI-D1_FOV1_PRE-000_mc.tif"),
        output_dir=OUTPUT_DIR,
        data_bin_path=data_bin_path,
        shape=shape,
        residual_S_path=residual_S_path,
        residual_S1_path=residual_S1_path,
        mean_M=mean_M, mean_S=mean_S, max_S=max_S, std_S=std_S,
        vcorr_S=vcorr_S, dog_map=dog_map, mean_L=mean_L,
        k_background=30,
        rois=rois,
    )

    cfg = PipelineConfig(fs=30.0, tau=1.0)

    # FREE DISK: once traces are reconstructed, we can drop residual_S.dat
    # (Stage 1 is done; it's no longer needed)
    print(f"[harness] Freeing disk: deleting residual_S.dat "
          f"({residual_S_path.stat().st_size / 1e9:.2f} GB)", flush=True)
    residual_S_path.unlink()
    fov.residual_S_path = residual_S1_path  # point to what's still on disk

    # ── Stage 2 ───────────────────────────────────────────────────────────
    print("\n[harness] Stage 2: Suite2p temporal detection", flush=True)
    t0 = time.time()
    stage2_candidates = run_stage2(
        fov, cfg, starting_label_id=max(r.label_id for r in rois) + 1,
    )
    stage1_for_gate2 = [r for r in rois if r.gate_outcome in ("accept", "flag")]
    stage2_rois = evaluate_gate2(stage2_candidates, stage1_for_gate2, cfg)
    n2_det = len(stage2_rois)
    n2_acc = sum(1 for r in stage2_rois if r.gate_outcome == "accept")
    n2_flag = sum(1 for r in stage2_rois if r.gate_outcome == "flag")
    n2_rej = sum(1 for r in stage2_rois if r.gate_outcome == "reject")
    print(f"  Stage 2: {n2_det} detected → {n2_acc} acc, {n2_flag} flag, {n2_rej} rej "
          f"in {time.time()-t0:.1f}s", flush=True)

    # Subtract Stage 2 → S₂
    print("\n[harness] Source subtraction (Stage 2 → S₂)", flush=True)
    t0 = time.time()
    s2_subtract = [r for r in stage2_rois if r.gate_outcome in ("accept", "flag")]
    if s2_subtract:
        std_S1 = compute_std_map(residual_S1_path, shape, chunk=500)
        residual_S2_path, val2, traces2 = run_source_subtraction(
            residual_S1_path, shape, std_S1, s2_subtract, OUTPUT_DIR, cfg,
            output_name="residual_S2",
        )
        for roi, tr in zip(s2_subtract, traces2):
            roi.trace = tr
        n2_pass = sum(1 for v in val2.values() if v.get("pass"))
        print(f"  S₂ written; {n2_pass}/{len(s2_subtract)} subtraction passes "
              f"in {time.time()-t0:.1f}s", flush=True)
        fov.residual_S2_path = residual_S2_path
        # FREE DISK: S1 no longer needed after S2 is written
        print(f"  freeing disk: deleting residual_S1.dat", flush=True)
        residual_S1_path.unlink()
    else:
        print("  no accept/flag ROIs to subtract; skipping", flush=True)
        residual_S2_path = residual_S1_path
        fov.residual_S2_path = residual_S2_path

    fov.rois.extend([r for r in stage2_rois if r.gate_outcome != "reject"])

    # ── Stage 3 ───────────────────────────────────────────────────────────
    print("\n[harness] Stage 3: template sweep on S₂", flush=True)
    t0 = time.time()
    template_bank = build_template_bank(cfg.fs, cfg.tau)
    next_label = max((r.label_id for r in fov.rois), default=0) + 1
    stage3_candidates = run_stage3(
        fov.residual_S2_path, fov, template_bank, cfg, starting_label_id=next_label,
    )
    prior_for_gate3 = [r for r in fov.rois if r.gate_outcome in ("accept", "flag")]
    stage3_rois = evaluate_gate3(
        stage3_candidates, prior_for_gate3, fov.residual_S2_path, shape,
        template_bank, cfg,
    )
    n3_det = len(stage3_rois)
    n3_acc = sum(1 for r in stage3_rois if r.gate_outcome == "accept")
    n3_flag = sum(1 for r in stage3_rois if r.gate_outcome == "flag")
    n3_rej = sum(1 for r in stage3_rois if r.gate_outcome == "reject")
    print(f"  Stage 3: {n3_det} candidates → {n3_acc} acc, {n3_flag} flag, {n3_rej} rej "
          f"in {time.time()-t0:.1f}s", flush=True)

    # Subtract Stage 3 → S₃
    print("\n[harness] Source subtraction (Stage 3 → S₃)", flush=True)
    t0 = time.time()
    s3_subtract = [r for r in stage3_rois if r.gate_outcome in ("accept", "flag")]
    if s3_subtract:
        std_S2 = compute_std_map(fov.residual_S2_path, shape, chunk=500)
        residual_S3_path, val3, traces3 = run_source_subtraction(
            fov.residual_S2_path, shape, std_S2, s3_subtract, OUTPUT_DIR, cfg,
            output_name="residual_S3",
        )
        n3_pass = sum(1 for v in val3.values() if v.get("pass"))
        print(f"  S₃ written; {n3_pass}/{len(s3_subtract)} subtraction passes "
              f"in {time.time()-t0:.1f}s", flush=True)
        # FREE: S2 no longer needed
        print(f"  freeing disk: deleting residual_S2.dat", flush=True)
        fov.residual_S2_path.unlink()
    else:
        print("  no accept/flag ROIs to subtract; skipping", flush=True)

    fov.rois.extend([r for r in stage3_rois if r.gate_outcome != "reject"])

    # Final summary
    print("\n[harness] === SUMMARY ===", flush=True)
    print(f"  Stage 1: {n_accept} accept + {n_flag} flag (loaded)", flush=True)
    print(f"  Stage 2: {n2_det} detected → {n2_acc} acc, {n2_flag} flag, {n2_rej} rej", flush=True)
    print(f"  Stage 3: {n3_det} detected → {n3_acc} acc, {n3_flag} flag, {n3_rej} rej", flush=True)
    print(f"  Total accept+flag ROIs: {sum(1 for r in fov.rois if r.gate_outcome in ('accept', 'flag'))}",
          flush=True)
    if n_accept and n2_acc > 1.5 * n_accept:
        print(f"  ⚠ CASCADE WARNING: Stage 2 acc ({n2_acc}) > 1.5× Stage 1 acc ({n_accept})", flush=True)
    if n2_acc and n3_acc > 0.5 * n2_acc:
        print(f"  ⚠ CASCADE WARNING: Stage 3 acc ({n3_acc}) > 0.5× Stage 2 acc ({n2_acc})", flush=True)


if __name__ == "__main__":
    main()
