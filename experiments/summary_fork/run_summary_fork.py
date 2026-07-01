"""Summary-image fork experiment — Phase 1 read-only diagnostic.

Tests whether the summary image *source* handed to Stage 1 changes recall, holding
the validated detector + Gate 1 + IoU matcher constant. One variable: the Cellpose
morphology-channel array.

PREMISE NOTE (verified against code): the production pipeline ALREADY detects Stage 1
on ``mean_M`` (raw movie mean, pre-SVD) — see ``run.py:597-603``. So the conditions are
relabeled truthfully:

  * ``mean_M``        = the DEPLOYED Stage-1 input (current baseline / live recall gap)
  * ``mean_S``        = COUNTERFACTUAL (mean of the SVD sparse residual; the pipeline
                        deliberately does NOT use this — quantifies the SVD-absorption
                        effect on the summary and feeds the retention number)
  * ``meanImgE``      = bounded sweep (Suite2p enhanced/high-pass mean, from ops.npy)
  * ``correlation_M`` = bounded sweep (8-neighbour Vcorr on the registered movie)

This is READ-ONLY against the spine: it consumes existing Foundation artifacts
(summary/*.tif, suite2p data.bin, ops.npy) and reruns only the read-only detector.
Nothing outside experiments/summary_fork/ is written.

Usage:
    python experiments/summary_fork/run_summary_fork.py --fov grin  --run-id 001
    python experiments/summary_fork/run_summary_fork.py --fov prism --run-id 002
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT = Path("/home/thejoshbq/Otis-Lab/Projects/Phoxel-Workbench/roigbiv")
sys.path.insert(0, str(PROJECT))

from roigbiv.eval.match import iou_match                       # noqa: E402
from roigbiv.eval.retention import retention_summary           # noqa: E402
from roigbiv.pipeline.foundation import vcorr_on_movie, _open_data_bin  # noqa: E402
from roigbiv.pipeline.gate1 import evaluate_gate1              # noqa: E402
from roigbiv.pipeline.stage1 import run_cellpose_detection    # noqa: E402
from roigbiv.pipeline.types import PipelineConfig             # noqa: E402

IOU_THRESHOLDS = (0.5, 0.3)
TAU_RETAIN = 0.5

# ── FOV registry ──────────────────────────────────────────────────────────
# Each entry pins the exact deployed cfg (the "held-constant" baseline) via a
# hardcoded EXPECTED dict; the drift-guard aborts if the on-disk cfg_snapshot
# diverges from it. Paths are the existing on-disk Foundation artifacts.
FOVS = {
    "grin": {
        "stem": "T1_230202_PrL-NAc-G6-6F_HI-D2_FOV2_pre-000",
        "lens": "grin",
        "run_root": PROJECT / "experiments/runs/T1_230202_robust",
        "summary": PROJECT / "experiments/runs/T1_230202_robust/summary",
        "s2p_plane": PROJECT / "experiments/runs/T1_230202_robust"
        / "T1_230202_PrL-NAc-G6-6F_HI-D2_FOV2_pre-000/suite2p/plane0",
        "manifest": PROJECT / "experiments/runs/T1_230202_robust/.roigbiv_manifest.json",
        "gt": PROJECT / "data/ROIGBIV-DATA/cellpose_ready/masks"
        / "T1_230202_PrL-NAc-G6-6F_HI-D2_FOV2_pre-000_mc_masks.tif",
        "gt_independent": True,   # independent Cellpose training annotation
        "ondisk_baseline": {"detected": 56, "accepted": 55},
        # Deployed GRIN cfg (from .roigbiv_manifest.json cfg_snapshot).
        "expected_cfg": {
            "cellpose_model": "models/deployed/current_model",
            "diameter": 12, "cellprob_threshold": -2.0, "flow_threshold": 0.6,
            "channels": [1, 2], "use_denoise": True, "min_area": 80, "max_area": 600,
        },
    },
    "prism": {
        "stem": "052126_DS-Prism-3_VI15_D2_FOV2_pre-005",
        "lens": "prism",
        "run_root": PROJECT / "output/prism_profile/052126_DS-Prism-3_VI15_D2_FOV2_pre-005",
        "summary": PROJECT / "output/prism_profile/052126_DS-Prism-3_VI15_D2_FOV2_pre-005/summary",
        "s2p_plane": PROJECT / "output/prism_profile/052126_DS-Prism-3_VI15_D2_FOV2_pre-005"
        / "052126_DS-Prism-3_VI15_D2_FOV2_pre-005_Ch2/suite2p/plane0",
        "manifest": PROJECT / "output/prism_profile/052126_DS-Prism-3_VI15_D2_FOV2_pre-005/.roigbiv_manifest.json",
        # HITL-edited seg (uint16 label TIFF). Weaker independence — pipeline-derived.
        "gt": PROJECT / "data/logan_cousa_trial/output/052126_DS-Prism-3_VI15_D2_FOV2_pre-005"
        / "hitl_staging/masks/052126_DS-Prism-3_VI15_D2_FOV2_pre-005_seg.tif",
        "gt_independent": False,
        "ondisk_baseline": {"detected": 18, "accepted": 11},
        # Deployed PRISM cfg (from cfg_snapshot). NOTE max_area=5000 here — the
        # deployed run predates the profiles.py widening to 9000; the directive's
        # 5000 matches THIS validated baseline, so we hold 5000 constant.
        "expected_cfg": {
            "cellpose_model": "cyto3",
            "diameter": 56, "cellprob_threshold": 0.0, "flow_threshold": 0.4,
            "channels": [0, 0], "use_denoise": False, "min_area": 1500, "max_area": 5000,
        },
    },
}


def load_cfg(manifest_path: Path, expected: dict) -> tuple[PipelineConfig, dict]:
    """Build the held-constant PipelineConfig from the deployed cfg_snapshot and
    drift-guard it against the hardcoded EXPECTED baseline. Aborts on drift."""
    snap = json.loads(manifest_path.read_text())["cfg_snapshot"]

    # Drift-guard: the deployed snapshot must match the pinned baseline exactly.
    drift = []
    for k, want in expected.items():
        got = snap.get(k)
        if isinstance(want, list):
            got = list(got) if got is not None else got
        if got != want:
            drift.append(f"{k}: snapshot={got!r} != pinned={want!r}")
    if drift:
        raise SystemExit(
            "DRIFT-GUARD ABORT — deployed cfg_snapshot diverged from pinned baseline:\n  "
            + "\n  ".join(drift)
        )

    fields = {f.name for f in dc.fields(PipelineConfig)}
    kw = {}
    for k, v in snap.items():
        if k not in fields:
            continue
        kw[k] = tuple(v) if k == "channels" and isinstance(v, list) else v
    cfg = PipelineConfig(**kw)

    # Resolve the deployed-model path to absolute (stored relative in the snapshot).
    if cfg.cellpose_model and not str(cfg.cellpose_model).startswith("cyto"):
        p = PROJECT / cfg.cellpose_model
        if p.exists():
            cfg.cellpose_model = str(p)
    return cfg, snap


def label_from_masks(masks_list, shape) -> np.ndarray:
    """Raw-detection label image (pre-gate) from a list of bool masks."""
    lab = np.zeros(shape, dtype=np.uint16)
    for i, m in enumerate(masks_list, start=1):
        lab[m] = i
    return lab


def accept_labels_from_rois(rois, shape) -> np.ndarray:
    lab = np.zeros(shape, dtype=np.uint16)
    for r in rois:
        if r.gate_outcome == "accept":
            lab[r.mask] = r.label_id
    return lab


def score(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Recall / missed-cell list at each IoU threshold."""
    out = {}
    n_gt = int(len(np.unique(gt)) - (1 if 0 in np.unique(gt) else 0))
    for thr in IOU_THRESHOLDS:
        mr = iou_match(gt, pred, min_iou=thr)
        recall = mr.n_tp / (mr.n_tp + mr.n_fn) if (mr.n_tp + mr.n_fn) else float("nan")
        out[f"iou_{thr}"] = {
            "recall": recall, "n_tp": mr.n_tp, "n_fp": mr.n_fp, "n_fn": mr.n_fn,
            "missed_gt_labels": sorted(int(x) for x in mr.fn),
        }
    out["n_gt"] = n_gt
    return out


def build_condition_morphs(fov: dict) -> dict:
    """Load / compute the four morphology-channel candidates."""
    summ = fov["summary"]
    mean_M = tifffile.imread(str(summ / "mean_M.tif")).astype(np.float32)
    mean_S = tifffile.imread(str(summ / "mean_S.tif")).astype(np.float32)
    conds = {"mean_M": mean_M, "mean_S": mean_S}

    # meanImgE from Suite2p ops (enhanced/high-pass mean).
    ops_p = fov["s2p_plane"] / "ops.npy"
    try:
        ops = np.load(str(ops_p), allow_pickle=True).item()
        e = ops.get("meanImgE")
        if e is not None and np.asarray(e).shape == mean_M.shape:
            conds["meanImgE"] = np.asarray(e, dtype=np.float32)
        else:
            print(f"  [sweep] meanImgE unavailable/shape-mismatch — skipping", flush=True)
    except Exception as exc:
        print(f"  [sweep] meanImgE load failed ({exc}) — skipping", flush=True)

    # correlation_M: 8-neighbour Vcorr on the registered movie (no SVD).
    data_bin = fov["s2p_plane"] / "data.bin"
    if data_bin.exists():
        Ly, Lx = mean_M.shape
        T = data_bin.stat().st_size // (Ly * Lx * 2)
        print(f"  [sweep] correlation_M: streaming data.bin (T={T}) ...", flush=True)
        summ_m = vcorr_on_movie(data_bin, Ly, Lx, int(T))
        conds["correlation_M"] = np.asarray(summ_m["vcorr"], dtype=np.float32)
    else:
        print(f"  [sweep] data.bin missing — skipping correlation_M", flush=True)
    return conds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov", required=True, choices=sorted(FOVS))
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    fov = FOVS[args.fov]
    run_dir = PROJECT / "experiments/summary_fork" / f"run_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = run_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    print(f"=== summary-fork Phase 1 | fov={args.fov} ({fov['lens']}) | run_{args.run_id} ===",
          flush=True)

    cfg, snap = load_cfg(fov["manifest"], fov["expected_cfg"])
    print(f"  cfg pinned OK: model={cfg.cellpose_model} chan={cfg.channels} "
          f"diam={cfg.diameter} cellprob={cfg.cellprob_threshold} flow={cfg.flow_threshold} "
          f"denoise={cfg.use_denoise} area=({cfg.min_area},{cfg.max_area})", flush=True)

    summ = fov["summary"]
    mean_M = tifffile.imread(str(summ / "mean_M.tif")).astype(np.float32)
    mean_S = tifffile.imread(str(summ / "mean_S.tif")).astype(np.float32)
    mean_L = tifffile.imread(str(summ / "mean_L.tif")).astype(np.float32)
    vcorr_S = tifffile.imread(str(summ / "vcorr_S.tif")).astype(np.float32)
    dog_map = tifffile.imread(str(summ / "dog_map.tif")).astype(np.float32)
    max_S = tifffile.imread(str(summ / "max_S.tif")).astype(np.float32)
    gt = tifffile.imread(str(fov["gt"])).astype(np.uint16)

    if gt.shape != mean_M.shape:
        raise SystemExit(f"GT shape {gt.shape} != summary shape {mean_M.shape}")

    gt_ids = np.unique(gt); gt_ids = gt_ids[gt_ids != 0]
    gt_masks = [(gt == i) for i in gt_ids]

    # ── Retention (SVD-absorption) number — cells present in mean_M, absent from mean_S ──
    ret = retention_summary(mean_S, mean_L, gt_masks, tau_retain=TAU_RETAIN)
    per = np.asarray(ret["per_mask"], dtype=np.float64)
    absorbed = [int(gt_ids[j]) for j in range(len(gt_ids))
                if np.isfinite(per[j]) and per[j] < TAU_RETAIN]
    ret["absorbed_gt_labels"] = absorbed
    ret["n_absorbed"] = len(absorbed)
    print(f"  RETENTION: {len(gt_ids)} GT somata | r_S median={ret['r_S_median']:.3f} "
          f"| frac_pass(τ={TAU_RETAIN})={ret['frac_pass']:.3f} "
          f"| n_absorbed(r_S<{TAU_RETAIN})={len(absorbed)}", flush=True)

    # ── Per-condition detection + scoring ──
    conds = build_condition_morphs(fov)
    results = {}
    for name, morph in conds.items():
        print(f"\n  --- condition: {name} ---", flush=True)
        masks_list, probs_list, label_image, cellprob_map = run_cellpose_detection(
            morph.astype(np.float32), vcorr_S, cfg, max_S=max_S,
        )
        rois = evaluate_gate1(masks_list, probs_list, mean_M, vcorr_S, dog_map, cfg,
                              starting_label_id=1)
        n_accept = sum(1 for r in rois if r.gate_outcome == "accept")
        raw_lab = label_from_masks(masks_list, mean_M.shape)
        acc_lab = accept_labels_from_rois(rois, mean_M.shape)
        tifffile.imwrite(str(masks_dir / f"{name}_raw_labels.tif"), raw_lab)
        tifffile.imwrite(str(masks_dir / f"{name}_accept_labels.tif"), acc_lab)

        raw_score = score(gt, raw_lab)
        acc_score = score(gt, acc_lab)
        results[name] = {
            "n_detected": len(masks_list), "n_accept": n_accept,
            "raw": raw_score, "accept": acc_score,
        }
        print(f"    detected={len(masks_list)} accepted={n_accept} | "
              f"raw recall@0.5={raw_score['iou_0.5']['recall']:.3f} "
              f"@0.3={raw_score['iou_0.3']['recall']:.3f} | "
              f"accept recall@0.5={acc_score['iou_0.5']['recall']:.3f} "
              f"@0.3={acc_score['iou_0.3']['recall']:.3f}", flush=True)

    # ── Persist manifest + cfg snapshot + metrics ──
    import subprocess
    branch = subprocess.run(["git", "-C", str(PROJECT), "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    manifest = {
        "experiment": "summary_image_fork", "phase": 1,
        "fov": args.fov, "lens": fov["lens"], "stem": fov["stem"],
        "gt_independent": fov["gt_independent"], "gt_path": str(fov["gt"]),
        "n_gt": int(len(gt_ids)), "gt_shape": list(gt.shape),
        "ondisk_baseline": fov["ondisk_baseline"],
        "iou_thresholds": list(IOU_THRESHOLDS), "tau_retain": TAU_RETAIN,
        "pinned_cfg": fov["expected_cfg"],
        "cfg_resolved": {
            "cellpose_model": str(cfg.cellpose_model), "channels": list(cfg.channels),
            "diameter": cfg.diameter, "cellprob_threshold": cfg.cellprob_threshold,
            "flow_threshold": cfg.flow_threshold, "use_denoise": cfg.use_denoise,
            "min_area": cfg.min_area, "max_area": cfg.max_area,
        },
        "git_branch": branch,
        "premise_note": "Pipeline detects Stage 1 on mean_M (pre-SVD), not mean_S — "
                        "run.py:597-603. mean_M = deployed baseline; mean_S = counterfactual.",
        "condition_note": {
            "mean_M": "DEPLOYED Stage-1 input (baseline / live recall gap)",
            "mean_S": "COUNTERFACTUAL (SVD sparse-residual mean; not used in production)",
            "meanImgE": "bounded sweep — Suite2p enhanced mean",
            "correlation_M": "bounded sweep — Vcorr on registered movie (no SVD)",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "cfg_snapshot.json").write_text(json.dumps(snap, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(
        {"retention": ret, "conditions": results}, indent=2))
    print(f"\n  wrote {run_dir}/{{manifest,cfg_snapshot,metrics}}.json + masks/", flush=True)


if __name__ == "__main__":
    main()
