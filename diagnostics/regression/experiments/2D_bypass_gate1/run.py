"""Experiment 2.D — Bypass Gate 1: render all 101 raw Cellpose candidates on mean_M.

Re-runs Stage 1 Cellpose on the previously-computed summary images (mean_M.tif,
vcorr_S.tif). This reproduces the 101 candidates that existed before Gate 1
rejected 17 + flagged 5. We render the full candidate set with per-ROI labels
so we can see exactly which cells Gate 1 removed.

Outputs (beside this script):
  raw_labels.tif            - (H, W) uint32 labeled image with all 101 candidates
  raw_candidates_overlay.png - annotated overlay of all candidates on mean_M
  rejects_flags_overlay.png  - only reject/flag ROIs highlighted with labels
  candidates_report.json     - per-label area/centroid + gate outcome from stage1_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

# Add project root to path so local imports resolve
PROJECT = Path("/home/thejoshbq/Otis-Lab/Projects/roigbiv")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from roigbiv.pipeline.stage1 import run_cellpose_detection  # noqa: E402
from roigbiv.pipeline.types import PipelineConfig  # noqa: E402
from diagnostic_compare import label_props, render_overlay, annotate_labels  # noqa: E402


def main() -> None:
    fov_dir = PROJECT / "test_output/cli_verify/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002"
    summary = fov_dir / "summary"
    out_dir = PROJECT / "diagnostics/regression/experiments/2D_bypass_gate1"
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_M = tifffile.imread(str(summary / "mean_M.tif")).astype(np.float32)
    vcorr_S = tifffile.imread(str(summary / "vcorr_S.tif")).astype(np.float32)
    print(f"Loaded mean_M {mean_M.shape} dtype={mean_M.dtype} range=[{mean_M.min():.1f},{mean_M.max():.1f}]")
    print(f"Loaded vcorr_S {vcorr_S.shape} dtype={vcorr_S.dtype} range=[{vcorr_S.min():.3f},{vcorr_S.max():.3f}]")

    # Reproduce the exact Stage 1 inference call (denoise ON matches the saved run)
    cfg = PipelineConfig()
    cfg.cellpose_model = str(PROJECT / "models/deployed/current_model")
    print(f"cfg: use_denoise={cfg.use_denoise}, diameter={cfg.diameter}, cp_thr={cfg.cellprob_threshold}, flow_thr={cfg.flow_threshold}, channels={cfg.channels}")

    masks_list, probs_list, label_image, cellprob_map = run_cellpose_detection(
        mean_M, vcorr_S, cfg
    )
    n = int(label_image.max())
    print(f"Reproduced {n} raw Cellpose candidates (saved stage had 101)")

    # Save raw label image
    tifffile.imwrite(str(out_dir / "raw_labels.tif"), label_image.astype(np.uint16))
    tifffile.imwrite(str(out_dir / "cellprob_map.tif"), cellprob_map.astype(np.float32))

    # Join regionprops with gate outcomes from the saved stage1_report
    report_path = fov_dir / "stage1/stage1_report.json"
    with open(report_path) as f:
        report = json.load(f)
    saved_props = {roi["label_id"]: roi for roi in report["rois"]}

    reproduced = label_props(label_image.astype(np.uint32), intensity=mean_M)
    by_area = sorted(reproduced, key=lambda p: -p["area"])

    # Match reproduced labels to saved report by centroid proximity (within 5 px).
    # If reproduction and saved indexing differ, this mapping helps us locate
    # the "rejected" set in the reproduced label image.
    saved_centroids = {}
    # saved_centroids keyed by (y, x) centroid — we don't have centroids in the
    # saved JSON; we use area as a proxy and then compute centroids from label_image
    # against the same area. Practically: reproduce and match by best IoU would be
    # more correct but we trust the label ordering is stable.

    # Simpler: compare label-by-label. Cellpose label IDs are assigned in raster
    # order by skimage.measure.label under the hood; stable across identical inputs.
    annotations: dict[int, str] = {}
    rejects_labels = np.zeros_like(label_image)
    flags_labels = np.zeros_like(label_image)
    accepts_labels = np.zeros_like(label_image)

    for p in reproduced:
        lid = p["label"]
        saved = saved_props.get(lid)
        if saved is None:
            continue
        outcome = saved["gate_outcome"]
        annotations[lid] = f"{lid}:{saved['area']}"
        mask = (label_image == lid)
        if outcome == "reject":
            rejects_labels[mask] = lid
        elif outcome == "flag":
            flags_labels[mask] = lid
        else:
            accepts_labels[mask] = lid

    n_acc = int((accepts_labels > 0).sum() > 0) and int(len(np.unique(accepts_labels)) - 1)
    n_flag = int(len(np.unique(flags_labels)) - 1)
    n_rej = int(len(np.unique(rejects_labels)) - 1)
    print(f"Reproduction → mapped to saved labels: {n_acc} accept, {n_flag} flag, {n_rej} reject")

    # Three-color overlay: accepts in green, flags in yellow, rejects in red
    render_overlay(
        mean_M,
        {
            "accept (79)": (accepts_labels, (0, 200, 0)),
            "flag (5)": (flags_labels, (255, 255, 0)),
            "reject (17)": (rejects_labels, (255, 50, 50)),
        },
        out_path=out_dir / "raw_candidates_overlay.png",
        title="Stage 1 Cellpose candidates on mean_M (before/after Gate 1)",
    )

    # Annotated figure showing only rejects/flags with id:area labels
    combined_rf = rejects_labels + flags_labels
    # annotate only rejects/flags
    rf_annot = {
        lid: f"{lid}:{saved_props[lid]['area']}"
        for lid in saved_props
        if saved_props[lid]["gate_outcome"] in ("reject", "flag")
    }
    annotate_labels(
        mean_M,
        combined_rf,
        annotations=rf_annot,
        out_path=out_dir / "rejects_flags_annotated.png",
        title="Gate 1 rejects (red/yellow) — labels = ID:area(px)",
        color=(255, 80, 80),
    )

    # Save per-label dump
    out_rows = []
    for p in reproduced:
        lid = p["label"]
        saved = saved_props.get(lid, {})
        out_rows.append({
            "label_id": lid,
            "reproduced_area": p["area"],
            "reproduced_centroid_yx": [p["centroid_y"], p["centroid_x"]],
            "saved_area": saved.get("area"),
            "gate_outcome": saved.get("gate_outcome"),
            "gate_reasons": saved.get("gate_reasons"),
            "cellpose_prob": saved.get("cellpose_prob"),
            "soma_surround_contrast": saved.get("soma_surround_contrast"),
        })
    with open(out_dir / "candidates_report.json", "w") as f:
        json.dump({
            "n_reproduced": n,
            "n_saved": report["detected"],
            "rows": out_rows,
        }, f, indent=2)
    print(f"Wrote overlay to {out_dir}")


if __name__ == "__main__":
    main()
