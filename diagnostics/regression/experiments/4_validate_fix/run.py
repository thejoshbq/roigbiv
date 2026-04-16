"""Phase 4 validation — re-run Stage 1 + Gate 1 with the fix in place.

Uses existing summary artifacts (mean_M, vcorr_S, dog_map) from the prior
test_output run, re-runs Cellpose + Gate 1 with the patched PipelineConfig
(max_area=600), and reports the new accept/flag/reject distribution.

Success criterion: no ROI rejected for `area_high` alone; the previously-missed
22 ROIs (17 rejects + 5 flags, all in 351–503 px) are now accepted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT = Path("/home/thejoshbq/Otis-Lab/Projects/roigbiv")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from roigbiv.pipeline.gate1 import evaluate_gate1  # noqa: E402
from roigbiv.pipeline.stage1 import run_cellpose_detection  # noqa: E402
from roigbiv.pipeline.types import PipelineConfig  # noqa: E402
from diagnostic_compare import render_overlay  # noqa: E402


def main() -> None:
    fov_dir = PROJECT / "test_output/cli_verify/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002"
    summary = fov_dir / "summary"
    out_dir = PROJECT / "diagnostics/regression/experiments/4_validate_fix"
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_M = tifffile.imread(str(summary / "mean_M.tif")).astype(np.float32)
    vcorr_S = tifffile.imread(str(summary / "vcorr_S.tif")).astype(np.float32)
    dog_map = tifffile.imread(str(summary / "dog_map.tif")).astype(np.float32)

    cfg = PipelineConfig()
    cfg.cellpose_model = str(PROJECT / "models/deployed/current_model")
    print(f"cfg.max_area = {cfg.max_area}  (was 350)")
    print(f"cfg.use_denoise = {cfg.use_denoise}")

    # Stage 1 — identical config to the saved run (denoise on)
    masks_list, probs_list, label_image, cellprob_map = run_cellpose_detection(
        mean_M, vcorr_S, cfg
    )
    n_detected = len(masks_list)
    print(f"Cellpose: {n_detected} candidates")

    # Gate 1 — using patched cfg
    rois = evaluate_gate1(
        masks_list, probs_list, mean_M, vcorr_S, dog_map, cfg, starting_label_id=1
    )
    by_out = {"accept": [], "flag": [], "reject": []}
    for r in rois:
        by_out[r.gate_outcome].append(r)

    n_accept = len(by_out["accept"])
    n_flag = len(by_out["flag"])
    n_reject = len(by_out["reject"])
    print(f"Gate 1 (fixed): {n_detected} detected → "
          f"{n_accept} accepted / {n_flag} flagged / {n_reject} rejected")

    # Compare against saved (pre-fix) report
    with open(fov_dir / "stage1/stage1_report.json") as f:
        saved = json.load(f)
    print(f"Prior (max_area=350): {saved['detected']} detected → "
          f"{saved['accepted']} / {saved['flagged']} / {saved['rejected']}")

    # What are the remaining rejects?
    remaining_reasons: dict[str, int] = {}
    for r in by_out["reject"]:
        for reason in r.gate_reasons:
            head = reason.split(":")[0]
            remaining_reasons[head] = remaining_reasons.get(head, 0) + 1
    print(f"Remaining reject reasons: {remaining_reasons}")

    # Any area_high rejects left?
    area_high_rejects = [r for r in by_out["reject"] if any("area_high" in x for x in r.gate_reasons)]
    print(f"area_high rejects remaining: {len(area_high_rejects)} (must be 0 — any left would exceed new max=600)")

    # Render overlay: previously-rejected cells should now be accepts (green)
    accept_labels = np.zeros(mean_M.shape, dtype=np.uint32)
    flag_labels = np.zeros(mean_M.shape, dtype=np.uint32)
    reject_labels = np.zeros(mean_M.shape, dtype=np.uint32)
    for r in rois:
        if r.gate_outcome == "accept":
            accept_labels[r.mask] = r.label_id
        elif r.gate_outcome == "flag":
            flag_labels[r.mask] = r.label_id
        else:
            reject_labels[r.mask] = r.label_id

    render_overlay(
        mean_M,
        {
            f"accept ({n_accept})": (accept_labels, (0, 200, 0)),
            f"flag ({n_flag})": (flag_labels, (255, 255, 0)),
            f"reject ({n_reject})": (reject_labels, (255, 50, 50)),
        },
        out_path=out_dir / "post_fix_overlay.png",
        title=f"Post-fix Stage 1 (max_area=600): {n_accept}A / {n_flag}F / {n_reject}R",
    )

    # Save detailed report
    report_out = {
        "config": {
            "max_area": cfg.max_area,
            "min_area": cfg.min_area,
            "min_solidity": cfg.min_solidity,
            "max_eccentricity": cfg.max_eccentricity,
            "min_contrast": cfg.min_contrast,
            "use_denoise": cfg.use_denoise,
        },
        "counts": {
            "detected": n_detected,
            "accepted": n_accept,
            "flagged": n_flag,
            "rejected": n_reject,
        },
        "prior_counts": {
            "detected": saved["detected"],
            "accepted": saved["accepted"],
            "flagged": saved["flagged"],
            "rejected": saved["rejected"],
        },
        "delta": {
            "accepted": n_accept - saved["accepted"],
            "flagged": n_flag - saved["flagged"],
            "rejected": n_reject - saved["rejected"],
        },
        "remaining_reject_reasons": remaining_reasons,
        "area_high_rejects_remaining": len(area_high_rejects),
    }
    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(report_out, f, indent=2)
    print(f"Saved overlay and validation_report.json to {out_dir}")


if __name__ == "__main__":
    main()
