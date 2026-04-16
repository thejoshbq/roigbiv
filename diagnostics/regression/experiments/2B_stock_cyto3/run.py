"""Experiment 2.B — Stock cyto3 vs fine-tuned run015 on identical input.

Runs three inference configurations on the same mean_M + vcorr_S pair:
  (a) cyto3, single-channel mean_M, permissive thresholds
  (b) cyto3, dual-channel [mean_M, vcorr_S], permissive thresholds
  (c) cyto3, single-channel mean_M, default thresholds (cp=0.0, flow=0.4)
  (d) cyto3, single-channel mean_M, diameter=None (auto)

Compares each configuration's masks to the 101 fine-tuned candidates via IoU.
Reports: counts, % of the 17 rejects that cyto3 also finds, % of new detections.
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

from diagnostic_compare import iou_match, render_overlay, label_props  # noqa: E402


def run_cyto3(mean_M: np.ndarray, vcorr_S: np.ndarray, *, dual: bool, diameter, cp: float, flow: float):
    import torch
    from cellpose.models import CellposeModel

    gpu = torch.cuda.is_available()
    model = CellposeModel(gpu=gpu, model_type="cyto3")
    if dual:
        x = np.stack([mean_M.astype(np.float32), vcorr_S.astype(np.float32)], axis=-1)
        channels = [1, 2]
        channel_axis = -1
    else:
        x = mean_M.astype(np.float32)
        channels = [0, 0]
        channel_axis = None

    masks, flows, styles = model.eval(
        x,
        diameter=diameter,
        cellprob_threshold=cp,
        flow_threshold=flow,
        channels=channels,
        channel_axis=channel_axis,
        normalize={"tile_norm_blocksize": 128},
    )
    return np.asarray(masks, dtype=np.uint32)


def main() -> None:
    fov_dir = PROJECT / "test_output/cli_verify/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002"
    summary = fov_dir / "summary"
    out_dir = PROJECT / "diagnostics/regression/experiments/2B_stock_cyto3"
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_M = tifffile.imread(str(summary / "mean_M.tif")).astype(np.float32)
    vcorr_S = tifffile.imread(str(summary / "vcorr_S.tif")).astype(np.float32)

    # Reference: fine-tuned 101-candidate label image from 2.D
    ref_labels = tifffile.imread(
        str(PROJECT / "diagnostics/regression/experiments/2D_bypass_gate1/raw_labels.tif")
    ).astype(np.uint32)
    print(f"Reference has {int(ref_labels.max())} fine-tuned candidates")

    # Load gate outcomes so we can check: how many of the 17 rejects does cyto3 find?
    with open(fov_dir / "stage1/stage1_report.json") as f:
        report = json.load(f)
    rejected_ids = {roi["label_id"] for roi in report["rois"] if roi["gate_outcome"] == "reject"}
    flagged_ids = {roi["label_id"] for roi in report["rois"] if roi["gate_outcome"] == "flag"}
    print(f"Rejected: {len(rejected_ids)}, Flagged: {len(flagged_ids)}")

    configs = [
        ("a_single_permissive", dict(dual=False, diameter=12, cp=-2.0, flow=0.6)),
        ("b_dual_permissive", dict(dual=True, diameter=12, cp=-2.0, flow=0.6)),
        ("c_single_default", dict(dual=False, diameter=12, cp=0.0, flow=0.4)),
        ("d_single_auto_diameter", dict(dual=False, diameter=None, cp=-2.0, flow=0.6)),
    ]

    results = []
    for name, kwargs in configs:
        print(f"\n=== Config {name} === {kwargs}")
        labels = run_cyto3(mean_M, vcorr_S, **kwargs)
        n = int(labels.max())
        print(f"cyto3 → {n} candidates")
        tifffile.imwrite(str(out_dir / f"cyto3_{name}_labels.tif"), labels.astype(np.uint16))

        # IoU match cyto3 vs fine-tuned 101 candidates
        matches, cyto_only, ft_only = iou_match(labels, ref_labels, min_iou=0.3)

        # How many of the 17 rejects does cyto3 also find?
        ft_found_by_cyto3 = {m[1] for m in matches}
        rejects_found = rejected_ids & ft_found_by_cyto3
        flags_found = flagged_ids & ft_found_by_cyto3
        ft_not_found_by_cyto3 = set(ft_only)  # fine-tuned labels cyto3 missed

        # Render: cyto3 (cyan) vs fine-tuned rejects (red)
        ft_rejects = np.zeros_like(ref_labels)
        ft_flags = np.zeros_like(ref_labels)
        for lid in rejected_ids:
            ft_rejects[ref_labels == lid] = lid
        for lid in flagged_ids:
            ft_flags[ref_labels == lid] = lid
        render_overlay(
            mean_M,
            {
                f"cyto3 ({n})": (labels, (0, 220, 220)),
                "FT rejects (17)": (ft_rejects, (255, 50, 50)),
                "FT flags (5)": (ft_flags, (255, 255, 0)),
            },
            out_path=out_dir / f"overlay_{name}.png",
            title=f"cyto3 {name}: cp={kwargs['cp']}, flow={kwargs['flow']}, diameter={kwargs['diameter']}, dual={kwargs['dual']}",
        )

        results.append({
            "config": name,
            "kwargs": {k: v for k, v in kwargs.items()},
            "n_cyto3_candidates": n,
            "n_matched_to_FT_any": len(matches),
            "n_FT_rejects_also_found_by_cyto3": len(rejects_found),
            "n_FT_flags_also_found_by_cyto3": len(flags_found),
            "n_FT_labels_missed_by_cyto3": len(ft_not_found_by_cyto3),
            "n_cyto3_only": len(cyto_only),
        })

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['config']}: cyto3={r['n_cyto3_candidates']:>3}, matches FT={r['n_matched_to_FT_any']:>3}, "
              f"rejects-found={r['n_FT_rejects_also_found_by_cyto3']}/17, "
              f"flags-found={r['n_FT_flags_also_found_by_cyto3']}/5, "
              f"FT-missed={r['n_FT_labels_missed_by_cyto3']}, cyto3-only={r['n_cyto3_only']}")


if __name__ == "__main__":
    main()
