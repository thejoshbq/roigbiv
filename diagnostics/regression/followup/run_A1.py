"""Phase A.1 — reference reproduction: fine-tuned Cellpose on raw mean_M,
denoise OFF, single-channel. Reproduces image 3 (golden reference).

Saves A1_reference_labels.tif and A1_reference_overlay.png (same green-fill
style as A.0 for visual comparability).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT = Path("/home/thejoshbq/Otis-Lab/Projects/roigbiv")
sys.path.insert(0, str(PROJECT))

from roigbiv.pipeline.stage1 import run_cellpose_detection  # noqa: E402
from roigbiv.pipeline.types import PipelineConfig  # noqa: E402

sys.path.insert(0, str(PROJECT / "diagnostics/regression/followup"))
from run_A0 import render_pipeline_style  # noqa: E402


def main() -> None:
    fov_dir = PROJECT / "test_output/cli_verify/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002"
    summary = fov_dir / "summary"
    out_dir = PROJECT / "diagnostics/regression/followup"

    mean_M = tifffile.imread(str(summary / "mean_M.tif")).astype(np.float32)
    vcorr_S = tifffile.imread(str(summary / "vcorr_S.tif")).astype(np.float32)

    cfg = PipelineConfig()
    cfg.cellpose_model = str(PROJECT / "models/deployed/current_model")
    cfg.use_denoise = False
    cfg.channels = (0, 0)
    print(f"cfg.use_denoise={cfg.use_denoise}  channels={cfg.channels}  (reference config)")

    # run_cellpose_detection stacks both inputs as (H, W, 2). With channels=[0,0]
    # Cellpose treats as grayscale / uses only channel 0. Second arg is unused
    # but still stacked — pass vcorr_S for code-path fidelity.
    masks_list, probs_list, label_image, _ = run_cellpose_detection(
        mean_M, vcorr_S, cfg
    )
    n = len(masks_list)
    print(f"Reference Cellpose (no denoise, single-channel): {n} candidates")

    ref_labels = np.asarray(label_image, dtype=np.uint32)
    tifffile.imwrite(str(out_dir / "A1_reference_labels.tif"),
                     ref_labels.astype(np.uint16))
    print(f"Saved {out_dir}/A1_reference_labels.tif")

    title = (f"Phase A.1 — reference (denoise=False, channels=(0,0))\n"
             f"{n} cells detected on raw mean_M")
    render_pipeline_style(
        mean_M, ref_labels,
        out_path=out_dir / "A1_reference_overlay.png",
        title=title,
    )
    print(f"Saved {out_dir}/A1_reference_overlay.png")


if __name__ == "__main__":
    main()
