"""Phase A.0 — current-pipeline overlay on mean_M.

Reruns Stage 1 + Gate 1 with the current (post-fix) config on the saved
foundation artifacts and renders an overlay in the same visual style as
/home/thejoshbq/Pictures/pipeline_rois.png (filled translucent green ROIs
with a darker-green outline, on a gamma-stretched grayscale mean_M).

Also saves the accepted-ROI label image so Phase A.2 can IoU-match it
against the A.1 reference set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT = Path("/home/thejoshbq/Otis-Lab/Projects/roigbiv")
sys.path.insert(0, str(PROJECT))

from roigbiv.pipeline.gate1 import evaluate_gate1  # noqa: E402
from roigbiv.pipeline.stage1 import run_cellpose_detection  # noqa: E402
from roigbiv.pipeline.types import PipelineConfig  # noqa: E402


def render_pipeline_style(
    image: np.ndarray,
    accept_labels: np.ndarray,
    out_path: str | Path,
    title: str,
    fill_color: tuple[int, int, int] = (40, 200, 80),
    outline_color: tuple[int, int, int] = (20, 110, 40),
    fill_alpha: float = 0.5,
    gamma: float = 0.7,
) -> None:
    """Render a matplotlib PNG matching the pipeline_rois.png aesthetic."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    lo, hi = np.quantile(image, [0.01, 0.995])
    norm = np.clip((image - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    norm = norm ** gamma

    rgb = np.stack([norm, norm, norm], axis=-1)  # (H, W, 3)

    fill_mask = accept_labels > 0
    fill = np.array(fill_color, dtype=np.float32) / 255.0
    rgb[fill_mask] = (1.0 - fill_alpha) * rgb[fill_mask] + fill_alpha * fill

    b = find_boundaries(accept_labels, mode="outer")
    rgb[b] = np.array(outline_color, dtype=np.float32) / 255.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fov_dir = PROJECT / "test_output/cli_verify/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002"
    summary = fov_dir / "summary"
    out_dir = PROJECT / "diagnostics/regression/followup"
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_M = tifffile.imread(str(summary / "mean_M.tif")).astype(np.float32)
    vcorr_S = tifffile.imread(str(summary / "vcorr_S.tif")).astype(np.float32)
    dog_map = tifffile.imread(str(summary / "dog_map.tif")).astype(np.float32)

    cfg = PipelineConfig()
    cfg.cellpose_model = str(PROJECT / "models/deployed/current_model")
    print(f"cfg.max_area={cfg.max_area}  use_denoise={cfg.use_denoise}  channels={cfg.channels}")

    masks_list, probs_list, label_image, cellprob_map = run_cellpose_detection(
        mean_M, vcorr_S, cfg
    )
    n_detected = len(masks_list)
    print(f"Cellpose: {n_detected} candidates")

    rois = evaluate_gate1(
        masks_list, probs_list, mean_M, vcorr_S, dog_map, cfg, starting_label_id=1
    )
    n_accept = sum(1 for r in rois if r.gate_outcome == "accept")
    n_flag = sum(1 for r in rois if r.gate_outcome == "flag")
    n_reject = sum(1 for r in rois if r.gate_outcome == "reject")
    print(f"Gate 1: {n_detected} detected → {n_accept} accepted / {n_flag} flagged / {n_reject} rejected")

    accept_labels = np.zeros(mean_M.shape, dtype=np.uint32)
    for r in rois:
        if r.gate_outcome == "accept":
            accept_labels[r.mask] = r.label_id

    tifffile.imwrite(str(out_dir / "A0_accept_labels.tif"),
                     accept_labels.astype(np.uint16))
    print(f"Saved {out_dir}/A0_accept_labels.tif")

    title = (f"Phase A.0 — current pipeline on mean_M "
             f"(max_area={cfg.max_area}, denoise={cfg.use_denoise}, channels={tuple(cfg.channels)})\n"
             f"{n_detected} detected → {n_accept} accepted")
    render_pipeline_style(
        mean_M, accept_labels,
        out_path=out_dir / "A0_current_pipeline_on_mean_M.png",
        title=title,
    )
    print(f"Saved {out_dir}/A0_current_pipeline_on_mean_M.png")


if __name__ == "__main__":
    main()
