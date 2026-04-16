"""
Stage 1 smoke test on real annotated projection TIFs.

Runs Cellpose (with denoising) on an actual PrL-NAc FOV's mean + Vcorr and
exercises Gate 1. Doesn't require motion correction — uses pre-computed
projections as stand-ins for mean_S / vcorr_S.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile


PROJ_DIR = Path("/home/thejoshbq/Otis-Lab/Projects/roigbiv/data/annotated")
STEM = "T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_PRE-002"


def test_stage1_on_real_projections():
    from roigbiv.pipeline.stage1 import run_cellpose_detection
    from roigbiv.pipeline.gate1 import evaluate_gate1
    from roigbiv.pipeline.foundation import compute_nuclear_shadow_map
    from roigbiv.pipeline.types import PipelineConfig

    mean_path = PROJ_DIR / f"{STEM}_mc_mean.tif"
    vcorr_path = PROJ_DIR / f"{STEM}_mc_vcorr.tif"
    if not mean_path.exists():
        mean_path = PROJ_DIR / f"{STEM}_mean.tif"
        vcorr_path = PROJ_DIR / f"{STEM}_vcorr.tif"
    if not mean_path.exists():
        print(f"SKIP: no projections for {STEM}")
        return

    mean_S = tifffile.imread(str(mean_path)).astype(np.float32)
    vcorr_S = tifffile.imread(str(vcorr_path)).astype(np.float32)
    print(f"  loaded {mean_S.shape} mean, {vcorr_S.shape} vcorr")

    # Use deployed fine-tuned model if it exists, else cyto3
    model_path = "/home/thejoshbq/Otis-Lab/Projects/roigbiv/models/deployed/current_model"
    if not Path(model_path).exists():
        model_path = "cyto3"
    cfg = PipelineConfig(
        fs=30.0,
        cellpose_model=model_path,
        use_denoise=True,  # exercises the Cellpose3 denoise path
    )

    candidates, probs, label_image, cellprob_map = run_cellpose_detection(
        mean_S, vcorr_S, cfg,
    )
    print(f"  Cellpose produced {len(candidates)} candidate ROIs")
    assert len(candidates) > 5, f"Expected >5 candidates; got {len(candidates)}"
    assert label_image.shape == mean_S.shape

    # Check label integrity
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels != 0]
    assert len(unique_labels) == len(candidates), (
        f"Label image has {len(unique_labels)} labels but {len(candidates)} candidates"
    )

    # Compute DoG on the real mean and run Gate 1
    dog_map = compute_nuclear_shadow_map(mean_S)
    rois = evaluate_gate1(candidates, probs, mean_S, vcorr_S, dog_map, cfg)
    n_accept = sum(1 for r in rois if r.gate_outcome == "accept")
    n_flag = sum(1 for r in rois if r.gate_outcome == "flag")
    n_reject = sum(1 for r in rois if r.gate_outcome == "reject")
    print(f"  Gate 1: {n_accept} accept, {n_flag} flag, {n_reject} reject")
    assert len(rois) == len(candidates)
    assert n_accept + n_flag > 0, "Gate 1 rejected ALL candidates; something's wrong"

    # Verify all features populated
    for r in rois:
        assert r.area > 0
        assert 0 <= r.solidity <= 1
        assert 0 <= r.eccentricity <= 1
        assert r.cellpose_prob is not None
    print(f"  ✓ All {len(rois)} ROIs have complete feature set")
    print(f"  ✓ Stage 1 + Gate 1 smoke test passed on real data")


if __name__ == "__main__":
    test_stage1_on_real_projections()
