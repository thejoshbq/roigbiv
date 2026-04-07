"""
ROI G. Biv — Three-branch ROI detection pipeline for two-photon calcium imaging.

Branch A: Cellpose spatial segmentation (mean + Vcorr projections)
Branch B: Suite2p spatiotemporal detection (SVD-based)
Branch C: Tonic neuron detection (bandpass SVD + local correlation clustering)

Steps 8-10: Merge, trace extraction, and classification.
"""

__version__ = "0.3.0"

from roigbiv.io import discover_tifs, extract_projections, download_model
from roigbiv.suite2p import run_suite2p_fov, run_suite2p_batch
from roigbiv.tonic import run_tonic_detection, local_corr_svd, bandpass_temporal
from roigbiv.merge import merge_three_branches, merge_fov, merge_batch, stat_to_mask
from roigbiv.traces import (
    extract_traces_fov, extract_raw_traces, build_neuropil_masks,
    estimate_alpha, compute_dff, deconvolve,
)
from roigbiv.classify import (
    classify_fov, compute_qc_features, classify_cell_nocell,
    classify_activity_type,
)
from roigbiv.union import build_union_batch
from roigbiv.viz import create_colab_viewer

__all__ = [
    # I/O
    "discover_tifs",
    "extract_projections",
    "download_model",
    # Branch B: Suite2p
    "run_suite2p_fov",
    "run_suite2p_batch",
    # Branch C: Tonic
    "run_tonic_detection",
    "local_corr_svd",
    "bandpass_temporal",
    # Step 8: Merge
    "merge_three_branches",
    "merge_fov",
    "merge_batch",
    "stat_to_mask",
    # Step 9: Traces
    "extract_traces_fov",
    "extract_raw_traces",
    "build_neuropil_masks",
    "estimate_alpha",
    "compute_dff",
    "deconvolve",
    # Step 10: Classify
    "classify_fov",
    "compute_qc_features",
    "classify_cell_nocell",
    "classify_activity_type",
    # Legacy
    "build_union_batch",
    "create_colab_viewer",
    "__version__",
]
