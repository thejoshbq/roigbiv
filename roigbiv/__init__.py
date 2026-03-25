"""
ROI G. Biv — Consensus cell detection pipeline for two-photon calcium imaging.

Combines Suite2p (activity-based + anatomy-based detection) and Cellpose (fine-tuned
segmentation) into a three-tier confidence system: GOLD, SILVER, BRONZE.
"""

__version__ = "0.1.1"

from roigbiv.io import discover_tifs, extract_projections, download_model
from roigbiv.suite2p import run_suite2p_fov, run_suite2p_batch
from roigbiv.union import build_union_batch
from roigbiv.viz import create_colab_viewer

__all__ = [
    "discover_tifs",
    "extract_projections",
    "download_model",
    "run_suite2p_fov",
    "run_suite2p_batch",
    "build_union_batch",
    "create_colab_viewer",
    "__version__",
]
