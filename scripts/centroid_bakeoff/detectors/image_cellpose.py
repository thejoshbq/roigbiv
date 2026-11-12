"""Cellpose on one named summary image, as a centroid proposer.

The activity-summary images (``bench.activity_images``) and the DeepCAD mean
are new inputs for Cellpose, not new models: the same production call
(``roigbiv.pipeline.stage1.run_cellpose_detection``) runs with the named image
in place of ``mean_M``. Each image is its own proposer so the union score can
credit it separately.
"""
# contract stub: U6 replaces the function bodies below; signatures are frozen.
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from centroid_bakeoff.detector import CentroidDetectorInputs, CentroidDetectorResult


class SummaryImageCellposeDetector:
    """Run Cellpose with ``inputs.summary[image_key]`` as the first channel.

    ``detect`` rules:

    1. ``image = inputs.summary[image_key]``; a missing key raises ``KeyError``
       whose message names ``image_key`` and the available keys.
    2. Second channel: ``inputs.summary["vcorr_S"]`` if present, else ``image``.
       ``max_S = inputs.summary.get("max_S")``.
    3. ``cfg = self.cfg or inputs.cfg``; if both are ``None``, ``cfg =
       PipelineConfig(fs=inputs.fs)`` (imported lazily from
       ``roigbiv.pipeline.types``). If ``self.cellprob_threshold`` is not
       ``None``, ``cfg = dataclasses.replace(cfg,
       cellprob_threshold=self.cellprob_threshold)``.
    4. ``run = self.run_fn``, else ``roigbiv.pipeline.stage1.run_cellpose_detection``
       imported lazily. Call ``masks_list, probs_list, _labels, _cellprob =
       run(image, ch2, cfg, max_S=max_S)``.
    5. Centroids: ``scipy.ndimage.center_of_mass`` of each mask, ``(N, 2)``
       float32 ``(y, x)``; scores: ``probs_list`` as float32. No masks gives
       ``(0, 2)`` and ``(0,)`` arrays.
    6. ``meta`` holds ``method`` (``self.name``), ``image_key``, ``n``,
       ``runtime_s`` and ``cellprob_threshold`` (``getattr(cfg,
       "cellprob_threshold", None)``).

    No import of ``roigbiv.pipeline`` happens at module import time or when
    both ``cfg`` and ``run_fn`` are supplied.
    """

    def __init__(
        self,
        image_key: str,
        name: Optional[str] = None,
        cfg=None,
        cellprob_threshold: Optional[float] = None,
        run_fn: Optional[Callable] = None,
    ):
        self.image_key = image_key
        self.name = name or f"cellpose_{image_key}"
        self.cfg = cfg
        self.cellprob_threshold = cellprob_threshold
        self.run_fn = run_fn

    def detect(self, inputs: CentroidDetectorInputs) -> CentroidDetectorResult:
        return CentroidDetectorResult(centroids=np.zeros((0, 2), dtype=np.float32))


def activity_max_detector(**kwargs) -> SummaryImageCellposeDetector:
    """Cellpose on ``summary["activity_max"]``, named ``cellpose_activity_max``."""
    return SummaryImageCellposeDetector(image_key="")


def temporal_sd_detector(**kwargs) -> SummaryImageCellposeDetector:
    """Cellpose on ``summary["temporal_sd"]``, named ``cellpose_temporal_sd``."""
    return SummaryImageCellposeDetector(image_key="")


def deepcad_mean_detector(**kwargs) -> SummaryImageCellposeDetector:
    """Cellpose on ``summary["deepcad_mean"]``, named ``cellpose_deepcad_mean``."""
    return SummaryImageCellposeDetector(image_key="")
