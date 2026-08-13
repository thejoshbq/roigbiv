"""The interchangeable-detector contract for the centroid bake-off.

Point-first counterpart to ``scripts/cv_bakeoff/detector.py``'s mask-first
``Detector`` protocol. Every method returns bare (y, x) centroids, not a label
mask — deliberately, since fitting a boundary is exactly what this benchmark
is trying to avoid for pyramidal neurons (apical dendrites make boundary
segmentation unreliable; see docs/adr/0003-centroid-canonical-roi-stamps.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class CentroidDetectorInputs:
    """Everything a detector may read for one FOV.

    ``summary`` maps summary-image channel names (``mean_M``, ``vcorr_S``,
    ``max_S``, ``dog_map``, …) to ``(H, W) float32`` arrays — the same
    convention as ``cv_bakeoff.detector.DetectorInputs``. ``soma_scale`` is
    measured once per FOV by the CLI (``roigbiv.pipeline.optics.measure_soma_scale``)
    and shared across all three detectors, so parameter auto-scaling is
    identical across methods regardless of which one is run.
    """

    summary: dict[str, np.ndarray]
    fov_stem: str
    shape: tuple[int, int]
    fs: float = 30.0
    raw_tif_path: Optional[Path] = None       # Suite2p needs the movie, not just summaries
    cfg: object = None                        # roigbiv.pipeline.types.PipelineConfig, for parity knobs
    soma_scale: object = None                 # roigbiv.pipeline.optics.SomaScale


@dataclass
class CentroidDetectorResult:
    """A detector's output for one FOV."""

    centroids: np.ndarray                      # (N, 2) float32, (y, x)
    scores: Optional[np.ndarray] = None         # (N,) confidence, None if unavailable
    meta: dict = field(default_factory=dict)    # method, params, timing, n, …

    @property
    def n(self) -> int:
        return int(self.centroids.shape[0])


@runtime_checkable
class CentroidDetector(Protocol):
    """Structural type for an interchangeable centroid-localization method."""

    name: str

    def detect(self, inputs: CentroidDetectorInputs) -> CentroidDetectorResult:
        ...
