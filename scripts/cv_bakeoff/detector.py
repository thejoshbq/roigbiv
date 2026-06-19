"""The interchangeable-detector contract for the CV bake-off.

Every method — in-process (CP3, classical) or out-of-process sidecar
(Cellpose-SAM, StarDist) — satisfies the same ``Detector`` protocol and returns
the same universal output: a ``(H, W) uint16`` label mask (0 = background),
identical to the codebase's ``merged_masks.tif`` / ``run_cnmfe.py`` convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class DetectorInputs:
    """Everything a detector may read for one FOV.

    ``summary`` maps summary-image channel names (``mean_M``, ``vcorr_S``,
    ``max_S``, ``std_S``, ``dog_map``, …) to ``(H, W) float32`` arrays. Detectors
    pick the channel(s) they need; missing channels are simply absent from the
    dict. ``params`` carries detector-specific knobs (diameter, model path, …).
    """

    summary: dict[str, np.ndarray]
    fov_stem: str
    fs: float = 7.5
    params: dict = field(default_factory=dict)


@dataclass
class DetectorResult:
    """A detector's output for one FOV."""

    label_mask: np.ndarray              # (H, W) uint16, 0 = background
    meta: dict = field(default_factory=dict)  # method, params, timing, n_rois, …

    @property
    def n_rois(self) -> int:
        return int(self.label_mask.max())


@runtime_checkable
class Detector(Protocol):
    """Structural type for an interchangeable segmentation method."""

    name: str

    def detect(self, inputs: DetectorInputs) -> DetectorResult:
        ...
