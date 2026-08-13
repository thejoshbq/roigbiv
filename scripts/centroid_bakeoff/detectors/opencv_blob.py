"""OpenCV centroid detector — cv2.SimpleBlobDetector on the Difference-of-
Gaussian map.

Why SimpleBlobDetector over the watershed approach already in
``scripts/cv_bakeoff/detectors/classical.py`` (skimage-based, not cv2, and
mask-first): ``KeyPoint.pt`` is a genuine centroid, computed internally via
image moments across multiple threshold layers — no extra
regionprops/center_of_mass step needed. That fits this benchmark's
centroid-first premise better than boundary-then-center-of-mass. Its
``minDistBetweenBlobs`` also gives built-in peak suppression, in the same
spirit as ``roi_stamp.py::resolve_crowding``'s "closer than the stamp radius"
merge, so no hand-rolled NMS is needed either.

Input: ``dog_map`` (Difference-of-Gaussian on mean_M) by default — the
pipeline's Foundation stage already computes this as a blob-detection
preprocessing step (``roigbiv.pipeline.foundation.compute_nuclear_shadow_map``),
so this detector reuses existing pipeline output rather than reimplementing
DoG. ``mean_M`` is available via ``--opencv-channel`` for an A/B.

``min_threshold``/``max_threshold``/``threshold_step``/``min_repeatability``
and ``area_scale_min``/``area_scale_max`` are exposed as constructor args
(defaults match this module's original hardcoded values, i.e. cv2's own
``SimpleBlobDetector_Params`` defaults for the threshold-scan knobs) so the
sweep infrastructure can vary them — the internal threshold-scan range was
never set relative to the actual image's intensity distribution, which is the
leading suspect for this detector's near-zero score in the first bake-off run.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from centroid_bakeoff.detector import CentroidDetectorInputs, CentroidDetectorResult

_FALLBACK_DIAMETER = 12.0  # roigbiv/pipeline/types.py's own GRIN-profile default


def _stretch01_to_uint8(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    """Percentile-stretch to uint8 — same convention as
    ``cv_bakeoff.detectors.classical._stretch01`` / ``roigbiv.overlay._stretch_to_uint8``."""
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


class OpenCVBlobDetector:
    name = "opencv"

    def __init__(
        self,
        channel: str = "dog_map",
        min_circularity: float = 0.3,
        min_convexity: float = 0.5,
        blob_color: int = 255,
        min_threshold: float = 50.0,
        max_threshold: float = 220.0,
        threshold_step: float = 10.0,
        min_repeatability: int = 2,
        area_scale_min: float = 0.4,
        area_scale_max: float = 3.0,
    ):
        self.channel = channel
        self.min_circularity = min_circularity
        self.min_convexity = min_convexity
        self.blob_color = blob_color
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.threshold_step = threshold_step
        self.min_repeatability = min_repeatability
        self.area_scale_min = area_scale_min
        self.area_scale_max = area_scale_max

    def detect(self, inputs: CentroidDetectorInputs) -> CentroidDetectorResult:
        img = inputs.summary.get(self.channel)
        if img is None:
            raise KeyError(
                f"opencv detector needs summary channel {self.channel!r}; "
                f"have {sorted(inputs.summary)}"
            )

        t0 = time.time()
        soma_scale = inputs.soma_scale
        diameter = (
            soma_scale.diameter_med
            if soma_scale is not None and getattr(soma_scale, "ok", False)
            else _FALLBACK_DIAMETER
        )
        a_circ = math.pi * (diameter / 2.0) ** 2

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        # Mirrors optics.py::derive_scale_params's own area-bound formulas
        # (0.4x-3.0x circular area), so the auto-scaling language matches the
        # rest of the pipeline's gates rather than inventing new bounds.
        params.minArea = self.area_scale_min * a_circ
        params.maxArea = self.area_scale_max * a_circ
        # Mirrors the crowding-guard's "closer than the stamp radius" merge
        # convention (roi_stamp.py::resolve_crowding) as built-in NMS.
        params.minDistBetweenBlobs = 0.6 * diameter
        params.filterByCircularity = True
        params.minCircularity = self.min_circularity
        params.filterByConvexity = True
        params.minConvexity = self.min_convexity
        params.filterByColor = True
        params.blobColor = self.blob_color
        params.filterByInertia = False
        params.minThreshold = self.min_threshold
        params.maxThreshold = self.max_threshold
        params.thresholdStep = self.threshold_step
        params.minRepeatability = self.min_repeatability

        u8 = _stretch01_to_uint8(img)
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(u8)

        centroids = np.asarray(
            [[kp.pt[1], kp.pt[0]] for kp in keypoints], dtype=np.float32,
        ).reshape(-1, 2)
        scores = np.asarray([kp.size for kp in keypoints], dtype=np.float32)
        elapsed = time.time() - t0

        return CentroidDetectorResult(
            centroids=centroids,
            scores=scores,
            meta={
                "method": self.name, "channel": self.channel, "diameter": diameter,
                "minArea": params.minArea, "maxArea": params.maxArea,
                "minDistBetweenBlobs": params.minDistBetweenBlobs,
                "minThreshold": params.minThreshold, "maxThreshold": params.maxThreshold,
                "thresholdStep": params.thresholdStep,
                "minCircularity": self.min_circularity, "minConvexity": self.min_convexity,
                "n": int(len(keypoints)), "runtime_s": round(elapsed, 2),
            },
        )
