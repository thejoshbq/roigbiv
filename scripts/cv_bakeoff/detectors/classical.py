"""Classical CV detector — training-free, fully interpretable baseline.

Pipeline: percentile-stretch → optional smoothing → threshold (Otsu/Li) →
distance transform → LoG/peak seeds → watershed → area filter. No weights, no
GPU. Useful as a floor and as a fast iteration loop for the image-enrichment
tactics (it has no learned priors, so summary-image quality is the only lever).
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import blob_log, peak_local_max
from skimage.filters import threshold_li, threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

from cv_bakeoff.detector import DetectorInputs, DetectorResult


def _stretch01(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


class ClassicalDetector:
    name = "classical"

    def __init__(
        self,
        channel: str = "mean_M",
        diameter: float = 12.0,
        threshold: str = "otsu",         # "otsu" | "li"
        seed_method: str = "distance",   # "distance" | "log"
        min_area: int = 20,
        max_area: int = 800,
        smooth_sigma: float = 1.0,
    ):
        self.channel = channel
        self.diameter = float(diameter)
        self.threshold = threshold
        self.seed_method = seed_method
        self.min_area = min_area
        self.max_area = max_area
        self.smooth_sigma = smooth_sigma

    def detect(self, inputs: DetectorInputs) -> DetectorResult:
        img = inputs.summary.get(self.channel)
        if img is None:
            raise KeyError(
                f"classical detector needs summary channel {self.channel!r}; "
                f"have {sorted(inputs.summary)}"
            )

        t0 = time.time()
        norm = _stretch01(img)
        if self.smooth_sigma > 0:
            norm = ndi.gaussian_filter(norm, self.smooth_sigma)

        thr = (threshold_otsu(norm) if self.threshold == "otsu"
               else threshold_li(norm))
        fg = norm > thr
        # Drop salt noise smaller than a fraction of a soma before seeding.
        fg = ndi.binary_opening(fg, iterations=1)

        distance = ndi.distance_transform_edt(fg)
        min_dist = max(1, int(round(self.diameter * 0.5)))

        if self.seed_method == "log":
            blobs = blob_log(
                norm, min_sigma=self.diameter / 6.0,
                max_sigma=self.diameter / 2.0, num_sigma=5, threshold=0.05,
            )
            seed_mask = np.zeros(norm.shape, dtype=bool)
            for y, x, _sigma in blobs:
                seed_mask[int(y), int(x)] = True
        else:
            peaks = peak_local_max(
                distance, min_distance=min_dist, labels=fg,
            )
            seed_mask = np.zeros(distance.shape, dtype=bool)
            seed_mask[tuple(peaks.T)] = True

        markers = label(seed_mask)
        labels = watershed(-distance, markers, mask=fg)

        # Area filter → relabel contiguous 1..N.
        out = np.zeros_like(labels, dtype=np.uint16)
        next_id = 1
        for region in regionprops(labels):
            if self.min_area <= region.area <= self.max_area:
                out[labels == region.label] = next_id
                next_id += 1
        elapsed = time.time() - t0

        return DetectorResult(
            label_mask=out,
            meta={
                "method": self.name,
                "channel": self.channel,
                "diameter": self.diameter,
                "threshold": self.threshold,
                "seed_method": self.seed_method,
                "min_area": self.min_area,
                "max_area": self.max_area,
                "n_rois": int(out.max()),
                "runtime_s": round(elapsed, 2),
            },
        )
