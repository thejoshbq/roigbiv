"""Annotation footprints and the registration shift into a run's movie frame.

``read_roiset`` rasterizes a hand-drawn ImageJ RoiSet with the benchmark's own rules
(``scripts/bench/manifest.load_annotation``: polygon via ``skimage.draw.polygon`` on the benchmark image shape,
centroid by centre of mass, point ROIs kept as centroids with no footprint), but keeps each cell's pixels.
``best_shift`` is the full-pipeline scorer's search: the offset (dy, dx) with ``bench[y, x] ~ movie[y + dy, x + dx]``
that maximises the correlation of the benchmark image with the run's mean image.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from roigbiv.workbench.match import Annotation


@dataclass(frozen=True)
class BenchCell:
    ann_id: int
    centroid: tuple  # (y, x) in the benchmark image frame
    footprint: np.ndarray | None  # flat indices into the benchmark image
    area: int | None


def read_roiset(path: Path, image_shape: tuple) -> list[BenchCell]:
    import roifile
    from scipy.ndimage import center_of_mass
    from skimage.draw import polygon as draw_polygon

    rois = roifile.roiread(str(path))
    if not isinstance(rois, list):
        rois = [rois]
    cells: list[BenchCell] = []
    for roi in rois:
        coords = roi.coordinates()
        if coords is None or len(coords) == 0:
            continue
        if roi.roitype == roifile.ROI_TYPE.POINT:
            for x, y in coords:
                cells.append(BenchCell(len(cells), (float(y), float(x)), None, None))
            continue
        if len(coords) < 3:
            continue
        rr, cc = draw_polygon(coords[:, 1], coords[:, 0], shape=image_shape)
        if rr.size == 0:
            continue
        mask = np.zeros(image_shape, dtype=bool)
        mask[rr, cc] = True
        cy, cx = center_of_mass(mask)
        cells.append(BenchCell(len(cells), (float(cy), float(cx)), np.flatnonzero(mask), int(rr.size)))
    return cells


def _zs(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


def best_shift(big: np.ndarray, small: np.ndarray, search: int = 12) -> tuple[int, int, float]:
    """Offset (dy, dx) of ``small`` inside ``big`` (small[y, x] ~ big[y + dy, x + dx]), by max correlation."""
    h, w = small.shape
    best = (0, 0, -2.0)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            y0, x0 = max(0, dy), max(0, dx)
            y1, x1 = min(big.shape[0], h + dy), min(big.shape[1], w + dx)
            if y1 - y0 < h * 0.8 or x1 - x0 < w * 0.8:
                continue
            c = float(np.mean(_zs(big[y0:y1, x0:x1]) * _zs(small[y0 - dy : y1 - dy, x0 - dx : x1 - dx])))
            if c > best[2]:
                best = (dy, dx, c)
    return best


def to_movie_frame(
    cells: Sequence[BenchCell], bench_shape: tuple, movie_shape: tuple, *, shift: tuple
) -> list[Annotation]:
    """Move benchmark cells into the movie frame; a cell whose centroid leaves the movie is dropped."""
    dy, dx = shift
    out = []
    for c in cells:
        cy, cx = c.centroid[0] + dy, c.centroid[1] + dx
        if not (0 <= cy < movie_shape[0] and 0 <= cx < movie_shape[1]):
            continue
        fp = None
        if c.footprint is not None:
            ys, xs = np.unravel_index(c.footprint, bench_shape)
            ys, xs = ys + dy, xs + dx
            ok = (ys >= 0) & (ys < movie_shape[0]) & (xs >= 0) & (xs < movie_shape[1])
            fp = np.ravel_multi_index((ys[ok], xs[ok]), movie_shape)
        out.append(Annotation(c.ann_id, (cy, cx), fp))
    return out
