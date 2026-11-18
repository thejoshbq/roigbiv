"""Cellpose-SAM as a second Stage 1 proposer next to Cellpose 3.

Cellpose-SAM (``cpsam_v2``, Cellpose 4) has no diameter setting, so it finds
somata the 12 px Cellpose 3 defaults miss. It runs in a separate env (the
``roigbiv`` env pins Cellpose 3) through ``scripts/cpsam_sidecar.py``; this
module holds the parts that need no Cellpose: the image it sees and the rule
that merges its masks into Cellpose 3's.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def keep_secondary(
    primary: list[np.ndarray],
    secondary: list[np.ndarray],
    *,
    radius: float,
    iou_min: float,
) -> list[int]:
    """Indices of ``secondary`` masks that duplicate no ``primary`` mask.

    A secondary mask is a duplicate when its centroid lies within ``radius``
    px (inclusive) of a primary centroid, or its IoU with any primary mask is
    ``iou_min`` or more. Primary wins every tie: it is the lab-trained model.
    """
    if not primary:
        return list(range(len(secondary)))
    p_cent = np.array([np.argwhere(m).mean(0) for m in primary], float)
    p_area = np.array([m.sum() for m in primary], float)
    keep = []
    for i, m in enumerate(secondary):
        c = np.argwhere(m).mean(0)
        if np.min(np.hypot(*(p_cent - c).T)) <= radius:
            continue
        inter = np.array([np.logical_and(m, p).sum() for p in primary], float)
        iou = inter / (p_area + m.sum() - inter)
        if np.max(iou) >= iou_min:
            continue
        keep.append(i)
    return keep


def binned_max_movie(
    data_bin_path: Path,
    shape: tuple,
    *,
    frames_per_bin: int = 7,
    chunk_bins: int = 8,
) -> np.ndarray:
    """Per-pixel maximum over ``frames_per_bin``-frame means of a registered movie.

    Reads the int16 ``(T, Ly, Lx)`` movie at ``data_bin_path`` by memmap,
    ``chunk_bins`` bins at a time. Trailing frames that do not fill a bin
    belong to no bin. No baseline is subtracted, so bright resting cells show
    as well as active ones. Returned as float32.
    """
    T, Ly, Lx = (int(s) for s in shape)
    n_bins = T // frames_per_bin
    if n_bins < 1:
        raise ValueError(f"movie has no complete bin of {frames_per_bin} frames (T={T})")
    movie = np.memmap(data_bin_path, dtype=np.int16, mode="r", shape=(T, Ly, Lx))
    best = np.full((Ly, Lx), -np.inf)
    for b0 in range(0, n_bins, chunk_bins):
        b1 = min(n_bins, b0 + chunk_bins)
        block = np.asarray(movie[b0 * frames_per_bin:b1 * frames_per_bin], dtype=np.float64)
        means = block.reshape(b1 - b0, frames_per_bin, Ly, Lx).mean(axis=1)
        np.maximum(best, means.max(axis=0), out=best)
    return best.astype(np.float32)
