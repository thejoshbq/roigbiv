"""Activity-summary images from a movie, streamed one frame at a time.

``activity_max``: the movie is averaged into bins of ``bin_s`` seconds; the
bins are grouped into blocks of ``block_s`` seconds with half overlap; within
each block the per-pixel median over the block's bins is subtracted from each
bin; the image is the per-pixel maximum of that over every bin of every block.
Cells that fire rarely stand out against their own local baseline.

``temporal_sd``: per-pixel population standard deviation over all frames.

Windows are in seconds, so the frame rate must be known. Memory stays at one
block of bins plus running sums, so a 27,000-frame movie never loads whole.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import tifffile

from bench.schema import ActivityImages


def iter_frames(path: Path, kind: str) -> Iterator[np.ndarray]:
    """Yield the movie's frames as 2-D arrays, one at a time.

    ``kind == "tif"``: one multi-page TIFF; yield ``page.asarray()`` for each
    page of ``tifffile.TiffFile(path).pages``. ``kind == "frame_dir"``: a
    folder of TIFFs; for each ``*.tif`` in sorted name order yield
    ``tifffile.imread(f, key=0)`` (first page only: Bruker frame files can
    carry an OME series that would otherwise load the whole recording). Any
    other kind raises ``ValueError``.
    """
    if kind == "tif":
        with tifffile.TiffFile(str(path)) as tf:
            for page in tf.pages:
                yield page.asarray()
    elif kind == "frame_dir":
        for f in sorted(Path(path).glob("*.tif")):
            yield tifffile.imread(str(f), key=0)
    else:
        raise ValueError(f"unknown movie kind: {kind!r}")


def activity_images(
    frames: Iterable[np.ndarray],
    fs: float,
    bin_s: float = 1.0,
    block_s: float = 30.0,
) -> ActivityImages:
    """Compute both images in one pass over ``frames``.

    Rules:

    1. ``fs``, ``bin_s`` and ``block_s`` must be positive (else ``ValueError``).
    2. ``frames_per_bin = max(1, round(bin_s * fs))``. Bin ``i`` is the mean
       of frames ``[i * frames_per_bin, (i + 1) * frames_per_bin)``, in
       float64. Trailing frames that do not fill a bin belong to no bin.
    3. ``k = max(1, round(block_s / bin_s))`` bins per block,
       ``stride = max(1, k // 2)``. Blocks start at ``0, stride, 2 * stride,
       ...`` while ``start + k <= n_bins``. If the last such block does not end
       at ``n_bins``, one more block ``[n_bins - k, n_bins)`` is added. If
       ``n_bins < k`` there is one block of all bins.
    4. For each block: ``d = bins - median(bins, axis=0)`` (``np.median``);
       ``activity_max`` is the maximum of ``d`` over every bin of every block.
    5. ``temporal_sd`` uses every frame (trailing frames included):
       ``sqrt(max(mean(x**2) - mean(x)**2, 0))`` from running float64 sums.
    6. No frames, or fewer frames than one bin, raises ``ValueError``.
    7. Both images are returned as float32; ``n_frames``, ``n_bins`` and
       ``n_blocks`` report the counts used.
    """
    if not fs > 0 or not bin_s > 0 or not block_s > 0:
        raise ValueError("fs, bin_s and block_s must be positive")

    frames_per_bin = max(1, round(bin_s * fs))
    k = max(1, round(block_s / bin_s))
    stride = max(1, k // 2)

    bins: deque[np.ndarray] = deque(maxlen=k)
    bin_sum: np.ndarray | None = None
    in_bin = 0
    n_frames = 0
    n_bins = 0
    n_blocks = 0
    next_start = 0
    last_end = 0
    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    act: np.ndarray | None = None

    def emit(block: np.ndarray) -> None:
        nonlocal act, n_blocks
        d = block - np.median(block, axis=0)
        peak = d.max(axis=0)
        act = peak if act is None else np.maximum(act, peak)
        n_blocks += 1

    for frame in frames:
        x = np.asarray(frame, dtype=np.float64)
        if sum_x is None:
            sum_x = np.zeros(x.shape)
            sum_x2 = np.zeros(x.shape)
            bin_sum = np.zeros(x.shape)
        sum_x += x
        sum_x2 += x * x
        n_frames += 1

        bin_sum += x
        in_bin += 1
        if in_bin == frames_per_bin:
            bins.append(bin_sum / frames_per_bin)
            bin_sum = np.zeros(x.shape)
            in_bin = 0
            n_bins += 1
            if n_bins == next_start + k:
                emit(np.stack(bins))
                last_end = n_bins
                next_start += stride

    if n_bins == 0:
        raise ValueError("movie has no complete bin")
    if n_bins < k or last_end != n_bins:
        emit(np.stack(bins))

    mean_x = sum_x / n_frames
    mean_x2 = sum_x2 / n_frames
    temporal_sd = np.sqrt(np.maximum(mean_x2 - mean_x * mean_x, 0.0))

    return ActivityImages(
        activity_max=act.astype(np.float32),
        temporal_sd=temporal_sd.astype(np.float32),
        n_frames=n_frames,
        n_bins=n_bins,
        n_blocks=n_blocks,
    )


def binned_max(frames: Iterable[np.ndarray], frames_per_bin: int = 7) -> np.ndarray:
    """Per-pixel maximum over bin means, streamed one frame at a time.

    Bin ``i`` is the float64 mean of frames ``[i * frames_per_bin, (i + 1) *
    frames_per_bin)``; trailing frames that do not fill a bin belong to no bin.
    No baseline is subtracted, so bright resting cells show as well as active
    ones. ``frames_per_bin`` below 1 or a movie with no complete bin raises
    ``ValueError``. Returned as float32.
    """
    if frames_per_bin < 1:
        raise ValueError("frames_per_bin must be at least 1")
    best: np.ndarray | None = None
    bin_sum: np.ndarray | None = None
    in_bin = 0
    for frame in frames:
        x = np.asarray(frame, dtype=np.float64)
        bin_sum = x.copy() if in_bin == 0 else bin_sum + x
        in_bin += 1
        if in_bin == frames_per_bin:
            mean = bin_sum / frames_per_bin
            best = mean if best is None else np.maximum(best, mean)
            in_bin = 0
    if best is None:
        raise ValueError("movie has no complete bin")
    return best.astype(np.float32)
