"""Validity mask: which pixels a rigid registration shift filled with content from the opposite edge.

Suite2p 0.14.5 applies the rigid shift with ``np.roll(frame, (-dy, -dx))`` (``registration/rigid.py
shift_frame``), so the band a shift exposes holds pixels wrapped from the other side of the field, not zeros. Pixel
(r, c) of a registered frame holds source (r + dy, c + dx); it is wrapped when that falls outside the frame. As the
shift changes from frame to frame, an edge pixel switches between its own content and content ~Ly away, a step
the Stage 3 template sweep can score as an event. The nonrigid step that follows clamps at the border (edge
replication, ``nonrigid.map_coordinates``), so it brings in no foreign content; ``margin`` covers it if wanted.

``suite2p_crop`` rebuilds Suite2p's own valid range (``register.compute_crop``): a symmetric band of
``ceil(max |shift|)`` over the frames that are not bad, on every side. ``wrapped_fraction`` and ``wrapped_frames``
give the exact per-pixel and per-frame picture from the rigid offsets saved in ``motion_trace.npz``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Crop:
    yrange: tuple  # [y0, y1): valid rows
    xrange: tuple  # [x0, x1): valid columns


def _whole(offsets) -> np.ndarray:
    a = np.asarray(offsets, dtype=np.float64)
    r = np.round(a)
    if not np.array_equal(a, r):
        raise ValueError("rigid shifts must be whole pixels")
    return r.astype(np.int64)


def _row_band(dy: int, ly: int) -> np.ndarray:
    """Rows wrapped by a vertical shift dy: the bottom dy rows when dy > 0, the top -dy rows when dy < 0."""
    rows = np.zeros(ly, bool)
    if dy > 0:
        rows[max(0, ly - dy) :] = True
    elif dy < 0:
        rows[: min(ly, -dy)] = True
    return rows


def wrapped_fraction(yoff, xoff, shape) -> np.ndarray:
    """(Ly, Lx): the share of frames in which each pixel holds wrapped content."""
    ly, lx = shape
    dy, dx = _whole(yoff), _whole(xoff)
    pairs, counts = np.unique(np.stack([dy, dx], axis=1), axis=0, return_counts=True)
    total = np.zeros(shape, dtype=np.float64)
    for (py, px), n in zip(pairs, counts):
        total += n * (_row_band(int(py), ly)[:, None] | _row_band(int(px), lx)[None, :])
    return total / max(1, dy.size)


def wrapped_frames(yoff, xoff, shape, ys, xs) -> np.ndarray:
    """(T, n) bool: whether pixel (ys[i], xs[i]) holds wrapped content in each frame."""
    ly, lx = shape
    ys, xs = np.asarray(ys), np.asarray(xs)
    if ys.size and (ys.min() < 0 or ys.max() >= ly or xs.min() < 0 or xs.max() >= lx):
        raise ValueError("pixel outside the frame")
    dy, dx = _whole(yoff)[:, None], _whole(xoff)[:, None]
    ys, xs = ys[None, :], xs[None, :]
    return ((ys + dy) < 0) | ((ys + dy) >= ly) | ((xs + dx) < 0) | ((xs + dx) >= lx)


def suite2p_crop(yoff, xoff, shape, badframes=None, margin: int = 0) -> Crop:
    """Suite2p's valid range from the rigid shifts, widened by ``margin`` on every side.

    ``badframes`` must be the run's saved ``ops["badframes"]``: ``compute_crop`` merges its own correlation
    outliers into it. Unlike Suite2p, each band is capped at half the frame, so the range never inverts.
    """
    if margin < 0:
        raise ValueError("margin must be >= 0")
    ly, lx = shape
    dy, dx = np.abs(_whole(yoff)), np.abs(_whole(xoff))
    if badframes is not None and np.mean(badframes) < 0.5:
        keep = ~np.asarray(badframes, bool)
        dy, dx = dy[keep], dx[keep]
    by = min(int(dy.max(initial=0)) + int(margin), ly // 2)
    bx = min(int(dx.max(initial=0)) + int(margin), lx // 2)
    return Crop(yrange=(by, ly - by), xrange=(bx, lx - bx))


def valid_mask(crop: Crop, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[crop.yrange[0] : crop.yrange[1], crop.xrange[0] : crop.xrange[1]] = True
    return m


def outside_share(footprint, mask: np.ndarray) -> float:
    """Share of a footprint's pixels outside the valid mask; NaN for an empty footprint."""
    fp = np.asarray(footprint, dtype=np.int64)
    if fp.size == 0:
        return float("nan")
    return float(1.0 - mask.flat[fp].mean())
