"""Random-access reads of a FOV's registered movie, cropped to what's on screen.

Backs the /discovery viewer's playback controls. The page's legacy equivalent is
Fiji: scrub the stack, watch for a transient, draw a polygon around it. The
static mean projection the viewer draws over cannot show the "watch for a
transient" half, so this module serves the movie itself.

What it serves, and why in this shape
-------------------------------------
One request returns a *block* of frames for a *rectangle* of the field at a
*decimation*, as raw uint8 — not PNGs, and not the whole frame.

* **Raw uint8, not PNG.** The client keeps a ring buffer around the playhead and
  draws with ``putImageData``; it needs pixels, not a container. PNG encoding
  each frame server-side (what :mod:`roigbiv.ui.routes.roi_editor` does) costs
  more CPU than the read it wraps.
* **Cropped.** Zoomed 8x into a 128x128 region, a frame is 16 KB instead of the
  262 KB a full 512² frame costs. That ratio is what makes playback responsive
  while zoomed in, which is exactly when a user is deciding whether a blob is a
  cell.
* **Decimated.** Zoomed *out*, the screen cannot resolve every pixel anyway, so
  the client asks for a stride matched to its zoom and full-field playback stays
  affordable.
* **Blocks, not single frames.** One round trip per frame cannot keep up with
  playback, and the resulting request storm is what forces a player to start
  dropping frames.

Normalisation is a **fixed global window** sampled once per movie, not a
per-frame percentile stretch. A per-frame stretch (again, what ``roi_editor``
does) re-normalises against each frame's own extremes, so the background pulses
in antiphase with any bright transient — precisely the signal the user is
looking for, hidden by the display.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

#: Frames per request. Bounds the work one HTTP handler can do and, with the
#: byte cap below, bounds how much a single client can ask the server to hold.
MAX_COUNT = 64

#: Bytes per request. 8 MB is ~32 full 512² frames — a comfortable lookahead at
#: any zoom, and small enough that a pathological request cannot stall the
#: process the UI's pipeline thread also lives in.
MAX_BYTES = 8 << 20

#: Frames sampled to derive the display window. Enough to see the movie's real
#: dynamic range; cheap enough (~33 MB read at 512²) to do lazily on first play.
_WINDOW_SAMPLES = 64

_DTYPE = np.int16


@dataclass(frozen=True)
class MovieSource:
    """A movie on disk plus everything needed to memmap it."""

    path: Path
    shape: tuple[int, int, int]   # (T, Ly, Lx)
    fps: float
    kind: str                     # "data_bin" | "mc_tif"
    #: ``(st_size, st_mtime_ns)`` at resolve time — the cache key's identity
    #: half, so re-running motion correction invalidates rather than aliases.
    stat: tuple[int, int]

    @property
    def n_frames(self) -> int:
        return self.shape[0]

    @property
    def height(self) -> int:
        return self.shape[1]

    @property
    def width(self) -> int:
        return self.shape[2]


# ── resolution ──────────────────────────────────────────────────────────────


def _stat_of(path: Path) -> tuple[int, int]:
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)


def resolve_movie(output_dir: Path) -> Optional[MovieSource]:
    """The best available registered movie for *output_dir*, or ``None``.

    Prefers Suite2p's ``data.bin`` — already int16 and contiguous, so a frame is
    one seek. Falls back to the ``{stem}_mc.tif`` export
    (:func:`roigbiv.pipeline.registration._write_mc_tif`) for FOVs whose
    ``data.bin`` was reclaimed; that export is uncompressed contiguous BigTIFF,
    so it memmaps too.
    """
    from roigbiv.pipeline.discovery_extract import resolve_suite2p

    try:
        data_bin, shape, fs = resolve_suite2p(output_dir)
    except FileNotFoundError:
        pass
    else:
        # A frameless data.bin is a motion-correction run that died partway,
        # not a movie. Reported as absent so the player says so, rather than
        # left to blow up in np.memmap ("cannot mmap an empty file") once
        # something asks for a frame.
        if shape[0] >= 1:
            return MovieSource(path=data_bin, shape=shape, fps=fs,
                               kind="data_bin", stat=_stat_of(data_bin))

    mc_tif = output_dir / f"{output_dir.name}_mc.tif"
    if not mc_tif.exists():
        return None
    try:
        import tifffile

        with tifffile.TiffFile(str(mc_tif)) as tf:
            series = tf.series[0]
            shape = tuple(int(v) for v in series.shape)
            dtype = series.dtype
    except (OSError, ValueError, IndexError):
        return None
    if len(shape) != 3 or shape[0] < 1 or dtype != np.uint16:
        return None
    return MovieSource(path=mc_tif, shape=shape, fps=0.0,
                       kind="mc_tif", stat=_stat_of(mc_tif))


# ── memmap pool ─────────────────────────────────────────────────────────────
#
# Keyed on (path, size, mtime) rather than path alone: a re-run of motion
# correction rewrites data.bin in place, and a path-only key would keep handing
# out a memmap over the old length.

_pool: dict[tuple, np.memmap] = {}
_pool_lock = threading.Lock()


def open_movie(src: MovieSource) -> np.memmap:
    """A pooled read-only memmap over *src*. Safe to read from any thread."""
    key = (str(src.path.resolve()), src.stat)
    mm = _pool.get(key)
    if mm is not None:
        return mm
    with _pool_lock:
        if key not in _pool:
            # Drop any older generation of the same file before mapping the new
            # one, so a long-lived session does not accumulate stale mappings.
            for stale in [k for k in _pool if k[0] == key[0]]:
                _pool.pop(stale, None)
            _pool[key] = _map(src)
        return _pool[key]


def _map(src: MovieSource) -> np.memmap:
    if src.kind == "mc_tif":
        import tifffile

        return tifffile.memmap(str(src.path), mode="r")
    return np.memmap(str(src.path), dtype=_DTYPE, mode="r", shape=src.shape)


# ── display window ──────────────────────────────────────────────────────────

_windows: dict[tuple, tuple[float, float]] = {}
_windows_lock = threading.Lock()


def display_window(src: MovieSource) -> tuple[float, float]:
    """``(lo, hi)`` intensities mapped to 0 and 255, fixed for the whole movie.

    p1/p99.7 over a strided sample. The high percentile sits well above p99 on
    purpose: transients are the bright tail here, and clipping them flat is
    worse than a slightly dim background.
    """
    key = (str(src.path.resolve()), src.stat)
    cached = _windows.get(key)
    if cached is not None:
        return cached
    with _windows_lock:
        if key not in _windows:
            _windows[key] = _compute_window(src)
        return _windows[key]


def _compute_window(src: MovieSource) -> tuple[float, float]:
    mm = open_movie(src)
    stride = max(1, src.n_frames // _WINDOW_SAMPLES)
    sample = np.asarray(mm[::stride], dtype=np.float32)
    if sample.size == 0:
        return (0.0, 1.0)
    lo, hi = (float(v) for v in np.percentile(sample, [1.0, 99.7]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(sample.min()), float(sample.max())
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def clear_cache() -> None:
    """Drop pooled memmaps and cached windows. For tests."""
    with _pool_lock:
        _pool.clear()
    with _windows_lock:
        _windows.clear()


# ── reads ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BlockRequest:
    """A clamped, serve-able read. Never constructed directly — see
    :func:`clamp_request`, which is the only thing that decides what is legal."""

    start: int
    count: int
    x: int
    y: int
    w: int
    h: int
    ds: int

    @property
    def cols(self) -> int:
        return len(range(0, self.w, self.ds))

    @property
    def rows(self) -> int:
        return len(range(0, self.h, self.ds))


def clamp_request(src: MovieSource, *, start: int, count: int, x: int, y: int,
                  w: int, h: int, ds: int) -> BlockRequest:
    """Force a client's ask into something this movie can actually serve.

    Clamping rather than rejecting is deliberate: the client echoes the headers
    back into its ring buffer, so a request trimmed at an edge still lands as
    valid data instead of a hole it has to retry around.
    """
    ds = max(1, min(int(ds), 32))
    x = max(0, min(int(x), src.width - 1))
    y = max(0, min(int(y), src.height - 1))
    w = max(1, min(int(w), src.width - x))
    h = max(1, min(int(h), src.height - y))
    start = max(0, min(int(start), max(0, src.n_frames - 1)))
    count = max(1, min(int(count), MAX_COUNT, src.n_frames - start))

    cols = len(range(0, w, ds))
    rows = len(range(0, h, ds))
    per_frame = max(1, cols * rows)
    count = max(1, min(count, MAX_BYTES // per_frame))
    return BlockRequest(start=start, count=count, x=x, y=y, w=w, h=h, ds=ds)


def read_block(src: MovieSource, req: BlockRequest) -> np.ndarray:
    """``(count, rows, cols)`` uint8, windowed by :func:`display_window`.

    One memmap slice and one vectorised rescale — no per-frame Python. The
    decimation is a stride on the slice, so decimated reads also touch less of
    the page cache, not just less of the wire.
    """
    mm = open_movie(src)
    block = np.asarray(
        mm[req.start:req.start + req.count,
           req.y:req.y + req.h:req.ds,
           req.x:req.x + req.w:req.ds],
        dtype=np.float32,
    )
    lo, hi = display_window(src)
    block -= lo
    block *= 255.0 / (hi - lo)
    np.clip(block, 0.0, 255.0, out=block)
    return np.ascontiguousarray(block.astype(np.uint8))
