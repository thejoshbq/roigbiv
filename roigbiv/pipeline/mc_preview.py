"""Live motion-correction preview sidecar.

While a FOV is being registered, the active backend hands this writer a
downsampled ``(raw, corrected)`` pair for the *same* frame every few hundred
milliseconds. The pair lands in ``{output_dir}/mc_preview/`` as seq-numbered
PNGs alongside a small ``state.json``, which the Dash UI polls over plain HTTP
(:mod:`roigbiv.ui.routes.mc_preview`) to render the FOV being corrected in real
time. When the run ends the accumulated records are a scrubbable timeline, which
is what makes an A/B of two MC backends on the same FOV possible after the fact.

Disk — rather than an in-process queue — is the channel because the producer is
in a different process depending on how the pipeline was launched: a daemon
thread of the Dash process (UI), a ``spawn`` worker (:mod:`roigbiv.pipeline.batch`),
or a bare CLI process. A file works for all three, survives a browser reload
mid-run, and *is* the post-hoc artifact rather than a second mechanism for it.

Contract with the pipeline
--------------------------
This module is **strictly diagnostic**. It never touches the registered data,
and no method here raises: every write is guarded, and after
``_MAX_WRITE_FAILURES`` consecutive failures (a full disk, a read-only mount)
the writer disables itself permanently and records ``phase="degraded"``. A
registration run must produce byte-identical output with the preview on or off.

Ordering invariant
------------------
PNG filenames carry the sequence number and are never overwritten; ``state.json``
is ``os.replace``d **last**, so the ``seq`` a reader sees always points at files
that are already complete on disk. Overwriting a fixed ``raw.png``/``corr.png``
pair would let the UI's A/B blink compare two *different* frames — exactly the
artifact that would make you distrust the registration.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from roigbiv.pipeline.diskguard import ensure_free_space

SCHEMA_VERSION = 1

# Consecutive failed writes after which the writer gives up for good. Small on
# purpose: a preview that keeps retrying a full disk inside the registration
# loop costs the run real time.
_MAX_WRITE_FAILURES = 3

# Quality metrics run on the *full-resolution* corrected frame (they are
# scale-dependent — see the note in :meth:`MCPreviewWriter.emit`), which costs
# tens of milliseconds inline in the MC loop. Throttle them well below the image
# cadence so the overhead stays ~1%; the readout still updates visibly.
_METRICS_MIN_INTERVAL_S = 2.0

# Cap on the shift trace embedded in state.json. The full-resolution trace goes
# to shifts.npz on close; state.json only needs enough points to draw.
_TRACE_POINTS = 500

# Bytes reserved up front so a full disk fails at open, not mid-loop.
_RESERVE_BYTES = 64 * 1024 * 1024

#: Terminal phases — the UI enables the scrubber once ``phase`` is one of these.
TERMINAL_PHASES = frozenset({
    "done", "skipped_precorrected", "skipped_resume", "unsupported",
    "degraded", "aborted",
})


def preview_dir(output_dir) -> Path:
    """Sidecar directory for a FOV's output dir."""
    return Path(output_dir) / "mc_preview"


def downsample(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Integer-stride decimation so the long edge is <= ``max_dim``.

    Striding rather than area-averaging is deliberate: averaging would smooth
    away exactly the high-frequency edge detail the preview exists to show, and
    would make residual motion look better than it is.
    """
    a = np.asarray(img)
    if max_dim <= 0:
        return a
    long_edge = max(a.shape[0], a.shape[1])
    stride = max(1, int(np.ceil(long_edge / float(max_dim))))
    return a[::stride, ::stride]


def encode_png(img: np.ndarray, lo: float, hi: float) -> bytes:
    """Map ``img`` through the fixed window ``[lo, hi]`` to 8-bit PNG bytes."""
    import io as _io

    from PIL import Image

    a = np.asarray(img, dtype=np.float32)
    span = float(hi) - float(lo)
    if span <= 0:
        span = 1.0
    a = (a - float(lo)) / span
    u8 = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = _io.BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _decimate(values: list, n: int) -> list:
    """Evenly subsample ``values`` down to at most ``n`` points."""
    if len(values) <= n:
        return list(values)
    idx = np.linspace(0, len(values) - 1, n).astype(int)
    return [values[i] for i in idx]


class MCPreviewWriter:
    """Writes the live preview sidecar for one FOV.

    Use as a context manager around the motion-correction call; ``__exit__``
    stamps the terminal phase (``done`` / ``aborted``) and flushes
    ``shifts.npz``. Disabled instances are inert, so backends can call the
    methods unconditionally.
    """

    def __init__(
        self,
        output_dir,
        *,
        stem: str,
        backend: str,
        enabled: bool = True,
        max_dim: int = 512,
        min_interval_s: float = 0.4,
        max_records: int = 300,
        metrics: bool = True,
        avg: bool = True,
    ) -> None:
        self.dir = preview_dir(output_dir)
        self.stem = stem
        self.backend = backend
        self.enabled = bool(enabled)
        self.max_dim = int(max_dim)
        self.min_interval_s = float(min_interval_s)
        self.max_records = max(1, int(max_records))
        self.want_metrics = bool(metrics)
        self.want_avg = bool(avg)

        self._seq = -1
        self._records: list[int] = []
        # Newest emit when it fell between retained records: kept on disk so the
        # live pane always has a frame, retired as soon as it is superseded.
        self._transient: int | None = None
        self._ordinal = 0
        self._stride = 1
        self._last_emit_t = -1e9
        self._last_metrics_t = -1e9
        self._failures = 0
        self._norm: tuple[float, float] | None = None
        # Running mean of every raw frame previewed so far this run (the same
        # throttled ~2.5 Hz stream that feeds the raw/corrected panes, not
        # every frame in the registration batch loop).
        self._avg_mean: np.ndarray | None = None
        self._avg_n: int = 0

        self._phase = "starting"
        self._note: str | None = None
        self._n_done = 0
        self._n_total = 0
        self._pass_index = 0
        self._frame_index = 0
        self._shape: tuple[int, int] | None = None
        self._preview_shape: tuple[int, int] | None = None
        self._metrics: dict | None = None
        self._started_at = time.time()

        # Full-resolution shift trace, appended per batch (no I/O).
        self._sh_frame: list[int] = []
        self._sh_y: list[float] = []
        self._sh_x: list[float] = []
        self._sh_c: list[float] = []

        if self.enabled:
            self._open()

    # ── lifecycle ────────────────────────────────────────────────────────

    def _open(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            # Wipe a prior run's records so the scrubber can't interleave two
            # different registrations of the same FOV.
            for p in self.dir.iterdir():
                if p.is_file():
                    p.unlink()
            ensure_free_space(self.dir / "state.json", _RESERVE_BYTES,
                              label="mc_preview sidecar")
            self._write_state()
        except Exception:
            # Cannot even open: stay silent and inert for the rest of the run.
            self.enabled = False

    def __enter__(self) -> "MCPreviewWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.enabled:
            if exc_type is not None:
                self.set_phase("aborted", note=f"{exc_type.__name__}: {exc}")
            elif self._phase not in TERMINAL_PHASES:
                self.set_phase("done")
            self._write_shifts()
        return False  # never swallow a pipeline exception

    # ── state ────────────────────────────────────────────────────────────

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def records(self) -> list[int]:
        """Scrubbable timeline: retained records plus the live frame."""
        if self._transient is None:
            return list(self._records)
        return sorted(set(self._records) | {self._transient})

    @property
    def n_done(self) -> int:
        """Frames registered so far in the current pass."""
        return self._n_done

    @property
    def pass_index(self) -> int:
        return self._pass_index

    def set_phase(self, phase: str, *, note: str | None = None) -> None:
        """Record a coarse progress phase (see module docstring for the set)."""
        if not self.enabled:
            return
        self._phase = phase
        if note is not None:
            self._note = note
        self._write_state()

    def set_progress(self, n_done: int) -> None:
        """Update the frame counter without writing an image record."""
        if not self.enabled:
            return
        self._n_done = int(n_done)
        self._write_state()

    def set_total(self, n_total: int, pass_index: int = 0) -> None:
        """Declare the frame count for the current registration pass.

        ``pass_index`` distinguishes the two passes of Suite2p's two-step
        registration, whose frame counter restarts from zero.
        """
        if not self.enabled:
            return
        self._n_total = int(n_total)
        if pass_index != self._pass_index:
            self._pass_index = int(pass_index)
            self._n_done = 0
        self._write_state()

    def record_shifts(self, frame_index: int, ys, xs, cmax=None) -> None:
        """Append per-frame rigid shifts for a whole batch. No I/O.

        Called on every batch regardless of the image throttle, so the trace
        (and ``shifts.npz``) keeps full temporal resolution even though only a
        few frames per second are rendered. ``cmax`` is the phase-correlation
        peak — a registration-confidence signal available nowhere else in the
        pipeline's outputs.
        """
        if not self.enabled:
            return
        try:
            ys = np.atleast_1d(np.asarray(ys, dtype=np.float32))
            xs = np.atleast_1d(np.asarray(xs, dtype=np.float32))
            n = min(ys.size, xs.size)
            if n == 0:
                return
            cs = (np.atleast_1d(np.asarray(cmax, dtype=np.float32))
                  if cmax is not None else np.full(n, np.nan, dtype=np.float32))
            self._sh_frame.extend(range(int(frame_index), int(frame_index) + n))
            self._sh_y.extend(ys[:n].tolist())
            self._sh_x.extend(xs[:n].tolist())
            self._sh_c.extend(cs[:n].tolist() if cs.size >= n
                              else [float("nan")] * n)
        except Exception:
            pass  # a broken trace must never stop registration

    # ── emit ─────────────────────────────────────────────────────────────

    def should_emit(self) -> bool:
        """Whether enough wall-clock has passed to render another frame.

        Deliberately syscall-free (a float compare) — this runs in the MC batch
        loop and is called far more often than it returns True.
        """
        if not self.enabled:
            return False
        return (time.monotonic() - self._last_emit_t) >= self.min_interval_s

    def emit(self, raw, corrected, *, frame_index: int, n_done: int) -> None:
        """Write one raw/corrected (and optionally running-average) record.

        ``raw`` and ``corrected`` must be the *same* frame index — the UI's A/B
        blink is meaningless otherwise. Callers holding a view into a buffer the
        registration mutates in place must copy before calling.
        """
        if not self.enabled:
            return
        self._last_emit_t = time.monotonic()
        try:
            self._emit_inner(raw, corrected, frame_index, n_done)
            self._failures = 0
        except Exception:
            self._failures += 1
            if self._failures >= _MAX_WRITE_FAILURES:
                # Stop trying, but leave a breadcrumb explaining the empty card.
                self._phase = "degraded"
                self._note = "preview writes failing; disabled for this run"
                try:
                    self._write_state()
                except Exception:
                    pass
                self.enabled = False

    def _emit_inner(self, raw, corrected, frame_index: int, n_done: int) -> None:
        raw = np.asarray(raw, dtype=np.float32)
        corr = np.asarray(corrected, dtype=np.float32)
        self._shape = (int(raw.shape[0]), int(raw.shape[1]))
        self._frame_index = int(frame_index)
        self._n_done = int(n_done)

        # Freeze the display window on the first frame we see, and reuse it for
        # every frame and both panes thereafter: a per-frame percentile stretch
        # would make the blink flicker on brightness, which reads as motion.
        if self._norm is None:
            lo, hi = (float(v) for v in np.percentile(raw, [1.0, 99.0]))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(np.min(raw)) if np.isfinite(raw).any() else 0.0
                hi = lo + 1.0   # flat frame: any window is arbitrary, avoid /0
            self._norm = (lo, hi)
        lo, hi = self._norm

        # Metrics on the FULL-resolution corrected frame. lap_var_smooth and
        # banding_score are both scale-dependent, so computing them on the
        # downsampled preview would print numbers that contradict the post-hoc
        # panel for the same FOV.
        now = time.monotonic()
        if (self.want_metrics
                and (now - self._last_metrics_t) >= _METRICS_MIN_INTERVAL_S):
            self._last_metrics_t = now
            from roigbiv.pipeline.mc_metrics import compute_metrics
            self._metrics = compute_metrics(corr)

        raw_ds = downsample(raw, self.max_dim)
        corr_ds = downsample(corr, self.max_dim)
        self._preview_shape = (int(raw_ds.shape[0]), int(raw_ds.shape[1]))

        seq = self._seq + 1
        self._write_bytes(f"raw_{seq:06d}.png", encode_png(raw_ds, lo, hi))
        self._write_bytes(f"corr_{seq:06d}.png", encode_png(corr_ds, lo, hi))
        if self.want_avg:
            if self._avg_mean is None:
                self._avg_mean = raw_ds.astype(np.float32).copy()
            else:
                self._avg_n += 1
                # Incremental mean: reuses the same frozen (lo, hi) window as
                # the raw/corrected panes, so brightness reads consistently
                # across all three rather than needing its own stretch.
                self._avg_mean += (raw_ds - self._avg_mean) / (self._avg_n + 1)
            self._write_bytes(f"avg_{seq:06d}.png", encode_png(self._avg_mean, lo, hi))

        self._seq = seq
        superseded, self._transient = self._transient, None
        if self._ordinal % self._stride == 0:
            self._records.append(seq)
            self._prune()
        else:
            self._transient = seq
        self._ordinal += 1
        if superseded is not None and superseded != self._transient:
            self._retire(superseded)
        self._write_state()  # last: the seq it names is now complete on disk

    def _prune(self) -> None:
        """Halve the retained timeline whenever it exceeds the record budget.

        Dropping every other record *and doubling the sampling stride* keeps
        coverage uniform across the whole run at a bounded footprint, without
        needing to know the movie length up front. (Halving alone, with every
        emit still retained, would collapse the timeline onto the last few
        seconds of the run.)
        """
        while len(self._records) > self.max_records:
            keep = self._records[::2]
            dropped = [s for s in self._records if s not in set(keep)]
            self._records = keep
            self._stride *= 2
            for seq in dropped:
                if seq == self._seq:
                    # Never unlink what the live pane is pointing at.
                    self._transient = seq
                else:
                    self._retire(seq)

    def _retire(self, seq: int) -> None:
        for kind in ("raw", "corr", "avg"):
            try:
                (self.dir / f"{kind}_{seq:06d}.png").unlink()
            except OSError:
                pass

    # ── disk ─────────────────────────────────────────────────────────────

    def _write_bytes(self, name: str, payload: bytes) -> None:
        tmp = self.dir / (name + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, self.dir / name)

    def _valid_crop_frac(self) -> list[float] | None:
        """Fraction-of-frame rectangle unaffected by the ``np.roll`` edge wrap.

        Registration shifts frames with ``np.roll`` (suite2p ``rigid.shift_frame``),
        so pixels rolled past an edge reappear on the opposite one. Reporting the
        union of valid rows/cols lets the UI outline it, otherwise the wrapped
        strip reads as a registration artifact.
        """
        if not self._sh_y or self._shape is None:
            return None
        Ly, Lx = self._shape
        dy = np.asarray(self._sh_y, dtype=np.float32)
        dx = np.asarray(self._sh_x, dtype=np.float32)
        # roll by (-dy, -dx): dy > 0 invalidates the bottom, dy < 0 the top.
        top = float(max(0.0, -float(dy.min())))
        bottom = float(max(0.0, float(dy.max())))
        left = float(max(0.0, -float(dx.min())))
        right = float(max(0.0, float(dx.max())))
        x0, x1 = left / Lx, (Lx - right) / Lx
        y0, y1 = top / Ly, (Ly - bottom) / Ly
        if x1 <= x0 or y1 <= y0:
            return None
        return [round(x0, 5), round(y0, 5), round(x1, 5), round(y1, 5)]

    def _state(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "stem": self.stem,
            "backend": self.backend,
            "phase": self._phase,
            "note": self._note,
            "seq": self._seq,
            "records": self.records,
            "frame_index": self._frame_index,
            "n_done": self._n_done,
            "n_total": self._n_total,
            "pass_index": self._pass_index,
            "plane": 0,
            "pid": os.getpid(),
            "started_at": self._started_at,
            "updated_at": time.time(),
            "shape": list(self._shape) if self._shape else None,
            "preview_shape": (list(self._preview_shape)
                              if self._preview_shape else None),
            "norm": list(self._norm) if self._norm else None,
            "has_avg": bool(self.want_avg),
            "valid_crop_frac": self._valid_crop_frac(),
            "live_metrics": self._metrics,
            "shifts": {
                "frame": _decimate(self._sh_frame, _TRACE_POINTS),
                "y": [round(v, 4) for v in _decimate(self._sh_y, _TRACE_POINTS)],
                "x": [round(v, 4) for v in _decimate(self._sh_x, _TRACE_POINTS)],
                "cmax": [None if not np.isfinite(v) else round(float(v), 5)
                         for v in _decimate(self._sh_c, _TRACE_POINTS)],
            },
        }

    def _write_state(self) -> None:
        if not self.enabled:
            return
        try:
            tmp = self.dir / "state.json.tmp"
            tmp.write_text(json.dumps(self._state()))
            os.replace(tmp, self.dir / "state.json")
        except Exception:
            pass  # state is a convenience; never break the run over it

    def _write_shifts(self) -> None:
        if not self._sh_frame:
            return
        try:
            np.savez(
                self.dir / "shifts.npz",
                frame=np.asarray(self._sh_frame, dtype=np.int32),
                y=np.asarray(self._sh_y, dtype=np.float32),
                x=np.asarray(self._sh_x, dtype=np.float32),
                cmax=np.asarray(self._sh_c, dtype=np.float32),
            )
        except Exception:
            pass


def writer_for(cfg, output_dir, *, stem: str, backend: str) -> MCPreviewWriter:
    """Build a writer from a :class:`~roigbiv.pipeline.types.PipelineConfig`."""
    return MCPreviewWriter(
        output_dir,
        stem=stem,
        backend=backend,
        enabled=getattr(cfg, "mc_preview_enabled", True),
        max_dim=getattr(cfg, "mc_preview_max_dim", 512),
        min_interval_s=getattr(cfg, "mc_preview_min_interval_s", 0.4),
        max_records=getattr(cfg, "mc_preview_max_records", 300),
        metrics=getattr(cfg, "mc_preview_metrics", True),
        avg=getattr(cfg, "mc_preview_avg", True),
    )
