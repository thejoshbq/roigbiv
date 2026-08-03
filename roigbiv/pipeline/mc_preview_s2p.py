"""Live-preview hooks for the ``phasecorr`` (Suite2p) motion-correction backend.

Suite2p exposes no callbacks: :func:`suite2p.run_s2p.run_s2p` is a monolith and
its registration loop (``suite2p/registration/register.py``) neither yields nor
notifies. But that loop calls the *module-level* ``register_frames`` once per
batch and unpacks eight values from it, so wrapping that global gives us —
exactly, and for free — the raw input frames, the corrected output, the
per-frame rigid shifts, and the phase-correlation peak ``cmax``. That last one
is a registration-confidence signal available nowhere in the pipeline's outputs.

The alternative (tailing ``data.bin`` against the ``"Registered k/n"`` stdout
line) was rejected: it yields only a frame *index*, costs two extra full-frame
memmap reads per emit, and silently shows black frames when
``mc_s2p_two_step_registration`` is on — that flips ``keep_movie_raw``, leaving
``data.bin`` zero-filled ahead of the registration cursor.

Two things the wrapper cannot see are covered by a thin stdout line tap: the
TIFF→binary conversion, and the ``nimg_init``-frame reference build — a
multi-second dead zone before the first batch that would otherwise render as a
stalled UI.

Safety
------
Nothing here changes Suite2p's behavior. The wrapper copies, records, and
delegates; the tap forwards every byte to the stdout it replaced. Both are
restored by identity. The wrapper is installed only after its signature is
feature-detected, so a Suite2p upgrade degrades the preview (``phase =
"unsupported"``) instead of breaking registration.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import re
import sys
import threading

import numpy as np

# Suite2p's registration progress lines. ``print`` in register.py is the plain
# builtin, so these arrive at our tap as soon as they are emitted.
_RE_REGISTERED = re.compile(r"^Registered (\d+)/(\d+) in")
_RE_CONVERTING = re.compile(r"^(\d+) frames of binary")

# Line prefix → phase. Matched by ``startswith``, first hit wins, so the two
# "NOTE: not running registration, ..." variants must precede their shared
# prefix. Suite2p distinguishes them and so must we: one means the input was
# already corrected, the other that a previous run registered this plane.
# ("NOTE: not registered / registration forced ..." is deliberately absent — it
# announces that registration *will* run, not that it was skipped.)
_PHASE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("----------- REGISTRATION", "registering"),
    ("Reference frame,", "registering"),
    ("List of reference frames", "building_reference"),
    ("NOTE: not running registration, plane already registered",
     "skipped_resume"),
    ("NOTE: not running registration, ops['do_registration']=0",
     "skipped_precorrected"),
    ("----------- ROI DETECTION", "done"),
)

# Only one preview may wrap the process-global Suite2p module at a time. The UI
# serializes runs with a process-level gate and batch mode uses separate
# processes, so this should never contend; if it somehow does, refuse rather
# than nest (nested restores would leak a wrapper).
_install_lock = threading.Lock()
_installed = False


def classify_line(line: str) -> tuple[str | None, tuple[int, int] | None]:
    """Map one Suite2p stdout line to ``(phase, (n_done, n_total))``.

    Either element may be ``None``. Pure function — the tap's whole parsing
    surface, so it can be tested without Suite2p or a running pipeline.
    """
    line = line.strip()
    if not line:
        return None, None
    m = _RE_REGISTERED.match(line)
    if m:
        return "registering", (int(m.group(1)), int(m.group(2)))
    m = _RE_CONVERTING.match(line)
    if m:
        return "converting", None
    for prefix, phase in _PHASE_PREFIXES:
        if line.startswith(prefix):
            return phase, None
    return None, None


class _PhaseTap(io.TextIOBase):
    """Line-buffered ``sys.stdout`` shim that also reports Suite2p phases.

    Chains to whatever stdout it replaced — under
    :mod:`roigbiv.pipeline.batch` that is already a queue shim, so
    ``sys.__stdout__`` would silently drop the worker's logs.

    ``print(x)`` issues two ``write`` calls (payload, then the newline), hence
    the buffering.
    """

    def __init__(self, prev, writer):
        self._prev = prev
        self._writer = writer
        self._buf = ""

    def write(self, s):  # noqa: D102 — file-like protocol
        try:
            n = self._prev.write(s)
        except Exception:
            n = len(s)
        try:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._consume(line)
        except Exception:
            self._buf = ""
        return n

    def flush(self):  # noqa: D102
        with contextlib.suppress(Exception):
            self._prev.flush()

    def isatty(self):  # noqa: D102
        try:
            return self._prev.isatty()
        except Exception:
            return False

    def _consume(self, line: str) -> None:
        from roigbiv.pipeline.mc_preview import TERMINAL_PHASES

        phase, progress = classify_line(line)
        if progress is not None:
            n_done, n_total = progress
            self._writer.set_total(n_total, self._writer.pass_index)
            self._writer.set_progress(n_done)
        if phase is None or phase == self._writer.phase:
            return
        # Suite2p prints "ROI DETECTION" after *any* registration outcome, so
        # letting it overwrite a terminal phase would relabel a skipped run as a
        # completed one and throw away the reason the card is empty.
        if self._writer.phase in TERMINAL_PHASES:
            return
        self._writer.set_phase(phase)


def _supports_hook(fn) -> bool:
    """Whether ``fn`` looks like Suite2p 0.14's ``register_frames``."""
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return False
    return params[:2] == ["refAndMasks", "frames"]


@contextlib.contextmanager
def suite2p_preview_hooks(writer):
    """Feed ``writer`` from a Suite2p registration run happening inside.

    A no-op (yielding immediately) when ``writer`` is None/disabled, when
    Suite2p's API has drifted, or when another install is already active.
    """
    global _installed

    if writer is None or not getattr(writer, "enabled", False):
        yield
        return

    try:
        from suite2p.registration import register as _reg
    except Exception:
        writer.set_phase("unsupported", note="suite2p registration module missing")
        yield
        return

    if not _supports_hook(getattr(_reg, "register_frames", None)):
        writer.set_phase(
            "unsupported",
            note="suite2p register_frames signature not recognised; "
                 "live preview disabled for this run")
        yield
        return

    with _install_lock:
        if _installed:
            writer.set_phase(
                "unsupported",
                note="another motion-correction preview is already active")
            yield
            return
        _installed = True

    orig_register = _reg.register_frames
    orig_compute = getattr(_reg, "compute_reference_and_register_frames", None)
    prev_stdout = sys.stdout
    tap = _PhaseTap(prev_stdout, writer)
    depth = threading.local()

    def register_frames_hook(refAndMasks, frames, *args, **kwargs):
        # Guard the nZ > 1 self-recursion: only the outermost call is a batch.
        if getattr(depth, "busy", False):
            return orig_register(refAndMasks, frames, *args, **kwargs)
        depth.busy = True
        try:
            raw = None
            mid = 0
            try:
                if writer.should_emit() and len(frames) > 0:
                    mid = len(frames) // 2
                    # MUST copy: register_frames shifts frames in place
                    # (`frame[:] = rigid.shift_frame(...)`), and when the caller
                    # is a BinaryFile the array is a memmap view. Without this
                    # the "raw" pane would show the corrected frame.
                    raw = np.array(frames[mid], dtype=np.float32)
            except Exception:
                raw = None

            out = orig_register(refAndMasks, frames, *args, **kwargs)

            try:
                _after_batch(writer, out, raw, mid)
            except Exception:
                pass
            return out
        finally:
            depth.busy = False

    def compute_hook(f_align_in, *args, **kwargs):
        try:
            # Suite2p's two-step registration calls this twice, and the second
            # pass restarts the frame counter at zero. A non-zero count here
            # means we are entering that second pass.
            second_pass = writer.n_done > 0
            writer.set_total(int(f_align_in.shape[0]),
                             writer.pass_index + 1 if second_pass else 0)
        except Exception:
            pass
        return orig_compute(f_align_in, *args, **kwargs)

    _reg.register_frames = register_frames_hook
    if orig_compute is not None:
        _reg.compute_reference_and_register_frames = compute_hook
    sys.stdout = tap
    try:
        yield
    finally:
        _reg.register_frames = orig_register
        if orig_compute is not None:
            _reg.compute_reference_and_register_frames = orig_compute
        if sys.stdout is tap:
            sys.stdout = prev_stdout
        with contextlib.suppress(Exception):
            tap.flush()
        with _install_lock:
            _installed = False


def _after_batch(writer, out, raw, mid: int) -> None:
    """Record shifts (always) and a frame pair (when one was snapshotted).

    ``out`` is Suite2p's 8-tuple
    ``(frames, ymax, xmax, cmax, ymax1, xmax1, cmax1, zest)``.

    Suite2p prints ``"Registered k/n"`` *after* this returns, so ``writer.n_done``
    is still the previous batch's total — which is exactly this batch's first
    frame index.
    """
    frames, ymax, xmax, cmax = out[0], out[1], out[2], out[3]
    base = writer.n_done
    writer.record_shifts(base, ymax, xmax, cmax)
    if raw is not None and mid < len(frames):
        writer.emit(raw, np.asarray(frames[mid], dtype=np.float32),
                    frame_index=base + mid, n_done=base + len(frames))
