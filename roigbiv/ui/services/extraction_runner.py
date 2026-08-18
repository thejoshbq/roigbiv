"""Background runner for Discovery-triggered trace extraction.

Runs :func:`roigbiv.pipeline.discovery_extract.extract_from_merged_masks` in a
daemon thread so the Discovery page can poll it the same way "Run centroid
discovery" polls :class:`~roigbiv.ui.services.pipeline_runner.PipelineRunner`.

Deliberately separate from ``PipelineRunner``/``TrackingRunner``: it has none
of their detection-run or registry-pass concerns. Unlike
:class:`~roigbiv.ui.services.tracking_runner.TrackingRunner`, which shares
``pipeline_runner._pipeline_gate`` because ROICaT alignment is as GPU-hungry
as detection, this runner does **not** acquire that gate — median/mode
extraction is pure CPU/disk ``data.bin`` streaming with no CUDA involvement,
so serializing it against GPU-heavy pipeline/tracking runs would be an
unnecessary wait. Its own per-session lock is enough to prevent
double-extraction on the same FOV.
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from roigbiv.pipeline.discovery_extract import extract_from_merged_masks
from roigbiv.pipeline.types import PipelineConfig

_MAX_LOG_LINES = 500


@dataclass
class ExtractionSnapshot:
    """Serializable snapshot for the Discovery page's interval callback."""

    active: bool
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    logs: list[str] = field(default_factory=list)
    error: Optional[str] = None
    bundle_dir: Optional[str] = None
    stem: Optional[str] = None


class ExtractionRunner:
    """Single-slot background runner for one FOV's trace extraction."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._active = False
        self._started_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._logs: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self._error: Optional[str] = None
        self._bundle_dir: Optional[Path] = None
        self._stem: Optional[str] = None
        self._last_accessed = time.monotonic()

    def start(
        self,
        fov_output_dir: Path,
        stem: str,
        stats: tuple[str, ...] = (),
        cfg: Optional[PipelineConfig] = None,
    ) -> bool:
        """Kick off extraction for one FOV.

        Returns ``True`` when started, ``False`` when this session already
        has an extraction running.
        """
        with self._lock:
            if self._active:
                return False
            self._reset_locked()
            self._active = True
            self._started_at = time.time()
            self._stem = stem

        t = threading.Thread(
            target=self._run,
            args=(Path(fov_output_dir), tuple(stats), cfg),
            name="roigbiv-ui-extraction",
            daemon=True,
        )
        self._thread = t
        t.start()
        return True

    def snapshot(self) -> ExtractionSnapshot:
        with self._lock:
            return ExtractionSnapshot(
                active=self._active,
                started_at=self._started_at,
                completed_at=self._completed_at,
                logs=list(self._logs),
                error=self._error,
                bundle_dir=str(self._bundle_dir) if self._bundle_dir else None,
                stem=self._stem,
            )

    # ── internals ─────────────────────────────────────────────────────────

    def _reset_locked(self) -> None:
        self._logs.clear()
        self._started_at = None
        self._completed_at = None
        self._error = None
        self._bundle_dir = None
        self._stem = None

    def _log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def _run(
        self,
        fov_output_dir: Path,
        stats: tuple[str, ...],
        cfg: Optional[PipelineConfig],
    ) -> None:
        label = ", ".join(("mean",) + stats)
        self._log(f"Extracting {label} for {fov_output_dir.name}...")
        try:
            bundle_dir = extract_from_merged_masks(
                fov_output_dir, cfg=cfg, stats=stats)
            self._log(f"Wrote {bundle_dir}")
            with self._lock:
                self._bundle_dir = bundle_dir
        except BaseException as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._logs.append(f"FATAL: {self._error}")
                for line in tb.strip().splitlines():
                    self._logs.append(line)
        finally:
            with self._lock:
                self._completed_at = time.time()
                self._active = False


_runners: dict[str, ExtractionRunner] = {}
_runners_lock = threading.Lock()


def get_extraction_runner() -> ExtractionRunner:
    """Return the :class:`ExtractionRunner` for the current browser session."""
    from roigbiv.ui.services.session import get_session_id

    sid = get_session_id()
    with _runners_lock:
        if sid not in _runners:
            _runners[sid] = ExtractionRunner()
    runner = _runners[sid]
    runner._last_accessed = time.monotonic()
    return runner
