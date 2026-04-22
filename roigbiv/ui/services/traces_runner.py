"""Background runner for HITL trace re-extraction.

Mirrors the pattern in ``pipeline_runner.py``: a single-slot daemon thread
that calls :func:`roigbiv.pipeline.reextract.reextract_from_corrections` so
the Review-page callback returns immediately. Status is polled by the UI on
a ``dcc.Interval``.
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_MAX_LOG_LINES = 500


@dataclass
class ReextractSnapshot:
    active: bool
    started_at: Optional[float]
    completed_at: Optional[float]
    fov_output_dir: Optional[str]
    target_dir: Optional[str]
    logs: list[str]
    error: Optional[str]


class TracesRunner:
    """Single-slot background runner for trace re-extraction."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._logs: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self._active = False
        self._started_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._fov_output_dir: Optional[Path] = None
        self._target_dir: Optional[Path] = None
        self._error: Optional[str] = None

    def start(self, fov_output_dir: Path) -> bool:
        """Kick off a re-extract; returns ``False`` if one is already running."""
        with self._lock:
            if self._active:
                return False
            self._reset_locked()
            self._active = True
            self._started_at = time.time()
            self._fov_output_dir = Path(fov_output_dir)
        t = threading.Thread(
            target=self._run,
            args=(Path(fov_output_dir),),
            name="roigbiv-ui-reextract",
            daemon=True,
        )
        self._thread = t
        t.start()
        return True

    def snapshot(self) -> ReextractSnapshot:
        with self._lock:
            return ReextractSnapshot(
                active=self._active,
                started_at=self._started_at,
                completed_at=self._completed_at,
                fov_output_dir=(str(self._fov_output_dir)
                                if self._fov_output_dir else None),
                target_dir=(str(self._target_dir)
                            if self._target_dir else None),
                logs=list(self._logs),
                error=self._error,
            )

    def _reset_locked(self) -> None:
        self._logs.clear()
        self._started_at = None
        self._completed_at = None
        self._fov_output_dir = None
        self._target_dir = None
        self._error = None

    def _log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def _run(self, fov_output_dir: Path) -> None:
        from roigbiv.pipeline.reextract import reextract_from_corrections

        self._log(f"reextract: start {fov_output_dir}")
        try:
            target = reextract_from_corrections(fov_output_dir)
        except BaseException as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._logs.append(f"FATAL: {self._error}")
                for line in tb.strip().splitlines():
                    self._logs.append(line)
                self._completed_at = time.time()
                self._active = False
            return

        with self._lock:
            self._target_dir = target
            self._logs.append(f"reextract OK: wrote {target}")
            self._completed_at = time.time()
            self._active = False


_runner: Optional[TracesRunner] = None
_runner_lock = threading.Lock()


def get_traces_runner() -> TracesRunner:
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = TracesRunner()
    return _runner
