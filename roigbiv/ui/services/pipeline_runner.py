"""Background pipeline runner for the Process page.

Runs :func:`roigbiv.pipeline.workspace.run_with_workspace` in a daemon thread
so the Dash callback that kicks it off returns immediately. Logs are buffered
in a thread-safe deque that the page polls on a ``dcc.Interval``.

Multi-user: each browser session gets its own :class:`PipelineRunner` keyed
on the Flask session UUID, so logs and results are fully isolated. A
process-level ``_pipeline_gate`` lock ensures only one pipeline run is active
at a time (preventing concurrent ``configure_registry_env()`` env-var races in
:mod:`roigbiv.pipeline.workspace`). Callers receive ``"busy"`` instead of a
silent failure when another session is running.
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from roigbiv.pipeline.workspace import (
    FOVRunResult,
    WorkspacePaths,
    run_with_workspace,
)

_MAX_LOG_LINES = 2000


@dataclass
class RunSnapshot:
    """Serializable snapshot for the Process page's interval callback."""

    active: bool
    started_at: Optional[float]
    completed_at: Optional[float]
    n_fovs: int
    n_done: int
    n_failed: int
    logs: list[str]
    error: Optional[str]
    results_summary: list[dict] = field(default_factory=list)


class PipelineRunner:
    """Single-slot background runner for workspace pipeline jobs."""

    def __init__(self, gate: threading.Lock) -> None:
        self._gate = gate
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._logs: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self._active: bool = False
        self._started_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._n_fovs: int = 0
        self._n_done: int = 0
        self._n_failed: int = 0
        self._error: Optional[str] = None
        self._results: list[FOVRunResult] = []
        self._last_accessed: float = time.monotonic()

    # ── control ───────────────────────────────────────────────────────────
    def start(self, workspace: WorkspacePaths, overrides: dict) -> bool | str:
        """Kick off a run.

        Returns:
          True    — run started successfully.
          False   — this session's runner is already active (shouldn't happen
                    if the UI disables the button, but guards re-entry).
          "busy"  — another session's pipeline is currently running.
        """
        if not self._gate.acquire(blocking=False):
            return "busy"
        try:
            with self._lock:
                if self._active:
                    self._gate.release()
                    return False
                self._reset_locked()
                self._active = True
                self._started_at = time.time()
                self._n_fovs = len(workspace.tifs)
        except Exception:
            self._gate.release()
            raise
        t = threading.Thread(
            target=self._run,
            args=(workspace, overrides),
            name="roigbiv-ui-pipeline",
            daemon=True,
        )
        self._thread = t
        t.start()
        return True

    def snapshot(self) -> RunSnapshot:
        with self._lock:
            return RunSnapshot(
                active=self._active,
                started_at=self._started_at,
                completed_at=self._completed_at,
                n_fovs=self._n_fovs,
                n_done=self._n_done,
                n_failed=self._n_failed,
                logs=list(self._logs),
                error=self._error,
                results_summary=[self._summarize(r) for r in self._results],
            )

    def results(self) -> list[FOVRunResult]:
        with self._lock:
            return list(self._results)

    # ── internals ─────────────────────────────────────────────────────────
    def _reset_locked(self) -> None:
        self._logs.clear()
        self._started_at = None
        self._completed_at = None
        self._n_fovs = 0
        self._n_done = 0
        self._n_failed = 0
        self._error = None
        self._results = []

    def _log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def _run(self, workspace: WorkspacePaths, overrides: dict) -> None:
        try:
            try:
                results = run_with_workspace(
                    workspace, overrides, log_cb=self._append_and_tally,
                )
            except BaseException as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._logs.append(f"FATAL: {self._error}")
                    for line in tb.strip().splitlines():
                        self._logs.append(line)
                results = []

            with self._lock:
                self._results = results
                self._n_done = sum(1 for r in results if r.error is None)
                self._n_failed = sum(1 for r in results if r.error is not None)
                self._completed_at = time.time()
                self._active = False
        finally:
            self._gate.release()

    def _append_and_tally(self, line: str) -> None:
        """Log callback that also counts completed FOVs from ``pipeline OK``."""
        self._log(line)
        low = line.lstrip()
        if low.startswith("pipeline OK"):
            with self._lock:
                self._n_done += 1

    @staticmethod
    def _summarize(r: FOVRunResult) -> dict:
        return {
            "stem": r.tif.stem.replace("_mc", ""),
            "tif": str(r.tif),
            "output_dir": str(r.output_dir),
            "duration_s": r.duration_s,
            "error": r.error,
            "roi_counts": dict(r.roi_counts),
            "registry_decision": (
                (r.registry or {}).get("decision") if r.registry else None
            ),
            "registry_fov_id": (
                (r.registry or {}).get("fov_id") if r.registry else None
            ),
        }


_runners: dict[str, PipelineRunner] = {}
_runners_lock = threading.Lock()
_pipeline_gate = threading.Lock()


def get_pipeline_runner() -> PipelineRunner:
    """Return the :class:`PipelineRunner` for the current browser session."""
    from roigbiv.ui.services.session import get_session_id
    sid = get_session_id()
    with _runners_lock:
        if sid not in _runners:
            _runners[sid] = PipelineRunner(_pipeline_gate)
    runner = _runners[sid]
    runner._last_accessed = time.monotonic()
    return runner
