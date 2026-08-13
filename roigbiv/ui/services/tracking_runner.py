"""Background runner for the cross-session tracking pass.

Runs :func:`roigbiv.pipeline.workspace.run_tracking` in a daemon thread so the
Track page can stream its log the same way the Pipeline page streams a run.

Deliberately separate from :class:`~roigbiv.ui.services.pipeline_runner.
PipelineRunner`: that class is built around ``FOVRunResult`` and carries
detection-run concerns (MC quality metrics, optics-confirmation pauses, Slack
overlays) that a registration pass has none of. It does share the same
process-wide GPU gate, because ROICaT's alignment and embedding steps are as
GPU-hungry as detection and must not overlap a pipeline run.
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from roigbiv.pipeline.workspace import (
    TrackingResult,
    WorkspacePaths,
    run_tracking,
)
from roigbiv.registry.config import RegistryConfig
from roigbiv.ui.services.pipeline_runner import _pipeline_gate

_MAX_LOG_LINES = 2000


@dataclass
class TrackingSnapshot:
    """Serializable snapshot for the Track page's interval callback."""

    active: bool
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    logs: list[str] = field(default_factory=list)
    error: Optional[str] = None
    results: list[dict] = field(default_factory=list)
    anomalies: Optional[dict] = None
    fov_ids: list[str] = field(default_factory=list)

    @property
    def n_tracked(self) -> int:
        return sum(1 for r in self.results if r.get("registry"))

    @property
    def n_skipped(self) -> int:
        return sum(1 for r in self.results if r.get("skipped"))

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if r.get("error"))


class TrackingRunner:
    """Single-slot background runner for a workspace's tracking pass."""

    def __init__(self, gate: threading.Lock) -> None:
        self._gate = gate
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._active = False
        self._started_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._logs: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self._error: Optional[str] = None
        self._results: list[TrackingResult] = []
        self._anomalies: Optional[dict] = None
        self._last_accessed = time.monotonic()

    def start(
        self,
        workspace: WorkspacePaths,
        overrides: Optional[dict] = None,
        registry_config: Optional[RegistryConfig] = None,
    ) -> bool | str:
        """Kick off a tracking pass.

        Returns ``True`` when started, ``False`` when this session already has
        one running, and ``"busy"`` when another session holds the GPU gate.
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
        except Exception:
            self._gate.release()
            raise

        t = threading.Thread(
            target=self._run,
            args=(workspace, dict(overrides or {}), registry_config),
            name="roigbiv-ui-tracking",
            daemon=True,
        )
        self._thread = t
        t.start()
        return True

    def snapshot(self) -> TrackingSnapshot:
        with self._lock:
            return TrackingSnapshot(
                active=self._active,
                started_at=self._started_at,
                completed_at=self._completed_at,
                logs=list(self._logs),
                error=self._error,
                results=[_summarize(r) for r in self._results],
                anomalies=(dict(self._anomalies)
                           if self._anomalies is not None else None),
                fov_ids=_fov_ids(self._results),
            )

    def results(self) -> list[TrackingResult]:
        with self._lock:
            return list(self._results)

    # ── internals ─────────────────────────────────────────────────────────

    def _reset_locked(self) -> None:
        self._logs.clear()
        self._started_at = None
        self._completed_at = None
        self._error = None
        self._results = []
        self._anomalies = None

    def _log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def _run(
        self,
        workspace: WorkspacePaths,
        overrides: dict,
        registry_config: Optional[RegistryConfig],
    ) -> None:
        try:
            try:
                results = run_tracking(
                    workspace, overrides,
                    log_cb=self._log,
                    registry_config=registry_config,
                )
            except BaseException as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._logs.append(f"FATAL: {self._error}")
                    for line in tb.strip().splitlines():
                        self._logs.append(line)
                results = []

            anomalies = self._collect_anomalies(results, registry_config)

            with self._lock:
                self._results = results
                self._anomalies = anomalies
                self._completed_at = time.time()
                self._active = False
        finally:
            self._gate.release()

    def _collect_anomalies(
        self,
        results: list[TrackingResult],
        registry_config: Optional[RegistryConfig],
    ) -> Optional[dict]:
        """Per-FOV anomaly counts, read back once the timeline is complete."""
        fov_ids = _fov_ids(results)
        if not fov_ids:
            return None
        try:
            from roigbiv.registry import build_store
            from roigbiv.registry.anomalies import cell_timeline
            from roigbiv.ui.services.registry_service import anomaly_payload

            store = build_store(registry_config)
            return {
                fov_id: anomaly_payload(cell_timeline(store, fov_id))
                for fov_id in fov_ids
            }
        except Exception as exc:  # noqa: BLE001
            self._log(f"anomaly report unavailable — {type(exc).__name__}: {exc}")
            return None


def _fov_ids(results: list[TrackingResult]) -> list[str]:
    """Distinct FOV ids touched by this pass, in first-seen order."""
    return list(dict.fromkeys(
        r.registry["fov_id"] for r in results
        if r.registry and r.registry.get("fov_id")
    ))


def _summarize(r: TrackingResult) -> dict:
    return {
        "stem": r.stem,
        "sequence_index": r.sequence_index,
        "output_dir": str(r.output_dir),
        "n_centroids": r.n_centroids,
        "n_overlapping_pairs": r.n_overlapping_pairs,
        "skipped": r.skipped,
        "error": r.error,
        "registry": r.registry,
        "decision": (r.registry or {}).get("decision"),
        "posterior": ((r.registry or {}).get("fov_posterior")
                      or (r.registry or {}).get("fov_sim")),
        "n_matched": (r.registry or {}).get("n_matched"),
        "n_new": (r.registry or {}).get("n_new"),
        "n_missing": (r.registry or {}).get("n_missing"),
        "match_errors": (r.registry or {}).get("match_errors") or [],
    }


_runners: dict[str, TrackingRunner] = {}
_runners_lock = threading.Lock()


def get_tracking_runner() -> TrackingRunner:
    """Return the :class:`TrackingRunner` for the current browser session."""
    from roigbiv.ui.services.session import get_session_id

    sid = get_session_id()
    with _runners_lock:
        if sid not in _runners:
            _runners[sid] = TrackingRunner(_pipeline_gate)
    runner = _runners[sid]
    runner._last_accessed = time.monotonic()
    return runner
