"""Background pipeline runner for the Process page.

Runs :func:`roigbiv.pipeline.workspace.run_with_workspace` in a daemon thread
so the Dash callback that kicks it off returns immediately. Logs are buffered
in a thread-safe deque that the page polls on a ``dcc.Interval``.

Multi-user: each browser session gets its own :class:`PipelineRunner` keyed
on the Flask session UUID, so logs and results are fully isolated. Each runner
receives the session's :class:`~roigbiv.registry.config.RegistryConfig` so
registry operations never touch ``os.environ`` and sessions can coexist safely.

A process-level ``_pipeline_gate`` lock serializes pipeline runs for GPU safety:
the RTX 5080 cannot service two concurrent pipeline jobs without CUDA OOM risk.
Callers receive ``"busy"`` instead of a silent failure when another session is
running.
"""
from __future__ import annotations

import re
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from roigbiv.pipeline.workspace import (
    FOVRunResult,
    WorkspacePaths,
    run_with_workspace,
)
from roigbiv.registry.config import RegistryConfig

_MAX_LOG_LINES = 2000

# ── stage-marker derivation ──────────────────────────────────────────────────
# Pipeline stages are emitted as ``fmt.stage_header(n, label)`` lines of the
# form ``--- Stage {n}: {label} ----``. In batch mode each line is prefixed
# with ``[FOV i/n] `` (see workspace._process_one). We map the ``{n}`` token to
# a short human label for the Run-status banner. Foundation has no stage_header
# (it runs first), so a run begins on the Foundation label until Stage 1 fires.
_FOUNDATION_STAGE = "Foundation · motion correction"

# Ordered (token -> label). The token is matched exactly against the value
# captured between "Stage " and ":".
_STAGE_LABELS: dict[str, str] = {
    "1": "Stage 1 · Cellpose detection",
    "1→S": "Source subtraction",
    "2": "Stage 2 · Temporal detection",
    "2→S": "Source subtraction",
    "3": "Stage 3 · Template sweep",
    "3→S": "Source subtraction",
    "4": "Stage 4 · Tonic search",
    "Post": "Trace extraction + QC",
    "Summary": "Detection complete",
}

# Optional leading "[FOV i/n] " prefix, then the stage marker.
_FOV_PREFIX_RE = re.compile(r"^\[FOV\s+(\d+/\d+)\]\s*")
_STAGE_MARKER_RE = re.compile(r"---\s*Stage\s+(\S+):")


def _derive_stage(line: str) -> Optional[str]:
    """Return a human stage label if ``line`` is a stage marker, else None.

    Captures any ``[FOV i/n]`` batch prefix so the banner can read e.g.
    ``FOV 2/5 · Stage 3 · Template sweep``.
    """
    fov_prefix = ""
    m_fov = _FOV_PREFIX_RE.match(line)
    if m_fov:
        fov_prefix = f"FOV {m_fov.group(1)} · "
        line = line[m_fov.end():]
    m = _STAGE_MARKER_RE.search(line)
    if not m:
        return None
    label = _STAGE_LABELS.get(m.group(1))
    if label is None:
        return None
    return f"{fov_prefix}{label}"


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
    current_stage: Optional[str] = None
    n_awaiting: int = 0          # FOVs paused for optics confirmation
    overrides: Optional[dict] = None   # config the run actually launched with
    stopping: bool = False       # stop requested, finishing current stage
    stopped: bool = False        # run ended after a cooperative stop


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
        self._registry_config: Optional[RegistryConfig] = None
        self._slack_channel: Optional[str] = None
        self._current_stage: Optional[str] = None
        self._overrides: Optional[dict] = None
        self._abort_event = threading.Event()
        self._stopping: bool = False
        self._last_accessed: float = time.monotonic()

    # ── control ───────────────────────────────────────────────────────────
    def start(
        self,
        workspace: WorkspacePaths,
        overrides: dict,
        registry_config: Optional[RegistryConfig] = None,
        slack_channel: Optional[str] = None,
        selected_tifs: Optional[Sequence[Path]] = None,
    ) -> bool | str:
        """Kick off a run.

        Pass ``registry_config`` (from :attr:`AppState.registry_config`) so the
        pipeline never reads ``os.environ`` for registry paths, enabling safe
        concurrent sessions on different workspaces.

        Pass ``slack_channel`` (a Slack channel ID) to post a run summary +
        overlay PNGs when the batch finishes; the bot token is read from
        ``ROIGBIV_SLACK_TOKEN`` in the environment that launched the UI.

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
                self._registry_config = registry_config
                self._slack_channel = slack_channel
                self._active = True
                self._started_at = time.time()
                self._n_fovs = (len(selected_tifs) if selected_tifs is not None
                                else len(workspace.tifs))
                # Foundation runs first and emits no stage_header; show it until
                # the first Stage marker streams in.
                self._current_stage = _FOUNDATION_STAGE
                # Capture the config that actually launched this run, for the
                # read-only "Launched config" echo on the Process page.
                self._overrides = dict(overrides)
        except Exception:
            self._gate.release()
            raise
        t = threading.Thread(
            target=self._run,
            args=(workspace, overrides, selected_tifs),
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
                current_stage=self._current_stage,
                n_awaiting=sum(1 for r in self._results
                               if r.awaiting_confirmation is not None),
                overrides=(dict(self._overrides)
                           if self._overrides is not None else None),
                stopping=self._stopping and self._active,
                stopped=(not self._active) and self._abort_event.is_set(),
            )

    def abort(self) -> bool:
        """Request a cooperative stop of the in-flight run.

        Sets the shared :class:`threading.Event` the pipeline polls at each
        stage boundary. The current stage finishes (its outputs + manifest
        entry land on disk) and then the run halts; a later ``--resume`` run
        continues from there. Returns True if a stop was requested, False if
        no run is active.
        """
        with self._lock:
            if not self._active:
                return False
            self._stopping = True
            self._logs.append(
                "Stop requested — finishing current stage, then halting…")
            self._abort_event.set()
            return True

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
        self._current_stage = None
        self._overrides = None
        self._abort_event.clear()
        self._stopping = False

    def _log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)

    def _run(self, workspace: WorkspacePaths, overrides: dict,
             selected_tifs: Optional[Sequence[Path]] = None) -> None:
        try:
            try:
                results = run_with_workspace(
                    workspace, overrides,
                    log_cb=self._append_and_tally,
                    registry_config=self._registry_config,
                    selected_tifs=selected_tifs,
                    abort_event=self._abort_event,
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
                # Awaiting-confirmation FOVs have no error but are not "done":
                # foundation ran, detection is paused pending an optics decision.
                self._n_done = sum(
                    1 for r in results
                    if r.error is None and r.awaiting_confirmation is None)
                self._n_failed = sum(1 for r in results if r.error is not None)
                self._completed_at = time.time()
                self._active = False
        finally:
            self._gate.release()
        # Post-run Slack notification (network + disk only; GPU gate already
        # released so it never blocks another session's pipeline).
        self._maybe_post_slack(results, overrides)

    def _maybe_post_slack(self, results: list[FOVRunResult], overrides: dict) -> None:
        """Render overlays + post a summary to Slack if a channel was set.

        ``run_with_workspace`` does not render overlays (the CLI does this in
        ``run.py::_run_workspace``), so we render here to have an image to
        upload, then map each :class:`FOVRunResult` to the
        :class:`~roigbiv.pipeline._email.EmailFOVResult` the notifier consumes.
        Failures are surfaced as a single log line — never raised.
        """
        import os
        from pathlib import Path

        channel = self._slack_channel
        if not channel or not results:
            return
        if not os.environ.get("ROIGBIV_SLACK_TOKEN"):
            self._log("Slack: ROIGBIV_SLACK_TOKEN not set in the UI's "
                      "environment; skipping notification.")
            return

        from roigbiv import overlay as _overlay
        from roigbiv.pipeline._email import EmailFOVResult
        from roigbiv.pipeline._slack import SlackParams, send_slack

        model_name = Path(overrides.get("cellpose_model", "")).name
        for r in results:
            if r.error is not None or r.fov is None or r.png_path is not None:
                continue
            fov_stem = r.tif.stem.replace("_mc", "")
            try:
                r.png_path = _overlay.render_overlay(
                    fov=r.fov, output_dir=r.output_dir,
                    model_name=model_name, fov_stem=fov_stem,
                )
            except BaseException as exc:  # noqa: BLE001
                self._log(f"Slack: overlay render failed for {fov_stem}: {exc}")

        summary = {
            "model_name": model_name,
            "fs": overrides.get("fs", "?"),
            "tau": overrides.get("tau", "?"),
            "flow_threshold": overrides.get("flow_threshold", "?"),
            "stage_flags": {
                2: overrides.get("enable_stage_2"),
                3: overrides.get("enable_stage_3"),
                4: overrides.get("enable_stage_4"),
            },
        }
        slack_results = [
            EmailFOVResult(
                tif=r.tif, output_dir=r.output_dir, duration_s=r.duration_s,
                error=r.error, png_path=r.png_path,
                roi_counts=r.roi_counts or {"accept": 0, "flag": 0, "reject": 0},
            )
            for r in results
        ]
        params = SlackParams(channel=channel, token_env="ROIGBIV_SLACK_TOKEN")
        ok = send_slack(slack_results, params, summary)
        self._log("Slack: summary + overlays posted." if ok else
                  "Slack FAILED — see server logs; overlays remain on disk.")

    def _append_and_tally(self, line: str) -> None:
        """Log callback that also counts completed FOVs and tracks the stage.

        Counts ``pipeline OK`` lines for FOV completion and derives the current
        stage from ``--- Stage N: ...`` markers so the Run-status banner can
        name what the pipeline is doing.
        """
        self._log(line)
        low = line.lstrip()
        stage = _derive_stage(line)
        if low.startswith("pipeline OK") or stage is not None:
            with self._lock:
                if low.startswith("pipeline OK"):
                    self._n_done += 1
                if stage is not None:
                    self._current_stage = stage

    @staticmethod
    def _summarize(r: FOVRunResult) -> dict:
        return {
            "stem": r.tif.stem.replace("_mc", ""),
            "tif": str(r.tif),
            "output_dir": str(r.output_dir),
            "duration_s": r.duration_s,
            "error": r.error,
            "awaiting_confirmation": r.awaiting_confirmation,
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
