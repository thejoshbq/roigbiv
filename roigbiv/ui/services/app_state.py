"""Per-session shared state for the Dash UI.

Held server-side so we don't ship heavy arrays through ``dcc.Store``. Each
Dash callback fetches its session's instance via :func:`get_app_state` and
reads or mutates it directly. Callbacks then return small serializable
receipts to trigger UI updates on the client.

Thread safety: Dash callbacks run on Flask's WSGI worker pool. All mutating
operations here take ``self._lock`` so concurrent callbacks don't corrupt
state. Caches are opportunistic — they can be blown away without data loss.

Multi-user: each browser session gets its own :class:`AppState` keyed on the
Flask session UUID. The module-level ``_instances`` dict is the only
shared mutable state; it is protected by ``_instances_lock``.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from roigbiv.pipeline.workspace import WorkspacePaths
from roigbiv.registry.config import RegistryConfig


@dataclass
class _FOVCache:
    """Lazily-loaded per-output-dir bundle used by the Viewer/Review pages."""
    bundle: object = None        # ui.services.loaders.FOVBundle
    corrections: list = field(default_factory=list)


@dataclass
class AppState:
    """Per-session source of truth for the Dash UI.

    Fields are intentionally simple values (paths, dicts) so they can be
    safely handed to callbacks and templates.
    """

    workspace: Optional[WorkspacePaths] = None
    registry_config: Optional[RegistryConfig] = None
    run_id: Optional[str] = None
    # Subset of workspace.tifs (as resolved path strings) the user selected to
    # run. None means "all" (no scan yet / no explicit subset). Reset to the
    # full set on every scan so a fresh workspace starts all-selected.
    selected_tifs: Optional[set[str]] = None
    # Stage-1 Cellpose diameter chosen on the motion-correction preview (drag the
    # circle / "Suggest"). A single global scalar: PipelineConfig.diameter is one
    # int applied to every FOV in the run. ``fov_stem`` is provenance for the
    # readout only. ``None`` means "uncalibrated — use the form/cfg default".
    # Survives a page reload (the form input reseeds from it) so a long
    # foundation-only run can be calibrated, then continued, across a refresh.
    calibration: Optional[dict] = None      # {"diameter_px": float, "fov_stem": str | None}
    _fov_cache: dict[str, _FOVCache] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _last_accessed: float = field(default_factory=time.monotonic)

    # ── workspace ─────────────────────────────────────────────────────────
    def set_workspace(self, workspace: WorkspacePaths) -> None:
        with self._lock:
            self.workspace = workspace
            self.selected_tifs = {str(t) for t in workspace.tifs}
            self.registry_config = RegistryConfig(
                dsn=workspace.db_dsn,
                blob_backend="local",
                blob_root=workspace.blob_root,
                endpoint=None,
                api_key=None,
                calibration_path=workspace.calibration_path,
            )
            self._fov_cache.clear()
            # A new scan starts uncalibrated — a diameter measured on the prior
            # workspace's FOVs must not leak into the next run.
            self.calibration = None

    def set_selected_tifs(self, values) -> None:
        """Store the user's TIF subset (an iterable of path strings)."""
        with self._lock:
            self.selected_tifs = {str(v) for v in (values or [])}

    # ── Stage-1 diameter calibration ──────────────────────────────────────
    def set_calibration(self, diameter_px: float, fov_stem: Optional[str] = None) -> None:
        """Persist the diameter (px) chosen on the MC preview for this session."""
        with self._lock:
            self.calibration = {
                "diameter_px": float(diameter_px),
                "fov_stem": fov_stem,
            }

    def clear_calibration(self) -> None:
        with self._lock:
            self.calibration = None

    def calibrated_diameter(self) -> Optional[int]:
        """Rounded calibrated diameter (px), or ``None`` if uncalibrated."""
        with self._lock:
            cal = self.calibration
        if not cal or cal.get("diameter_px") is None:
            return None
        return int(round(cal["diameter_px"]))

    def require_workspace(self) -> WorkspacePaths:
        if self.workspace is None:
            raise RuntimeError("no workspace selected — use the Process page first")
        return self.workspace

    # ── per-FOV bundle cache ──────────────────────────────────────────────
    def fov_cache(self, output_dir: Path) -> _FOVCache:
        key = str(Path(output_dir).resolve())
        with self._lock:
            if key not in self._fov_cache:
                self._fov_cache[key] = _FOVCache()
            return self._fov_cache[key]

    def invalidate_fov(self, output_dir: Path) -> None:
        key = str(Path(output_dir).resolve())
        with self._lock:
            self._fov_cache.pop(key, None)


_instances: dict[str, AppState] = {}
_instances_lock = threading.Lock()


def get_app_state() -> AppState:
    """Return the :class:`AppState` for the current browser session."""
    from roigbiv.ui.services.session import get_session_id
    sid = get_session_id()
    if sid not in _instances:
        with _instances_lock:
            if sid not in _instances:
                state = AppState()
                preset = _get_preset_workspace()
                if preset is not None:
                    state.set_workspace(preset)
                _instances[sid] = state
    inst = _instances[sid]
    inst._last_accessed = time.monotonic()
    return inst


def _get_preset_workspace() -> Optional[WorkspacePaths]:
    try:
        from flask import current_app
        return current_app.config.get("ROIGBIV_PRESET_WORKSPACE")
    except RuntimeError:
        return None
