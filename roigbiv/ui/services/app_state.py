"""Process-local shared state for the Dash UI.

Held server-side so we don't ship heavy arrays through ``dcc.Store``. Each
Dash callback fetches the singleton via :func:`get_app_state` and reads or
mutates it directly. Callbacks then return small serializable receipts to
trigger UI updates on the client.

Thread safety: Dash callbacks run on Flask's WSGI worker pool. All mutating
operations here take ``self._lock`` so concurrent callbacks don't corrupt
state. Caches are opportunistic — they can be blown away without data loss.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from roigbiv.pipeline.workspace import WorkspacePaths


@dataclass
class _FOVCache:
    """Lazily-loaded per-output-dir bundle used by the Viewer/Review pages."""
    bundle: object = None        # ui.services.loaders.FOVBundle
    corrections: list = field(default_factory=list)


@dataclass
class AppState:
    """Single source of truth for the Dash session.

    Fields are intentionally simple values (paths, dicts) so they can be
    safely handed to callbacks and templates.
    """

    workspace: Optional[WorkspacePaths] = None
    run_id: Optional[str] = None
    _fov_cache: dict[str, _FOVCache] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ── workspace ─────────────────────────────────────────────────────────
    def set_workspace(self, workspace: WorkspacePaths) -> None:
        with self._lock:
            self.workspace = workspace
            self._fov_cache.clear()

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


_instance: Optional[AppState] = None
_instance_lock = threading.Lock()


def get_app_state() -> AppState:
    """Return the process-wide :class:`AppState` singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AppState()
    return _instance
