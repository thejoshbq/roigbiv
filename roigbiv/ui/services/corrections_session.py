"""In-memory wrapper around ``roigbiv.pipeline.corrections``.

The Review page keeps an :class:`CorrectionsSession` per FOV output dir so
the user can:

* queue up adds / deletes / merges / splits / edits / relabels
* undo / redo without touching disk
* "Commit" to persist the log and materialize corrected artifacts
* "Re-register" to run ``register_or_match`` against the corrected artifacts

Staying server-side keeps the Dash wire format small (we only ship op ids
and short summaries to the client).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from roigbiv.pipeline.corrections import (
    CorrectionOp,
    apply_corrections,
    append_correction,
    load_corrections,
    materialize,
    write_corrections,
)
from roigbiv.pipeline.loaders import load_fov_from_output_dir
from roigbiv.pipeline.types import ROI


@dataclass
class CommitResult:
    masks_path: Path
    metadata_path: Path
    n_ops: int
    n_rois: int


@dataclass
class CorrectionsSession:
    """Session-scoped corrections staging for one FOV output dir.

    ``pending`` is the list of *unsaved* ops staged since the last commit or
    load. ``undo_stack`` gets a snapshot before each append so the user can
    undo one step at a time. ``redo_stack`` holds undone ops for re-apply.

    All mutation is guarded by ``self._lock``; Dash callbacks share one
    :class:`CorrectionsSession` across worker threads.
    """

    output_dir: Path
    shape_hw: tuple[int, int]
    base_rois: list[ROI] = field(default_factory=list)
    persisted: list[CorrectionOp] = field(default_factory=list)
    pending: list[CorrectionOp] = field(default_factory=list)
    undo_stack: list[list[CorrectionOp]] = field(default_factory=list)
    redo_stack: list[list[CorrectionOp]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ── lifecycle ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls, output_dir: Path) -> "CorrectionsSession":
        fov, _rq = load_fov_from_output_dir(output_dir)
        if fov.mean_M is not None:
            shape_hw = (int(fov.mean_M.shape[0]), int(fov.mean_M.shape[1]))
        elif isinstance(fov.shape, tuple) and len(fov.shape) >= 3:
            shape_hw = (int(fov.shape[1]), int(fov.shape[2]))
        else:
            raise ValueError("cannot determine FOV shape for corrections replay")

        return cls(
            output_dir=output_dir,
            shape_hw=shape_hw,
            base_rois=list(fov.rois),
            persisted=list(load_corrections(output_dir)),
            pending=[],
        )

    # ── op queueing ───────────────────────────────────────────────────────
    def add(self, op: CorrectionOp) -> None:
        with self._lock:
            self.undo_stack.append(list(self.pending))
            self.redo_stack.clear()
            self.pending.append(op)

    def undo(self) -> bool:
        with self._lock:
            if not self.undo_stack:
                return False
            self.redo_stack.append(list(self.pending))
            self.pending = self.undo_stack.pop()
            return True

    def redo(self) -> bool:
        with self._lock:
            if not self.redo_stack:
                return False
            self.undo_stack.append(list(self.pending))
            self.pending = self.redo_stack.pop()
            return True

    def discard_pending(self) -> None:
        with self._lock:
            if self.pending:
                self.undo_stack.append(list(self.pending))
                self.redo_stack.clear()
                self.pending = []

    # ── state ─────────────────────────────────────────────────────────────
    def all_ops(self) -> list[CorrectionOp]:
        with self._lock:
            return list(self.persisted) + list(self.pending)

    def corrected_rois(self) -> list[ROI]:
        return apply_corrections(self.base_rois, self.all_ops(), self.shape_hw)

    def summary(self) -> dict:
        with self._lock:
            return {
                "output_dir": str(self.output_dir),
                "n_persisted": len(self.persisted),
                "n_pending": len(self.pending),
                "can_undo": bool(self.undo_stack),
                "can_redo": bool(self.redo_stack),
            }

    # ── commit / rewrite ──────────────────────────────────────────────────
    def commit(self) -> CommitResult:
        """Persist ``pending`` ops to disk and materialize corrected outputs."""
        with self._lock:
            if self.pending:
                for op in self.pending:
                    append_correction(self.output_dir, op)
                self.persisted.extend(self.pending)
                self.pending = []
                self.undo_stack.clear()
                self.redo_stack.clear()
            rois_corrected = apply_corrections(
                self.base_rois, self.persisted, self.shape_hw,
            )
            masks_path, meta_path = materialize(
                rois_corrected, self.output_dir, self.shape_hw,
            )
            return CommitResult(
                masks_path=masks_path,
                metadata_path=meta_path,
                n_ops=len(self.persisted),
                n_rois=len(rois_corrected),
            )

    def rewrite_persisted(self, ops: list[CorrectionOp]) -> None:
        """Overwrite the on-disk log — used when the user revokes persisted ops."""
        with self._lock:
            write_corrections(self.output_dir, ops)
            self.persisted = list(ops)
            self.pending = []
            self.undo_stack.clear()
            self.redo_stack.clear()


# ── module-level singleton registry ────────────────────────────────────────

_sessions: dict[str, CorrectionsSession] = {}
_sessions_lock = threading.Lock()


def get_corrections_session(output_dir: Path) -> CorrectionsSession:
    """Return (or lazily create) the session for ``output_dir``."""
    key = str(Path(output_dir).resolve())
    with _sessions_lock:
        sess = _sessions.get(key)
        if sess is None:
            sess = CorrectionsSession.load(Path(output_dir))
            _sessions[key] = sess
        return sess


def reset_corrections_session(output_dir: Path) -> None:
    """Drop the cached session so the next lookup reloads from disk."""
    key = str(Path(output_dir).resolve())
    with _sessions_lock:
        _sessions.pop(key, None)


def reregister_corrected_session(
    output_dir: Path,
    session: CorrectionsSession,
) -> Optional[dict]:
    """Run ``register_or_match`` against the corrected artifacts.

    Uses the materialized ``corrected_masks.tif`` + mean_M. Returns the same
    report dict shape as the pipeline-time registration, written alongside
    ``registry_match.json`` in the output dir (overwriting the pipeline-time
    file is intentional — it's now stale).
    """
    import numpy as np
    import tifffile

    from roigbiv.registry import (
        RegistryConfig,
        build_adapter_config,
        build_blob_store,
        build_store,
        load_calibration,
        register_or_match,
    )
    from roigbiv.registry.roicat_adapter import SessionInput

    # Ensure corrected artifacts are fresh on disk.
    session.commit()

    mean_m_path = output_dir / "summary" / "mean_M.tif"
    corrected_masks_path = output_dir / "corrections" / "corrected_masks.tif"
    if not mean_m_path.exists() or not corrected_masks_path.exists():
        return None

    mean_m = np.asarray(tifffile.imread(str(mean_m_path)), dtype=np.float32)
    merged_masks = np.asarray(
        tifffile.imread(str(corrected_masks_path)), dtype=np.uint16,
    )
    if not (merged_masks > 0).any():
        return None

    cfg = RegistryConfig.from_env()
    store = build_store(cfg)
    blob_store = build_blob_store(cfg)
    adapter_cfg = build_adapter_config(cfg)
    calibration = load_calibration(cfg)

    query = SessionInput(
        session_key=output_dir.name,
        mean_m=mean_m,
        merged_masks=merged_masks,
    )
    return register_or_match(
        fov_stem=output_dir.name,
        query=query,
        output_dir=output_dir,
        store=store,
        blob_store=blob_store,
        adapter_config=adapter_cfg,
        calibration=calibration,
        accept_threshold=cfg.fov_accept_threshold,
        review_threshold=cfg.fov_review_threshold,
    )
