"""Trace-bundle loaders for the Traces page.

Reads the per-FOV ``traces/`` bundle written by
:mod:`roigbiv.pipeline.traces_io` (primary matrix + sidecar) and surfaces
just what the UI needs: per-session trace matrices keyed by row index plus
the cross-session lookup ``global_cell_id -> [(session, row_index)]``.

No re-extraction is performed; sessions that predate the bundle just come
back as :class:`SessionTraces` with ``F_corrected = None``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from roigbiv.ui.services.loaders import CrossSessionBundle, load_cross_session_bundle


SignalKind = Literal["dff", "f"]


@dataclass
class SessionTraces:
    session_id: Optional[str]
    session_date: Optional[date]
    stem: str                            # output dir name
    output_dir: Path
    fs: float
    n_frames: int
    matrix: Optional[np.ndarray]         # (n_rois, n_frames) for requested kind
    kind: SignalKind
    rows: list[dict]                     # sidecar "rois" entries, label-id sorted
    source_label: str                    # "pipeline" or "corrections-<hash12>"
    note: Optional[str] = None           # set when the requested bundle is missing

    def row_for_local_label(self, local_label_id: int) -> Optional[int]:
        for r in self.rows:
            if int(r.get("local_label_id", -1)) == int(local_label_id):
                return int(r["row_index"])
        return None

    def row_for_global_cell(self, global_cell_id: str) -> Optional[int]:
        for r in self.rows:
            if r.get("global_cell_id") == global_cell_id:
                return int(r["row_index"])
        return None


@dataclass
class GlobalCellRow:
    global_cell_id: str
    n_sessions: int
    label_ids_by_session: dict[str, int] = field(default_factory=dict)


Y_LABELS: dict[SignalKind, str] = {
    "dff": "dF/F",
    "f": "F (neuropil-corrected)",
}


# ── primary bundle selection ──────────────────────────────────────────────


def _select_bundle_dir(output_dir: Path) -> tuple[Optional[Path], str]:
    """Return (bundle_dir, source_label) for the freshest available bundle.

    Prefers ``traces/corrections-*/`` when its sidecar is newer than the
    primary ``traces/``. Falls back to the primary. Returns ``(None, "")``
    when neither exists.
    """
    traces_root = output_dir / "traces"
    primary_meta = traces_root / "traces_meta.json"

    newest_corr: Optional[Path] = None
    newest_corr_mtime = -1.0
    if traces_root.is_dir():
        for sub in traces_root.iterdir():
            if sub.is_dir() and sub.name.startswith("corrections-"):
                meta = sub / "traces_meta.json"
                if meta.exists() and meta.stat().st_mtime > newest_corr_mtime:
                    newest_corr = sub
                    newest_corr_mtime = meta.stat().st_mtime

    primary_mtime = primary_meta.stat().st_mtime if primary_meta.exists() else -1.0
    if newest_corr is not None and newest_corr_mtime >= primary_mtime:
        return newest_corr, newest_corr.name       # e.g. "corrections-abc123def012"
    if primary_meta.exists():
        return traces_root, "pipeline"
    return None, ""


# ── sidecar + matrix loading ──────────────────────────────────────────────


@lru_cache(maxsize=32)
def _load_cached(
    bundle_key: str,
    kind: SignalKind,
    mtime_key: float,   # part of the cache key so file edits invalidate
) -> tuple[Optional[np.ndarray], dict]:
    """Load ``(matrix, sidecar)`` for a given bundle + signal kind.

    ``bundle_key`` is the stringified absolute path of the bundle dir.
    ``mtime_key`` is the modification time of ``traces_meta.json`` — two
    separate runs against the same bundle dir will cache-hit only when the
    sidecar hasn't been rewritten.
    """
    bundle_dir = Path(bundle_key)
    sidecar = json.loads((bundle_dir / "traces_meta.json").read_text())

    if kind == "f":
        primary = bundle_dir / "traces.npy"
        matrix = np.load(primary) if primary.exists() else None
        return matrix, sidecar

    # dF/F row count must match this bundle's ROI set, so never fall back
    # across bundles. Corrections bundles write their own dFF.npy alongside
    # traces.npy; the primary bundle's dF/F lives at output_dir/dFF.npy
    # (one level up from `traces/`).
    local_dff = bundle_dir / "dFF.npy"
    if local_dff.exists():
        return np.load(local_dff), sidecar
    if bundle_dir.name == "traces":
        parent_dff = bundle_dir.parent / "dFF.npy"
        if parent_dff.exists():
            return np.load(parent_dff), sidecar
    return None, sidecar


def load_session_traces(
    output_dir: Path,
    *,
    kind: SignalKind,
    session_date: Optional[date] = None,
    session_id: Optional[str] = None,
) -> SessionTraces:
    """Load a :class:`SessionTraces` for one FOV output directory.

    Callers who already have session metadata (date, session_id) from the
    registry should pass them through; otherwise these fall back to the
    sidecar values or ``None``.
    """
    output_dir = Path(output_dir)
    bundle_dir, source_label = _select_bundle_dir(output_dir)
    if bundle_dir is None:
        return SessionTraces(
            session_id=session_id, session_date=session_date,
            stem=output_dir.name, output_dir=output_dir,
            fs=0.0, n_frames=0, matrix=None, kind=kind,
            rows=[], source_label="",
            note="No traces/ bundle found for this session.",
        )

    meta_path = bundle_dir / "traces_meta.json"
    matrix, sidecar = _load_cached(
        str(bundle_dir.resolve()), kind, meta_path.stat().st_mtime,
    )
    note: Optional[str] = None
    if matrix is None:
        note = f"{Y_LABELS[kind]} matrix not written for this session."
    return SessionTraces(
        session_id=session_id or sidecar.get("session_id"),
        session_date=session_date,
        stem=output_dir.name,
        output_dir=output_dir,
        fs=float(sidecar.get("fs") or 0.0),
        n_frames=int(sidecar.get("n_frames") or 0),
        matrix=matrix,
        kind=kind,
        rows=list(sidecar.get("rois") or []),
        source_label=source_label,
        note=note,
    )


# ── cross-session helpers ─────────────────────────────────────────────────


def list_local_rois_for_session(output_dir: Path) -> list[dict]:
    """List ROI identifiers for a single session.

    Returns the sidecar's ``rois`` array sorted by ``local_label_id``.
    Each entry is the raw dict from ``traces_meta.json`` — at minimum
    ``row_index``, ``local_label_id``; optionally ``global_cell_id``,
    ``activity_type``.
    """
    bundle_dir, _ = _select_bundle_dir(Path(output_dir))
    if bundle_dir is None:
        return []
    sidecar = json.loads((bundle_dir / "traces_meta.json").read_text())
    rows = list(sidecar.get("rois") or [])
    rows.sort(key=lambda r: int(r.get("local_label_id", 0)))
    return rows


def list_global_cells_for_fov(fov_id: str) -> list[GlobalCellRow]:
    """Enumerate cross-session cells for a FOV (n_sessions >= 2 only).

    Uses the authoritative mapping in the registry DB: a cell is "cross-
    session" if it has observations across at least two distinct sessions.
    """
    from roigbiv.registry import build_store

    store = build_store()
    store.ensure_schema()
    cells = store.list_cells(fov_id)
    out: list[GlobalCellRow] = []
    for cell in cells:
        obs = store.list_observations_for_cell(cell.global_cell_id)
        if len(obs) < 2:
            continue
        label_by_session = {
            o.session_id: int(o.local_label_id) for o in obs
        }
        out.append(GlobalCellRow(
            global_cell_id=cell.global_cell_id,
            n_sessions=len(label_by_session),
            label_ids_by_session=label_by_session,
        ))
    out.sort(key=lambda r: (-r.n_sessions, r.global_cell_id))
    return out


def collect_cross_session_traces(
    fov_id: str,
    global_cell_id: str,
    kind: SignalKind,
) -> list[tuple[SessionTraces, int]]:
    """Return [(SessionTraces, row_index)] for every session containing the
    given persistent cell.

    The row_index is resolved from the registry DB (authoritative) and then
    validated against the session's sidecar — if the sidecar reports a
    different row for the same ``local_label_id``, the sidecar wins.
    """
    from roigbiv.registry import build_store

    store = build_store()
    store.ensure_schema()
    observations = store.list_observations_for_cell(global_cell_id)
    if not observations:
        return []

    bundle: CrossSessionBundle = load_cross_session_bundle(fov_id)
    session_meta_by_id = {s.session_id: s for s in bundle.sessions}

    out: list[tuple[SessionTraces, int]] = []
    for obs in observations:
        ref = session_meta_by_id.get(obs.session_id)
        if ref is None:
            continue
        sess = load_session_traces(
            Path(ref.output_dir),
            kind=kind,
            session_date=ref.session_date,
            session_id=ref.session_id,
        )
        row = sess.row_for_local_label(int(obs.local_label_id))
        if row is None:
            continue
        out.append((sess, row))

    out.sort(key=lambda pair: (pair[0].session_date or date.min,
                               pair[0].session_id or ""))
    return out


def collect_sessions_for_fov(
    fov_id: str,
    kind: SignalKind,
) -> list[SessionTraces]:
    """Return every session on this FOV as :class:`SessionTraces`, ordered
    by session_date. Sessions with no ``traces/`` bundle come back with
    ``matrix=None`` and a ``note`` set — the UI decides whether to skip or
    display a placeholder."""
    bundle: CrossSessionBundle = load_cross_session_bundle(fov_id)
    out: list[SessionTraces] = []
    for ref in bundle.sessions:
        out.append(load_session_traces(
            Path(ref.output_dir),
            kind=kind,
            session_date=ref.session_date,
            session_id=ref.session_id,
        ))
    out.sort(key=lambda s: (s.session_date or date.min, s.session_id or ""))
    return out
