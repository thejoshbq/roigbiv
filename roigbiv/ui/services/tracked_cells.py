"""One FOV's cells, assembled across every session that saw them.

The Track page can already say *how many* cells matched. This module supplies
what it takes to say *which* — the per-session geometry, joined to the
cross-session ``global_cell_id``, so a viewer can light the same neuron up in
several sessions at once.

Why not :func:`roigbiv.ui.services.loaders.load_cross_session_bundle`
--------------------------------------------------------------------
That path goes through ``load_fov_from_output_dir``, which short-circuits on
``foundation_only.json`` and returns ``rois=[]``. Every centroid-tracked
workspace writes that sentinel, so the existing bundle loader returns zero ROIs
for exactly the FOVs this module exists to show.

Here the geometry comes from ``merged_masks.tif`` instead — via
:func:`roigbiv.registry.roicat_adapter.load_session_input`, the same reader the
matcher itself uses. Its label values *are* ``CellObservation.local_label_id``,
so what gets drawn is precisely what was matched, with no second
interpretation of the data sitting in between.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.registry.anomalies import CellTimeline
from roigbiv.ui.services.loaders import ROIRender

# A cell that is *absent* from a session still gets an outline, drawn at the
# position it last held. It owns no label in this session's mask, so it is
# given a negative id — real labels are uint16 and start at 1, so the two can
# never collide when a click is resolved back to a cell.
def _ghost_label_id(cell_index: int) -> int:
    return -int(cell_index)


@dataclass
class TrackedCell:
    """One global cell's presence across a FOV's ordered sessions."""

    global_cell_id: str
    index: int                                  # display "#N", 1-based
    present: list[bool]
    local_label_ids: list[Optional[int]]
    centroids: list[Optional[tuple[float, float]]]
    anomalies: list[str]

    @property
    def n_present(self) -> int:
        return sum(self.present)

    def label_in(self, session_index: int) -> Optional[int]:
        return self.local_label_ids[session_index]


@dataclass
class TrackedSession:
    """One session of a tracked FOV, decoded for the viewer."""

    session_id: str
    stem: str
    session_date: Optional[date]
    sequence_index: Optional[int]
    output_dir: Path
    mean_M: Optional[np.ndarray]
    rois: list[ROIRender]
    n_matched: int
    n_new: int
    n_missing: int
    # The on-disk registry_match.json names a different session than the
    # registry does — see :func:`_is_stale`.
    stale: bool = False

    @property
    def label(self) -> str:
        return self.session_date.isoformat() if self.session_date else self.stem

    @property
    def short_label(self) -> str:
        """The part of the stem that tells this session from its siblings.

        Not the date: this lab routinely records several sessions on one day —
        the reference prism FOV has ``pre-005`` / ``beh-006`` / ``post-007`` all
        on 2026-05-21 — so a date-labelled panel is indistinguishable from its
        neighbours, which is the whole failure the Track page's manual ordering
        exists to work around. The trailing token of the stem is the lab's own
        name for the session and is what discriminates.
        """
        return self.stem.rsplit("_", 1)[-1] or self.stem


@dataclass
class TrackedFOV:
    fov_id: str
    animal_id: Optional[str]
    region: Optional[str]
    sessions: list[TrackedSession] = field(default_factory=list)
    cells: list[TrackedCell] = field(default_factory=list)
    ordering_is_confirmed: bool = True

    @property
    def n_complete(self) -> int:
        """Cells seen in every session — the ones needing no explanation."""
        n = len(self.sessions)
        return sum(1 for c in self.cells if c.n_present == n)

    def cell_by_gcid(self, gcid: Optional[str]) -> Optional[TrackedCell]:
        if not gcid:
            return None
        return next((c for c in self.cells if c.global_cell_id == gcid), None)

    def gcid_for_label(self, session_index: int, label_id: int) -> Optional[str]:
        """Which cell owns *label_id* in session *session_index*."""
        for cell in self.cells:
            if cell.local_label_ids[session_index] == label_id:
                return cell.global_cell_id
            if _ghost_label_id(cell.index) == label_id:
                return cell.global_cell_id
        return None


# ── loading ────────────────────────────────────────────────────────────────


def load_tracked_fov(fov_id: str, cfg=None) -> TrackedFOV:
    """Assemble every session of *fov_id* with its cross-session cell identities.

    Sessions come back in timeline order (``store.list_sessions`` honours the
    human-set ``sequence_index``), which is the order that makes "arrived late"
    and "dropped out" mean anything. Sessions whose output directory has gone
    missing are skipped rather than raising, matching
    :func:`~roigbiv.ui.services.loaders.load_cross_session_bundle`.
    """
    from roigbiv.registry import build_store
    from roigbiv.registry.anomalies import cell_timeline

    store = build_store(cfg=cfg)
    store.ensure_schema()

    fov = store.get_fov(fov_id)
    report = cell_timeline(store, fov_id)
    records = {r.session_id: r for r in store.list_sessions(fov_id)}

    slots = [s for s in report.sessions if Path(s.output_dir).exists()]
    kept_positions = [s.sequence_index for s in slots]

    cells = _number_cells(report.cells, kept_positions)
    sessions = _load_sessions(slots, records, cells, kept_positions)

    return TrackedFOV(
        fov_id=fov_id,
        animal_id=getattr(fov, "animal_id", None),
        region=getattr(fov, "region", None),
        sessions=sessions,
        cells=cells,
        ordering_is_confirmed=report.ordering_is_confirmed,
    )


def _number_cells(timelines, kept_positions: list[int]) -> list[TrackedCell]:
    """Assign the display ``#N``, restricted to the sessions we can render.

    Numbering follows first appearance, then label order within that session.
    Sorting on the ``global_cell_id`` instead — as
    :func:`~roigbiv.registry.anomalies.cell_timeline` does — would scatter the
    numbers randomly across the image, since the id is a UUID.

    A cell seen only in sessions we cannot render is dropped, so the numbers
    stay contiguous over what is actually on screen.
    """
    trimmed = []
    for t in timelines:
        present = [t.present[i] for i in kept_positions]
        if not any(present):
            continue
        labels = [t.local_label_ids[i] for i in kept_positions]
        first = next(i for i, p in enumerate(present) if p)
        trimmed.append((first, labels[first], t, present, labels))

    trimmed.sort(key=lambda row: (row[0], row[1], row[2].global_cell_id))
    return [
        TrackedCell(
            global_cell_id=t.global_cell_id,
            index=n,
            present=present,
            local_label_ids=labels,
            centroids=[None] * len(present),
            # Re-derived over the sessions actually shown: an anomaly computed
            # against a session the viewer cannot render would contradict the
            # timeline the user is looking at.
            anomalies=CellTimeline(
                global_cell_id=t.global_cell_id,
                present=present, local_label_ids=labels,
            ).anomalies,
        )
        for n, (_, _, t, present, labels) in enumerate(trimmed, start=1)
    ]


def _load_sessions(slots, records, cells, kept_positions) -> list[TrackedSession]:
    """Decode each session's masks and attach cross-session status per ROI."""
    from roigbiv.registry.roicat_adapter import load_session_input

    cell_by_label: list[dict[int, TrackedCell]] = [
        {c.local_label_ids[i]: c for c in cells if c.local_label_ids[i] is not None}
        for i in range(len(kept_positions))
    ]
    # A cell absent from this session is drawn where it last was; without this
    # carry-forward a dropout is simply invisible, which is the opposite of
    # what the anomaly is for.
    last_known: dict[str, tuple[tuple[float, float], list]] = {}

    sessions: list[TrackedSession] = []
    for i, slot in enumerate(slots):
        out_dir = Path(slot.output_dir)
        try:
            session_input = load_session_input(out_dir)
        except (FileNotFoundError, ValueError):
            continue

        shapes = _label_shapes(session_input.merged_masks)
        rois: list[ROIRender] = []
        for label_id, (centroid, contours, area) in sorted(shapes.items()):
            cell = cell_by_label[i].get(label_id)
            status = "matched"
            if cell is not None:
                cell.centroids[i] = centroid
                last_known[cell.global_cell_id] = (centroid, contours)
                if all(not p for p in cell.present[:i]):
                    status = "new"
            else:
                # A label with no observation row: the session was registered
                # before this ROI existed, or the DB is out of step with disk.
                status = None
            rois.append(ROIRender(
                label_id=label_id, source_stage=1, gate_outcome="accept",
                activity_type=None, area=area, centroid_yx=centroid,
                contours=contours,
                global_cell_id=cell.global_cell_id if cell else None,
                match_status=status,
            ))

        rois.extend(_ghosts(cells, i, last_known))
        record = records.get(slot.session_id)
        sessions.append(TrackedSession(
            session_id=slot.session_id,
            stem=out_dir.name,
            session_date=slot.session_date,
            sequence_index=slot.stored_sequence_index,
            output_dir=out_dir,
            mean_M=session_input.mean_m,
            rois=rois,
            n_matched=int(getattr(record, "n_matched", 0) or 0),
            n_new=int(getattr(record, "n_new", 0) or 0),
            n_missing=int(getattr(record, "n_missing", 0) or 0),
            stale=_is_stale(out_dir, slot.session_id),
        ))
    return sessions


def _ghosts(cells, i: int, last_known: dict) -> list[ROIRender]:
    """Outlines for cells that were seen before this session but not in it."""
    out = []
    for cell in cells:
        if cell.present[i] or not any(cell.present[:i]):
            continue
        seen = last_known.get(cell.global_cell_id)
        if seen is None:
            continue
        centroid, contours = seen
        out.append(ROIRender(
            label_id=_ghost_label_id(cell.index), source_stage=1,
            gate_outcome="accept", activity_type=None, area=0,
            centroid_yx=centroid, contours=contours,
            global_cell_id=cell.global_cell_id, match_status="lost",
        ))
    return out


def _label_shapes(merged_masks: np.ndarray) -> dict[int, tuple]:
    """``{label_id: (centroid_yx, contours, area)}`` for one label image.

    Contours are traced inside each label's padded bounding box rather than the
    full frame: a 1024x1024 FOV with 17 ROIs would otherwise trace 17 full-size
    arrays for a few hundred pixels of actual footprint.
    """
    from scipy.ndimage import find_objects
    from skimage.measure import find_contours

    masks = np.asarray(merged_masks)
    out: dict[int, tuple] = {}
    for label_id, window in enumerate(find_objects(masks), start=1):
        if window is None:
            continue
        y0, x0 = window[0].start, window[1].start
        # Pad by one pixel so a footprint touching its own bbox edge still
        # closes into a ring instead of being clipped open.
        sub = np.pad((masks[window] == label_id).astype(float), 1)
        ys, xs = np.nonzero(sub)
        centroid = (float(ys.mean()) + y0 - 1.0, float(xs.mean()) + x0 - 1.0)
        contours = [
            ((ring[:, 0] + y0 - 1.0).tolist(), (ring[:, 1] + x0 - 1.0).tolist())
            for ring in find_contours(sub, 0.5)
        ]
        out[int(label_id)] = (centroid, contours, int(ys.size))
    return out


def _is_stale(output_dir: Path, session_id: str) -> bool:
    """Whether this directory's ``registry_match.json`` names another session.

    The pipeline writes that file and the registry rows in the same pass, so a
    disagreement means one of them was later replaced — the registry can then
    report "nothing matched" for a FOV whose own match record says otherwise.
    Better to say so than to draw the emptier of the two answers in silence.
    """
    path = Path(output_dir) / "registry_match.json"
    if not path.exists():
        return False
    try:
        recorded = json.loads(path.read_text()).get("session_id")
    except (json.JSONDecodeError, OSError):
        return False
    return bool(recorded) and recorded != session_id


# ── cache ──────────────────────────────────────────────────────────────────

_cache: dict[tuple, TrackedFOV] = {}
_cache_lock = threading.Lock()


def load_tracked_fov_cached(fov_id: str, cfg=None) -> TrackedFOV:
    """:func:`load_tracked_fov`, memoised until the registry or masks change.

    Every click on the /cells page needs the whole FOV again; re-tracing
    contours on 1024x1024 label images each time would make selection feel
    broken. Invalidated by the registry DB's mtime plus each session's mask
    mtime, so a re-run or a re-ingest is picked up without a restart.

    Only ever keeps *one* entry per ``(fov_id, dsn)``. Before instant-apply
    edits, this cache only churned on a re-run, so an unbounded dict was
    harmless. Under instant apply, every click writes a new mtime fingerprint
    and adds another full ``TrackedFOV`` — traced contours for every
    session — without ever dropping the one from before the click; a 30-edit
    session would otherwise leak 30 of them.
    """
    dsn = getattr(cfg, "dsn", None)
    key = (fov_id, dsn, _fingerprint(fov_id, cfg))
    cached = _cache.get(key)
    if cached is not None:
        return cached
    loaded = load_tracked_fov(fov_id, cfg=cfg)
    with _cache_lock:
        _evict_other_fingerprints(fov_id, dsn, keep=key)
        return _cache.setdefault(key, loaded)


def invalidate_tracked_fov(fov_id: str, cfg=None) -> None:
    """Drop every cached entry for ``(fov_id, this cfg's dsn)``.

    ``apply_tracking_edits`` writes new masks and observations directly, and
    an edit's write and the next read can land inside the same mtime tick —
    the fingerprint would then look unchanged and the cache would hand back a
    pre-edit snapshot. Calling this right after an edit closes that window
    instead of relying on the mtime granularity to save it.
    """
    dsn = getattr(cfg, "dsn", None)
    with _cache_lock:
        _evict_other_fingerprints(fov_id, dsn, keep=None)


def _evict_other_fingerprints(fov_id: str, dsn, *, keep: Optional[tuple]) -> None:
    """Remove every cached key for ``(fov_id, dsn)`` other than ``keep``.

    Caller holds ``_cache_lock``.
    """
    for stale_key in [k for k in _cache
                       if k[0] == fov_id and k[1] == dsn and k != keep]:
        del _cache[stale_key]


def _fingerprint(fov_id: str, cfg) -> tuple:
    """Cheap staleness token: the DB's mtime plus every session mask's."""
    from roigbiv.registry import build_store

    parts: list = []
    dsn = getattr(cfg, "dsn", "") or ""
    if dsn.startswith("sqlite:///"):
        parts.append(_mtime(Path(dsn[len("sqlite:///"):])))
    try:
        store = build_store(cfg=cfg)
        store.ensure_schema()
        for row in store.list_sessions(fov_id):
            parts.append(_mtime(Path(row.output_dir) / "merged_masks.tif"))
    except Exception:  # noqa: BLE001 — an unreadable store simply won't cache
        return (None,)
    return tuple(parts)


def _mtime(path: Path) -> Optional[int]:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None
