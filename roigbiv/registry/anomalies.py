"""Which cells went missing, arrived late, or blinked — derived, not stored.

A cell's presence across a FOV's timeline is already fully determined by its
``CellObservation`` rows; nothing new needs persisting. This module reads that
relation in timeline order (``store.list_sessions`` honours the human-set
``sequence_index``) and names the three patterns worth a researcher's attention:

``late_arrival``
    First seen after the timeline started. Either a genuinely new cell or one
    the earlier sessions' detection missed.

``dropout``
    Last seen before the timeline ended. Either the cell stopped expressing /
    left the plane, or later detection missed it.

``intermittent``
    Absent from a session that falls *between* two sessions where it was seen.
    The most suspicious of the three — a cell cannot un-exist and then exist
    again, so this is nearly always a missed detection rather than biology.

A cell can be both ``late_arrival`` and ``dropout``; the flags are independent.
None of them is an error — they are the questions the registry can ask on a
researcher's behalf, not answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class SessionSlot:
    """One position in a FOV's timeline."""

    session_id: str
    sequence_index: int          # position in the timeline, always 0..N-1 here
    session_date: Optional[date]
    output_dir: str
    stored_sequence_index: Optional[int] = None   # None when never human-ordered


@dataclass
class CellTimeline:
    """One cell's presence across a FOV's ordered sessions."""

    global_cell_id: str
    present: list[bool] = field(default_factory=list)
    local_label_ids: list[Optional[int]] = field(default_factory=list)

    @property
    def first_seen(self) -> Optional[int]:
        return next((i for i, p in enumerate(self.present) if p), None)

    @property
    def last_seen(self) -> Optional[int]:
        seen = [i for i, p in enumerate(self.present) if p]
        return seen[-1] if seen else None

    @property
    def n_present(self) -> int:
        return sum(self.present)

    @property
    def is_late_arrival(self) -> bool:
        return self.first_seen is not None and self.first_seen > 0

    @property
    def is_dropout(self) -> bool:
        last = self.last_seen
        return last is not None and last < len(self.present) - 1

    @property
    def gap_indices(self) -> list[int]:
        """Timeline positions where the cell is absent between two sightings."""
        first, last = self.first_seen, self.last_seen
        if first is None or last is None:
            return []
        return [i for i in range(first + 1, last) if not self.present[i]]

    @property
    def is_intermittent(self) -> bool:
        return bool(self.gap_indices)

    @property
    def anomalies(self) -> list[str]:
        out = []
        if self.is_late_arrival:
            out.append("late_arrival")
        if self.is_dropout:
            out.append("dropout")
        if self.is_intermittent:
            out.append("intermittent")
        return out


@dataclass
class FOVAnomalyReport:
    fov_id: str
    sessions: list[SessionSlot]
    cells: list[CellTimeline]

    @property
    def complete(self) -> list[CellTimeline]:
        """Cells seen in every session — the ones needing no explanation."""
        n = len(self.sessions)
        return [c for c in self.cells if c.n_present == n]

    def with_anomaly(self, kind: str) -> list[CellTimeline]:
        return [c for c in self.cells if kind in c.anomalies]

    @property
    def counts(self) -> dict:
        return {
            "n_sessions": len(self.sessions),
            "n_cells": len(self.cells),
            "n_complete": len(self.complete),
            "late_arrival": len(self.with_anomaly("late_arrival")),
            "dropout": len(self.with_anomaly("dropout")),
            "intermittent": len(self.with_anomaly("intermittent")),
        }

    @property
    def ordering_is_confirmed(self) -> bool:
        """Whether a human ordered this timeline.

        Worth surfacing next to the counts: on an unordered FOV the sequence is
        whatever the filename dates implied, so "late" and "dropout" inherit
        that guess.
        """
        return bool(self.sessions) and all(
            s.stored_sequence_index is not None for s in self.sessions
        )


def cell_timeline(store, fov_id: str) -> FOVAnomalyReport:
    """Build one FOV's cell-presence report from its observation rows."""
    sessions = [
        SessionSlot(
            session_id=s.session_id,
            sequence_index=i,
            session_date=s.session_date,
            output_dir=s.output_dir,
            stored_sequence_index=s.sequence_index,
        )
        for i, s in enumerate(store.list_sessions(fov_id))
    ]
    if not sessions:
        return FOVAnomalyReport(fov_id=fov_id, sessions=[], cells=[])

    slot_of = {s.session_id: s.sequence_index for s in sessions}
    n = len(sessions)

    timelines: dict[str, CellTimeline] = {}
    for session in sessions:
        for obs in store.list_observations_for_session(session.session_id):
            timeline = timelines.get(obs.global_cell_id)
            if timeline is None:
                timeline = CellTimeline(
                    global_cell_id=obs.global_cell_id,
                    present=[False] * n,
                    local_label_ids=[None] * n,
                )
                timelines[obs.global_cell_id] = timeline
            idx = slot_of[obs.session_id]
            timeline.present[idx] = True
            timeline.local_label_ids[idx] = obs.local_label_id

    cells = sorted(
        timelines.values(),
        key=lambda c: (c.first_seen if c.first_seen is not None else n,
                       c.global_cell_id),
    )
    return FOVAnomalyReport(fov_id=fov_id, sessions=sessions, cells=cells)
