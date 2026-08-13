"""The human-confirmed chronological order of a workspace's sessions.

Filename dates cannot order this lab's timelines on their own. Six-digit date
groups are ambiguous between the two conventions in use
(:mod:`roigbiv.registry.filename`), and sessions routinely share a date — the
reference prism workspace records ``pre-005`` / ``beh-006`` / ``post-007`` on
one day, a sequence no date can express.

So the parsed date only ever *proposes* an order. A human confirms it on the
Track page, and the result is persisted here as ``session_order.json`` at the
workspace root, beside ``registry.db``. Registration then walks that order,
which matters beyond display: the earliest-registered cell in a ROICaT cluster
wins the ``global_cell_id`` (``registry/orchestrator.py``), so registration
order *is* cell-identity seniority.

Entries a human has touched are ``locked``; re-scanning a workspace appends new
FOVs after them rather than reshuffling a confirmed timeline.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

_SCHEMA = 1
ORDER_FILENAME = "session_order.json"


@dataclass
class SessionOrderEntry:
    """One FOV's place in the workspace timeline."""

    stem: str
    index: int
    session_date: Optional[str] = None   # ISO date, or None when unparseable
    date_source: str = "unparsed"        # see registry.filename.DATE_SOURCES
    locked: bool = False                 # a human confirmed this position

    @property
    def needs_review(self) -> bool:
        """Whether a human should look at this entry's date before trusting it."""
        return not self.locked and self.date_source in ("ambiguous", "unparsed")

    def as_date(self) -> Optional[date]:
        if not self.session_date:
            return None
        try:
            return date.fromisoformat(self.session_date)
        except ValueError:
            return None


def order_path(input_root: Path) -> Path:
    return Path(input_root) / ORDER_FILENAME


def discover_trackable_stems(workspace) -> list[str]:
    """The FOVs in *workspace* that tracking can consider, as output stems.

    Read from ``output/`` rather than from ``workspace.tifs``: a session is
    something the pipeline has produced output for, not every stack on disk.
    Prairie View drops single-frame reference snapshots beside each recording
    (``..._beh-006-Ch2-16bit-Reference.tif``, ``...-Window1-Ch2-8bit-...``),
    and the reference workspace has six of them against three real sessions —
    ordering those by hand would be busywork over files that are not sessions.
    """
    output_root = Path(workspace.output_root)
    if not output_root.exists():
        return []
    return sorted(p.name for p in output_root.iterdir() if p.is_dir())


def propose_order(stems: Iterable[str]) -> list[SessionOrderEntry]:
    """Best-effort initial ordering of *stems* from their filename dates.

    Datable stems sort by date then stem. Stems whose date is ambiguous or
    unparseable sort last (by stem) rather than being silently interleaved on a
    guess — they are exactly the ones a human needs to place.
    """
    from roigbiv.registry.filename import parse_filename_metadata

    datable: list[tuple[date, str, str]] = []
    undatable: list[tuple[str, str, Optional[str]]] = []

    for stem in stems:
        meta = parse_filename_metadata(stem)
        iso = meta.session_date.isoformat() if meta.session_date else None
        if meta.session_date is not None and meta.date_source in ("mmddyy", "yymmdd"):
            datable.append((meta.session_date, stem, meta.date_source))
        else:
            undatable.append((stem, meta.date_source, iso))

    entries: list[SessionOrderEntry] = []
    for session_date, stem, source in sorted(datable, key=lambda t: (t[0], t[1])):
        entries.append(SessionOrderEntry(
            stem=stem,
            index=len(entries),
            session_date=session_date.isoformat(),
            date_source=source,
        ))
    for stem, source, iso in sorted(undatable, key=lambda t: t[0]):
        entries.append(SessionOrderEntry(
            stem=stem,
            index=len(entries),
            session_date=iso,
            date_source=source,
        ))
    return entries


def load_order(input_root: Path) -> list[SessionOrderEntry]:
    """Read ``session_order.json``. Returns ``[]`` when absent or unreadable."""
    path = order_path(input_root)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    entries = [
        SessionOrderEntry(
            stem=str(e["stem"]),
            index=int(e.get("index", i)),
            session_date=e.get("session_date"),
            date_source=e.get("date_source", "unparsed"),
            locked=bool(e.get("locked", False)),
        )
        for i, e in enumerate(payload.get("order", []))
        if e.get("stem")
    ]
    return _renumber(sorted(entries, key=lambda e: e.index))


def save_order(input_root: Path, entries: list[SessionOrderEntry]) -> Path:
    path = order_path(input_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": _SCHEMA,
        "order": [asdict(e) for e in _renumber(entries)],
    }, indent=2))
    return path


def resolve_order(input_root: Path, stems: Iterable[str]) -> list[SessionOrderEntry]:
    """The workspace's current order, reconciled against the stems on disk.

    Saved positions are authoritative for stems already known. Stems that have
    appeared since are appended in proposal order; stems that have vanished are
    dropped. A confirmed timeline is never reshuffled by a re-scan.
    """
    stems = list(stems)
    saved = {e.stem: e for e in load_order(input_root)}
    known = [saved[s] for s in
             sorted((s for s in stems if s in saved), key=lambda s: saved[s].index)]
    new = [e for e in propose_order(s for s in stems if s not in saved)]

    for entry in new:
        entry.index = 0  # renumbered below; proposal order is preserved
    return _renumber(known + new)


def reorder(entries: list[SessionOrderEntry], stems: list[str]) -> list[SessionOrderEntry]:
    """Apply a human-supplied *stems* ordering, marking the result locked.

    Stems not present in *stems* keep their relative position at the end, so a
    partial reorder can never drop a session off the timeline.
    """
    by_stem = {e.stem: e for e in entries}
    ordered: list[SessionOrderEntry] = []
    for stem in stems:
        entry = by_stem.pop(stem, None)
        if entry is not None:
            entry.locked = True
            ordered.append(entry)
    ordered.extend(sorted(by_stem.values(), key=lambda e: e.index))
    return _renumber(ordered)


def _renumber(entries: list[SessionOrderEntry]) -> list[SessionOrderEntry]:
    for i, entry in enumerate(entries):
        entry.index = i
    return entries
