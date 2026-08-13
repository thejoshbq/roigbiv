"""Human-in-the-loop cross-session cell links — the correspondence half of
tracking HITL.

:mod:`roigbiv.pipeline.centroid_edits` lets a human fix *which centroids
exist*; this module lets them fix *which centroids are the same cell*. Same
pattern as :mod:`roigbiv.pipeline.corrections`: an append-only JSONL log of
:class:`MatchOp`, replayed purely to get an effective state. What's replayed
here is not a mask but an assignment — ``(session_stem, local_label_id) ->
global_cell_id`` — because that pair is what survives a fresh ROICaT run
whereas a ``global_cell_id`` does not (a full re-match mints new gcids for
every cell). One log per FOV, at
``{input_root}/corrections/matches/{fov_id}.jsonl``, so an "undo last" on one
FOV can never touch another's.

Two operations:

``link``    merge the whole cells that own every named member into one cell.
            Associative — which pair a human happened to click first cannot
            change the merged result — and *whole cells*, not just the two
            clicked members, because leaving the rest of each cell behind
            would silently orphan sessions the human never looked at.
``unlink``  give one member a cell of its own again.

Both are replayed **sequentially** against a mutable working copy of the
assignment, not resolved as one static graph over the whole op list — a human
can unlink something and then re-link it differently, and only replaying in
order reproduces that. This is why ``apply_match_ops`` is not simply
"union-find over every link op ever written."

**A ``link`` that would leave two members of one session in one cell is
rejected outright**, not silently applied. ``cell_observation`` allows two
observations to share a ``global_cell_id`` (only ``(session_id,
local_label_id)`` is unique), so nothing at the database layer stops it — but
readers built on "one member per session" — the display index in
``ui/services/tracked_cells.py``, ``CellTimeline.local_label_ids`` in
``anomalies.py`` — would silently take the last-written one and drop the
other, which is a worse failure than refusing the click.

**Unlink's new cell id is deterministic**, derived from the op's own id via
``uuid5``, not ``uuid4``. A full re-match replays this log from scratch every
time (see ``run_tracking``), and a random id would hand the same physical cell
a new identity on every replay — which is exactly the churn this whole feature
exists to stop.

**Merge survivor is the gcid whose earliest member has the smallest
``sequence_index``**, ties broken by the smaller gcid string. This deliberately
does *not* reuse ``orchestrator.py``'s "earliest-created wins" convention: that
convention compares ``Cell.first_seen_session_id``, which
``_resolve_session_id`` mints as a fresh ``uuid4`` on every registration, so it
actually preserves the lexicographically smallest random id, not the
earliest-created cell. Reusing it here would propagate that bug into a second
place. ``sequence_index`` is what the human actually ordered sessions by, so
it is the only signal here that means what its name says.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from roigbiv.registry.store.base import CellRecord, ObservationRecord

# A member is one ROI: which session (by stem, not session_id — see module
# docstring) and which local label within it.
Member = tuple[str, int]


@dataclass
class MatchOp:
    """One human-in-the-loop cross-session correspondence edit."""

    op: str                        # "link" | "unlink"
    fov_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    members: Optional[list] = None   # link: [[stem, label], ...]
    member: Optional[list] = None    # unlink: [stem, label]

    notes: str = ""

    @classmethod
    def link(cls, fov_id: str, members: list, notes: str = "") -> "MatchOp":
        return cls(op="link", fov_id=fov_id,
                  members=[[str(stem), int(label)] for stem, label in members],
                  notes=notes)

    @classmethod
    def unlink(cls, fov_id: str, member, notes: str = "") -> "MatchOp":
        stem, label = member
        return cls(op="unlink", fov_id=fov_id, member=[str(stem), int(label)],
                  notes=notes)

    def to_jsonable(self) -> dict:
        out = {k: v for k, v in asdict(self).items() if v is not None and v != ""}
        out["op"] = self.op
        out["fov_id"] = self.fov_id
        out["id"] = self.id
        out["ts"] = self.ts
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "MatchOp":
        return cls(
            op=d["op"],
            fov_id=d["fov_id"],
            id=d.get("id", str(uuid.uuid4())),
            ts=d.get("ts", datetime.now(timezone.utc).isoformat()),
            members=d.get("members"),
            member=d.get("member"),
            notes=d.get("notes", ""),
        )


# ── On-disk layout helpers ─────────────────────────────────────────────────


def _matches_dir(input_root: Path) -> Path:
    """``{input_root}/corrections/matches/`` — created on demand."""
    d = Path(input_root) / "corrections" / "matches"
    d.mkdir(parents=True, exist_ok=True)
    return d


def match_log_path(input_root: Path, fov_id: str) -> Path:
    """Path to one FOV's cross-session correspondence log."""
    return _matches_dir(input_root) / f"{fov_id}.jsonl"


def append_match_op(input_root: Path, op: MatchOp) -> None:
    """Append one op to its FOV's JSONL log. Creates the file if missing."""
    log_path = match_log_path(input_root, op.fov_id)
    with log_path.open("a") as f:
        f.write(json.dumps(op.to_jsonable()) + "\n")


def load_match_ops(input_root: Path, fov_id: str) -> list[MatchOp]:
    """Read all ops for one FOV (empty list if no log exists)."""
    log_path = match_log_path(input_root, fov_id)
    if not log_path.exists():
        return []
    ops: list[MatchOp] = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ops.append(MatchOp.from_dict(json.loads(line)))
    return ops


def write_match_ops(input_root: Path, fov_id: str, ops: list[MatchOp]) -> None:
    """Replace one FOV's entire log with ``ops`` (used for undo: drop the tail)."""
    log_path = match_log_path(input_root, fov_id)
    if not ops:
        if log_path.exists():
            log_path.unlink()
        return
    with log_path.open("w") as f:
        for op in ops:
            f.write(json.dumps(op.to_jsonable()) + "\n")


# ── Replay ─────────────────────────────────────────────────────────────────


def apply_match_ops(
    assignment: dict,
    ops: list[MatchOp],
    *,
    order: dict,
) -> tuple[dict, list[str]]:
    """Replay ``ops`` against ``assignment`` and return the edited assignment.

    Parameters
    ----------
    assignment : dict[Member, str]
        ``(stem, label) -> global_cell_id`` for every ROI currently present.
        A member absent from this dict is treated as not present (e.g. its
        centroid was since deleted) — a ``link``/``unlink`` naming it is
        adjusted or skipped, never invented.
    ops : list[MatchOp]
        Replayed strictly in order — see module docstring for why this must
        be sequential rather than a static graph.
    order : dict[str, int]
        ``stem -> sequence_index``, used only to pick a merge's surviving
        ``global_cell_id`` (see module docstring). A stem missing from
        ``order`` sorts last.

    Returns
    -------
    (dict[Member, str], list[str])
        The effective assignment, and human-readable warnings for every op
        that was adjusted or skipped. Never mutates ``assignment``.
    """
    current: dict = dict(assignment)
    warnings: list[str] = []

    for op in ops:
        if op.op == "link":
            _apply_link(current, op, order, warnings)
        elif op.op == "unlink":
            _apply_unlink(current, op, warnings)
        else:
            warnings.append(f"unknown: op {op.id} has unknown op '{op.op}' — skipped")

    return current, warnings


def _apply_link(current: dict, op: MatchOp, order: dict, warnings: list[str]) -> None:
    requested = [tuple(m) for m in (op.members or [])]
    if len(requested) < 2:
        warnings.append(f"link: op {op.id} names fewer than 2 members — skipped")
        return

    present = [m for m in requested if m in current]
    for m in requested:
        if m not in current:
            warnings.append(
                f"link: member {m[0]}:{m[1]} in op {op.id} is not present "
                f"(deleted?) — dropped from the link")
    if len(present) < 2:
        warnings.append(
            f"link: op {op.id} has fewer than 2 present members after drops "
            f"— skipped")
        return

    # Merge whole cells, not just the clicked members: every member currently
    # sharing a touched cell comes along, so nothing a human didn't look at
    # gets silently orphaned.
    touched_gcids = {current[m] for m in present}
    all_members = [m for m, g in current.items() if g in touched_gcids]

    stems = [m[0] for m in all_members]
    if len(stems) != len(set(stems)):
        dupe_stem = next(s for s in stems if stems.count(s) > 1)
        warnings.append(
            f"link: op {op.id} rejected — session {dupe_stem!r} would end up "
            f"with two members in one cell; unlink one first")
        return

    def earliest_seq(gcid: str) -> float:
        seqs = [order.get(m[0], float("inf"))
                for m, g in current.items() if g == gcid]
        return min(seqs) if seqs else float("inf")

    survivor = min(touched_gcids, key=lambda g: (earliest_seq(g), g))
    for m in all_members:
        current[m] = survivor


def _apply_unlink(current: dict, op: MatchOp, warnings: list[str]) -> None:
    if not op.member:
        warnings.append(f"unlink: op {op.id} names no member — skipped")
        return
    member = tuple(op.member)
    if member not in current:
        warnings.append(
            f"unlink: member {member[0]}:{member[1]} in op {op.id} is not "
            f"present (deleted?) — skipped")
        return

    gcid = current[member]
    siblings = [m for m, g in current.items() if g == gcid and m != member]
    if not siblings:
        warnings.append(
            f"unlink: {member[0]}:{member[1]} is already the only member of "
            f"its cell — no-op")
        return

    # Deterministic — see module docstring on why not uuid4.
    new_gcid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"roigbiv-unlink/{op.id}"))
    current[member] = new_gcid


# ── DB materializer ──────────────────────────────────────────────────────


@dataclass
class EditApplyReport:
    """Outcome of replaying one FOV's centroid + match edits into the registry."""

    fov_id: str
    n_sessions: int
    n_observations: int
    n_cells_created: int
    warnings: list


def apply_tracking_edits(
    fov_id: str, input_root: Path, store, cfg=None,
) -> EditApplyReport:
    """Re-stamp every session of ``fov_id`` and rebuild its observations.

    This is the single place both a fresh ``run_tracking`` pass and an
    interactive ``/cells`` edit call to reach the same end state — a FOV
    whose centroid and correspondence logs have been fully replayed. No
    ROICaT match runs here: every observation is derived either from an
    existing one (the label was already assigned a cell) or from a
    deterministic new cell, then the match log adjusts correspondences on
    top. That's what makes an edit apply in well under a second.

    Labels are restricted to what ``write_merged_masks`` actually stamped for
    each session — never to what the ops *asked* for. A ``move`` can bury one
    label completely under another (see ``StampedMasks.present_labels``), and
    building the assignment from the ops instead of the stamped image would
    write an observation for a label the sheet can never draw, while still
    counting the cell as present.
    """
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    if cfg is None:
        from roigbiv.pipeline.types import PipelineConfig
        cfg = PipelineConfig(no_viewer=True)

    sessions = store.list_sessions(fov_id)  # sequence_index order, guaranteed
    warnings: list = []
    order: dict = {}                # stem -> sequence_index
    session_id_by_stem: dict = {}
    present_by_session: dict = {}   # session_id -> [label, ...]

    for i, sess in enumerate(sessions):
        stem = Path(sess.output_dir).name
        session_id_by_stem[stem] = sess.session_id
        order[stem] = sess.sequence_index if sess.sequence_index is not None else i

        stamped = write_merged_masks(Path(sess.output_dir), cfg)
        if stamped is None:
            # No centroids.json — either a pre-centroid-discovery FOV (a real
            # merged_masks.tif from a full cascade may still be on disk) or a
            # genuinely empty session (present_labels_on_disk then returns []).
            present_by_session[sess.session_id] = _present_labels_on_disk(
                Path(sess.output_dir))
            continue
        warnings.extend(f"{stem}: {w}" for w in stamped.edit_warnings)
        present_by_session[sess.session_id] = (
            list(stamped.present_labels) if stamped.written
            else list(_present_labels_on_disk(Path(sess.output_dir)))
        )

    # Assignment restricted to present labels; existing observations are
    # reused so their gcid (and the human's prior link decisions) carries
    # forward, and a present label with none gets a deterministic new cell.
    assignment: dict = {}
    meta: dict = {}  # member -> (match_score, cluster_label)
    for sess in sessions:
        stem = Path(sess.output_dir).name
        present = present_by_session[sess.session_id]
        present_set = set(present)
        obs_by_label = {
            int(o.local_label_id): o
            for o in store.list_observations_for_session(sess.session_id)
        }
        for label in present:
            o = obs_by_label.get(label)
            if o is not None:
                assignment[(stem, label)] = o.global_cell_id
                meta[(stem, label)] = (o.match_score, o.cluster_label)
            else:
                assignment[(stem, label)] = str(uuid.uuid5(
                    uuid.NAMESPACE_URL, f"roigbiv-newcell/{sess.session_id}/{label}"))
        for label in obs_by_label:
            if label not in present_set:
                warnings.append(
                    f"{stem}: observation for label {label} has no matching "
                    f"centroid — dropped")

    ops = load_match_ops(input_root, fov_id)
    effective, match_warnings = apply_match_ops(assignment, ops, order=order)
    warnings.extend(match_warnings)

    existing_cells = {c.global_cell_id for c in store.list_cells(fov_id)}
    new_gcids = sorted(set(effective.values()) - existing_cells)
    for gcid in new_gcids:
        members = [m for m, g in effective.items() if g == gcid]
        first_stem = min(members, key=lambda m: order.get(m[0], float("inf")))[0]
        store.insert_cell(CellRecord(
            global_cell_id=gcid, fov_id=fov_id,
            first_seen_session_id=session_id_by_stem[first_stem],
        ))

    observations = [
        ObservationRecord(
            observation_id=str(uuid.uuid4()),
            global_cell_id=gcid,
            session_id=session_id_by_stem[stem],
            local_label_id=label,
            match_score=meta.get((stem, label), (None, None))[0],
            cluster_label=meta.get((stem, label), (None, None))[1],
        )
        for (stem, label), gcid in effective.items()
    ]
    store.replace_observations(list(session_id_by_stem.values()), observations)

    return EditApplyReport(
        fov_id=fov_id,
        n_sessions=len(sessions),
        n_observations=len(observations),
        n_cells_created=len(new_gcids),
        warnings=warnings,
    )


def _present_labels_on_disk(output_dir: Path) -> list:
    """Labels in an on-disk ``merged_masks.tif`` that ``write_merged_masks``
    left untouched (a full-cascade FOV — see ``StampedMasks.written``)."""
    import numpy as np
    import tifffile

    masks_path = output_dir / "merged_masks.tif"
    if not masks_path.exists():
        return []
    arr = np.asarray(tifffile.imread(str(masks_path)))
    return sorted(int(x) for x in np.unique(arr) if x != 0)
