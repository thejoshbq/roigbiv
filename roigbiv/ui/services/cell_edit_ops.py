"""Cross-session cell edits, decided server-side and applied as additive ops.

One entry point — :func:`apply_gesture` — turns a described user gesture into
appended JSONL ops, replays them through
:func:`roigbiv.registry.cell_edits.apply_tracking_edits`, and hands back the
refreshed FOV. Pipeline output is never mutated; see ADR-0004.

Why this is not in ``pages/tracking.py``
----------------------------------------
It has two callers: the Flask endpoint the browser posts to
(:mod:`roigbiv.ui.routes.cells_api`) and its own tests. Neither wants Dash
imported, and the endpoint cannot use a callback's seven-tuple return. Nothing
here may import ``dash``.

Gestures, not modes
-------------------
The page used to carry a mode radio and a two-click "pickup" for move and
link. Both are gone: the browser knows which shape was hit and can report an
exact image coordinate, so every gesture arrives complete. In particular a
*move* carries the raw destination pixel. The old two-click path resolved that
second click through a nearest-centroid snap, which silently turned any nudge
shorter than the stamp radius into a no-op — the destination resolved back to
the centroid being moved.

Selection is client state and is never posted; it arrives only as
``selected_gcid``, the context that makes "place here" and "link" mean
something. The one exception is ``confirm``, where the ghost that was clicked
names its own cell — which is exactly the selection click it saves.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

from roigbiv.ui.services.tracked_cells import (
    TrackedFOV,
    invalidate_tracked_fov,
    load_tracked_fov_cached,
)

GESTURE_KINDS = ("add", "delete", "move", "link", "undo", "confirm")


@dataclass
class Gesture:
    """One described user action, complete enough to decide on by itself."""

    kind: str
    stem: Optional[str] = None
    label: Optional[int] = None
    # Destination for "move", placement for "add" — full-resolution image
    # pixels, exactly as clicked. Never snapped to a centroid.
    y: Optional[float] = None
    x: Optional[float] = None
    # The cell the gesture is performed against: "add" links to it when it is
    # missing from this session, "link" merges the clicked member into it.
    selected_gcid: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: dict) -> "Gesture":
        """Build from a posted JSON body, raising ``ValueError`` on nonsense.

        Validation is by gesture kind rather than a blanket schema: a delete
        carries no coordinate and an undo carries nothing at all, so requiring
        every field would reject valid gestures.
        """
        if not isinstance(payload, dict):
            raise ValueError("gesture must be an object")
        kind = payload.get("kind")
        if kind not in GESTURE_KINDS:
            raise ValueError(f"unknown gesture kind: {kind!r}")

        gesture = cls(
            kind=kind,
            stem=_opt_str(payload.get("stem")),
            label=_opt_int(payload.get("label")),
            y=_opt_float(payload.get("y")),
            x=_opt_float(payload.get("x")),
            selected_gcid=_opt_str(payload.get("selected_gcid")),
        )
        if kind != "undo" and not gesture.stem:
            raise ValueError(f"{kind} requires a stem")
        if kind in ("delete", "move", "link") and gesture.label is None:
            raise ValueError(f"{kind} requires a label")
        if kind in ("add", "move", "confirm") and (gesture.y is None
                                                   or gesture.x is None):
            raise ValueError(f"{kind} requires y and x")
        return gesture


@dataclass
class GestureResult:
    """What the gesture did, and what the page should show afterwards.

    ``fov`` is ``None`` when nothing was written — the caller then leaves the
    rendered sheet alone rather than repainting it identically.
    """

    ok: bool
    message: str
    selected_gcid: Optional[str] = None
    warnings: list = field(default_factory=list)
    fov: Optional[TrackedFOV] = None
    # Advisory HTTP status for the route. 409 is reserved for "tracking is
    # running", which is a temporary conflict rather than a bad request.
    status: int = 200


def _opt_str(value) -> Optional[str]:
    return str(value) if value not in (None, "") else None


def _opt_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected an integer, got {value!r}") from None


def _opt_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected a number, got {value!r}") from None


# ── op writers ─────────────────────────────────────────────────────────────
#
# Thin wrappers over the two edit logs, kept as separate named functions so a
# test can assert *which* op a gesture chose without reading JSONL back off
# disk. Imports are local: the centroid and registry modules pull in tifffile
# and SQLAlchemy, and importing this module must stay cheap.


def _next_centroid_label(output_dir: Path) -> int:
    from roigbiv.pipeline.centroid_edits import load_centroid_ops, next_label
    from roigbiv.pipeline.centroid_masks import load_effective_centroids

    effective, _warnings = load_effective_centroids(output_dir)
    return next_label(effective, load_centroid_ops(output_dir))


def _do_add(output_dir: Path, y: float, x: float) -> int:
    from roigbiv.pipeline.centroid_edits import CentroidOp, append_centroid_op

    label = _next_centroid_label(output_dir)
    append_centroid_op(output_dir, CentroidOp.add(label, y, x))
    return label


def _do_delete(output_dir: Path, label: int) -> None:
    from roigbiv.pipeline.centroid_edits import CentroidOp, append_centroid_op

    append_centroid_op(output_dir, CentroidOp.delete(label))


def _do_move(output_dir: Path, label: int, y: float, x: float) -> None:
    from roigbiv.pipeline.centroid_edits import CentroidOp, append_centroid_op

    append_centroid_op(output_dir, CentroidOp.move(label, y, x))


def _do_link(fov_id: str, input_root: Path, members: list) -> str:
    from roigbiv.registry.cell_edits import MatchOp, append_match_op

    op = MatchOp.link(fov_id, members)
    append_match_op(input_root, op)
    return op.id


def _do_unlink(fov_id: str, input_root: Path, member: tuple) -> str:
    from roigbiv.registry.cell_edits import MatchOp, append_match_op

    op = MatchOp.unlink(fov_id, member)
    append_match_op(input_root, op)
    return op.id


def _rejection_for(report, op_id: Optional[str]) -> Optional[str]:
    """The replay's complaint about *op_id*, if it refused to apply it.

    ``apply_match_ops`` names the offending op in its warning, which is the
    only way to tell "your link was applied" from "your link was dropped" —
    both otherwise return a perfectly valid FOV.
    """
    if not op_id:
        return None
    for warning in getattr(report, "warnings", []) or []:
        if str(op_id) in str(warning):
            return warning
    return None


def _rollback_match_op(fov_id: str, input_root: Path, op_id: str) -> None:
    """Drop a rejected op back off the log.

    A rejected op is inert but not silent: it stays in the log and re-emits its
    warning on *every* later replay, so one impossible link would otherwise
    append its complaint to every message the page shows from then on.
    """
    from roigbiv.registry.cell_edits import load_match_ops, write_match_ops

    ops = load_match_ops(input_root, fov_id)
    write_match_ops(input_root, fov_id, [o for o in ops if o.id != op_id])


def _apply_and_reload(fov_id: str, input_root: Path, registry_cfg):
    """Replay both logs into the registry, then return the fresh FOV.

    The single choke point every gesture runs through. Reloads through the
    *same* ``registry_cfg`` that was just written with, via
    ``load_tracked_fov_cached`` directly rather than any request-scoped
    helper — this function must work outside a live Flask request context.
    """
    from roigbiv.registry import build_store
    from roigbiv.registry.cell_edits import apply_tracking_edits

    store = build_store(cfg=registry_cfg)
    report = apply_tracking_edits(fov_id, input_root, store)
    invalidate_tracked_fov(fov_id, cfg=registry_cfg)
    return load_tracked_fov_cached(fov_id, cfg=registry_cfg), report


def _tracking_is_active() -> bool:
    from roigbiv.ui.services.tracking_runner import get_tracking_runner

    return get_tracking_runner().snapshot().active


# ── FOV lookups ────────────────────────────────────────────────────────────


def _session_index_for_stem(fov: TrackedFOV, stem: str) -> Optional[int]:
    for i, session in enumerate(fov.sessions):
        if session.stem == stem:
            return i
    return None


def _gcid_present_in_session(fov: TrackedFOV, gcid: Optional[str],
                             session_index: Optional[int]) -> bool:
    if not gcid or session_index is None:
        return False
    cell = fov.cell_by_gcid(gcid)
    return cell is not None and cell.present[session_index]


def _any_member_of(fov: TrackedFOV, gcid: Optional[str]) -> Optional[tuple]:
    """One ``(stem, label)`` this cell actually owns, to link a new one against.

    Merging happens by whole cell (see ``cell_edits.apply_match_ops``), so any
    one existing member is enough to pull the new centroid into the group.
    """
    cell = fov.cell_by_gcid(gcid)
    if cell is None:
        return None
    for i, label in enumerate(cell.local_label_ids):
        if label is not None:
            return fov.sessions[i].stem, label
    return None


def _label_exists(fov: TrackedFOV, session_index: int, label: int) -> bool:
    """Whether *label* is a real (non-ghost) footprint in this session.

    Ghosts carry a negative id (see ``tracked_cells._ghost_label_id``) and own
    nothing in this session's mask, so they can be selected but never edited.
    """
    if label <= 0:
        return False
    return any(r.label_id == label for r in fov.sessions[session_index].rois)


def _point_in_ring(y: float, x: float, ys: list, xs: list) -> bool:
    """Standard even-odd ray cast, in image coordinates."""
    inside = False
    n = min(len(ys), len(xs))
    j = n - 1
    for i in range(n):
        # The bracket test guarantees ys[j] != ys[i], so the division is safe.
        if (ys[i] > y) != (ys[j] > y):
            if x < xs[i] + (y - ys[i]) * (xs[j] - xs[i]) / (ys[j] - ys[i]):
                inside = not inside
        j = i
    return inside


def _roi_containing(fov: TrackedFOV, session_index: int, y: float, x: float):
    """The real footprint enclosing *(y, x)* in this session, if there is one.

    Containment rather than nearest-within-a-radius: sessions are not
    co-registered, so any distance threshold would be a guess about drift this
    codebase cannot measure. "The point is inside this outline" is a claim the
    geometry actually supports.

    Rings are combined even-odd so a mask with a hole answers correctly; in
    practice ADR-0003 stamps disks and there is one ring.
    """
    for roi in fov.sessions[session_index].rois:
        if int(roi.label_id) <= 0:
            continue
        if sum(_point_in_ring(y, x, ys, xs)
               for ys, xs in roi.contours if ys) % 2:
            return roi
    return None


def _note(result: GestureResult, note: str) -> GestureResult:
    """Append context without rewriting what the operation said it did."""
    if not result.ok:
        return result
    return replace(result, message=f"{result.message} ({note})")


def _report_msg(report, verb: str) -> str:
    warnings = list(getattr(report, "warnings", []) or [])
    return f"{verb} — " + "; ".join(warnings) if warnings else verb


def _refused(message: str, *, status: int = 200,
             selected_gcid: Optional[str] = None) -> GestureResult:
    return GestureResult(ok=False, message=message, selected_gcid=selected_gcid,
                         status=status)


# ── the dispatcher ─────────────────────────────────────────────────────────


def apply_gesture(fov: TrackedFOV, gesture: Gesture, *, input_root: Path,
                  registry_cfg) -> GestureResult:
    """Write the ops *gesture* implies, replay them, return the fresh FOV.

    *fov* is passed in already loaded so a caller that has one (and a test
    that fabricates one) need not go through the store twice.
    """
    if _tracking_is_active():
        return _refused(
            "tracking is running for this workspace — try again once it finishes",
            status=409, selected_gcid=gesture.selected_gcid)

    if gesture.kind == "undo":
        return _undo(fov, input_root, registry_cfg, gesture.selected_gcid)

    session_index = _session_index_for_stem(fov, gesture.stem)
    if session_index is None:
        return _refused(f"no session named {gesture.stem!r} in this FOV",
                        status=400, selected_gcid=gesture.selected_gcid)
    session = fov.sessions[session_index]
    if session.output_dir is None:
        return _refused("this session has no on-disk output to edit",
                        selected_gcid=gesture.selected_gcid)
    output_dir = Path(session.output_dir)

    if gesture.kind == "add":
        return _add(fov, gesture, session_index, output_dir, input_root,
                    registry_cfg)
    if gesture.kind == "delete":
        return _delete(fov, gesture, session_index, output_dir, input_root,
                       registry_cfg)
    if gesture.kind == "move":
        return _move(fov, gesture, session_index, output_dir, input_root,
                     registry_cfg)
    if gesture.kind == "link":
        return _link(fov, gesture, session_index, input_root, registry_cfg)
    if gesture.kind == "confirm":
        return _confirm(fov, gesture, session_index, output_dir, input_root,
                        registry_cfg)
    return _refused(f"unknown gesture: {gesture.kind}", status=400,
                    selected_gcid=gesture.selected_gcid)


def _confirm(fov: TrackedFOV, gesture: Gesture, session_index: int,
             output_dir: Path, input_root: Path, registry_cfg) -> GestureResult:
    """Ctrl-click a ghost: "this cell *is* here", in one gesture.

    A ghost already names its cell, so unlike shift-click-to-link this needs no
    prior selection — which is the click it removes. What "is here" should
    write depends on whether anything was detected under the ghost:

    * nothing there — place a centroid and adopt it. Note that the position
      comes from the session the cell was last *seen* in, and sessions are not
      co-registered (ROICaT computes an alignment transform during matching and
      discards it), so the centroid lands wherever that other frame put it plus
      whatever drift exists. The message says so; a drag fixes it.
    * an unmatched outline there — link that instead. It is both the real
      detection and immune to the drift above.
    * an outline belonging to a cell already seen in several sessions —
      refuse. Adopting it would merge two multi-session cells on the strength
      of one click, which is a claim big enough to be made deliberately.
    """
    gcid = gesture.selected_gcid
    if not gcid:
        return _refused("ctrl-click a ghost outline to confirm its cell is here")
    cell = fov.cell_by_gcid(gcid)
    if cell is None:
        return _refused("that cell is no longer part of this FOV")
    if _gcid_present_in_session(fov, gcid, session_index):
        return _refused(
            f"cell #{cell.index} already has a centroid in this session",
            selected_gcid=gcid)

    hit = _roi_containing(fov, session_index, gesture.y, gesture.x)
    if hit is None:
        return _note(
            _add(fov, replace(gesture, kind="add"), session_index, output_dir,
                 input_root, registry_cfg),
            "placed from another session's frame — check the position")

    other = fov.cell_by_gcid(fov.gcid_for_label(session_index, hit.label_id))
    if other is not None and other.n_present > 1:
        return _refused(
            f"that outline is already cell #{other.index}, seen in "
            f"{other.n_present} sessions — shift-click it if you really mean "
            f"to merge the two", selected_gcid=gcid)

    return _note(
        _link(fov, replace(gesture, kind="link", label=int(hit.label_id)),
              session_index, input_root, registry_cfg),
        "an outline was already there")


def _add(fov: TrackedFOV, gesture: Gesture, session_index: int,
         output_dir: Path, input_root: Path, registry_cfg) -> GestureResult:
    """Place a centroid, and adopt it into the selected cell when that fits.

    "Place here": when a cell is selected but missing from *this* session,
    one click both adds its centroid and links it — by far the most common
    repair, and three separate gestures if composed by hand.
    """
    stem = gesture.stem
    selected = gesture.selected_gcid
    place_here = bool(selected) and not _gcid_present_in_session(
        fov, selected, session_index)

    new_label = _do_add(output_dir, gesture.y, gesture.x)
    verb = "added a centroid"
    link_op_id = None
    if place_here:
        anchor = _any_member_of(fov, selected)
        if anchor is not None:
            link_op_id = _do_link(fov.fov_id, input_root,
                                  [anchor, (stem, new_label)])
            verb = "added and linked a centroid"

    new_fov, report = _apply_and_reload(fov.fov_id, input_root, registry_cfg)

    # The centroid is placed either way; only the adoption can be refused.
    rejection = _rejection_for(report, link_op_id)
    if rejection is not None:
        _rollback_match_op(fov.fov_id, input_root, link_op_id)
        new_fov, report = _apply_and_reload(fov.fov_id, input_root, registry_cfg)
        verb = f"added the centroid but could not link it — {rejection}"
        place_here = False

    new_index = _session_index_for_stem(new_fov, stem)
    new_gcid = (new_fov.gcid_for_label(new_index, new_label)
                if new_index is not None else None)
    if place_here:
        # The link may have re-minted the group's id; falling back to the
        # selection keeps the drawer pointed at the cell just repaired.
        new_gcid = new_gcid or selected
    return GestureResult(ok=True, message=_report_msg(report, verb),
                         selected_gcid=new_gcid,
                         warnings=list(getattr(report, "warnings", []) or []),
                         fov=new_fov)


def _delete(fov: TrackedFOV, gesture: Gesture, session_index: int,
            output_dir: Path, input_root: Path, registry_cfg) -> GestureResult:
    label = gesture.label
    if not _label_exists(fov, session_index, label):
        return _refused("that outline has no centroid in this session to delete",
                        selected_gcid=gesture.selected_gcid)

    deleted_gcid = fov.gcid_for_label(session_index, label)
    _do_delete(output_dir, label)
    new_fov, report = _apply_and_reload(fov.fov_id, input_root, registry_cfg)
    # Only the deleted cell's own selection is cleared; an unrelated one
    # survives so a delete does not interrupt whatever was being reviewed.
    selected = (None if gesture.selected_gcid == deleted_gcid
                else gesture.selected_gcid)
    return GestureResult(ok=True, message=_report_msg(report, "deleted a centroid"),
                         selected_gcid=selected,
                         warnings=list(getattr(report, "warnings", []) or []),
                         fov=new_fov)


def _move(fov: TrackedFOV, gesture: Gesture, session_index: int,
          output_dir: Path, input_root: Path, registry_cfg) -> GestureResult:
    """Relocate a centroid to the exact posted coordinate.

    No snapping of any kind: a two-pixel nudge onto the real soma is the
    common case and has to land where it was dropped.
    """
    label = gesture.label
    if not _label_exists(fov, session_index, label):
        return _refused("that outline has no centroid in this session to move",
                        selected_gcid=gesture.selected_gcid)

    _do_move(output_dir, label, gesture.y, gesture.x)
    new_fov, report = _apply_and_reload(fov.fov_id, input_root, registry_cfg)
    new_index = _session_index_for_stem(new_fov, gesture.stem)
    new_gcid = (new_fov.gcid_for_label(new_index, label)
                if new_index is not None else None)
    return GestureResult(ok=True, message=_report_msg(report, "moved a centroid"),
                         selected_gcid=new_gcid or gesture.selected_gcid,
                         warnings=list(getattr(report, "warnings", []) or []),
                         fov=new_fov)


def _link(fov: TrackedFOV, gesture: Gesture, session_index: int,
          input_root: Path, registry_cfg) -> GestureResult:
    """Merge the clicked member into the selected cell — or pull it back out.

    Shift-clicking a member that is *already* part of the selected cell reads
    as "this one does not belong", which is an unlink. A link that would merge
    a cell with itself is the only other reading and is a no-op, so the
    ambiguity is not real.
    """
    label = gesture.label
    selected = gesture.selected_gcid
    if not selected:
        return _refused(
            "select a cell first, then shift-click its counterpart in "
            "another session")
    if not _label_exists(fov, session_index, label):
        return _refused("that outline has no centroid in this session to link")

    member = (gesture.stem, label)
    member_gcid = fov.gcid_for_label(session_index, label)

    if member_gcid is not None and member_gcid == selected:
        op_id = _do_unlink(fov.fov_id, input_root, member)
        verb = "unlinked"
    else:
        anchor = _any_member_of(fov, selected)
        if anchor is None:
            return _refused("the selected cell owns no centroid to link against")
        if anchor == member:
            return _refused("that is the selected cell itself")
        op_id = _do_link(fov.fov_id, input_root, [anchor, member])
        verb = "linked"

    new_fov, report = _apply_and_reload(fov.fov_id, input_root, registry_cfg)

    # The replay is allowed to refuse — two members of one session cannot share
    # a cell. Saying "linked" over the top of that would be a lie the sheet
    # then visibly contradicts, so the refusal is reported as one.
    rejection = _rejection_for(report, op_id)
    if rejection is not None:
        _rollback_match_op(fov.fov_id, input_root, op_id)
        new_fov, _report = _apply_and_reload(fov.fov_id, input_root, registry_cfg)
        return GestureResult(ok=False, message=rejection,
                             selected_gcid=selected, fov=new_fov)

    new_index = _session_index_for_stem(new_fov, gesture.stem)
    new_gcid = (new_fov.gcid_for_label(new_index, label)
                if new_index is not None else None)
    return GestureResult(ok=True, message=_report_msg(report, verb),
                         selected_gcid=new_gcid,
                         warnings=list(getattr(report, "warnings", []) or []),
                         fov=new_fov)


def _undo(fov: TrackedFOV, input_root: Path, registry_cfg,
          selected: Optional[str]) -> GestureResult:
    undone = undo_last(fov, input_root)
    if undone is None:
        return _refused("nothing to undo", selected_gcid=selected)
    new_fov, report = _apply_and_reload(fov.fov_id, input_root, registry_cfg)
    return GestureResult(ok=True, message=_report_msg(report, undone),
                         selected_gcid=selected,
                         warnings=list(getattr(report, "warnings", []) or []),
                         fov=new_fov)


def undo_last(fov: TrackedFOV, input_root: Path) -> Optional[str]:
    """Drop the most-recently-written op across every log this FOV owns.

    "Most recent" is decided by comparing the last line's timestamp across
    every session's centroid log *and* the FOV's one match log — undo has to
    span all of them, since a gesture could have been a link just as easily as
    a centroid edit. Two ops in the same clock tick is a coin flip between
    them; both are valid things to undo.
    """
    from roigbiv.pipeline.centroid_edits import load_centroid_ops, write_centroid_ops
    from roigbiv.registry.cell_edits import load_match_ops, write_match_ops

    candidates: list = []
    for session in fov.sessions:
        if session.output_dir is None:
            continue
        ops = load_centroid_ops(session.output_dir)
        if ops:
            candidates.append((ops[-1].ts, "centroid", session.output_dir, ops))
    match_ops = load_match_ops(input_root, fov.fov_id)
    if match_ops:
        candidates.append((match_ops[-1].ts, "match", None, match_ops))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    _ts, kind, output_dir, ops = candidates[-1]
    if kind == "centroid":
        write_centroid_ops(output_dir, ops[:-1])
        return f"undid the last centroid edit in {Path(output_dir).name}"
    write_match_ops(input_root, fov.fov_id, ops[:-1])
    return "undid the last cross-session link edit"
