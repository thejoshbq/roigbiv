"""Gesture semantics for the cross-session cell editor.

These are the decisions the browser is *not* trusted to make: what a drop
coordinate means, when an add should also link, when a link is really an
unlink. The pointer handling that produces the gestures lives in
``assets/cells_sheet.js``; everything it can get wrong about *meaning* is
decided here.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from roigbiv.ui.services import cell_edit_ops as ops
from roigbiv.ui.services.cell_edit_ops import Gesture, apply_gesture
from roigbiv.ui.services.loaders import ROIRender
from roigbiv.ui.services.tracked_cells import TrackedCell, TrackedFOV, TrackedSession

STEMS = ["fov_pre-005", "fov_beh-006", "fov_post-007"]
INPUT_ROOT = Path("/fake/root")


# ── fixtures ───────────────────────────────────────────────────────────────


class _StubReport:
    warnings: list = []


def _roi(label_id, cy, cx, status, gcid):
    ys = [cy - 5, cy - 5, cy + 5, cy + 5]
    xs = [cx - 5, cx + 5, cx + 5, cx - 5]
    return ROIRender(
        label_id=label_id, source_stage=1, gate_outcome="accept",
        activity_type=None, area=78, centroid_yx=(float(cy), float(cx)),
        contours=[(ys, xs)], global_cell_id=gcid, match_status=status,
    )


def _fov() -> TrackedFOV:
    """Two cells over three sessions: A throughout, B dropping out at the end."""
    layout = [
        [("A", 1, 20, 20, "new"), ("B", 2, 20, 60, "new")],
        [("A", 1, 22, 21, "matched"), ("B", 2, 21, 61, "matched")],
        [("A", 1, 23, 22, "matched"), ("B", -2, 21, 61, "lost")],
    ]
    sessions = []
    for i, entries in enumerate(layout):
        sessions.append(TrackedSession(
            session_id=f"s{i}", stem=STEMS[i], session_date=None,
            sequence_index=i, output_dir=Path(f"/fake/{STEMS[i]}"),
            mean_M=np.zeros((80, 80), dtype=np.float32),
            rois=[_roi(lid, cy, cx, status, f"gcid-{name}")
                  for name, lid, cy, cx, status in entries],
            n_matched=2 if i else 0, n_new=0 if i else 2,
            n_missing=1 if i == 2 else 0,
        ))
    cells = [
        TrackedCell(global_cell_id="gcid-A", index=1, present=[True] * 3,
                    local_label_ids=[1, 1, 1],
                    centroids=[(20.0, 20.0), (22.0, 21.0), (23.0, 22.0)],
                    anomalies=[]),
        TrackedCell(global_cell_id="gcid-B", index=2, present=[True, True, False],
                    local_label_ids=[2, 2, None],
                    centroids=[(20.0, 60.0), (21.0, 61.0), None],
                    anomalies=["dropout"]),
    ]
    return TrackedFOV(fov_id="fov-1", animal_id="DS-Prism-3", region="DS-Prism",
                      sessions=sessions, cells=cells)


# ── payload validation ─────────────────────────────────────────────────────


def test_an_unknown_gesture_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown gesture kind"):
        Gesture.from_payload({"kind": "teleport"})


@pytest.mark.parametrize("payload,missing", [
    ({"kind": "delete", "label": 1}, "stem"),
    ({"kind": "delete", "stem": "s"}, "label"),
    ({"kind": "move", "stem": "s", "label": 1}, "y and x"),
    ({"kind": "add", "stem": "s"}, "y and x"),
])
def test_each_gesture_requires_the_fields_it_acts_on(payload, missing):
    with pytest.raises(ValueError, match=missing):
        Gesture.from_payload(payload)


def test_undo_needs_nothing_at_all():
    assert Gesture.from_payload({"kind": "undo"}).kind == "undo"


def test_a_non_numeric_coordinate_is_rejected_rather_than_silently_zeroed():
    with pytest.raises(ValueError):
        Gesture.from_payload({"kind": "add", "stem": "s", "y": "here", "x": 1})


# ── move — the regression the rewrite exists for ───────────────────────────


def test_a_move_writes_the_exact_dropped_coordinate():
    """The Plotly page resolved a move's destination through a
    nearest-centroid snap, so any nudge shorter than the stamp radius resolved
    back to the centroid being moved and wrote a no-op. A two-pixel correction
    onto the real soma is the common case; it has to land where it was put."""
    fov = _fov()
    origin_y, origin_x = fov.sessions[0].rois[0].centroid_yx   # label 1 @ (20, 20)
    nudged_y, nudged_x = origin_y + 2.0, origin_x - 1.0

    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_move") as mock_move:
        result = apply_gesture(
            fov, Gesture(kind="move", stem=STEMS[0], label=1,
                         y=nudged_y, x=nudged_x),
            input_root=INPUT_ROOT, registry_cfg=None)

    assert result.ok
    assert mock_move.call_args[0][1:] == (1, nudged_y, nudged_x)


def test_moving_a_label_that_is_not_in_this_session_is_refused():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_do_move") as mock_move:
        result = apply_gesture(fov, Gesture(kind="move", stem=STEMS[0],
                                            label=77, y=1.0, x=1.0),
                               input_root=INPUT_ROOT, registry_cfg=None)
    mock_move.assert_not_called()
    assert not result.ok
    assert "no centroid" in result.message


def test_a_ghost_cannot_be_moved():
    """Session 2 draws B as a ghost — a negative label with no footprint."""
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_do_move") as mock_move:
        result = apply_gesture(fov, Gesture(kind="move", stem=STEMS[2],
                                            label=-2, y=5.0, x=5.0),
                               input_root=INPUT_ROOT, registry_cfg=None)
    mock_move.assert_not_called()
    assert not result.ok


# ── delete ─────────────────────────────────────────────────────────────────


def test_deleting_writes_the_op_and_clears_a_matching_selection():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_delete") as mock_delete:
        result = apply_gesture(fov, Gesture(kind="delete", stem=STEMS[0],
                                            label=1, selected_gcid="gcid-A"),
                               input_root=INPUT_ROOT, registry_cfg=None)
    assert mock_delete.call_args[0][1] == 1
    assert result.selected_gcid is None
    assert result.message == "deleted a centroid"


def test_deleting_leaves_an_unrelated_selection_alone():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_delete"):
        result = apply_gesture(fov, Gesture(kind="delete", stem=STEMS[0],
                                            label=1, selected_gcid="gcid-B"),
                               input_root=INPUT_ROOT, registry_cfg=None)
    assert result.selected_gcid == "gcid-B"


def test_deleting_something_that_is_not_there_is_refused():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_do_delete") as mock_delete:
        result = apply_gesture(fov, Gesture(kind="delete", stem=STEMS[0],
                                            label=99),
                               input_root=INPUT_ROOT, registry_cfg=None)
    mock_delete.assert_not_called()
    assert not result.ok


# ── add ────────────────────────────────────────────────────────────────────


def test_adding_with_nothing_selected_is_a_plain_add():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_add", return_value=99) as mock_add, \
            patch.object(ops, "_do_link") as mock_link:
        result = apply_gesture(fov, Gesture(kind="add", stem=STEMS[0],
                                            y=5.0, x=5.0),
                               input_root=INPUT_ROOT, registry_cfg=None)
    mock_add.assert_called_once()
    mock_link.assert_not_called()
    assert result.message == "added a centroid"


def test_place_here_composes_add_and_link_in_one_gesture():
    """A selected cell missing from this session: one click both adds the
    centroid and links it to that cell — the repair this data needs most, and
    three separate gestures if composed by hand."""
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_add", return_value=99), \
            patch.object(ops, "_do_link") as mock_link:
        result = apply_gesture(
            fov, Gesture(kind="add", stem=STEMS[2], y=21.0, x=61.0,
                         selected_gcid="gcid-B"),   # B is absent from session 2
            input_root=INPUT_ROOT, registry_cfg=None)

    mock_link.assert_called_once()
    assert (STEMS[2], 99) in mock_link.call_args[0][2]
    assert result.message == "added and linked a centroid"


def test_no_place_here_when_the_selected_cell_is_already_in_this_session():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_add", return_value=99), \
            patch.object(ops, "_do_link") as mock_link:
        apply_gesture(fov, Gesture(kind="add", stem=STEMS[0], y=5.0, x=5.0,
                                   selected_gcid="gcid-A"),
                      input_root=INPUT_ROOT, registry_cfg=None)
    mock_link.assert_not_called()


def test_place_here_keeps_the_repaired_cell_selected_even_if_the_id_is_reminted():
    """apply_match_ops can mint a fresh global id for the merged group, and the
    drawer must stay on the cell that was just repaired rather than blanking."""
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_add", return_value=99), \
            patch.object(ops, "_do_link"):
        result = apply_gesture(
            fov, Gesture(kind="add", stem=STEMS[2], y=21.0, x=61.0,
                         selected_gcid="gcid-B"),
            input_root=INPUT_ROOT, registry_cfg=None)
    # Label 99 does not exist in the stubbed reload, so the lookup misses.
    assert result.selected_gcid == "gcid-B"


# ── link / unlink ──────────────────────────────────────────────────────────


def test_shift_clicking_a_different_cell_links_the_two():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_link") as mock_link, \
            patch.object(ops, "_do_unlink") as mock_unlink:
        result = apply_gesture(
            fov, Gesture(kind="link", stem=STEMS[1], label=2,   # gcid-B
                         selected_gcid="gcid-A"),
            input_root=INPUT_ROOT, registry_cfg=None)
    mock_link.assert_called_once()
    mock_unlink.assert_not_called()
    assert mock_link.call_args[0][2] == [(STEMS[0], 1), (STEMS[1], 2)]
    assert result.message == "linked"


def test_shift_clicking_a_member_of_the_selected_cell_unlinks_it():
    """Session 1's label 1 is already gcid-A, same as the selection — reading
    that as a link would merge a cell with itself, so it means "this one does
    not belong"."""
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_link") as mock_link, \
            patch.object(ops, "_do_unlink") as mock_unlink:
        result = apply_gesture(
            fov, Gesture(kind="link", stem=STEMS[1], label=1,
                         selected_gcid="gcid-A"),
            input_root=INPUT_ROOT, registry_cfg=None)
    mock_unlink.assert_called_once()
    mock_link.assert_not_called()
    assert mock_unlink.call_args[0][2] == (STEMS[1], 1)
    assert result.message == "unlinked"


def test_linking_with_no_selection_says_what_to_do_instead_of_failing_silently():
    """The old two-click pickup gave no feedback at all on the first click,
    which is most of why link read as broken."""
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_do_link") as mock_link:
        result = apply_gesture(fov, Gesture(kind="link", stem=STEMS[0], label=1),
                               input_root=INPUT_ROOT, registry_cfg=None)
    mock_link.assert_not_called()
    assert not result.ok
    assert "select a cell first" in result.message


def test_a_link_the_replay_refuses_is_reported_as_a_refusal_not_a_success():
    """apply_match_ops rejects a link that would put two members of one session
    in one cell. Reporting "linked" over the top of that would be a claim the
    sheet visibly contradicts a moment later."""
    fov = _fov()

    class _Rejecting:
        warnings = ["link: op the-op-id rejected — session 'x' would end up "
                    "with two members in one cell; unlink one first"]

    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _Rejecting())), \
            patch.object(ops, "_do_link", return_value="the-op-id"), \
            patch.object(ops, "_rollback_match_op") as mock_rollback:
        result = apply_gesture(
            fov, Gesture(kind="link", stem=STEMS[1], label=2,
                         selected_gcid="gcid-A"),
            input_root=INPUT_ROOT, registry_cfg=None)

    assert not result.ok
    assert "two members in one cell" in result.message
    mock_rollback.assert_called_once()
    assert mock_rollback.call_args[0][2] == "the-op-id"


def test_a_rejected_op_is_rolled_off_the_log_rather_than_left_to_repeat():
    """A rejected op is inert but not silent: left in place it re-emits its
    warning on every later replay, so one impossible link would append its
    complaint to every message the page shows from then on."""
    from roigbiv.registry.cell_edits import MatchOp, append_match_op, load_match_ops

    root = Path(__import__("tempfile").mkdtemp())
    keep = MatchOp.link("fov-1", [("a", 1), ("b", 2)])
    doomed = MatchOp.link("fov-1", [("c", 3), ("d", 4)])
    append_match_op(root, keep)
    append_match_op(root, doomed)

    ops._rollback_match_op("fov-1", root, doomed.id)

    remaining = [o.id for o in load_match_ops(root, "fov-1")]
    assert remaining == [keep.id]


def test_a_refused_link_still_returns_state_because_the_log_changed():
    """Unlike a validation refusal, this one wrote and then unwrote an op, so
    the browser has to repaint from the reconciled state."""
    fov = _fov()

    class _Rejecting:
        warnings = ["link: op the-op-id rejected — nope"]

    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _Rejecting())), \
            patch.object(ops, "_do_link", return_value="the-op-id"), \
            patch.object(ops, "_rollback_match_op"):
        result = apply_gesture(
            fov, Gesture(kind="link", stem=STEMS[1], label=2,
                         selected_gcid="gcid-A"),
            input_root=INPUT_ROOT, registry_cfg=None)
    assert result.fov is fov
    assert result.selected_gcid == "gcid-A"


def test_place_here_keeps_the_centroid_when_only_the_adoption_is_refused():
    """The add and the link are one gesture but two ops; a refused link must
    not take the placed centroid down with it."""
    fov = _fov()

    class _Rejecting:
        warnings = ["link: op the-op-id rejected — nope"]

    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _Rejecting())), \
            patch.object(ops, "_do_add", return_value=99) as mock_add, \
            patch.object(ops, "_do_link", return_value="the-op-id"), \
            patch.object(ops, "_rollback_match_op") as mock_rollback:
        result = apply_gesture(
            fov, Gesture(kind="add", stem=STEMS[2], y=21.0, x=61.0,
                         selected_gcid="gcid-B"),
            input_root=INPUT_ROOT, registry_cfg=None)

    mock_add.assert_called_once()
    mock_rollback.assert_called_once()
    assert result.ok                      # the centroid is placed
    assert "could not link it" in result.message


# ── confirm — ctrl-click a ghost ───────────────────────────────────────────


def _ghost_at(fov: TrackedFOV) -> tuple:
    """B's ghost in the last session: (stem, gcid, y, x)."""
    ghost = next(r for r in fov.sessions[2].rois if r.label_id < 0)
    return STEMS[2], ghost.global_cell_id, *ghost.centroid_yx


def test_confirming_a_ghost_over_bare_background_places_and_links_it():
    """No prior selection: the ghost names its own cell, which is the click
    this gesture removes versus select-then-shift-click."""
    fov = _fov()
    stem, gcid, y, x = _ghost_at(fov)
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_add", return_value=9) as mock_add, \
            patch.object(ops, "_do_link", return_value="op-1") as mock_link:
        result = apply_gesture(
            fov, Gesture(kind="confirm", stem=stem, y=y, x=x,
                         selected_gcid=gcid),
            input_root=INPUT_ROOT, registry_cfg=None)

    assert result.ok
    assert mock_add.call_args[0][1:] == (y, x)
    assert (stem, 9) in mock_link.call_args[0][2]


def test_confirming_says_the_position_came_from_another_frame():
    """Sessions are not co-registered, so the ghost's coordinate carries
    whatever drift exists between the two frames. Silently placing a centroid
    at a borrowed position without saying so would be the lie."""
    fov = _fov()
    stem, gcid, y, x = _ghost_at(fov)
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_add", return_value=9), \
            patch.object(ops, "_do_link", return_value="op-1"):
        result = apply_gesture(
            fov, Gesture(kind="confirm", stem=stem, y=y, x=x,
                         selected_gcid=gcid),
            input_root=INPUT_ROOT, registry_cfg=None)

    assert "check the position" in result.message


def test_confirming_over_an_existing_outline_links_it_instead_of_duplicating():
    """A detection is already there — adopting it beats stacking a second
    centroid on top of it, and it is immune to the cross-session drift."""
    fov = _fov()
    # An unmatched outline in the last session, right where B's ghost is drawn.
    fov.sessions[2].rois.append(_roi(7, 21, 61, "new", "gcid-C"))
    fov.cells.append(TrackedCell(
        global_cell_id="gcid-C", index=3, present=[False, False, True],
        local_label_ids=[None, None, 7], centroids=[None, None, (21.0, 61.0)],
        anomalies=[]))
    stem, gcid, y, x = _ghost_at(fov)

    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())), \
            patch.object(ops, "_do_add") as mock_add, \
            patch.object(ops, "_do_link", return_value="op-1") as mock_link:
        result = apply_gesture(
            fov, Gesture(kind="confirm", stem=stem, y=y, x=x,
                         selected_gcid=gcid),
            input_root=INPUT_ROOT, registry_cfg=None)

    mock_add.assert_not_called()
    assert (stem, 7) in mock_link.call_args[0][2]
    assert result.ok
    assert "an outline was already there" in result.message


def test_confirming_onto_a_multi_session_cell_is_refused():
    """Adopting it would merge two tracked cells on one click. That is a claim
    big enough to have to be made deliberately, so it points at shift-click."""
    fov = _fov()
    # A's outline in the last session, moved under B's ghost. A is seen in all
    # three sessions, so this would be a merge, not an adoption.
    fov.sessions[2].rois[0] = _roi(1, 21, 61, "matched", "gcid-A")
    stem, gcid, y, x = _ghost_at(fov)

    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_do_add") as mock_add, \
            patch.object(ops, "_do_link") as mock_link:
        result = apply_gesture(
            fov, Gesture(kind="confirm", stem=stem, y=y, x=x,
                         selected_gcid=gcid),
            input_root=INPUT_ROOT, registry_cfg=None)

    mock_add.assert_not_called()
    mock_link.assert_not_called()
    assert not result.ok
    assert "already cell #1" in result.message
    assert "shift-click" in result.message


def test_confirming_a_cell_that_is_already_here_is_refused():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_do_add") as mock_add:
        result = apply_gesture(
            fov, Gesture(kind="confirm", stem=STEMS[0], y=20.0, x=20.0,
                         selected_gcid="gcid-A"),
            input_root=INPUT_ROOT, registry_cfg=None)
    mock_add.assert_not_called()
    assert not result.ok
    assert "already has a centroid" in result.message


def test_confirming_without_a_cell_says_what_to_do():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_do_add") as mock_add:
        result = apply_gesture(fov, Gesture(kind="confirm", stem=STEMS[2],
                                            y=21.0, x=61.0),
                               input_root=INPUT_ROOT, registry_cfg=None)
    mock_add.assert_not_called()
    assert not result.ok


@pytest.mark.parametrize("y,x,expected", [
    (20.0, 20.0, True),     # dead centre of label 1's box
    (25.0, 20.0, False),    # below it
    (20.0, 26.0, False),    # beside it
])
def test_containment_is_the_test_not_proximity(y, x, expected):
    """Sessions are not co-registered, so a distance threshold would be a guess
    about drift this codebase cannot measure. Inside-the-outline is a claim the
    geometry supports."""
    fov = _fov()
    hit = ops._roi_containing(fov, 0, y, x)
    assert (hit is not None and hit.label_id == 1) is expected


def test_linking_a_ghost_is_refused():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_do_link") as mock_link:
        result = apply_gesture(fov, Gesture(kind="link", stem=STEMS[2],
                                            label=-2, selected_gcid="gcid-A"),
                               input_root=INPUT_ROOT, registry_cfg=None)
    mock_link.assert_not_called()
    assert not result.ok


# ── guards ─────────────────────────────────────────────────────────────────


def test_every_gesture_is_blocked_while_tracking_runs():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=True), \
            patch.object(ops, "_do_delete") as mock_delete:
        result = apply_gesture(fov, Gesture(kind="delete", stem=STEMS[0],
                                            label=1),
                               input_root=INPUT_ROOT, registry_cfg=None)
    mock_delete.assert_not_called()
    assert result.status == 409
    assert "tracking is running" in result.message


def test_a_gesture_naming_an_unknown_session_is_refused():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False):
        result = apply_gesture(fov, Gesture(kind="delete", stem="not_a_session",
                                            label=1),
                               input_root=INPUT_ROOT, registry_cfg=None)
    assert result.status == 400
    assert not result.ok


def test_a_refusal_carries_no_fresh_state_so_the_sheet_is_left_alone():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=True):
        result = apply_gesture(fov, Gesture(kind="delete", stem=STEMS[0],
                                            label=1),
                               input_root=INPUT_ROOT, registry_cfg=None)
    assert result.fov is None


def test_replay_warnings_are_surfaced_next_to_the_verb():
    class _Noisy:
        warnings = ["label 3 was already gone"]

    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "_apply_and_reload", return_value=(fov, _Noisy())), \
            patch.object(ops, "_do_delete"):
        result = apply_gesture(fov, Gesture(kind="delete", stem=STEMS[0],
                                            label=1),
                               input_root=INPUT_ROOT, registry_cfg=None)
    assert "deleted a centroid — label 3 was already gone" == result.message
    assert result.warnings == ["label 3 was already gone"]


# ── undo ───────────────────────────────────────────────────────────────────


def test_undo_with_an_empty_log_says_so():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "undo_last", return_value=None):
        result = apply_gesture(fov, Gesture(kind="undo"),
                               input_root=INPUT_ROOT, registry_cfg=None)
    assert not result.ok
    assert result.message == "nothing to undo"


def test_undo_keeps_the_current_selection():
    fov = _fov()
    with patch.object(ops, "_tracking_is_active", return_value=False), \
            patch.object(ops, "undo_last", return_value="undid the last edit"), \
            patch.object(ops, "_apply_and_reload",
                         return_value=(fov, _StubReport())):
        result = apply_gesture(fov, Gesture(kind="undo", selected_gcid="gcid-A"),
                               input_root=INPUT_ROOT, registry_cfg=None)
    assert result.selected_gcid == "gcid-A"


# ── end to end ─────────────────────────────────────────────────────────────


def test_an_add_writes_a_real_centroid_and_a_real_observation(tmp_path):
    """The mocked tests above prove the branching; this one proves the real
    pieces — the op log, apply_tracking_edits, the store — wire together the
    way the mocks assumed."""
    import json
    import uuid
    from datetime import date, datetime, timezone

    import tifffile

    from roigbiv.pipeline.centroid_masks import write_merged_masks
    from roigbiv.pipeline.types import PipelineConfig
    from roigbiv.registry.config import RegistryConfig
    from roigbiv.registry.store.base import FOVRecord, SessionRecord
    from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore
    from roigbiv.ui.services.tracked_cells import load_tracked_fov

    cfg = RegistryConfig(
        dsn=f"sqlite:///{tmp_path / 'registry.db'}", blob_backend="local",
        blob_root=tmp_path / "blobs", endpoint=None, api_key=None,
    )
    store = SQLAlchemyStore(dsn=cfg.dsn)
    store.ensure_schema()

    fov_id = str(uuid.uuid4())
    store.insert_fov(FOVRecord(
        fov_id=fov_id, fingerprint_hash="a" * 64, animal_id="X", region="Y",
        mean_m_uri="file:///m", centroid_table_uri="file:///c",
        created_at=datetime.now(timezone.utc)))

    out_dir = tmp_path / "sess-a"
    (out_dir / "summary").mkdir(parents=True)
    tifffile.imwrite(str(out_dir / "summary" / "mean_M.tif"),
                     np.zeros((64, 64), dtype=np.float32))
    (out_dir / "centroids.json").write_text(json.dumps({
        "stem": "sess-a", "schema": 4,
        "centroids": [{"label_id": 0, "y": 10.0, "x": 10.0, "npix": 50,
                       "cellpose_prob": 0.9}],
    }))
    write_merged_masks(out_dir, PipelineConfig())

    session_id = str(uuid.uuid4())
    store.upsert_session(SessionRecord(
        session_id=session_id, fov_id=fov_id, session_date=date(2026, 1, 1),
        output_dir=str(out_dir), created_at=datetime.now(timezone.utc),
        sequence_index=0))

    fov = load_tracked_fov(fov_id, cfg=cfg)
    with patch.object(ops, "_tracking_is_active", return_value=False):
        result = apply_gesture(
            fov, Gesture(kind="add", stem="sess-a", y=40.0, x=40.0),
            input_root=tmp_path, registry_cfg=cfg)

    assert result.message == "added a centroid"
    assert (out_dir / "corrections" / "centroids.jsonl").exists()

    reloaded = load_tracked_fov(fov_id, cfg=cfg)
    assert len(reloaded.cells) == 2   # the original centroid + the new one
    assert len(store.list_observations_for_session(session_id)) == 2


def test_a_move_survives_replay_and_lands_where_it_was_dropped(tmp_path):
    """The end-to-end counterpart of the snap-back regression: a two-pixel
    nudge has to reach the mask on disk, not just the op log."""
    import json
    import uuid
    from datetime import date, datetime, timezone

    import tifffile

    from roigbiv.pipeline.centroid_masks import (
        load_effective_centroids,
        write_merged_masks,
    )
    from roigbiv.pipeline.types import PipelineConfig
    from roigbiv.registry.config import RegistryConfig
    from roigbiv.registry.store.base import FOVRecord, SessionRecord
    from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore
    from roigbiv.ui.services.tracked_cells import load_tracked_fov

    cfg = RegistryConfig(
        dsn=f"sqlite:///{tmp_path / 'registry.db'}", blob_backend="local",
        blob_root=tmp_path / "blobs", endpoint=None, api_key=None,
    )
    store = SQLAlchemyStore(dsn=cfg.dsn)
    store.ensure_schema()

    fov_id = str(uuid.uuid4())
    store.insert_fov(FOVRecord(
        fov_id=fov_id, fingerprint_hash="b" * 64, animal_id="X", region="Y",
        mean_m_uri="file:///m", centroid_table_uri="file:///c",
        created_at=datetime.now(timezone.utc)))

    out_dir = tmp_path / "sess-b"
    (out_dir / "summary").mkdir(parents=True)
    tifffile.imwrite(str(out_dir / "summary" / "mean_M.tif"),
                     np.zeros((64, 64), dtype=np.float32))
    (out_dir / "centroids.json").write_text(json.dumps({
        "stem": "sess-b", "schema": 4,
        "centroids": [{"label_id": 0, "y": 30.0, "x": 30.0, "npix": 50,
                       "cellpose_prob": 0.9}],
    }))
    write_merged_masks(out_dir, PipelineConfig())

    store.upsert_session(SessionRecord(
        session_id=str(uuid.uuid4()), fov_id=fov_id, session_date=date(2026, 1, 1),
        output_dir=str(out_dir), created_at=datetime.now(timezone.utc),
        sequence_index=0))

    fov = load_tracked_fov(fov_id, cfg=cfg)
    label = fov.sessions[0].rois[0].label_id
    with patch.object(ops, "_tracking_is_active", return_value=False):
        result = apply_gesture(
            fov, Gesture(kind="move", stem="sess-b", label=label,
                         y=32.0, x=29.0),
            input_root=tmp_path, registry_cfg=cfg)

    assert result.ok
    effective, _warnings = load_effective_centroids(out_dir)
    moved = effective[label]
    assert (round(float(moved[0])), round(float(moved[1]))) == (32, 29)
