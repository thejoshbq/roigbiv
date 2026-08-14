"""Track page — session ordering UI and its server-side callbacks.

The drag interaction itself is browser-only (``assets/reorder.js``) and not
unit-testable here; what is testable is everything on the Python side of it —
the row rendering that tells a human which dates to distrust, the order the
rows come out in, and the persistence the drag ultimately triggers.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from roigbiv.ui.pages import tracking as track

PRISM = [
    "052126_DS-Prism-3_VI15_D2_FOV2_pre-005",
    "052126_DS-Prism-3_VI15_D2_FOV2_beh-006",
    "052126_DS-Prism-3_VI15_D2_FOV2_post-007",
]


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _walk(child)


def _ids(root):
    return {getattr(c, "id", None) for c in _walk(root)}


def _text(root) -> str:
    """All string content in a component tree, flattened."""
    out = []
    for c in _walk(root):
        children = getattr(c, "children", None)
        if isinstance(children, str):
            out.append(children)
        elif isinstance(children, (list, tuple)):
            out.extend(x for x in children if isinstance(x, str))
    return " ".join(out)


def _make_workspace(root: Path, stems, *, centroids=True):
    from roigbiv.pipeline.workspace import resolve_workspace

    for stem in stems:
        tifffile.imwrite(root / f"{stem}_mc.tif",
                         np.zeros((2, 16, 16), dtype=np.uint16))
        out_dir = root / "output" / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        if centroids:
            out_dir.joinpath("centroids.json").write_text(json.dumps({
                "stem": stem, "schema": 4,
                "centroids": [{"label_id": 0, "y": 5.0, "x": 5.0, "npix": 10}],
            }))
    return resolve_workspace(root)


class _FakeState:
    def __init__(self, workspace=None):
        self.workspace = workspace
        self.registry_config = None


def test_layout_has_the_reorder_sink_and_action_buttons():
    """The drag script publishes into the sink; it must exist in the tree."""
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    class _IdleRunner:
        def snapshot(self):
            return TrackingSnapshot(active=False)

    with patch.object(track, "get_app_state", return_value=_FakeState()), \
            patch.object(track, "get_tracking_runner", return_value=_IdleRunner()):
        ids = _ids(track.layout())

    assert track.ORDER_SINK_ID in ids
    assert "roigbiv-track-save-btn" in ids
    assert "roigbiv-track-run-btn" in ids
    assert "roigbiv-track-reset-btn" in ids


def test_without_a_workspace_the_list_asks_for_a_scan():
    with patch.object(track, "get_app_state", return_value=_FakeState()):
        assert "Scan a workspace" in _text(track._session_list())


def test_rows_render_one_per_session_with_drag_metadata():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _make_workspace(root, PRISM)
        with patch.object(track, "get_app_state",
                          return_value=_FakeState(workspace)):
            listing = track._session_list()

    rows = [c for c in _walk(listing)
            if getattr(c, "draggable", None) == "true"]
    assert len(rows) == 3
    stems = {r.__dict__["data-track-stem"] for r in rows}
    assert stems == set(PRISM)


def test_ambiguous_dates_are_badged_for_the_human():
    """060126 is valid as both MMDDYY and YYMMDD — the human has to decide."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _make_workspace(
            root, ["060126_DS-Prism-3_VI15_D3_FOV2_beh-007"])
        with patch.object(track, "get_app_state",
                          return_value=_FakeState(workspace)):
            listing = track._session_list()

    assert "ambiguous date" in _text(listing)


def test_undatable_stems_are_badged_too():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _make_workspace(root, ["999999_DS-Prism-3_FOV1_beh-001"])
        with patch.object(track, "get_app_state",
                          return_value=_FakeState(workspace)):
            listing = track._session_list()

    assert "no date" in _text(listing)


def test_missing_centroids_are_called_out_on_the_row():
    """Tracking will skip these, so say so before the user hits Run."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _make_workspace(root, PRISM[:1], centroids=False)
        with patch.object(track, "get_app_state",
                          return_value=_FakeState(workspace)):
            listing = track._session_list()

    assert "run discovery first" in _text(listing)


def test_centroid_counts_are_shown():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _make_workspace(root, PRISM[:1])
        with patch.object(track, "get_app_state",
                          return_value=_FakeState(workspace)):
            listing = track._session_list()

    assert "1 centroids" in _text(listing)


def test_saved_order_is_reflected_in_row_order():
    from roigbiv.pipeline.session_order import propose_order, reorder, save_order

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _make_workspace(root, PRISM)
        wanted = [PRISM[2], PRISM[0], PRISM[1]]
        save_order(root, reorder(propose_order(PRISM), wanted))

        with patch.object(track, "get_app_state",
                          return_value=_FakeState(workspace)):
            listing = track._session_list()

    rows = [c for c in _walk(listing) if getattr(c, "draggable", None) == "true"]
    assert [r.__dict__["data-track-stem"] for r in rows] == wanted
    # A confirmed order is shown as such rather than left ambiguous.
    assert "confirmed" in _text(listing)


def test_entries_are_numbered_from_one_for_display():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _make_workspace(root, PRISM)
        with patch.object(track, "get_app_state",
                          return_value=_FakeState(workspace)):
            listing = track._session_list()

    text = _text(listing)
    assert "1" in text and "3" in text


def test_idle_status_reads_as_idle():
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    assert "Idle" in _text(track._status(TrackingSnapshot(active=False)))


def test_active_status_reads_as_running():
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(active=True, started_at=1.0)
    assert "in progress" in _text(track._status(snap))


def test_finished_status_counts_outcomes():
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(
        active=False, started_at=1.0, completed_at=2.0,
        results=[
            {"registry": {"decision": "auto_match"}},
            {"skipped": "no centroids.json"},
            {"error": "boom"},
        ],
    )
    text = _text(track._status(snap))
    assert "1 session(s) tracked" in text
    assert "1 skipped" in text
    assert "1 failed" in text


def test_anomaly_panel_is_empty_before_anything_is_tracked():
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    with patch.object(track, "get_app_state", return_value=_FakeState()):
        panel = track._anomaly_panel(TrackingSnapshot(active=False))
    assert "No tracked sessions yet" in _text(panel)


def test_anomaly_panel_falls_back_to_the_registry():
    """A workspace tracked from the CLI, or before this browser session,
    still has a report — it lives in the observation rows, not in run state."""
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    stored = {"fov-9": {
        "counts": {"n_sessions": 3, "n_cells": 4, "n_complete": 2,
                   "late_arrival": 1, "dropout": 1, "intermittent": 0},
        "ordering_is_confirmed": True,
        "cells": [{"global_cell_id": "beadfeed-1", "present": [True, True, False],
                   "anomalies": ["dropout"], "first_seen": 0, "last_seen": 1}],
        "sessions": [],
    }}
    with patch.object(track, "_stored_anomalies", return_value=stored):
        text = _text(track._anomaly_panel(TrackingSnapshot(active=False)))

    assert "4 cells over 3 sessions" in text
    assert "beadfeed" in text
    # The provenance matters: these are not this run's numbers.
    assert "From the registry" in text


def test_a_run_snapshot_wins_over_the_stored_report():
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(active=False, anomalies={"fov-1": {
        "counts": {"n_sessions": 2, "n_cells": 7, "n_complete": 7,
                   "late_arrival": 0, "dropout": 0, "intermittent": 0},
        "ordering_is_confirmed": True, "cells": [], "sessions": [],
    }})
    with patch.object(track, "_stored_anomalies") as stored:
        text = _text(track._anomaly_panel(snap))

    stored.assert_not_called()
    assert "7 cells over 2 sessions" in text
    assert "From the registry" not in text


def test_an_active_run_does_not_query_the_registry():
    """Mid-run the registry is half-written; polling it every tick would also
    hammer the store for a report that is about to be handed over anyway."""
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    with patch.object(track, "_stored_anomalies") as stored:
        text = _text(track._anomaly_panel(TrackingSnapshot(active=True)))

    stored.assert_not_called()
    assert "in progress" in text


def test_an_unreadable_registry_reports_instead_of_blanking():
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    with patch.object(track, "_stored_anomalies",
                      side_effect=RuntimeError("no such table: session")):
        text = _text(track._anomaly_panel(TrackingSnapshot(active=False)))

    assert "no such table" in text


# ── per-session registration outcome ───────────────────────────────────────


def _result(**kw) -> dict:
    base = {
        "stem": PRISM[0], "sequence_index": 0, "output_dir": "/out",
        "n_centroids": 12, "n_overlapping_pairs": 0, "skipped": None,
        "error": None, "registry": {}, "decision": None, "posterior": None,
        "n_matched": None, "n_new": None, "n_missing": None,
    }
    base.update(kw)
    return base


def test_results_table_is_absent_before_a_run():
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    assert _text(track._results_table(TrackingSnapshot(active=False))) == ""


def test_results_table_shows_each_session_decision_and_counts():
    """The aggregate 'N tracked' cannot say which session matched what."""
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(active=False, completed_at=1.0, results=[
        _result(stem=PRISM[0], sequence_index=0, decision="new_fov", n_new=12),
        _result(stem=PRISM[1], sequence_index=1, decision="auto_match",
                posterior=0.938, n_centroids=8, n_matched=8, n_new=0,
                n_missing=4),
    ])
    text = _text(track._results_table(snap))

    assert PRISM[1] in text
    assert "auto_match" in text and "new_fov" in text
    assert "0.94" in text          # posterior, not the raw float
    assert "8" in text and "4" in text


def test_results_table_reports_skips_and_failures_in_place():
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(active=False, completed_at=1.0, results=[
        _result(skipped="no centroid discovery output"),
        _result(stem=PRISM[1], sequence_index=1, error="OSError: disk full"),
    ])
    text = _text(track._results_table(snap))

    assert "no centroid discovery output" in text
    assert "disk full" in text


def test_overlapping_stamps_are_surfaced_next_to_the_cell_count():
    """Crowded disks degrade the embeddings; the count makes that visible."""
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(active=False, completed_at=1.0, results=[
        _result(n_centroids=12, n_overlapping_pairs=3, decision="new_fov"),
    ])
    assert "3 overlap" in _text(track._results_table(snap))


def test_result_rows_are_numbered_to_match_the_session_list():
    """Both halves of the page must agree; the list numbers from one."""
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(active=False, completed_at=1.0, results=[
        _result(sequence_index=0, decision="new_fov"),
        _result(stem=PRISM[1], sequence_index=1, decision="auto_match"),
    ])
    cells = _text(track._results_table(snap)).split()
    assert "1" in cells and "2" in cells


def test_a_review_decision_renders_without_match_counts():
    """The review branch writes no session row, so m/n/x are simply absent."""
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(active=False, completed_at=1.0, results=[
        _result(decision="review", posterior=0.61),
    ])
    text = _text(track._results_table(snap))
    assert "review" in text and "0.61" in text


def test_anomaly_panel_renders_counts_and_per_cell_timelines():
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(active=False, anomalies={
        "fov-1": {
            "counts": {"n_sessions": 3, "n_cells": 5, "n_complete": 3,
                       "late_arrival": 1, "dropout": 1, "intermittent": 0},
            "ordering_is_confirmed": True,
            "cells": [
                {"global_cell_id": "abcdef12-0000", "present": [False, True, True],
                 "anomalies": ["late_arrival"], "first_seen": 1, "last_seen": 2},
            ],
            "sessions": [],
        }
    })
    text = _text(track._anomaly_panel(snap))

    assert "5 cells over 3 sessions" in text
    assert "late_arrival" in text
    assert "abcdef12" in text


def test_unconfirmed_ordering_is_warned_about_in_the_panel():
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(active=False, anomalies={
        "fov-1": {
            "counts": {"n_sessions": 2, "n_cells": 1, "n_complete": 1,
                       "late_arrival": 0, "dropout": 0, "intermittent": 0},
            "ordering_is_confirmed": False,
            "cells": [],
            "sessions": [],
        }
    })
    assert "not human-ordered" in _text(track._anomaly_panel(snap))


def test_saving_an_order_persists_it():
    """The end of the drag interaction: sink value -> session_order.json."""
    from roigbiv.pipeline.session_order import load_order

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _make_workspace(root, PRISM)
        wanted = [PRISM[2], PRISM[1], PRISM[0]]

        captured = {}

        class _App:
            def callback(self, *a, **k):
                def deco(fn):
                    captured[fn.__name__] = fn
                    return fn
                return deco

            def clientside_callback(self, *a, **k):
                # The merged page also wires the contact sheet's browser-side
                # handoffs; nothing to capture, but the fake app has to accept
                # them or registration stops before the server callbacks.
                return None

        track.register_callbacks(_App())
        with patch.object(track, "get_app_state",
                          return_value=_FakeState(workspace)):
            captured["_on_save"](1, json.dumps(wanted))

        assert [e.stem for e in load_order(root)] == wanted
        assert all(e.locked for e in load_order(root))


def test_a_crashed_matcher_is_flagged_above_the_results_table():
    """new_fov from a crash looks identical to new_fov from a new FOV."""
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(active=False, completed_at=1.0, results=[
        _result(decision="new_fov", match_errors=[
            {"candidate_fov_id": "c", "error": "ValueError: zero-size array"}]),
        _result(stem=PRISM[1], sequence_index=1, decision="new_fov",
                match_errors=[{"candidate_fov_id": "c",
                               "error": "ValueError: zero-size array"}]),
    ])
    text = _text(track._results_table(snap))

    assert "matching failed" in text.lower()
    assert "zero-size array" in text
    # Both sessions hit the same upstream bug; report it once.
    assert text.count("zero-size array") == 1


def test_a_clean_run_shows_no_failure_banner():
    from roigbiv.ui.services.tracking_runner import TrackingSnapshot

    snap = TrackingSnapshot(active=False, completed_at=1.0, results=[
        _result(decision="auto_match", posterior=0.95, n_matched=8),
    ])
    assert "matching failed" not in _text(track._results_table(snap)).lower()
