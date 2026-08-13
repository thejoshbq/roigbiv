"""
Contract tests for the cross-session tracking pass
(:func:`roigbiv.pipeline.workspace.run_tracking`).

``register_or_match`` is mocked — ROICaT is heavy and covered by the registry's
own tests. What matters here is this pass's own contract: it walks the *human*
order, stamps registry-readable masks from centroids, records each session's
timeline position, and skips rather than crashes on an unmarked FOV.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

PRISM = [
    "052126_DS-Prism-3_VI15_D2_FOV2_pre-005",
    "052126_DS-Prism-3_VI15_D2_FOV2_beh-006",
    "052126_DS-Prism-3_VI15_D2_FOV2_post-007",
]


def _workspace(root: Path, stems, *, with_centroids=True):
    """A workspace with one _mc.tif per stem and matching centroid output."""
    from roigbiv.pipeline.workspace import resolve_workspace

    for stem in stems:
        tifffile.imwrite(root / f"{stem}_mc.tif",
                         np.zeros((2, 64, 64), dtype=np.uint16))
        out_dir = root / "output" / stem
        (out_dir / "summary").mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out_dir / "summary" / "mean_M.tif",
                         np.zeros((64, 64), dtype=np.float32))
        if with_centroids:
            (out_dir / "centroids.json").write_text(json.dumps({
                "stem": stem, "schema": 4,
                "centroids": [
                    {"label_id": 0, "y": 10.0, "x": 10.0, "npix": 100,
                     "cellpose_prob": 0.9},
                    {"label_id": 1, "y": 40.0, "x": 40.0, "npix": 100,
                     "cellpose_prob": 0.8},
                ],
            }))
    return resolve_workspace(root)


class _FakeStore:
    def __init__(self):
        self.sequences: dict[str, int] = {}

    def update_session_sequence(self, session_id, sequence_index):
        self.sequences[session_id] = sequence_index

    def list_sessions(self, fov_id):
        return []


class _StubEditReport:
    def __init__(self, warnings=()):
        self.warnings = list(warnings)


def _patched_registry(store, calls, *, edit_calls=None, edit_warnings=None):
    """Patch the registry surface run_tracking imports, recording each call.

    ``apply_tracking_edits`` is stubbed too: its own contract is covered by
    ``registry/tests/test_cell_edits.py`` against a real store, and
    ``_FakeStore`` here does not implement the full protocol that function
    needs (``list_cells``, ``replace_observations``, ...). What this module
    owns is the *replay is called* contract — see
    ``test_replay_is_called_once_per_fov_after_the_session_loop``.
    """
    def _register(**kwargs):
        calls.append(kwargs)
        return {
            "decision": "auto_match",
            "fov_id": "fov-1",
            "session_id": f"sess-{len(calls)}",
            "fov_posterior": 0.95,
            "n_matched": 2, "n_new": 0, "n_missing": 0,
        }

    def _apply_edits(fov_id, input_root, store):
        if edit_calls is not None:
            edit_calls.append({"fov_id": fov_id, "input_root": input_root})
        return _StubEditReport(edit_warnings or [])

    return [
        patch("roigbiv.registry.register_or_match", side_effect=_register),
        patch("roigbiv.registry.build_store", return_value=store),
        patch("roigbiv.registry.build_blob_store", return_value=object()),
        patch("roigbiv.registry.build_adapter_config", return_value=None),
        patch("roigbiv.registry.load_calibration", return_value=None),
        patch("roigbiv.registry.cell_edits.apply_tracking_edits",
             side_effect=_apply_edits),
    ]


def _run(workspace, store, calls, *, logs=None, edit_calls=None, edit_warnings=None):
    from roigbiv.pipeline.workspace import run_tracking

    patches = _patched_registry(store, calls, edit_calls=edit_calls,
                                edit_warnings=edit_warnings)
    for p in patches:
        p.start()
    try:
        log_cb = logs.append if logs is not None else (lambda _m: None)
        return run_tracking(workspace, {"fs": 7.5}, log_cb=log_cb)
    finally:
        for p in patches:
            p.stop()


def test_sessions_register_in_the_confirmed_human_order():
    """Registration order is cell-identity seniority — it must follow the human."""
    from roigbiv.pipeline.session_order import propose_order, reorder, save_order

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # Discovery order is alphabetical: beh, post, pre. The human wants
        # pre -> beh -> post, which no date can express (all one day).
        workspace = _workspace(root, PRISM)
        save_order(root, reorder(propose_order(PRISM), PRISM))

        store, calls = _FakeStore(), []
        results = _run(workspace, store, calls)

        assert [c["fov_stem"] for c in calls] == PRISM
        assert [r.stem for r in results] == PRISM
        assert [r.sequence_index for r in results] == [0, 1, 2]


def test_timeline_positions_are_persisted():
    from roigbiv.pipeline.session_order import propose_order, reorder, save_order

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM)
        save_order(root, reorder(propose_order(PRISM), PRISM))

        store, calls = _FakeStore(), []
        _run(workspace, store, calls)

        assert store.sequences == {"sess-1": 0, "sess-2": 1, "sess-3": 2}


def test_merged_masks_are_written_for_the_registry():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM[:1])

        store, calls = _FakeStore(), []
        results = _run(workspace, store, calls)

        masks = root / "output" / PRISM[0] / "merged_masks.tif"
        assert masks.exists()
        assert sorted(np.unique(tifffile.imread(masks)).tolist()) == [0, 1, 2]
        assert results[0].n_centroids == 2


def test_replay_is_called_once_per_fov_after_the_session_loop():
    """A centroid edit changes the FOV fingerprint, so a fresh registration
    misses the idempotency guard and rebuilds observations from scratch —
    without a post-loop replay, that would silently destroy every edit made
    since the last run.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM)

        store, calls, edit_calls = _FakeStore(), [], []
        _run(workspace, store, calls, edit_calls=edit_calls)

        # All three sessions register into the same stubbed fov_id ("fov-1"),
        # so the replay must fire exactly once, not once per session.
        assert len(edit_calls) == 1
        assert edit_calls[0]["fov_id"] == "fov-1"
        assert edit_calls[0]["input_root"] == workspace.input_root


def test_replay_warnings_reach_the_log():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM[:1])

        store, calls, logs = _FakeStore(), [], []
        _run(workspace, store, calls, logs=logs,
            edit_warnings=["s1: something stale"])

        assert any("fov-1" in line and "replaying edits" in line for line in logs)
        assert any("something stale" in line for line in logs)


def test_centroid_edit_warnings_reach_the_log():
    """A stale correction (e.g. naming a since-deleted label) must not be silent."""
    from roigbiv.pipeline.centroid_edits import CentroidOp, append_centroid_op

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM[:1])
        append_centroid_op(root / "output" / PRISM[0], CentroidOp.delete(99))

        store, calls, logs = _FakeStore(), [], []
        _run(workspace, store, calls, logs=logs)

        assert any("centroid edit" in line and "99" in line for line in logs)


def test_fov_without_centroids_is_skipped_not_failed():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM[:2], with_centroids=False)

        store, calls = _FakeStore(), []
        results = _run(workspace, store, calls)

        assert calls == []
        assert all(r.skipped for r in results)
        assert all(r.error is None for r in results)
        assert "centroid discovery" in results[0].skipped


def test_a_partially_marked_workspace_still_tracks_what_it_can():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM)
        (root / "output" / PRISM[1] / "centroids.json").unlink()

        store, calls = _FakeStore(), []
        results = _run(workspace, store, calls)

        assert len(calls) == 2
        assert sum(1 for r in results if r.skipped) == 1
        assert sum(1 for r in results if r.registry) == 2


def test_order_file_is_created_when_absent():
    """A first run proposes an order rather than refusing to start."""
    from roigbiv.pipeline.session_order import ORDER_FILENAME, load_order

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM)
        assert not (root / ORDER_FILENAME).exists()

        store, calls = _FakeStore(), []
        _run(workspace, store, calls)

        assert (root / ORDER_FILENAME).exists()
        assert len(load_order(root)) == 3


def test_parsed_date_is_passed_as_the_session_date():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM[:1])

        store, calls = _FakeStore(), []
        _run(workspace, store, calls)

        from datetime import date
        assert calls[0]["session_date_override"] == date(2026, 5, 21)


def test_overlapping_stamps_are_reported_on_the_result():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM[:1])
        # Two centroids 3 px apart — well inside one stamp diameter.
        (root / "output" / PRISM[0] / "centroids.json").write_text(json.dumps({
            "stem": PRISM[0], "schema": 4,
            "centroids": [
                {"label_id": 0, "y": 20.0, "x": 20.0, "npix": 100},
                {"label_id": 1, "y": 20.0, "x": 23.0, "npix": 100},
            ],
        }))

        store, calls = _FakeStore(), []
        results = _run(workspace, store, calls)

        assert results[0].n_overlapping_pairs == 1


def test_missing_roicat_is_announced_before_the_run():
    """Without ROICaT nothing can match — say so, don't fake success."""
    from roigbiv.pipeline.workspace import run_tracking

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM[:1])
        logs: list[str] = []
        store = _FakeStore()

        patches = _patched_registry(store, [])
        patches.append(patch("importlib.util.find_spec", return_value=None))
        for p in patches:
            p.start()
        try:
            run_tracking(workspace, {"fs": 7.5}, log_cb=logs.append)
        finally:
            for p in patches:
                p.stop()

        assert any("ROICaT is not installed" in line for line in logs)
        assert any("register as a NEW FOV" in line for line in logs)


def test_matching_nothing_is_called_out_even_though_the_fov_holds_together():
    """Forced grouping must not disguise a matcher that found no cells.

    Sessions can no longer scatter into separate FOVs, so the old tell — every
    session minting its own FOV — is gone. The failure it stood for is not:
    every session still lands in the timeline while every ROI registers as a
    brand-new cell, and the per-FOV summary below reads as a healthy run.
    """
    from roigbiv.pipeline.workspace import run_tracking

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM)
        logs: list[str] = []
        store = _FakeStore()

        def _register(**kwargs):
            return {"decision": "forced_fov", "fov_id": "fov-1",
                    "session_id": f"sess-{kwargs['fov_stem']}",
                    "n_matched": 0, "n_new": 2, "n_missing": 0}

        patches = [
            patch("roigbiv.registry.register_or_match", side_effect=_register),
            patch("roigbiv.registry.build_store", return_value=store),
            patch("roigbiv.registry.build_blob_store", return_value=object()),
            patch("roigbiv.registry.build_adapter_config", return_value=None),
            patch("roigbiv.registry.load_calibration", return_value=None),
            patch("roigbiv.registry.cell_edits.apply_tracking_edits",
                 side_effect=lambda *a, **k: _StubEditReport()),
        ]
        for p in patches:
            p.start()
        try:
            run_tracking(workspace, {"fs": 7.5}, log_cb=logs.append)
        finally:
            for p in patches:
                p.stop()

        assert any("no cell matched across sessions" in line for line in logs)


def test_review_decision_writes_no_sequence_index():
    """The review branch creates no session row, so there is nothing to place."""
    from roigbiv.pipeline.workspace import run_tracking

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM[:1])
        store = _FakeStore()

        patches = [
            patch("roigbiv.registry.register_or_match",
                  return_value={"decision": "review", "fov_posterior": 0.6}),
            patch("roigbiv.registry.build_store", return_value=store),
            patch("roigbiv.registry.build_blob_store", return_value=object()),
            patch("roigbiv.registry.build_adapter_config", return_value=None),
            patch("roigbiv.registry.load_calibration", return_value=None),
        ]
        for p in patches:
            p.start()
        try:
            results = run_tracking(workspace, {"fs": 7.5}, log_cb=lambda _m: None)
        finally:
            for p in patches:
                p.stop()

        assert store.sequences == {}
        assert results[0].registry["decision"] == "review"


def test_a_crashed_matcher_is_distinguished_from_genuinely_new_cells():
    """Both come back with nothing matched, and they mean opposite things.

    A matcher that raised says nothing about whether these cells correspond;
    without this the run reports a clean "no cell matched" and the researcher
    concludes the sessions really do share no cells.
    """
    from roigbiv.pipeline.workspace import run_tracking

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM)
        logs: list[str] = []
        store = _FakeStore()

        def _register(**kwargs):
            return {
                "decision": "forced_fov",
                "fov_id": "fov-1",
                "session_id": f"sess-{kwargs['fov_stem']}",
                "n_matched": 0,
                "n_new": 2,
                "n_missing": 0,
                "match_errors": [{
                    "candidate_fov_id": "fov-earlier",
                    "error": "ValueError: zero-size array to reduction "
                             "operation fmin which has no identity",
                }],
            }

        patches = _patched_registry(store, [])
        patches[0] = patch("roigbiv.registry.register_or_match",
                           side_effect=_register)
        for p in patches:
            p.start()
        try:
            run_tracking(workspace, {"fs": 7.5}, log_cb=logs.append)
        finally:
            for p in patches:
                p.stop()

        text = "\n".join(logs)
        assert "no cell matched across sessions" in text
        assert "it failed" in text
        assert "zero-size array" in text
        # The benign explanation must not be offered when the matcher crashed.
        assert "really are the same field of view" not in text


def test_identical_match_errors_are_reported_once():
    """Every session hits the same upstream bug; don't print it N times."""
    from roigbiv.pipeline.workspace import run_tracking

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM)
        logs: list[str] = []
        store = _FakeStore()

        def _register(**kwargs):
            return {
                "decision": "new_fov", "fov_id": f"fov-{kwargs['fov_stem']}",
                "session_id": f"sess-{kwargs['fov_stem']}", "n_new_cells": 2,
                "match_errors": [{"candidate_fov_id": "c",
                                  "error": "TypeError: boom"}],
            }

        patches = _patched_registry(store, [])
        patches[0] = patch("roigbiv.registry.register_or_match",
                           side_effect=_register)
        for p in patches:
            p.start()
        try:
            run_tracking(workspace, {"fs": 7.5}, log_cb=logs.append)
        finally:
            for p in patches:
                p.stop()

        assert "\n".join(logs).count("TypeError: boom") == 1


def test_genuinely_distinct_fovs_still_get_the_benign_explanation():
    from roigbiv.pipeline.workspace import run_tracking

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = _workspace(root, PRISM)
        logs: list[str] = []
        store = _FakeStore()

        def _register(**kwargs):
            return {"decision": "new_fov", "fov_id": f"fov-{kwargs['fov_stem']}",
                    "session_id": f"sess-{kwargs['fov_stem']}", "n_new_cells": 2,
                    "match_errors": []}

        patches = _patched_registry(store, [])
        patches[0] = patch("roigbiv.registry.register_or_match",
                           side_effect=_register)
        for p in patches:
            p.start()
        try:
            run_tracking(workspace, {"fs": 7.5}, log_cb=logs.append)
        finally:
            for p in patches:
                p.stop()

        text = "\n".join(logs)
        assert "really are the same field of view" in text
        assert "it failed" not in text
