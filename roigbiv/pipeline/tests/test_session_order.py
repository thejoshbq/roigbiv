"""
Contract tests for the workspace session ordering
(:mod:`roigbiv.pipeline.session_order`).

The order file is what makes a human's chronology authoritative over filename
dates. These cover the proposal heuristic, round-tripping, and the invariant
that matters most: re-scanning a workspace must never reshuffle a timeline a
human already confirmed.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

# The reference prism workspace: three sessions, one date, order pre->beh->post.
PRISM = [
    "052126_DS-Prism-3_VI15_D2_FOV2_post-007",
    "052126_DS-Prism-3_VI15_D2_FOV2_pre-005",
    "052126_DS-Prism-3_VI15_D2_FOV2_beh-006",
]


def test_proposal_sorts_datable_stems_chronologically():
    from roigbiv.pipeline.session_order import propose_order

    entries = propose_order([
        "T1_230116_PrL-NAc-G6-5M_EXT-D9_FOV1_PRE-000",
        "T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_PRE-002",
        "T1_221215_PrL-NAc-G6-5M_LOW-D1_FOV1_PRE-000",
    ])

    assert [e.session_date for e in entries] == [
        "2022-12-09", "2022-12-15", "2023-01-16",
    ]
    assert [e.index for e in entries] == [0, 1, 2]


def test_same_day_stems_get_a_stable_but_unconfirmed_order():
    """Dates cannot order these — the proposal is a starting point, not truth."""
    from roigbiv.pipeline.session_order import propose_order

    entries = propose_order(PRISM)

    assert {e.session_date for e in entries} == {"2026-05-21"}
    assert [e.index for e in entries] == [0, 1, 2]
    # Nothing is locked until a human says so.
    assert not any(e.locked for e in entries)


def test_ambiguous_and_unparsed_stems_sort_last_and_flag_for_review():
    from roigbiv.pipeline.session_order import propose_order

    entries = propose_order([
        "999999_DS-Prism-3_VI15_D2_FOV1_beh-001",   # unparseable
        "060126_DS-Prism-3_VI15_D3_FOV2_beh-007",   # ambiguous MMDDYY/YYMMDD
        "052126_DS-Prism-3_VI15_D2_FOV2_pre-005",   # unambiguous
    ])

    assert entries[0].stem.startswith("052126")
    assert entries[0].date_source == "mmddyy"
    assert entries[0].needs_review is False

    trailing = {e.date_source for e in entries[1:]}
    assert trailing == {"ambiguous", "unparsed"}
    assert all(e.needs_review for e in entries[1:])


def test_save_and_load_round_trip():
    from roigbiv.pipeline.session_order import load_order, propose_order, save_order

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        entries = propose_order(PRISM)
        entries[0].locked = True
        save_order(root, entries)

        loaded = load_order(root)
        assert [e.stem for e in loaded] == [e.stem for e in entries]
        assert [e.index for e in loaded] == [0, 1, 2]
        assert loaded[0].locked is True


def test_missing_order_file_loads_as_empty():
    from roigbiv.pipeline.session_order import load_order

    with tempfile.TemporaryDirectory() as td:
        assert load_order(Path(td)) == []


def test_corrupt_order_file_loads_as_empty_rather_than_raising():
    from roigbiv.pipeline.session_order import ORDER_FILENAME, load_order

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / ORDER_FILENAME).write_text("{not json")
        assert load_order(root) == []


def test_reorder_marks_touched_entries_locked():
    from roigbiv.pipeline.session_order import propose_order, reorder

    entries = propose_order(PRISM)
    wanted = [
        "052126_DS-Prism-3_VI15_D2_FOV2_pre-005",
        "052126_DS-Prism-3_VI15_D2_FOV2_beh-006",
        "052126_DS-Prism-3_VI15_D2_FOV2_post-007",
    ]

    result = reorder(entries, wanted)

    assert [e.stem for e in result] == wanted
    assert [e.index for e in result] == [0, 1, 2]
    assert all(e.locked for e in result)


def test_partial_reorder_never_drops_a_session():
    from roigbiv.pipeline.session_order import propose_order, reorder

    entries = propose_order(PRISM)
    result = reorder(entries, ["052126_DS-Prism-3_VI15_D2_FOV2_beh-006"])

    assert len(result) == 3
    assert result[0].stem.endswith("beh-006")
    assert result[0].locked is True
    assert not any(e.locked for e in result[1:])


def test_rescan_preserves_a_confirmed_order():
    """The core invariant: a re-scan must not undo a human's decision."""
    from roigbiv.pipeline.session_order import propose_order, reorder, resolve_order, save_order

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wanted = [
            "052126_DS-Prism-3_VI15_D2_FOV2_pre-005",
            "052126_DS-Prism-3_VI15_D2_FOV2_beh-006",
            "052126_DS-Prism-3_VI15_D2_FOV2_post-007",
        ]
        save_order(root, reorder(propose_order(PRISM), wanted))

        # A re-scan sees the same stems in a different discovery order.
        resolved = resolve_order(root, sorted(PRISM))

        assert [e.stem for e in resolved] == wanted


def test_new_stems_are_appended_after_a_confirmed_order():
    from roigbiv.pipeline.session_order import propose_order, reorder, resolve_order, save_order

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        wanted = [
            "052126_DS-Prism-3_VI15_D2_FOV2_pre-005",
            "052126_DS-Prism-3_VI15_D2_FOV2_beh-006",
            "052126_DS-Prism-3_VI15_D2_FOV2_post-007",
        ]
        save_order(root, reorder(propose_order(PRISM), wanted))

        newcomer = "052226_DS-Prism-3_VI15_D3_FOV2_beh-008"
        resolved = resolve_order(root, PRISM + [newcomer])

        assert [e.stem for e in resolved] == wanted + [newcomer]
        assert resolved[-1].locked is False
        assert resolved[-1].index == 3


def test_removed_stems_drop_out_of_the_order():
    from roigbiv.pipeline.session_order import propose_order, resolve_order, save_order

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        save_order(root, propose_order(PRISM))

        resolved = resolve_order(root, PRISM[:2])

        assert len(resolved) == 2
        assert [e.index for e in resolved] == [0, 1]


def test_trackable_stems_come_from_output_not_from_input_tifs():
    """Prairie View reference snapshots are files, not sessions.

    The reference workspace holds three recordings plus six single-frame
    reference/thumbnail TIFs that ``resolve_workspace`` also discovers. Only
    the three the pipeline produced output for are orderable sessions.
    """
    import numpy as np
    import tifffile

    from roigbiv.pipeline.session_order import discover_trackable_stems
    from roigbiv.pipeline.workspace import resolve_workspace

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        frame = np.zeros((2, 16, 16), dtype=np.uint16)
        for stem in PRISM:
            tifffile.imwrite(root / f"{stem}.tif", frame)
            tifffile.imwrite(root / f"{stem}-Ch2-16bit-Reference.tif", frame)
            tifffile.imwrite(
                root / f"{stem}-Window1-Ch2-8bit-Reference.tif", frame)
            (root / "output" / stem).mkdir(parents=True, exist_ok=True)

        workspace = resolve_workspace(root)

        assert len(workspace.tifs) == 9
        assert sorted(discover_trackable_stems(workspace)) == sorted(PRISM)


def test_trackable_stems_is_empty_before_any_output_exists():
    from roigbiv.pipeline.session_order import discover_trackable_stems

    class _WS:
        output_root = Path("/nonexistent/output")

    assert discover_trackable_stems(_WS()) == []


def test_saved_file_is_readable_json_with_a_schema():
    from roigbiv.pipeline.session_order import ORDER_FILENAME, propose_order, save_order

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        save_order(root, propose_order(PRISM))

        payload = json.loads((root / ORDER_FILENAME).read_text())
        assert payload["schema"] == 1
        assert len(payload["order"]) == 3
        assert payload["order"][0]["stem"]
