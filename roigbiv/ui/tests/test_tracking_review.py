"""Cells page — the server-rendered shell around the contact sheet.

The sheet itself is not here. It is OpenSeadragon viewers built by
``assets/cells_sheet.js`` against ``roigbiv/ui/routes/cells_api.py``, and what
it draws is guarded by ``test_cells_api.py``; what a gesture *means* is guarded
by ``test_cell_edit_ops.py``. What this module still owns is the frame around
it: the picker, the header, the cell rail, and the handoff that tells the
browser what to render.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from roigbiv.ui.pages import tracking as cells
from roigbiv.ui.services.loaders import ROIRender
from roigbiv.ui.services.registry_service import FOVRow
from roigbiv.ui.services.tracked_cells import TrackedCell, TrackedFOV, TrackedSession

STEMS = ["fov_pre-005", "fov_beh-006", "fov_post-007"]


# ── component-tree helpers (shared idiom with test_track_page) ─────────────


def _walk(component):
    """Every node in a component tree."""
    if isinstance(component, (list, tuple)):
        for item in component:
            yield from _walk(item)
        return
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        yield from _walk(child)


def _ids(root):
    return {_hashable(getattr(c, "id", None)) for c in _walk(root)}


def _hashable(value):
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


def _text(root) -> str:
    return " ".join(n for n in _walk(root) if isinstance(n, str))


# ── fixtures ───────────────────────────────────────────────────────────────


class _FakeState:
    def __init__(self, workspace=object()):
        self.workspace = workspace
        self.registry_config = None


def _roi(label_id, cy, cx, status, gcid):
    ys = [cy - 5, cy - 5, cy + 5, cy + 5]
    xs = [cx - 5, cx + 5, cx + 5, cx - 5]
    return ROIRender(
        label_id=label_id, source_stage=1, gate_outcome="accept",
        activity_type=None, area=78, centroid_yx=(float(cy), float(cx)),
        contours=[(ys, xs)], global_cell_id=gcid, match_status=status,
    )


def _fov(*, ordering_is_confirmed: bool = True) -> TrackedFOV:
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
            sequence_index=i, output_dir=f"/fake/{STEMS[i]}",
            mean_M=np.zeros((80, 80), dtype=np.float32),
            rois=[_roi(lid, cy, cx, status, f"gcid-{name}")
                  for name, lid, cy, cx, status in entries],
            n_matched=2 if i else 0, n_new=0 if i else 2,
            n_missing=1 if i == 2 else 0,
        ))
    cells_ = [
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
                      sessions=sessions, cells=cells_,
                      ordering_is_confirmed=ordering_is_confirmed)


@pytest.fixture
def callbacks():
    captured = {}

    class _App:
        def callback(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

        def clientside_callback(self, *a, **k):
            """The browser-side halves — captured with their dependencies."""
            captured.setdefault("_clientside", []).append(a)

    cells.register_callbacks(_App())
    return captured


# ── layout ─────────────────────────────────────────────────────────────────


def _layout():
    """The review half of the Tracking page.

    Only that half: the setup half above it needs a real workspace to resolve a
    session order from, and none of these cases is about session ordering.
    ``tracking.layout()`` is what stitches the two together, and
    ``test_tracking_setup.py`` covers the seam.
    """
    with patch.object(cells, "get_app_state", return_value=_FakeState()), \
            patch.object(cells, "list_fovs", return_value=[]):
        return cells._review_section()


def test_layout_carries_the_stores_and_navigation_controls():
    assert {cells.SELECTED_ID, cells.VIEW_ID, cells.SHEET_ID,
            cells.CELL_LIST_ID, cells.PREV_ID, cells.NEXT_ID, cells.NUMBERS_ID,
            cells.BOUNDARIES_ID,
            cells.RAIL_TOGGLE_ID} <= _ids(_layout())


def test_the_sheet_is_an_empty_mount_point_for_the_browser_to_fill():
    """Panels never travel through a Dash payload — the browser fetches its own
    geometry and images. This is what keeps an edit from re-shipping every
    projection, and therefore what keeps the zoom."""
    sheet = next(c for c in _walk(_layout())
                 if getattr(c, "id", None) == cells.SHEET_ID)
    assert not getattr(sheet, "children", None)


def test_the_sheet_and_the_cell_rail_are_separately_addressable():
    """The rail toggle flips both column classes, so both need ids."""
    assert {cells.SHEET_COL_ID, cells.RAIL_COL_ID} <= _ids(_layout())


def test_the_rail_collapse_control_sits_on_the_rail_not_in_the_toolbar():
    """It hides the cell list, so it belongs beside the cell list. In the
    toolbar it read as *go to cells* and went unfound."""
    assert cells.RAIL_TOGGLE_ID in _ids(cells._rail())
    with patch.object(cells, "get_app_state", return_value=_FakeState()), \
            patch.object(cells, "list_fovs", return_value=[]):
        assert cells.RAIL_TOGGLE_ID not in _ids(cells._toolbar())


def test_collapsing_leaves_a_way_back():
    """The rail is removed outright when collapsed, so the control that brings
    it back cannot live inside it."""
    assert cells.RAIL_TAB_ID in _ids(_layout())
    assert cells.RAIL_TAB_ID not in _ids(cells._rail())


def test_the_rail_state_is_remembered_across_reloads():
    """Derived from n_clicks parity it could not be: a restored store and a
    zeroed click count are two different answers to the same question."""
    store = next(c for c in _walk(_layout())
                 if getattr(c, "id", None) == cells.RAIL_STATE_ID)
    assert store.storage_type == "local"
    assert store.data is False


def test_one_class_on_the_body_drives_the_whole_split():
    """Rail, sheet width and edge tab all follow `is-collapsed`, so they cannot
    drift out of agreement the way three separate className outputs could."""
    assert cells.BODY_ID in _ids(_layout())


def test_layout_carries_the_edit_controls():
    assert {cells.EDIT_ID, cells.EDIT_ROW_ID, cells.UNDO_ID,
            cells.EDIT_MSG_ID} <= _ids(_layout())


def test_the_boundaries_switch_defaults_off():
    """Disks are the canonical registry geometry (ADR-0003) and load first;
    seeded boundaries (ADR-0005) are opt-in."""
    with patch.object(cells, "get_app_state", return_value=_FakeState()), \
            patch.object(cells, "list_fovs", return_value=[]):
        switch = next(c for c in _walk(cells._toolbar())
                     if getattr(c, "id", None) == cells.BOUNDARIES_ID)
    assert switch.value is False


def test_there_is_no_mode_picker_left():
    """Editing is modeless: click selects, drag moves, right-click deletes,
    shift-click links. A mode radio was most of the click count the page was
    rebuilt to cut."""
    assert not hasattr(cells, "MODE_ID")
    assert not hasattr(cells, "EDIT_MODES")
    assert not hasattr(cells, "PICKUP_ID")


def test_the_gestures_are_named_on_screen_rather_than_left_to_be_discovered():
    """Direct manipulations have no on-screen controls to find them from, and
    this page is used about once a fortnight."""
    text = _text(_layout())
    for phrase in ["drag", "right-click", "shift-click", "Ctrl+Z"]:
        assert phrase in text


def test_edit_row_starts_hidden():
    row = next(c for c in _walk(_layout())
               if getattr(c, "id", None) == cells.EDIT_ROW_ID)
    assert "d-none" in row.className


def test_edit_toggle_shows_and_hides_the_edit_row(callbacks):
    assert "d-none" in callbacks["_on_edit_toggle"](False)
    assert "d-none" not in callbacks["_on_edit_toggle"](True)


def test_without_a_workspace_the_page_asks_for_a_scan():
    with patch.object(cells, "get_app_state", return_value=_FakeState(None)):
        assert "Scan a workspace" in _text(cells._fov_picker())


def test_with_no_tracked_fovs_the_page_points_at_the_setup_above_it():
    """The way out is now on this page, not on another one.

    Tracking and review were two pages, so the empty state had to name the
    other one. They are two halves of one page now, and the setup half is
    directly above.
    """
    with patch.object(cells, "get_app_state", return_value=_FakeState()), \
            patch.object(cells, "list_fovs", return_value=[]):
        assert "run tracking above" in _text(cells._fov_picker())


def test_a_registry_failure_is_reported_rather_than_blanking():
    with patch.object(cells, "get_app_state", return_value=_FakeState()), \
            patch.object(cells, "list_fovs", side_effect=RuntimeError("db gone")):
        assert "db gone" in _text(cells._fov_picker())


def test_fovs_without_sessions_are_not_offered():
    """A FOV with no sessions has nothing to draw; offering it would render an
    empty sheet with no explanation."""
    def _row(fov_id, n_sessions):
        return FOVRow(fov_id=fov_id, animal_id="A", region="R", created_at=None,
                      latest_session_date=None, fingerprint_version=1,
                      n_sessions=n_sessions)

    rows = [_row("f-empty", 0), _row("f-real", 3)]
    with patch.object(cells, "get_app_state", return_value=_FakeState()), \
            patch.object(cells, "list_fovs", return_value=rows):
        picker = cells._fov_picker()
    dropdown = next(c for c in _walk(picker)
                    if getattr(c, "id", None) == cells.FOV_ID)
    assert [o["value"] for o in dropdown.options] == ["f-real"]


def test_the_legend_names_all_three_outcomes():
    text = _text(cells._legend())
    for label in ("Matched", "New here", "Not detected"):
        assert label in text


# ── cell rail and header ───────────────────────────────────────────────────


def test_the_cell_list_shows_a_presence_timeline_per_cell():
    text = _text(cells._cell_list(_fov(), None))
    assert "●●●" in text        # A, seen throughout
    assert "●●○" in text        # B, dropping out
    assert "dropout" in text


def test_each_cell_row_is_addressable_by_its_registry_id():
    ids = _ids(cells._cell_list(_fov(), None))
    assert _hashable({"type": "roigbiv-cells-row", "gcid": "gcid-A"}) in ids


def test_the_selected_row_is_marked_and_the_others_are_not():
    rows = [c for c in _walk(cells._cell_list(_fov(), "gcid-B"))
            if isinstance(getattr(c, "id", None), dict)]
    marked = {r.id["gcid"] for r in rows if "roigbiv-cells-row-active" in r.className}
    assert marked == {"gcid-B"}


def test_an_empty_fov_says_so_rather_than_rendering_a_blank_table():
    empty = TrackedFOV(fov_id="f", animal_id=None, region=None)
    assert "No tracked cells" in _text(cells._cell_list(empty, None))


def test_the_header_summarises_the_timeline():
    text = _text(cells._header(_fov()))
    assert "3 sessions" in text
    assert "2 cells" in text
    assert "1 seen throughout" in text     # only A is present in all three


def test_an_unconfirmed_session_order_is_flagged_in_the_header():
    assert "not confirmed" in _text(cells._header(_fov(ordering_is_confirmed=False)))
    assert "not confirmed" not in _text(cells._header(_fov()))


# ── callbacks ──────────────────────────────────────────────────────────────


def test_choosing_a_fov_fills_the_rail_and_clears_any_selection(callbacks):
    with patch.object(cells, "_load", return_value=_fov()):
        cell_list, header, selected = callbacks["_on_fov"]("fov-1")
    assert selected is None
    assert "●●●" in _text(cell_list)
    assert "3 sessions" in _text(header)


def test_choosing_no_fov_renders_nothing_rather_than_raising(callbacks):
    cell_list, header, selected = callbacks["_on_fov"](None)
    assert selected is None
    assert header == []


def test_a_failure_loading_a_fov_is_reported_in_place(callbacks):
    with patch.object(cells, "_load", side_effect=RuntimeError("masks missing")):
        _cell_list, header, _selected = callbacks["_on_fov"]("fov-1")
    assert "masks missing" in _text(header)


def test_selecting_a_cell_highlights_it_in_the_rail(callbacks):
    with patch.object(cells, "_load", return_value=_fov()):
        cell_list = callbacks["_on_select"]("gcid-B", "fov-1")
    assert "roigbiv-cells-row-active" in " ".join(
        c.className for c in _walk(cell_list)
        if isinstance(getattr(c, "id", None), dict))


def test_clearing_the_selection_removes_highlighting(callbacks):
    with patch.object(cells, "_load", return_value=_fov()):
        cell_list = callbacks["_on_select"](None, "fov-1")
    # No row should have the active class
    assert not any("roigbiv-cells-row-active" in getattr(c, "className", "")
                   for c in _walk(cell_list))


def test_clicking_a_row_selects_that_cell(callbacks):
    class _Ctx:
        triggered_id = {"type": "roigbiv-cells-row", "gcid": "gcid-B"}

    with patch.object(cells, "ctx", _Ctx()):
        assert callbacks["_on_row_click"]([0, 1]) == "gcid-B"


def test_a_row_render_alone_does_not_count_as_a_click(callbacks):
    assert callbacks["_on_row_click"]([0, 0]) is cells.no_update


def test_stepping_walks_the_cells_and_wraps_at_the_end(callbacks):
    def _step(trigger, selected):
        class _Ctx:
            triggered_id = trigger
        with patch.object(cells, "ctx", _Ctx()), \
                patch.object(cells, "_load", return_value=_fov()):
            return callbacks["_on_step"](1, 1, selected, "fov-1")

    assert _step(cells.NEXT_ID, "gcid-A") == "gcid-B"
    assert _step(cells.NEXT_ID, "gcid-B") == "gcid-A"     # wraps forward
    assert _step(cells.PREV_ID, "gcid-A") == "gcid-B"     # wraps backward
    assert _step(cells.NEXT_ID, None) == "gcid-A"         # nothing selected yet


# ── the browser handoff ────────────────────────────────────────────────────


def test_the_sheet_is_told_about_every_control_it_reacts_to(callbacks):
    """One render call rather than a listener per switch: the sheet has a
    single entry point, so nothing can update half of it."""
    render = callbacks["_clientside"][0]
    inputs = [dep.component_id for dep in render[2:]]
    assert inputs == [cells.FOV_ID, cells.SELECTED_ID, cells.NUMBERS_ID,
                      cells.EDIT_ID, cells.BOUNDARIES_ID]


def test_selection_and_the_switches_never_go_through_the_server(callbacks):
    """A repaint for a selection is what used to throw the zoom away. Every
    handoff stays in the browser."""
    assert len(callbacks["_clientside"]) == 4
    assert "roigbivCells.render" in callbacks["_clientside"][0][0]
    assert "roigbivCells.undo" in callbacks["_clientside"][1][0]
    assert "roigbivCells.resize" in callbacks["_clientside"][3][0]


def test_the_rail_toggle_writes_the_store_rather_than_counting_clicks(callbacks):
    """Both the rail chevron and the edge tab flip one persisted boolean, and
    the class mapping reads only that — so a reload paints the state it left."""
    toggle = callbacks["_clientside"][2]
    assert [dep.component_id for dep in toggle[2:4]] == [cells.RAIL_TOGGLE_ID,
                                                         cells.RAIL_TAB_ID]
    assert toggle[1].component_id == cells.RAIL_STATE_ID

    apply_ = callbacks["_clientside"][3]
    assert apply_[2].component_id == cells.RAIL_STATE_ID
    assert apply_[1].component_id == cells.BODY_ID
    assert "is-collapsed" in apply_[0]
