"""Cells page — the cross-session contact sheet and its callbacks.

The interaction being guarded is the one the page exists for: a click anywhere
in any session panel has to resolve to *one* cell, and that cell has to light
up in every other panel at the same time. Everything else here is the scaffold
that makes that possible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest

from roigbiv.ui.pages import cells
from roigbiv.ui.services.loaders import ROIRender
from roigbiv.ui.services.registry_service import FOVRow
from roigbiv.ui.services.tracked_cells import TrackedCell, TrackedFOV, TrackedSession

STEMS = ["fov_pre-005", "fov_beh-006", "fov_post-007"]


# ── component-tree helpers (shared idiom with test_track_page) ─────────────


def _walk(component):
    """Every node in a component tree.

    Unlike the Track page's walker this recurses unconditionally: a
    ``dcc.Graph`` with no explicit id exposes neither ``children`` nor ``id``,
    and the filmstrip is built entirely out of those.
    """
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


def _graphs(root):
    return [c for c in _walk(root) if getattr(c, "figure", None) is not None]


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


def _session_with(mean) -> TrackedSession:
    """The bare minimum ``_sheet_step`` reads — a frame with a size."""
    return TrackedSession(
        session_id="s", stem="stem", session_date=None, sequence_index=0,
        output_dir=None, mean_M=mean, rois=[],
        n_matched=0, n_new=0, n_missing=0,
    )


def _fov(*, stale_session: Optional[int] = None, with_output_dirs: bool = False) -> TrackedFOV:
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
            sequence_index=i,
            output_dir=(f"/fake/{STEMS[i]}" if with_output_dirs else None),
            mean_M=np.zeros((80, 80), dtype=np.float32),
            rois=[_roi(lid, cy, cx, status, f"gcid-{name}")
                  for name, lid, cy, cx, status in entries],
            n_matched=2 if i else 0, n_new=0 if i else 2, n_missing=1 if i == 2 else 0,
            stale=(i == stale_session),
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
                      sessions=sessions, cells=cells_)


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
            """The browser-side halves — captured only so registration runs."""
            captured.setdefault("_clientside", []).append(a[0])

    cells.register_callbacks(_App())
    return captured


# ── layout ─────────────────────────────────────────────────────────────────


def _layout():
    with patch.object(cells, "get_app_state", return_value=_FakeState()), \
            patch.object(cells, "list_fovs", return_value=[]):
        return cells.layout()


def test_layout_carries_the_stores_and_navigation_controls():
    assert {cells.SELECTED_ID, cells.INDEX_ID, cells.SHEET_ID, cells.STRIP_ID,
            cells.LIST_ID, cells.PREV_ID, cells.NEXT_ID, cells.NUMBERS_ID,
            cells.DRAWER_ID, cells.RAIL_TOGGLE_ID} <= _ids(_layout())


def test_the_cell_controls_sit_outside_the_drawer():
    """The drawer only opens once a cell is selected, so anything that makes
    the *first* selection has to live outside it. An earlier revision put
    prev/next inside and they were unreachable from a cold page."""
    drawer = next(c for c in _walk(_layout())
                  if getattr(c, "id", None) == cells.DRAWER_ID)
    inside = _ids(drawer)
    assert cells.PREV_ID not in inside and cells.NEXT_ID not in inside
    assert cells.STRIP_ID in inside, "the filmstrip itself belongs in the drawer"


def test_the_sheet_and_the_cell_rail_are_separately_addressable():
    """The rail collapse swaps both columns' classes, so both need ids."""
    assert {cells.SHEET_COL_ID, cells.RAIL_COL_ID} <= _ids(_layout())


def test_layout_carries_the_edit_mode_controls():
    assert {cells.EDIT_ID, cells.MODE_ID, cells.MODE_ROW_ID, cells.UNDO_ID,
            cells.EDIT_MSG_ID, cells.PICKUP_ID} <= _ids(_layout())


def test_edit_mode_row_starts_hidden():
    """The read-only view is unchanged until a researcher opts into editing."""
    row = next(c for c in _walk(_layout())
              if getattr(c, "id", None) == cells.MODE_ROW_ID)
    assert "d-none" in row.className


def test_edit_toggle_shows_and_hides_the_mode_row(callbacks):
    assert "d-none" not in callbacks["_on_edit_toggle"](True)
    assert "d-none" in callbacks["_on_edit_toggle"](False)
    assert "d-none" in callbacks["_on_edit_toggle"](None)


def test_without_a_workspace_the_page_asks_for_a_scan():
    with patch.object(cells, "get_app_state",
                      return_value=_FakeState(workspace=None)):
        assert "Scan a workspace" in _text(cells.layout())


def test_with_no_tracked_fovs_the_page_points_at_the_track_page():
    with patch.object(cells, "get_app_state", return_value=_FakeState()), \
            patch.object(cells, "list_fovs", return_value=[]):
        assert "Track page" in _text(cells.layout())


def test_a_registry_failure_is_reported_rather_than_blanking():
    with patch.object(cells, "get_app_state", return_value=_FakeState()), \
            patch.object(cells, "list_fovs", side_effect=OSError("db is locked")):
        assert "db is locked" in _text(cells.layout())


def test_fovs_without_sessions_are_not_offered():
    rows = [FOVRow("a" * 32, "an", "rg", None, None, 3, 0),
            FOVRow("b" * 32, "an", "rg", None, None, 3, 2)]
    with patch.object(cells, "get_app_state", return_value=_FakeState()), \
            patch.object(cells, "list_fovs", return_value=rows):
        dropdown = next(c for c in _walk(cells.layout())
                        if getattr(c, "id", None) == cells.FOV_ID)
    assert [o["value"] for o in dropdown.options] == ["b" * 32]


def test_the_legend_names_all_three_outcomes():
    with patch.object(cells, "get_app_state", return_value=_FakeState()), \
            patch.object(cells, "list_fovs", return_value=[]):
        text = _text(cells.layout())
    assert "Matched" in text and "New here" in text and "Not detected" in text


# ── contact sheet ──────────────────────────────────────────────────────────


def test_one_panel_per_session_in_timeline_order():
    sheet, index = cells._sheet(_fov())
    assert len(_graphs(sheet)) == 3
    assert set(index) == {"0", "1", "2"}
    assert _text(sheet).index(STEMS[0]) < _text(sheet).index(STEMS[2])


def test_panels_carry_the_class_the_browser_side_selects_on():
    """Both clientside callbacks reach the plots through
    ``.roigbiv-cells-panel .js-plotly-plot``. Renaming the class silently
    breaks linked zoom and the rail's resize, neither of which pytest sees."""
    sheet, _index = cells._sheet(_fov())
    wrappers = [c for c in _walk(sheet)
                if "roigbiv-cells-panel" == (getattr(c, "className", "") or "")]
    assert len(wrappers) == 3
    assert all(_graphs(w) for w in wrappers), "each wrapper must contain its plot"


def test_panels_are_told_apart_when_every_session_shares_a_date():
    """The reference prism FOV records pre-005 / beh-006 / post-007 on one day.
    Labelling panels by date alone makes all three read identically, which
    defeats a page whose whole job is comparing them."""
    from datetime import date

    fov = _fov()
    for session in fov.sessions:
        session.session_date = date(2026, 5, 21)
    text = _text(cells._sheet(fov)[0])
    assert all(stem.rsplit("_", 1)[-1] in text for stem in STEMS)


def test_the_sheet_says_the_shared_window_is_only_approximate():
    """Linked zoom implies co-registration it does not have."""
    sheet, _index = cells._sheet(_fov())
    assert "not co-registered" in _text(sheet)


def test_each_panel_counts_what_it_is_actually_drawing():
    """The caption is read against the picture, so it must come from it."""
    sheet, _index = cells._sheet(_fov())
    text = _text(sheet)
    assert "2 new" in text            # session 1: both cells first seen
    assert "2 matched" in text        # session 2: both carried over
    assert "1 not detected" in text   # session 3: B's ghost


def test_the_caption_never_contradicts_the_ghosts_beside_it():
    """A registration row can say 0 missing while observations show otherwise."""
    fov = _fov()
    fov.sessions[2].n_missing = 0     # what the matcher recorded
    assert "1 not detected" in _text(cells._sheet(fov)[0])


def test_a_session_the_registry_disagrees_with_is_called_out():
    """Silently drawing the emptier of two answers would be the worse failure."""
    sheet, _index = cells._sheet(_fov(stale_session=1))
    assert "Re-run tracking" in _text(sheet)


def test_a_consistent_sheet_shows_no_stale_warning():
    sheet, _index = cells._sheet(_fov())
    assert "Re-run tracking" not in _text(sheet)


def test_a_fov_with_no_readable_sessions_says_so():
    empty = TrackedFOV(fov_id="f", animal_id=None, region=None)
    sheet, index = cells._sheet(empty)
    assert index == {}
    assert "no readable session" in _text(sheet)


def test_the_index_maps_every_outline_to_its_cell():
    _sheet, index = cells._sheet(_fov())
    assert index["0"]["gcid"] == {"1": "gcid-A", "2": "gcid-B"}
    assert set(index["0"]["outline"]) == {"1", "2"}
    assert set(index["0"]["badge"]) == {"1", "2"}


def test_a_dropped_cells_ghost_is_still_clickable_in_the_last_session():
    _sheet, index = cells._sheet(_fov())
    assert index["2"]["gcid"]["-2"] == "gcid-B"


@pytest.mark.parametrize("show_numbers", [True, False])
def test_badges_start_in_the_state_the_switch_asks_for(show_numbers):
    """The resting sheet is what the numbers switch is for — it has to hold
    from the first render, not only after the first restyle."""
    fov = _fov()
    figure, index = cells._panel_figure(
        fov, fov.sessions[0], 0, 1.0, show_numbers)
    for trace_id in index["badge"].values():
        assert figure.data[trace_id].visible is show_numbers


def test_the_sheet_background_is_downsampled_but_the_outlines_follow():
    """Contours are in frame pixels; a scaled background needs scaled contours."""
    big = np.zeros((2048, 2048), dtype=np.float32)
    scale = cells._sheet_step([_session_with(big)])
    mean = cells._downsampled(big, scale)
    assert max(mean.shape) <= cells._SHEET_MAX_PX
    roi = _roi(1, 1000, 1000, "matched", "g")
    scaled = cells._scaled(roi, scale)
    assert scaled.centroid_yx == (1000 / scale, 1000 / scale)
    assert max(max(xs) for _ys, xs in scaled.contours) <= max(mean.shape)


def test_a_panel_draws_its_outlines_on_the_background_it_actually_shipped():
    """The failure this catches is an outline sitting 4x off its own image."""
    fov = _fov()
    session = fov.sessions[0]
    session.mean_M = np.zeros((2048, 2048), dtype=np.float32)
    session.rois = [_roi(1, 1000, 1000, "matched", "gcid-A")]

    step = cells._sheet_step([session])
    figure, _index = cells._panel_figure(fov, session, 0, step, True)
    height, width = np.asarray(figure.data[0].z).shape
    outline = next(t for t in figure.data
                   if t.type == "scatter" and t.mode == "lines")
    assert max(outline.x) <= width and max(outline.y) <= height, (
        "outline falls outside the downsampled background it is drawn on")
    # And it still lands where the cell is, not collapsed into a corner.
    assert width * 0.25 < np.mean(outline.x) < width * 0.75


def test_every_panel_shares_one_coordinate_scale():
    """The precondition for linked zoom, and silently wrong when the reduction
    is derived per session: the synchroniser copies axis ranges between panels
    verbatim, so a panel reduced by 1 beside one reduced by 2 would jump to a
    window twice the size.

    Read off the built figures rather than recomputed here — recomputing the
    step and comparing it against itself proves nothing about what shipped.
    """
    fov = _fov()
    for session, size in zip(fov.sessions, (800, 2048, 2048)):
        session.mean_M = np.zeros((size, size), dtype=np.float32)
        # The same place in the frame, in all three sessions.
        session.rois = [_roi(1, 400, 400, "matched", "gcid-A")]

    sheet, _index = cells._sheet(fov)
    drawn = set()
    for graph in _graphs(sheet):
        outline = next(t for t in graph.figure.data
                       if t.type == "scatter" and t.mode == "lines")
        drawn.add((round(float(np.mean(outline.x)), 6),
                   round(float(np.mean(outline.y)), 6)))
    assert len(drawn) == 1, f"panels disagree on where frame (400,400) is: {drawn}"


def test_the_background_is_quantised_without_inverting_the_picture():
    """Payload measure only — it must not reorder intensities."""
    mean = np.linspace(0, 4.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    out = cells._quantized(mean)
    assert out.dtype == np.uint8
    assert out[0, 0] < out[-1, -1]
    assert out.min() == 0 and out.max() == 255, "display window is fully used"


def test_a_flat_frame_quantises_to_a_blank_rather_than_dividing_by_zero():
    assert cells._quantized(np.full((8, 8), 3.0)).max() == 0


# ── cell list ──────────────────────────────────────────────────────────────


def test_the_cell_list_shows_a_presence_timeline_per_cell():
    text = _text(cells._cell_list(_fov(), None))
    assert "●●●" in text        # A, seen throughout
    assert "●●○" in text        # B, dropped out
    assert "dropout" in text


def test_each_cell_row_is_addressable_by_its_registry_id():
    ids = _ids(cells._cell_list(_fov(), None))
    assert _hashable({"type": "roigbiv-cells-row", "gcid": "gcid-A"}) in ids


def test_the_selected_row_is_marked_and_the_others_are_not():
    rows = [c for c in _walk(cells._cell_list(_fov(), "gcid-B"))
            if isinstance(getattr(c, "id", None), dict)]
    active = [r for r in rows if "active" in (r.className or "")]
    assert len(active) == 1
    assert active[0].id["gcid"] == "gcid-B"


def test_the_header_summarises_the_timeline():
    text = _text(cells._header(_fov()))
    assert "3 sessions" in text and "2 cells" in text and "1 seen throughout" in text


def test_an_unconfirmed_session_order_is_flagged_in_the_header():
    fov = _fov()
    fov.ordering_is_confirmed = False
    assert "not confirmed" in _text(cells._header(fov))


# ── selection ──────────────────────────────────────────────────────────────


def _values(ops, key):
    return [op["params"]["value"] for op in ops if op["location"][-1] == key]


def test_selecting_a_cell_thickens_only_that_cell_and_enlarges_only_its_badge():
    _sheet, index = cells._sheet(_fov())
    ops = cells._restyle(index["0"], "gcid-A", True)._operations

    widths = _values(ops, "width")
    assert len(set(widths)) == 2, "one cell must differ from the rest"
    assert max(widths) == cells._HIGHLIGHT_WIDTH

    sizes = _values(ops, "size")
    assert sorted(sizes) == [cells._BADGE_SIZE, cells._BADGE_SIZE_SELECTED], (
        "exactly one badge is enlarged")


def test_clearing_the_selection_puts_every_outline_back():
    _sheet, index = cells._sheet(_fov())
    ops = cells._restyle(index["0"], None, False)._operations
    assert set(_values(ops, "width")) == {cells._OUTLINE_WIDTH}
    assert all(v is False for v in _values(ops, "visible"))
    assert set(_values(ops, "size")) == {cells._BADGE_SIZE}


def test_a_cell_lights_up_in_every_session_at_once():
    """The point of the page: one click, every panel responds."""
    _sheet, index = cells._sheet(_fov())
    for panel in ("0", "1", "2"):
        ops = cells._restyle(index[panel], "gcid-B", True)._operations
        assert cells._HIGHLIGHT_WIDTH in _values(ops, "width")


def test_the_numbers_switch_shows_or_hides_every_badge():
    _sheet, index = cells._sheet(_fov())
    assert all(_values(cells._restyle(index["0"], None, True)._operations,
                       "visible"))
    assert not any(_values(cells._restyle(index["0"], None, False)._operations,
                           "visible"))


def test_a_selected_cell_keeps_its_number_with_the_switch_off():
    """Otherwise turning numbers off leaves the sheet with no identity cue at
    all — the thickened outline alone does not say *which* cell it is."""
    _sheet, index = cells._sheet(_fov())
    ops = cells._restyle(index["0"], "gcid-A", False)._operations
    assert sorted(_values(ops, "visible")) == [False, True]


# ── callbacks ──────────────────────────────────────────────────────────────


def test_choosing_a_fov_renders_the_sheet_and_clears_any_selection(callbacks):
    with patch.object(cells, "_load", return_value=_fov()):
        sheet, index, cell_list, header, selected = callbacks["_on_fov"](
            "fov-1", True)
    assert len(_graphs(sheet)) == 3
    assert set(index) == {"0", "1", "2"}
    assert "●●○" in _text(cell_list)
    assert "3 sessions" in _text(header)
    assert selected is None


def test_choosing_no_fov_renders_nothing_rather_than_raising(callbacks):
    sheet, index, cell_list, header, selected = callbacks["_on_fov"](None, True)
    assert index is None and selected is None
    assert _graphs(sheet) == []


def test_a_failure_loading_a_fov_is_reported_in_place(callbacks):
    with patch.object(cells, "_load", side_effect=OSError("mask unreadable")):
        sheet, index, _list, _header, _sel = callbacks["_on_fov"]("fov-1", True)
    assert "mask unreadable" in _text(sheet)
    assert index is None


def _click(callbacks, index, click, panel_idx, *, edit_on=False, mode="select",
          pickup=None, fov_id="fov-1", selected=None, show_numbers=True,
          trigger=None, n_panels=3):
    """Fire ``_on_panel_click`` for one panel click (or the undo button)."""
    clicks = [None] * n_panels
    if panel_idx is not None and click is not None:
        clicks[panel_idx] = click

    class _Ctx:
        triggered = [{"prop_id": "x", "value": click}] if click is not None else []
        triggered_id = (trigger if trigger is not None else
                        {"type": "roigbiv-cells-panel", "index": panel_idx})

    with patch.object(cells, "ctx", _Ctx):
        return callbacks["_on_panel_click"](
            clicks, None, index, edit_on, mode, pickup, fov_id, selected,
            show_numbers)


def test_clicking_an_outline_selects_the_cell_that_owns_it(callbacks):
    _sheet, index = cells._sheet(_fov())
    click = {"points": [{"customdata": [2]}]}
    result = _click(callbacks, index, click, 0)
    assert result[0] == "gcid-B"
    assert result[1:] == (cells.no_update,) * 6


def test_clicking_a_ghost_selects_the_cell_that_went_missing(callbacks):
    _sheet, index = cells._sheet(_fov())
    click = {"points": [{"customdata": [-2]}]}
    result = _click(callbacks, index, click, 2)
    assert result[0] == "gcid-B"


def test_clicking_the_background_changes_nothing(callbacks):
    _sheet, index = cells._sheet(_fov())
    result = _click(callbacks, index, {"points": [{}]}, 0)
    assert result == cells._NOOP


def test_edit_mode_off_ignores_the_mode_and_behaves_read_only(callbacks):
    """Edit controls are inert until the switch is on, regardless of what
    the (hidden) mode radio happens to be set to."""
    _sheet, index = cells._sheet(_fov())
    click = {"points": [{"customdata": [2]}]}
    result = _click(callbacks, index, click, 0, edit_on=False, mode="delete")
    assert result[0] == "gcid-B"
    assert result[1:] == (cells.no_update,) * 6


def test_clicking_a_row_selects_that_cell(callbacks):
    class _Ctx:
        triggered_id = {"type": "roigbiv-cells-row", "gcid": "gcid-A"}

    with patch.object(cells, "ctx", _Ctx):
        assert callbacks["_on_row_click"]([1, 0]) == "gcid-A"


def test_stepping_walks_the_cells_and_wraps_at_the_end(callbacks):
    def _step(button, selected):
        class _Ctx:
            triggered_id = button
        with patch.object(cells, "ctx", _Ctx), \
                patch.object(cells, "_load", return_value=_fov()):
            return callbacks["_on_step"](1, 1, selected, "fov-1")

    assert _step(cells.NEXT_ID, None) == "gcid-A"
    assert _step(cells.NEXT_ID, "gcid-A") == "gcid-B"
    assert _step(cells.NEXT_ID, "gcid-B") == "gcid-A"     # wraps
    assert _step(cells.PREV_ID, "gcid-A") == "gcid-B"     # wraps the other way


def _select(callbacks, index, selected, *, show_numbers=True, trigger=None):
    class _Ctx:
        outputs_list = [[{"id": {"type": cells.PANEL_TYPE, "index": i}}
                         for i in range(3)]]
        triggered_id = trigger if trigger is not None else cells.SELECTED_ID

    with patch.object(cells, "ctx", _Ctx), \
            patch.object(cells, "_load", return_value=_fov()):
        return callbacks["_on_select"](selected, show_numbers, "fov-1", index)


def test_selecting_renders_the_filmstrip_and_one_patch_per_panel(callbacks):
    _sheet, index = cells._sheet(_fov())
    patches, strip, cell_list, is_open = _select(callbacks, index, "gcid-B")

    assert len(patches) == 3
    assert len(_graphs(strip)) == 3, "one crop per session"
    assert "Cell #2" in _text(strip)
    assert "not detected" in _text(strip)
    assert "roigbiv-cells-row-active" in str(cell_list)
    assert is_open is True


def test_the_drawer_stays_shut_until_a_cell_is_chosen(callbacks):
    """The filmstrip covers the sheet, so it may only appear on request."""
    _sheet, index = cells._sheet(_fov())
    assert _select(callbacks, index, None)[3] is False


def test_dismissing_the_drawer_clears_the_selection(callbacks):
    """Otherwise the drawer is shut while a cell is still selected, and
    clicking that same cell writes an unchanged value that reopens nothing."""
    assert callbacks["_on_drawer_closed"](False, "gcid-A") is None


def test_the_drawer_closing_itself_does_not_loop(callbacks):
    """Clearing the selection re-asserts is_open=False; that second pass has
    to stop rather than bounce off the selection callback again."""
    assert callbacks["_on_drawer_closed"](False, None) is cells.no_update
    assert callbacks["_on_drawer_closed"](True, "gcid-A") is cells.no_update


def test_toggling_numbers_repaints_the_panels_and_nothing_else(callbacks):
    """A switch flip must not re-crop three sessions for a selection that
    did not move."""
    _sheet, index = cells._sheet(_fov())
    patches, strip, cell_list, is_open = _select(
        callbacks, index, "gcid-B", show_numbers=False,
        trigger=cells.NUMBERS_ID)

    assert len(patches) == 3
    assert strip is cells.no_update
    assert cell_list is cells.no_update
    assert is_open is cells.no_update


# ── _resolve_click ───────────────────────────────────────────────────────


def _panel_index(*, step=1.0, radius=5, centroid=None):
    return {
        "step": step,
        "radius": radius,
        "centroid": centroid or {"7": [20.0, 30.0], "8": [60.0, 60.0]},
    }


def test_resolve_click_customdata_hit_uses_the_centroid_table():
    index = _panel_index()
    # The click point is on the outline ring, not the centroid — resolution
    # must report the centroid position, not the (less precise) click point.
    point = {"customdata": [7], "y": 19.0, "x": 30.0}
    outcome, label, pos = cells._resolve_click(index, point)
    assert outcome == "hit"
    assert label == 7
    assert pos == (20.0, 30.0)


def test_resolve_click_interior_of_stamp_hit_without_customdata():
    """A click inside the disk but away from its ring carries no customdata —
    resolution must fall back to nearest-centroid-within-radius."""
    index = _panel_index()
    point = {"y": 22.0, "x": 31.0}  # 2px from label 7's centroid
    outcome, label, pos = cells._resolve_click(index, point)
    assert outcome == "hit"
    assert label == 7
    assert pos == (20.0, 30.0)


def test_resolve_click_empty_space_outside_every_radius():
    index = _panel_index()
    point = {"y": 0.0, "x": 0.0}
    outcome, label, pos = cells._resolve_click(index, point)
    assert outcome == "empty"
    assert label is None
    assert pos == (0.0, 0.0)


def test_resolve_click_zero_radius_disables_the_fallback():
    """No calibration available (radius 0) — only a direct outline hit counts."""
    index = _panel_index(radius=0)
    point = {"y": 20.0, "x": 30.0}  # exactly on the centroid, no customdata
    outcome, label, pos = cells._resolve_click(index, point)
    assert outcome == "empty"


def test_resolve_click_rescales_downsampled_points_by_step():
    """The heatmap has no x=/y= arrays, so Plotly reports points in
    downsampled pixel space — a click must be scaled back to full-res before
    it's compared against the (full-res) centroid table."""
    index = _panel_index(step=4.0, centroid={"7": [80.0, 120.0]})
    point = {"y": 20.0, "x": 30.0}  # downsampled: 20*4=80, 30*4=120
    outcome, label, pos = cells._resolve_click(index, point)
    assert outcome == "hit"
    assert label == 7
    assert pos == (80.0, 120.0)


def test_panel_figure_excludes_ghosts_from_the_centroid_table():
    """Ghosts (negative label_id) have no footprint in this session, so a
    click near one must resolve to empty space — exactly the "place here"
    gesture edit mode composes add+link from — not a false hit on a cell
    that isn't actually present here."""
    fov = _fov()
    session = fov.sessions[2]  # B is a ghost ("lost") in this session
    assert any(r.label_id < 0 for r in session.rois)

    _figure, index = cells._panel_figure(fov, session, 2, 1.0, True)
    assert all(int(label) > 0 for label in index["centroid"])


# ── _handle_edit_click — the gesture dispatcher ─────────────────────────────


class _StubReport:
    warnings: list = []


def _edit_state():
    class _Workspace:
        input_root = Path("/fake/root")

    class _State:
        workspace = _Workspace()
        registry_config = None

    return _State()


def _panel_index_for(fov, session_idx):
    _figure, index = cells._panel_figure(
        fov, fov.sessions[session_idx], session_idx, 1.0, True)
    return index


def test_select_mode_hit_selects_and_touches_nothing_else():
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 0)
    with patch.object(cells, "_tracking_is_active", return_value=False):
        result = cells._handle_edit_click(
            {"customdata": [1]}, index, "select", None, fov, _edit_state(),
            None, True)
    assert result[0] == "gcid-A"
    assert result[1:] == (cells.no_update,) * 6


def test_select_mode_empty_click_is_a_noop():
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 0)
    with patch.object(cells, "_tracking_is_active", return_value=False):
        result = cells._handle_edit_click(
            {"y": 0.0, "x": 0.0}, index, "select", None, fov, _edit_state(),
            None, True)
    assert result == cells._NOOP


def test_delete_mode_hit_calls_do_delete_and_clears_a_matching_selection():
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 0)
    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_delete") as mock_delete, \
            patch.object(cells, "_apply_and_reload",
                         return_value=(fov, _StubReport())):
        result = cells._handle_edit_click(
            {"customdata": [1]}, index, "delete", None, fov, _edit_state(),
            "gcid-A", True)
    assert mock_delete.call_args[0][1] == 1
    assert result[0] is None          # the deleted cell was selected — cleared
    assert result[2] is not None      # sheet rebuilt


def test_delete_mode_leaves_an_unrelated_selection_alone():
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 0)
    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_delete"), \
            patch.object(cells, "_apply_and_reload",
                         return_value=(fov, _StubReport())):
        result = cells._handle_edit_click(
            {"customdata": [1]}, index, "delete", None, fov, _edit_state(),
            "gcid-B", True)
    assert result[0] == "gcid-B"


def test_delete_mode_miss_is_a_noop():
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 0)
    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_delete") as mock_delete:
        result = cells._handle_edit_click(
            {"y": 0.0, "x": 0.0}, index, "delete", None, fov, _edit_state(),
            None, True)
    mock_delete.assert_not_called()
    assert result == cells._NOOP


def test_add_mode_plain_add_when_nothing_is_selected():
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 0)
    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_add", return_value=99) as mock_add, \
            patch.object(cells, "_do_link") as mock_link, \
            patch.object(cells, "_apply_and_reload",
                         return_value=(fov, _StubReport())):
        cells._handle_edit_click(
            {"y": 5.0, "x": 5.0}, index, "add", None, fov, _edit_state(),
            None, True)
    mock_add.assert_called_once()
    mock_link.assert_not_called()


def test_add_mode_place_here_composes_add_and_link():
    """A selected cell missing from this session: clicking empty space here
    both adds the centroid and links it to that cell in one gesture — the
    repair this data needs most, per the diagnosis."""
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 2)  # B is a ghost (absent) in session 2

    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_add", return_value=99) as mock_add, \
            patch.object(cells, "_do_link") as mock_link, \
            patch.object(cells, "_apply_and_reload",
                         return_value=(fov, _StubReport())):
        cells._handle_edit_click(
            {"y": 21.0, "x": 61.0}, index, "add", None, fov, _edit_state(),
            "gcid-B", True)

    mock_add.assert_called_once()
    mock_link.assert_called_once()
    members = mock_link.call_args[0][2]
    assert (STEMS[2], 99) in members


def test_add_mode_does_not_place_here_when_the_cell_is_already_present():
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 0)  # A is present here — nothing missing
    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_add", return_value=99), \
            patch.object(cells, "_do_link") as mock_link, \
            patch.object(cells, "_apply_and_reload",
                         return_value=(fov, _StubReport())):
        cells._handle_edit_click(
            {"y": 5.0, "x": 5.0}, index, "add", None, fov, _edit_state(),
            "gcid-A", True)
    mock_link.assert_not_called()


def test_add_mode_hit_is_rejected_with_a_message():
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 0)
    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_add") as mock_add:
        result = cells._handle_edit_click(
            {"customdata": [1]}, index, "add", None, fov, _edit_state(),
            None, True)
    mock_add.assert_not_called()
    assert "already a cell" in result[6]


def test_move_mode_first_click_picks_up():
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 0)
    with patch.object(cells, "_tracking_is_active", return_value=False):
        result = cells._handle_edit_click(
            {"customdata": [1]}, index, "move", None, fov, _edit_state(),
            None, True)
    assert result[1] == {"stem": STEMS[0], "label": 1, "session_id": "s0"}
    assert result[0] is cells.no_update


def test_move_mode_second_click_in_the_same_panel_moves():
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 0)
    pickup = {"stem": STEMS[0], "label": 1, "session_id": "s0"}
    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_move") as mock_move, \
            patch.object(cells, "_apply_and_reload",
                         return_value=(fov, _StubReport())):
        result = cells._handle_edit_click(
            {"y": 30.0, "x": 30.0}, index, "move", pickup, fov, _edit_state(),
            None, True)
    assert mock_move.call_args[0][1] == 1
    assert result[1] is None  # pickup cleared once the move completes


def test_move_mode_switching_panels_restarts_the_pickup_instead_of_moving():
    fov = _fov(with_output_dirs=True)
    index1 = _panel_index_for(fov, 1)
    pickup = {"stem": STEMS[0], "label": 1, "session_id": "s0"}
    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_move") as mock_move:
        result = cells._handle_edit_click(
            {"customdata": [1]}, index1, "move", pickup, fov, _edit_state(),
            None, True)
    mock_move.assert_not_called()
    assert result[1] == {"stem": STEMS[1], "label": 1, "session_id": "s1"}


def test_link_mode_two_different_cells_links_them():
    fov = _fov(with_output_dirs=True)
    index1 = _panel_index_for(fov, 1)  # session 1's label 2 is gcid-B
    pickup = {"stem": STEMS[0], "label": 1, "session_id": "s0"}  # gcid-A
    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_link") as mock_link, \
            patch.object(cells, "_do_unlink") as mock_unlink, \
            patch.object(cells, "_apply_and_reload",
                         return_value=(fov, _StubReport())):
        cells._handle_edit_click(
            {"customdata": [2]}, index1, "link", pickup, fov, _edit_state(),
            None, True)
    mock_link.assert_called_once()
    mock_unlink.assert_not_called()


def test_link_mode_second_click_on_the_same_cell_unlinks_instead():
    """Session 1's label 1 is already gcid-A, same as the pickup — clicking
    it again means "pull this one back out", not a link that would be a
    no-op merge of a cell with itself."""
    fov = _fov(with_output_dirs=True)
    index1 = _panel_index_for(fov, 1)
    pickup = {"stem": STEMS[0], "label": 1, "session_id": "s0"}  # gcid-A
    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_link") as mock_link, \
            patch.object(cells, "_do_unlink") as mock_unlink, \
            patch.object(cells, "_apply_and_reload",
                         return_value=(fov, _StubReport())):
        cells._handle_edit_click(
            {"customdata": [1]}, index1, "link", pickup, fov, _edit_state(),
            None, True)
    mock_unlink.assert_called_once()
    mock_link.assert_not_called()


def test_link_mode_clicking_the_same_member_again_cancels():
    fov = _fov(with_output_dirs=True)
    index0 = _panel_index_for(fov, 0)
    pickup = {"stem": STEMS[0], "label": 1, "session_id": "s0"}
    with patch.object(cells, "_tracking_is_active", return_value=False), \
            patch.object(cells, "_do_link") as mock_link:
        result = cells._handle_edit_click(
            {"customdata": [1]}, index0, "link", pickup, fov, _edit_state(),
            None, True)
    mock_link.assert_not_called()
    assert result[1] is None


def test_link_mode_missing_first_click_picks_up():
    fov = _fov(with_output_dirs=True)
    index0 = _panel_index_for(fov, 0)
    with patch.object(cells, "_tracking_is_active", return_value=False):
        result = cells._handle_edit_click(
            {"customdata": [1]}, index0, "link", None, fov, _edit_state(),
            None, True)
    assert result[1] == {"stem": STEMS[0], "label": 1, "session_id": "s0"}


def test_tracking_running_blocks_every_edit_with_a_message():
    fov = _fov(with_output_dirs=True)
    index = _panel_index_for(fov, 0)
    with patch.object(cells, "_tracking_is_active", return_value=True), \
            patch.object(cells, "_do_delete") as mock_delete:
        result = cells._handle_edit_click(
            {"customdata": [1]}, index, "delete", None, fov, _edit_state(),
            None, True)
    mock_delete.assert_not_called()
    assert "tracking is running" in result[6]


def test_add_mode_end_to_end_writes_a_real_centroid_and_observation(tmp_path):
    """The mocked dispatch tests above prove the branching; this one proves
    the real pieces (op log, apply_tracking_edits, the store) actually wire
    together the way the mocks assumed."""
    import json
    import uuid
    from datetime import date, datetime, timezone

    import tifffile

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
    from roigbiv.pipeline.centroid_masks import write_merged_masks
    from roigbiv.pipeline.types import PipelineConfig
    write_merged_masks(out_dir, PipelineConfig())  # produces merged_masks.tif

    session_id = str(uuid.uuid4())
    store.upsert_session(SessionRecord(
        session_id=session_id, fov_id=fov_id, session_date=date(2026, 1, 1),
        output_dir=str(out_dir), created_at=datetime.now(timezone.utc),
        sequence_index=0))

    fov = load_tracked_fov(fov_id, cfg=cfg)
    index = _panel_index_for(fov, 0)

    class _Workspace:
        input_root = tmp_path

    class _State:
        workspace = _Workspace()
        registry_config = cfg

    with patch.object(cells, "_tracking_is_active", return_value=False):
        result = cells._handle_edit_click(
            {"y": 40.0, "x": 40.0}, index, "add", None, fov, _State(), None, True)

    assert result[6] == "added a centroid"
    assert (out_dir / "corrections" / "centroids.jsonl").exists()

    reloaded = load_tracked_fov(fov_id, cfg=cfg)
    assert len(reloaded.cells) == 2  # the original centroid + the new one
    assert len(store.list_observations_for_session(session_id)) == 2
