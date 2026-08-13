"""Cells page — the same neuron, seen across every session of one FOV.

Tracking already assigns each cell a ``global_cell_id`` that survives from
session to session, but until now that only ever surfaced as a count: *13
matched*. A count cannot be checked. This page shows **which** 13, and lets a
researcher decide whether to believe any one of them.

Layout
------
1. A **contact sheet** — one panel per session, in timeline order, each showing
   that session's own mean projection with its ROIs outlined. Sessions are
   *not* warped into a shared frame: ROICaT computes an alignment transform
   during matching and discards it, so nothing on disk could place two
   sessions in one coordinate system without re-running the matcher. Separate
   frames are the honest rendering, and cross-session identity is carried by
   annotation instead. Any panel can be promoted to focus by clicking its
   header.
2. A **filmstrip** (:mod:`roigbiv.ui.components.cell_strip`) — the selected
   cell cropped out of every session at one fixed scale. This is the evidence;
   the sheet is only the index into it.

Color carries outcome, not identity: matched / new here / not detected, three
hues total. Identity is revealed on demand — click any cell and it thickens
and shows its ``#N`` in every panel at once. Coloring 40-odd cells by hashed
hue would put the whole answer on screen permanently and make none of it
readable.

Where the sheet actually lives
------------------------------
Not here. The panels are OpenSeadragon viewers built by
``assets/cells_sheet.js`` against :mod:`roigbiv.ui.routes.cells_api`; this
module contributes an empty mount point and the state that drives it. The
previous Plotly implementation shipped each projection as a heatmap — ~1.9 MB
of JSON per panel, re-sent on every edit, because applying one meant rebuilding
the figure and therefore resetting the zoom.

What stays server-rendered is what is cheap and small: the FOV picker, the
header, the cell rail, and the filmstrip drawer.

Editing
-------
The **Edit** switch turns pointer gestures into corrections. There are no
modes: click selects, drag moves, right-click deletes, clicking empty
background adds, and shift-click links the clicked cell to the selected one
(or unlinks it, when it is already part of that cell). Ctrl-clicking a ghost —
the dashed outline of a cell this session never detected — asserts that the
cell *is* here, without first having to select it. Ctrl+Z undoes.

Edits apply instantly and are additive — they append to a per-session centroid
log and a per-FOV correspondence log, replayed by
:func:`roigbiv.registry.cell_edits.apply_tracking_edits` rather than mutating
pipeline output in place. The semantics live in
:mod:`roigbiv.ui.services.cell_edit_ops`, which is also what the browser posts
to; this module never decides what a gesture means.
"""
from __future__ import annotations

from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from roigbiv.ui.components.cell_strip import cell_strip
from roigbiv.ui.components.errors import user_error
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.colors import MATCH_STATUS_LABELS, MATCH_STATUS_PALETTE
from roigbiv.ui.services.registry_service import list_fovs
from roigbiv.ui.services.tracked_cells import TrackedFOV, load_tracked_fov_cached

FOV_ID = "roigbiv-cells-fov"
SELECTED_ID = "roigbiv-cells-selected"
SHEET_ID = "roigbiv-cells-sheet"
STRIP_ID = "roigbiv-cells-strip"
LIST_ID = "roigbiv-cells-list"
HEADER_ID = "roigbiv-cells-header"
PREV_ID = "roigbiv-cells-prev"
NEXT_ID = "roigbiv-cells-next"
NUMBERS_ID = "roigbiv-cells-numbers"
DRAWER_ID = "roigbiv-cells-drawer"
RAIL_TOGGLE_ID = "roigbiv-cells-rail-toggle"
RAIL_TAB_ID = "roigbiv-cells-rail-tab"
RAIL_COL_ID = "roigbiv-cells-rail-col"
RAIL_STATE_ID = "roigbiv-cells-rail-state"
SHEET_COL_ID = "roigbiv-cells-sheet-col"
BODY_ID = "roigbiv-cells-body"

# Edit mode — see the module docstring's "Editing" section.
EDIT_ID = "roigbiv-cells-edit"
EDIT_ROW_ID = "roigbiv-cells-edit-row"
UNDO_ID = "roigbiv-cells-undo"
EDIT_MSG_ID = "roigbiv-cells-edit-msg"

# The one store the browser-side sheet reads. Written by a clientside callback
# that folds together every control the sheet has to react to, so the sheet has
# a single entry point instead of one listener per switch.
VIEW_ID = "roigbiv-cells-view"


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    return html.Div([
        dcc.Store(id=SELECTED_ID),
        dcc.Store(id=VIEW_ID),
        # Persisted: the rail is furniture, and a researcher who hides it to get
        # the width back should not have to hide it again after every reload.
        dcc.Store(id=RAIL_STATE_ID, storage_type="local", data=False),
        _toolbar(),
        _edit_row(),
        html.Div(id=HEADER_ID, className="text-muted small mb-2"),
        # Flex rather than a Bootstrap row: the collapsed state needs the edge
        # tab to sit *beside* a full-width sheet, which a col-md-12 cannot do
        # without wrapping to a second row.
        html.Div([
            # Filled by assets/cells_sheet.js — see the module docstring.
            html.Div(html.Div(id=SHEET_ID), id=SHEET_COL_ID,
                     className="roigbiv-cells-sheet-col"),
            html.Div(_rail(), id=RAIL_COL_ID, className="roigbiv-cells-rail-col"),
            _rail_tab(),
        ], id=BODY_ID, className="roigbiv-cells-body"),
        _drawer(),
    ])


def _toolbar() -> dbc.Row:
    """One compact row — every control that must be reachable without a selection.

    Prev/next live here rather than in the drawer: the drawer only opens once a
    cell is selected, so controls inside it could never make the first
    selection.
    """
    return dbc.Row([
        dbc.Col(_fov_picker(), md=4),
        dbc.Col(dbc.Switch(id=NUMBERS_ID, label="numbers", value=True,
                           className="mb-0 mt-3"), width="auto"),
        dbc.Col(dbc.ButtonGroup([
            dbc.Button("◀", id=PREV_ID, size="sm", color="secondary",
                       outline=True, title="previous cell"),
            dbc.Button("▶", id=NEXT_ID, size="sm", color="secondary",
                       outline=True, title="next cell"),
        ], className="mt-3"), width="auto"),
        dbc.Col(dbc.Switch(id=EDIT_ID, label="edit", value=False,
                           className="mb-0 mt-3"), width="auto"),
        dbc.Col(_legend(), className="text-md-end"),
    ], className="align-items-center g-2")


def _edit_row() -> html.Div:
    """Undo, the last gesture's outcome, and the gesture reference.

    Hidden (``d-none``) rather than left unmounted while the Edit switch is
    off, so ``_on_edit_toggle`` only has to flip a className.

    The cheat sheet is permanent rather than a tooltip: these are direct
    manipulations with no on-screen controls to discover them from, and a
    researcher who edits once a fortnight should not have to remember that
    shift-click is the link.
    """
    return html.Div([
        dbc.Button("Undo last", id=UNDO_ID, size="sm", color="secondary",
                   outline=True, className="me-3"),
        html.Span(id=EDIT_MSG_ID, className="text-muted small me-3"),
        html.Span(
            "drag to move · right-click to delete · click empty space to add "
            "· ctrl-click a dashed outline to confirm the cell is there "
            "· shift-click to link to the selected cell · Ctrl+Z to undo",
            className="text-muted small font-monospace"),
    ], id=EDIT_ROW_ID,
        className="d-none align-items-center g-2 mb-2 flex-wrap")


def _rail() -> html.Div:
    """The cell list, with its own collapse control in its own header.

    The control used to be a ``cells ▸`` button sitting fifth in the toolbar
    between the step buttons and the Edit switch, which is nowhere near the
    thing it hides and reads as *go to cells* rather than *hide this*.
    """
    return html.Div([
        html.Div([
            html.Span("cells", className="small text-muted"),
            # ›, because the rail is on the right and collapsing pushes it that
            # way. The tab that brings it back points the other way.
            dbc.Button("›", id=RAIL_TOGGLE_ID, size="sm", color="link",
                       n_clicks=0, className="ms-auto py-0 px-2 lh-1",
                       title="hide the cell list  ["),
        ], className="d-flex align-items-center mb-1"),
        html.Div("Numbers are for reading this page only — they follow the "
                 "session order and are not the cell's registry id.",
                 className="text-muted small mb-2"),
        html.Div(id=LIST_ID, className="roigbiv-cells-list"),
    ])


def _rail_tab() -> html.Div:
    """The way back, once the rail is hidden.

    Collapsing used to remove the rail outright, leaving the toolbar button as
    the only route back — the same button that was hard enough to find on the
    way out. This one is always where the rail was.
    """
    return html.Div(
        dbc.Button("‹ cells", id=RAIL_TAB_ID, size="sm", color="link",
                   n_clicks=0, className="p-1 lh-1",
                   title="show the cell list  ["),
        id=RAIL_TAB_ID + "-wrap", className="roigbiv-cells-rail-tab")


def _drawer() -> dbc.Offcanvas:
    """The filmstrip, raised from the bottom when a cell is selected.

    ``backdrop=False`` deliberately: the sheet stays live underneath, so
    picking a different cell just re-fills the drawer instead of forcing a
    dismiss-then-click round trip.
    """
    return dbc.Offcanvas(
        html.Div(id=STRIP_ID),
        id=DRAWER_ID,
        placement="bottom",
        backdrop=False,
        scrollable=True,
        is_open=False,
        className="roigbiv-cells-drawer",
    )


def _fov_picker() -> html.Div:
    state = get_app_state()
    if state.workspace is None:
        return dbc.Alert("Scan a workspace on the Pipeline page first.",
                         color="secondary", className="mb-0")
    try:
        rows = [r for r in list_fovs(state.registry_config) if r.n_sessions]
    except Exception as exc:  # noqa: BLE001 — any store failure is the user's
        return user_error(exc, "listing tracked FOVs")
    if not rows:
        return dbc.Alert(
            "No tracked FOVs in this workspace yet — run tracking on the "
            "Track page first.", color="secondary", className="mb-0")

    options = [{
        "label": f"{r.animal_id or '?'} / {r.region or '?'} · "
                 f"{r.n_sessions} session(s) · {r.fov_id[:8]}",
        "value": r.fov_id,
    } for r in rows]
    return html.Div([
        dbc.Label("Field of view", className="small mb-1"),
        dcc.Dropdown(id=FOV_ID, options=options, value=options[0]["value"],
                     clearable=False),
    ])


def _legend() -> html.Div:
    swatches = [
        html.Span([
            html.Span(className="roigbiv-cells-swatch",
                      style={"backgroundColor": MATCH_STATUS_PALETTE[key]}),
            html.Span(MATCH_STATUS_LABELS[key], className="small text-muted"),
        ], className="me-3")
        for key in ("matched", "new", "lost")
    ]
    return html.Div(swatches, className="d-inline-flex flex-wrap mt-3")


# ── cell list ──────────────────────────────────────────────────────────────


def _cell_list(fov: TrackedFOV, selected: Optional[str]) -> html.Div:
    if not fov.cells:
        return html.Div("No tracked cells for this FOV.",
                        className="text-muted small")
    rows = [
        html.Tr([
            html.Td(f"#{cell.index}", className="roigbiv-track-seq"),
            html.Td("".join("●" if p else "○" for p in cell.present),
                    className="font-monospace"),
            html.Td(", ".join(cell.anomalies) or "—", className="small"),
        ],
            id={"type": "roigbiv-cells-row", "gcid": cell.global_cell_id},
            className=("roigbiv-cells-row"
                       + (" roigbiv-cells-row-active"
                          if cell.global_cell_id == selected else "")),
            n_clicks=0,
        )
        for cell in fov.cells
    ]
    return dbc.Table([html.Tbody(rows)], size="sm", bordered=False,
                     className="mb-0")


def _header(fov: TrackedFOV) -> list:
    if not fov.sessions:
        return []
    out: list = [html.Span(
        f"{len(fov.sessions)} sessions · {len(fov.cells)} cells · "
        f"{fov.n_complete} seen throughout")]
    if not fov.ordering_is_confirmed:
        out.append(dbc.Badge("session order not confirmed", color="warning",
                             className="ms-2"))
    return out


def _strip_header(fov: TrackedFOV, selected: Optional[str]) -> html.Div:
    cell = fov.cell_by_gcid(selected)
    if cell is None:
        return html.Div()
    return html.Div([
        html.Span(f"Cell #{cell.index}", className="fw-bold me-2"),
        html.Span("".join("●" if p else "○" for p in cell.present),
                  className="font-monospace me-2"),
        html.Span(", ".join(cell.anomalies), className="small text-warning me-3"),
        html.Span(cell.global_cell_id, className="font-monospace small text-muted"),
    ], className="mb-2 d-flex align-items-center flex-wrap")


def _load(fov_id: str) -> TrackedFOV:
    state = get_app_state()
    return load_tracked_fov_cached(fov_id, cfg=state.registry_config)


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:

    @app.callback(
        Output(EDIT_ROW_ID, "className"),
        Input(EDIT_ID, "value"),
    )
    def _on_edit_toggle(edit_on: Optional[bool]):
        base = "align-items-center g-2 mb-2 flex-wrap"
        return base if edit_on else f"{base} d-none"

    @app.callback(
        Output(LIST_ID, "children"),
        Output(HEADER_ID, "children"),
        Output(SELECTED_ID, "data"),
        Input(FOV_ID, "value"),
    )
    def _on_fov(fov_id: Optional[str]):
        """The server-rendered half of a FOV change.

        The sheet itself is not here: the clientside callback below sees the
        same FOV_ID and fetches its own geometry, so the panels never travel
        through a Dash payload.
        """
        if not fov_id:
            return html.Div(), [], None
        try:
            fov = _load(fov_id)
        except Exception as exc:  # noqa: BLE001 — store or disk, both the user's
            return (html.Div(),
                    user_error(exc, "loading this FOV's tracked cells"), None)
        return _cell_list(fov, None), _header(fov), None

    @app.callback(
        Output(STRIP_ID, "children"),
        Output(LIST_ID, "children", allow_duplicate=True),
        Output(DRAWER_ID, "is_open"),
        Input(SELECTED_ID, "data"),
        State(FOV_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_select(selected: Optional[str], fov_id: Optional[str]):
        """The filmstrip and the rail follow the selection, wherever it came from.

        "Wherever" includes the browser: a click on a panel writes ``SELECTED_ID``
        through ``dash_clientside.set_props``, which lands here exactly as a rail
        click does.
        """
        if not fov_id:
            return html.Div(), no_update, False
        try:
            fov = _load(fov_id)
        except Exception as exc:  # noqa: BLE001
            return (user_error(exc, "loading this FOV's tracked cells"),
                    no_update, False)

        cell = fov.cell_by_gcid(selected)
        strip = html.Div([_strip_header(fov, selected), cell_strip(fov, cell)])
        return strip, _cell_list(fov, selected), cell is not None

    @app.callback(
        Output(SELECTED_ID, "data", allow_duplicate=True),
        Input({"type": "roigbiv-cells-row", "gcid": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_row_click(clicks):
        if not any(clicks or []):
            return no_update
        return ctx.triggered_id["gcid"]

    @app.callback(
        Output(SELECTED_ID, "data", allow_duplicate=True),
        Input(PREV_ID, "n_clicks"),
        Input(NEXT_ID, "n_clicks"),
        State(SELECTED_ID, "data"),
        State(FOV_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_step(_prev, _next, selected: Optional[str], fov_id: Optional[str]):
        """Walk the cell list in display order, wrapping at both ends."""
        if not fov_id:
            return no_update
        try:
            cells = _load(fov_id).cells
        except Exception:  # noqa: BLE001 — the header already reported it
            return no_update
        if not cells:
            return no_update

        gcids = [c.global_cell_id for c in cells]
        step = -1 if ctx.triggered_id == PREV_ID else 1
        if selected not in gcids:
            return gcids[0] if step > 0 else gcids[-1]
        return gcids[(gcids.index(selected) + step) % len(gcids)]

    @app.callback(
        Output(SELECTED_ID, "data", allow_duplicate=True),
        Input(DRAWER_ID, "is_open"),
        State(SELECTED_ID, "data"),
        prevent_initial_call=True,
    )
    def _on_drawer_closed(is_open: bool, selected: Optional[str]):
        """Dismissing the drawer clears the selection.

        Otherwise the two disagree: the drawer is shut but a cell is still
        selected, so clicking that same cell again writes an unchanged value
        and nothing reopens. Keeping "drawer open" and "cell selected" as one
        state makes every click do what it looks like it does.
        """
        if is_open or selected is None:
            return no_update
        return None

    _register_clientside(app)


# ── clientside ─────────────────────────────────────────────────────────────


def _register_clientside(app: dash.Dash) -> None:
    """The three handoffs to ``assets/cells_sheet.js``.

    All three are browser-side because none of them has a server-side answer:
    the sheet's state lives in OpenSeadragon viewers that Dash does not own,
    and routing any of this through a callback would mean tearing those viewers
    down — which is exactly the zoom loss this page was rebuilt to fix.
    """
    # Every control the sheet reacts to, folded into one render call.
    app.clientside_callback(
        """
        function(fovId, selected, showNumbers, editOn) {
            const config = {
                fov_id: fovId || null,
                selected_gcid: selected || null,
                show_numbers: showNumbers !== false,
                edit_on: !!editOn,
            };
            // The first render fires as the page mounts, which can beat the
            // asset that answers it; without the retry the sheet would stay
            // blank until some unrelated control moved.
            (function attempt(tries) {
                if (window.roigbivCells) {
                    window.roigbivCells.render(config);
                } else if (tries > 0) {
                    setTimeout(function() { attempt(tries - 1); }, 50);
                }
            })(20);
            return config;
        }
        """,
        Output(VIEW_ID, "data"),
        Input(FOV_ID, "value"),
        Input(SELECTED_ID, "data"),
        Input(NUMBERS_ID, "value"),
        Input(EDIT_ID, "value"),
    )

    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks && window.roigbivCells) {
                window.roigbivCells.undo();
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output(VIEW_ID, "data", allow_duplicate=True),
        Input(UNDO_ID, "n_clicks"),
        prevent_initial_call=True,
    )

    # The rail collapse, in two halves. It used to be one callback deriving
    # open/closed from `n_clicks % 2`, which cannot survive a reload: a restored
    # store and a zeroed click count are two different answers to the same
    # question. The store is now the only answer, and the buttons just flip it.
    app.clientside_callback(
        """
        function(hide, show, collapsed) { return !collapsed; }
        """,
        Output(RAIL_STATE_ID, "data"),
        Input(RAIL_TOGGLE_ID, "n_clicks"),
        Input(RAIL_TAB_ID, "n_clicks"),
        State(RAIL_STATE_ID, "data"),
        prevent_initial_call=True,
    )

    # One class on the wrapper drives the rail, the sheet's width and the edge
    # tab, so they cannot disagree. No prevent_initial_call: this is what paints
    # the persisted state on mount. OpenSeadragon only re-reads its container
    # when asked, and the sheet just changed width.
    app.clientside_callback(
        """
        function(collapsed) {
            setTimeout(function() {
                if (window.roigbivCells) { window.roigbivCells.resize(); }
            }, 80);
            return "roigbiv-cells-body" + (collapsed ? " is-collapsed" : "");
        }
        """,
        Output(BODY_ID, "className"),
        Input(RAIL_STATE_ID, "data"),
    )
