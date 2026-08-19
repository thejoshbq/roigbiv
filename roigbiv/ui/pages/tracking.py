"""Tracking — order the sessions, register them, then check the result.

Two halves of one job, which is why they are one page. The top half decides
*which sessions, in what order*; the bottom half is where you find out whether
the answer is believable, and correct it if it is not. Split across two pages
they were a loop with a navigation step in the middle: run tracking, go
elsewhere, look, come back, re-run.

Setup — session order
---------------------
The pipeline can mark every FOV in a workspace with centroids, but it cannot
know what order those sessions happened in. Filename dates are unreliable here:
six-digit groups are ambiguous between the lab's two conventions
(:mod:`roigbiv.registry.filename`), and sessions routinely share a date — the
reference prism workspace records ``pre-005`` / ``beh-006`` / ``post-007`` on
one day, which no date can order.

So a human orders them, and that order is authoritative. It is not cosmetic:
within a ROICaT cluster the earliest-registered observation owns the
``global_cell_id``, so registration order *is* cell-identity seniority, and it
is what makes "arrived late" and "dropped out" mean anything.

The setup section is a collapse, opening itself when this workspace has nothing
tracked yet and staying shut once it does — ordering is done once per workspace
and reviewing is done every time.

Review — the contact sheet
--------------------------
Tracking assigns each cell a ``global_cell_id`` that survives from session to
session. The contact sheet shows all tracked cells overlaid on each session's
mean projection in timeline order, with color indicating match status: matched,
new, or not detected. Click any cell to highlight it across all panels and see
its index.

Sessions are not warped into a shared frame — ROICaT computes an alignment
transform during matching and discards it, so nothing on disk could place two
sessions in one coordinate system without re-running the matcher. Separate
frames are the honest rendering, and cross-session identity is carried by
annotation.

The **boundaries** switch is a display choice, not a data source. Off (the
default), each panel outlines ADR-0003's canonical disk stamps — what the
registry actually matched on. On, it draws ADR-0005's seeded segmentation
instead — closer to the real soma. Both tracks carry the same label ids, so
toggling never changes which cell a click resolves to.

Where the sheet actually lives
------------------------------
Not here. The panels are OpenSeadragon viewers built by
``assets/cells_sheet.js`` against :mod:`roigbiv.ui.routes.cells_api`; this
module contributes an empty mount point and the state that drives it. The
previous Plotly implementation shipped each projection as a heatmap — ~1.9 MB
of JSON per panel, re-sent on every edit, because applying one meant rebuilding
the figure and therefore resetting the zoom.

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
pipeline output in place. That replay also redraws each session's seeded
boundaries, so the outlines on screen can never contradict the centroids they
were drawn from. The semantics live in
:mod:`roigbiv.ui.services.cell_edit_ops`, which is also what the browser posts
to; this module never decides what a gesture means.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from roigbiv.pipeline.session_order import (
    SessionOrderEntry,
    discover_trackable_stems,
    reorder,
    resolve_order,
    save_order,
)
from roigbiv.ui.components import workspace_bar
from roigbiv.ui.components.errors import user_error
from roigbiv.ui.components.forms import HELP_TEXT, button_tooltip, help_icon
from roigbiv.ui.components.log_stream import log_stream
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.colors import MATCH_STATUS_LABELS, MATCH_STATUS_PALETTE
from roigbiv.ui.services.registry_service import list_fovs, workspace_anomalies
from roigbiv.ui.services.tracked_cells import TrackedFOV, load_tracked_fov_cached
from roigbiv.ui.services.tracking_runner import (
    TrackingSnapshot,
    get_tracking_runner,
)

# ── setup half (was pages/track.py) ────────────────────────────────────────
LIST_ID = "roigbiv-track-list"
ORDER_SINK_ID = "roigbiv-track-order-sink"
SETUP_COLLAPSE_ID = "roigbiv-track-setup-collapse"
SETUP_TOGGLE_ID = "roigbiv-track-setup-toggle"
SETUP_STORE_ID = "roigbiv-track-setup-open"
SETUP_SUMMARY_ID = "roigbiv-track-setup-summary"

# ── review half (was pages/cells.py) ───────────────────────────────────────
FOV_ID = "roigbiv-cells-fov"
SELECTED_ID = "roigbiv-cells-selected"
SHEET_ID = "roigbiv-cells-sheet"
CELL_LIST_ID = "roigbiv-cells-list"
HEADER_ID = "roigbiv-cells-header"
PREV_ID = "roigbiv-cells-prev"
NEXT_ID = "roigbiv-cells-next"
NUMBERS_ID = "roigbiv-cells-numbers"
BOUNDARIES_ID = "roigbiv-cells-boundaries"
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

_DATE_BADGE = {
    "ambiguous": ("ambiguous date", "warning"),
    "unparsed": ("no date", "danger"),
}

# How the registry's four outcomes read at a glance. `review` is not a failure
# — it means the posterior landed between the accept and review thresholds and
# a human has to say whether this is the same FOV.
_DECISION_COLOR = {
    "hash_match": "success",
    "auto_match": "success",
    "new_fov": "info",
    "review": "warning",
}


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    snap = get_tracking_runner().snapshot()
    tracked = _tracked_fov_rows()
    return html.Div([
        dcc.Interval(id="roigbiv-track-tick", interval=1500,
                     disabled=not snap.active),
        # The drag script publishes the reordered stems here as a JSON array.
        # Hidden rather than removed: Dash only listens to inputs in the tree.
        dcc.Input(id=ORDER_SINK_ID, type="text", value="",
                  style={"display": "none"}),
        dcc.Store(id=SETUP_STORE_ID, storage_type="session",
                  data=not tracked),
        _setup_section(snap, tracked),
        html.Hr(className="roigbiv-h-line"),
        _review_section(),
    ])


def _tracked_fov_rows() -> list:
    """Tracked FOVs in this workspace, or ``[]`` when there is nothing to ask.

    Swallows store failures deliberately: this only decides whether the setup
    section starts open, and the section itself reports a broken registry far
    better than a stack trace at layout time would.
    """
    state = get_app_state()
    if state.workspace is None:
        return []
    try:
        return [r for r in list_fovs(state.registry_config) if r.n_sessions]
    except Exception:  # noqa: BLE001 — see docstring
        return []


# ── setup half ─────────────────────────────────────────────────────────────


def _setup_section(snap: TrackingSnapshot, tracked: list) -> html.Div:
    return html.Div([
        html.Div([
            dbc.Button(
                [html.I(className="bi bi-sliders me-2"),
                 html.Span("Session order & tracking run")],
                id=SETUP_TOGGLE_ID, color="link", size="sm", n_clicks=0,
                className="text-decoration-none px-0"),
            button_tooltip(SETUP_TOGGLE_ID, HELP_TEXT[SETUP_TOGGLE_ID]),
            html.Span(_setup_summary(tracked), id=SETUP_SUMMARY_ID,
                      className="text-muted small ms-3"),
        ], className="d-flex align-items-center"),
        dbc.Collapse(
            dbc.Row([
                dbc.Col([
                    html.Div(id="roigbiv-track-list-wrap",
                             children=_session_list()),
                    html.Div([
                        dbc.Button("Save order", id="roigbiv-track-save-btn",
                                   color="secondary", className="me-2"),
                        dbc.Button("Reset to filename dates",
                                   id="roigbiv-track-reset-btn",
                                   color="link", className="me-2"),
                        dbc.Button("Run tracking", id="roigbiv-track-run-btn",
                                   color="primary"),
                    ], className="mt-3"),
                    html.Div(id="roigbiv-track-save-status", className="mt-2"),
                ], md=6),
                dbc.Col([
                    html.H6("Run status"),
                    html.Div(id="roigbiv-track-status", children=_status(snap)),
                    html.Div(id="roigbiv-track-results",
                             children=_results_table(snap), className="mt-2"),
                    html.Div(id="roigbiv-track-logs",
                             children=log_stream(
                                 snap.logs, empty_hint="No tracking run yet."),
                             className="mt-2"),
                    html.H6("Anomalies", className="mt-3"),
                    html.Div(id="roigbiv-track-anomalies",
                             children=_anomaly_panel(snap)),
                ], md=6),
            ], className="mt-2"),
            id=SETUP_COLLAPSE_ID,
            is_open=not tracked,
        ),
    ])


def _setup_summary(tracked: list) -> str:
    if not tracked:
        return "nothing tracked yet"
    sessions = sum(r.n_sessions for r in tracked)
    return (f"{len(tracked)} FOV(s) · {sessions} session(s) tracked")


def _entries() -> tuple[Optional[Path], list[SessionOrderEntry]]:
    """This workspace's session order, or ``(None, [])`` before a scan."""
    workspace = get_app_state().workspace
    if workspace is None:
        return None, []
    return workspace.input_root, resolve_order(
        workspace.input_root, discover_trackable_stems(workspace))


def _session_list() -> html.Div:
    root, entries = _entries()
    if root is None:
        return dbc.Alert("Scan a workspace from the navbar first.",
                         color="secondary")
    if not entries:
        return dbc.Alert("No FOVs discovered in this workspace.", color="warning")

    rows = [_session_row(e, _centroid_count(root, e.stem)) for e in entries]
    return html.Div(rows, id=LIST_ID)


def _session_row(entry: SessionOrderEntry, n_centroids: Optional[int]) -> html.Div:
    badge_spec = _DATE_BADGE.get(entry.date_source) if not entry.locked else None
    meta: list = [html.Span(entry.session_date or "date unknown",
                            className="text-muted small")]
    if badge_spec:
        label, color = badge_spec
        meta.append(dbc.Badge(label, color=color, className="ms-2"))
    if entry.locked:
        meta.append(dbc.Badge("confirmed", color="success", className="ms-2"))
    meta.append(html.Span(
        f"{n_centroids} centroids" if n_centroids is not None
        else "no centroids — run discovery first",
        className=("ms-2 small text-muted" if n_centroids is not None
                   else "ms-2 small text-warning"),
    ))

    return html.Div(
        dbc.Row([
            dbc.Col(html.Span("⠿", className="roigbiv-track-handle"), width="auto"),
            dbc.Col(html.Span(str(entry.index + 1), className="roigbiv-track-seq"),
                    width="auto"),
            dbc.Col([
                html.Div(entry.stem, className="font-monospace small"),
                html.Div(meta),
            ]),
        ], className="align-items-center g-2"),
        className="roigbiv-track-row p-2 mb-2",
        draggable="true",
        **{"data-track-stem": entry.stem},
    )


def _centroid_count(input_root: Path, stem: str) -> Optional[int]:
    path = Path(input_root) / "output" / stem / "centroids.json"
    if not path.exists():
        return None
    try:
        return len(json.loads(path.read_text()).get("centroids", []))
    except (json.JSONDecodeError, OSError):
        return None


def _stored_anomalies() -> dict:
    """What the registry already knows about this workspace's sessions."""
    state = get_app_state()
    workspace = state.workspace
    if workspace is None:
        return {}
    return workspace_anomalies(
        [Path(workspace.output_root) / stem
         for stem in discover_trackable_stems(workspace)],
        cfg=state.registry_config,
    )


def _status(snap: TrackingSnapshot) -> dbc.Alert:
    if snap.error:
        return dbc.Alert(f"Tracking failed: {snap.error}", color="danger")
    if snap.active:
        return dbc.Alert("Tracking in progress…", color="info")
    if snap.completed_at is None:
        return dbc.Alert("Idle — confirm the order, then run tracking.",
                         color="secondary")
    parts = [f"{snap.n_tracked} session(s) tracked"]
    if snap.n_skipped:
        parts.append(f"{snap.n_skipped} skipped")
    if snap.n_failed:
        parts.append(f"{snap.n_failed} failed")
    return dbc.Alert(" · ".join(parts),
                     color="warning" if snap.n_failed else "success")


def _results_table(snap: TrackingSnapshot) -> html.Div:
    """Per-session registration outcome — the same detail the CLI logs.

    Aggregate counts alone can't answer the question a researcher actually has
    after a run: *which* session matched, how confidently, and how many cells
    it lost or gained relative to the ones before it.
    """
    if not snap.results:
        return html.Div()

    blocks: list = []
    # A crashed matcher decides "new_fov" exactly like a genuinely new field of
    # view does. Without this the table below reads as a clean result.
    failures = dict.fromkeys(
        e["error"] for r in snap.results for e in (r.get("match_errors") or []))
    if failures:
        blocks.append(dbc.Alert(
            [html.Div("Cross-session matching failed — these sessions were not "
                      "compared, they errored.", className="fw-bold mb-1")]
            + [html.Div(msg, className="font-monospace small") for msg in failures],
            color="danger", className="py-2 px-3"))

    rows = [_result_row(r) for r in snap.results]
    blocks.append(dbc.Table([
        html.Thead(html.Tr([
            html.Th("#"), html.Th("session"), html.Th("cells"),
            html.Th("outcome"), html.Th("p"),
            html.Th("matched"), html.Th("new"), html.Th("missing"),
        ])),
        html.Tbody(rows),
    ], size="sm", bordered=False, responsive=True, className="mb-0"))
    return html.Div(blocks)


def _result_row(r: dict) -> html.Tr:
    stem = html.Td(r["stem"], className="font-monospace small text-truncate",
                   style={"maxWidth": "16rem"})
    seq = html.Td("" if r["sequence_index"] is None
                  else str(r["sequence_index"] + 1),
                  className="roigbiv-track-seq")

    if r.get("skipped"):
        return html.Tr([seq, stem, html.Td(r["skipped"], colSpan=6,
                                           className="small text-warning")])
    if r.get("error"):
        return html.Tr([seq, stem, html.Td(r["error"], colSpan=6,
                                           className="small text-danger")])

    cells = f"{r['n_centroids']}"
    if r.get("n_overlapping_pairs"):
        # Stamped disks that touch — the pair count makes crowding visible
        # rather than letting it quietly degrade the embeddings.
        cells += f" ({r['n_overlapping_pairs']} overlap)"
    decision = r.get("decision") or "—"
    posterior = r.get("posterior")

    return html.Tr([
        seq, stem,
        html.Td(cells, className="small"),
        html.Td(dbc.Badge(decision,
                          color=_DECISION_COLOR.get(decision, "secondary"))),
        html.Td("—" if posterior is None else f"{posterior:.2f}",
                className="font-monospace small"),
        html.Td(_count_cell(r.get("n_matched"))),
        html.Td(_count_cell(r.get("n_new"))),
        html.Td(_count_cell(r.get("n_missing"))),
    ])


def _count_cell(value) -> html.Span:
    return html.Span("—" if value is None else str(value),
                     className="font-monospace small")


def _anomaly_panel(snap: TrackingSnapshot) -> html.Div:
    """Anomalies from this run, falling back to what the registry already holds.

    Without the fallback the panel would be blank for any workspace tracked
    before this browser session or from the CLI, even though the registry can
    answer for it — the report is derived from observation rows, not run state.
    """
    reports, from_registry = snap.anomalies, False
    if reports is None:
        if snap.active:
            return html.Div("Tracking in progress…", className="text-muted")
        try:
            reports = _stored_anomalies()
        except Exception as exc:  # noqa: BLE001 — any store failure is the user's
            return user_error(exc, "reading anomalies from the registry")
        from_registry = True

    if not reports:
        return html.Div("No tracked sessions yet for this workspace.",
                        className="text-muted")

    blocks: list = []
    if from_registry:
        blocks.append(html.Div("From the registry — not this session's run.",
                               className="text-muted small mb-2"))
    for fov_id, report in reports.items():
        counts = report["counts"]
        header = [
            html.Div(f"FOV {fov_id}", className="font-monospace small"),
            html.Div(
                f"{counts['n_cells']} cells over {counts['n_sessions']} "
                f"sessions · {counts['n_complete']} seen throughout",
                className="text-muted small",
            ),
        ]
        if not report.get("ordering_is_confirmed"):
            header.append(dbc.Alert(
                "This timeline was not human-ordered, so 'late' and 'dropout' "
                "follow the filename dates.",
                color="warning", className="py-1 px-2 small mt-1"))

        summary = dbc.Row([
            dbc.Col(_counter("late", counts["late_arrival"])),
            dbc.Col(_counter("dropout", counts["dropout"])),
            dbc.Col(_counter("intermittent", counts["intermittent"])),
        ], className="my-2")

        rows = [
            html.Tr([
                html.Td(cell["global_cell_id"][:8],
                        className="font-monospace small"),
                html.Td("".join("●" if p else "○" for p in cell["present"]),
                        className="font-monospace"),
                html.Td(", ".join(cell["anomalies"]), className="small"),
            ])
            for cell in report["cells"]
        ]
        table = dbc.Table([
            html.Thead(html.Tr([html.Th("cell"), html.Th("timeline"),
                                html.Th("anomaly")])),
            html.Tbody(rows),
        ], size="sm", bordered=False, className="mb-0") if rows else html.Div(
            "No anomalies — every cell was seen in every session.",
            className="text-muted small")

        blocks.append(dbc.Card(dbc.CardBody(header + [summary, table]),
                               className="mb-3"))
    return html.Div(blocks)


def _counter(label: str, value: int) -> html.Div:
    return html.Div([
        html.Div(str(value), className="roigbiv-track-seq h5 mb-0"),
        html.Div(label, className="text-muted small"),
    ])


# ── review half ────────────────────────────────────────────────────────────


def _review_section() -> html.Div:
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
    ])


def _toolbar() -> dbc.Row:
    """One compact row — controls for FOV selection, display options, and navigation."""
    return dbc.Row([
        dbc.Col(_fov_picker(), md=4),
        dbc.Col(dbc.Switch(id=NUMBERS_ID, label="numbers", value=True,
                           className="mb-0 mt-3"), width="auto"),
        dbc.Col(dbc.Switch(id=BOUNDARIES_ID, label="boundaries", value=False,
                           className="mb-0 mt-3"), width="auto"),
        dbc.Col(dbc.ButtonGroup([
            dbc.Button("◀", id=PREV_ID, size="sm", color="secondary",
                       outline=True, title="previous cell"),
            dbc.Button("▶", id=NEXT_ID, size="sm", color="secondary",
                       outline=True, title="next cell"),
        ], className="mt-3"), width="auto"),
        dbc.Col(html.Div([
            dbc.Switch(id=EDIT_ID, label="edit", value=False,
                       className="mb-0 mt-3"),
            *help_icon(EDIT_ID, HELP_TEXT[EDIT_ID]),
        ], className="d-flex align-items-center"), width="auto"),
        dbc.Col(_legend(), className="text-md-end"),
    ], className="align-items-center g-2")


def _edit_row() -> html.Div:
    """Undo and the last gesture's outcome.

    Hidden (``d-none``) rather than left unmounted while the Edit switch is
    off, so ``_on_edit_toggle`` only has to flip a className.

    The gesture reference used to live here as a permanent cheat sheet; it's
    now the Edit switch's hover-help tooltip (``HELP_TEXT[EDIT_ID]``).
    """
    return html.Div([
        dbc.Button("Undo last", id=UNDO_ID, size="sm", color="secondary",
                   outline=True, className="me-3"),
        html.Span(id=EDIT_MSG_ID, className="text-muted small me-3"),
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
            *help_icon("roigbiv-cells-list-heading",
                       HELP_TEXT["roigbiv-cells-list-heading"]),
            # ›, because the rail is on the right and collapsing pushes it that
            # way. The tab that brings it back points the other way.
            dbc.Button("›", id=RAIL_TOGGLE_ID, size="sm", color="link",
                       n_clicks=0, className="ms-auto py-0 px-2 lh-1",
                       title="hide the cell list  ["),
        ], className="d-flex align-items-center mb-1"),
        html.Div(id=CELL_LIST_ID, className="roigbiv-cells-list"),
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
        className="roigbiv-cells-rail-tab")


def _fov_picker() -> html.Div:
    state = get_app_state()
    if state.workspace is None:
        return dbc.Alert("Scan a workspace from the navbar first.",
                         color="secondary", className="mb-0")
    try:
        rows = [r for r in list_fovs(state.registry_config) if r.n_sessions]
    except Exception as exc:  # noqa: BLE001 — any store failure is the user's
        return user_error(exc, "listing tracked FOVs")
    if not rows:
        return dbc.Alert(
            "No tracked FOVs in this workspace yet — run tracking above first.",
            color="secondary", className="mb-0")

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


def _load(fov_id: str) -> TrackedFOV:
    state = get_app_state()
    return load_tracked_fov_cached(fov_id, cfg=state.registry_config)


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    _register_setup_callbacks(app)
    _register_review_callbacks(app)
    _register_clientside(app)


def _register_setup_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output(SETUP_STORE_ID, "data"),
        Input(SETUP_TOGGLE_ID, "n_clicks"),
        State(SETUP_STORE_ID, "data"),
        prevent_initial_call=True,
    )
    def _on_setup_toggle(_n: int, is_open):
        return not bool(is_open)

    @app.callback(
        Output(SETUP_COLLAPSE_ID, "is_open"),
        Input(SETUP_STORE_ID, "data"),
    )
    def _apply_setup_open(is_open):
        # The store is the only answer, so a remount (navigation zeroes every
        # n_clicks) cannot invert a state the user set deliberately.
        return bool(is_open)

    @app.callback(
        Output("roigbiv-track-save-status", "children"),
        Output("roigbiv-track-list-wrap", "children"),
        Input("roigbiv-track-save-btn", "n_clicks"),
        State(ORDER_SINK_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_save(_n: int, raw: Optional[str]):
        root, entries = _entries()
        if root is None:
            return dbc.Alert("Scan a workspace first.", color="warning"), no_update
        try:
            stems = json.loads(raw) if raw else []
        except json.JSONDecodeError:
            stems = []
        if not stems:
            return (dbc.Alert("Nothing to save — drag a session first.",
                              color="secondary"), no_update)
        try:
            save_order(root, reorder(entries, stems))
        except OSError as exc:
            return user_error(exc, "saving session order"), no_update
        return (dbc.Alert(f"Order saved to {root / 'session_order.json'}",
                          color="success"), _session_list())

    @app.callback(
        Output("roigbiv-track-save-status", "children", allow_duplicate=True),
        Output("roigbiv-track-list-wrap", "children", allow_duplicate=True),
        Output(ORDER_SINK_ID, "value"),
        Input("roigbiv-track-reset-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_reset(_n: int):
        """Drop the saved order and re-propose from filename dates."""
        from roigbiv.pipeline.session_order import order_path, propose_order

        root, _ = _entries()
        if root is None:
            return dbc.Alert("Scan a workspace first.", color="warning"), no_update, ""
        stems = discover_trackable_stems(get_app_state().workspace)
        try:
            path = order_path(root)
            if path.exists():
                path.unlink()
            save_order(root, propose_order(stems))
        except OSError as exc:
            return user_error(exc, "resetting session order"), no_update, ""
        return (dbc.Alert("Order reset to filename dates.", color="secondary"),
                _session_list(), "")

    @app.callback(
        Output("roigbiv-track-status", "children"),
        Output("roigbiv-track-tick", "disabled"),
        Input("roigbiv-track-run-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_run(_n: int):
        state = get_app_state()
        if state.workspace is None:
            return dbc.Alert("Scan a workspace first.", color="warning"), True
        started = get_tracking_runner().start(
            state.workspace, {}, registry_config=state.registry_config)
        if started == "busy":
            return (dbc.Alert("Another run is using the GPU — try again shortly.",
                              color="warning"), True)
        if started is False:
            return dbc.Alert("A tracking run is already active.",
                             color="warning"), False
        return dbc.Alert("Tracking started…", color="info"), False

    @app.callback(
        Output("roigbiv-track-status", "children", allow_duplicate=True),
        Output("roigbiv-track-results", "children"),
        Output("roigbiv-track-logs", "children"),
        Output("roigbiv-track-anomalies", "children"),
        Output("roigbiv-track-tick", "disabled", allow_duplicate=True),
        Output(SETUP_SUMMARY_ID, "children"),
        Input("roigbiv-track-tick", "n_intervals"),
        prevent_initial_call=True,
    )
    def _on_tick(_n: int):
        snap = get_tracking_runner().snapshot()
        return (_status(snap),
                _results_table(snap),
                log_stream(snap.logs, empty_hint="No tracking run yet."),
                _anomaly_panel(snap),
                not snap.active,
                _setup_summary(_tracked_fov_rows()))

    @app.callback(
        Output("roigbiv-track-list-wrap", "children", allow_duplicate=True),
        Input(workspace_bar.WORKSPACE_VERSION, "data"),
        prevent_initial_call=True,
    )
    def _on_workspace_change(_version):
        # A scan while this page is mounted has to repopulate the session list;
        # nothing else on the page reads the workspace directly.
        return _session_list()


def _register_review_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output(EDIT_ROW_ID, "className"),
        Input(EDIT_ID, "value"),
    )
    def _on_edit_toggle(edit_on: Optional[bool]):
        base = "align-items-center g-2 mb-2 flex-wrap"
        return base if edit_on else f"{base} d-none"

    @app.callback(
        Output(CELL_LIST_ID, "children"),
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
        Output(CELL_LIST_ID, "children", allow_duplicate=True),
        Input(SELECTED_ID, "data"),
        State(FOV_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_select(selected: Optional[str], fov_id: Optional[str]):
        """Update the cell list highlighting when a cell is selected."""
        if not fov_id:
            return no_update
        try:
            fov = _load(fov_id)
        except Exception:
            return no_update
        return _cell_list(fov, selected)

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
        function(fovId, selected, showNumbers, editOn, showBoundaries) {
            const config = {
                fov_id: fovId || null,
                selected_gcid: selected || null,
                show_numbers: showNumbers !== false,
                edit_on: !!editOn,
                show_boundaries: !!showBoundaries,
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
        Input(BOUNDARIES_ID, "value"),
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
        function(hide, show, collapsed) {
            // prevent_initial_call covers the first mount, but navigating back
            // to this page remounts both buttons at zero. Without this a
            // remount would silently invert a state the user set deliberately.
            if (!hide && !show) { return window.dash_clientside.no_update; }
            return !collapsed;
        }
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
