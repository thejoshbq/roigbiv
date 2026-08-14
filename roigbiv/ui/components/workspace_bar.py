"""The workspace scanner, as a disclosure in the navbar.

Every page needs a resolved workspace and none of them owns one. It used to
live on the Pipeline page, which meant Track, Cells and anything new could only
say "scan a workspace on the Pipeline page first" — a dead end that named a
page rather than offering the field.

So it sits in the navbar: a summary button that reports the current workspace,
and a collapse holding the path field, the Scan button, the registry indicator
and the TIF checklist. It opens itself when no workspace is resolved, so a cold
start still lands on the field.

The one thing every page reads
------------------------------
``WORKSPACE_VERSION`` is a counter bumped on each successful scan. Pages listen
to it to refresh their own FOV dropdowns and enable their own Run buttons.
Before the split the scan callback wrote *directly* into those controls
(``roigbiv-run-btn.disabled``, the MC dropdown's options) — which only worked
while every one of them lived on the same page. A callback whose Output is not
mounted fails at runtime, and ``suppress_callback_exceptions=True`` hides that
until the callback happens to fire. The counter is what makes the pages
independent of each other.

Element ids are unchanged from the Pipeline page they came from, so the
``localStorage`` persistence keys on the TIF checklist survive the move.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from roigbiv.io import validate_tif
from roigbiv.pipeline.workspace import WorkspacePaths, resolve_workspace
from roigbiv.ui.services.app_state import current_workspace, get_app_state

WORKSPACE_VERSION = "roigbiv-workspace-version"
COLLAPSE_ID = "roigbiv-workspace-collapse"
TOGGLE_ID = "roigbiv-workspace-toggle"
OPEN_STORE_ID = "roigbiv-workspace-open"
REGISTRY_ID = "roigbiv-active-registry"

PATH_ID = "roigbiv-input-path"
SCAN_ID = "roigbiv-scan-btn"
RESULT_ID = "roigbiv-scan-result"
TIF_SELECT_ID = "roigbiv-tif-select"
TIF_SELECT_ALL_ID = "roigbiv-tif-select-all"
TIF_SINK_ID = "roigbiv-tif-select-sink"


# ── layout ─────────────────────────────────────────────────────────────────


def toggle_button() -> dbc.Button:
    """The navbar control — reports the workspace and opens the disclosure."""
    return dbc.Button(
        summary_label(current_workspace()),
        id=TOGGLE_ID, color="link", size="sm", n_clicks=0,
        className="roigbiv-workspace-toggle text-decoration-none",
        title="scan or change the workspace",
    )


def summary_label(workspace: Optional[WorkspacePaths]) -> list:
    """One line naming the workspace, or asking for one.

    Reads defensively. Dash validates the *whole* layout on every request now
    that it is built per request, so an unexpected workspace object here would
    500 every route in the app — including the image endpoints that have
    nothing to do with the navbar. A label is not worth that.
    """
    icon = html.I(className="bi bi-folder2-open me-2")
    root = getattr(workspace, "input_root", None) if workspace else None
    if root is None:
        return [icon, html.Span("no workspace — scan one")]
    n_tifs = len(getattr(workspace, "tifs", ()) or ())
    return [
        icon,
        html.Span(Path(root).name, className="fw-semibold"),
        html.Span(f" · {n_tifs} TIFs", className="text-muted"),
    ]


def collapse() -> dbc.Collapse:
    """The disclosure body — path, Scan, registry, and the TIF checklist.

    Same defensiveness as :func:`summary_label`, and for the same reason: this
    is built on every request, so it cannot be allowed to raise.
    """
    workspace = current_workspace()
    root = getattr(workspace, "input_root", None) if workspace else None
    try:
        summary = workspace_summary(workspace) if root is not None else None
    except (AttributeError, TypeError, OSError):
        summary = None
    return dbc.Collapse(
        dbc.Card(dbc.CardBody([
            dbc.InputGroup([
                dbc.Input(
                    id=PATH_ID,
                    placeholder="Path to a .tif file or a directory of stacks",
                    value=str(root) if root is not None else "",
                    type="text",
                ),
                dbc.Button("Scan", id=SCAN_ID, color="primary", n_clicks=0),
            ], className="mb-2"),
            html.Small(registry_label(workspace), id=REGISTRY_ID,
                       className="text-muted d-block mb-2",
                       title="Active registry DSN"),
            html.Div(id=RESULT_ID, children=summary),
        ]), className="roigbiv-card-accent"),
        id=COLLAPSE_ID,
        is_open=root is None,
        className="mb-3",
    )


def stores() -> html.Div:
    """State that has to exist before any page mounts.

    ``TIF_SINK_ID`` is the benign Output of the selection-sync callback, which
    fires on the checklist's *restored* value too — so it has to be in the tree
    from the first render, not only after a scan.
    """
    return html.Div([
        dcc.Store(id=WORKSPACE_VERSION, data=0),
        dcc.Store(id=OPEN_STORE_ID, storage_type="session",
                  data=current_workspace() is None),
        dcc.Store(id=TIF_SINK_ID),
    ], style={"display": "none"})


def registry_label(workspace: Optional[WorkspacePaths]) -> str:
    db_path = getattr(workspace, "db_path", None) if workspace else None
    if db_path is None:
        return "registry: scan a workspace to begin"
    return f"registry: {db_path}"


def workspace_summary(workspace: WorkspacePaths) -> dbc.Card:
    """The discovered TIFs, as the checklist deciding which ones a run touches.

    All selected by default; the "Select all" master toggles the set. Living
    inside ``RESULT_ID`` means a re-scan rebuilds it, and
    ``AppState.set_workspace`` resets the stored selection to all — so no
    separate seeding output is needed on the scan callback.
    """
    options, values = tif_options_and_values(workspace)
    # Persisted against workspace identity so the choice survives a reload but
    # never bleeds a stale selection onto a different workspace.
    ws_key = str(workspace.input_root)
    return dbc.Card(dbc.CardBody([
        html.H6("Workspace resolved", className="mb-2"),
        html.Div([
            _row("Output", str(workspace.output_root)),
            _row("TIFs", f"{len(workspace.tifs)} discovered"),
        ], className="mb-2"),
        html.Small("Select which detected TIF stacks to run.",
                   className="text-muted d-block mb-2"),
        dbc.Checklist(
            id=TIF_SELECT_ALL_ID,
            options=[{"label": "Select all", "value": "all"}],
            value=["all"] if values else [],
            className="fw-bold mb-1",
            persistence=ws_key, persistence_type="local",
        ),
        dbc.Checklist(
            id=TIF_SELECT_ID,
            options=options,
            value=list(values),
            className="ms-3 mb-0",
            persistence=ws_key, persistence_type="local",
        ),
    ]), className="mt-2")


def _row(label: str, value: str) -> html.Div:
    return html.Div([
        html.Span(label, className="text-muted me-2"),
        html.Span(value, className="roigbiv-muted-code"),
    ], className="small")


def tif_options_and_values(workspace):
    """Build the TIF-selection checklist ``(options, all_values)``.

    One option per ``workspace.tifs`` entry; ``value`` is ``str(tif)`` (the
    resolved path already stored in the workspace — stable and unique, so the
    run can map a selection back to ``Path`` objects). Each label carries the
    validity tick + name + shape so the checklist doubles as the discovery
    summary.
    """
    options: list[dict] = []
    values: list[str] = []
    if workspace is None:
        return options, values
    for tif in workspace.tifs:
        value = str(tif)
        values.append(value)
        try:
            _, shape = validate_tif(tif)
            label = html.Span([
                html.Span("OK ", className="text-success fw-bold"),
                html.Span(tif.name, className="me-2"),
                html.Span(f"{shape[0]}×{shape[1]}×{shape[2]}",
                          className="text-muted small"),
            ])
        except ValueError as exc:
            label = html.Span([
                html.Span("! ", className="text-danger fw-bold"),
                html.Span(tif.name, className="me-2"),
                html.Span(str(exc), className="text-danger small"),
            ])
        options.append({"label": label, "value": value})
    return options, values


def selected_run_paths(workspace, selected):
    """Map the stored selection (path strings, or ``None`` = all) to the ordered
    ``Path`` subset of ``workspace.tifs`` to run."""
    return [t for t in workspace.tifs
            if selected is None or str(t) in selected]


def sync_select_all_values(trigger, master_value, child_value, all_values):
    """Pure decision core for the Select-all ↔ checklist sync.

    Returns ``(child_value_out, master_value_out)``; either element may be
    :data:`no_update` to leave that control untouched. Breaks the master/child
    feedback loop: a master *uncheck* only clears the children when they are
    *currently* all-selected (a genuine user toggle) — not when it is the
    programmatic echo of a partial child selection that just drove the master
    to empty.
    """
    child_set = set(child_value or [])
    full = set(all_values)
    if trigger == TIF_SELECT_ALL_ID:
        checked = bool(master_value and "all" in master_value)
        if checked:
            return (no_update if child_set == full else list(all_values)), no_update
        if all_values and child_set == full:
            return [], no_update
        return no_update, no_update
    # Child changed → reflect all-or-not in the master, but only if it differs
    # (else the master update would re-trigger this callback needlessly).
    desired = ["all"] if (all_values and child_set == full) else []
    if list(master_value or []) == desired:
        return no_update, no_update
    return no_update, desired


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    """Wire the workspace bar. Called **once** from ``build_app``, not per page.

    The bar lives outside the routed page container, so registering it per page
    would mean duplicate Outputs on the same ids.
    """

    @app.callback(
        Output(RESULT_ID, "children"),
        Output(REGISTRY_ID, "children"),
        Output(TOGGLE_ID, "children"),
        Output(WORKSPACE_VERSION, "data"),
        Output(OPEN_STORE_ID, "data", allow_duplicate=True),
        Input(SCAN_ID, "n_clicks"),
        State(PATH_ID, "value"),
        State(WORKSPACE_VERSION, "data"),
        prevent_initial_call=True,
    )
    def _on_scan(_n: int, path: Optional[str], version):
        # A failed scan must not bump the version: pages would refresh their
        # controls against the workspace they already had, and the collapse
        # would close over an error the user never got to read.
        state = get_app_state()
        if not path:
            return (dbc.Alert("Enter a path first.", color="warning"),
                    no_update, no_update, no_update, True)
        try:
            workspace = resolve_workspace(Path(path))
        except FileNotFoundError as exc:
            return (dbc.Alert(str(exc), color="danger"),
                    no_update, no_update, no_update, True)
        state.set_workspace(workspace)
        return (
            workspace_summary(workspace),
            registry_label(workspace),
            summary_label(workspace),
            int(version or 0) + 1,
            False,
        )

    @app.callback(
        Output(OPEN_STORE_ID, "data"),
        Input(TOGGLE_ID, "n_clicks"),
        State(OPEN_STORE_ID, "data"),
        prevent_initial_call=True,
    )
    def _on_toggle(_n: int, is_open):
        return not bool(is_open)

    @app.callback(
        Output(COLLAPSE_ID, "is_open"),
        Input(OPEN_STORE_ID, "data"),
    )
    def _apply_open(is_open):
        # The store is the only answer to open/closed, so a remount (page
        # navigation zeroes every n_clicks) cannot invert a deliberate choice.
        return bool(is_open)

    @app.callback(
        Output(TIF_SELECT_ID, "value", allow_duplicate=True),
        Output(TIF_SELECT_ALL_ID, "value", allow_duplicate=True),
        Input(TIF_SELECT_ALL_ID, "value"),
        Input(TIF_SELECT_ID, "value"),
        State(TIF_SELECT_ID, "options"),
        prevent_initial_call=True,
    )
    def _on_select_all(master_value, child_value, options):
        # Single combined callback (master + child as inputs) keyed on the
        # trigger id: avoids the destructive feedback loop a two-callback
        # master/child pair hits when the programmatic master update echoes
        # back. Pure logic in sync_select_all_values.
        all_values = [opt["value"] for opt in (options or [])]
        return sync_select_all_values(dash.ctx.triggered_id, master_value,
                                      child_value, all_values)

    @app.callback(
        Output(TIF_SINK_ID, "data"),
        Input(TIF_SELECT_ID, "value"),
        prevent_initial_call=False,
    )
    def _sync_selected_tifs(value):
        # Mirror the checklist into server-side AppState so each page's run path
        # reads it without a new callback State. Fires on the *restored* value
        # too (persistence remounts the checklist on nav/reload), keeping
        # AppState in step with what is displayed.
        if get_app_state().workspace is None:
            return no_update
        get_app_state().set_selected_tifs(value or [])
        return len(value or [])
