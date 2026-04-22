"""Process page — scan a workspace, set pipeline params, run.

Flow
----
1. User pastes / types a path into the input field and clicks **Scan**.
2. Workspace summary card shows what was discovered (input / output /
   registry / TIF count + TIF list with validity ticks).
3. User sets ``fs`` + tunables in the form and clicks **Run pipeline**.
4. Background runner streams logs; interval polls render them live.
5. Per-FOV summary rows show up under the log as they complete, including
   the registry decision (``hash_match`` / ``auto_match`` / ``review`` /
   ``new_fov``) so no tab switch is required to see the full outcome.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from roigbiv.io import validate_tif
from roigbiv.pipeline.workspace import WorkspacePaths, resolve_workspace
from roigbiv.ui.components.log_stream import log_stream
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.pipeline_runner import get_pipeline_runner


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    state = get_app_state()
    workspace = state.workspace
    return html.Div([
        dcc.Interval(id="roigbiv-process-tick", interval=1500, disabled=True),
        dbc.Row([
            dbc.Col(_left_column(workspace), md=5, lg=4, className="pe-md-4"),
            dbc.Col(_right_column(),         md=7, lg=8),
        ], className="g-3"),
    ])


def _left_column(workspace: Optional[WorkspacePaths]) -> html.Div:
    return html.Div([
        html.H4("Workspace", className="mb-3"),
        dbc.InputGroup([
            dbc.Input(
                id="roigbiv-input-path",
                placeholder="Path to a .tif file or a directory of stacks",
                value=str(workspace.input_root) if workspace else "",
                type="text",
            ),
            dbc.Button("Scan", id="roigbiv-scan-btn", color="primary",
                       n_clicks=0),
        ], className="mb-3"),
        html.Div(id="roigbiv-scan-result"),
        html.H5("Pipeline parameters", className="mt-4 mb-2"),
        _params_form(),
        dbc.Button("Run pipeline", id="roigbiv-run-btn",
                   color="success", className="mt-3 w-100", n_clicks=0,
                   disabled=workspace is None),
    ])


def _params_form() -> dbc.Card:
    row = lambda *children: dbc.Row(children, className="mb-2")    # noqa: E731
    return dbc.Card(dbc.CardBody([
        row(
            dbc.Col(dbc.Label("fs (Hz)", html_for="roigbiv-param-fs"), md=6),
            dbc.Col(dbc.Input(id="roigbiv-param-fs",
                              type="number", value=7.5, step=0.5), md=6),
        ),
        row(
            dbc.Col(dbc.Label("tau (s)", html_for="roigbiv-param-tau"), md=6),
            dbc.Col(dbc.Input(id="roigbiv-param-tau",
                              type="number", value=1.0, step=0.1), md=6),
        ),
        row(
            dbc.Col(dbc.Label("k_background",
                              html_for="roigbiv-param-k"), md=6),
            dbc.Col(dbc.Input(id="roigbiv-param-k",
                              type="number", value=30, step=1), md=6),
        ),
        row(
            dbc.Col(dbc.Label("Cellpose model",
                              html_for="roigbiv-param-model"), md=6),
            dbc.Col(dbc.Input(id="roigbiv-param-model", type="text",
                              value="models/deployed/current_model"), md=6),
        ),
    ]))


def _right_column() -> html.Div:
    return html.Div([
        html.H4("Run status", className="mb-3"),
        dbc.Progress(id="roigbiv-run-progress", value=0, striped=True,
                     className="mb-3"),
        html.Div(id="roigbiv-run-banner"),
        html.Div(id="roigbiv-run-log", children=log_stream([])),
        html.Hr(),
        html.H5("Per-FOV results", className="mb-2"),
        html.Div(id="roigbiv-run-results"),
    ])


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("roigbiv-scan-result", "children"),
        Output("roigbiv-run-btn", "disabled"),
        Input("roigbiv-scan-btn", "n_clicks"),
        State("roigbiv-input-path", "value"),
        prevent_initial_call=True,
    )
    def _on_scan(_n: int, path: Optional[str]):
        state = get_app_state()
        if not path:
            return dbc.Alert("Enter a path first.", color="warning"), True
        try:
            workspace = resolve_workspace(Path(path))
        except FileNotFoundError as exc:
            return dbc.Alert(str(exc), color="danger"), True
        state.set_workspace(workspace)
        return _workspace_summary(workspace), False

    @app.callback(
        Output("roigbiv-process-tick", "disabled"),
        Output("roigbiv-run-banner", "children"),
        Input("roigbiv-run-btn", "n_clicks"),
        State("roigbiv-param-fs", "value"),
        State("roigbiv-param-tau", "value"),
        State("roigbiv-param-k", "value"),
        State("roigbiv-param-model", "value"),
        prevent_initial_call=True,
    )
    def _on_run(_n: int, fs, tau, k, model):
        state = get_app_state()
        if state.workspace is None:
            return True, dbc.Alert("Scan a workspace first.", color="warning")
        overrides = {
            "fs": float(fs or 7.5),
            "tau": float(tau or 1.0),
            "k_background": int(k or 30),
            "cellpose_model": model or "models/deployed/current_model",
        }
        runner = get_pipeline_runner()
        started = runner.start(state.workspace, overrides)
        if not started:
            return False, dbc.Alert(
                "A pipeline run is already active — wait for it to finish.",
                color="warning",
            )
        return False, dbc.Alert("Pipeline run started.",
                                color="info", className="py-2 mb-2")

    @app.callback(
        Output("roigbiv-run-log", "children"),
        Output("roigbiv-run-progress", "value"),
        Output("roigbiv-run-progress", "label"),
        Output("roigbiv-run-results", "children"),
        Output("roigbiv-process-tick", "disabled", allow_duplicate=True),
        Input("roigbiv-process-tick", "n_intervals"),
        prevent_initial_call="initial_duplicate",
    )
    def _on_tick(_n):
        runner = get_pipeline_runner()
        snap = runner.snapshot()

        if snap.n_fovs > 0:
            progress = int(
                round(100 * (snap.n_done + snap.n_failed) / snap.n_fovs)
            )
            label = f"{snap.n_done + snap.n_failed} / {snap.n_fovs}"
        else:
            progress = 0
            label = ""

        results_block = _render_results(snap.results_summary)
        interval_disabled = not snap.active
        return (
            log_stream(snap.logs),
            progress, label,
            results_block,
            interval_disabled,
        )


# ── rendering helpers ──────────────────────────────────────────────────────


def _workspace_summary(workspace: WorkspacePaths) -> html.Div:
    tif_rows = []
    for tif in workspace.tifs:
        try:
            _, shape = validate_tif(tif)
            tif_rows.append(html.Tr([
                html.Td("OK", className="text-success fw-bold"),
                html.Td(tif.name),
                html.Td(f"{shape[0]}×{shape[1]}×{shape[2]}"),
            ]))
        except ValueError as exc:
            tif_rows.append(html.Tr([
                html.Td("!",  className="text-danger fw-bold"),
                html.Td(tif.name),
                html.Td(str(exc), className="text-danger"),
            ]))
    db_hint = "(new)" if not workspace.db_path.exists() else "(found)"
    return dbc.Card(dbc.CardBody([
        html.H6("Workspace resolved", className="mb-2"),
        html.Div([
            html.Span("Input: "),
            html.Code(str(workspace.input_root), className="roigbiv-muted-code")
        ], className="mb-1"),
        html.Div([
            html.Span("Output: "),
            html.Code(str(workspace.output_root),
                      className="roigbiv-muted-code"),
        ], className="mb-1"),
        html.Div([
            html.Span("Registry DB: "),
            html.Code(str(workspace.db_path), className="roigbiv-muted-code"),
            html.Span(f" {db_hint}", className="ms-2 text-muted"),
        ], className="mb-2"),
        dbc.Table(
            [html.Thead(html.Tr([html.Th(""), html.Th("File"), html.Th("Shape")])),
             html.Tbody(tif_rows)],
            size="sm", striped=True, borderless=True, className="mt-2 mb-0",
        ),
    ]), className="roigbiv-card-accent mt-2")


def _render_results(summaries: list[dict]) -> html.Div:
    if not summaries:
        return html.Div(html.Em("No FOV results yet.", className="text-muted"))
    rows = []
    for s in summaries:
        status = "FAILED" if s.get("error") else "OK"
        decision = s.get("registry_decision") or "—"
        counts = s.get("roi_counts") or {}
        duration = f"{s.get('duration_s', 0):.1f}s"
        rows.append(html.Tr([
            html.Td(status,
                    className=("text-danger fw-bold" if s.get("error")
                               else "text-success fw-bold")),
            html.Td(s.get("stem")),
            html.Td(duration),
            html.Td(f"A {counts.get('accept', 0)} / "
                    f"F {counts.get('flag', 0)} / "
                    f"R {counts.get('reject', 0)}"),
            html.Td(decision),
        ]))
    return dbc.Table(
        [html.Thead(html.Tr([
            html.Th(""), html.Th("FOV"), html.Th("Duration"),
            html.Th("ROIs"), html.Th("Registry"),
        ])), html.Tbody(rows)],
        size="sm", striped=True, borderless=False,
        className="mb-0",
    )
