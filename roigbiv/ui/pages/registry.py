"""Registry page — FOV list, per-FOV sessions, and maintenance actions.

Most users never need this page — migrate and backfill are folded into the
Process page's pipeline runner. It's exposed as an escape hatch for
debugging, manually re-running backfill, or clicking into a specific FOV to
open it in the Viewer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from roigbiv.ui.components.sidebar import workspace_summary_card
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.registry_service import (
    list_fovs,
    list_sessions_for_fov,
    run_backfill_now,
    run_migrations,
)


def layout() -> html.Div:
    state = get_app_state()
    return html.Div([
        dcc.Store(id="roigbiv-selected-fov", storage_type="memory"),
        dbc.Row([
            dbc.Col([
                workspace_summary_card(state.workspace),
                _maintenance_card(),
            ], md=4, className="pe-md-4"),
            dbc.Col([
                html.H4("FOVs in registry", className="mb-3"),
                dbc.Button("Refresh", id="roigbiv-registry-refresh",
                           color="primary", outline=True, size="sm",
                           className="mb-3", n_clicks=0),
                html.Div(id="roigbiv-fov-table"),
                html.Hr(),
                html.H5("Sessions", className="mb-3"),
                html.Div(id="roigbiv-sessions-detail",
                         children=html.Em("Select a FOV to see its sessions.",
                                          className="text-muted")),
            ], md=8),
        ], className="g-3"),
    ])


def _maintenance_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("Maintenance (escape hatches)", className="mb-2"),
        html.P(
            "These are run automatically on every pipeline invocation. "
            "Buttons below are for manual intervention only.",
            className="text-muted small",
        ),
        dbc.Button("Run migrations", id="roigbiv-migrate-btn",
                   color="secondary", outline=True, size="sm",
                   className="me-2 mb-2"),
        dbc.Button("Backfill (dry-run)", id="roigbiv-backfill-dry-btn",
                   color="secondary", outline=True, size="sm",
                   className="me-2 mb-2"),
        dbc.Button("Backfill now", id="roigbiv-backfill-btn",
                   color="warning", outline=True, size="sm",
                   className="mb-2"),
        html.Div(id="roigbiv-maintenance-output"),
    ]))


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("roigbiv-fov-table", "children"),
        Input("roigbiv-registry-refresh", "n_clicks"),
        Input("roigbiv-maintenance-output", "children"),
    )
    def _refresh_fovs(_n, _maint):
        try:
            rows = list_fovs()
        except Exception as exc:  # noqa: BLE001
            return dbc.Alert(
                f"Unable to read registry: {type(exc).__name__}: {exc}. "
                "Pick a workspace on the Process page and scan it.",
                color="warning",
            )
        if not rows:
            return html.Em("No FOVs registered yet. Run the pipeline on "
                           "the Process page to populate the registry.",
                           className="text-muted")
        header = html.Thead(html.Tr([
            html.Th("Animal"), html.Th("Region"),
            html.Th("Latest session"), html.Th("Sessions"),
            html.Th("v"), html.Th("FOV id"), html.Th(""),
        ]))
        body_rows = []
        for r in rows:
            body_rows.append(html.Tr([
                html.Td(r.animal_id or "—"),
                html.Td(r.region or "—"),
                html.Td(r.latest_session_date or "—"),
                html.Td(str(r.n_sessions)),
                html.Td(str(r.fingerprint_version or "—")),
                html.Td(html.Code(r.fov_id[:8] + "…",
                                  className="roigbiv-muted-code")),
                html.Td(dbc.Button(
                    "Open",
                    id={"type": "roigbiv-fov-open", "fov_id": r.fov_id},
                    color="primary", outline=True, size="sm",
                )),
            ]))
        return dbc.Table([header, html.Tbody(body_rows)],
                         striped=True, hover=True, responsive=True, size="sm")

    @app.callback(
        Output("roigbiv-sessions-detail", "children"),
        Output("roigbiv-selected-fov", "data"),
        Input({"type": "roigbiv-fov-open", "fov_id": dash.ALL}, "n_clicks"),
        State({"type": "roigbiv-fov-open", "fov_id": dash.ALL}, "id"),
        prevent_initial_call=True,
    )
    def _open_fov(n_clicks_list, ids_list):
        if not any(n_clicks_list or []):
            return no_update, no_update
        trig = dash.callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update, no_update
        fov_id = trig.get("fov_id")
        if not fov_id:
            return no_update, no_update
        try:
            sessions = list_sessions_for_fov(fov_id)
        except Exception as exc:  # noqa: BLE001
            return dbc.Alert(str(exc), color="danger"), no_update
        return _render_sessions(fov_id, sessions), fov_id

    @app.callback(
        Output("roigbiv-maintenance-output", "children"),
        Input("roigbiv-migrate-btn", "n_clicks"),
        Input("roigbiv-backfill-dry-btn", "n_clicks"),
        Input("roigbiv-backfill-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _maintenance(_m, _d, _b):
        trig = dash.callback_context.triggered_id
        state = get_app_state()
        if trig == "roigbiv-migrate-btn":
            try:
                msg = run_migrations()
            except Exception as exc:  # noqa: BLE001
                return dbc.Alert(f"Migrate failed: {exc}",
                                 color="danger", className="mt-2")
            return dbc.Alert(f"Migrations: {msg}",
                             color="success", className="mt-2")
        if trig in ("roigbiv-backfill-dry-btn", "roigbiv-backfill-btn"):
            if state.workspace is None:
                return dbc.Alert(
                    "Scan a workspace first so backfill knows what to walk.",
                    color="warning", className="mt-2",
                )
            dry = trig == "roigbiv-backfill-dry-btn"
            try:
                reports = run_backfill_now(state.workspace.output_root, dry_run=dry)
            except Exception as exc:  # noqa: BLE001
                return dbc.Alert(f"Backfill failed: {exc}",
                                 color="danger", className="mt-2")
            if not reports:
                return dbc.Alert("Backfill: nothing to do.",
                                 color="secondary", className="mt-2")
            decisions = _tally_decisions(reports)
            tag = "dry-run" if dry else "executed"
            return dbc.Alert(f"Backfill {tag}: {decisions}",
                             color="info", className="mt-2")
        return no_update


def _render_sessions(fov_id: str, sessions) -> html.Div:
    if not sessions:
        return html.Em("This FOV has no registered sessions.",
                       className="text-muted")
    header = html.Thead(html.Tr([
        html.Th("Date"), html.Th("Posterior"),
        html.Th("Matched"), html.Th("New"), html.Th("Missing"),
        html.Th("Output dir"),
    ]))
    body_rows = []
    for s in sessions:
        body_rows.append(html.Tr([
            html.Td(s.session_date or "—"),
            html.Td(f"{s.fov_posterior:.3f}" if s.fov_posterior is not None else "—"),
            html.Td(s.n_matched),
            html.Td(s.n_new),
            html.Td(s.n_missing),
            html.Td(html.Code(s.output_dir, className="roigbiv-muted-code")),
        ]))
    link = dcc.Link(
        dbc.Button("Open in Viewer", color="primary", size="sm"),
        href=f"/viewer?fov_id={fov_id}",
        className="mb-2 d-inline-block",
    )
    return html.Div([
        html.Div([html.Span("FOV: "), html.Code(fov_id,
                  className="roigbiv-muted-code")], className="mb-2"),
        link,
        dbc.Table([header, html.Tbody(body_rows)],
                  size="sm", striped=True, responsive=True),
    ])


def _tally_decisions(reports: list[dict]) -> str:
    counts: dict[str, int] = {}
    errors = 0
    for r in reports:
        if "error" in r:
            errors += 1
            continue
        d = r.get("decision") or r.get("action") or "unknown"
        counts[d] = counts.get(d, 0) + 1
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    if errors:
        parts.append(f"errors={errors}")
    return ", ".join(parts) if parts else "no reports"
