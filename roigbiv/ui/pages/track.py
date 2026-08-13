"""Track page — confirm session chronology, then track cells across them.

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

Flow
----
1. Scan a workspace on the Pipeline page (this page reuses that selection).
2. Sessions render in their saved order, or a proposal from filename dates.
   Rows whose date is ambiguous or unreadable are badged for attention.
3. Drag rows to reorder (``assets/reorder.js`` publishes the new order into a
   hidden input; the callback below persists it to ``session_order.json``).
4. **Run tracking** stamps each FOV's centroids into the label image the
   registry reads and registers the sessions in that order, reporting each
   session's outcome — matched, new, missing, and how confident the match was.
5. The anomaly panel reports cells that arrive late, drop out, or blink. It
   falls back to the registry when this browser session has no run of its own,
   so a workspace tracked from the CLI still reports here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from roigbiv.pipeline.session_order import (
    SessionOrderEntry,
    discover_trackable_stems,
    reorder,
    resolve_order,
    save_order,
)
from roigbiv.ui.components.errors import user_error
from roigbiv.ui.components.log_stream import log_stream
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.registry_service import workspace_anomalies
from roigbiv.ui.services.tracking_runner import (
    TrackingSnapshot,
    get_tracking_runner,
)

LIST_ID = "roigbiv-track-list"
ORDER_SINK_ID = "roigbiv-track-order-sink"

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
    return html.Div([
        dcc.Interval(id="roigbiv-track-tick", interval=1500,
                     disabled=not snap.active),
        # The drag script publishes the reordered stems here as a JSON array.
        # Hidden rather than removed: Dash only listens to inputs in the tree.
        dcc.Input(id=ORDER_SINK_ID, type="text", value="",
                  style={"display": "none"}),
        dbc.Row([
            dbc.Col([
                html.H4("Session order"),
                html.P(
                    "Drag sessions into the order they were recorded. This "
                    "order decides which session owns each cell's identity, "
                    "so it is worth getting right.",
                    className="text-muted",
                ),
                html.Div(id="roigbiv-track-list-wrap", children=_session_list()),
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
                html.H4("Run status"),
                html.Div(id="roigbiv-track-status", children=_status(snap)),
                html.Div(id="roigbiv-track-results",
                         children=_results_table(snap), className="mt-2"),
                html.Div(id="roigbiv-track-logs",
                         children=log_stream(snap.logs,
                                             empty_hint="No tracking run yet."),
                         className="mt-2"),
                html.H4("Anomalies", className="mt-4"),
                html.Div(id="roigbiv-track-anomalies",
                         children=_anomaly_panel(snap)),
            ], md=6),
        ]),
    ])


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
        return dbc.Alert(
            "Scan a workspace on the Pipeline page first.", color="secondary")
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


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
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
        Input("roigbiv-track-tick", "n_intervals"),
        prevent_initial_call=True,
    )
    def _on_tick(_n: int):
        snap = get_tracking_runner().snapshot()
        return (_status(snap),
                _results_table(snap),
                log_stream(snap.logs, empty_hint="No tracking run yet."),
                _anomaly_panel(snap),
                not snap.active)
