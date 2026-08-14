"""Run status for the one pipeline runner, shared by the pages that start it.

There is a single :class:`~roigbiv.ui.services.pipeline_runner.PipelineRunner`
per browser session, behind a process-wide GPU gate — the RTX 5080 cannot
service two pipeline jobs at once. So the Motion-correction and Centroids pages
cannot each own a runner, and this panel is what they both look at.

The split of responsibility is deliberate:

* the **panel** owns the status — banner, progress, timer, log, stop, results —
  and its callbacks are registered **once** from ``build_app``. Registering per
  page would give Dash duplicate Outputs on the same ids.
* the **page** owns its Run button and supplies only its own overrides.

When a run started from the other page is active, the banner names it and the
page's own Run button disables. Hiding a foreign run would be worse than
showing it: the user would press Run, get "already active", and have nothing on
screen explaining why.
"""
from __future__ import annotations

import time
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, html

from roigbiv.ui.components.log_stream import log_stream
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.pipeline_runner import RunSnapshot, get_pipeline_runner

TICK_ID = "roigbiv-run-tick"
BANNER_ID = "roigbiv-run-banner"
PROGRESS_ID = "roigbiv-run-progress"
TIMER_ID = "roigbiv-run-timer"
LOG_ID = "roigbiv-run-log"
RESULTS_ID = "roigbiv-run-results"
STOP_ID = "roigbiv-stop-btn"
CONFIG_ID = "roigbiv-run-config"

def run_mode(overrides: Optional[dict]) -> str:
    """What kind of run this is, from the overrides it was launched with.

    The page that started a run is not recorded anywhere else, and a bare
    "Running · Stage 1" on the motion page would read as the motion run having
    stalled rather than as a centroid run someone started next door.
    """
    if not overrides:
        return "pipeline"
    foundation = bool(overrides.get("foundation_only"))
    centroids = bool(overrides.get("run_centroids"))
    if foundation and centroids:
        return "motion correction + centroid discovery"
    if foundation:
        return "motion correction"
    if centroids:
        return "centroid discovery"
    return "full pipeline"


# ── layout ─────────────────────────────────────────────────────────────────


def layout(*, title: str = "Run status") -> html.Div:
    """The status panel, seeded from the live runner.

    Seeding matters: the runner persists per Flask session, so a mid-run reload
    or a navigation to the other page repaints the last snapshot rather than
    dropping back to an empty panel.
    """
    snap = get_pipeline_runner().snapshot()
    has_run = snap.started_at is not None
    progress, label = progress_for(snap)
    return html.Div([
        html.H4(title, className="mb-3"),
        html.Div(id=CONFIG_ID,
                 children=launched_config(snap if has_run else None)),
        html.Div(id=TIMER_ID, className="mb-2",
                 children=(format_timer(snap.started_at, snap.completed_at)
                           if has_run else "")),
        dbc.Progress(id=PROGRESS_ID, value=progress, label=label,
                     striped=True, className="mb-3"),
        html.Div(id=BANNER_ID, children=render_banner(snap)),
        dbc.Button("Stop run", id=STOP_ID, color="danger", outline=True,
                   size="sm", className="mb-2",
                   n_clicks=0, disabled=not snap.active),
        html.Div(id=LOG_ID, children=log_stream(snap.logs if has_run else [])),
        html.Hr(),
        html.H5("Per-FOV results", className="mb-2"),
        html.Div(id=RESULTS_ID,
                 children=render_results(snap.results_summary if has_run else [])),
    ])


def tick() -> "dash.dcc.Interval":
    """The poll driving the panel. One per page layout, disabled while idle.

    Drives *only* this panel's own outputs. A page's own periodic work belongs
    on :func:`page_tick`: this interval is mounted on several pages, so binding
    a page-local Output to it would ask the server to compute an update for a
    component that is not on screen.
    """
    from dash import dcc

    return dcc.Interval(id=TICK_ID, interval=1500,
                        disabled=not get_pipeline_runner().snapshot().active)


def page_tick(tick_id: str, *, interval: int = 2000) -> "dash.dcc.Interval":
    """A page's own poll, for refreshing that page's own controls.

    Always enabled while the page is mounted, and deliberately slower than the
    run panel's: what rides it (a FOV list, a Run button's disabled state) is a
    directory glob and a snapshot read, and it must stay fresh whether or not a
    run happens to be in flight.
    """
    from dash import dcc

    return dcc.Interval(id=tick_id, interval=interval)


# ── rendering ──────────────────────────────────────────────────────────────


def progress_for(snap: Optional[RunSnapshot]) -> tuple[int, str]:
    """Progress-bar value (0–100) and ``done / total`` label from a snapshot."""
    if snap and snap.n_fovs > 0:
        done = snap.n_done + snap.n_failed
        return int(round(100 * done / snap.n_fovs)), f"{done} / {snap.n_fovs}"
    return 0, ""


def render_banner(snap: Optional[RunSnapshot]):
    """Live run-status banner: current stage while active, outcome when done."""
    if snap is None or snap.started_at is None:
        return None
    # Error wins over stopped: a crash on the post-stop path (e.g. backfill)
    # still sets the abort event, so guard stopped on the absence of an error
    # or the failure would be masked as a clean "Run stopped."
    if snap.error:
        return dbc.Alert("Run failed — see log below.",
                         color="danger", className="py-2 mb-2")
    if snap.stopped:
        return dbc.Alert("Run stopped.",
                         color="secondary", className="py-2 mb-2")
    mode = run_mode(snap.overrides)
    if not snap.active:
        return dbc.Alert(f"{mode.capitalize()} complete.",
                         color="success", className="py-2 mb-2")
    if snap.stopping:
        stage = snap.current_stage or "current stage"
        return dbc.Alert(
            [html.Span("Stopping · ", className="fw-bold"),
             html.Span(f"finishing {stage}, then halting…")],
            color="warning", className="py-2 mb-2",
        )
    stage = snap.current_stage or "started"
    return dbc.Alert(
        [html.Span(f"Running {mode} · ", className="fw-bold"),
         html.Span(stage)],
        color="info", className="py-2 mb-2",
    )


def launched_config(snap: Optional[RunSnapshot]):
    """Read-only echo of the overrides that actually launched the run.

    Rendered from the runner snapshot, so it survives navigation / reload and —
    unlike the live, persisted parameter forms — never misrepresents an
    in-progress run if the user edits a form afterward. ``None`` before any run.
    """
    if snap is None or snap.started_at is None or not snap.overrides:
        return None
    ov = snap.overrides

    def _item(label: str, value) -> html.Div:
        return html.Div(
            [html.Span(f"{label}: ", className="text-muted"),
             html.Span(str(value), className="fw-semibold")],
            className="small me-3 d-inline-block",
        )

    items = [_item("mode", run_mode(ov)), _item("FOVs", snap.n_fovs)]
    if ov.get("foundation_only"):
        items += [_item("fs", ov.get("fs")), _item("tau", ov.get("tau")),
                  _item("MC", ov.get("motion_correction_backend"))]
    return dbc.Card(dbc.CardBody([
        html.H6("Launched config", className="mb-2"),
        html.Div(items),
    ]), className="roigbiv-card-accent mb-3")


def format_timer(started_at: Optional[float],
                 completed_at: Optional[float]) -> "str | html.Div":
    if started_at is None:
        return ""
    start_str = time.strftime("%H:%M:%S", time.localtime(started_at))
    end_ts = completed_at if completed_at is not None else time.time()
    elapsed_s = int(end_ts - started_at)
    h, rem = divmod(elapsed_s, 3600)
    m, s = divmod(rem, 60)
    return html.Div(
        [html.Span(f"Started: {start_str}", className="me-4"),
         html.Span(f"Elapsed: {h:02d}:{m:02d}:{s:02d}")],
        style={"fontFamily": "var(--roigbiv-font-mono)",
               "fontSize": "0.80rem",
               "color": "var(--roigbiv-accent)"},
    )


def fmt_metric(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def render_results(summaries: list[dict]) -> html.Div:
    """Per-FOV outcomes, with only the columns this run actually produced.

    A motion-correction run has no centroid counts and a centroids-only run has
    no MC metrics; showing both sets unconditionally left a column of em-dashes
    that reads as a failure rather than as "not applicable".
    """
    if not summaries:
        return html.Div(html.Em("No FOV results yet.", className="text-muted"))

    show_mc = any(s.get("mc_metrics") for s in summaries)
    show_centroids = any(s.get("centroid_count") is not None for s in summaries)

    head = [html.Th(""), html.Th("FOV"), html.Th("Duration")]
    if show_mc:
        head += [html.Th("Sharpness"), html.Th("Banding"),
                 html.Th("Anisotropy"), html.Th("Contrast")]
    if show_centroids:
        head.append(html.Th("Centroids"))

    rows = []
    for s in summaries:
        m = s.get("mc_metrics") or {}
        cells = [
            html.Td("FAILED" if s.get("error") else "OK",
                    className=("text-danger fw-bold" if s.get("error")
                               else "text-success fw-bold")),
            html.Td(s.get("stem")),
            html.Td(f"{s.get('duration_s', 0):.1f}s"),
        ]
        if show_mc:
            cells += [html.Td(fmt_metric(m.get(k))) for k in
                      ("lap_var_smooth", "banding_score",
                       "grad_anisotropy_xy", "contrast_rms")]
        if show_centroids:
            count = s.get("centroid_count")
            cells.append(html.Td("—" if count is None else str(count)))
        rows.append(html.Tr(cells))

    return dbc.Table([html.Thead(html.Tr(head)), html.Tbody(rows)],
                     size="sm", striped=True, borderless=False,
                     className="mb-0")


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    """Wire the panel. Called **once** from ``build_app``, not per page."""

    @app.callback(
        Output(LOG_ID, "children"),
        Output(PROGRESS_ID, "value"),
        Output(PROGRESS_ID, "label"),
        Output(RESULTS_ID, "children"),
        Output(TICK_ID, "disabled", allow_duplicate=True),
        Output(TIMER_ID, "children"),
        Output(BANNER_ID, "children", allow_duplicate=True),
        Output(STOP_ID, "disabled", allow_duplicate=True),
        Output(CONFIG_ID, "children"),
        Input(TICK_ID, "n_intervals"),
        prevent_initial_call="initial_duplicate",
    )
    def _on_tick(_n):
        snap = get_pipeline_runner().snapshot()
        progress, label = progress_for(snap)
        return (
            log_stream(snap.logs),
            progress, label,
            render_results(snap.results_summary),
            not snap.active,
            format_timer(snap.started_at, snap.completed_at),
            render_banner(snap),
            # Stop is actionable only while a run is in flight and not already
            # stopping.
            (not snap.active) or snap.stopping,
            launched_config(snap),
        )

    @app.callback(
        Output(BANNER_ID, "children", allow_duplicate=True),
        Output(STOP_ID, "disabled", allow_duplicate=True),
        Input(STOP_ID, "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_stop(_n: int):
        # Cooperative stop: flag the in-flight run to halt at the next stage
        # boundary. Disable the button once requested; the tick refreshes the
        # banner from "Stopping…" to "Run stopped." when it ends.
        runner = get_pipeline_runner()
        requested = runner.abort()
        return render_banner(runner.snapshot()), (not requested)


def run_disabled() -> bool:
    """Whether a page's own Run button should be disabled right now."""
    return (get_app_state().workspace is None
            or get_pipeline_runner().snapshot().active)
