"""Boundary drawing — see the seeded outlines, and tune the one knob they have.

Last in the order of operations, and the only page with nothing to run: seeded
boundaries are pure numpy over a flow field centroid discovery already cached,
so everything here is instant and nothing touches the GPU.

What is being drawn
-------------------
Cellpose forms masks in two steps — a learned flow field, then a histogram
heuristic that clusters the pixels it converges. The heuristic has no idea which
cells are real. :mod:`roigbiv.pipeline.seeded_masks` keeps the flow field and
replaces the heuristic with the centroids a human confirmed on the Tracking
page: a pixel is cell material if its trajectory lands within ``capture_px`` of
some seed, and a watershed on ``-cellprob`` decides *which* seed owns it.

``capture_px`` is the one real tuning knob and, until this page, had no surface
at all. Too tight and every seed falls through to its canonical disk; too loose
and basins belonging to nothing survive. Both failure modes are reported below
the preview rather than left to be inferred from the picture.

Fallbacks are not failures
--------------------------
A seed that captures no basin still gets a disk, so a confirmed cell can never
vanish. But a *high* fallback rate means the detector never fired there, which
``capture_px`` cannot fix — on the bakeoff dataset, sweeping it from 6 to 45 px
moved the fallback count only 419 → 393. That is why the counts are on screen:
a boundary page that only showed outlines would make a detector problem look
like a tuning problem.

What Save writes
----------------
``boundaries.tif`` plus a ``settings`` block in ``boundaries.json``. Later
automatic redraws — a centroid edit, a fresh tracking run — read that block back
and reuse it, so a tuned FOV stays tuned without pinning anything for FOVs
nobody tuned.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, dcc, html

from roigbiv.ui.components import fov_select, workspace_bar
from roigbiv.ui.components.errors import user_error
from roigbiv.ui.components.figure import build_roi_figure
from roigbiv.ui.services import boundary_preview
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.loaders import ROIRender, label_shapes

FOV_SELECT_ID = "roigbiv-boundaries-fov"
CAPTURE_ID = "roigbiv-boundaries-capture"
MIN_AREA_ID = "roigbiv-boundaries-min-area"
PREVIEW_ID = "roigbiv-boundaries-preview"
STATS_ID = "roigbiv-boundaries-stats"
SAVE_ID = "roigbiv-boundaries-save"
SAVE_ALL_ID = "roigbiv-boundaries-save-all"
STATUS_ID = "roigbiv-boundaries-status"

# Colour by where a label's pixels came from — the only distinction that
# matters here, and the one the statistics line counts.
_ORIGIN_COLOR = {"flow": "#3FC1C9", "disk_fallback": "#FF7A45"}


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    workspace = get_app_state().workspace
    _, value = fov_select.processed_options_and_value(workspace)
    capture, min_area = _settings_for(value)
    figure, stats = draw(value, capture, min_area)
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("Seeded boundaries", className="mb-3"),
                fov_select.select(FOV_SELECT_ID, workspace, processed_only=True,
                                  className="mb-3"),
                dbc.Card(dbc.CardBody([
                    html.Div([
                        html.Span("capture_px", className="small me-2"),
                        html.Span(
                            "how far a flow trajectory may land from a seed "
                            "and still be that cell",
                            className="text-muted small"),
                    ], className="mb-1"),
                    dcc.Slider(id=CAPTURE_ID, min=2, max=60, step=1,
                               value=capture, marks=None,
                               tooltip={"placement": "bottom",
                                        "always_visible": True}),
                    html.Div([
                        html.Span("min_area (px)", className="small me-2"),
                        html.Span("drop anything smaller",
                                  className="text-muted small"),
                    ], className="mb-1 mt-3"),
                    dbc.Input(id=MIN_AREA_ID, type="number", min=0, step=10,
                              value=min_area, style={"width": "120px"}),
                ]), className="mb-3"),
                html.Div(id=STATS_ID, children=stats),
                html.Div([
                    dbc.Button("Save this FOV", id=SAVE_ID, color="primary",
                               className="me-2"),
                    dbc.Button("Apply to every FOV", id=SAVE_ALL_ID,
                               color="secondary", outline=True),
                ], className="mt-3"),
                html.Div(id=STATUS_ID, className="mt-2"),
            ], md=4, className="pe-md-4"),
            dbc.Col([
                dcc.Graph(id=PREVIEW_ID, figure=figure,
                          config={"displaylogo": False, "scrollZoom": True},
                          style={"height": "760px"}),
            ], md=8),
        ], className="g-3"),
    ])


def _cfg():
    """A config carrying no boundary overrides, so the page's controls win.

    Reading ``PipelineConfig``'s own defaults here would let a ``boundary_*``
    set in ``configs/pipeline.yaml`` outrank the slider the user is dragging.
    """
    from roigbiv.pipeline.types import PipelineConfig

    cfg = PipelineConfig(no_viewer=True)
    cfg.boundary_capture_px = None
    cfg.boundary_min_area = None
    return cfg


def _settings_for(value: Optional[str]) -> tuple[float, int]:
    out_dir = _output_dir(value)
    if out_dir is None:
        return 12.0, 0
    try:
        capture, min_area = boundary_preview.default_settings(out_dir, _cfg())
    except Exception:  # noqa: BLE001 — an unreadable FOV still gets a slider
        return 12.0, 0
    return float(capture), int(min_area)


def _output_dir(value: Optional[str]) -> Optional[Path]:
    _, _, out_dir = fov_select.mean_and_title(value)
    return out_dir


def draw(value: Optional[str], capture, min_area):
    """``(figure, statistics)`` from a **single** partition.

    Both come from one :func:`boundary_preview.preview` call. Deriving them
    separately doubles the per-tick cost of the slider — measured at ~196 ms a
    partition on a 509x509 FOV, which is the difference between a control that
    tracks the drag and one that lags it.
    """
    mean, title, out_dir = fov_select.mean_and_title(value)
    bare = build_roi_figure(mean, [], show_overlay=False, title=title)

    if out_dir is None:
        return bare, dbc.Alert(
            "Run motion correction and centroid discovery first.",
            color="secondary", className="py-2 mb-0")
    try:
        result = boundary_preview.preview(
            out_dir, _cfg(), capture_px=float(capture or 12.0),
            min_area=int(min_area or 0))
    except boundary_preview.NoFlowCache as exc:
        # Drawing the bare projection is more honest than an empty axis; the
        # way out is named in the statistics panel beside it.
        return bare, dbc.Alert(str(exc), color="warning", className="py-2 mb-0")
    except Exception as exc:  # noqa: BLE001 — disk or decode, both the user's
        return bare, user_error(exc, "computing boundaries for this FOV")

    return (build_roi_figure(mean, _rois(result), show_overlay=True, title=title),
            _stats_card(result))




def _rois(result) -> list[ROIRender]:
    """Boundaries as the same ``ROIRender`` contours the Tracking sheet draws.

    Shared tracer (:func:`roigbiv.ui.services.loaders.label_shapes`) on purpose:
    a boundary that looked different here than on the page it is drawn for would
    make this a preview of something else.
    """
    out: list[ROIRender] = []
    for label_id, (centroid, contours, area) in sorted(
            label_shapes(result.labels).items()):
        origin = result.origins.get(label_id, "flow")
        out.append(ROIRender(
            label_id=label_id, source_stage=1, gate_outcome="accept",
            activity_type=None, area=area, centroid_yx=centroid,
            contours=contours, global_cell_id=None,
            match_status=("new" if origin == "disk_fallback" else "matched"),
        ))
    return out


def _stats_for(value: Optional[str], capture, min_area):
    """Just the statistics half of :func:`draw`, for tests reading the copy."""
    return draw(value, capture, min_area)[1]


def _stats_card(result):
    """Seeds, fallbacks, orphan pixels and the size the disks would have been.

    The last comparison is what makes a boundary judgeable at a glance: if the
    median flow-derived area is not meaningfully different from the disk it
    replaces, this whole track is buying nothing on this FOV.
    """
    flow_areas = [a for label, a in result.areas.items()
                  if result.origins.get(label) == "flow" and a > 0]
    disk_area = np.pi * result.fallback_radius ** 2

    rows = [
        _stat("seeds", result.n_seeds),
        _stat("disk fallbacks", result.n_disk_fallback,
              warn=result.n_disk_fallback > result.n_seeds // 2),
        _stat("orphan px", result.n_orphan_basin_px),
    ]
    detail = [
        html.Div(
            f"median flow-derived area "
            f"{int(np.median(flow_areas))} px — a disk of radius "
            f"{result.fallback_radius} would be {int(disk_area)} px",
            className="small text-muted mt-2")
        if flow_areas else
        html.Div("no seed captured a flow basin at this capture_px",
                 className="small text-warning mt-2"),
    ]
    if result.n_disk_fallback > result.n_seeds // 2:
        detail.append(html.Div(
            "More than half the seeds fell back to disks. capture_px rarely "
            "fixes this — it usually means the detector never fired there.",
            className="small text-warning mt-1"))
    for warning in result.warnings:
        detail.append(html.Div(warning, className="small text-warning mt-1"))

    return dbc.Card(dbc.CardBody(
        [dbc.Row([dbc.Col(r) for r in rows], className="g-2")] + detail),
        className="mb-0")


def _stat(label: str, value: int, *, warn: bool = False) -> html.Div:
    return html.Div([
        html.Div(str(value),
                 className=("roigbiv-track-seq h5 mb-0"
                            + (" text-warning" if warn else ""))),
        html.Div(label, className="text-muted small"),
    ])


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output(FOV_SELECT_ID, "options"),
        Output(FOV_SELECT_ID, "value"),
        Input(workspace_bar.WORKSPACE_VERSION, "data"),
        State(FOV_SELECT_ID, "value"),
        prevent_initial_call=True,
    )
    def _refresh_fovs(_version, current):
        return fov_select.processed_options_and_value(
            get_app_state().workspace, current)

    @app.callback(
        Output(CAPTURE_ID, "value"),
        Output(MIN_AREA_ID, "value"),
        Input(FOV_SELECT_ID, "value"),
        prevent_initial_call=True,
    )
    def _seed_settings(value):
        # Each FOV carries its own pinned settings; switching without this would
        # show the previous FOV's numbers over the new FOV's picture.
        return _settings_for(value)

    @app.callback(
        Output(PREVIEW_ID, "figure"),
        Output(STATS_ID, "children"),
        Input(FOV_SELECT_ID, "value"),
        Input(CAPTURE_ID, "value"),
        Input(MIN_AREA_ID, "value"),
        prevent_initial_call=True,
    )
    def _render(value, capture, min_area):
        # One callback, one partition — see draw(). Two callbacks, or two calls
        # here, would double the per-tick cost of the slider.
        return draw(value, capture, min_area)

    @app.callback(
        Output(STATUS_ID, "children"),
        Input(SAVE_ID, "n_clicks"),
        Input(SAVE_ALL_ID, "n_clicks"),
        State(FOV_SELECT_ID, "value"),
        State(CAPTURE_ID, "value"),
        State(MIN_AREA_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_save(_n, _n_all, value, capture, min_area):
        from roigbiv.pipeline.boundaries import write_boundaries

        state = get_app_state()
        if state.workspace is None:
            return dbc.Alert("Scan a workspace first.", color="warning",
                             className="py-2 mb-0")
        if dash.ctx.triggered_id == SAVE_ALL_ID:
            targets = sorted(
                p.parent for p in
                Path(state.workspace.output_root).glob("*/centroids.json"))
        else:
            out_dir = _output_dir(value)
            if out_dir is None:
                return dbc.Alert("Select a FOV first.", color="warning",
                                 className="py-2 mb-0")
            targets = [out_dir]

        written, skipped, failed = [], [], []
        for out_dir in targets:
            try:
                drawn = write_boundaries(
                    out_dir, _cfg(), capture_px=float(capture or 12.0),
                    min_area=int(min_area or 0))
            except Exception as exc:  # noqa: BLE001 — reported per FOV below
                failed.append(f"{out_dir.name}: {type(exc).__name__}: {exc}")
                continue
            if drawn is None or not drawn.written:
                skipped.append(out_dir.name)
            else:
                written.append(out_dir.name)

        return _save_report(written, skipped, failed)


def _save_report(written: list, skipped: list, failed: list):
    """Name what did *not* happen, not just what did.

    A silent skip count would read as success on a workspace where most FOVs
    have no flow cache — which is exactly the workspace where this page is
    least useful and most needs to say so.
    """
    if failed:
        color = "danger"
    elif written:
        color = "success"
    else:
        color = "warning"
    lines = [html.Div(f"{len(written)} FOV(s) written.")]
    if skipped:
        lines.append(html.Div(
            f"{len(skipped)} skipped — no flow cache, or a full cascade owns "
            f"the geometry: {', '.join(skipped)}",
            className="small"))
    for msg in failed:
        lines.append(html.Div(msg, className="small font-monospace"))
    return dbc.Alert(lines, color=color, className="py-2 mb-0")
