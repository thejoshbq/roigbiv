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
   annotation instead.
2. A **filmstrip** (:mod:`roigbiv.ui.components.cell_strip`) — the selected
   cell cropped out of every session at one fixed scale. This is the evidence;
   the sheet is only the index into it.

Color carries outcome, not identity: matched / new here / not detected, three
hues total. Identity is revealed on demand — click any cell and it thickens
and shows its ``#N`` in every panel at once. Coloring 40-odd cells by hashed
hue would put the whole answer on screen permanently and make none of it
readable.

Editing
-------
The **Edit** switch turns clicks from selection into a correction: delete a
centroid, add one, move one, or link/unlink two cells across sessions. Off by
default — every click above describes read-only behaviour. Edits apply
instantly (no separate "Apply" step) and are additive: they append to a
per-session centroid log and a per-FOV correspondence log, replayed by
:func:`roigbiv.registry.cell_edits.apply_tracking_edits` rather than mutating
pipeline output in place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, Patch, State, ctx, dcc, html, no_update

from roigbiv.ui.components.cell_strip import cell_strip
from roigbiv.ui.components.errors import user_error
from roigbiv.ui.components.figure import build_roi_figure, trace_index_map
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.colors import (
    MATCH_STATUS_LABELS,
    MATCH_STATUS_PALETTE,
    color_for_match_status,
)
from roigbiv.ui.services.loaders import ROIRender
from roigbiv.ui.services.registry_service import list_fovs
from roigbiv.ui.services.tracked_cells import (
    TrackedFOV,
    invalidate_tracked_fov,
    load_tracked_fov_cached,
)

FOV_ID = "roigbiv-cells-fov"
SELECTED_ID = "roigbiv-cells-selected"
INDEX_ID = "roigbiv-cells-trace-index"
SHEET_ID = "roigbiv-cells-sheet"
STRIP_ID = "roigbiv-cells-strip"
LIST_ID = "roigbiv-cells-list"
HEADER_ID = "roigbiv-cells-header"
PREV_ID = "roigbiv-cells-prev"
NEXT_ID = "roigbiv-cells-next"
NUMBERS_ID = "roigbiv-cells-numbers"
DRAWER_ID = "roigbiv-cells-drawer"
RAIL_TOGGLE_ID = "roigbiv-cells-rail-toggle"
RAIL_COL_ID = "roigbiv-cells-rail-col"
SHEET_COL_ID = "roigbiv-cells-sheet-col"
SYNC_SINK_ID = "roigbiv-cells-sync-sink"

# Edit mode — see the module docstring's "Editing" section.
EDIT_ID = "roigbiv-cells-edit"
MODE_ID = "roigbiv-cells-mode"
MODE_ROW_ID = "roigbiv-cells-mode-row"
UNDO_ID = "roigbiv-cells-undo"
EDIT_MSG_ID = "roigbiv-cells-edit-msg"
# Holds the label a "move" or "link" gesture picked up, waiting for the
# second click that completes it. Cleared on mode change or apply.
PICKUP_ID = "roigbiv-cells-pickup"

PANEL_TYPE = "roigbiv-cells-panel"

EDIT_MODES = ("select", "add", "move", "delete", "link")

# Panels are now viewport-height and zoomable, so the background has to survive
# being zoomed into. Full 1024x1024 float projections would be ~6 MB of JSON
# each; quantising to uint16 (see _quantized) keeps the same picture at roughly
# a third of the text. The filmstrip crops stay untouched full-resolution
# floats, which is where detail actually decides anything.
_SHEET_MAX_PX = 1024

_OUTLINE_WIDTH = 3.2
_HIGHLIGHT_WIDTH = 6.0

# Resting numbers have to be readable without competing with the image; the
# selected one has to be findable in a field of forty of them.
_BADGE_SIZE = 11
_BADGE_SIZE_SELECTED = 16


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    return html.Div([
        dcc.Store(id=SELECTED_ID),
        dcc.Store(id=INDEX_ID),
        # Written only so the zoom-sync clientside callback has somewhere to
        # return to; it does its work through Plotly directly.
        dcc.Store(id=SYNC_SINK_ID),
        # (stem, label) of a move/link gesture's first click, or None.
        dcc.Store(id=PICKUP_ID),
        _toolbar(),
        _mode_row(),
        html.Div(id=HEADER_ID, className="text-muted small mb-2"),
        dbc.Row([
            dbc.Col(html.Div(id=SHEET_ID), id=SHEET_COL_ID, className="col-md-9"),
            dbc.Col(_rail(), id=RAIL_COL_ID, className="col-md-3"),
        ], className="g-2"),
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
        dbc.Col(dbc.Button("cells ▸", id=RAIL_TOGGLE_ID, size="sm",
                           color="secondary", outline=True, n_clicks=0,
                           className="mt-3", title="show / hide the cell list"),
                width="auto"),
        dbc.Col(dbc.Switch(id=EDIT_ID, label="edit", value=False,
                           className="mb-0 mt-3"), width="auto"),
        dbc.Col(_legend(), className="text-md-end"),
    ], className="align-items-center g-2")


def _mode_row() -> html.Div:
    """Edit-mode controls — mode choice, undo, and the last op's outcome.

    Hidden (``d-none``) rather than left unmounted while the Edit switch is
    off, so ``_on_edit_toggle`` only has to flip a className: no branch has to
    reconstruct these controls from scratch when edit mode turns on.
    """
    return html.Div([
        dbc.RadioItems(
            id=MODE_ID,
            options=[{"label": m, "value": m} for m in EDIT_MODES],
            value="select", inline=True, className="me-3",
        ),
        dbc.Button("Undo last", id=UNDO_ID, size="sm", color="secondary",
                   outline=True, className="me-3"),
        html.Span(id=EDIT_MSG_ID, className="text-muted small"),
    ], id=MODE_ROW_ID, className="d-none align-items-center g-2 mb-2 flex-wrap")


def _rail() -> html.Div:
    return html.Div([
        html.Div("Numbers are for reading this page only — they follow the "
                 "session order and are not the cell's registry id.",
                 className="text-muted small mb-2"),
        html.Div(id=LIST_ID, className="roigbiv-cells-list"),
    ])


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


# ── contact sheet ──────────────────────────────────────────────────────────


def _sheet(fov: TrackedFOV, show_numbers: bool = True) -> tuple[html.Div, dict]:
    """The per-session panels, plus the trace map a selection restyles through."""
    if not fov.sessions:
        return (dbc.Alert("This FOV has no readable session output.",
                          color="warning"), {})

    panels: list = []
    index: dict = {}
    step = _sheet_step(fov.sessions)
    for i, session in enumerate(fov.sessions):
        figure, panel_index = _panel_figure(fov, session, i, step, show_numbers)
        index[str(i)] = panel_index
        panels.append(_panel(session, figure, i))
    return html.Div([
        html.Div(panels, className="roigbiv-cells-sheet"),
        html.Div("Panels share one zoom window, but sessions are not "
                 "co-registered — the same cell sits tens of pixels apart "
                 "between them.", className="text-muted small mt-1"),
    ]), index


def _sheet_step(sessions) -> float:
    """One reduction factor for the whole sheet.

    Deriving it per session would put panels of differing frame size into
    different pixel coordinate systems — harmless while each panel zoomed on
    its own, wrong the moment they share a view, since the synchroniser copies
    axis ranges verbatim between them.
    """
    largest = max(
        (max(np.asarray(s.mean_M).shape[:2])
         for s in sessions if s.mean_M is not None),
        default=0,
    )
    return float(max(1, int(np.ceil(largest / _SHEET_MAX_PX)))) if largest else 1.0


def _panel(session, figure, i: int) -> html.Div:
    header = [
        html.Div([
            html.Span(f"{i + 1}.", className="roigbiv-track-seq me-1"),
            html.Span(session.short_label, className="font-monospace"),
            html.Span(session.label, className="text-muted small ms-2"),
        ], className="small text-truncate"),
        html.Div(_counts_line(session), className="text-muted small"),
    ]
    if session.stale:
        header.append(dbc.Alert(
            "This session's registry_match.json names a different session than "
            "the registry does — the counts below come from the registry. "
            "Re-run tracking to reconcile them.",
            color="warning", className="py-1 px-2 small mt-1 mb-0"))
    return html.Div(header + [
        dcc.Graph(
            id={"type": PANEL_TYPE, "index": i},
            figure=figure,
            config={"displayModeBar": False, "scrollZoom": True,
                    "responsive": True},
            # An explicit height rather than 100%: Plotly measures its own
            # container, and a percentage height through a flex parent is not
            # reliably resolvable at mount time.
            className="roigbiv-cells-panel-graph",
        ),
    ], className="roigbiv-cells-panel")


def _counts_line(session) -> str:
    """What this panel is actually showing, counted off the panel itself.

    Deliberately *not* the session row's ``n_matched`` / ``n_new`` /
    ``n_missing``: those count what the matcher decided at registration time
    and can disagree with the observations — a session recorded as
    ``n_missing=0`` can still be drawing a dozen cells that earlier sessions
    saw and it did not. A caption that contradicts the picture above it is
    worse than either number alone, so the caption follows the picture. The
    registration figures are still reported by ``roigbiv-registry show``.
    """
    counts: dict[str, int] = {}
    for roi in session.rois:
        if roi.match_status:
            counts[roi.match_status] = counts.get(roi.match_status, 0) + 1
    parts = [f"{counts.get('matched', 0)} matched"]
    if counts.get("new"):
        parts.append(f"{counts['new']} new")
    if counts.get("lost"):
        parts.append(f"{counts['lost']} not detected")
    return " · ".join(parts)


def _panel_figure(
    fov: TrackedFOV, session, i: int, step: float, show_numbers: bool,
) -> tuple[go.Figure, dict]:
    mean = _downsampled(session.mean_M, step)
    rois = [_scaled(r, step) for r in session.rois]
    figure = build_roi_figure(mean=_quantized(mean), rois=rois,
                              color_mode="status", hide_rejected=False)
    index = {"outline": {str(k): v for k, v in trace_index_map(figure).items()}}
    index["badge"] = _add_badges(figure, fov, rois, i, show_numbers)
    index["gcid"] = {
        str(r.label_id): fov.gcid_for_label(i, r.label_id) for r in rois
    }
    # Everything edit mode needs to turn a raw Plotly click into a decision —
    # see _resolve_click. Kept in full-resolution coordinates (unlike `rois`
    # above, which are pre-scaled for drawing) so a click resolves to the same
    # (y, x) regardless of how far the sheet has downsampled this panel.
    index["step"] = step
    index["session_id"] = session.session_id
    index["output_dir"] = str(session.output_dir) if session.output_dir else None
    index["stem"] = session.stem
    index["radius"] = _stamp_radius(session.output_dir)
    # Ghosts (negative label_id) are excluded: they have no footprint in this
    # session, so a click near one should resolve to "empty" — the gesture
    # that adds a centroid there — not to a false "hit" on a cell that isn't
    # actually present. A direct click on a ghost's own outline still selects
    # it via customdata, unaffected by this table.
    index["centroid"] = {
        str(r.label_id): list(r.centroid_yx) for r in session.rois
        if r.label_id > 0
    }
    return figure, index


def _stamp_radius(output_dir) -> int:
    """The canonical disk radius for this session, or 0 when it can't be read.

    0 disables the nearest-centroid fallback in ``_resolve_click`` rather than
    guessing — a session with no on-disk output (as in tests) or no
    centroids.json yet has nothing to calibrate against.
    """
    if output_dir is None:
        return 0
    try:
        from roigbiv.pipeline.centroid_masks import resolve_stamp_radius
        from roigbiv.pipeline.types import PipelineConfig

        return resolve_stamp_radius(Path(output_dir), PipelineConfig())
    except Exception:  # noqa: BLE001 — best-effort; missing calibration is fine
        return 0


def _resolve_click(
    panel_index: dict, point: dict,
) -> tuple[str, Optional[int], tuple[float, float]]:
    """Resolve one Plotly click point to a label, in full-resolution coordinates.

    Returns ``("hit", label, (y, x))`` when the click lands on or near a real
    (non-ghost) cell, else ``("empty", None, (y, x))``.

    Two paths, because a click can land inside a stamp's disk without ever
    touching the thin outline ring that carries ``customdata``:

    * ``customdata`` present — the click hit an outline trace directly. Its
      label is authoritative; its position (when known) comes from the
      centroid table rather than the click point, since the point is exact
      only on the ring, not the centroid.
    * ``customdata`` absent — the panel's heatmap was hit instead (see
      ``figure.py``'s ``hoverinfo="none"``). The click point is in
      *downsampled* pixel coordinates (the heatmap trace carries no explicit
      ``x=``/``y=`` arrays), so it is rescaled by ``step`` before searching
      for the nearest real centroid within the session's stamp radius.
    """
    step = panel_index.get("step") or 1.0
    centroids = panel_index.get("centroid", {})

    customdata = point.get("customdata")
    if customdata:
        label = int(customdata[0])
        centroid = centroids.get(str(label))
        if centroid is not None:
            return "hit", label, (float(centroid[0]), float(centroid[1]))
        y = float(point.get("y", 0.0)) * step
        x = float(point.get("x", 0.0)) * step
        return "hit", label, (y, x)

    y = float(point.get("y", 0.0)) * step
    x = float(point.get("x", 0.0)) * step
    radius = panel_index.get("radius") or 0

    best_label: Optional[int] = None
    best_dist: Optional[float] = None
    for label_str, (cy, cx) in centroids.items():
        dist = ((float(cy) - y) ** 2 + (float(cx) - x) ** 2) ** 0.5
        if best_dist is None or dist < best_dist:
            best_label, best_dist = int(label_str), dist

    if best_label is not None and radius and best_dist <= radius:
        cy, cx = centroids[str(best_label)]
        return "hit", best_label, (float(cy), float(cx))
    return "empty", None, (y, x)


def _add_badges(
    figure: go.Figure, fov: TrackedFOV, rois, i: int, visible: bool,
) -> dict:
    """Lay down every cell's ``#N`` once, and record where each one landed.

    Adding traces on selection would mean shipping the panel's whole figure
    back — background included — on every click. Laying them down once and
    toggling ``visible`` keeps a selection, and the numbers switch, to a few
    bytes.
    """
    out: dict[str, int] = {}
    for roi in rois:
        cell = fov.cell_by_gcid(roi.global_cell_id)
        if cell is None:
            continue
        cy, cx = roi.centroid_yx
        figure.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="text", text=[f"#{cell.index}"],
            textfont=dict(color=color_for_match_status(roi.match_status),
                          size=_BADGE_SIZE, family="monospace"),
            hoverinfo="skip", showlegend=False, visible=bool(visible),
        ))
        out[str(roi.label_id)] = len(figure.data) - 1
    return out


def _downsampled(mean, step: float) -> Optional[np.ndarray]:
    """*mean* reduced by *step*, the factor shared across the whole sheet."""
    if mean is None:
        return None
    arr = np.asarray(mean)
    return arr if step <= 1 else arr[::int(step), ::int(step)]


def _quantized(mean) -> Optional[np.ndarray]:
    """*mean* rescaled onto uint8 across its own display window.

    Purely a payload measure: ``build_roi_figure`` sets the colour limits from
    the 1st and 99.5th percentiles, and this maps exactly that window onto the
    integer range, so the colour ramp is unchanged while the JSON carries
    three-character integers instead of eighteen-character floats — the
    difference between 5.7 MB and 1.9 MB for one 1024x1024 panel.

    Eight bits, not sixteen, because the browser renders this through an 8-bit
    colourscale anyway: the 256 levels are the ones a screenshot of the panel
    would have had regardless. The filmstrip crops keep the original floats.
    """
    if mean is None:
        return None
    arr = np.asarray(mean, dtype=np.float64)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99.5))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _scaled(roi: ROIRender, scale: float) -> ROIRender:
    if scale == 1.0:
        return roi
    cy, cx = roi.centroid_yx
    return ROIRender(
        label_id=roi.label_id, source_stage=roi.source_stage,
        gate_outcome=roi.gate_outcome, activity_type=roi.activity_type,
        area=roi.area, centroid_yx=(cy / scale, cx / scale),
        contours=[([y / scale for y in ys], [x / scale for x in xs])
                  for ys, xs in roi.contours],
        global_cell_id=roi.global_cell_id, match_status=roi.match_status,
    )


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


# ── editing ────────────────────────────────────────────────────────────────
#
# Every edit gesture ends the same way: append one or two ops to a JSONL log,
# replay them through apply_tracking_edits (no ROICaT, no GPU — see that
# function's docstring), invalidate the cached TrackedFOV, and rebuild the
# sheet from the fresh one. The helpers below build that "append, apply,
# rebuild" sequence for each gesture; _on_panel_click (in register_callbacks)
# is the dispatcher that decides which one a click means.
#
# The seven-tuple every helper returns matches _on_panel_click's Output order
# exactly: (selected, pickup, sheet, index, cell_list, header, msg). A no_update
# in a slot means that piece of the page is untouched by this particular click.

_NOOP = (no_update,) * 7


def _msg_only(msg: str) -> tuple:
    return (no_update, no_update, no_update, no_update, no_update, no_update, msg)


def _select_only(gcid: Optional[str]) -> tuple:
    return (gcid or no_update, no_update, no_update, no_update, no_update,
            no_update, no_update)


def _pickup_only(pickup: Optional[dict]) -> tuple:
    return (no_update, pickup, no_update, no_update, no_update, no_update, no_update)


def _rebuild(fov: TrackedFOV, show_numbers, selected: Optional[str],
            msg: str) -> tuple:
    """The result of any mutation: a fresh sheet, and the pickup cleared.

    Pickup always clears on a completed edit — a move or link that just
    finished has nothing left pending, and starting the *next* gesture from a
    stale first click would silently combine two unrelated actions.
    """
    sheet, index = _sheet(fov, bool(show_numbers))
    return (selected, None, sheet, index, _cell_list(fov, selected),
            _header(fov), msg)


def _session_index_for_stem(fov: TrackedFOV, stem: str) -> Optional[int]:
    for i, session in enumerate(fov.sessions):
        if session.stem == stem:
            return i
    return None


def _gcid_present_in_session(fov: TrackedFOV, gcid: Optional[str],
                             session_index: Optional[int]) -> bool:
    if not gcid or session_index is None:
        return False
    cell = fov.cell_by_gcid(gcid)
    return cell is not None and cell.present[session_index]


def _any_member_of(fov: TrackedFOV, gcid: Optional[str]) -> Optional[tuple]:
    """One ``(stem, label)`` this cell actually owns, to link a new one against.

    Merging happens by whole cell (see ``cell_edits.apply_match_ops``), so any
    one existing member is enough to pull the new centroid into the group.
    """
    cell = fov.cell_by_gcid(gcid)
    if cell is None:
        return None
    for i, label in enumerate(cell.local_label_ids):
        if label is not None:
            return fov.sessions[i].stem, label
    return None


def _report_msg(report, verb: str) -> str:
    if report.warnings:
        return f"{verb} — " + "; ".join(report.warnings)
    return verb


def _next_centroid_label(output_dir: Path) -> int:
    from roigbiv.pipeline.centroid_edits import load_centroid_ops, next_label
    from roigbiv.pipeline.centroid_masks import load_effective_centroids

    effective, _warnings = load_effective_centroids(output_dir)
    return next_label(effective, load_centroid_ops(output_dir))


def _do_add(output_dir: Path, y: float, x: float) -> int:
    from roigbiv.pipeline.centroid_edits import CentroidOp, append_centroid_op

    label = _next_centroid_label(output_dir)
    append_centroid_op(output_dir, CentroidOp.add(label, y, x))
    return label


def _do_delete(output_dir: Path, label: int) -> None:
    from roigbiv.pipeline.centroid_edits import CentroidOp, append_centroid_op

    append_centroid_op(output_dir, CentroidOp.delete(label))


def _do_move(output_dir: Path, label: int, y: float, x: float) -> None:
    from roigbiv.pipeline.centroid_edits import CentroidOp, append_centroid_op

    append_centroid_op(output_dir, CentroidOp.move(label, y, x))


def _do_link(fov_id: str, input_root: Path, members: list) -> None:
    from roigbiv.registry.cell_edits import MatchOp, append_match_op

    append_match_op(input_root, MatchOp.link(fov_id, members))


def _do_unlink(fov_id: str, input_root: Path, member: tuple) -> None:
    from roigbiv.registry.cell_edits import MatchOp, append_match_op

    append_match_op(input_root, MatchOp.unlink(fov_id, member))


def _apply_and_reload(fov_id: str, input_root: Path, registry_cfg):
    """Replay both logs into the registry, then return the fresh FOV.

    The single choke point every gesture below runs through — see the
    section docstring. Reloads through the *same* ``registry_cfg`` that was
    just written with, via ``load_tracked_fov_cached`` directly rather than
    the module's ``_load`` helper — that helper re-derives its config from
    ``get_app_state()``, which is one more hop than this function needs and
    depends on a live Flask request context this function otherwise wouldn't.
    """
    from roigbiv.registry import build_store
    from roigbiv.registry.cell_edits import apply_tracking_edits

    store = build_store(cfg=registry_cfg)
    report = apply_tracking_edits(fov_id, input_root, store)
    invalidate_tracked_fov(fov_id, cfg=registry_cfg)
    return load_tracked_fov_cached(fov_id, cfg=registry_cfg), report


def _tracking_is_active() -> bool:
    from roigbiv.ui.services.tracking_runner import get_tracking_runner

    return get_tracking_runner().snapshot().active


def _undo_last(fov: TrackedFOV, input_root: Path) -> Optional[str]:
    """Drop the most-recently-written op across every log this FOV owns.

    "Most recent" is decided by comparing the last line's timestamp across
    every session's centroid log *and* the FOV's one match log — undo has to
    span all of them, since a click could have been a link just as easily as
    a centroid edit. Two ops in the same clock tick is a coin flip between
    them; both are valid things to undo.
    """
    from roigbiv.pipeline.centroid_edits import load_centroid_ops, write_centroid_ops
    from roigbiv.registry.cell_edits import load_match_ops, write_match_ops

    candidates: list = []
    for session in fov.sessions:
        if session.output_dir is None:
            continue
        ops = load_centroid_ops(session.output_dir)
        if ops:
            candidates.append((ops[-1].ts, "centroid", session.output_dir, ops))
    match_ops = load_match_ops(input_root, fov.fov_id)
    if match_ops:
        candidates.append((match_ops[-1].ts, "match", None, match_ops))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    _ts, kind, output_dir, ops = candidates[-1]
    if kind == "centroid":
        write_centroid_ops(output_dir, ops[:-1])
        return f"undid the last centroid edit in {Path(output_dir).name}"
    write_match_ops(input_root, fov.fov_id, ops[:-1])
    return "undid the last cross-session link edit"


def _handle_edit_click(
    point: dict, panel_index: dict, mode: str, pickup: Optional[dict],
    fov: TrackedFOV, state, selected: Optional[str], show_numbers,
) -> tuple:
    """What one click means in edit mode, dispatched by the mode radio.

    Every branch that mutates anything ends by calling ``_apply_and_reload``
    and ``_rebuild`` — there is no incremental patch path for an edit, unlike
    read-only selection. See the section docstring for why.
    """
    if _tracking_is_active():
        return _msg_only(
            "tracking is running for this workspace — try again once it finishes")

    outcome, label, (y, x) = _resolve_click(panel_index, point)
    stem = panel_index.get("stem")
    session_id = panel_index.get("session_id")
    output_dir_str = panel_index.get("output_dir")
    if not stem or not output_dir_str:
        return _msg_only("this session has no on-disk output to edit")
    output_dir = Path(output_dir_str)
    fov_id = fov.fov_id
    input_root = state.workspace.input_root
    registry_cfg = state.registry_config

    if mode == "select":
        if outcome == "hit":
            return _select_only(panel_index.get("gcid", {}).get(str(label)))
        return _NOOP

    if mode == "add":
        if outcome == "hit":
            return _msg_only(
                f"there is already a cell here (label {label}) — "
                f"use move or delete instead")
        session_index = _session_index_for_stem(fov, stem)
        if selected and not _gcid_present_in_session(fov, selected, session_index):
            # "Place here": the selected cell is missing from this session —
            # add its missing centroid and link it in one gesture, rather than
            # making the human add-then-select-then-link three separate clicks
            # for what is, by far, the most common repair.
            new_label = _do_add(output_dir, y, x)
            anchor = _any_member_of(fov, selected)
            if anchor is not None:
                _do_link(fov_id, input_root, [anchor, (stem, new_label)])
            new_fov, report = _apply_and_reload(fov_id, input_root, registry_cfg)
            new_idx = _session_index_for_stem(new_fov, stem)
            new_gcid = (new_fov.gcid_for_label(new_idx, new_label)
                       if new_idx is not None else None) or selected
            return _rebuild(new_fov, show_numbers, new_gcid,
                            _report_msg(report, "added and linked a centroid"))

        new_label = _do_add(output_dir, y, x)
        new_fov, report = _apply_and_reload(fov_id, input_root, registry_cfg)
        new_idx = _session_index_for_stem(new_fov, stem)
        new_gcid = new_fov.gcid_for_label(new_idx, new_label) if new_idx is not None else None
        return _rebuild(new_fov, show_numbers, new_gcid,
                        _report_msg(report, "added a centroid"))

    if mode == "delete":
        if outcome != "hit" or label is None or label <= 0:
            return _NOOP
        deleted_gcid = panel_index.get("gcid", {}).get(str(label))
        _do_delete(output_dir, label)
        new_fov, report = _apply_and_reload(fov_id, input_root, registry_cfg)
        next_selected = None if selected == deleted_gcid else selected
        return _rebuild(new_fov, show_numbers, next_selected,
                        _report_msg(report, "deleted a centroid"))

    if mode == "move":
        if pickup is None:
            if outcome != "hit" or label is None or label <= 0:
                return _msg_only(
                    "click the cell to move first, then click where it should go")
            return _pickup_only({"stem": stem, "label": label,
                                 "session_id": session_id})
        if pickup.get("session_id") != session_id:
            # A move only makes sense within one session's own frame — a
            # click in a different panel restarts the pickup there instead of
            # silently moving a label into a session it was never part of.
            if outcome == "hit" and label is not None and label > 0:
                return _pickup_only({"stem": stem, "label": label,
                                     "session_id": session_id})
            return _pickup_only(None)

        _do_move(output_dir, pickup["label"], y, x)
        new_fov, report = _apply_and_reload(fov_id, input_root, registry_cfg)
        new_idx = _session_index_for_stem(new_fov, stem)
        new_gcid = (new_fov.gcid_for_label(new_idx, pickup["label"])
                   if new_idx is not None else None)
        return _rebuild(new_fov, show_numbers, new_gcid,
                        _report_msg(report, "moved a centroid"))

    if mode == "link":
        if outcome != "hit" or label is None or label <= 0:
            return _pickup_only(None) if pickup is not None else _NOOP
        member = (stem, label)
        if pickup is None:
            return _pickup_only({"stem": stem, "label": label,
                                 "session_id": session_id})
        first = (pickup["stem"], pickup["label"])
        if first == member:
            return _pickup_only(None)  # clicked the same one again — cancel

        first_idx = _session_index_for_stem(fov, pickup["stem"])
        member_idx = _session_index_for_stem(fov, stem)
        first_gcid = fov.gcid_for_label(first_idx, pickup["label"]) if first_idx is not None else None
        member_gcid = fov.gcid_for_label(member_idx, label) if member_idx is not None else None

        if first_gcid is not None and first_gcid == member_gcid:
            # The second click landed on a member already in the same cell as
            # the first — read that as "pull this one back out", not as a
            # link that would be a no-op anyway.
            _do_unlink(fov_id, input_root, member)
            verb = "unlinked"
        else:
            _do_link(fov_id, input_root, [first, member])
            verb = "linked"

        new_fov, report = _apply_and_reload(fov_id, input_root, registry_cfg)
        new_idx = _session_index_for_stem(new_fov, stem)
        new_gcid = new_fov.gcid_for_label(new_idx, label) if new_idx is not None else None
        return _rebuild(new_fov, show_numbers, new_gcid, _report_msg(report, verb))

    return _NOOP


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:

    @app.callback(
        Output(MODE_ROW_ID, "className"),
        Input(EDIT_ID, "value"),
    )
    def _on_edit_toggle(edit_on: Optional[bool]):
        base = "align-items-center g-2 mb-2 flex-wrap"
        return base if edit_on else f"{base} d-none"

    @app.callback(
        Output(SHEET_ID, "children"),
        Output(INDEX_ID, "data"),
        Output(LIST_ID, "children"),
        Output(HEADER_ID, "children"),
        Output(SELECTED_ID, "data"),
        Input(FOV_ID, "value"),
        State(NUMBERS_ID, "value"),
    )
    def _on_fov(fov_id: Optional[str], show_numbers: Optional[bool]):
        if not fov_id:
            return html.Div(), None, html.Div(), [], None
        try:
            fov = _load(fov_id)
        except Exception as exc:  # noqa: BLE001 — store or disk, both the user's
            return (user_error(exc, "loading this FOV's tracked cells"),
                    None, html.Div(), [], None)
        sheet, index = _sheet(fov, bool(show_numbers))
        return sheet, index, _cell_list(fov, None), _header(fov), None

    @app.callback(
        Output(SELECTED_ID, "data", allow_duplicate=True),
        Output(PICKUP_ID, "data"),
        Output(SHEET_ID, "children", allow_duplicate=True),
        Output(INDEX_ID, "data", allow_duplicate=True),
        Output(LIST_ID, "children", allow_duplicate=True),
        Output(HEADER_ID, "children", allow_duplicate=True),
        Output(EDIT_MSG_ID, "children"),
        Input({"type": PANEL_TYPE, "index": ALL}, "clickData"),
        Input(UNDO_ID, "n_clicks"),
        State(INDEX_ID, "data"),
        State(EDIT_ID, "value"),
        State(MODE_ID, "value"),
        State(PICKUP_ID, "data"),
        State(FOV_ID, "value"),
        State(SELECTED_ID, "data"),
        State(NUMBERS_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_panel_click(_clicks, _undo_clicks, index, edit_on, mode, pickup,
                        fov_id, selected, show_numbers):
        """Read-only: a click selects the cell that owns the clicked outline.

        Edit mode: dispatches to :func:`_handle_edit_click` / the undo button,
        each of which ends in a full sheet rebuild — see the "editing"
        section's module docstring for why there is no incremental path here.
        """
        if ctx.triggered_id == UNDO_ID:
            if not edit_on or not fov_id:
                return _NOOP
            state = get_app_state()
            if state.workspace is None:
                return _msg_only("no workspace selected")
            try:
                fov = _load(fov_id)
                undone = _undo_last(fov, state.workspace.input_root)
                if undone is None:
                    return _msg_only("nothing to undo")
                new_fov, _report = _apply_and_reload(
                    fov_id, state.workspace.input_root, state.registry_config)
            except Exception as exc:  # noqa: BLE001 — store or disk, the user's
                return _msg_only(f"undo failed: {exc}")
            return _rebuild(new_fov, show_numbers, selected, undone)

        triggered = ctx.triggered[0] if ctx.triggered else None
        if not triggered or not triggered.get("value") or not index:
            return _NOOP
        points = triggered["value"].get("points") or []
        point = points[0] if points else None
        if not point:
            return _NOOP
        panel = str(ctx.triggered_id["index"])
        panel_index = index.get(panel, {})

        if not edit_on:
            customdata = point.get("customdata")
            if not customdata:
                return _NOOP
            gcid = panel_index.get("gcid", {}).get(str(int(customdata[0])))
            return _select_only(gcid)

        if not fov_id:
            return _NOOP
        state = get_app_state()
        if state.workspace is None:
            return _msg_only("no workspace selected")
        try:
            fov = _load(fov_id)
            return _handle_edit_click(point, panel_index, mode or "select",
                                      pickup, fov, state, selected, show_numbers)
        except Exception as exc:  # noqa: BLE001 — store or disk, the user's
            return _msg_only(f"edit failed: {exc}")

    @app.callback(
        Output(PICKUP_ID, "data", allow_duplicate=True),
        Input(MODE_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_mode_change(_mode):
        """A pending move/link pickup means nothing once the mode changes."""
        return None

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
        except Exception:  # noqa: BLE001 — the sheet already reported it
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

    @app.callback(
        Output({"type": PANEL_TYPE, "index": ALL}, "figure"),
        Output(STRIP_ID, "children"),
        Output(LIST_ID, "children", allow_duplicate=True),
        Output(DRAWER_ID, "is_open"),
        Input(SELECTED_ID, "data"),
        Input(NUMBERS_ID, "value"),
        State(FOV_ID, "value"),
        State(INDEX_ID, "data"),
        prevent_initial_call=True,
    )
    def _on_select(selected: Optional[str], show_numbers: Optional[bool],
                   fov_id: Optional[str], index):
        n_panels = len(ctx.outputs_list[0]) if ctx.outputs_list else 0
        if not fov_id or not index:
            return [no_update] * n_panels, html.Div(), no_update, False
        try:
            fov = _load(fov_id)
        except Exception as exc:  # noqa: BLE001
            return ([no_update] * n_panels,
                    user_error(exc, "loading this FOV's tracked cells"),
                    no_update, False)

        patches = [
            _restyle(index.get(str(out["id"]["index"]), {}), selected,
                     bool(show_numbers))
            for out in (ctx.outputs_list[0] if n_panels else [])
        ]
        # Flipping the numbers switch changes only the badges. Rebuilding the
        # filmstrip and the cell list for it would re-crop every session for a
        # selection that did not move.
        if ctx.triggered_id == NUMBERS_ID:
            return patches, no_update, no_update, no_update

        cell = fov.cell_by_gcid(selected)
        strip = html.Div([_strip_header(fov, selected), cell_strip(fov, cell)])
        return patches, strip, _cell_list(fov, selected), cell is not None

    _register_clientside(app)


def _restyle(panel_index: dict, selected: Optional[str],
             show_numbers: bool) -> Patch:
    """Thicken the selected cell's outline and size its badge, in one panel.

    Every label is rewritten, not just the two that changed, so the patch does
    not need to know what was selected before it.

    A selected cell keeps its number even with the switch off — the switch
    controls the resting field of numbers, and suppressing the one the user
    just clicked would leave the sheet with no identity cue at all.
    """
    patch = Patch()
    gcids = panel_index.get("gcid", {})
    for label_id, trace_ids in panel_index.get("outline", {}).items():
        lit = selected is not None and gcids.get(label_id) == selected
        for trace_id in trace_ids:
            patch["data"][trace_id]["line"]["width"] = (
                _HIGHLIGHT_WIDTH if lit else _OUTLINE_WIDTH)
    for label_id, trace_id in panel_index.get("badge", {}).items():
        lit = selected is not None and gcids.get(label_id) == selected
        patch["data"][trace_id]["visible"] = bool(show_numbers) or lit
        patch["data"][trace_id]["textfont"]["size"] = (
            _BADGE_SIZE_SELECTED if lit else _BADGE_SIZE)
    return patch


def _load(fov_id: str) -> TrackedFOV:
    state = get_app_state()
    return load_tracked_fov_cached(fov_id, cfg=state.registry_config)


# ── clientside ─────────────────────────────────────────────────────────────


def _register_clientside(app: dash.Dash) -> None:
    """Panel zoom sync and the cell-rail collapse, both done in the browser.

    Neither goes through a Dash figure output on purpose. Zoom fires on every
    wheel tick, and a server round-trip that returns figures would ship the
    backgrounds back each time; the rail toggle only needs Plotly to re-measure
    after a CSS reflow. Both drive Plotly imperatively instead, the same way
    ``app.py`` resizes the Review page's plots.
    """
    # Copy one panel's view onto the others. Plotly re-emits a relayout for
    # every programmatic change, and `scaleanchor` makes it echo an adjusted
    # range back on top of that, so without the guard this loops.
    app.clientside_callback(
        """
        function(relayouts) {
            const dc = window.dash_clientside;
            if (window.__roigbivCellsSyncing || !window.Plotly) {
                return dc.no_update;
            }
            const trig = (dc.callback_context.triggered || [])[0];
            if (!trig || !trig.value) { return dc.no_update; }

            const ev = trig.value;
            let update = null;
            if (ev["xaxis.autorange"] || ev["autosize"]) {
                update = {"xaxis.autorange": true, "yaxis.autorange": true};
            } else if (ev["xaxis.range[0]"] !== undefined
                       && ev["yaxis.range[0]"] !== undefined) {
                update = {
                    "xaxis.range[0]": ev["xaxis.range[0]"],
                    "xaxis.range[1]": ev["xaxis.range[1]"],
                    "yaxis.range[0]": ev["yaxis.range[0]"],
                    "yaxis.range[1]": ev["yaxis.range[1]"]
                };
            }
            if (update === null) { return dc.no_update; }

            let source = null;
            try {
                source = JSON.parse(
                    trig.prop_id.slice(0, trig.prop_id.lastIndexOf("."))
                ).index;
            } catch (e) { return dc.no_update; }

            const plots = document.querySelectorAll(
                ".roigbiv-cells-panel .js-plotly-plot");
            window.__roigbivCellsSyncing = true;
            try {
                plots.forEach(function(plot, i) {
                    if (i !== source) { window.Plotly.relayout(plot, update); }
                });
            } finally {
                setTimeout(function() {
                    window.__roigbivCellsSyncing = false;
                }, 0);
            }
            return Date.now();
        }
        """,
        Output(SYNC_SINK_ID, "data"),
        Input({"type": PANEL_TYPE, "index": ALL}, "relayoutData"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(n_clicks) {
            const open = !(n_clicks % 2);
            setTimeout(function() {
                if (!window.Plotly) { return; }
                document.querySelectorAll(
                    ".roigbiv-cells-panel .js-plotly-plot"
                ).forEach(function(plot) {
                    try { window.Plotly.Plots.resize(plot); } catch (e) {}
                });
            }, 80);
            return [
                open ? "col-md-3" : "d-none",
                open ? "col-md-9" : "col-md-12",
                open ? "cells \\u25b8" : "cells \\u25c2"
            ];
        }
        """,
        Output(RAIL_COL_ID, "className"),
        Output(SHEET_COL_ID, "className"),
        Output(RAIL_TOGGLE_ID, "children"),
        Input(RAIL_TOGGLE_ID, "n_clicks"),
    )
