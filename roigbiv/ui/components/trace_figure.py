"""Plotly figure builders for the Traces page.

Four view modes, all producing a single :class:`go.Figure`:

* :func:`build_mean_single`  — mean-FOV trace, one session (single panel)
* :func:`build_mean_multi`   — mean-FOV trace across sessions (stacked subplots)
* :func:`build_roi_single`   — single ROI in one session (single panel)
* :func:`build_roi_multi`    — one persistent cell across sessions (stacked subplots)

Each figure exposes the identifiers the criteria require: FOV id in the
figure title, session id/date in subplot titles, and for ROI views the
``local_label_id`` (per-session mask id) plus ``global_cell_id`` (the
persistent cross-session id) in both the subplot title and the hover
template.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from roigbiv.ui.services.colors import color_for_gcid
from roigbiv.ui.services.trace_viz import SessionTraces, Y_LABELS


LINE_COLOR = "rgba(52, 152, 219, 0.9)"
MEAN_COLOR = "rgba(46, 204, 113, 0.9)"
EMPTY_HEIGHT = 420


# ── shared helpers ────────────────────────────────────────────────────────


def _time_axis(sess: SessionTraces) -> np.ndarray:
    if sess.fs and sess.n_frames:
        return np.arange(sess.n_frames, dtype=np.float32) / float(sess.fs)
    return np.arange(sess.n_frames, dtype=np.float32)


def _session_label(sess: SessionTraces) -> str:
    date_str = sess.session_date.isoformat() if sess.session_date else "—"
    sid = (sess.session_id or "")[:8]
    return f"{date_str} · {sid}…" if sid else date_str


def _fov_title(fov_meta: dict, suffix: str) -> str:
    animal = fov_meta.get("animal_id") or "—"
    region = fov_meta.get("region") or "—"
    fov_short = (fov_meta.get("fov_id") or "")[:8]
    return f"FOV {fov_short}… · {animal} / {region} — {suffix}"


def _empty_fig(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=EMPTY_HEIGHT,
        annotations=[{
            "text": message,
            "showarrow": False,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "font": {"size": 14, "color": "#888"},
        }],
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _note_for(sess: SessionTraces) -> str:
    base = sess.note or "No trace data."
    if sess.source_label and sess.source_label != "pipeline":
        return f"{base} (source: {sess.source_label})"
    return base


# ── mean-FOV views ────────────────────────────────────────────────────────


def build_mean_single(
    fov_meta: dict,
    sess: SessionTraces,
) -> go.Figure:
    title = _fov_title(
        fov_meta,
        f"mean trace · {_session_label(sess)} · {Y_LABELS[sess.kind]}",
    )
    if sess.matrix is None or sess.matrix.size == 0:
        return _empty_fig(title, _note_for(sess))

    t = _time_axis(sess)
    mean = np.nanmean(sess.matrix, axis=0)
    n_rois = int(sess.matrix.shape[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=mean,
        mode="lines",
        line={"color": MEAN_COLOR, "width": 1.5},
        name=f"mean of {n_rois} ROIs",
        hovertemplate=(
            f"session {sess.session_id or '—'}<br>"
            f"t = %{{x:.2f}} s<br>{Y_LABELS[sess.kind]} = %{{y:.4f}}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=EMPTY_HEIGHT,
        xaxis_title="time (s)" if sess.fs else "frame",
        yaxis_title=Y_LABELS[sess.kind],
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        hovermode="x unified",
    )
    if sess.source_label and sess.source_label != "pipeline":
        fig.add_annotation(
            text=f"source: {sess.source_label}",
            xref="paper", yref="paper",
            x=1.0, y=1.02, xanchor="right", yanchor="bottom",
            showarrow=False, font={"size": 10, "color": "#888"},
        )
    return fig


def build_mean_multi(
    fov_meta: dict,
    sessions: list[SessionTraces],
) -> go.Figure:
    kinds = {s.kind for s in sessions} or {"dff"}
    ylabel = Y_LABELS[next(iter(kinds))]
    title = _fov_title(
        fov_meta, f"mean trace across {len(sessions)} session(s) · {ylabel}",
    )
    if not sessions:
        return _empty_fig(title, "No sessions registered on this FOV yet.")

    subplot_titles = [_session_label(s) for s in sessions]
    fs_set = {round(s.fs, 3) for s in sessions if s.fs}
    mixed_fs = len(fs_set) > 1

    fig = make_subplots(
        rows=len(sessions), cols=1,
        shared_xaxes=False,
        vertical_spacing=max(0.02, 0.25 / max(len(sessions), 1)),
        subplot_titles=subplot_titles,
    )

    for i, sess in enumerate(sessions, start=1):
        if sess.matrix is None or sess.matrix.size == 0:
            fig.add_annotation(
                text=_note_for(sess),
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, showarrow=False,
                font={"size": 12, "color": "#888"},
                row=i, col=1,
            )
            continue
        t = _time_axis(sess)
        mean = np.nanmean(sess.matrix, axis=0)
        fig.add_trace(
            go.Scatter(
                x=t, y=mean, mode="lines",
                line={"color": MEAN_COLOR, "width": 1.3},
                name=_session_label(sess),
                hovertemplate=(
                    f"session {sess.session_id or '—'}<br>"
                    f"t = %{{x:.2f}} s<br>{Y_LABELS[sess.kind]} = %{{y:.4f}}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=i, col=1,
        )
        fig.update_xaxes(
            title_text="time (s)" if sess.fs else "frame",
            row=i, col=1,
        )
        fig.update_yaxes(title_text=ylabel, row=i, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(260, 220 * len(sessions)),
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
        hovermode="x unified",
    )
    if mixed_fs:
        fig.add_annotation(
            text="fs differs across sessions — each panel uses its own time axis",
            xref="paper", yref="paper",
            x=0.0, y=1.02, xanchor="left", yanchor="bottom",
            showarrow=False, font={"size": 10, "color": "#c77"},
        )
    return fig


# ── single-ROI views ──────────────────────────────────────────────────────


def _roi_id_suffix(local_label_id: int, gcid: Optional[str]) -> str:
    if gcid:
        return f"local {local_label_id} · gcid {gcid[:8]}…"
    return f"local {local_label_id} · gcid —"


def build_roi_single(
    fov_meta: dict,
    sess: SessionTraces,
    local_label_id: int,
) -> go.Figure:
    row = sess.row_for_local_label(local_label_id)
    gcid = None
    if row is not None and row < len(sess.rows):
        gcid = sess.rows[row].get("global_cell_id")

    title = _fov_title(
        fov_meta,
        f"ROI {_roi_id_suffix(local_label_id, gcid)} · "
        f"{_session_label(sess)} · {Y_LABELS[sess.kind]}",
    )
    if sess.matrix is None or row is None:
        msg = (f"ROI {local_label_id} not present in this session's bundle."
               if sess.matrix is not None else _note_for(sess))
        return _empty_fig(title, msg)

    t = _time_axis(sess)
    y = np.asarray(sess.matrix[row], dtype=np.float32)
    color = color_for_gcid(gcid) if gcid else LINE_COLOR

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=y, mode="lines",
        line={"color": color, "width": 1.5},
        name=f"ROI {local_label_id}",
        hovertemplate=(
            f"session {sess.session_id or '—'}<br>"
            f"local_label_id = {local_label_id}<br>"
            f"global_cell_id = {gcid or '—'}<br>"
            f"t = %{{x:.2f}} s<br>{Y_LABELS[sess.kind]} = %{{y:.4f}}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=EMPTY_HEIGHT,
        xaxis_title="time (s)" if sess.fs else "frame",
        yaxis_title=Y_LABELS[sess.kind],
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        hovermode="x unified",
    )
    return fig


def build_roi_multi(
    fov_meta: dict,
    global_cell_id: str,
    sessions_with_rows: list[tuple[SessionTraces, int]],
) -> go.Figure:
    kinds = {s.kind for (s, _) in sessions_with_rows} or {"dff"}
    ylabel = Y_LABELS[next(iter(kinds))]
    title = _fov_title(
        fov_meta,
        f"cell {global_cell_id[:8]}… across {len(sessions_with_rows)} session(s) · {ylabel}",
    )
    if not sessions_with_rows:
        return _empty_fig(title,
                          "This cell has no observations with recoverable traces.")

    color = color_for_gcid(global_cell_id)
    subplot_titles = [
        f"{_session_label(s)} · local {s.rows[row].get('local_label_id', '—')}"
        for (s, row) in sessions_with_rows
    ]
    fs_set = {round(s.fs, 3) for (s, _) in sessions_with_rows if s.fs}
    mixed_fs = len(fs_set) > 1

    fig = make_subplots(
        rows=len(sessions_with_rows), cols=1,
        shared_xaxes=False,
        vertical_spacing=max(0.02, 0.25 / max(len(sessions_with_rows), 1)),
        subplot_titles=subplot_titles,
    )

    for i, (sess, row) in enumerate(sessions_with_rows, start=1):
        if sess.matrix is None:
            fig.add_annotation(
                text=_note_for(sess),
                xref="x domain", yref="y domain",
                x=0.5, y=0.5, showarrow=False,
                font={"size": 12, "color": "#888"},
                row=i, col=1,
            )
            continue
        t = _time_axis(sess)
        y = np.asarray(sess.matrix[row], dtype=np.float32)
        local_lid = sess.rows[row].get("local_label_id", "—")
        fig.add_trace(
            go.Scatter(
                x=t, y=y, mode="lines",
                line={"color": color, "width": 1.4},
                name=_session_label(sess),
                hovertemplate=(
                    f"session {sess.session_id or '—'}<br>"
                    f"local_label_id = {local_lid}<br>"
                    f"global_cell_id = {global_cell_id}<br>"
                    f"t = %{{x:.2f}} s<br>{Y_LABELS[sess.kind]} = %{{y:.4f}}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=i, col=1,
        )
        fig.update_xaxes(
            title_text="time (s)" if sess.fs else "frame",
            row=i, col=1,
        )
        fig.update_yaxes(title_text=ylabel, row=i, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(260, 220 * len(sessions_with_rows)),
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
        hovermode="x unified",
    )
    if mixed_fs:
        fig.add_annotation(
            text="fs differs across sessions — each panel uses its own time axis",
            xref="paper", yref="paper",
            x=0.0, y=1.02, xanchor="left", yanchor="bottom",
            showarrow=False, font={"size": 10, "color": "#c77"},
        )
    return fig
