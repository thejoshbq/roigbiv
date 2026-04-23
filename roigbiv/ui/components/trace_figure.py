"""Plotly figure builders for the Viewer page's trace panels.

Three builders, all producing a single :class:`go.Figure`:

* :func:`build_mean_single`          — mean-FOV trace for one session.
* :func:`build_mean_multi`           — per-session mean-FOV traces overlaid on
                                       one axes (one line per session), with an
                                       optional grand-average line.
* :func:`build_roi_across_sessions`  — one ROI across every available session,
                                       overlaid with the session where it was
                                       selected drawn bold on top.

Each figure exposes the identifiers the criteria require: FOV id in the
figure title, session id/date in the legend, and for ROI views the
``local_label_id`` (per-session mask id) plus ``global_cell_id`` (the
persistent cross-session id) in the hover template.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from roigbiv.ui.services.colors import session_colors
from roigbiv.ui.services.trace_viz import SessionTraces, Y_LABELS


MEAN_COLOR = "rgba(46, 204, 113, 0.9)"
GRAND_AVERAGE_COLOR = "#111"
HIGHLIGHT_COLOR = "rgba(231, 76, 60, 0.95)"       # brick-red, prominent
EMPTY_HEIGHT = 420
MULTI_HEIGHT = 520


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


def _sort_sessions(sessions: list[SessionTraces]) -> list[SessionTraces]:
    return sorted(
        sessions,
        key=lambda s: (s.session_date or date.min, s.session_id or ""),
    )


def _grand_average(
    lines: list[tuple[np.ndarray, np.ndarray, Optional[float]]],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return ``(t, y_mean)`` if every line shares one ``fs``; else ``None``.

    Truncates all y-arrays to the shortest length before averaging with
    :func:`numpy.nanmean` along axis 0. Callers should only pass lines that
    successfully plotted.
    """
    if not lines:
        return None
    if any(fs is None or fs <= 0 for (_, _, fs) in lines):
        return None
    fs_set = {round(float(fs), 3) for (_, _, fs) in lines}
    if len(fs_set) != 1:
        return None
    ys = [np.asarray(y, dtype=np.float32) for (_, y, _) in lines]
    min_len = min(len(y) for y in ys)
    if min_len == 0:
        return None
    stack = np.stack([y[:min_len] for y in ys], axis=0)
    y_mean = np.nanmean(stack, axis=0)
    fs = float(next(iter(fs_set)))
    t = np.arange(min_len, dtype=np.float32) / fs
    return t, y_mean


def _multi_fs_mismatch(sessions: list[SessionTraces]) -> bool:
    fs_values = [s.fs for s in sessions]
    if any(f in (None, 0) for f in fs_values):
        return True
    return len({round(float(f), 3) for f in fs_values}) > 1


def _maybe_mixed_fs_note(
    fig: go.Figure, *, mixed_fs: bool, show_grand_average: bool,
) -> None:
    if not mixed_fs:
        return
    text = (
        "fs differs across sessions — grand average disabled"
        if show_grand_average
        else "fs differs across sessions"
    )
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0.0, y=1.02, xanchor="left", yanchor="bottom",
        showarrow=False, font={"size": 10, "color": "#c77"},
    )


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
    *,
    show_grand_average: bool = False,
) -> go.Figure:
    kinds = {s.kind for s in sessions} or {"dff"}
    ylabel = Y_LABELS[next(iter(kinds))]
    title = _fov_title(
        fov_meta, f"mean trace across {len(sessions)} session(s) · {ylabel}",
    )
    if not sessions:
        return _empty_fig(title, "No sessions registered on this FOV yet.")

    sessions_sorted = _sort_sessions(sessions)
    usable = [s for s in sessions_sorted
              if s.matrix is not None and s.matrix.size > 0]
    skipped = len(sessions_sorted) - len(usable)

    if not usable:
        return _empty_fig(title, "No sessions on this FOV have trace data yet.")

    palette = session_colors(len(usable))
    mixed_fs = _multi_fs_mismatch(usable)

    fig = go.Figure()
    lines_for_avg: list[tuple[np.ndarray, np.ndarray, Optional[float]]] = []

    for color, sess in zip(palette, usable):
        t = _time_axis(sess)
        y = np.nanmean(sess.matrix, axis=0).astype(np.float32)
        fig.add_trace(go.Scatter(
            x=t, y=y, mode="lines",
            line={"color": color, "width": 1.3},
            name=_session_label(sess),
            hovertemplate=(
                f"session {sess.session_id or '—'}<br>"
                f"t = %{{x:.2f}} s<br>{Y_LABELS[sess.kind]} = %{{y:.4f}}"
                "<extra></extra>"
            ),
        ))
        lines_for_avg.append((t, y, sess.fs or None))

    if show_grand_average:
        avg = _grand_average(lines_for_avg)
        if avg is not None:
            t_avg, y_avg = avg
            fig.add_trace(go.Scatter(
                x=t_avg, y=y_avg, mode="lines",
                line={"color": GRAND_AVERAGE_COLOR, "width": 2.5},
                name=f"grand average (n={len(usable)})",
                hovertemplate=(
                    "grand average<br>"
                    f"t = %{{x:.2f}} s<br>{ylabel} = %{{y:.4f}}<extra></extra>"
                ),
            ))

    any_fs = any(s.fs for s in usable)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=MULTI_HEIGHT,
        xaxis_title="time (s)" if any_fs else "frame",
        yaxis_title=ylabel,
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
        hovermode="x unified",
        legend={"orientation": "v", "x": 1.02, "y": 1.0, "xanchor": "left"},
    )
    _maybe_mixed_fs_note(fig, mixed_fs=mixed_fs,
                         show_grand_average=show_grand_average)
    if skipped:
        fig.add_annotation(
            text=f"skipped {skipped} session(s) with no trace data",
            xref="paper", yref="paper",
            x=1.0, y=1.02, xanchor="right", yanchor="bottom",
            showarrow=False, font={"size": 10, "color": "#888"},
        )
    return fig


# ── single-ROI across sessions ────────────────────────────────────────────


def build_roi_across_sessions(
    fov_meta: dict,
    sessions_with_rows: list[tuple[SessionTraces, int]],
    highlighted_session_id: Optional[str],
) -> go.Figure:
    """Overlay per-session traces of one ROI, highlighting the source session.

    ``sessions_with_rows`` is the list of ``(SessionTraces, row_index)`` pairs
    produced by :func:`collect_cross_session_traces` (or a single-session
    equivalent when no ``global_cell_id`` is available). ``highlighted_session_id``
    is the session whose line should be drawn bold and on top — this is the
    session the user clicked the ROI from in the canvas.
    """
    kinds = {s.kind for (s, _) in sessions_with_rows} or {"dff"}
    ylabel = Y_LABELS[next(iter(kinds))]
    title = _fov_title(
        fov_meta,
        f"ROI traces across {len(sessions_with_rows)} session(s) · {ylabel}",
    )
    if not sessions_with_rows:
        return _empty_fig(title, "Click an ROI in a session above to load traces.")

    entries_sorted = sorted(
        sessions_with_rows,
        key=lambda sr: (sr[0].session_date or date.min, sr[0].session_id or ""),
    )
    usable = [(s, r) for (s, r) in entries_sorted if s.matrix is not None]
    skipped = len(entries_sorted) - len(usable)

    if not usable:
        return _empty_fig(title, "No recoverable traces for this ROI.")

    palette = session_colors(len(usable))
    sessions_only = [s for (s, _) in usable]
    mixed_fs = _multi_fs_mismatch(sessions_only)

    highlight_pair: Optional[tuple[np.ndarray, np.ndarray, str, SessionTraces, int]] = None

    fig = go.Figure()
    for color, (sess, row) in zip(palette, usable):
        t = _time_axis(sess)
        y = np.asarray(sess.matrix[row], dtype=np.float32)
        local_lid = (sess.rows[row].get("local_label_id", "—")
                     if row < len(sess.rows) else "—")
        gcid = (sess.rows[row].get("global_cell_id")
                if row < len(sess.rows) else None) or "—"
        is_highlighted = (
            highlighted_session_id is not None
            and sess.session_id == highlighted_session_id
        )
        if is_highlighted:
            # draw last, on top — cache and continue
            highlight_pair = (t, y, color, sess, row)
            continue
        fig.add_trace(go.Scatter(
            x=t, y=y, mode="lines",
            line={"color": _dim(color), "width": 1.1},
            name=_session_label(sess),
            opacity=0.7,
            hovertemplate=(
                f"session {sess.session_id or '—'}<br>"
                f"local_label_id = {local_lid}<br>"
                f"global_cell_id = {gcid}<br>"
                f"t = %{{x:.2f}} s<br>{Y_LABELS[sess.kind]} = %{{y:.4f}}"
                "<extra></extra>"
            ),
        ))

    if highlight_pair is not None:
        t, y, _color, sess, row = highlight_pair
        local_lid = (sess.rows[row].get("local_label_id", "—")
                     if row < len(sess.rows) else "—")
        gcid = (sess.rows[row].get("global_cell_id")
                if row < len(sess.rows) else None) or "—"
        fig.add_trace(go.Scatter(
            x=t, y=y, mode="lines",
            line={"color": HIGHLIGHT_COLOR, "width": 2.5},
            name=f"{_session_label(sess)} · selected",
            hovertemplate=(
                f"session {sess.session_id or '—'} (selected)<br>"
                f"local_label_id = {local_lid}<br>"
                f"global_cell_id = {gcid}<br>"
                f"t = %{{x:.2f}} s<br>{Y_LABELS[sess.kind]} = %{{y:.4f}}"
                "<extra></extra>"
            ),
        ))

    any_fs = any(s.fs for s in sessions_only)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=MULTI_HEIGHT,
        xaxis_title="time (s)" if any_fs else "frame",
        yaxis_title=ylabel,
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
        hovermode="x unified",
        legend={"orientation": "v", "x": 1.02, "y": 1.0, "xanchor": "left"},
    )
    _maybe_mixed_fs_note(fig, mixed_fs=mixed_fs, show_grand_average=False)
    if skipped:
        fig.add_annotation(
            text=f"skipped {skipped} session(s) with no trace data",
            xref="paper", yref="paper",
            x=1.0, y=1.02, xanchor="right", yanchor="bottom",
            showarrow=False, font={"size": 10, "color": "#888"},
        )
    return fig


def _dim(rgba: str, *, alpha: float = 0.35) -> str:
    """Return an rgba string with the alpha channel replaced."""
    if not rgba.startswith("rgba"):
        return rgba
    inner = rgba[rgba.index("(") + 1: rgba.rindex(")")]
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) < 3:
        return rgba
    r, g, b = parts[:3]
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"
