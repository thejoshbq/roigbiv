"""Plotly figure builders for the Viewer page's trace panels.

Four builders, all producing a single :class:`go.Figure`:

* :func:`build_mean_single`          — mean-FOV trace for one session.
* :func:`build_mean_multi`           — per-session mean-FOV traces overlaid on
                                       one axes (one line per session), with an
                                       optional grand-average line.
* :func:`build_roi_across_sessions`  — one ROI across every available session,
                                       overlaid with the session where it was
                                       selected drawn bold on top.
* :func:`build_roi_single`           — single-session ROI: dF/F + transient
                                       markers + optional raw/neuropil subplot
                                       + activity-type badge.

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
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

from roigbiv.ui.services.colors import session_colors
from roigbiv.ui.services.theme import (
    axis_muted_color,
    plotly_template,
    warning_color,
)
from roigbiv.ui.services.trace_viz import SessionTraces, SingleROIData, Y_LABELS


MEAN_COLOR = "rgba(46, 204, 113, 0.9)"
GRAND_AVERAGE_COLOR = "#111"
HIGHLIGHT_COLOR = "rgba(231, 76, 60, 0.95)"       # brick-red, prominent


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


def _empty_fig(title: str, message: str,
               theme: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template=plotly_template(theme),
        autosize=True,
        annotations=[{
            "text": message,
            "showarrow": False,
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "font": {"size": 14, "color": axis_muted_color(theme)},
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
    theme: Optional[str] = None,
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
        showarrow=False, font={"size": 10, "color": warning_color(theme)},
    )


# ── mean-FOV views ────────────────────────────────────────────────────────


def build_mean_single(
    fov_meta: dict,
    sess: SessionTraces,
    *,
    gate_filter: Optional[set[str]] = None,
    theme: Optional[str] = None,
) -> go.Figure:
    title = _fov_title(
        fov_meta,
        f"mean trace · {_session_label(sess)} · {Y_LABELS[sess.kind]}",
    )
    if sess.matrix is None or sess.matrix.size == 0:
        return _empty_fig(title, _note_for(sess), theme=theme)

    matrix = sess.matrix
    if gate_filter is not None:
        if not sess.rows:
            return _empty_fig(
                title, "No ROI metadata — cannot apply gate filter.", theme=theme)
        idx = [r["row_index"] for r in sess.rows
               if r.get("gate_outcome") in gate_filter]
        matrix = matrix[idx] if idx else np.empty(
            (0, sess.n_frames), dtype=np.float32)
    if matrix.size == 0:
        return _empty_fig(title, "No accepted ROIs in this session.", theme=theme)

    t = _time_axis(sess)
    mean = np.nanmean(matrix, axis=0)
    n_rois = int(matrix.shape[0])

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
        template=plotly_template(theme),
        autosize=True,
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
            showarrow=False,
            font={"size": 10, "color": axis_muted_color(theme)},
        )
    return fig


def build_mean_multi(
    fov_meta: dict,
    sessions: list[SessionTraces],
    *,
    gate_filter: Optional[set[str]] = None,
    show_grand_average: bool = False,
    theme: Optional[str] = None,
) -> go.Figure:
    kinds = {s.kind for s in sessions} or {"dff"}
    ylabel = Y_LABELS[next(iter(kinds))]
    title = _fov_title(
        fov_meta, f"mean trace across {len(sessions)} session(s) · {ylabel}",
    )
    if not sessions:
        return _empty_fig(title, "No sessions registered on this FOV yet.",
                          theme=theme)

    sessions_sorted = _sort_sessions(sessions)
    usable = [s for s in sessions_sorted
              if s.matrix is not None and s.matrix.size > 0]
    skipped = len(sessions_sorted) - len(usable)

    if not usable:
        return _empty_fig(title,
                          "No sessions on this FOV have trace data yet.",
                          theme=theme)

    palette = session_colors(len(usable))

    fig = go.Figure()
    lines_for_avg: list[tuple[np.ndarray, np.ndarray, Optional[float]]] = []
    gate_skipped = 0

    for color, sess in zip(palette, usable):
        matrix = sess.matrix
        if gate_filter is not None:
            if not sess.rows:
                gate_skipped += 1
                continue
            idx = [r["row_index"] for r in sess.rows
                   if r.get("gate_outcome") in gate_filter]
            matrix = matrix[idx] if idx else np.empty(
                (0, sess.n_frames), dtype=np.float32)
        if matrix.size == 0:
            skipped += 1
            continue
        t = _time_axis(sess)
        y = np.nanmean(matrix, axis=0).astype(np.float32)
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

    if not lines_for_avg:
        return _empty_fig(title, "No accepted ROIs in any selected session.",
                          theme=theme)

    # Recompute mixed_fs from contributing sessions only; gate_filter may have
    # pruned sessions that would have caused a false mismatch warning.
    plotted_fs = [fs for (_, _, fs) in lines_for_avg]
    mixed_fs = (any(f in (None, 0) for f in plotted_fs) or
                len({round(float(f), 3) for f in plotted_fs if f}) > 1)

    if show_grand_average:
        avg = _grand_average(lines_for_avg)
        if avg is not None:
            t_avg, y_avg = avg
            fig.add_trace(go.Scatter(
                x=t_avg, y=y_avg, mode="lines",
                line={"color": GRAND_AVERAGE_COLOR, "width": 2.5},
                name=f"grand average (n={len(lines_for_avg)})",
                hovertemplate=(
                    "grand average<br>"
                    f"t = %{{x:.2f}} s<br>{ylabel} = %{{y:.4f}}<extra></extra>"
                ),
            ))

    any_fs = any(fs for (_, _, fs) in lines_for_avg)
    fig.update_layout(
        title=title,
        template=plotly_template(theme),
        autosize=True,
        xaxis_title="time (s)" if any_fs else "frame",
        yaxis_title=ylabel,
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
        hovermode="x unified",
        legend={"orientation": "v", "x": 1.02, "y": 1.0, "xanchor": "left"},
    )
    _maybe_mixed_fs_note(fig, mixed_fs=mixed_fs,
                         show_grand_average=show_grand_average,
                         theme=theme)
    if skipped:
        fig.add_annotation(
            text=f"skipped {skipped} session(s) with no trace data",
            xref="paper", yref="paper",
            x=1.0, y=1.02, xanchor="right", yanchor="bottom",
            showarrow=False,
            font={"size": 10, "color": axis_muted_color(theme)},
        )
    if gate_skipped:
        fig.add_annotation(
            text=f"skipped {gate_skipped} session(s) with no ROI metadata",
            xref="paper", yref="paper",
            x=1.0, y=1.06, xanchor="right", yanchor="bottom",
            showarrow=False,
            font={"size": 10, "color": axis_muted_color(theme)},
        )
    return fig


# ── single-ROI across sessions ────────────────────────────────────────────


def build_roi_across_sessions(
    fov_meta: dict,
    sessions_with_rows: list[tuple[SessionTraces, int]],
    highlighted_session_id: Optional[str],
    *,
    theme: Optional[str] = None,
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
        return _empty_fig(title,
                          "Click an ROI in a session above to load traces.",
                          theme=theme)

    entries_sorted = sorted(
        sessions_with_rows,
        key=lambda sr: (sr[0].session_date or date.min, sr[0].session_id or ""),
    )
    usable = [(s, r) for (s, r) in entries_sorted if s.matrix is not None]
    skipped = len(entries_sorted) - len(usable)

    if not usable:
        return _empty_fig(title, "No recoverable traces for this ROI.",
                          theme=theme)

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
        template=plotly_template(theme),
        autosize=True,
        xaxis_title="time (s)" if any_fs else "frame",
        yaxis_title=ylabel,
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
        hovermode="x unified",
        legend={"orientation": "v", "x": 1.02, "y": 1.0, "xanchor": "left"},
    )
    _maybe_mixed_fs_note(fig, mixed_fs=mixed_fs, show_grand_average=False,
                         theme=theme)
    if skipped:
        fig.add_annotation(
            text=f"skipped {skipped} session(s) with no trace data",
            xref="paper", yref="paper",
            x=1.0, y=1.02, xanchor="right", yanchor="bottom",
            showarrow=False,
            font={"size": 10, "color": axis_muted_color(theme)},
        )
    return fig


# ── single-session single-ROI view ───────────────────────────────────────


_ACTIVITY_COLORS: dict[str, str] = {
    "phasic":    "rgba(46, 204, 113, 0.85)",
    "sparse":    "rgba(52, 152, 219, 0.85)",
    "tonic":     "rgba(230, 126, 34, 0.85)",
    "silent":    "rgba(127, 140, 141, 0.85)",
    "ambiguous": "rgba(241, 196, 15, 0.85)",
}
_ACTIVITY_COLOR_DEFAULT = "rgba(149, 165, 166, 0.85)"

DFF_COLOR = "rgba(231, 76, 60, 0.95)"          # matches HIGHLIGHT_COLOR
F_CORRECTED_COLOR = "rgba(52, 152, 219, 0.65)"
F_NEUROPIL_COLOR = "rgba(149, 165, 166, 0.50)"


def build_roi_single(
    fov_meta: dict,
    data: SingleROIData,
    *,
    theme: Optional[str] = None,
) -> go.Figure:
    """Single-session ROI: dF/F with transient markers + raw/neuropil subplot.

    Primary panel: dF/F trace with scipy MAD-threshold peak markers and an
    activity-type badge. Secondary panel (when f_corrected or f_neuropil are
    available): corrected F (solid) and neuropil F (dotted) on a shared x-axis.
    """
    title = _fov_title(fov_meta, f"ROI {data.local_label_id} · dF/F")

    if data.dff is None:
        return _empty_fig(title, "dF/F not available for this ROI.", theme=theme)

    dff = np.asarray(data.dff, dtype=np.float32)
    has_secondary = data.f_corrected is not None or data.f_neuropil is not None

    if has_secondary:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.68, 0.32],
            shared_xaxes=True,
            vertical_spacing=0.06,
        )
        dff_row, sec_row = 1, 2
    else:
        fig = go.Figure()
        dff_row = sec_row = None

    # Time axis
    if data.fs and data.n_frames:
        t = np.arange(data.n_frames, dtype=np.float32) / float(data.fs)
        xaxis_label = "time (s)"
    else:
        t = np.arange(len(dff), dtype=np.float32)
        xaxis_label = "frame"

    n = min(len(t), len(dff))
    t, dff = t[:n], dff[:n]

    add_kw: dict = ({"row": dff_row, "col": 1} if has_secondary else {})

    fig.add_trace(go.Scatter(
        x=t, y=dff,
        mode="lines",
        line={"color": DFF_COLOR, "width": 1.5},
        name="dF/F",
        hovertemplate="t = %{x:.2f} s<br>dF/F = %{y:.4f}<extra></extra>",
    ), **add_kw)

    # Transient markers — MAD-based threshold, handles NaN (silent ROIs)
    valid_mask = ~np.isnan(dff)
    if np.any(valid_mask):
        dff_filled = dff.copy()
        med = float(np.nanmedian(dff_filled))
        dff_filled[~valid_mask] = med
        mad = float(np.nanmedian(np.abs(dff_filled - med))) or 1e-6
        min_dist = max(1, int((data.fs or 7.5) * 0.5))
        try:
            peaks, _ = find_peaks(
                dff_filled, height=med + 2.0 * mad, distance=min_dist,
            )
        except Exception:
            peaks = np.array([], dtype=int)
        if len(peaks):
            fig.add_trace(go.Scatter(
                x=t[peaks], y=dff[peaks],
                mode="markers",
                marker={
                    "symbol": "triangle-up", "size": 7,
                    "color": "rgba(255, 255, 255, 0.9)",
                    "line": {"color": DFF_COLOR, "width": 1},
                },
                name="transients",
                hovertemplate=(
                    "transient<br>t = %{x:.2f} s<br>dF/F = %{y:.4f}"
                    "<extra></extra>"
                ),
            ), **add_kw)

    # Secondary panel — raw channels
    if has_secondary:
        if data.f_corrected is not None:
            fc = np.asarray(data.f_corrected, dtype=np.float32)
            n2 = min(len(t), len(fc))
            fig.add_trace(go.Scatter(
                x=t[:n2], y=fc[:n2],
                mode="lines",
                line={"color": F_CORRECTED_COLOR, "width": 1.0},
                name="F corrected",
                hovertemplate="t = %{x:.2f} s<br>F = %{y:.2f}<extra></extra>",
            ), row=sec_row, col=1)
        if data.f_neuropil is not None:
            fn = np.asarray(data.f_neuropil, dtype=np.float32)
            n2 = min(len(t), len(fn))
            fig.add_trace(go.Scatter(
                x=t[:n2], y=fn[:n2],
                mode="lines",
                line={"color": F_NEUROPIL_COLOR, "width": 0.9, "dash": "dot"},
                name="F neuropil",
                hovertemplate="t = %{x:.2f} s<br>F_neu = %{y:.2f}<extra></extra>",
            ), row=sec_row, col=1)

    fig.update_layout(
        title=title,
        template=plotly_template(theme),
        autosize=True,
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
        hovermode="x unified",
        legend={"orientation": "h", "x": 0.0, "y": -0.12,
                "xanchor": "left", "yanchor": "top"},
    )

    if has_secondary:
        fig.update_yaxes(title_text="dF/F", row=1, col=1)
        fig.update_yaxes(title_text="F (a.u.)", row=2, col=1)
        fig.update_xaxes(title_text=xaxis_label, row=2, col=1)
    else:
        fig.update_layout(
            xaxis_title=xaxis_label,
            yaxis_title="dF/F",
        )

    # Activity-type badge
    activity = data.activity_type or "—"
    fig.add_annotation(
        text=f" {activity} ",
        xref="paper", yref="paper",
        x=0.01, y=1.0,
        xanchor="left", yanchor="bottom",
        showarrow=False,
        font={"size": 11, "color": "#fff"},
        bgcolor=_ACTIVITY_COLORS.get(activity, _ACTIVITY_COLOR_DEFAULT),
        borderpad=3,
    )

    if data.source_label and data.source_label != "pipeline":
        fig.add_annotation(
            text=f"source: {data.source_label}",
            xref="paper", yref="paper",
            x=1.0, y=1.02,
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font={"size": 10, "color": axis_muted_color(theme)},
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
