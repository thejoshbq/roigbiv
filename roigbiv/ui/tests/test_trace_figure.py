"""Tests for the Viewer page's trace figure builders.

Covers:

* :func:`build_mean_multi` — per-session overlay with optional grand average.
* :func:`build_roi_across_sessions` — per-session ROI overlay with one session
  drawn bold and on top.
* :func:`session_colors` helper shape/format.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from roigbiv.ui.components.trace_figure import (
    GRAND_AVERAGE_COLOR,
    HIGHLIGHT_COLOR,
    build_mean_multi,
    build_roi_across_sessions,
)
from roigbiv.ui.services.colors import session_colors
from roigbiv.ui.services.trace_viz import SessionTraces


def _make_session(
    *,
    session_id: str,
    session_date: date,
    matrix: np.ndarray,
    fs: float = 7.5,
    kind: str = "dff",
    rows: list[dict] | None = None,
) -> SessionTraces:
    n_rows = int(matrix.shape[0])
    default_rows = [
        {"row_index": i, "local_label_id": i + 1, "global_cell_id": "cell-x"}
        for i in range(n_rows)
    ]
    return SessionTraces(
        session_id=session_id,
        session_date=session_date,
        stem=f"stem-{session_id}",
        output_dir=Path(f"/tmp/{session_id}"),
        fs=fs,
        n_frames=int(matrix.shape[1]),
        matrix=matrix.astype(np.float32),
        kind=kind,
        rows=rows or default_rows,
        source_label="pipeline",
    )


FOV_META = {"fov_id": "fov-1234-5678-9abc", "animal_id": "A1", "region": "V1"}


# ── build_mean_multi ──────────────────────────────────────────────────────


def test_mean_multi_empty_returns_placeholder_figure() -> None:
    fig = build_mean_multi(FOV_META, [])
    assert len(fig.data) == 0
    # the placeholder layout uses a single centered annotation
    assert fig.layout.annotations
    assert fig.layout.annotations[0].text.startswith("No sessions")


def test_mean_multi_overlays_one_trace_per_session() -> None:
    a = _make_session(
        session_id="sess-a", session_date=date(2026, 1, 1),
        matrix=np.ones((3, 10)),
    )
    b = _make_session(
        session_id="sess-b", session_date=date(2026, 1, 8),
        matrix=np.full((4, 10), 2.0),
    )
    fig = build_mean_multi(FOV_META, [a, b])
    assert len(fig.data) == 2
    # per-session means are uniform → we should see [1,1,...] and [2,2,...]
    ys = [np.asarray(tr.y) for tr in fig.data]
    assert pytest.approx(ys[0].mean()) == 1.0
    assert pytest.approx(ys[1].mean()) == 2.0


def test_mean_multi_grand_average_equals_analytical_mean() -> None:
    a = _make_session(
        session_id="sess-a", session_date=date(2026, 1, 1),
        matrix=np.ones((3, 10)),
    )
    b = _make_session(
        session_id="sess-b", session_date=date(2026, 1, 8),
        matrix=np.full((4, 10), 3.0),
    )
    fig = build_mean_multi(FOV_META, [a, b], show_grand_average=True)
    assert len(fig.data) == 3  # two sessions + grand-average
    avg_trace = fig.data[-1]
    assert avg_trace.line.color == GRAND_AVERAGE_COLOR
    # analytical: mean of (1.0, 3.0) = 2.0 everywhere
    assert np.allclose(np.asarray(avg_trace.y), 2.0)


def test_mean_multi_mixed_fs_drops_grand_average() -> None:
    a = _make_session(
        session_id="sess-a", session_date=date(2026, 1, 1),
        matrix=np.ones((3, 10)), fs=7.5,
    )
    b = _make_session(
        session_id="sess-b", session_date=date(2026, 1, 8),
        matrix=np.ones((3, 10)), fs=30.0,
    )
    fig = build_mean_multi(FOV_META, [a, b], show_grand_average=True)
    # no third trace because fs differs
    assert len(fig.data) == 2
    texts = [a.text for a in fig.layout.annotations]
    assert any("fs differs" in t for t in texts)


def test_mean_multi_truncates_to_shortest_session() -> None:
    a = _make_session(
        session_id="sess-a", session_date=date(2026, 1, 1),
        matrix=np.ones((3, 8)),
    )
    b = _make_session(
        session_id="sess-b", session_date=date(2026, 1, 8),
        matrix=np.full((3, 12), 3.0),
    )
    fig = build_mean_multi(FOV_META, [a, b], show_grand_average=True)
    avg_trace = fig.data[-1]
    # grand-average length should be min(n_frames) = 8
    assert len(avg_trace.y) == 8


# ── build_roi_across_sessions ─────────────────────────────────────────────


def _make_roi_session(
    *,
    session_id: str,
    session_date: date,
    local_label_id: int,
    trace: np.ndarray,
    fs: float = 7.5,
) -> tuple[SessionTraces, int]:
    row_index = 0
    matrix = trace.reshape(1, -1).astype(np.float32)
    sess = _make_session(
        session_id=session_id,
        session_date=session_date,
        matrix=matrix,
        fs=fs,
        rows=[{
            "row_index": row_index,
            "local_label_id": local_label_id,
            "global_cell_id": "gcid-target",
        }],
    )
    return sess, row_index


def test_roi_across_sessions_empty_returns_placeholder() -> None:
    fig = build_roi_across_sessions(FOV_META, [], highlighted_session_id=None)
    assert len(fig.data) == 0
    assert fig.layout.annotations
    assert "click" in fig.layout.annotations[0].text.lower()


def test_roi_across_sessions_overlays_one_trace_per_session() -> None:
    a = _make_roi_session(
        session_id="sess-a", session_date=date(2026, 1, 1),
        local_label_id=1, trace=np.ones(10),
    )
    b = _make_roi_session(
        session_id="sess-b", session_date=date(2026, 1, 8),
        local_label_id=5, trace=np.full(10, 3.0),
    )
    fig = build_roi_across_sessions(
        FOV_META, [a, b], highlighted_session_id="sess-a",
    )
    assert len(fig.data) == 2


def test_roi_across_sessions_highlight_line_has_highlight_color() -> None:
    a = _make_roi_session(
        session_id="sess-a", session_date=date(2026, 1, 1),
        local_label_id=1, trace=np.ones(10),
    )
    b = _make_roi_session(
        session_id="sess-b", session_date=date(2026, 1, 8),
        local_label_id=5, trace=np.full(10, 3.0),
    )
    fig = build_roi_across_sessions(
        FOV_META, [a, b], highlighted_session_id="sess-b",
    )
    # Highlighted line is drawn last and wears the brand highlight color.
    last = fig.data[-1]
    assert last.line.color == HIGHLIGHT_COLOR
    assert last.line.width == 2.5
    assert "selected" in (last.name or "")
    assert np.allclose(np.asarray(last.y), 3.0)


def test_roi_across_sessions_no_highlight_if_session_not_present() -> None:
    a = _make_roi_session(
        session_id="sess-a", session_date=date(2026, 1, 1),
        local_label_id=1, trace=np.ones(10),
    )
    fig = build_roi_across_sessions(
        FOV_META, [a], highlighted_session_id="sess-unknown",
    )
    # Still renders one trace, just with no highlighted styling.
    assert len(fig.data) == 1
    assert fig.data[0].line.color != HIGHLIGHT_COLOR


def test_roi_across_sessions_sorts_by_session_date() -> None:
    later = _make_roi_session(
        session_id="sess-later", session_date=date(2026, 2, 1),
        local_label_id=5, trace=np.full(10, 2.0),
    )
    earlier = _make_roi_session(
        session_id="sess-earlier", session_date=date(2026, 1, 1),
        local_label_id=1, trace=np.full(10, 1.0),
    )
    # Pass in reverse date order; builder should sort ascending.
    fig = build_roi_across_sessions(
        FOV_META, [later, earlier], highlighted_session_id=None,
    )
    first_name = fig.data[0].name
    assert "sess-ear" in first_name  # earlier session appears first


# ── session_colors helper ─────────────────────────────────────────────────


def test_session_colors_length_and_format() -> None:
    colors = session_colors(5)
    assert len(colors) == 5
    for c in colors:
        assert c.startswith("rgba(")


def test_session_colors_zero_returns_empty() -> None:
    assert session_colors(0) == []


def test_session_colors_one_is_midpoint() -> None:
    colors = session_colors(1)
    assert len(colors) == 1
    assert colors[0].startswith("rgba(")
