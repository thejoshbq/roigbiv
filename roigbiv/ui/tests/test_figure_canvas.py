"""Regression guard: FOV canvas is always pure black, colorscale is always Greys."""
import numpy as np
import pytest

from roigbiv.ui.components.figure import build_roi_figure, trace_index_map
from roigbiv.ui.services.loaders import ROIRender


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_pure_black_canvas(theme):
    mean = np.random.default_rng(0).random((8, 8)).astype(np.float32)
    fig = build_roi_figure(mean=mean, rois=[], theme=theme)
    assert fig.layout.plot_bgcolor == "#000000", (
        f"plot_bgcolor must be #000000 for theme={theme!r}, got {fig.layout.plot_bgcolor!r}"
    )
    assert fig.layout.paper_bgcolor == "#000000", (
        f"paper_bgcolor must be #000000 for theme={theme!r}, got {fig.layout.paper_bgcolor!r}"
    )


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_colorscale_reversed(theme):
    """reversescale must be True: Plotly renders Greys reversed (0=dark, 1=bright).

    Plotly stores the original (un-reversed) colorscale stops and flips them
    only during rendering, so we check the reversescale flag directly.
    """
    mean = np.ones((4, 4), dtype=np.float32)
    fig = build_roi_figure(mean=mean, rois=[], theme=theme)
    heatmap = next(t for t in fig.data if t.type == "heatmap")
    assert heatmap.reversescale is True, (
        f"theme={theme!r}: reversescale must be True for dark-background rendering, "
        f"got {heatmap.reversescale!r}"
    )


def test_zmin_zmax_set():
    mean = np.linspace(0, 1, 64).reshape(8, 8).astype(np.float32)
    fig = build_roi_figure(mean=mean, rois=[], theme=None)
    heatmap = next(t for t in fig.data if t.type == "heatmap")
    assert heatmap.zmin is not None, "zmin should be set for contrast clipping"
    assert heatmap.zmax is not None, "zmax should be set for contrast clipping"
    assert heatmap.zmax > heatmap.zmin


def test_flat_image_guard():
    """Flat (all-same-value) images must not produce zmin == zmax."""
    mean = np.zeros((4, 4), dtype=np.float32)
    fig = build_roi_figure(mean=mean, rois=[], theme=None)
    heatmap = next(t for t in fig.data if t.type == "heatmap")
    assert heatmap.zmax > heatmap.zmin, "zmax must exceed zmin even for flat input"


def test_background_clicks_reach_the_callback():
    """hoverinfo="skip" swallows plotly_click too, not just the tooltip.

    /cells' edit mode needs a click on empty background (no ROI under the
    cursor) to fire, so it can place a new centroid there.
    """
    mean = np.zeros((4, 4), dtype=np.float32)
    fig = build_roi_figure(mean=mean, rois=[], theme=None)
    heatmap = next(t for t in fig.data if t.type == "heatmap")
    assert heatmap.hoverinfo == "none"


# ── status color mode + selection highlight (the /cells page) ──────────────


def _roi(label_id, status, *, cy=4.0, cx=4.0):
    """A tiny square ROI carrying a cross-session match status."""
    ys = [cy - 1, cy - 1, cy + 1, cy + 1]
    xs = [cx - 1, cx + 1, cx + 1, cx - 1]
    return ROIRender(
        label_id=label_id, source_stage=1, gate_outcome="accept",
        activity_type=None, area=9, centroid_yx=(cy, cx),
        contours=[(ys, xs)], match_status=status,
    )


def _is_badge(trace):
    return trace.type == "scatter" and trace.mode == "text"


def _outlines(fig):
    """ROI outline traces, keyed by label_id (badges carry no meta)."""
    return {t.meta: t for t in fig.data
            if t.type == "scatter" and isinstance(t.meta, int)}


def test_status_mode_gives_each_outcome_its_own_color():
    rois = [_roi(1, "matched"), _roi(2, "new"), _roi(3, "lost")]
    fig = build_roi_figure(mean=np.zeros((8, 8), np.float32), rois=rois,
                           color_mode="status")
    colors = {t.line.color for t in _outlines(fig).values()}
    assert len(colors) == 3, f"expected 3 distinct status colors, got {colors}"


def test_a_cell_not_detected_here_is_drawn_dotted():
    """A 'lost' outline sits at a position where nothing was detected."""
    rois = [_roi(1, "matched"), _roi(2, "lost")]
    fig = build_roi_figure(mean=np.zeros((8, 8), np.float32), rois=rois,
                           color_mode="status")
    outlines = _outlines(fig)
    assert outlines[2].line.dash == "dot"
    assert outlines[1].line.dash == "solid"


def test_status_dashing_does_not_leak_into_other_color_modes():
    fig = build_roi_figure(mean=np.zeros((8, 8), np.float32),
                           rois=[_roi(1, "lost")], color_mode="stage")
    assert _outlines(fig)[1].line.dash == "solid"


def test_highlighting_thickens_only_the_named_labels():
    rois = [_roi(1, "matched"), _roi(2, "matched", cx=6.0)]
    fig = build_roi_figure(mean=np.zeros((8, 8), np.float32), rois=rois,
                           color_mode="status", highlight_labels={2: "#7"})
    outlines = _outlines(fig)
    assert outlines[2].line.width > outlines[1].line.width


def test_highlighting_prints_the_badge_at_the_centroid():
    rois = [_roi(1, "matched", cy=3.0, cx=5.0)]
    fig = build_roi_figure(mean=np.zeros((8, 8), np.float32), rois=rois,
                           color_mode="status", highlight_labels={1: "#7"})
    badge = next(t for t in fig.data if _is_badge(t))
    assert badge.text == ("#7",)
    assert (badge.y[0], badge.x[0]) == (3.0, 5.0)


def test_no_badge_without_a_highlight():
    fig = build_roi_figure(mean=np.zeros((8, 8), np.float32),
                           rois=[_roi(1, "matched")], color_mode="status")
    assert not [t for t in fig.data if _is_badge(t)]


def test_default_call_is_unchanged_by_the_new_parameters():
    """Every existing caller passes neither param — width stays 3.2, solid."""
    fig = build_roi_figure(mean=np.zeros((8, 8), np.float32),
                           rois=[_roi(1, None)], color_mode="stage")
    line = _outlines(fig)[1].line
    assert (line.width, line.dash) == (3.2, "solid")


def test_trace_index_map_points_at_the_outlines_it_describes():
    rois = [_roi(1, "matched"), _roi(2, "new", cx=6.0)]
    fig = build_roi_figure(mean=np.zeros((8, 8), np.float32), rois=rois,
                           color_mode="status")
    index_map = trace_index_map(fig)
    assert set(index_map) == {1, 2}
    for label_id, indices in index_map.items():
        for i in indices:
            assert fig.data[i].meta == label_id


def test_trace_index_map_skips_the_heatmap_and_badges():
    fig = build_roi_figure(mean=np.zeros((8, 8), np.float32),
                           rois=[_roi(1, "matched")], color_mode="status",
                           highlight_labels={1: "#1"})
    flat = [i for ids in trace_index_map(fig).values() for i in ids]
    assert flat
    assert all(fig.data[i].type == "scatter" and fig.data[i].mode == "lines"
               for i in flat)
