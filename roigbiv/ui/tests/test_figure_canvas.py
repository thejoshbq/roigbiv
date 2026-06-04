"""Regression guard: FOV canvas is always pure black, colorscale is always Greys."""
import numpy as np
import pytest

from roigbiv.ui.components.figure import build_roi_figure


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
