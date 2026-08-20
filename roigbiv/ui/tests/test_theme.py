"""Regression guard: plotly_template() always returns roigbiv-reacher and it is registered.

Colour assertions name ``phoxel_tokens`` rather than literals. Pinning a hex
here would just be a second copy of the palette, free to drift from the one the
app actually loads.
"""
import phoxel_tokens as pt
import pytest

from roigbiv.ui.services.theme import (
    axis_muted_color,
    danger_color,
    figure_paper_bg,
    heatmap_reverse,
    hover_bg,
    hover_border,
    plotly_template,
    text_color,
    warning_color,
)


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_plotly_template_is_constant(theme):
    assert plotly_template(theme) == "roigbiv-reacher", (
        f"plotly_template({theme!r}) must always return 'roigbiv-reacher'"
    )


def test_template_registered():
    import plotly.io as pio
    assert "roigbiv-reacher" in pio.templates, (
        "roigbiv-reacher template must be registered in pio.templates after import"
    )


def test_template_pure_black_canvas():
    import plotly.io as pio
    tmpl = pio.templates["roigbiv-reacher"]
    assert tmpl.layout.paper_bgcolor == "#000000"
    assert tmpl.layout.plot_bgcolor == "#000000"


def test_template_uses_the_system_ui_face():
    import plotly.io as pio
    tmpl = pio.templates["roigbiv-reacher"]
    assert "JetBrains Mono" in tmpl.layout.font.family


def test_template_inherits_the_shared_axis_treatment():
    """Everything except the canvas comes from phoxel-tokens, unmodified."""
    import plotly.io as pio
    tmpl = pio.templates["roigbiv-reacher"]
    shared = pt.PLOTLY_TEMPLATE["layout"]
    assert tmpl.layout.xaxis.gridcolor == shared["xaxis"]["gridcolor"]
    assert tmpl.layout.xaxis.linecolor == shared["xaxis"]["linecolor"]
    assert tmpl.layout.font.color == shared["font"]["color"]
    assert tuple(tmpl.layout.colorway) == tuple(shared["colorway"])


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_figure_paper_bg_always_black(theme):
    assert figure_paper_bg(theme) == "#000000"


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_heatmap_reverse_always_true(theme):
    assert heatmap_reverse(theme) is True


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_axis_muted_color_constant(theme):
    assert axis_muted_color(theme) == pt.color("text-muted")


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_warning_color_constant(theme):
    assert warning_color(theme) == pt.color("warn")


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_figure_colours_come_from_the_shared_palette(theme):
    """No accessor may invent a colour the design system does not define."""
    palette = set(pt.COLOR.values())
    for fn in (axis_muted_color, warning_color, figure_paper_bg,
               hover_bg, hover_border, text_color, danger_color):
        assert fn(theme) in palette, f"{fn.__name__}({theme!r}) is off-palette"
