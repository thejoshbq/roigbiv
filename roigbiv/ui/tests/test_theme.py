"""Regression guard: plotly_template() always returns roigbiv-reacher and it is registered."""
import pytest

from roigbiv.ui.services.theme import (
    axis_muted_color,
    figure_paper_bg,
    heatmap_reverse,
    plotly_template,
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


def test_template_rajdhani_font():
    import plotly.io as pio
    tmpl = pio.templates["roigbiv-reacher"]
    assert "Rajdhani" in tmpl.layout.font.family


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_figure_paper_bg_always_black(theme):
    assert figure_paper_bg(theme) == "#000000"


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_heatmap_reverse_always_true(theme):
    assert heatmap_reverse(theme) is True


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_axis_muted_color_constant(theme):
    assert axis_muted_color(theme) == "#4A7070"


@pytest.mark.parametrize("theme", ["dark", "light", None])
def test_warning_color_constant(theme):
    assert warning_color(theme) == "#FFB454"
