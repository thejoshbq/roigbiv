"""Theme constants shared by the app shell and figure builders.

The runtime toggle in :mod:`roigbiv.ui.app` flips ``data-bs-theme`` on
the document root and writes the chosen theme name to a ``dcc.Store``.
Pages thread that store value into figure callbacks; figure builders call
:func:`plotly_template` to convert it to the Plotly template name that
``dash-bootstrap-templates.load_figure_template`` registers.
"""
from __future__ import annotations

from typing import Optional

LIGHT = "light"
DARK = "dark"

LIGHT_TEMPLATE = "flatly"
DARK_TEMPLATE = "darkly"


def _register_templates() -> None:
    """Idempotently register the bootstrap-matched Plotly templates.

    Called at import time so figure builders are usable from tests and
    scripts without booting a Dash app first.
    """
    try:
        import plotly.io as pio
        from dash_bootstrap_templates import load_figure_template
    except ImportError:
        return
    if LIGHT_TEMPLATE in pio.templates and DARK_TEMPLATE in pio.templates:
        return
    load_figure_template([LIGHT_TEMPLATE, DARK_TEMPLATE])


_register_templates()


def normalize(theme: Optional[str]) -> str:
    """Map an arbitrary theme value to ``"light"`` or ``"dark"``."""
    return LIGHT if theme == LIGHT else DARK


def plotly_template(theme: Optional[str]) -> str:
    """Plotly template name for the given theme — safe for ``None``."""
    return DARK_TEMPLATE if normalize(theme) == DARK else LIGHT_TEMPLATE


def is_dark(theme: Optional[str]) -> bool:
    return normalize(theme) == DARK


def axis_muted_color(theme: Optional[str]) -> str:
    """Color for axis-margin annotations (muted, theme-aware)."""
    return "#aaa" if is_dark(theme) else "#888"


def warning_color(theme: Optional[str]) -> str:
    """Color for in-figure warning annotations (e.g. ``mixed fs``)."""
    return "#e6b56c" if is_dark(theme) else "#c77"


def figure_paper_bg(theme: Optional[str]) -> str:
    """Background color for figures that are layered on a card body.

    Returned as a plain hex/rgb string rather than ``"transparent"`` so PNG
    export still has a predictable background.
    """
    return "#222" if is_dark(theme) else "#ffffff"


def heatmap_colorscale(theme: Optional[str]) -> str:
    """Colorscale for the mean-projection heatmap (per theme).

    Light backgrounds want reverse-Greys (dark = high signal); dark
    backgrounds want straight Greys (light = high signal).
    """
    return "Greys" if is_dark(theme) else "Greys"


def heatmap_reverse(theme: Optional[str]) -> bool:
    """Whether to reverse the heatmap colorscale for the given theme."""
    return not is_dark(theme)
