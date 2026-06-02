"""Theme constants shared by the app shell and figure builders.

The runtime toggle in :mod:`roigbiv.ui.app` flips ``data-bs-theme`` on
the document root and writes the chosen theme name to a ``dcc.Store``.
Pages thread that store value into figure callbacks; figure builders call
:func:`plotly_template` to convert it to the registered Plotly template name.

FOV invariant: the mean-projection heatmap canvas is ALWAYS pure black
(``#000000``) regardless of UI theme.  ``heatmap_colorscale`` / ``heatmap_reverse``
/ ``figure_paper_bg`` are unconditional; they accept ``theme`` for API
compatibility but ignore it.
"""
from __future__ import annotations

from typing import Optional

LIGHT = "light"
DARK = "dark"

_TEMPLATE = "roigbiv-reacher"


def _register_roigbiv_template() -> None:
    """Register the custom REACHER Plotly template, idempotent."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return
    if _TEMPLATE in pio.templates:
        return
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(family="Rajdhani, system-ui, sans-serif", color="#C8E8E8", size=12),
        colorway=["#00E5FF", "#3498db", "#e67e22", "#9b59b6", "#f1c40f", "#2ecc71", "#e74c3c"],
        xaxis=dict(
            gridcolor="#0D2626",
            zerolinecolor="#0D2626",
            linecolor="#4A7070",
            tickcolor="#4A7070",
            tickfont=dict(color="#4A7070"),
        ),
        yaxis=dict(
            gridcolor="#0D2626",
            zerolinecolor="#0D2626",
            linecolor="#4A7070",
            tickcolor="#4A7070",
            tickfont=dict(color="#4A7070"),
        ),
        hoverlabel=dict(
            bgcolor="#0A1818",
            bordercolor="#00E5FF",
            font=dict(color="#C8E8E8", family="JetBrains Mono, monospace"),
        ),
        legend=dict(
            bgcolor="rgba(10,24,24,0.85)",
            bordercolor="#0D2626",
            font=dict(color="#C8E8E8"),
        ),
        title=dict(font=dict(color="#00E5FF")),
    )
    pio.templates[_TEMPLATE] = tmpl


_register_roigbiv_template()


def normalize(theme: Optional[str]) -> str:
    """Map an arbitrary theme value to ``"light"`` or ``"dark"``."""
    return LIGHT if theme == LIGHT else DARK


def plotly_template(theme: Optional[str] = None) -> str:
    """REACHER Plotly template name — theme arg kept for API compatibility."""
    return _TEMPLATE


def is_dark(theme: Optional[str]) -> bool:
    return normalize(theme) == DARK


def axis_muted_color(theme: Optional[str] = None) -> str:
    """Muted color for axis-margin annotations."""
    return "#4A7070"


def warning_color(theme: Optional[str] = None) -> str:
    """Warm amber — readable on pure-black background."""
    return "#FFB454"


def figure_paper_bg(theme: Optional[str] = None) -> str:
    """Always pure black — FOV canvas must never show a light background."""
    return "#000000"


def heatmap_colorscale(theme: Optional[str] = None) -> str:
    """Greys colorscale: 0 → black (inactive), 1 → white (active)."""
    return "Greys"


def heatmap_reverse(theme: Optional[str] = None) -> bool:
    """Always True — Greys runs white→black, so reverse to get black→white (0=dark, 1=bright)."""
    return True
