"""Theme constants shared by the app shell and figure builders.

Colour comes from ``phoxel_tokens`` — the shared Phoxel Workbench design
system — not from values held here.  The Plotly template is the package's
layout template with one deliberate departure, described below.

The shell is dark-only. Figure builders still accept a ``theme`` argument
for call-site compatibility; :func:`plotly_template` ignores it and always
returns the registered ``roigbiv-reacher`` Plotly template.

FOV invariant: the mean-projection heatmap canvas is ALWAYS pure black
(``#000000``) regardless of UI theme.  ``heatmap_colorscale`` / ``heatmap_reverse``
/ ``figure_paper_bg`` are unconditional; they accept ``theme`` for API
compatibility but ignore it.
"""
from __future__ import annotations

import copy
from typing import Optional

import phoxel_tokens as pt

LIGHT = "light"
DARK = "dark"

_TEMPLATE = "roigbiv-reacher"


def _register_roigbiv_template() -> None:
    """Register the custom ROIGBIV Plotly template, idempotent."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        return
    if _TEMPLATE in pio.templates:
        return
    layout = copy.deepcopy(pt.PLOTLY_TEMPLATE["layout"])
    # The one departure from the shared template, and the reason ROIGBIV keeps
    # a template of its own: every canvas in this app is a fluorescence image
    # or a trace read against one, so the ground is pure black rather than the
    # system's --surface-1 panel.
    layout["paper_bgcolor"] = pt.color("surface-black")
    layout["plot_bgcolor"] = pt.color("surface-black")
    # The system's margins assume a page-width figure; ROIGBIV packs several
    # into a contact sheet and sets its own per figure.
    layout.pop("margin", None)
    pio.templates[_TEMPLATE] = go.layout.Template(layout=go.Layout(**layout))


_register_roigbiv_template()


def normalize(theme: Optional[str]) -> str:
    """Map an arbitrary theme value to ``"light"`` or ``"dark"``."""
    return LIGHT if theme == LIGHT else DARK


def plotly_template(theme: Optional[str] = None) -> str:
    """ROIGBIV Plotly template name — theme arg kept for API compatibility."""
    return _TEMPLATE


def is_dark(theme: Optional[str]) -> bool:
    return normalize(theme) == DARK


def axis_muted_color(theme: Optional[str] = None) -> str:
    """Muted color for axis-margin annotations."""
    return pt.color("text-muted")


def warning_color(theme: Optional[str] = None) -> str:
    """Warm amber — readable on the black figure canvas."""
    return pt.color("warn")


def figure_paper_bg(theme: Optional[str] = None) -> str:
    """Always pure black — FOV canvas must never show a light background."""
    return pt.color("surface-black")


def hover_bg(theme: Optional[str] = None) -> str:
    """Background for figure hover labels."""
    return pt.color("surface-2")


def hover_border(theme: Optional[str] = None) -> str:
    """Border for figure hover labels."""
    return pt.color("border-control")


def text_color(theme: Optional[str] = None) -> str:
    """Body text on the figure canvas."""
    return pt.color("text")


def danger_color(theme: Optional[str] = None) -> str:
    """Error text on the figure canvas."""
    return pt.color("err")


def heatmap_colorscale(theme: Optional[str] = None) -> str:
    """Greys colorscale: 0 → black (inactive), 1 → white (active)."""
    return "Greys"


def heatmap_reverse(theme: Optional[str] = None) -> bool:
    """Always True — Greys runs white→black, so reverse to get black→white (0=dark, 1=bright)."""
    return True
