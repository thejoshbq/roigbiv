"""Dash app factory.

Pages in a top nav:

* **Process** — scan a workspace, set params, run the pipeline.
* **Registry** — browse FOVs and sessions; migrate / backfill escape hatches.
* **Viewer** — per-FOV and cross-session viewer with stage / feature /
  cross-session color modes (ROIs always rendered as outlines), plus
  FOV-level and ROI-level signal trace panels driven by session selection
  and ROI click.
* **Review** — HITL correction tools: polygon / freehand / eraser drawing,
  merge / split / edit / relabel, additive corrections log.

State is held server-side in a single shared :class:`AppState` instance and
mirrored to the client via ``dcc.Store`` only for the pieces the UI needs
to react to (selected FOV, selected session, view mode, etc). Heavy arrays
— mean projections, masks — stay server-side and are streamed into figures
on demand.
"""
from __future__ import annotations

import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from roigbiv.ui.pages import process, registry, review, viewer
from roigbiv.ui.services.app_state import get_app_state


THEME = dbc.themes.FLATLY
PAGES = (
    ("/process",  "Process",  process),
    ("/registry", "Registry", registry),
    ("/viewer",   "Viewer",   viewer),
    ("/review",   "Review",   review),
)


def build_app() -> dash.Dash:
    """Create and wire the Dash app (layout + callbacks)."""
    app = dash.Dash(
        __name__,
        title="ROIGBIV",
        update_title=None,
        external_stylesheets=[THEME, dbc.icons.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder="assets",
    )

    # Prime the shared app-state singleton so pages can read/write it.
    get_app_state()

    app.layout = _build_layout()
    _wire_routes(app)
    for _, _, page in PAGES:
        page.register_callbacks(app)
    return app


def _build_layout() -> html.Div:
    nav_items = [
        dbc.NavItem(dbc.NavLink(label, href=path, active="exact"))
        for path, label, _ in PAGES
    ]
    brand = html.Span([
        html.Span("ROI", className="roigbiv-brand roigbiv-brand-accent"),
        html.Span("GBIV", className="roigbiv-brand"),
    ], className="d-flex align-items-center")
    registry_indicator = html.Small(
        _active_registry_label(),
        id="roigbiv-active-registry",
        className="text-muted ms-3",
        title="Active registry DSN (change with `roigbiv-ui --workspace PATH`)",
    )
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dcc.Link(brand, href="/process",
                         style={"textDecoration": "none", "color": "inherit"}),
                registry_indicator,
                dbc.Nav(nav_items, navbar=True, className="ms-auto"),
            ],
            fluid=True,
        ),
        color="light",
        dark=False,
        sticky="top",
        className="roigbiv-navbar mb-3",
    )
    return html.Div([
        dcc.Location(id="roigbiv-url", refresh=False),
        navbar,
        dbc.Container(id="roigbiv-page-content", fluid=True, className="pb-5"),
    ])


def _active_registry_label() -> str:
    """One-line human-readable description of the active registry DSN."""
    dsn = os.environ.get("ROIGBIV_REGISTRY_DSN")
    if dsn and dsn.startswith("sqlite:///"):
        return f"registry: {dsn[len('sqlite:///'):]}"
    if dsn:
        return f"registry: {dsn}"
    default = Path.cwd() / "inference" / "registry.db"
    return f"registry: {default} (default — pass --workspace PATH to override)"


def _wire_routes(app: dash.Dash) -> None:
    @app.callback(
        Output("roigbiv-page-content", "children"),
        Input("roigbiv-url", "pathname"),
    )
    def _render(pathname: str):  # noqa: ANN001
        if not pathname or pathname == "/":
            return process.layout()
        for path, _, page in PAGES:
            if pathname.rstrip("/") == path.rstrip("/"):
                return page.layout()
        return dbc.Alert(
            f"Unknown page: {pathname}. Navigate via the top bar.",
            color="warning",
        )
