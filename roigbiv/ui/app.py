"""Dash app factory.

Pages in a top nav:

* **Process** — scan a workspace, set params, run the pipeline.
* **Review** — unified viewing + HITL corrections. Multi-session grid
  canvas, slide-in metadata drawer, overlay toggle, Add ROI draw mode,
  polygon / freehand / eraser, merge / split / edit / relabel, additive
  corrections log. (The former /viewer path redirects here.)

Registry browsing/maintenance lives in the ``roigbiv-registry`` CLI
(``list``, ``show``, ``migrate``, ``backfill``, ...). The UI never needed
to expose those — they're admin-grade operations.

State is held server-side in a single shared :class:`AppState` instance and
mirrored to the client via ``dcc.Store`` only for the pieces the UI needs
to react to (selected FOV, selected session, view mode, etc). Heavy arrays
— mean projections, masks — stay server-side and are streamed into figures
on demand.
"""
from __future__ import annotations

import os
import secrets as _secrets
import threading
import time
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from dash_bootstrap_templates import load_figure_template

from roigbiv.ui.components import errors as error_components
from roigbiv.ui.logging import configure_ui_logging
from roigbiv.ui.pages import process, review
from roigbiv.ui.pages.review import (
    MAIN_COL_ID,
    RIGHT_SIDEBAR_COL_ID,
    RIGHT_SIDEBAR_STORE_ID,
    RIGHT_SIDEBAR_TOGGLE_ID,
    SIDEBAR_COL_ID,
    SIDEBAR_STORE_ID,
    SIDEBAR_TOGGLE_ID,
)


# Both stylesheets are served at boot; the runtime toggle just flips
# ``data-bs-theme`` on <html>, so we never round-trip the server to swap CSS.
LIGHT_THEME = dbc.themes.FLATLY
DARK_THEME = dbc.themes.DARKLY
LIGHT_TEMPLATE = "flatly"
DARK_TEMPLATE = "darkly"

THEME_STORE_ID = "roigbiv-theme"
THEME_TOGGLE_ID = "roigbiv-theme-toggle"
THEME_TOGGLE_ICON_ID = "roigbiv-theme-toggle-icon"

PAGES = (
    ("/process",  "Process",  process),
    ("/review",   "Review",   review),
)


_SESSION_TTL = 7200   # seconds before an idle session's state is evicted


def _start_session_cleanup() -> None:
    """Daemon thread that evicts stale per-session state every 30 minutes."""
    from roigbiv.ui.services.app_state import _instances, _instances_lock
    from roigbiv.ui.services.cellpose_trainer import _trainers, _trainers_lock
    from roigbiv.ui.services.pipeline_runner import _runners, _runners_lock

    def _loop() -> None:
        while True:
            time.sleep(1800)
            cutoff = time.monotonic() - _SESSION_TTL
            for store, lock in (
                (_instances, _instances_lock),
                (_runners, _runners_lock),
                (_trainers, _trainers_lock),
            ):
                with lock:
                    stale = [
                        sid for sid, obj in store.items()
                        if getattr(obj, "_last_accessed", 0) < cutoff
                    ]
                    for sid in stale:
                        del store[sid]

    threading.Thread(target=_loop, name="roigbiv-session-cleanup",
                     daemon=True).start()


def build_app() -> dash.Dash:
    """Create and wire the Dash app (layout + callbacks)."""
    configure_ui_logging()
    # Register both Plotly templates so figure builders can reference either
    # by name. ``load_figure_template`` is idempotent across calls.
    load_figure_template([LIGHT_TEMPLATE, DARK_TEMPLATE])
    app = dash.Dash(
        __name__,
        title="ROIGBIV",
        update_title=None,
        external_stylesheets=[LIGHT_THEME, DARK_THEME, dbc.icons.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder="assets",
    )
    app.server.secret_key = os.environ.get(
        "ROIGBIV_SECRET_KEY", _secrets.token_hex(32)
    )
    if not os.environ.get("ROIGBIV_SECRET_KEY"):
        import warnings
        warnings.warn(
            "ROIGBIV_SECRET_KEY is not set — a random key is used. "
            "Browser sessions will be lost on every server restart. "
            "Set ROIGBIV_SECRET_KEY=<hex-string> for persistence.",
            stacklevel=2,
        )
    _start_session_cleanup()

    app.layout = _build_layout()
    _wire_routes(app)
    _wire_sidebar_toggles(app)
    _wire_theme_toggle(app)
    error_components.register_callbacks(app)
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
    theme_toggle = dbc.Button(
        html.I(id=THEME_TOGGLE_ICON_ID, className="bi bi-sun-fill"),
        id=THEME_TOGGLE_ID,
        color="link",
        className="roigbiv-theme-toggle ms-2 p-1",
        title="Toggle light / dark theme",
        n_clicks=0,
    )
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dcc.Link(brand, href="/process",
                         style={"textDecoration": "none", "color": "inherit"}),
                registry_indicator,
                dbc.Nav(nav_items, navbar=True, className="ms-auto"),
                theme_toggle,
            ],
            fluid=True,
        ),
        sticky="top",
        className="roigbiv-navbar mb-3",
    )
    return html.Div([
        dcc.Location(id="roigbiv-url", refresh=False),
        # Default = "dark" (the lab uses ROIGBIV in a darkened scope room).
        # Persisted in localStorage so the choice survives reloads.
        dcc.Store(id=THEME_STORE_ID, storage_type="local", data="dark"),
        navbar,
        dbc.Container(id="roigbiv-page-content", fluid=True, className="pb-5"),
    ])


def _active_registry_label() -> str:
    """One-line initial registry indicator shown before a workspace is scanned."""
    return "registry: scan a workspace to begin"


def _wire_routes(app: dash.Dash) -> None:
    @app.callback(
        Output("roigbiv-page-content", "children"),
        Input("roigbiv-url", "pathname"),
    )
    def _render(pathname: str):  # noqa: ANN001
        if not pathname or pathname == "/":
            return process.layout()
        # Backward-compat: the old standalone Viewer has been folded into
        # Review. Bookmarks to /viewer still land somewhere sensible.
        if pathname.rstrip("/") == "/viewer":
            return review.layout()
        # Registry browsing was retired — route stragglers to Process.
        if pathname.rstrip("/") == "/registry":
            return process.layout()
        for path, _, page in PAGES:
            if pathname.rstrip("/") == path.rstrip("/"):
                return page.layout()
        return dbc.Alert(
            f"Unknown page: {pathname}. Navigate via the top bar.",
            color="warning",
        )


def _wire_theme_toggle(app: dash.Dash) -> None:
    """Theme toggle — flips ``data-bs-theme`` on <html> and persists choice.

    Two clientside callbacks:

    * Button click → toggle the stored theme.
    * Store value  → apply ``data-bs-theme`` to the document root and update
      the toggle icon. Runs on initial load too, so the persisted choice is
      respected without a click.
    """
    app.clientside_callback(
        """
        function(n_clicks, current) {
            if (!n_clicks) {
                return current || "dark";
            }
            return (current === "dark") ? "light" : "dark";
        }
        """,
        Output(THEME_STORE_ID, "data"),
        Input(THEME_TOGGLE_ID, "n_clicks"),
        State(THEME_STORE_ID, "data"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        """
        function(theme) {
            const t = (theme === "light") ? "light" : "dark";
            document.documentElement.setAttribute("data-bs-theme", t);
            // Sun icon when in dark mode (click to go light); moon when in light.
            return (t === "dark") ? "bi bi-sun-fill" : "bi bi-moon-fill";
        }
        """,
        Output(THEME_TOGGLE_ICON_ID, "className"),
        Input(THEME_STORE_ID, "data"),
    )


def _wire_sidebar_toggles(app: dash.Dash) -> None:
    """Clientside toggles for the Review page's two collapsible sidebars.

    Three small callbacks, one responsibility each:

    * left-toggle button  → left col className + left store
    * right-toggle button → right col className + right store
    * both stores         → main col className (depends on BOTH states)

    State is mirrored to ``dcc.Store`` in local storage so the choices
    survive page navigation. Main-col width expands to reclaim space
    whenever either (or both) sidebars collapse:

    | left | right | main                 |
    |------|-------|----------------------|
    | open | open  | ``col-md-6``         |
    | clos | open  | ``col-md-9``         |
    | open | clos  | ``col-md-9``         |
    | clos | clos  | ``col-md-12``        |
    """
    app.clientside_callback(
        """
        function(n_clicks, stored) {
            let is_open = !(stored && stored.is_open === false);
            if (n_clicks) {
                is_open = !is_open;
            }
            const sidebar_class = is_open
                ? "col-md-3 pe-md-3"
                : "d-none";
            return [sidebar_class, {is_open: is_open}];
        }
        """,
        Output(SIDEBAR_COL_ID, "className"),
        Output(SIDEBAR_STORE_ID, "data"),
        Input(SIDEBAR_TOGGLE_ID, "n_clicks"),
        State(SIDEBAR_STORE_ID, "data"),
    )
    app.clientside_callback(
        """
        function(n_clicks, stored) {
            let is_open = !(stored && stored.is_open === false);
            if (n_clicks) {
                is_open = !is_open;
            }
            const sidebar_class = is_open
                ? "col-md-3 ps-md-3"
                : "d-none";
            return [sidebar_class, {is_open: is_open}];
        }
        """,
        Output(RIGHT_SIDEBAR_COL_ID, "className"),
        Output(RIGHT_SIDEBAR_STORE_ID, "data"),
        Input(RIGHT_SIDEBAR_TOGGLE_ID, "n_clicks"),
        State(RIGHT_SIDEBAR_STORE_ID, "data"),
    )
    app.clientside_callback(
        """
        function(left_stored, right_stored) {
            const left_open = !(left_stored && left_stored.is_open === false);
            const right_open = !(right_stored && right_stored.is_open === false);
            // After the flex reflow settles, force every Plotly graph to
            // re-measure its container. `d-none` → open transitions hide
            // zero-width caching that otherwise sticks until a toggle cycle.
            setTimeout(function() {
                if (window.Plotly) {
                    document.querySelectorAll('.js-plotly-plot').forEach(function(el) {
                        try { window.Plotly.Plots.resize(el); } catch (e) {}
                    });
                }
            }, 80);
            if (left_open && right_open)   return "col-md-6";
            if (!left_open && !right_open) return "col-md-12";
            return "col-md-9";
        }
        """,
        Output(MAIN_COL_ID, "className"),
        Input(SIDEBAR_STORE_ID, "data"),
        Input(RIGHT_SIDEBAR_STORE_ID, "data"),
    )
