"""Dash app factory.

One page per operation, in the order the operations happen:

1. **Motion correction** — parameters, a live view of the registration, and the
   per-FOV quality metrics (``pages/motion.py``).
2. **Discovery** — per-FOV Cellpose calibration, the detection run, and
   tuning the seeded boundaries detection produces
   (``pages/discovery.py``). Was two pages (Centroids, Boundaries); merged
   because they are one workflow on one FOV, not two. Its viewer can also
   play the registered movie under the markers (**Live movie**), so the Fiji
   habit of scrubbing for a transient before drawing around it has an
   equivalent here — see ``services/movie_source.py``.
3. **Tracking** — session order, cross-session registration, and the contact
   sheet where cells are reviewed and corrected (``pages/tracking.py``).

Each page owns its own run and nothing else. They used to be one 1400-line
Pipeline page with a run-mode radio deciding which of four jobs its single Run
button meant, which made "did motion correction finish" and "did detection
work" the same question with the same answer.

Two things are *not* per page, because every page needs them and none owns one:

* the workspace scanner, a disclosure in the navbar
  (``components/workspace_bar.py``);
* the pipeline run status, since there is one runner behind one GPU gate
  (``components/run_panel.py``).

Both register their callbacks once, here — a page-level registration would give
Dash duplicate Outputs on the same ids. Pages coordinate through
``workspace_bar.WORKSPACE_VERSION`` rather than by writing into each other's
controls.

The Review page (unified viewing + HITL corrections; ``pages/review.py``) is
unrouted and dormant, not deleted. Registry browsing/maintenance lives in the
``roigbiv-registry`` CLI — admin-grade operations the UI never needed.

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
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from roigbiv import __version__ as _ROIGBIV_VERSION
from roigbiv.ui.components import errors as error_components
from roigbiv.ui.components import run_panel, workspace_bar
from roigbiv.ui.logging import configure_ui_logging
from roigbiv.ui.pages import discovery, motion, tracking
from roigbiv.ui.pages.review import (
    MAIN_COL_ID,
    RIGHT_SIDEBAR_COL_ID,
    RIGHT_SIDEBAR_STORE_ID,
    RIGHT_SIDEBAR_TOGGLE_ID,
    SIDEBAR_COL_ID,
    SIDEBAR_STORE_ID,
    SIDEBAR_TOGGLE_ID,
)


#: The three families the Phoxel template names: mono for UI and data, a sans
#: for running prose, a display face for the wordmark.
_GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400"
    "&family=IBM+Plex+Sans:wght@400;500;600"
    "&family=Chakra+Petch:wght@600;700"
    "&display=swap"
)

#: Served from the installed ``phoxel_tokens`` package rather than copied into
#: ``assets/`` — ``pt.install_assets()`` would write into the package directory
#: on every boot, which a non-editable install cannot do. Same Flask-static
#: pattern as the ROI editor's ``/roi-assets/`` route.
_PHOXEL_ROUTE = "/phoxel-assets"
_PHOXEL_STYLESHEETS = (
    f"{_PHOXEL_ROUTE}/phoxel-tokens.css",
    f"{_PHOXEL_ROUTE}/phoxel-bootstrap-bridge.css",
    f"{_PHOXEL_ROUTE}/phoxel-tokens_chrome.css",
)

PAGES = (
    ("/motion-correction", "Motion correction", motion),
    ("/discovery", "Discovery", discovery),
    ("/tracking", "Tracking", tracking),
)

#: Paths that used to exist, and where their content went. ``/cells`` and
#: ``/track`` are two halves of one page now; ``/centroids`` and
#: ``/boundaries`` are two more; ``/registry`` was retired to the CLI.
#: Bookmarks and half-remembered URLs are cheap to honour.
_REDIRECTS = {
    "/pipeline": "/motion-correction",
    "/process": "/motion-correction",
    "/registry": "/motion-correction",
    "/track": "/tracking",
    "/cells": "/tracking",
    "/viewer": "/tracking",
    "/centroids": "/discovery",
    "/boundaries": "/discovery",
}


_SESSION_TTL = 7200   # seconds before an idle session's state is evicted


def _start_session_cleanup() -> None:
    """Daemon thread that evicts stale per-session state every 30 minutes."""
    from roigbiv.ui.services.app_state import _instances, _instances_lock
    from roigbiv.ui.services.pipeline_runner import _runners, _runners_lock

    def _loop() -> None:
        while True:
            time.sleep(1800)
            cutoff = time.monotonic() - _SESSION_TTL
            for store, lock in (
                (_instances, _instances_lock),
                (_runners, _runners_lock),
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


def build_app(preset_workspace: "Optional[WorkspacePaths]" = None) -> dash.Dash:
    """Create and wire the Dash app (layout + callbacks)."""
    configure_ui_logging()
    import roigbiv.ui.services.theme  # noqa: F401 — registers roigbiv-reacher template at import
    app = dash.Dash(
        __name__,
        title="ROIGBIV",
        update_title=None,
        external_stylesheets=[
            dbc.themes.CYBORG,
            dbc.icons.BOOTSTRAP,
            _GOOGLE_FONTS,
            # After CYBORG so the tokens win; before assets/roigbiv.css, which
            # Dash appends and which aliases these onto its --roigbiv-* names.
            *_PHOXEL_STYLESHEETS,
        ],
        suppress_callback_exceptions=True,
        assets_folder="assets",
    )
    app.index_string = (
        "<!DOCTYPE html>\n<html>\n    <head>\n        {%metas%}\n"
        "        <title>{%title%}</title>\n        {%favicon%}\n"
        '        <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">\n'
        "        {%css%}\n    </head>\n    <body>\n        {%app_entry%}\n"
        "        <footer>\n            {%config%}\n            {%scripts%}\n"
        "            {%renderer%}\n        </footer>\n    </body>\n</html>"
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
    app.server.config["ROIGBIV_PRESET_WORKSPACE"] = preset_workspace

    # A *callable* layout: Dash evaluates it per page load, inside a request
    # context. The navbar now reports the session's workspace, which a layout
    # built once at import time could not know about.
    app.layout = _build_layout
    _wire_routes(app)
    _register_phoxel_assets(app.server)
    from roigbiv.ui.routes.roi_editor import register_flask_routes
    register_flask_routes(app.server)
    from roigbiv.ui.routes.mc_preview import (
        register_flask_routes as register_mc_preview_routes)
    register_mc_preview_routes(app.server)
    from roigbiv.ui.routes.cells_api import (
        register_flask_routes as register_cells_api_routes)
    register_cells_api_routes(app.server)
    from roigbiv.ui.routes.discovery_api import (
        register_flask_routes as register_discovery_api_routes)
    register_discovery_api_routes(app.server)
    _wire_sidebar_toggles(app)
    error_components.register_callbacks(app)
    # Registered once, not per page: both live outside the routed container and
    # a second registration would be duplicate Outputs on the same ids.
    workspace_bar.register_callbacks(app)
    run_panel.register_callbacks(app)
    for _, _, page in PAGES:
        page.register_callbacks(app)
    return app


def _register_phoxel_assets(server) -> None:  # noqa: ANN001
    """Serve the shared token stylesheets straight out of ``phoxel_tokens``."""
    import phoxel_tokens as pt
    from flask import send_from_directory

    assets = pt.assets_dir()

    @server.route(f"{_PHOXEL_ROUTE}/<path:filename>")
    def phoxel_assets(filename: str):  # noqa: ANN202
        return send_from_directory(assets, filename)


def _fov_mark() -> html.Span:
    """Header mark: FOV frame, two ROI contours, one centroid.

    Dash has no ``html.Svg``, so the glyph lives in ``assets/favicon.svg``
    and is painted here with ``--accent`` through a CSS mask.
    """
    return html.Span(
        className="phoxel-mark",
        role="img",
        **{"aria-label": "ROIGBIV"},
    )


def _build_layout() -> html.Div:
    nav_items = [
        dbc.NavItem(dbc.NavLink(label, href=path, active="exact"))
        for path, label, _ in PAGES
    ]
    brand = html.Div([
        _fov_mark(),
        html.H1("// ROIGBIV",
                className="phoxel-wordmark title-glow glitch-hover"),
    ], className="roigbiv-brand")
    sys_online = html.Span([
        html.Span(className="pulse-dot"),
        html.Span("SYS_ONLINE", className="sys-label"),
    ], className="sys-online")
    # `crt` is the Phoxel scanline layer, opt-in and chrome-only. The title
    # bar is the one surface in this app that qualifies — it carries no body
    # text, no table and no plot.
    header = html.Header([
        dcc.Link(brand, href=PAGES[0][0],
                 className="text-decoration-none"),
        sys_online,
        html.Span(workspace_bar.toggle_button(), className="ms-3"),
        dbc.Nav(nav_items, navbar=True, className="ms-auto"),
    ], className="phoxel-header crt sticky-top roigbiv-navbar")
    footer = html.Footer(
        f"(c) 2026 LOGISTECH // ALL RIGHTS RESERVED // BUILD {_ROIGBIV_VERSION}",
        className="phoxel-footer",
    )
    return html.Div([
        dcc.Location(id="roigbiv-url", refresh=False),
        dcc.Location(id="roigbiv-redirect", refresh=False),
        workspace_bar.stores(),
        html.Div(className="roigbiv-grid", **{"aria-hidden": "true"}),
        header,
        dbc.Container([workspace_bar.collapse()], fluid=True),
        dbc.Container(id="roigbiv-page-content", fluid=True,
                      className="flex-grow-1 pb-4"),
        footer,
    ], className="roigbiv-shell")


def resolve_route(pathname: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """``(page_path, redirect_to)`` for a URL. Either may be ``None``.

    ``redirect_to`` is set for a retired path, so the address bar can be
    rewritten rather than the new page being served under the old URL — the
    nav's ``active="exact"`` matching keys on the path, and a stale one would
    leave every nav item unhighlighted.
    """
    if not pathname or pathname == "/":
        return PAGES[0][0], PAGES[0][0]
    path = pathname.rstrip("/") or "/"
    if path in _REDIRECTS:
        return _REDIRECTS[path], _REDIRECTS[path]
    for page_path, _, _ in PAGES:
        if path == page_path.rstrip("/"):
            return page_path, None
    return None, None


def _wire_routes(app: dash.Dash) -> None:
    _page_by_path = {path: page for path, _, page in PAGES}

    @app.callback(
        Output("roigbiv-page-content", "children"),
        Input("roigbiv-url", "pathname"),
    )
    def _render(pathname: str):  # noqa: ANN001
        page_path, _ = resolve_route(pathname)
        if page_path is None:
            return dbc.Alert(
                f"Unknown page: {pathname}. Navigate via the top bar.",
                color="warning")
        return _page_by_path[page_path].layout()

    @app.callback(
        Output("roigbiv-redirect", "pathname"),
        Input("roigbiv-url", "pathname"),
    )
    def _redirect(pathname: str):  # noqa: ANN001
        # A second Location writes the URL; one component cannot be both the
        # Input and the Output of the same callback.
        _, target = resolve_route(pathname)
        return target if target is not None else no_update


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
