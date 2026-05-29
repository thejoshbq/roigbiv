"""Consistent error surfacing for Dash callbacks.

Every ``except`` in the UI should route through :func:`user_error` (for alert
bodies) or :func:`user_error_figure` (for Plotly placeholders). Both log the
full traceback to the terminal — users no longer see a bare class name in
the browser with no corresponding server-side trace.
"""
from __future__ import annotations

import traceback
import uuid
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import MATCH, Input, Output, State, html

from roigbiv.ui.logging import get_logger


def user_error(
    exc: BaseException,
    context: str,
    *,
    include_traceback: bool = True,
    logger_name: str = "errors",
) -> dbc.Alert:
    """Render a uniform red alert with context + message + collapsible trace.

    The traceback is always logged to stderr via ``logger.exception``; the
    ``include_traceback`` flag only controls whether the *UI* also exposes it.
    """
    get_logger(logger_name).exception(
        "%s — %s: %s", context, type(exc).__name__, exc or "(no message)",
    )
    message = str(exc) or type(exc).__name__
    body: list[Any] = [
        html.Div(context, className="fw-bold mb-1"),
        html.Div(message, className="small"),
    ]
    if include_traceback:
        uid = uuid.uuid4().hex[:8]
        body.append(
            dbc.Button(
                "Show details",
                id={"type": "roigbiv-err-toggle", "uid": uid},
                size="sm", outline=True, color="danger",
                className="mt-1",
                n_clicks=0,
            )
        )
        body.append(
            dbc.Collapse(
                html.Pre(
                    traceback.format_exc(),
                    className="small text-muted mt-2 mb-0",
                    style={"whiteSpace": "pre-wrap", "maxHeight": "260px",
                           "overflowY": "auto"},
                ),
                id={"type": "roigbiv-err-collapse", "uid": uid},
                is_open=False,
            )
        )
    return dbc.Alert(body, color="danger", className="mb-2")


def user_error_figure(exc: BaseException, context: str,
                      *, logger_name: str = "errors",
                      theme: str | None = None) -> dict:
    """Same idea as :func:`user_error` but returns a Plotly figure dict.

    Use this for ``dcc.Graph.figure`` outputs where a full alert component
    would break the figure prop type. The terminal still receives the full
    traceback — only the UI version is terse.
    """
    from roigbiv.ui.services.theme import (
        axis_muted_color,
        plotly_template,
    )

    get_logger(logger_name).exception(
        "%s — %s: %s", context, type(exc).__name__, exc or "(no message)",
    )
    message = str(exc) or type(exc).__name__
    return {
        "data": [],
        "layout": {
            "height": 420,
            "template": plotly_template(theme),
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [
                {
                    "text": f"<b>{context}</b>",
                    "showarrow": False,
                    "xref": "paper", "yref": "paper",
                    "x": 0.5, "y": 0.58,
                    "font": {"size": 14, "color": "#e74c3c"},
                },
                {
                    "text": message,
                    "showarrow": False,
                    "xref": "paper", "yref": "paper",
                    "x": 0.5, "y": 0.42,
                    "font": {"size": 12, "color": axis_muted_color(theme)},
                },
            ],
        },
    }


def register_callbacks(app: dash.Dash) -> None:
    """Wire the pattern-matched toggle for every ``user_error`` alert.

    Registered once from ``app.build_app()``.
    """
    @app.callback(
        Output({"type": "roigbiv-err-collapse", "uid": MATCH}, "is_open"),
        Input({"type": "roigbiv-err-toggle", "uid": MATCH}, "n_clicks"),
        State({"type": "roigbiv-err-collapse", "uid": MATCH}, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle(n_clicks, is_open):
        if not n_clicks:
            return is_open
        return not is_open
