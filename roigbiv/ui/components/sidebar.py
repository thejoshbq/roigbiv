"""Shared layout primitives used across pages."""
from __future__ import annotations

from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from roigbiv.pipeline.workspace import WorkspacePaths


def workspace_summary_card(workspace: Optional[WorkspacePaths]) -> dbc.Card:
    """Compact card summarising the currently-selected workspace."""
    if workspace is None:
        body = [html.P(
            "No workspace selected. Open the Process page to scan a directory.",
            className="mb-0 text-muted",
        )]
    else:
        body = [
            _row("Input",    str(workspace.input_root)),
            _row("Output",   str(workspace.output_root)),
            _row("Registry", str(workspace.db_path)),
            _row("TIFs",     f"{len(workspace.tifs)} discovered"),
        ]
    return dbc.Card(
        dbc.CardBody(body),
        className="roigbiv-card-accent mb-3",
    )


def _row(label: str, value: str) -> html.Div:
    return html.Div([
        html.Span(label, className="text-muted me-2"),
        html.Span(value, className="roigbiv-muted-code"),
    ], className="mb-1")


def segmented(name: str, options: list[tuple[str, str]], value: str) -> dbc.RadioItems:
    """Bootstrap segmented-control group using ``dbc.RadioItems``."""
    return dbc.RadioItems(
        id=name,
        options=[{"label": label, "value": val} for val, label in options],
        value=value,
        inline=True,
        className="roigbiv-segmented",
        inputClassName="btn-check",
        labelClassName="btn btn-sm btn-outline-primary",
        labelCheckedClassName="active",
    )


def sidebar_toggle(*, toggle_id: str, store_id: str,
                    default_open: bool = True) -> html.Div:
    """Render the sidebar chevron-toggle button and its persistence store.

    The page is responsible for:
    * placing this helper *before* the left column,
    * giving the left ``dbc.Col`` a stable id (so a callback can swap its
      ``className``: ``"d-none"`` vs. its normal class),
    * giving the right ``dbc.Col`` a stable id (so its width class can swap
      between ``col-md-8`` and ``col-md-12``).

    The state is mirrored to a ``dcc.Store`` in local storage so navigating
    between pages preserves the open/closed choice. ``default_open=False``
    is for a page whose form most users won't need to touch on a first
    visit — see :func:`register_collapsible_toggle` for the sibling pattern
    that keeps a Run button visible outside the collapse.
    """
    return html.Div([
        dcc.Store(id=store_id, storage_type="local",
                  data={"is_open": default_open}),
        dbc.Button(
            html.I(className="bi bi-chevron-double-left"),
            id=toggle_id,
            color="link", size="sm",
            className="roigbiv-sidebar-toggle",
            n_clicks=0,
        ),
    ], className="d-inline-block")


def register_collapsible_toggle(
    app: dash.Dash, *, toggle_id: str, store_id: str, collapse_id: str,
    left_col_id: str, right_col_id: str,
    left_open_class: str, left_closed_class: str,
    right_open_class: str, right_closed_class: str,
) -> None:
    """Wire a left-column form collapse that narrows/widens the columns.

    Unlike the Review page's sidebars (``app.py::_wire_sidebar_toggles``,
    whole column ``d-none``), the left column here stays visible — only its
    form content (wrapped by the page in a ``dbc.Collapse`` with id
    ``collapse_id``) hides, so a Run button placed after the collapse but
    still inside the column stays on screen and clickable. The freed width
    goes to ``right_col_id`` rather than sitting empty.

    Widening and the height-collapse are deliberately staggered via
    ``set_props`` + ``setTimeout`` rather than fired together as one set of
    declarative Outputs: flipping the column's width class at the same
    instant Bootstrap's ``.collapsing`` transition starts forces the
    still-visible form content to reflow into a much narrower column *while*
    it is animating away, which reads as the cards jittering/fading out from
    the bottom. Opening widens the columns first, so the form has room to
    lay out correctly as it animates open; closing collapses the form first
    at full width, then narrows the columns once Bootstrap's transition
    (350ms, its default) has finished.
    """
    app.clientside_callback(
        f"""
        function(n_clicks, stored) {{
            const D = window.dash_clientside;
            let is_open = !!(stored && stored.is_open === true);
            if (n_clicks) {{
                is_open = !is_open;
            }}
            if (is_open) {{
                D.set_props({left_col_id!r}, {{className: {left_open_class!r}}});
                D.set_props({right_col_id!r}, {{className: {right_open_class!r}}});
                D.set_props({collapse_id!r}, {{is_open: true}});
            }} else {{
                D.set_props({collapse_id!r}, {{is_open: false}});
                setTimeout(function() {{
                    D.set_props({left_col_id!r}, {{className: {left_closed_class!r}}});
                    D.set_props({right_col_id!r}, {{className: {right_closed_class!r}}});
                }}, 350);
            }}
            return {{is_open: is_open}};
        }}
        """,
        Output(store_id, "data"),
        Input(toggle_id, "n_clicks"),
        State(store_id, "data"),
    )
