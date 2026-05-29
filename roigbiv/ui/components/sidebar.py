"""Shared layout primitives used across pages."""
from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import dcc, html

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
        labelClassName="btn btn-sm btn-outline-secondary",
        labelCheckedClassName="active",
    )


def sidebar_toggle(*, toggle_id: str, store_id: str) -> html.Div:
    """Render the sidebar chevron-toggle button and its persistence store.

    The page is responsible for:
    * placing this helper *before* the left column,
    * giving the left ``dbc.Col`` a stable id (so a callback can swap its
      ``className``: ``"d-none"`` vs. its normal class),
    * giving the right ``dbc.Col`` a stable id (so its width class can swap
      between ``col-md-8`` and ``col-md-12``).

    The state is mirrored to a ``dcc.Store`` in local storage so navigating
    between pages preserves the open/closed choice.
    """
    return html.Div([
        dcc.Store(id=store_id, storage_type="local",
                  data={"is_open": True}),
        dbc.Button(
            html.I(className="bi bi-chevron-double-left"),
            id=toggle_id,
            color="light", size="sm",
            className="roigbiv-sidebar-toggle",
            n_clicks=0,
        ),
    ], className="d-inline-block")
