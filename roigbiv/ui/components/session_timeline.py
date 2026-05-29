"""Horizontal timeline of sessions for one FOV."""
from __future__ import annotations

from typing import Iterable

import dash_bootstrap_components as dbc
from dash import html

from roigbiv.ui.services.loaders import SessionRef


def session_timeline(
    sessions: Iterable[SessionRef],
    selected_ids: set[str],
) -> html.Div:
    """Render a one-row ordered list of session chips."""
    items = []
    for s in sessions:
        active = s.session_id in selected_ids
        label = s.session_date.isoformat() if s.session_date else s.session_id[:8]
        post = f" · p={s.fov_posterior:.2f}" if s.fov_posterior is not None else ""
        items.append(
            dbc.Button(
                label + post,
                id={"type": "roigbiv-session-chip", "session_id": s.session_id},
                color="primary" if active else "light",
                outline=not active,
                size="sm",
                className="me-2 mb-2",
            )
        )
    if not items:
        return html.Div(html.Em("No sessions for this FOV.",
                                className="text-muted"),
                        className="mb-2")
    return html.Div(items, className="d-flex flex-wrap align-items-center")
