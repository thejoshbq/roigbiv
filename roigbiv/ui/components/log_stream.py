"""Simple rolling log console for background jobs."""
from __future__ import annotations

from dash import html


def log_stream(lines: list[str], *, empty_hint: str = "Waiting for activity…") -> html.Div:
    """Return a scrollable monospace block showing ``lines``.

    Newest lines are at the bottom — matches terminal expectations.
    """
    body = "\n".join(lines) if lines else empty_hint
    return html.Div(body, className="roigbiv-log-stream", style={"whiteSpace": "pre-wrap"})
