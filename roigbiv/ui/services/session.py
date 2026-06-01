"""Per-browser session ID for multi-user state isolation."""
from __future__ import annotations

import secrets

from flask import session

SESSION_KEY = "roigbiv_session_id"


def get_session_id() -> str:
    """Return (or create) the per-browser session UUID from the Flask cookie.

    Must only be called from within a Dash callback (Flask request context).
    Background threads must not call this — they're already bound to a
    specific session's object via the factory that spawned them.
    """
    if SESSION_KEY not in session:
        session[SESSION_KEY] = secrets.token_hex(16)
        session.modified = True
    return session[SESSION_KEY]
