"""A Flask request context for every UI test.

Page ``layout()`` functions read per-session state — the workspace, the run
snapshot — through ``flask.session``, and since the app's layout became a
*callable* (evaluated per page load rather than once at import) that is exactly
the context they run in for real. Tests that call ``layout()`` directly need
the same context, or they exercise a situation production never reaches.

Deliberately not solved by making ``get_session_id`` tolerant of a missing
context: background threads must never share a session id, and a silent
fallback there would cross-wire two users' runners.
"""
from __future__ import annotations

import pytest
from flask import Flask


@pytest.fixture(autouse=True)
def _request_context():
    """Push a throwaway request context around each test.

    A fresh Flask app per test, so the session cookie — and therefore the
    ``AppState`` / runner keyed on it — cannot leak between tests.
    """
    app = Flask(__name__)
    app.secret_key = "roigbiv-tests"
    with app.test_request_context("/"):
        yield
