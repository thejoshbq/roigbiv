"""Structured logging for the Dash UI.

Every UI callback that catches an exception goes through
:func:`roigbiv.ui.components.errors.user_error` (or its figure sibling),
which calls ``logger.exception(...)`` — that way the terminal always shows a
full traceback even when the UI only renders a short message. Without this,
UI-surfaced errors were silent in the terminal and users had no way to
diagnose them.
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_ui_logging(level: int = logging.INFO) -> None:
    """Attach a stderr StreamHandler to the ``roigbiv.ui`` logger tree.

    Idempotent — safe to call from ``build_app()`` on every start.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("roigbiv.ui")
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return ``roigbiv.ui.<name>`` — pass the page/module as ``name``."""
    return logging.getLogger(f"roigbiv.ui.{name}")
